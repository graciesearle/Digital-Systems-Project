import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import glob
import librosa
from moviepy import VideoFileClip, AudioFileClip
import torch
import torch.nn as nn
import torch.optim as optim

# --- SETTINGS ---
LATENT_DIM = 16  # Size of the compressed genome
SEQUENCE_LENGTH = 32  # Number of frames per dance segment
JOINTS_COUNT = 11 # Number of key joints used (Hips, Neck, Head, L_Elbow, L_Hand, R_Elbow, R_Hand, L_Knee, L_Foot, R_Knee, R_Foot)
INPUT_SIZE = SEQUENCE_LENGTH * JOINTS_COUNT * 3  # 3D coordinates
MAX_FILES_TO_LOAD = 50 
TRAIN_EPOCHS = 50

# --- DATA LOADING & PROCESSING ---
REAL_DANCE_BANK = []

def load_aist_data(dataset_path, max_files=10):
    """
    1. Loads AIST++ .pkl files.
    2. SWAPS AXES (Y-up -> Z-up) so they stand up.
    3. ROTATES them to face forward.
    4. NORMALIZES to 0-1 range.
    """
    file_list = glob.glob(f"{dataset_path}/**/*.pkl", recursive=True)
    
    if not file_list:
        print(f"!!! ERROR: No .pkl files found in {dataset_path}")
        return

    print(f"Found {len(file_list)} files. Loading {max_files} sequences...")
    
    TARGET_HIP = np.array([0.5, 0.5, 0.4])

    for file_path in file_list[:max_files]:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            raw_keypoints = data['keypoints3d'] # (N, 17, 3)

            # --- AXIS SWAP PRE-CHECK ---
            # We need to decide scaling based on the NEW height axis (which was Y, index 1)
            # But let's just swap first to avoid confusion.
            
            for frame in raw_keypoints:
                # --- 1. AXIS SWAP (Y-up to Z-up) ---
                # AIST: x, y, z (where y is height)
                # GA:   x, y, z (where z is height)
                # Fix:  New X = Old X
                #       New Y = Old Z (Depth)
                #       New Z = Old Y (Height)
                
                # Create a temporary dictionary with SWAPPED axes
                swapped_frame = {}
                for i in range(17):
                    ox, oy, oz = frame[i]
                    swapped_frame[i] = np.array([ox, oz, oy]) # Swap Y and Z

                # --- 2. RETARGETING ---
                # Now we use 'swapped_frame' which is standing upright (Z-up)
                l_hip, r_hip = swapped_frame[11], swapped_frame[12]
                center_hips = (l_hip + r_hip) / 2.0
                center_neck = (swapped_frame[5] + swapped_frame[6]) / 2.0

                pose = {
                    "Hips": center_hips, "Neck": center_neck, "Head": swapped_frame[0],
                    "L_Elbow": swapped_frame[7], "R_Elbow": swapped_frame[8],
                    "L_Hand": swapped_frame[9],  "R_Hand": swapped_frame[10],
                    "L_Knee": swapped_frame[13], "R_Knee": swapped_frame[14],
                    "L_Foot": swapped_frame[15], "R_Foot": swapped_frame[16]
                }
                
                # --- 3. AUTO-SCALE (Updated for Z-up) ---
                # Check Head Z. If > 10, it's cm. If < 2, it's meters.
                head_z = pose["Head"][2]
                if head_z > 10: SCALE = 0.0035 
                else:           SCALE = 0.35   

                # --- 4. ROTATION ALIGNMENT ---
                # Calculate hip angle in XY plane
                hip_vec_x = r_hip[0] - l_hip[0]
                hip_vec_y = r_hip[1] - l_hip[1]
                angle = math.atan2(hip_vec_y, hip_vec_x)
                
                # Rotate to face -X (Math.Pi)
                rot_angle = math.pi - angle 
                cos_a = math.cos(rot_angle)
                sin_a = math.sin(rot_angle)

                normalized_pose = {}
                for joint, coord in pose.items():
                    # Center
                    rel_x = (coord[0] - center_hips[0])
                    rel_y = (coord[1] - center_hips[1])
                    rel_z = (coord[2] - center_hips[2])

                    # Rotate (Z-axis rotation)
                    rot_x = rel_x * cos_a - rel_y * sin_a
                    rot_y = rel_x * sin_a + rel_y * cos_a
                    
                    # Scale & Move
                    final_x = TARGET_HIP[0] + (rot_x * SCALE)
                    final_y = TARGET_HIP[1] + (rot_y * SCALE)
                    final_z = TARGET_HIP[2] + (rel_z * SCALE)

                    normalized_pose[joint] = (final_x, final_y, final_z)
                    
                REAL_DANCE_BANK.append(normalized_pose)
                
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    print(f"Successfully loaded {len(REAL_DANCE_BANK)} upright poses.")

def analyze_music(audio_path):
    print(f"Analyzing {audio_path}...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray): tempo = tempo.item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"Tempo: {tempo:.1f} BPM | Beats: {len(beat_times)}")
    return beat_times, y, sr

audio_path = "Supernatural - Newjeans.mp3"
beat_times, _, _ = analyze_music(audio_path)

# --- SEGMENT CALCULATION ---
NUM_CLIPS = math.ceil(len(beat_times) / SEQUENCE_LENGTH)
TOTAL_GENES = NUM_CLIPS * LATENT_DIM
print(f"Song needs {NUM_CLIPS} clips (Total {TOTAL_GENES} genes) to cover {len(beat_times)} beats.")

# --- AUTOENCODER MODEL ---
class DanceAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Compresses 32 frames of 40-dim pose data into 4 latent vars
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM),
            )
        
        # Decoder: Unzips 4 latent vars back to 32 frames of pose data
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, INPUT_SIZE),
            nn.Sigmoid() # Force output between 0-1
            )
        
    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent
    
ae_model = DanceAutoEncoder()

def train_autoencoder():
    print("Preparing training data...")
    training_data = []
    JOINT_ORDER = ["Hips", "Neck", "Head", "L_Elbow", "L_Hand", "R_Elbow", "R_Hand", "L_Knee", "L_Foot", "R_Knee", "R_Foot"]
    
    curr_seq = []
    for frame_dict in REAL_DANCE_BANK:
        flat = []
        for j in JOINT_ORDER: flat.extend(frame_dict[j])
        curr_seq.extend(flat)
        if len(curr_seq) == INPUT_SIZE:
            training_data.append(curr_seq)
            curr_seq = []
            
    if not training_data: return
    print(f"Training on {len(training_data)} sequences...")
    
    opt = optim.Adam(ae_model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    dset = torch.utils.data.DataLoader(torch.tensor(training_data, dtype=torch.float32), batch_size=32, shuffle=True)
    
    print(f"Training AutoEncoder for {TRAIN_EPOCHS} epochs...")
    ae_model.train()
    for e in range(1, TRAIN_EPOCHS + 1): 
        loss_sum = 0
        for b in dset:
            opt.zero_grad()
            rec, _ = ae_model(b)
            loss = crit(rec, b)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        
        if e % 5 == 0: 
            print(f"Epoch {e} Loss: {loss_sum/len(dset):.4f}")
            
    print("Training Complete.")
    ae_model.eval()

# --- GENETIC ALGORITHM IN LATENT SPACE ---
def create_genome():
    return [random.uniform(0, 1) for _ in range(LATENT_DIM)]

def enforce_rigid_skeleton(pose):
    """
    Takes a loose/stretchy pose from the AI and forces strict bone lengths.
    It preserves the DIRECTION of the limbs but fixes the LENGTH.
    """
    # 1. Define the Skeleton Hierarchy (Parent -> Child, Bone Name)
    # Root is Hips. Everything branches from there.
    hierarchy = [
        ("Hips", "Neck", "TORSO"),
        ("Neck", "Head", "HEAD"),
        ("Neck", "L_Elbow", "UPPER_ARM"), ("L_Elbow", "L_Hand", "FOREARM"),
        ("Neck", "R_Elbow", "UPPER_ARM"), ("R_Elbow", "R_Hand", "FOREARM"),
        ("Hips", "L_Knee", "THIGH"),      ("L_Knee", "L_Foot", "SHIN"),
        ("Hips", "R_Knee", "THIGH"),      ("R_Knee", "R_Foot", "SHIN")
    ]
    
    fixed_pose = pose.copy()
    
    for parent, child, bone_name in hierarchy:
        p_coords = np.array(fixed_pose[parent])
        c_coords = np.array(fixed_pose[child])
        
        # A. Calculate Direction (Unit Vector)
        direction = c_coords - p_coords
        length = np.linalg.norm(direction)
        
        if length == 0: continue # Prevent divide by zero
        
        unit_vector = direction / length
        
        # B. Force strict length
        target_length = BONE_LENGTHS[bone_name]
        
        # C. Calculate new Child position
        new_child_pos = p_coords + (unit_vector * target_length)
        
        # Update the pose dictionary so the next bone starts from the correct place
        fixed_pose[child] = tuple(new_child_pos)
        
    return fixed_pose

def decode_genome(full_genome):
    # --- SAFETY: Ensure genome length ---
    required_length = NUM_CLIPS * LATENT_DIM
    if len(full_genome) < required_length:
        missing = required_length - len(full_genome)
        full_genome = full_genome + (full_genome * (missing // len(full_genome) + 1))[:missing]

    raw_frames = []
    
    # 1. DECODE ALL BLOCKS
    with torch.no_grad():
        for i in range(NUM_CLIPS):
            start = i * LATENT_DIM
            end = start + LATENT_DIM
            gene_chunk = full_genome[start:end]
            
            z = torch.tensor(gene_chunk, dtype=torch.float32)
            if z.dim() == 0: continue
            if z.dim() == 1: z = z.unsqueeze(0)

            # Decode to 32 frames (flat list of numbers)
            block_data = ae_model.decoder(z).numpy().flatten()
            
            # Reshape into list of 32 frames (each frame is 33 numbers)
            block_frames = []
            for f in range(0, len(block_data), 33):
                frame_data = block_data[f:f+33]
                if len(frame_data) == 33:
                    block_frames.append(frame_data)
            
            raw_frames.extend(block_frames)

    # 2. SMOOTHING (Moving Average)
    # This removes the "jitter/vibration" that makes it look robotic
    smoothed_frames = []
    window_size = 3 # Average 3 frames together
    
    for i in range(len(raw_frames)):
        # Get neighbors
        start_win = max(0, i - 1)
        end_win = min(len(raw_frames), i + 2)
        window = raw_frames[start_win:end_win]
        
        # Calculate average of this window
        avg_frame = np.mean(window, axis=0)
        smoothed_frames.append(avg_frame)

    # 3. PARSE TO DICTIONARIES & ENFORCE RIGIDITY
    final_poses = []
    for fd in smoothed_frames:
        pose = {
            "Hips": (fd[0], fd[1], fd[2]), "Neck": (fd[3], fd[4], fd[5]),
            "Head": (fd[6], fd[7], fd[8]),
            "L_Elbow": (fd[9], fd[10], fd[11]), "R_Elbow": (fd[12], fd[13], fd[14]),
            "L_Hand": (fd[15], fd[16], fd[17]), "R_Hand": (fd[18], fd[19], fd[20]),
            "L_Knee": (fd[21], fd[22], fd[23]), "R_Knee": (fd[24], fd[25], fd[26]),
            "L_Foot": (fd[27], fd[28], fd[29]), "R_Foot": (fd[30], fd[31], fd[32]),
        }
        
        # Apply the bone length fix you added earlier
        rigid_pose = enforce_rigid_skeleton(pose)
        final_poses.append(rigid_pose)
                
    return final_poses[:len(beat_times)]

# --- CONSTANTS & PARAMETERS ---


BONE_LENGTHS = {
    "TORSO": 0.2, "HEAD": 0.08,
    "UPPER_ARM": 0.12, "FOREARM": 0.13,
    "THIGH": 0.15, "SHIN": 0.15,
}
GROUND_LEVEL = 0.0 

POPULATION_SIZE = 150
GENOME_LENGTH = len(beat_times)
MUTATION_RATE = 0.1       
MUTATION_AMOUNT = 0.3
MAX_VELOCITY = 0.2

# --- MATH HELPERS ---
def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z

def get_vector_angle(p_start, p_mid, p_end):
    v1 = (p_start[0]-p_mid[0], p_start[1]-p_mid[1], p_start[2]-p_mid[2])
    v2 = (p_end[0]-p_mid[0], p_end[1]-p_mid[1], p_end[2]-p_mid[2])
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(sum(k**2 for k in v1))
    mag2 = math.sqrt(sum(k**2 for k in v2))
    if mag1*mag2 == 0: return 0
    return math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))

# --- GENOME & SKELETON ---
def create_random_pose_gene():
    # T-Pose Start
    hip_x, hip_y, hip_z = 0.5, 0.5, 0.45
    
    # Angles [Theta, Phi]
    tor_t, tor_p = 0.0, 0.1 
    hed_t, hed_p = 0.0, 0.1
    la_t, la_p = 0.0, 1.57; lf_t, lf_p = 0.0, 1.57 # L Arm Out
    ra_t, ra_p = 3.14, 1.57; rf_t, rf_p = 3.14, 1.57 # R Arm Out
    ll_t, ll_p = 0.0, 3.0; ls_t, ls_p = 0.0, 3.0 # Legs Down
    rl_t, rl_p = 0.0, 3.0; rs_t, rs_p = 0.0, 3.0

    gene = [hip_x, hip_y, hip_z, 
            tor_t, tor_p, hed_t, hed_p,
            la_t, la_p, lf_t, lf_p,
            ra_t, ra_p, rf_t, rf_p,
            ll_t, ll_p, ls_t, ls_p,
            rl_t, rl_p, rs_t, rs_p]
    return [g + random.uniform(-0.1, 0.1) for g in gene]

def create_random_dance():
    return [create_random_pose_gene() for _ in range(GENOME_LENGTH)]

def calculate_pose_coords(g):
    hx, hy, hz = g[0], g[1], g[2]
    idx = 3
    def get_vec(length):
        nonlocal idx
        t, p = g[idx], g[idx+1]
        idx += 2
        return spherical_to_cartesian(length, t, p)
    
    c = {"Hips": (hx, hy, hz)}
    
    # Torso Chain
    dx, dy, dz = get_vec(BONE_LENGTHS["TORSO"])
    c["Neck"] = (hx+dx, hy+dy, hz+dz)
    dx, dy, dz = get_vec(BONE_LENGTHS["HEAD"])
    c["Head"] = (c["Neck"][0]+dx, c["Neck"][1]+dy, c["Neck"][2]+dz)
    
    # Arms
    nk = c["Neck"]
    dx, dy, dz = get_vec(BONE_LENGTHS["UPPER_ARM"])
    c["L_Elbow"] = (nk[0]+dx, nk[1]+dy, nk[2]+dz)
    dx, dy, dz = get_vec(BONE_LENGTHS["FOREARM"])
    c["L_Hand"] = (c["L_Elbow"][0]+dx, c["L_Elbow"][1]+dy, c["L_Elbow"][2]+dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["UPPER_ARM"])
    c["R_Elbow"] = (nk[0]+dx, nk[1]+dy, nk[2]+dz)
    dx, dy, dz = get_vec(BONE_LENGTHS["FOREARM"])
    c["R_Hand"] = (c["R_Elbow"][0]+dx, c["R_Elbow"][1]+dy, c["R_Elbow"][2]+dz)

    # Legs
    dx, dy, dz = get_vec(BONE_LENGTHS["THIGH"])
    c["L_Knee"] = (hx+dx, hy+dy, hz+dz)
    dx, dy, dz = get_vec(BONE_LENGTHS["SHIN"])
    c["L_Foot"] = (c["L_Knee"][0]+dx, c["L_Knee"][1]+dy, c["L_Knee"][2]+dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["THIGH"])
    c["R_Knee"] = (hx+dx, hy+dy, hz+dz)
    dx, dy, dz = get_vec(BONE_LENGTHS["SHIN"])
    c["R_Foot"] = (c["R_Knee"][0]+dx, c["R_Knee"][1]+dy, c["R_Knee"][2]+dz)
    return c

# --- STYLE & FITNESS ---

def calculate_innovation_score(pose_coords):
    """
    Scores the pose based on the trade-off between AIST++ Similarity and Novelty.
    Returns: A fitness score component.
    """
    if not REAL_DANCE_BANK: return 0

    # 1. Find distance to the NEAREST pose in the AIST++ bank
    # Optimization: Sample 50 random poses for speed
    sample_bank = random.sample(REAL_DANCE_BANK, min(50, len(REAL_DANCE_BANK)))
    min_dist = float('inf')
    
    check_joints = ["L_Hand", "R_Hand", "L_Foot", "R_Foot", "Head"]
    
    for ref_pose in sample_bank:
        dist_sum = 0
        for joint in check_joints:
            p1, p2 = pose_coords[joint], ref_pose[joint]
            # 3D Euclidean Distance
            d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            dist_sum += d
            
        if dist_sum < min_dist:
            min_dist = dist_sum
            
    # 2. THE INNOVATION CURVE
    # We want moves that are "Kind of close" but "Not identical".
    
    if min_dist < 0.15:
        # Zone 1: Too Similar (Plagiarism)
        # It's valid, so we give points, but not many.
        return 5 
        
    elif min_dist < 0.6:
        # Zone 2: THE SWEET SPOT (Novelty + Validity)
        # It's close enough to be "Dance", but far enough to be "New".
        # We reward distance within this safe zone.
        # Score scales from 20 to 50 based on how unique it is.
        return 20 + (min_dist * 50) 
        
    else:
        # Zone 3: Too Weird (Broken Physics/Noise)
        return -20 

def calculate_fitness(full_genome):
    dance_poses = decode_genome(full_genome)
    score = 0
    
    for i in range(len(dance_poses)-1):
        curr, next_p = dance_poses[i], dance_poses[i+1]
        dist = 0
        for j in ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]:
            p1, p2 = curr[j], next_p[j]
            dist += math.sqrt(sum((p1[k]-p2[k])**2 for k in range(3)))
        
        # --- FIX: CONTINUOUS REWARD ---
        # Old way: if dist > 0.05: score += 5 (Too hard to reach!)
        # New way: Reward ANY movement, scaled up.
        score += dist * 100 
        
        # 2. SEAM PENALTY (Critical for stitching)
        if (i+1) % SEQUENCE_LENGTH == 0:
            # If the transition between clips is unnatural/teleporting
            if dist > 0.4: 
                score -= 200 # Heavy penalty for broken seams
            
    return score
        
# --- EVOLUTION ---
def select_parent(pop):
    return max(random.sample(pop, 5), key=lambda d: d['fitness'])['genome']

def crossover_latent(p1, p2):
    pt = random.randint(1, LATENT_DIM-1)
    return p1[:pt] + p2[pt:]

def mutate_latent(genome):
    # Latent mutation is just nudging numbers
    new_g = list(genome)
    if random.random() < 0.2:
        idx = random.randint(0, LATENT_DIM-1)
        new_g[idx] += random.gauss(0, 0.5)
    return new_g

def run_hybrid_system():
    # A. Setup Data & Train Model
    load_aist_data("AISTpop") # Folder must exist!
    train_autoencoder()
    
    # B. Run GA
    population = []
    print("Evolving Latent Codes...")
    for _ in range(100):
        g = create_genome()
        population.append({'genome': g, 'fitness': calculate_fitness(g)})
        
    for gen in range(50):
        population.sort(key=lambda x: x['fitness'], reverse=True)
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness {population[0]['fitness']:.2f}")
            
        new_pop = population[:5] # Elitism
        while len(new_pop) < 100:
            p1 = random.choice(population[:20])['genome']
            p2 = random.choice(population[:20])['genome']
            child = mutate_latent(crossover_latent(p1, p2))
            new_pop.append({'genome': child, 'fitness': calculate_fitness(child)})
        population = new_pop

    return population[0]['genome']

# --- VISUALISATION ---
def save_3d_dance_gif(genome, fname="dance_pop.gif"):
    print(f"Saving to {fname}...")
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    bones = [("Hips", "Neck"), ("Neck", "Head"),
             ("Neck", "L_Elbow"), ("L_Elbow", "L_Hand"),
             ("Neck", "R_Elbow"), ("R_Elbow", "R_Hand"),
             ("Hips", "L_Knee"), ("L_Knee", "L_Foot"),
             ("Hips", "R_Knee"), ("R_Knee", "R_Foot")]
    
    def update(i):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_title(f"Red: AI | Green: Real Data (Frame {i})")
        ax.view_init(elev=20, azim=i*2)
        
        # Floor
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        ax.plot_surface(x, y, x*0, alpha=0.1, color='k')
        
        # Draw AI
        coords = calculate_pose_coords(genome[i])
        for a, b in bones:
            p1, p2 = coords[a], coords[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=3)
        for k, v in coords.items():
            ax.scatter(v[0], v[1], v[2], c='b')
            
        # Draw Ghost
        if REAL_DANCE_BANK:
            # Find closest match
            min_d = float('inf')
            match = None
            sample = random.sample(REAL_DANCE_BANK, min(50, len(REAL_DANCE_BANK)))
            check = ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]
            for ref in sample:
                d = sum((coords[k][dim]-ref[k][dim])**2 for k in check for dim in range(3))
                if d < min_d: min_d = d; match = ref
            
            if match:
                for a, b in bones:
                    p1, p2 = match[a], match[b]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g--', alpha=0.4)

    anim = FuncAnimation(fig, update, frames=len(genome), interval=200)
    anim.save(fname, writer=PillowWriter(fps=5))
    print("Done.")

# --- FILE HELPER ---
def get_next_filename(base_name="dance_pop", ext="mp4"):
    """
    Checks the directory and returns the next available filename 
    (e.g., dance_pop_001.mp4).
    """
    counter = 1
    while True:
        # Check for mp4 specifically since that's what we save
        filename = f"{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1

def save_video(genome, audio_path, beat_times, fname="result.mp4"):
    if not fname.endswith(".mp4"): fname = fname.replace(".gif", ".mp4")
    print("Decoding full dance...")
    poses = decode_genome(genome)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    bones = [("Hips", "Neck"), ("Neck", "Head"),
             ("Neck", "L_Elbow"), ("L_Elbow", "L_Hand"), ("Neck", "R_Elbow"), ("R_Elbow", "R_Hand"),
             ("Hips", "L_Knee"), ("L_Knee", "L_Foot"), ("Hips", "R_Knee"), ("R_Knee", "R_Foot")]

    # Calculate FPS based on beat count
    duration = beat_times[-1] - beat_times[0]
    fps = len(poses) / duration
    print(f"Rendering {len(poses)} frames over {duration:.1f}s ({fps:.1f} FPS)...")

    def update(i):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_title(f"Beat {i}/{len(poses)}")
        ax.view_init(elev=20, azim=i*0.5)
        coords = poses[i]
        
        # Floor
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        ax.plot_surface(x, y, x*0, alpha=0.1, color='k')
        
        # AI (Red)
        for a, b in bones:
            p1, p2 = coords[a], coords[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=3)
        
        # Ghost (Green) - Simplified for speed
        if REAL_DANCE_BANK and i % 5 == 0: # Only check every 5th frame to speed up rendering
             # Simple random ghost for vibe (Full KNN is too slow for 500+ frames)
             ref = random.choice(REAL_DANCE_BANK)
             for a, b in bones:
                p1, p2 = ref[a], ref[b]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g--', alpha=0.1)

    anim = FuncAnimation(fig, update, frames=len(poses))
    anim.save("temp.mp4", writer='ffmpeg', fps=fps)
    plt.close()

    try:
        vc = VideoFileClip("temp.mp4")
        ac = AudioFileClip(audio_path).subclipped(0, vc.duration)
        vc.with_audio(ac).write_videofile(fname, codec='libx264', audio_codec='aac')
    except:
        os.rename("temp.mp4", fname.replace(".mp4", "_silent.mp4"))
    if os.path.exists("temp.mp4"): os.remove("temp.mp4")

# --- RUN ---
load_aist_data("AISTpop") # Ensure this folder exists
final = run_hybrid_system()

# Use auto-incrementing filename
unique_filename = get_next_filename("dance_pop", "mp4")
save_video(final, audio_path, beat_times, unique_filename)