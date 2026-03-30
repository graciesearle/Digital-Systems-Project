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
SAFE_LIMIT = 100

# --- DATA LOADING & PROCESSING ---
REAL_DANCE_BANK = []

def load_aist_data(dataset_path, max_files=60):
    global REAL_DANCE_BANK
    global SAFE_LIMIT
    global BONE_LENGTHS # <--- We will calculate this dynamically
    
    file_list = glob.glob(f"{dataset_path}/**/*.pkl", recursive=True)
    if not file_list: return

    print(f"Found {len(file_list)} files. Loading {max_files} sequences...")
    
    REAL_DANCE_BANK = [] 
    all_relative_poses = []
    
    # 1. LOAD RAW DATA
    for file_path in file_list[:max_files]:
        try:
            with open(file_path, 'rb') as f: data = pickle.load(f)
            raw = data['keypoints3d'] 
            raw_downsampled = raw[::3] 

            for frame in raw_downsampled:
                # Standard Swap & Center
                swapped = {i: np.array([p[0], p[2], p[1]]) for i, p in enumerate(frame)}
                l_hip, r_hip = swapped[11], swapped[12]
                c_hip = (l_hip + r_hip) / 2.0
                c_neck = (swapped[5] + swapped[6]) / 2.0
                
                vec_x, vec_y = r_hip[0]-l_hip[0], r_hip[1]-l_hip[1]
                rot = math.pi - math.atan2(vec_y, vec_x) 
                cos_a, sin_a = math.cos(rot), math.sin(rot)
                
                pose_dict = {
                    "Hips": c_hip, "Neck": c_neck, "Head": swapped[0],
                    "L_Elbow": swapped[7], "R_Elbow": swapped[8],
                    "L_Hand": swapped[9],  "R_Hand": swapped[10],
                    "L_Knee": swapped[13], "R_Knee": swapped[14],
                    "L_Foot": swapped[15], "R_Foot": swapped[16]
                }
                
                relative_frame = {}
                for k, v in pose_dict.items():
                    rx, ry, rz = v[0]-c_hip[0], v[1]-c_hip[1], v[2]-c_hip[2]
                    rot_x = rx*cos_a - ry*sin_a
                    rot_y = rx*sin_a + ry*cos_a
                    rot_z = rz
                    relative_frame[k] = [rot_x, rot_y, rot_z]
                
                all_relative_poses.append(relative_frame)
        except Exception: pass

    # 2. AUTO-CALIBRATE BONE LENGTHS (The Critical Fix)
    # We measure the average length of every bone in the dataset.
    print("Measuring dancer size...")
    avg_bones = { "TORSO": [], "HEAD": [], "UPPER_ARM": [], "FOREARM": [], "THIGH": [], "SHIN": [] }
    
    for f in all_relative_poses:
        def dist(k1, k2):
            p1, p2 = f[k1], f[k2]
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            
        avg_bones["TORSO"].append(dist("Hips", "Neck"))
        avg_bones["HEAD"].append(dist("Neck", "Head"))
        avg_bones["UPPER_ARM"].append(dist("Neck", "L_Elbow")) 
        avg_bones["FOREARM"].append(dist("L_Elbow", "L_Hand"))
        avg_bones["THIGH"].append(dist("Hips", "L_Knee"))
        avg_bones["SHIN"].append(dist("L_Knee", "L_Foot"))

    # Set the global BONE_LENGTHS to the exact average of the data
    BONE_LENGTHS = {k: sum(v)/len(v) for k, v in avg_bones.items()}
    print(f"Auto-Calibrated Bones: {BONE_LENGTHS}")

    # 3. SET SAFE LIMIT
    # Find the max reach relative to these units
    max_val = 0
    for frame in all_relative_poses:
        for joint, coords in frame.items():
            m = max(abs(c) for c in coords)
            if m > max_val: max_val = m
            
    SAFE_LIMIT = max_val * 1.5 # 50% buffer to be safe
    print(f"Safe Limit set to: {SAFE_LIMIT:.4f}")

    # 4. NORMALIZE
    print(f"Normalizing data...")
    for frame in all_relative_poses:
        norm_pose = {}
        for k, coords in frame.items():
            nx = (coords[0] / (2 * SAFE_LIMIT)) + 0.5
            ny = (coords[1] / (2 * SAFE_LIMIT)) + 0.5
            nz = (coords[2] / (2 * SAFE_LIMIT)) + 0.5
            norm_pose[k] = (nx, ny, nz)
        REAL_DANCE_BANK.append(norm_pose)

    print(f"Database ready: {len(REAL_DANCE_BANK)} frames loaded.")

def analyze_music(audio_path):
    print(f"Analyzing {audio_path}...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray): tempo = tempo.item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"Tempo: {tempo:.1f} BPM | Beats: {len(beat_times)}")
    return beat_times, y, sr

audio_path = "Supernatural (Short).mp3"  
beat_times, _, _ = analyze_music(audio_path)

TARGET_FPS = 20

# Calculate duration
song_duration = beat_times[-1] - 2.0 # Add buffer
total_frames_needed = int(song_duration * TARGET_FPS)

# --- SEGMENT CALCULATION ---
NUM_CLIPS = math.ceil(total_frames_needed / SEQUENCE_LENGTH)
TOTAL_GENES = NUM_CLIPS * LATENT_DIM

print(f"Song Duration: {song_duration:.1f}s")
print(f"Target: {TARGET_FPS} FPS -> {total_frames_needed} Frames")
print(f"Architecture: {NUM_CLIPS} Latent Clips ({TOTAL_GENES} Genes)")

# --- AUTOENCODER MODEL ---
class DanceAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, INPUT_SIZE),
            nn.Sigmoid() # <--- BRING THIS BACK
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
    # 1. Flatten Data
    for frame_dict in REAL_DANCE_BANK:
        flat = []
        for j in JOINT_ORDER: flat.extend(frame_dict[j])
        curr_seq.extend(flat)
        if len(curr_seq) == INPUT_SIZE:
            training_data.append(curr_seq)
            curr_seq = []
            
    if not training_data: return

    # 2. SANITY CHECK: Remove sequences with NaNs
    clean_data = []
    for seq in training_data:
        if not np.isnan(seq).any():
            clean_data.append(seq)
    
    print(f"Training on {len(clean_data)} clean sequences (dropped {len(training_data)-len(clean_data)} bad ones)...")
    
    # 3. Setup Training with Gradient Clipping
    # Lower LR slightly to 0.0005 for stability
    opt = optim.Adam(ae_model.parameters(), lr=0.001) 
    crit = nn.MSELoss()
    dset = torch.utils.data.DataLoader(torch.tensor(clean_data, dtype=torch.float32), batch_size=32, shuffle=True)
    
    print(f"Training AutoEncoder for {TRAIN_EPOCHS} epochs...")
    ae_model.train()
    
    for e in range(1, TRAIN_EPOCHS + 1): 
        loss_sum = 0
        batch_count = 0
        
        for b in dset:
            opt.zero_grad()
            rec, _ = ae_model(b)
            loss = crit(rec, b)
            
            if torch.isnan(loss):
                print("!!! WARNING: Loss became NaN in batch. Skipping update.")
                continue
                
            loss.backward()
            
            # --- CRITICAL FIX: CLIP GRADIENTS ---
            # Prevents the "Explosion" that causes NaNs
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            # ------------------------------------
            
            opt.step()
            loss_sum += loss.item()
            batch_count += 1
        
        if batch_count > 0 and e % 5 == 0: 
            print(f"Epoch {e} Loss: {loss_sum/batch_count:.5f}")
            
    print("Training Complete.")
    ae_model.eval()

# --- GENETIC ALGORITHM IN LATENT SPACE ---
def create_genome():
    return [random.uniform(0, 1) for _ in range(LATENT_DIM)]

def enforce_rigid_skeleton(pose):
    global BONE_LENGTHS 
    
    # Hierarchy with Default Directions (T-Pose)
    hierarchy = [
        ("Hips", "Neck", "TORSO", np.array([0, 0, 1])),      # Up
        ("Neck", "Head", "HEAD", np.array([0, 0, 1])),       # Up
        ("Neck", "L_Elbow", "UPPER_ARM", np.array([0, 1, 0])), # Left
        ("L_Elbow", "L_Hand", "FOREARM", np.array([0, 1, 0])), # Left
        ("Neck", "R_Elbow", "UPPER_ARM", np.array([0, -1, 0])),# Right
        ("R_Elbow", "R_Hand", "FOREARM", np.array([0, -1, 0])),# Right
        ("Hips", "L_Knee", "THIGH", np.array([0, 0, -1])),   # Down
        ("L_Knee", "L_Foot", "SHIN", np.array([0, 0, -1])),  # Down
        ("Hips", "R_Knee", "THIGH", np.array([0, 0, -1])),   # Down
        ("R_Knee", "R_Foot", "SHIN", np.array([0, 0, -1]))   # Down
    ]
    
    fixed_pose = pose.copy()
    
    for parent, child, bone_name, default_dir in hierarchy:
        p_coords = np.array(fixed_pose[parent])
        c_coords = np.array(fixed_pose[child])
        
        direction = c_coords - p_coords
        length = np.linalg.norm(direction)
        
        # --- SAFETY FIX ---
        # If the bone has collapsed (length is near 0), use the Default T-Pose Direction
        # This prevents the "crumpled/broken" look.
        if length < (SAFE_LIMIT * 0.01): 
            unit_vector = default_dir 
        else:
            unit_vector = direction / length
        
        target_length = BONE_LENGTHS[bone_name]
        
        # Calculate new Child position based on Parent
        new_child_pos = p_coords + (unit_vector * target_length)
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
    smoothed_frames = []
    window_size = 3 
    
    for i in range(len(raw_frames)):
        start_win = max(0, i - 1)
        end_win = min(len(raw_frames), i + 2)
        window = raw_frames[start_win:end_win]
        avg_frame = np.mean(window, axis=0)
        smoothed_frames.append(avg_frame)

    # 3. PARSE, DENORMALIZE & RIGIDITY
    final_poses = []
    
    # DEBUG: Print the height of the first frame to check scale
    first_frame_checked = False
    
    for fd in smoothed_frames:
        # A. Parse the Normalized AI Output (0.0 to 1.0)
        pose = {
            "Hips": (fd[0], fd[1], fd[2]), "Neck": (fd[3], fd[4], fd[5]),
            "Head": (fd[6], fd[7], fd[8]),
            "L_Elbow": (fd[9], fd[10], fd[11]), "R_Elbow": (fd[12], fd[13], fd[14]),
            "L_Hand": (fd[15], fd[16], fd[17]), "R_Hand": (fd[18], fd[19], fd[20]),
            "L_Knee": (fd[21], fd[22], fd[23]), "R_Knee": (fd[24], fd[25], fd[26]),
            "L_Foot": (fd[27], fd[28], fd[29]), "R_Foot": (fd[30], fd[31], fd[32]),
        }
        
        # B. Denormalize (Expand 0-1 range to -120 to +120 cm)
        denorm_pose = {}
        for k, v in pose.items():
            dx = (v[0] - 0.5) * (2 * SAFE_LIMIT)
            dy = (v[1] - 0.5) * (2 * SAFE_LIMIT)
            dz = (v[2] - 0.5) * (2 * SAFE_LIMIT)
            denorm_pose[k] = (dx, dy, dz)

        # C. Apply Rigidity (Now that we are in CM, 35.0 makes sense)
        rigid_pose = enforce_rigid_skeleton(denorm_pose)
        final_poses.append(rigid_pose)

        # DEBUG CHECK
        if not first_frame_checked:
            h_z = rigid_pose["Head"][2]
            f_z = rigid_pose["L_Foot"][2]
            height = h_z - f_z
            print(f"DEBUG: Skeleton Height is approx {height:.2f} units.")
            # If this says "0.40", your Bone Lengths are still meters.
            # If this says "160.00", you are fixed.
            first_frame_checked = True
                
    return final_poses

# --- CONSTANTS & PARAMETERS ---


# --- CONSTANTS (CENTIMETERS) ---
BONE_LENGTHS = {
    "TORSO": 45.0,    # Spine length
    "HEAD": 20.0,     # Neck to top of head
    "UPPER_ARM": 30.0,
    "FOREARM": 28.0,
    "THIGH": 48.0,    # Upper leg
    "SHIN": 45.0,     # Lower leg
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
    score = 0.0
    
    beat_indices = [int(t * TARGET_FPS) for t in beat_times]
    
    # 1. Rewards
    for beat_idx in beat_indices:
        window_start = max(0, beat_idx - 2)
        window_end = min(len(dance_poses) - 1, beat_idx + 2)
        max_beat_energy = 0
        
        for i in range(window_start, window_end):
            curr = dance_poses[i]
            next_p = dance_poses[i+1] if i+1 < len(dance_poses) else curr
            
            frame_energy = 0
            for j in ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]:
                p1, p2 = curr[j], next_p[j]
                frame_energy += math.sqrt(sum((p1[k]-p2[k])**2 for k in range(3)))
            
            if frame_energy > max_beat_energy:
                max_beat_energy = frame_energy
        
        # Scale reward relative to world size
        # (Energy / Size) gives a standardized "effort" score
        score += (max_beat_energy / SAFE_LIMIT) * 500.0 

    # 2. Penalties
    penalty = 0
    jitter_threshold = SAFE_LIMIT * 0.05 # 5% movement is noise
    teleport_threshold = SAFE_LIMIT * 0.2 # 20% movement is teleporting
    
    for i in range(len(dance_poses)-1):
        curr, next_p = dance_poses[i], dance_poses[i+1]
        
        # Center of Mass Flow
        hip_dist = math.sqrt(sum((curr["Hips"][k]-next_p["Hips"][k])**2 for k in range(3)))
            
        if hip_dist > teleport_threshold: 
            penalty += 100 

        # Hand Jitter
        hand_dist = 0
        for h in ["L_Hand", "R_Hand"]:
             hand_dist += math.sqrt(sum((curr[h][k]-next_p[h][k])**2 for k in range(3)))
        
        if hand_dist > teleport_threshold * 2: # Hands can move faster than hips
            penalty += 50
            
        # Clip Seams
        if (i+1) % SEQUENCE_LENGTH == 0:
            if hip_dist > jitter_threshold: 
                penalty += 500 

    return max(0, score - penalty)
        
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
    load_aist_data("AISTpop") 
    train_autoencoder()
    
    # B. Run GA with INTELLIGENT INITIALIZATION
    population = []
    print("Evolving Latent Codes...")
    
    # --- NEW: SEEDING ---
    # Instead of 100% random noise, let's create genomes 
    # that are variations of "0.5" (The average pose)
    for _ in range(100):
        # Start with 0.5 (Center of latent space = Average Pose)
        # Add slight noise so they aren't all identical
        g = [0.5 + random.uniform(-0.2, 0.2) for _ in range(TOTAL_GENES)]
        
        population.append({'genome': g, 'fitness': 0}) # Calc fitness later
        
    for gen in range(50):
        # ... (rest of the loop is the same) ...
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
    print(f"Rendering AI-generated frames to {fname}...")
    
    poses = decode_genome(genome)
    final_frame_count = int((beat_times[-1] + 1.0) * TARGET_FPS)
    poses = poses[:final_frame_count]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    bones = [("Hips", "Neck"), ("Neck", "Head"),
             ("Neck", "L_Elbow"), ("L_Elbow", "L_Hand"), ("Neck", "R_Elbow"), ("R_Elbow", "R_Hand"),
             ("Hips", "L_Knee"), ("L_Knee", "L_Foot"), ("Hips", "R_Knee"), ("R_Knee", "R_Foot")]

    # --- NEW: CALCULATE PLOT LIMITS ---
    # Center is 0,0,0. We need to see roughly -100 to 100.
    limit = 150 # Zoom in slightly so it fills the frame
    
    def update(i):
        ax.clear()
        # Set limits to Centimeters, centered at 0
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
        ax.set_title(f"Frame {i} ({i/TARGET_FPS:.1f}s)")
        
        # Rotate camera
        ax.view_init(elev=20, azim=i*0.5)
        
        coords = poses[i]
        
        # Floor (at roughly -80cm usually)
        floor_level = -80 
        x, y = np.meshgrid(np.linspace(-limit, limit, 5), np.linspace(-limit, limit, 5))
        ax.plot_surface(x, y, x*0 + floor_level, alpha=0.1, color='k')
        
        for a, b in bones:
            p1, p2 = coords[a], coords[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=3)

    anim = FuncAnimation(fig, update, frames=len(poses))
    anim.save("temp.mp4", writer='ffmpeg', fps=TARGET_FPS)
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