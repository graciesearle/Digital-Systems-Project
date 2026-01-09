import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import glob

# --- 1. DATA LOADING & PROCESSING ---
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

# --- 2. CONSTANTS & PARAMETERS ---
BONE_LENGTHS = {
    "TORSO": 0.2, "HEAD": 0.08,
    "UPPER_ARM": 0.12, "FOREARM": 0.13,
    "THIGH": 0.15, "SHIN": 0.15,
}
GROUND_LEVEL = 0.0 

POPULATION_SIZE = 150
GENOME_LENGTH = 32
MUTATION_RATE = 0.1       
MUTATION_AMOUNT = 0.3
MAX_VELOCITY = 0.2

# --- 3. MATH HELPERS ---
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

# --- 4. GENOME & SKELETON ---
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

# --- 5. STYLE & FITNESS ---

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

def calculate_fitness(dance_genome, debug=False):
    dance_poses = [calculate_pose_coords(gene) for gene in dance_genome]
    total_score = 0
    POP_MAX_VELOCITY = 0.35
    
    # 1. DIVERSITY BONUS (Internal Novelty)
    # Check if the dancer actually moves (variance of angles)
    # This prevents them from finding one "Novel Pose" and freezing there.
    import numpy as np
    all_genes = np.array(dance_genome)
    angle_variances = np.std(all_genes[:, 3:], axis=0)
    total_score += np.sum(angle_variances) * 200

    for i in range(GENOME_LENGTH):
        pose = dance_poses[i]
        frame_score = 0
        
        # --- A. INNOVATION SCORE (Similarity vs Novelty) ---
        # This replaces the old style score
        innovation = calculate_innovation_score(pose)
        frame_score += innovation
        
        if debug and i == 0: 
            print(f"  > Innovation Score: {innovation:.1f}")

        # --- B. BIOMECHANICS (Strict Constraints) ---
        # Spine Check
        p1, p2 = pose["Hips"], pose["Head"]
        if math.sqrt(sum((p1[k]-p2[k])**2 for k in range(3))) < 0.15: 
            frame_score -= 500
        
        # Floor Clipping
        lz = pose["L_Foot"][2]
        if lz < 0: frame_score -= abs(lz) * 800
        
        # --- C. DYNAMICS ---
        if i > 0:
            prev = dance_poses[i-1]
            for joint in ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]:
                p1, p2 = prev[joint], pose[joint]
                d = math.sqrt(sum((p1[k]-p2[k])**2 for k in range(3)))
                
                if d > POP_MAX_VELOCITY: frame_score -= 50
                elif d > 0.05: frame_score += d * 40 
        
        total_score += frame_score

    return total_score

# --- 6. EVOLUTION ---
def select_parent(pop):
    return max(random.sample(pop, 5), key=lambda d: d['fitness'])['genome']

def crossover(p1, p2):
    pt = random.randint(1, GENOME_LENGTH - 1)
    return p1[:pt] + p2[pt:]

def mutate(dance):
    res = []
    for gene in dance:
        if random.random() < MUTATION_RATE:
            new_gene = list(gene)
            # Mutate Hips
            new_gene[0] += random.uniform(-0.05, 0.05)
            new_gene[1] += random.uniform(-0.05, 0.05)
            new_gene[2] = max(0.3, min(0.6, new_gene[2] + random.uniform(-0.05, 0.05)))
            
            # Mutate Angles (Occasional big moves)
            for k in range(3, len(new_gene)):
                change = random.uniform(-0.5, 0.5) if random.random() < 0.2 else random.uniform(-0.1, 0.1)
                new_gene[k] += change
            res.append(new_gene)
        else:
            res.append(gene)
    return res

def run_evolution():
    pop = [{'genome': create_random_dance(), 'fitness': 0} for _ in range(POPULATION_SIZE)]
    
    # DEBUG CHECK
    print("\n--- DIAGNOSTIC (First Dancer) ---")
    calculate_fitness(pop[0]['genome'], debug=True)
    print("---------------------------------\n")

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(1, 101): # 100 Gens
        for p in pop: p['fitness'] = calculate_fitness(p['genome'])
        pop.sort(key=lambda d: d['fitness'], reverse=True)
        
        if pop[0]['fitness'] > best_fitness:
            best_fitness = pop[0]['fitness']
            best_genome = pop[0]['genome'][:]

        new_pop = pop[:2] # Elitism
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = select_parent(pop), select_parent(pop)
            c = mutate(crossover(p1, p2))
            new_pop.append({'genome': c, 'fitness': 0})
        pop = new_pop
        
        if gen % 10 == 0: print(f"Gen {gen} Best: {best_fitness:.2f}")

    return best_genome

# --- 7. VISUALIZATION ---
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

# --- 8. FILE HELPER ---
def get_next_filename(base_name="dance_pop", ext="gif"):
    """
    Checks the directory and returns the next available filename 
    (e.g., dance_pop_001.gif).
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1

# --- RUN ---
load_aist_data("AISTpop") # Ensure this folder exists
final = run_evolution()

# Use auto-incrementing filename
unique_filename = get_next_filename("dance_pop")
save_3d_dance_gif(final, unique_filename)