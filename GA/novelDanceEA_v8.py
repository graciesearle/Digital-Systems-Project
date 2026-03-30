import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 1. Biomechanical Constants ---
BONE_LENGTHS = {
    "TORSO": 0.2,
    "HEAD": 0.08,
    "UPPER_ARM": 0.12, 
    "FOREARM": 0.13,
    "THIGH": 0.15,
    "SHIN": 0.15,
}

GROUND_LEVEL = 0.0 # Y-coordinate of the ground

# --- 2. GA Parameters ---
POPULATION_SIZE = 150
GENOME_LENGTH = 32  # Number of poses (frames) in a dance
MUTATION_RATE = 0.1       
MUTATION_AMOUNT = 0.3
MAX_VELOCITY = 0.2
REFERENCE_POSES = []  # To be filled with standard poses

# --- 3. Core GA Functions ---

def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical angles (theta, phi) to 3D Cartesian coordinates (x, y, z).
    Z is UP
    """
    # Maths convention: Z is r*cos(phi), X/Y are derived from sin(phi)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z

def get_vector_angle(p_start, p_mid, p_end):
    # Returns angle in radians (0 to pi)
    v1 = (p_start[0]-p_mid[0], p_start[1]-p_mid[1], p_start[2]-p_mid[2])
    v2 = (p_end[0]-p_mid[0], p_end[1]-p_mid[1], p_end[2]-p_mid[2])
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(sum(k**2 for k in v1))
    mag2 = math.sqrt(sum(k**2 for k in v2))
    if mag1*mag2 == 0: return 0
    return math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))

def create_random_pose_gene():
    # FORCE T-POSE INITIALIZATION
    # This prevents them from starting "crumpled"
    hip_x, hip_y, hip_z = 0.5, 0.5, 0.45

    # Angles (Theta=Spin, Phi=Up/Down)
    # Z-axis is Up (Phi=0)
    
    # Torso: Up (Phi=0)
    tor_t, tor_p = 0.0, 0.1 
    # Head: Up
    hed_t, hed_p = 0.0, 0.1
    
    # L Arm: Left (Theta=0, Phi=pi/2)
    la_t, la_p = 0.0, 1.57
    lf_t, lf_p = 0.0, 1.57 # Forearm straight out
    
    # R Arm: Right (Theta=pi, Phi=pi/2)
    ra_t, ra_p = 3.14, 1.57
    rf_t, rf_p = 3.14, 1.57
    
    # Legs: Down (Phi=pi)
    ll_t, ll_p = 0.0, 3.0 # Thigh
    ls_t, ls_p = 0.0, 3.0 # Shin
    
    rl_t, rl_p = 0.0, 3.0
    rs_t, rs_p = 0.0, 3.0
    
    # Add slight noise so they aren't identical robots
    gene = [hip_x, hip_y, hip_z, 
            tor_t, tor_p, hed_t, hed_p,
            la_t, la_p, lf_t, lf_p,
            ra_t, ra_p, rf_t, rf_p,
            ll_t, ll_p, ls_t, ls_p,
            rl_t, rl_p, rs_t, rs_p]
            
    # Perturb slightly
    noisy_gene = [g + random.uniform(-0.1, 0.1) for g in gene]
    return noisy_gene

def create_random_dance():
    return [create_random_pose_gene() for _ in range(GENOME_LENGTH)]

def calculate_pose_coords(g):
    # Unpack the massive list
    # Hips
    hx, hy, hz = g[0], g[1], g[2]

    # Helper to unpack pairs easily
    # Gene indices start from 3
    idx = 3
    def get_vec(length):
        nonlocal idx
        t, p = g[idx], g[idx+1]
        idx += 2
        return spherical_to_cartesian(length, t, p)
    
    coords = {}
    coords["Hips"] = (hx, hy, hz)

    # Torso & Head
    dx, dy, dz = get_vec(BONE_LENGTHS["TORSO"])
    coords["Neck"] = (hx + dx, hy + dy, hz + dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["HEAD"])
    nk = coords["Neck"]
    coords["Head"] = (nk[0] + dx, nk[1] + dy, nk[2] + dz)

    # Left Arm
    dx, dy, dz = get_vec(BONE_LENGTHS["UPPER_ARM"])
    coords["L_Elbow"] = (nk[0] + dx, nk[1] + dy, nk[2] + dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["FOREARM"])
    el = coords["L_Elbow"]
    coords["L_Hand"] = (el[0] + dx, el[1] + dy, el[2] + dz)

    # Right Arm
    dx, dy, dz = get_vec(BONE_LENGTHS["UPPER_ARM"])
    coords["R_Elbow"] = (nk[0] + dx, nk[1] + dy, nk[2] + dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["FOREARM"])
    el = coords["R_Elbow"]
    coords["R_Hand"] = (el[0] + dx, el[1] + dy, el[2] + dz)

    # Left Leg
    dx, dy, dz = get_vec(BONE_LENGTHS["THIGH"])
    coords["L_Knee"] = (hx + dx, hy + dy, hz + dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["SHIN"])
    kn = coords["L_Knee"]
    coords["L_Foot"] = (kn[0] + dx, kn[1] + dy, kn[2] + dz)

    # Right Leg
    dx, dy, dz = get_vec(BONE_LENGTHS["THIGH"])
    coords["R_Knee"] = (hx + dx, hy + dy, hz + dz)

    dx, dy, dz = get_vec(BONE_LENGTHS["SHIN"])
    kn = coords["R_Knee"]
    coords["R_Foot"] = (kn[0] + dx, kn[1] + dy, kn[2] + dz)

    return coords

def get_angle(vec):
    return math.atan2(vec[1], vec[0])

def generate_standard_poses_3d():
# Pose 1: Standing Neutral (Hands down)
    REFERENCE_POSES.append({
        "Hips": (0.5, 0.5, 0.4),
        "L_Hand": (0.6, 0.5, 0.25), "R_Hand": (0.4, 0.5, 0.25), # Hands low Z
        "L_Foot": (0.55, 0.5, 0.0), "R_Foot": (0.45, 0.5, 0.0)  # Feet on floor
    })
    # Pose 2: T-Pose (Hands out)
    REFERENCE_POSES.append({
        "Hips": (0.5, 0.5, 0.4),
        "L_Hand": (0.8, 0.5, 0.6), "R_Hand": (0.2, 0.5, 0.6),   # Hands high Z
        "L_Foot": (0.55, 0.5, 0.0), "R_Foot": (0.45, 0.5, 0.0)
    })
    # Pose 3: Hands Up (Cheering)
    REFERENCE_POSES.append({
        "Hips": (0.5, 0.5, 0.4),
        "L_Hand": (0.6, 0.5, 0.8), "R_Hand": (0.4, 0.5, 0.8),   # Hands very high Z
        "L_Foot": (0.55, 0.5, 0.0), "R_Foot": (0.45, 0.5, 0.0)
    })

generate_standard_poses_3d()

def calculate_novelty(pose_coords):
    """
    Returns the distance to the NEAREST standard pose.
    Higher value = More Novel (less like the standard bank).
    """
    min_dist = float('inf')
    
    check_joints = ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]
    
    for ref_pose in REFERENCE_POSES:
        dist_sum = 0
        for joint in check_joints:
            p1 = pose_coords[joint] # (x, y, z)
            p2 = ref_pose[joint]    # (x, y, z)
            
            # 3D Euclidean Distance: sqrt(dx^2 + dy^2 + dz^2)
            d = math.sqrt(
                (p1[0] - p2[0])**2 + 
                (p1[1] - p2[1])**2 + 
                (p1[2] - p2[2])**2
            )
            dist_sum += d
        
        # Track the closest match
        if dist_sum < min_dist:
            min_dist = dist_sum
            
    return min_dist

def calculate_fitness(dance_genome, debug=False):
    dance_poses = [calculate_pose_coords(gene) for gene in dance_genome]
    total_score = 0
    
    # Define groups for different rewards
    limb_joints = ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]
    core_joints = ["Head", "Hips", "L_Knee", "R_Knee"]
    
    consecutive_air_frames = 0

    # --- 1. RANGE OF MOTION (ROM) BONUS ---
    # Check if the dancer is actually using their joints over the full 32 frames
    # We calculate the standard deviation of the angles.
    # Genes [3:] are angles.
    import numpy as np
    all_genes = np.array(dance_genome)
    # Calculate std dev for each angle column (vertical variance)
    angle_variances = np.std(all_genes[:, 3:], axis=0)
    
    # Reward high variance (moving joints a lot)
    # Sum of std devs * 100
    rom_score = np.sum(angle_variances) * 200
    total_score += rom_score
    if debug: print(f"  > ROM Bonus: {rom_score:.2f}")

    for i in range(GENOME_LENGTH):
        pose = dance_poses[i]
        frame_score = 0
        
        # --- A. STRUCTURAL CHECKS (Keep these, they are working) ---
        p1, p2 = pose["Hips"], pose["Head"]
        spine_len = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        if spine_len < 0.15: frame_score -= 500 
        
        # Elbows
        l_el_ang = get_vector_angle(pose["Neck"], pose["L_Elbow"], pose["L_Hand"])
        if l_el_ang < 0.3: frame_score -= 50
        r_el_ang = get_vector_angle(pose["Neck"], pose["R_Elbow"], pose["R_Hand"])
        if r_el_ang < 0.3: frame_score -= 50

        # --- B. PHYSICS ---
        lz, rz, hz = pose["L_Foot"][2], pose["R_Foot"][2], pose["Hips"][2]
        
        if lz < 0: frame_score -= abs(lz) * 800
        if rz < 0: frame_score -= abs(rz) * 800
        
        if hz > 0.65: frame_score -= (hz-0.65)*400
        if hz < 0.25: frame_score -= (0.25-hz)*400

        # Gravity
        if lz > 0.05 and rz > 0.05: consecutive_air_frames += 1
        else: consecutive_air_frames = 0
        if consecutive_air_frames > 3: frame_score -= 100

        # --- C. VELOCITY REWARDS (The Fix!) ---
        if i > 0:
            prev = dance_poses[i-1]
            
            # 1. Limbs (High Reward)
            for j in limb_joints:
                p1, p2 = prev[j], pose[j]
                d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                
                if d > MAX_VELOCITY: frame_score -= 50
                elif d < 0.01: frame_score -= 5 # Increased Statue Penalty
                else: 
                    # HUGE REWARD for moving hands/feet
                    frame_score += d * 30 
            
            # 2. Core/Hips (Low Reward)
            for j in core_joints:
                p1, p2 = prev[j], pose[j]
                d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                
                if d > MAX_VELOCITY: frame_score -= 50
                else:
                    # Low reward to discourage just hopping
                    frame_score += d * 2

        total_score += frame_score

    return total_score

def select_parent(population):
    tournament = random.sample(population, 5)
    winner = max(tournament, key=lambda d: d['fitness'])
    return winner['genome']

def crossover(parent1, parent2):
    crossover_point = random.randint(1, GENOME_LENGTH - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(dance):
    res = []
    for gene in dance:
        # Standard Mutation
        if random.random() < MUTATION_RATE:
            new_gene = list(gene) # Copy
            
            # 1. Mutate Hips (Small amount, keep them stable)
            new_gene[0] += random.uniform(-0.05, 0.05) # X
            new_gene[1] += random.uniform(-0.05, 0.05) # Y
            new_gene[2] += random.uniform(-0.05, 0.05) # Z
            new_gene[2] = max(0.3, min(0.6, new_gene[2])) # Clamp Z
            
            # 2. Mutate Angles (LARGE amount)
            # Genes 3 to end are angles
            for k in range(3, len(new_gene)):
                # 20% chance to change an angle drastically
                if random.random() < 0.2:
                    change = random.uniform(-0.5, 0.5) # Big move (approx 30 degrees)
                else:
                    change = random.uniform(-0.1, 0.1) # Small adjustment
                
                new_gene[k] += change
                
            res.append(new_gene)
        else:
            res.append(gene)
    return res

# --- 4. Main Evolution Loop ---

def run_evolution():
    population = []
    for _ in range(POPULATION_SIZE):
        dance = create_random_dance()
        fitness = calculate_fitness(dance)
        population.append({'genome': dance, 'fitness': fitness})

    # Track best found
    global_best_genome = None
    global_best_fitness = -float('inf')

    print(f"Generation 0 - Best Fitness: {max(population, key=lambda d: d['fitness'])['fitness']:.2f}")

    for gen in range(1, 1000):
        # 1. Sort Population by Fitness
        population.sort(key=lambda d: d['fitness'], reverse=True)

        # 2. Update Global Best Dance if new record
        if population[0]['fitness'] > global_best_fitness:
            global_best_fitness = population[0]['fitness']
            global_best_genome = population[0]['genome'][:] # Copy

        # 3. ELITISM: Keep top 2 dances unchanged
        new_population = [population[0], population[1]]

        # 4. Fill the rest of the population
        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append({'genome': child, 'fitness': calculate_fitness(child)})
        population = new_population
        
        if gen % 25 == 0:
            best_dance = max(population, key=lambda d: d['fitness'])
            print(f"Generation {gen} - Best Fitness: {best_dance['fitness']:.2f}")

    # Final check to ensure global best is returned
    population.sort(key=lambda d: d['fitness'], reverse=True)
    if population[0]['fitness'] > global_best_fitness:
        global_best_genome = population[0]['genome']
        global_best_fitness = population[0]['fitness']

    print("\n--- Final Result ---")
    best_dance = max(population, key=lambda d: d['fitness'])
    print(f"Final Best Fitness: {best_dance['fitness']:.2f}")

    return global_best_genome

# -----------------------------------------------------------------
# --- NEW VIDEO/GIF SAVING FUNCTION ---
# -----------------------------------------------------------------

def save_3d_dance_gif(dance_genome, filename="dance_animation.gif"):
    print(f"Rendering 3D GIF to {filename}...")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    bones_list = [
        ("Hips", "Neck"), ("Neck", "Head"),
        ("Neck", "L_Elbow"), ("L_Elbow", "L_Hand"),
        ("Neck", "R_Elbow"), ("R_Elbow", "R_Hand"),
        ("Hips", "L_Knee"), ("L_Knee", "L_Foot"),
        ("Hips", "R_Knee"), ("R_Knee", "R_Foot")
    ]

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Frame {frame_idx}")
        
        # Camera rotation for cinematic effect
        ax.view_init(elev=20, azim=frame_idx * 2)

        # Draw Floor
        xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        z = xx * 0 
        ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')

        pose_coords = calculate_pose_coords(dance_genome[frame_idx])
        
        # Draw Skeleton
        for j1, j2 in bones_list:
            p1 = pose_coords[j1]
            p2 = pose_coords[j2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=3)
            
        for joint, (x, y, z) in pose_coords.items():
            ax.scatter(x, y, z, c='b', s=50)

    anim = FuncAnimation(fig, update, frames=len(dance_genome), interval=150)
    writer = PillowWriter(fps=7)
    anim.save(filename, writer=writer)
    print("Done!")
    plt.close()

# -----------------------------------------------------------------
# --- HELPER: INCREMENT FILENAME ---
# -----------------------------------------------------------------

def get_next_filename(base_name="dance_animation", ext="gif"):
    """
    Checks the directory and returns the next available filename 
    (e.g., dance_animation_001.gif).
    """
    counter = 1
    while True:
        # Format with 3 digits (001, 002, etc.) for better sorting
        filename = f"{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1

# --- RUN THE SCRIPT ---

# 1. Run the evolution
final_dance_genome = run_evolution()

# 2. Get next available filename
unique_filename = get_next_filename("dance_animation")

# 3. Save as GIF
save_3d_dance_gif(final_dance_genome, unique_filename)