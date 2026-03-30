import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

# --- 3. Core GA Functions ---
# (create_random_pose_gene, create_random_dance, 
#  calculate_pose_coords, get_angle, calculate_fitness, 
#  select_parent, crossover, mutate)

def create_random_pose_gene():
    # 1. Random Hips 
    # X between 0.3 and 0.7 (stay roughly centered)
    hip_x = random.uniform(0.3, 0.7)
    hip_y = random.uniform(0.3, 0.6) 

    # Random Angles for 10 joints
    angles = [random.uniform(0, 2 * math.pi) for _ in range(10)]

    return [hip_x, hip_y] + angles

def create_random_dance():
    return [create_random_pose_gene() for _ in range(GENOME_LENGTH)]

def calculate_pose_coords(pose_gene):
    hip_x = pose_gene[0]
    hip_y = pose_gene[1]
    angles = pose_gene[2:]

    [tor_ang, head_ang, 
    l_sh_ang, l_el_ang, 
    r_sh_ang, r_el_ang, 
    l_hip_ang, l_knee_ang, 
    r_hip_ang, r_knee_ang] = angles
    
    pose_coords = {}

    # --- Torso ---
    pose_coords["Hips"] = (hip_x, hip_y)

    pose_coords["Neck"] = (
        pose_coords["Hips"][0] + math.cos(tor_ang) * BONE_LENGTHS["TORSO"],
        pose_coords["Hips"][1] + math.sin(tor_ang) * BONE_LENGTHS["TORSO"]
    )
    pose_coords["Head"] = (
        pose_coords["Neck"][0] + math.cos(head_ang) * BONE_LENGTHS["HEAD"],
        pose_coords["Neck"][1] + math.sin(head_ang) * BONE_LENGTHS["HEAD"]
    )

    # --- Left Arm ---
    pose_coords["L_Elbow"] = (
        pose_coords["Neck"][0] + math.cos(l_sh_ang) * BONE_LENGTHS["UPPER_ARM"],
        pose_coords["Neck"][1] + math.sin(l_sh_ang) * BONE_LENGTHS["UPPER_ARM"]
    )
    pose_coords["L_Hand"] = (
        pose_coords["L_Elbow"][0] + math.cos(l_el_ang) * BONE_LENGTHS["FOREARM"],
        pose_coords["L_Elbow"][1] + math.sin(l_el_ang) * BONE_LENGTHS["FOREARM"]
    )

    # --- Right Arm ---
    pose_coords["R_Elbow"] = (
        pose_coords["Neck"][0] + math.cos(r_sh_ang) * BONE_LENGTHS["UPPER_ARM"],
        pose_coords["Neck"][1] + math.sin(r_sh_ang) * BONE_LENGTHS["UPPER_ARM"]
    )
    pose_coords["R_Hand"] = (
        pose_coords["R_Elbow"][0] + math.cos(r_el_ang) * BONE_LENGTHS["FOREARM"],
        pose_coords["R_Elbow"][1] + math.sin(r_el_ang) * BONE_LENGTHS["FOREARM"]
    )

    # --- Left Leg ---
    pose_coords["L_Knee"] = (
        pose_coords["Hips"][0] + math.cos(l_hip_ang) * BONE_LENGTHS["THIGH"],
        pose_coords["Hips"][1] + math.sin(l_hip_ang) * BONE_LENGTHS["THIGH"]
    )
    pose_coords["L_Foot"] = (
        pose_coords["L_Knee"][0] + math.cos(l_knee_ang) * BONE_LENGTHS["SHIN"],
        pose_coords["L_Knee"][1] + math.sin(l_knee_ang) * BONE_LENGTHS["SHIN"]
    )

    # --- Right Leg ---
    pose_coords["R_Knee"] = (
        pose_coords["Hips"][0] + math.cos(r_hip_ang) * BONE_LENGTHS["THIGH"],
        pose_coords["Hips"][1] + math.sin(r_hip_ang) * BONE_LENGTHS["THIGH"]
    ) 
    pose_coords["R_Foot"] = (
        pose_coords["R_Knee"][0] + math.cos(r_knee_ang) * BONE_LENGTHS["SHIN"],
        pose_coords["R_Knee"][1] + math.sin(r_knee_ang) * BONE_LENGTHS["SHIN"]
    )
    return pose_coords

def get_angle(vec):
    return math.atan2(vec[1], vec[0])

def calculate_fitness(dance_genome):
    dance_poses = [calculate_pose_coords(gene) for gene in dance_genome]
    total_score = 0
    
    moving_joints = ["L_Hand", "R_Hand", "L_Foot", "R_Foot", "Head", 
                     "L_Elbow", "R_Elbow", "L_Knee", "R_Knee", "Hips"]
    
    consecutive_air_frames = 0
    MAX_AIR_TIME = 3 # Approx 0.5 seconds at 6 FPS

    for i in range(GENOME_LENGTH - 1):
        pose = dance_poses[i]

        # --- A. Gravity Check ---
        l_foot_y = pose["L_Foot"][1]
        r_foot_y = pose["R_Foot"][1]

        # Check if both feet are above the ground threshold (allow 0.02 wiggle room)
        if l_foot_y > (GROUND_LEVEL + 0.02) and r_foot_y > (GROUND_LEVEL + 0.02):
            consecutive_air_frames += 1
        else:
            consecutive_air_frames = 0 # Landed. Reset counter.

        if consecutive_air_frames > MAX_AIR_TIME:
            total_score -= 100  # Penalty for being in air too long (flying)

        # Check if feet are under the floor (clipping)
        if l_foot_y < GROUND_LEVEL or r_foot_y < GROUND_LEVEL:
            total_score -= 500  # Penalty for clipping through floor

        # --- B. Movement & Velocity Check ---
        if i < GENOME_LENGTH - 1:
            pose_A = dance_poses[i]
            pose_B = dance_poses[i + 1]

            frame_penalty = 0
        
            # 1. Check Biomechanical Angle Limits (The "Broken Neck" check)
            vec_torso = (pose_A["Neck"][0] - pose_A["Hips"][0], pose_A["Neck"][1] - pose_A["Hips"][1])
            vec_head = (pose_A["Head"][0] - pose_A["Neck"][0], pose_A["Head"][1] - pose_A["Neck"][1])
            angle_diff = get_angle(vec_head) - get_angle(vec_torso)
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > (math.pi / 2.2):
                frame_penalty += 500 # Huge penalty for broken neck

            # 2. Check Velocity (The "Smoothness" check)
            for joint in moving_joints:
                p1 = pose_A[joint]
                p2 = pose_B[joint]
                # Calculate distance moved in this 1 frame
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                if dist > MAX_VELOCITY:
                    # If moving too fast (teleporting), massive penalty
                    frame_penalty += 200 
                elif dist < 0.01:
                    # If barely moving (statue), small penalty
                    frame_penalty += 5 
                else:
                    # We reward "Flow": steady movement
                    total_score += dist * 10 

            total_score -= frame_penalty

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
    mutated_dance = []
    for pose_gene in dance:
        if random.random() < MUTATION_RATE:
            mutated_pose = []

            # Mutate Hips Position Slightly
            hip_x = pose_gene[0] + random.uniform(-0.05, 0.05)
            hip_y = pose_gene[1] + random.uniform(-0.05, 0.05)

            # Clamp hips to screen limits
            hip_x = max(0.0, min(1.0, hip_x))
            hip_y = max(0.0, min(1.0, hip_y))

            mutated_pose.extend([hip_x, hip_y])

            # Mutate Angles (Indices 2 onwards)
            for angle in pose_gene[2:]:
                new_angle = angle + random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
                mutated_pose.append(new_angle)

            mutated_dance.append(mutated_pose)
        else:
            mutated_dance.append(pose_gene)
    return mutated_dance

# --- 4. Main Evolution Loop ---

def run_evolution():
    population = []
    for _ in range(POPULATION_SIZE):
        dance = create_random_dance()
        fitness = calculate_fitness(dance)
        population.append({'genome': dance, 'fitness': fitness})
    print(f"Generation 0 - Best Fitness: {max(population, key=lambda d: d['fitness'])['fitness']:.2f}")

    for gen in range(1, 121):
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append({'genome': child, 'fitness': calculate_fitness(child)})
        population = new_population
        
        if gen % 20 == 0:
            best_dance = max(population, key=lambda d: d['fitness'])
            print(f"Generation {gen} - Best Fitness: {best_dance['fitness']:.2f}")

    print("\n--- Final Result ---")
    best_dance = max(population, key=lambda d: d['fitness'])
    print(f"Final Best Fitness: {best_dance['fitness']:.2f}")
    return best_dance['genome']

# -----------------------------------------------------------------
# --- NEW VIDEO/GIF SAVING FUNCTION ---
# -----------------------------------------------------------------

def save_dance_gif(dance_genome, filename="dance_animation.gif"):
    print(f"\nGenerating animation ({len(dance_genome)} frames)...")
    
    bones_list = [
        ("Hips", "Neck"), ("Neck", "Head"),
        # Arms
        ("Neck", "L_Elbow"), ("L_Elbow", "L_Hand"),
        ("Neck", "R_Elbow"), ("R_Elbow", "R_Hand"),
        # Legs
        ("Hips", "L_Knee"), ("L_Knee", "L_Foot"),
        ("Hips", "R_Knee"), ("R_Knee", "R_Foot")
    ]

    # 1. Setup the Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 2. The Update Function (Called for every frame of video)
    def update(frame_index):
        ax.clear() # Clear the previous frame
        
        # Set limits and hide axis
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Frame {frame_index}")
        
        # Get coords for this frame
        pose_gene = dance_genome[frame_index]
        pose_coords = calculate_pose_coords(pose_gene)
        
        # Draw "Ghost" of previous frame (light grey) for visual trail
        if frame_index > 0:
            prev_coords = calculate_pose_coords(dance_genome[frame_index-1])
            for j1, j2 in bones_list:
                p1 = prev_coords[j1]
                p2 = prev_coords[j2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgrey', linestyle='-', linewidth=1)

        # Draw Current Frame (Red bones, Blue joints)
        bones = [("Hips", "Neck"), ("Neck", "Head"), ("Neck", "L_Hand"),
                 ("Neck", "R_Hand"), ("Hips", "L_Foot"), ("Hips", "R_Foot")]
        
        for j1, j2 in bones_list:
            p1 = pose_coords[j1]
            p2 = pose_coords[j2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3)
            
        for joint, (x, y) in pose_coords.items():
            ax.plot(x, y, 'bo', markersize=8)

    # 3. Create Animation Object
    anim = FuncAnimation(fig, update, frames=len(dance_genome), interval=200)
    
    # 4. Save using PillowWriter (Standard tool for GIFs)
    # fps=5 means 5 frames per second
    writer = PillowWriter(fps=5) 
    anim.save(filename, writer=writer)
    
    print(f"Successfully saved animation to {filename}")
    plt.close() # Close plot to prevent it showing up in some IDEs

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
save_dance_gif(final_dance_genome, unique_filename)