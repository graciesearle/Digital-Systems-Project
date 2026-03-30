import random
import math
import matplotlib.pyplot as plt


# --- 1. Biomechanical Constants ---
BONE_LENGTHS = {
    "TORSO": 0.2, "ARM": 0.25, "LEG": 0.3, "HEAD": 0.08,
}

# --- 2. GA Parameters ---
POPULATION_SIZE = 100
GENOME_LENGTH = 16        # A dance is 16 poses (frames) long
MUTATION_RATE = 0.1       
MUTATION_AMOUNT = 0.5     

# --- 3. Core GA Functions ---
# (create_random_pose_gene, create_random_dance, 
#  calculate_pose_coords, get_angle, calculate_fitness, 
#  select_parent, crossover, mutate)

def create_random_pose_gene():
    return [random.uniform(0, 2 * math.pi) for _ in range(6)]

def create_random_dance():
    return [create_random_pose_gene() for _ in range(GENOME_LENGTH)]

def calculate_pose_coords(pose_gene):
    [tor_ang, head_ang, l_arm_ang, r_arm_ang, l_leg_ang, r_leg_ang] = pose_gene
    pose_coords = {}
    pose_coords["Hips"] = (0.5, 0.3)
    pose_coords["Neck"] = (
        pose_coords["Hips"][0] + math.cos(tor_ang) * BONE_LENGTHS["TORSO"],
        pose_coords["Hips"][1] + math.sin(tor_ang) * BONE_LENGTHS["TORSO"]
    )
    pose_coords["Head"] = (
        pose_coords["Neck"][0] + math.cos(head_ang) * BONE_LENGTHS["HEAD"],
        pose_coords["Neck"][1] + math.sin(head_ang) * BONE_LENGTHS["HEAD"]
    )
    pose_coords["L_Hand"] = (
        pose_coords["Neck"][0] + math.cos(l_arm_ang) * BONE_LENGTHS["ARM"],
        pose_coords["Neck"][1] + math.sin(l_arm_ang) * BONE_LENGTHS["ARM"]
    )
    pose_coords["R_Hand"] = (
        pose_coords["Neck"][0] + math.cos(r_arm_ang) * BONE_LENGTHS["ARM"],
        pose_coords["Neck"][1] + math.sin(r_arm_ang) * BONE_LENGTHS["ARM"]
    )
    pose_coords["L_Foot"] = (
        pose_coords["Hips"][0] + math.cos(l_leg_ang) * BONE_LENGTHS["LEG"],
        pose_coords["Hips"][1] + math.sin(l_leg_ang) * BONE_LENGTHS["LEG"]
    )
    pose_coords["R_Foot"] = (
        pose_coords["Hips"][0] + math.cos(r_leg_ang) * BONE_LENGTHS["LEG"],
        pose_coords["Hips"][1] + math.sin(r_leg_ang) * BONE_LENGTHS["LEG"]
    )
    return pose_coords

def get_angle(vec):
    return math.atan2(vec[1], vec[0])

def calculate_fitness(dance_genome):
    dance_poses = [calculate_pose_coords(gene) for gene in dance_genome]
    total_movement = 0
    total_penalty = 0
    moving_joints = ["L_Hand", "R_Hand", "L_Foot", "R_Foot"]

    for i in range(GENOME_LENGTH - 1):
        pose_A = dance_poses[i]
        pose_B = dance_poses[i+1]
        
        for joint in moving_joints:
            p1 = pose_A[joint]
            p2 = pose_B[joint]
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            total_movement += distance
            
        vec_torso = (pose_A["Neck"][0] - pose_A["Hips"][0], pose_A["Neck"][1] - pose_A["Hips"][1])
        vec_head = (pose_A["Head"][0] - pose_A["Neck"][0], pose_A["Head"][1] - pose_A["Neck"][1])
        ang_torso = get_angle(vec_torso)
        ang_head = get_angle(vec_head)
        angle_diff = ang_head - ang_torso
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > (math.pi / 2):
            total_penalty += 100
            
    return total_movement - total_penalty

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
            for angle in pose_gene:
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

    for gen in range(1, 101):
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
# --- VISUALISATION FUNCTION ---
# -----------------------------------------------------------------

def draw_pose_on_axis(ax, pose_coords):
    """Helper function to draw a single pose on a given plot axis."""
    
    # Set the plot limits to keep the person centered
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    # Turn off the axes and ticks for a cleaner look
    ax.axis('off')
    
    bones = [
        ("Hips", "Neck"), ("Neck", "Head"), ("Neck", "L_Hand"),
        ("Neck", "R_Hand"), ("Hips", "L_Foot"), ("Hips", "R_Foot")
    ]

    # Plot the bones (lines)
    for joint1_name, joint2_name in bones:
        joint1 = pose_coords[joint1_name]
        joint2 = pose_coords[joint2_name]
        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'r-')

    # Plot the joints (dots)
    for joint_name, (x, y) in pose_coords.items():
        ax.plot(x, y, 'bo')

def save_dance_storyboard(dance_genome, filename="dance_storyboard.png"):
    """
    Creates a 4x4 grid of all 16 poses and saves it to a single PNG file.
    """
    print(f"\nCreating storyboard... (for {len(dance_genome)} frames)")
    
    # Our genome is 16 frames, so we'll make a 4x4 grid
    grid_size = math.ceil(math.sqrt(len(dance_genome))) # 4
    
    # Create a new figure with a 4x4 grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Flatten the 2D axes array into a 1D list for easy iteration
    axes = axes.flatten()

    for i, pose_gene in enumerate(dance_genome):
        if i >= len(axes): # In case dance is not a perfect square
            break
            
        # Get the (x, y) coords for this pose
        pose_coords = calculate_pose_coords(pose_gene)
        
        # Get the specific subplot for this pose
        ax = axes[i]
        
        # Draw the pose on that subplot
        draw_pose_on_axis(ax, pose_coords)
        ax.set_title(f"Frame {i}") # Add a title to each frame

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout()
    
    # Save the entire figure to a file
    plt.savefig(filename)
    print(f"Successfully saved visualization to {filename}")

# --- RUN THE SCRIPT ---

# 1. Run the evolution to get the best dance
final_dance_genome = run_evolution()

# 2. Save the final dance to a file
save_dance_storyboard(final_dance_genome)