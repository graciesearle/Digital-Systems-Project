"""
Novel Dance Generator using Genetic Algorithm with AIST++ Dataset
Combines the GA from novelDanceEA_v6.py with AIST++ dataset loading and 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
import math
import os
from pathlib import Path

# =============================================================================
# --- SETTINGS ---
# =============================================================================
DATA_FOLDER = 'keypoints3d'
SOURCE_FPS = 60  # AIST++ dataset is recorded at 60 FPS

# GA Parameters
POPULATION_SIZE = 100
GENOME_LENGTH = 240  # Number of frames in generated dance (4 seconds at 60 FPS)
NUM_GENERATIONS = 200
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

# Segment length for splicing dance moves
SEGMENT_MIN_LENGTH = 30  # 0.5 seconds
SEGMENT_MAX_LENGTH = 90  # 1.5 seconds

# Genre codes from AIST++ dataset
GENRES = ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']
GENRE_NAMES = {
    'BR': 'Break', 'PO': 'Pop', 'LO': 'Lock', 'MH': 'Middle Hip-hop',
    'LH': 'LA Hip-hop', 'HO': 'House', 'WA': 'Waack', 'KR': 'Krump',
    'JS': 'Street Jazz', 'JB': 'Ballet Jazz'
}

# COCO 17-keypoint skeleton connections
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head: nose, eyes, ears
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (6, 8), (8, 10),  # Right arm
    (5, 7), (7, 9),   # Left arm
    (12, 14), (14, 16),  # Right leg
    (11, 13), (13, 15)   # Left leg
]

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Body part colors for visualization
BONE_COLORS = {
    'head': '#FF6B6B', 'torso': '#4ECDC4',
    'right_arm': '#45B7D1', 'left_arm': '#96CEB4',
    'right_leg': '#FFEAA7', 'left_leg': '#DDA0DD'
}


# =============================================================================
# --- DATA LOADING ---
# =============================================================================

def extract_genre(filename):
    """Extract genre code from filename like 'gBR_sBM_cAll_d04_mBR0_ch01.pkl'"""
    return filename[1:3]


def load_dance_files():
    """Load all pkl files and organize by genre"""
    dances_by_genre = {genre: [] for genre in GENRES}
    
    data_path = Path(DATA_FOLDER)
    for pkl_file in data_path.glob('*.pkl'):
        genre = extract_genre(pkl_file.name)
        if genre in dances_by_genre:
            dances_by_genre[genre].append(pkl_file)
    
    return dances_by_genre


def load_pkl(filepath):
    """Load a single pkl file and return keypoints"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    # Use optimized keypoints if available
    return data.get('keypoints3d_optim', data.get('keypoints3d'))


def load_all_dance_data(dances_by_genre):
    """Load all dance data into memory for faster GA operations"""
    all_dances = []
    
    for genre, files in dances_by_genre.items():
        for filepath in files:
            keypoints = load_pkl(filepath)
            all_dances.append({
                'genre': genre,
                'name': GENRE_NAMES.get(genre, genre),
                'file': filepath.name,
                'data': keypoints
            })
    
    return all_dances


# =============================================================================
# --- GENETIC ALGORITHM ---
# =============================================================================

class DanceGenome:
    """
    Represents a dance as a sequence of segments from the dataset.
    Each segment is (dance_index, start_frame, end_frame)
    """
    def __init__(self, segments=None):
        self.segments = segments if segments else []
        self.fitness = None
        self.frames = None  # Cached rendered frames
    
    def render(self, all_dances):
        """Convert segments to actual frame data"""
        if self.frames is not None:
            return self.frames
        
        frames = []
        for dance_idx, start, end in self.segments:
            dance_data = all_dances[dance_idx]['data']
            segment_frames = dance_data[start:end]
            frames.append(segment_frames)
        
        if frames:
            self.frames = np.concatenate(frames, axis=0)
        else:
            self.frames = np.array([])
        
        return self.frames
    
    def get_total_frames(self, all_dances):
        """Get total frame count without full render"""
        total = 0
        for dance_idx, start, end in self.segments:
            total += (end - start)
        return total


def create_random_genome(all_dances, target_length=GENOME_LENGTH):
    """Create a random dance genome by selecting random segments"""
    segments = []
    total_frames = 0
    
    while total_frames < target_length:
        # Pick random dance
        dance_idx = random.randint(0, len(all_dances) - 1)
        dance_data = all_dances[dance_idx]['data']
        dance_length = len(dance_data)
        
        # Pick random segment
        seg_length = random.randint(SEGMENT_MIN_LENGTH, 
                                    min(SEGMENT_MAX_LENGTH, dance_length))
        max_start = dance_length - seg_length
        if max_start <= 0:
            continue
        
        start = random.randint(0, max_start)
        end = start + seg_length
        
        # Don't exceed target
        if total_frames + seg_length > target_length:
            end = start + (target_length - total_frames)
        
        segments.append((dance_idx, start, end))
        total_frames += (end - start)
    
    return DanceGenome(segments)


def calculate_fitness(genome, all_dances):
    """
    Fitness function evaluating:
    1. Smoothness of transitions between segments
    2. Genre diversity
    3. Movement quality (not too static, not too jerky)
    """
    if genome.fitness is not None:
        return genome.fitness
    
    frames = genome.render(all_dances)
    if len(frames) < 10:
        genome.fitness = -1000
        return genome.fitness
    
    score = 0
    
    # --- 1. Smoothness Score ---
    # Calculate velocity between consecutive frames
    velocities = np.linalg.norm(np.diff(frames, axis=0), axis=2)
    mean_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # Reward moderate, consistent movement
    if 0.5 < mean_velocity < 5.0:
        score += 100
    
    # Penalize jerky movement (high velocity variance)
    score -= velocity_std * 10
    
    # --- 2. Transition Smoothness ---
    # Check velocity at segment boundaries
    current_frame = 0
    for i, (dance_idx, start, end) in enumerate(genome.segments[:-1]):
        seg_length = end - start
        transition_frame = current_frame + seg_length - 1
        
        if transition_frame < len(frames) - 1:
            # Velocity at transition
            transition_vel = np.linalg.norm(frames[transition_frame + 1] - frames[transition_frame])
            
            # Penalize large jumps (bad transitions)
            if transition_vel > 10:
                score -= 50
            elif transition_vel < 3:
                score += 20  # Smooth transition bonus
        
        current_frame += seg_length
    
    # --- 3. Genre Diversity Score ---
    genres_used = set()
    for dance_idx, _, _ in genome.segments:
        genres_used.add(all_dances[dance_idx]['genre'])
    
    # Bonus for using multiple genres
    score += len(genres_used) * 30
    
    # --- 4. Physical Plausibility ---
    # Check that feet don't go through floor (Y should be positive for height)
    # In AIST++ data, Y is height
    feet_indices = [15, 16]  # Left and right ankle
    min_foot_height = np.min(frames[:, feet_indices, 1])
    
    if min_foot_height < 0:
        score -= abs(min_foot_height) * 100
    
    # --- 5. Head Above Hips Check ---
    head_idx = 0  # Nose
    hip_indices = [11, 12]  # Left and right hip
    
    head_heights = frames[:, head_idx, 1]
    hip_heights = np.mean(frames[:, hip_indices, 1], axis=1)
    
    upright_ratio = np.mean(head_heights > hip_heights)
    score += upright_ratio * 50
    
    genome.fitness = score
    return score


def tournament_select(population, tournament_size=TOURNAMENT_SIZE):
    """Select parent using tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda g: g.fitness)


def crossover(parent1, parent2, all_dances):
    """Single-point crossover on segments"""
    if random.random() > CROSSOVER_RATE:
        return DanceGenome(parent1.segments.copy())
    
    if len(parent1.segments) < 2 or len(parent2.segments) < 2:
        return DanceGenome(parent1.segments.copy())
    
    # Single point crossover
    point1 = random.randint(1, len(parent1.segments) - 1)
    point2 = random.randint(1, len(parent2.segments) - 1)
    
    child_segments = parent1.segments[:point1] + parent2.segments[point2:]
    
    # Trim to target length
    child = DanceGenome(child_segments)
    total = child.get_total_frames(all_dances)
    
    while total > GENOME_LENGTH and len(child.segments) > 1:
        # Remove last segment or trim it
        dance_idx, start, end = child.segments[-1]
        excess = total - GENOME_LENGTH
        if (end - start) > excess:
            child.segments[-1] = (dance_idx, start, end - excess)
            break
        else:
            child.segments.pop()
            total = child.get_total_frames(all_dances)
    
    return child


def mutate(genome, all_dances):
    """Mutate a genome by replacing, adding, or modifying segments"""
    if random.random() > MUTATION_RATE:
        return genome
    
    mutated_segments = genome.segments.copy()
    
    mutation_type = random.choice(['replace', 'shift', 'resize'])
    
    if mutation_type == 'replace' and len(mutated_segments) > 0:
        # Replace a random segment with a new one
        idx = random.randint(0, len(mutated_segments) - 1)
        old_dance_idx, old_start, old_end = mutated_segments[idx]
        old_length = old_end - old_start
        
        # Pick new dance
        new_dance_idx = random.randint(0, len(all_dances) - 1)
        dance_data = all_dances[new_dance_idx]['data']
        max_start = len(dance_data) - old_length
        if max_start > 0:
            new_start = random.randint(0, max_start)
            mutated_segments[idx] = (new_dance_idx, new_start, new_start + old_length)
    
    elif mutation_type == 'shift' and len(mutated_segments) > 0:
        # Shift a segment's start/end within same dance
        idx = random.randint(0, len(mutated_segments) - 1)
        dance_idx, start, end = mutated_segments[idx]
        dance_data = all_dances[dance_idx]['data']
        
        shift = random.randint(-20, 20)
        new_start = max(0, min(start + shift, len(dance_data) - (end - start)))
        new_end = new_start + (end - start)
        mutated_segments[idx] = (dance_idx, new_start, new_end)
    
    elif mutation_type == 'resize' and len(mutated_segments) > 0:
        # Resize a segment
        idx = random.randint(0, len(mutated_segments) - 1)
        dance_idx, start, end = mutated_segments[idx]
        dance_data = all_dances[dance_idx]['data']
        
        resize = random.randint(-15, 15)
        new_end = max(start + 10, min(end + resize, len(dance_data)))
        mutated_segments[idx] = (dance_idx, start, new_end)
    
    child = DanceGenome(mutated_segments)
    child.frames = None  # Clear cache
    child.fitness = None
    return child


def run_evolution(all_dances):
    """Main GA loop"""
    print(f"\nInitializing population of {POPULATION_SIZE} dances...")
    
    # Initialize population
    population = [create_random_genome(all_dances) for _ in range(POPULATION_SIZE)]
    
    # Evaluate initial fitness
    for genome in population:
        calculate_fitness(genome, all_dances)
    
    population.sort(key=lambda g: g.fitness, reverse=True)
    best_ever = population[0]
    
    print(f"Generation 0 - Best Fitness: {population[0].fitness:.2f}")
    
    for gen in range(1, NUM_GENERATIONS + 1):
        # Create new population
        new_population = []
        
        # Elitism: keep top 2
        new_population.append(population[0])
        new_population.append(population[1])
        
        # Fill rest with offspring
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            child = crossover(parent1, parent2, all_dances)
            child = mutate(child, all_dances)
            
            calculate_fitness(child, all_dances)
            new_population.append(child)
        
        population = new_population
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        if population[0].fitness > best_ever.fitness:
            best_ever = population[0]
        
        if gen % 20 == 0:
            # Show genres used in best
            genres = [all_dances[d[0]]['genre'] for d in population[0].segments]
            genre_str = ', '.join(set(genres))
            print(f"Generation {gen} - Best: {population[0].fitness:.2f} | Genres: {genre_str}")
    
    print(f"\n--- Evolution Complete ---")
    print(f"Best Fitness: {best_ever.fitness:.2f}")
    
    # Show composition of best dance
    print("\nBest dance composition:")
    for i, (dance_idx, start, end) in enumerate(best_ever.segments):
        dance = all_dances[dance_idx]
        print(f"  Segment {i+1}: {dance['name']} ({dance['genre']}) - frames {start}-{end}")
    
    return best_ever


# =============================================================================
# --- VISUALIZATION ---
# =============================================================================

def get_bone_color(bone_idx):
    """Assign colors based on bone index"""
    if bone_idx < 4:
        return BONE_COLORS['head']
    elif bone_idx < 8:
        return BONE_COLORS['torso']
    elif bone_idx < 10:
        return BONE_COLORS['right_arm']
    elif bone_idx < 12:
        return BONE_COLORS['left_arm']
    elif bone_idx < 14:
        return BONE_COLORS['right_leg']
    else:
        return BONE_COLORS['left_leg']


def visualize_dance(frames, title="Generated Dance", all_dances=None, genome=None):
    """Visualize a dance sequence in 3D"""
    print(f"\nPreparing visualization ({len(frames)} frames)...")
    
    # Calculate axis limits
    x_min, x_max = frames[:,:,0].min(), frames[:,:,0].max()
    y_min, y_max = frames[:,:,2].min(), frames[:,:,2].max()  # Z -> plot Y (depth)
    z_min, z_max = frames[:,:,1].min(), frames[:,:,1].max()  # Y -> plot Z (height)
    
    padding = 20
    
    # Setup figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim3d([x_min - padding, x_max + padding])
    ax.set_ylim3d([y_min - padding, y_max + padding])
    ax.set_zlim3d([z_min - padding, z_max + padding])
    ax.set_xlabel('X')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    ax.view_init(elev=10, azim=-60)
    
    # Initialize lines
    lines = []
    for i, _ in enumerate(BONES):
        color = get_bone_color(i)
        line, = ax.plot([], [], [], color=color, lw=2, marker='o', markersize=3)
        lines.append(line)
    
    title_text = ax.set_title('', fontsize=14, fontweight='bold')
    info_text = fig.text(0.02, 0.02, '', fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Build segment info for display
    segment_info = []
    if genome and all_dances:
        current_frame = 0
        for dance_idx, start, end in genome.segments:
            dance = all_dances[dance_idx]
            segment_info.append({
                'start': current_frame,
                'end': current_frame + (end - start),
                'name': dance['name'],
                'genre': dance['genre']
            })
            current_frame += (end - start)
    
    def get_segment_info(frame_idx):
        for seg in segment_info:
            if seg['start'] <= frame_idx < seg['end']:
                return seg
        return segment_info[-1] if segment_info else None
    
    def update(frame_idx):
        current_pose = frames[frame_idx]
        
        for line, bone in zip(lines, BONES):
            start, end = bone
            if start < len(current_pose) and end < len(current_pose):
                xs = [current_pose[start, 0], current_pose[end, 0]]
                ys = [current_pose[start, 2], current_pose[end, 2]]
                zs = [current_pose[start, 1], current_pose[end, 1]]
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
        
        # Update title
        seg = get_segment_info(frame_idx)
        if seg:
            title_text.set_text(f"{title} - {seg['name']} ({seg['genre']})")
        else:
            title_text.set_text(title)
        
        # Update info
        info_str = f"Frame: {frame_idx}/{len(frames)}"
        if seg:
            seg_idx = segment_info.index(seg) + 1
            info_str += f"\nSegment: {seg_idx}/{len(segment_info)}"
        info_text.set_text(info_str)
        
        return lines + [title_text, info_text]
    
    print("Starting animation... Close the window to continue.")
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000/SOURCE_FPS, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()


def save_dance_animation(frames, filename, genome=None, all_dances=None):
    """Save dance animation to file"""
    print(f"\nSaving animation to {filename}...")
    
    # Calculate axis limits
    x_min, x_max = frames[:,:,0].min(), frames[:,:,0].max()
    y_min, y_max = frames[:,:,2].min(), frames[:,:,2].max()
    z_min, z_max = frames[:,:,1].min(), frames[:,:,1].max()
    
    padding = 20
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim3d([x_min - padding, x_max + padding])
    ax.set_ylim3d([y_min - padding, y_max + padding])
    ax.set_zlim3d([z_min - padding, z_max + padding])
    ax.view_init(elev=10, azim=-60)
    
    lines = []
    for i, _ in enumerate(BONES):
        color = get_bone_color(i)
        line, = ax.plot([], [], [], color=color, lw=2, marker='o', markersize=3)
        lines.append(line)
    
    title_text = ax.set_title('', fontsize=12)
    
    def update(frame_idx):
        current_pose = frames[frame_idx]
        
        for line, bone in zip(lines, BONES):
            start, end = bone
            xs = [current_pose[start, 0], current_pose[end, 0]]
            ys = [current_pose[start, 2], current_pose[end, 2]]
            zs = [current_pose[start, 1], current_pose[end, 1]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        
        title_text.set_text(f'Generated Dance - Frame {frame_idx}/{len(frames)}')
        return lines + [title_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/30, blit=False)
    
    # Save as mp4 or gif
    if filename.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(filename, writer=writer)
    else:
        ani.save(filename, writer='pillow', fps=30)
    
    plt.close()
    print(f"Saved to {filename}")


def get_next_filename(base_name="generated_dance", ext="mp4"):
    """Get next available filename"""
    counter = 1
    while True:
        filename = f"{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1


# =============================================================================
# --- MAIN ---
# =============================================================================

def main():
    print("=" * 60)
    print("Novel Dance Generator - GA with AIST++ Dataset")
    print("=" * 60)
    
    # Load data
    print("\nLoading AIST++ dance data...")
    dances_by_genre = load_dance_files()
    
    print("\nAvailable genres:")
    for genre, files in dances_by_genre.items():
        if files:
            print(f"  {GENRE_NAMES.get(genre, genre)} ({genre}): {len(files)} files")
    
    all_dances = load_all_dance_data(dances_by_genre)
    print(f"\nTotal dances loaded: {len(all_dances)}")
    
    # Run evolution
    best_genome = run_evolution(all_dances)
    
    # Render best dance
    frames = best_genome.render(all_dances)
    
    # Visualize
    visualize_dance(frames, "GA Generated Dance", all_dances, best_genome)
    
    # Ask to save
    save = input("\nSave animation? (y/n): ").strip().lower()
    if save == 'y':
        filename = get_next_filename("generated_dance", "mp4")
        save_dance_animation(frames, filename, best_genome, all_dances)


if __name__ == '__main__':
    main()
