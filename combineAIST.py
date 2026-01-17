import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import random
from pathlib import Path

# --- SETTINGS ---
DATA_FOLDER = 'keypoints3d'
NUM_DANCES = 5
FRAMES_PER_DANCE = 200  # Limit frames per dance for quicker viewing
SOURCE_FPS = 60  # AIST++ dataset is recorded at 60 FPS
PLAYBACK_FPS = 60  # Target playback FPS (set equal to SOURCE_FPS for real-time)
REAL_TIME = True  # True = normal speed, False = slow motion (half speed)

# Genre codes from AIST++ dataset
GENRES = ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']
GENRE_NAMES = {
    'BR': 'Break',
    'PO': 'Pop', 
    'LO': 'Lock',
    'MH': 'Middle Hip-hop',
    'LH': 'LA Hip-hop',
    'HO': 'House',
    'WA': 'Waack',
    'KR': 'Krump',
    'JS': 'Street Jazz',
    'JB': 'Ballet Jazz'
}

# COCO 17-keypoint skeleton connections
BONES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose, eyes, ears
    # Torso
    (5, 6),   # shoulders
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    # Right arm
    (6, 8), (8, 10),
    # Left arm
    (5, 7), (7, 9),
    # Right leg
    (12, 14), (14, 16),
    # Left leg
    (11, 13), (13, 15)
]

# Colors for different body parts
BONE_COLORS = {
    'head': '#FF6B6B',
    'torso': '#4ECDC4',
    'right_arm': '#45B7D1',
    'left_arm': '#96CEB4',
    'right_leg': '#FFEAA7',
    'left_leg': '#DDA0DD'
}


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
    """Load a single pkl file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def select_random_dances(dances_by_genre, num_dances=5):
    """Select random dances ensuring genre variety"""
    selected = []
    available_genres = [g for g in GENRES if len(dances_by_genre[g]) > 0]
    
    # Try to get different genres
    chosen_genres = random.sample(available_genres, min(num_dances, len(available_genres)))
    
    # If we need more, allow repeats
    while len(chosen_genres) < num_dances:
        chosen_genres.append(random.choice(available_genres))
    
    for genre in chosen_genres:
        dance_file = random.choice(dances_by_genre[genre])
        selected.append((genre, dance_file))
    
    return selected


def concatenate_dances(selected_dances, frames_per_dance=200):
    """Load and concatenate dance sequences"""
    all_frames = []
    dance_info = []  # Store (start_frame, end_frame, genre, filename)
    
    current_frame = 0
    for genre, filepath in selected_dances:
        data = load_pkl(filepath)
        
        # Extract the keypoints array from the dictionary
        # Use 'keypoints3d_optim' (optimized) or 'keypoints3d'
        keypoints = data.get('keypoints3d_optim', data.get('keypoints3d'))
        
        # Limit frames per dance
        num_frames = min(len(keypoints), frames_per_dance)
        frames = keypoints[:num_frames]
        
        dance_info.append({
            'start': current_frame,
            'end': current_frame + num_frames,
            'genre': genre,
            'name': GENRE_NAMES.get(genre, genre),
            'file': filepath.name
        })
        
        all_frames.append(frames)
        current_frame += num_frames
    
    return np.concatenate(all_frames, axis=0), dance_info


def visualize_dances():
    """Main visualization function"""
    print("Loading dance files...")
    dances_by_genre = load_dance_files()
    
    # Print available genres
    print("\nAvailable genres:")
    for genre, files in dances_by_genre.items():
        if len(files) > 0:
            print(f"  {GENRE_NAMES.get(genre, genre)} ({genre}): {len(files)} files")
    
    print(f"\nSelecting {NUM_DANCES} random dances from different genres...")
    selected = select_random_dances(dances_by_genre, NUM_DANCES)
    
    print("\nSelected dances:")
    for i, (genre, filepath) in enumerate(selected, 1):
        print(f"  {i}. {GENRE_NAMES.get(genre, genre)} - {filepath.name}")
    
    print("\nLoading and concatenating dance data...")
    all_data, dance_info = concatenate_dances(selected, FRAMES_PER_DANCE)
    
    # Handle playback speed
    # AIST++ is 60 FPS - we play every frame at the correct interval for real-time
    if REAL_TIME:
        step = 1  # Use all frames
        fps_out = SOURCE_FPS  # Play at original speed (60 FPS)
    else:
        step = 1  # Use all frames
        fps_out = SOURCE_FPS // 2  # Half speed (30 FPS)
    
    dance_subset = all_data[::step]
    total_frames = len(dance_subset)
    
    print(f"\nTotal frames to animate: {total_frames}")
    print(f"Playback at {fps_out} FPS ({'real-time' if REAL_TIME else 'slow motion'})")
    
    # Calculate axis limits from the data
    # Note: In this dataset, Y is height (vertical), Z is depth
    # We swap Y and Z for plotting: X stays X, Y becomes depth, Z becomes height
    x_min, x_max = all_data[:,:,0].min(), all_data[:,:,0].max()
    y_min, y_max = all_data[:,:,2].min(), all_data[:,:,2].max()  # Z -> plot Y (depth)
    z_min, z_max = all_data[:,:,1].min(), all_data[:,:,1].max()  # Y -> plot Z (height)
    
    # Add some padding
    padding = 20
    
    # Setup figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the axes based on actual data range
    ax.set_xlim3d([x_min - padding, x_max + padding])
    ax.set_ylim3d([y_min - padding, y_max + padding])
    ax.set_zlim3d([z_min - padding, z_max + padding])
    ax.set_xlabel('X')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    
    # Set camera to eye level with the dancer
    # elev=10 is near eye level, azim=-60 gives a nice front-side view
    ax.view_init(elev=10, azim=-60)
    
    # Initialize lines for bones
    lines = []
    for i, _ in enumerate(BONES):
        color = get_bone_color(i)
        line, = ax.plot([], [], [], color=color, lw=2, marker='o', markersize=3)
        lines.append(line)
    
    # Title text
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    # Info text
    info_text = fig.text(0.02, 0.02, '', fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def get_current_dance_info(frame_idx):
        """Get the dance info for the current frame"""
        actual_frame = frame_idx * step
        for info in dance_info:
            if info['start'] <= actual_frame < info['end']:
                return info
        return dance_info[-1]
    
    def update(frame_idx):
        current_pose = dance_subset[frame_idx]
        
        for line, bone in zip(lines, BONES):
            start, end = bone
            if start < len(current_pose) and end < len(current_pose):
                xs = [current_pose[start, 0], current_pose[end, 0]]  # X stays X
                ys = [current_pose[start, 2], current_pose[end, 2]]  # Z -> plot Y (depth)
                zs = [current_pose[start, 1], current_pose[end, 1]]  # Y -> plot Z (height)
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
        
        # Update title with current dance info
        dance = get_current_dance_info(frame_idx)
        title.set_text(f"Genre: {dance['name']} ({dance['genre']})")
        
        # Update info text
        dance_num = dance_info.index(dance) + 1
        info_str = f"Dance {dance_num}/{NUM_DANCES}\nFrame: {frame_idx}/{total_frames}\nFile: {dance['file']}"
        info_text.set_text(info_str)
        
        return lines + [title, info_text]
    
    print("\nStarting animation...")
    print("Close the window to exit.")
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000/fps_out, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_dances()
