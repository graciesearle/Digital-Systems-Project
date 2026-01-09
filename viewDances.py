import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- SETTINGS ---
INPUT_FILE = 'dance_level_e _k_ch0.npy'  # Put your actual filename here
OUTPUT_FILE = 'dance_video.mp4'
SAMPLE_INDEX = 0         # Which dance to save (0 to 99)
REAL_TIME_SPEED = True   # True = normal speed; False = Slow Motion (smooth)
# ----------------

def save_dance_video():
    # 1. Load Data
    try:
        full_data = np.load(INPUT_FILE)
        print(f"Loaded file. Shape: {full_data.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Check the filename!")
        return

    # Extract the specific dance sample
    # Shape becomes (3, 1000, 21) -> (Coords, Frames, Joints)
    raw_dance = full_data[SAMPLE_INDEX]
    
    # Transpose to (Frames, Joints, Coords) for plotting
    # Result: (1000, 21, 3)
    dance_data = raw_dance.transpose(1, 2, 0)

    # 2. Handle Speed Logic
    # Original data is 100 FPS. Video standard is 30 FPS.
    if REAL_TIME_SPEED:
        # To look real-time, we need to skip frames.
        # 100 / 30 approx 3.33. We take every 3rd frame.
        step = 3 
        fps_out = 33 # Closest match to keep timing right
        print("Rendering at Real-Time speed (skipping frames)...")
    else:
        # Render every single frame at 60 FPS (Slow Motion / High Detail)
        step = 1
        fps_out = 60
        print("Rendering in Slow Motion (high detail)...")

    dance_subset = dance_data[::step] 

    # 3. Define Skeleton (ImperialDance Format)
    bones = [
        (19, 2), (2, 0), (0, 10), (10, 1),   # Right Leg
        (19, 3), (3, 12), (12, 4), (4, 15),  # Left Leg
        (19, 16), (16, 13), (13, 11), (11, 14), # Spine
        (13, 5), (5, 17), (17, 20), (20, 7), # Right Arm
        (13, 9), (9, 18), (18, 8), (8, 6)    # Left Arm
    ]

    # 4. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fix the camera view so it doesn't wobble
    radius = 1.2
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius]) # Note: In this dataset, Z is up, Y is depth
    ax.set_zlim3d([0, 2.0])
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Depth (Y)')
    ax.set_zlabel('Height (Z)')

    # Initialize lines
    lines = [ax.plot([], [], [], 'bo-', lw=2, markersize=4)[0] for _ in bones]

    def update(frame_idx):
        current_pose = dance_subset[frame_idx]
        
        for line, bone in zip(lines, bones):
            start, end = bone
            # ImperialDataset is usually (X, Y, Z). 
            # We plot X as x, Y as y, Z as z.
            xs = [current_pose[start, 0], current_pose[end, 0]]
            ys = [current_pose[start, 1], current_pose[end, 1]]
            zs = [current_pose[start, 2], current_pose[end, 2]]
            
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{len(dance_subset)}...")

    # 5. Write Video
    print(f"Starting video export to {OUTPUT_FILE}...")
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(dance_subset), blit=False
    )
    
    writer = animation.FFMpegWriter(fps=fps_out, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(OUTPUT_FILE, writer=writer)
    
    print(f"Success! Video saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    save_dance_video()