import random
import math
import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from moviepy import VideoFileClip, AudioFileClip
import librosa

# ==========================================
# 1. SETTINGS & CONSTANTS
# ==========================================
LATENT_DIM = 4           
SEQUENCE_LENGTH = 16     
JOINTS_COUNT = 11        
INPUT_SIZE = SEQUENCE_LENGTH * JOINTS_COUNT * 3 
TARGET_FPS = 20.0

# ==========================================
# 2. THE "BODY" (Autoencoder)
# ==========================================
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, INPUT_SIZE),
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent

# ==========================================
# 3. DATA LOADING (SAFETY VALVES ADDED)
# ==========================================
def load_and_normalize_data(folder_path, max_files=50):
    print("Loading Data...")
    training_data = []
    
    files = glob.glob(f"{folder_path}/**/*.pkl", recursive=True)[:max_files]
    if not files: 
        print("No files found! Check path."); return [], 1.0

    all_poses = []

    for f in files:
        try:
            with open(f, 'rb') as pkl: 
                data = pickle.load(pkl)['keypoints3d'][::3] 
            
            for frame in data:
                # 1. Swap Axes (Y-up to Z-up)
                swapped = np.array([[p[0], p[2], p[1]] for p in frame])
                
                # 2. Center Hips at (0,0,0)
                # SAFETY 1: Check if hip data exists/is valid
                hips = (swapped[11] + swapped[12]) / 2.0
                centered = swapped - hips
                
                # 3. Filter to 11 joints
                c_neck = (centered[5] + centered[6]) / 2.0
                c_hips = (centered[11] + centered[12]) / 2.0 
                
                final_pose = np.array([
                    c_hips, c_neck, centered[0],      
                    centered[7], centered[9],         
                    centered[8], centered[10],        
                    centered[13], centered[15],       
                    centered[14], centered[16]        
                ])
                all_poses.append(final_pose)
        except: pass

    if len(all_poses) == 0:
        print("Error: Files found but no valid poses extracted.")
        return [], 1.0

    # 4. Normalize to 0-1 Range
    all_poses = np.array(all_poses)
    
    # SAFETY 2: Ensure scale factor is never Zero
    max_val = np.max(np.abs(all_poses))
    if max_val == 0: max_val = 1.0 
    scale_factor = max_val * 1.2 
    
    normalized = (all_poses / (2 * scale_factor)) + 0.5
    
    # 5. Create Sequences (SAFETY 3: Filter NaNs)
    num_sequences = len(normalized) // SEQUENCE_LENGTH
    
    for i in range(num_sequences):
        seq = normalized[i*SEQUENCE_LENGTH : (i+1)*SEQUENCE_LENGTH]
        flat_seq = seq.flatten()
        
        # FINAL CHECK: If this sequence has ANY NaN, throw it away
        if not np.isnan(flat_seq).any():
            training_data.append(flat_seq)
        
    print(f"Loaded {len(training_data)} clean sequences. Scale Factor: {scale_factor:.2f}")
    return training_data, scale_factor

# ==========================================
# 4. THE "CHOREOGRAPHER" (Genetic Algo)
# ==========================================
def calculate_fitness(genome, ae_model, beat_frames):
    decoded_motion = []
    with torch.no_grad():
        for gene in genome:
            z = torch.tensor(gene, dtype=torch.float32).unsqueeze(0)
            motion_chunk = ae_model.decoder(z).numpy().reshape(SEQUENCE_LENGTH, JOINTS_COUNT, 3)
            decoded_motion.extend(motion_chunk)
            
    score = 0
    # Reward velocity on beat frames
    for beat_idx in beat_frames:
        if beat_idx < len(decoded_motion) - 1:
            curr = decoded_motion[beat_idx]
            next_p = decoded_motion[beat_idx+1]
            # Simple Euclidean distance sum
            velocity = np.sum(np.sqrt(np.sum((curr - next_p)**2, axis=1)))
            score += velocity * 10 
            
    return score

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def run_simple_hybrid():
    # A. Load Audio
    # Replace with your actual filename
    audio_file = "Supernatural (Short).mp3" 
    if not os.path.exists(audio_file):
        print("Audio file not found!"); return

    y, sr = librosa.load(audio_file)
    tempo, beat_frames_raw = librosa.beat.beat_track(y=y, sr=sr)
    
    video_beat_indices = librosa.frames_to_time(beat_frames_raw, sr=sr) * TARGET_FPS
    video_beat_indices = [int(b) for b in video_beat_indices]
    
    total_frames = int(librosa.get_duration(y=y, sr=sr) * TARGET_FPS)
    num_genes_needed = math.ceil(total_frames / SEQUENCE_LENGTH)
    
    print(f"Song Duration: {total_frames} frames. Genome Size: {num_genes_needed}")

    # B. Train Autoencoder
    # Make sure this folder name matches your real folder
    data, scale = load_and_normalize_data("AISTpop") 
    if not data: return
    
    ae = SimpleAutoencoder()
    opt = optim.Adam(ae.parameters(), lr=0.001)
    dataset = torch.utils.data.DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=32, shuffle=True)
    
    print("Training Body (Autoencoder)...")
    for epoch in range(40):
        total_loss = 0
        batch_count = 0
        for batch in dataset:
            opt.zero_grad()
            recon, _ = ae(batch)
            loss = nn.MSELoss()(recon, batch)
            
            # SAFETY 4: Skip bad gradients
            if torch.isnan(loss): continue
            
            loss.backward()
            
            # SAFETY 5: Clip Gradients (Prevents Exploding Math)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            
            opt.step()
            total_loss += loss.item()
            batch_count += 1
            
        if batch_count > 0 and epoch % 10 == 0: 
            print(f"Epoch {epoch}: {total_loss/batch_count:.5f}")

    # C. Evolve Choreography
    print("Training Brain (Evolution)...")
    POP_SIZE = 50
    population = [[np.random.rand(LATENT_DIM) for _ in range(num_genes_needed)] for _ in range(POP_SIZE)]
    
    for gen in range(15):
        scores = [(g, calculate_fitness(g, ae, video_beat_indices)) for g in population]
        scores.sort(key=lambda x: x[1], reverse=True)
        population = [s[0] for s in scores]
        
        print(f"Gen {gen}: Best Score {scores[0][1]:.2f}")
        
        new_pop = population[:10]
        while len(new_pop) < POP_SIZE:
            parent1 = random.choice(population[:20])
            parent2 = random.choice(population[:20])
            cut = random.randint(1, max(1, num_genes_needed-1))
            child = parent1[:cut] + parent2[cut:]
            
            if random.random() < 0.3:
                idx = random.randint(0, num_genes_needed-1)
                child[idx] = np.random.rand(LATENT_DIM)
            new_pop.append(child)
        population = new_pop

    # D. Save Result
    best_genome = population[0]
    final_motion = []
    with torch.no_grad():
        for gene in best_genome:
            z = torch.tensor(gene, dtype=torch.float32).unsqueeze(0)
            chunk = ae.decoder(z).numpy().reshape(SEQUENCE_LENGTH, JOINTS_COUNT, 3)
            final_motion.extend(chunk)
            
    final_motion = (np.array(final_motion) - 0.5) * (2 * scale)
    
    print("Saving Video...")
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simple bone connections for visualization
    bones = [(0,1), (1,2), (1,3), (3,4), (1,5), (5,6), (0,7), (7,8), (0,9), (9,10)]
    
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-scale, scale); ax.set_ylim(-scale, scale); ax.set_zlim(-scale, scale)
        pose = final_motion[frame_idx] if frame_idx < len(final_motion) else final_motion[-1]
        
        # Floor
        x, y = np.meshgrid(np.linspace(-scale, scale, 5), np.linspace(-scale, scale, 5))
        ax.plot_surface(x, y, x*0 - (scale*0.8), alpha=0.1, color='k')
        
        for i, j in bones:
            ax.plot([pose[i,0], pose[j,0]], [pose[i,1], pose[j,1]], [pose[i,2], pose[j,2]], 'r-', lw=2)

    ani = FuncAnimation(fig, update, frames=len(final_motion), interval=50)
    ani.save("simple_result.mp4", writer='ffmpeg', fps=20)
    
    try:
        vc = VideoFileClip("simple_result.mp4")
        ac = AudioFileClip(audio_file).subclipped(0, vc.duration)
        vc.with_audio(ac).write_videofile("FINAL_DANCE.mp4", codec='libx264', audio_codec='aac')
    except:
        print("Could not add audio, saved silent video.")

if __name__ == "__main__":
    run_simple_hybrid()