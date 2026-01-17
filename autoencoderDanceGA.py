"""
Autoencoder-based Dance Generator with Genetic Algorithm Evaluation
1. Train an autoencoder to learn latent representations of AIST++ dances
2. Use GA to evolve latent vectors that produce novel, high-quality dances
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# --- SETTINGS ---
# =============================================================================
DATA_FOLDER = 'keypoints3d'
SOURCE_FPS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Autoencoder Parameters
SEQUENCE_LENGTH = 60  # 1 second of dance (60 frames at 60 FPS)
LATENT_DIM = 64  # Dimension of latent space
HIDDEN_DIM = 512
NUM_KEYPOINTS = 17
NUM_COORDS = 3
INPUT_DIM = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

# GA Parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.3
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 5
NUM_SEQUENCES = 4  # Number of latent sequences to evolve per genome

# Genre codes from AIST++ dataset
GENRES = ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']
GENRE_NAMES = {
    'BR': 'Break', 'PO': 'Pop', 'LO': 'Lock', 'MH': 'Middle Hip-hop',
    'LH': 'LA Hip-hop', 'HO': 'House', 'WA': 'Waack', 'KR': 'Krump',
    'JS': 'Street Jazz', 'JB': 'Ballet Jazz'
}

# COCO 17-keypoint skeleton connections
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (6, 8), (8, 10),  # Right arm
    (5, 7), (7, 9),   # Left arm
    (12, 14), (14, 16),  # Right leg
    (11, 13), (13, 15)   # Left leg
]

BONE_COLORS = {
    'head': '#FF6B6B', 'torso': '#4ECDC4',
    'right_arm': '#45B7D1', 'left_arm': '#96CEB4',
    'right_leg': '#FFEAA7', 'left_leg': '#DDA0DD'
}


# =============================================================================
# --- DATA LOADING ---
# =============================================================================

def extract_genre(filename):
    return filename[1:3]


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data.get('keypoints3d_optim', data.get('keypoints3d'))


def load_all_sequences(data_folder, sequence_length=SEQUENCE_LENGTH, max_files=50):
    """Load dance data and split into fixed-length sequences
    
    Args:
        data_folder: Path to the data folder
        sequence_length: Number of frames per sequence
        max_files: Maximum number of pkl files to load (None for all)
    """
    sequences = []
    data_path = Path(data_folder)
    
    print("Loading dance sequences...")
    all_files = list(data_path.glob('*.pkl'))
    
    # Limit files if specified
    if max_files is not None:
        all_files = random.sample(all_files, min(max_files, len(all_files)))
        print(f"Using {len(all_files)} files (limited from full dataset)")
    
    for pkl_file in all_files:
        keypoints = load_pkl(pkl_file)
        num_frames = len(keypoints)
        
        # Extract overlapping sequences
        for start in range(0, num_frames - sequence_length, sequence_length // 2):
            seq = keypoints[start:start + sequence_length]
            sequences.append(seq)
    
    print(f"Loaded {len(sequences)} sequences of {sequence_length} frames each")
    return np.array(sequences)


class DanceDataset(Dataset):
    """PyTorch Dataset for dance sequences"""
    def __init__(self, sequences):
        # Normalize the data
        self.mean = sequences.mean(axis=(0, 1), keepdims=True)
        self.std = sequences.std(axis=(0, 1), keepdims=True) + 1e-8
        
        # Normalize and flatten: (N, seq_len, 17, 3) -> (N, seq_len, 51)
        normalized = (sequences - self.mean) / self.std
        self.data = normalized.reshape(len(sequences), SEQUENCE_LENGTH, -1)
        self.data = torch.FloatTensor(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def denormalize(self, data):
        """Convert normalized data back to original scale"""
        # data shape: (batch, seq_len, 51) or (seq_len, 51)
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Reshape to (batch, seq_len, 17, 3) or (seq_len, 17, 3)
        original_shape = data.shape
        if len(original_shape) == 2:
            data = data.reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.std.squeeze() + self.mean.squeeze()
        else:
            data = data.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.std + self.mean
        
        return data


# =============================================================================
# --- AUTOENCODER MODEL ---
# =============================================================================

class DanceEncoder(nn.Module):
    """Encodes a dance sequence into a latent vector"""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        
        # LSTM to process sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Project to latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class DanceDecoder(nn.Module):
    """Decodes a latent vector into a dance sequence"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_len = SEQUENCE_LENGTH
        
        # Project latent to initial hidden state
        self.fc_init = nn.Linear(latent_dim, hidden_dim * 2)
        
        # LSTM to generate sequence
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
    def forward(self, z, target=None, teacher_forcing_ratio=0.5, seq_len=SEQUENCE_LENGTH):
        """Decode latent vector to sequence.
        
        Args:
            z: Latent vector
            target: Target sequence for teacher forcing (during training)
            teacher_forcing_ratio: Probability of using target vs prediction
            seq_len: Length of sequence to generate
        """
        batch_size = z.size(0)
        
        # Initialize hidden state from latent
        h_init = self.fc_init(z)
        h0 = h_init[:, :self.hidden_dim].unsqueeze(0).repeat(2, 1, 1).contiguous()
        c0 = h_init[:, self.hidden_dim:].unsqueeze(0).repeat(2, 1, 1).contiguous()
        
        # Start with learned token
        outputs = []
        current_input = self.start_token.expand(batch_size, -1, -1)
        h, c = h0, c0
        
        use_teacher_forcing = target is not None and random.random() < teacher_forcing_ratio
        
        for t in range(seq_len):
            out, (h, c) = self.lstm(current_input, (h, c))
            frame = self.fc_out(out)
            outputs.append(frame)
            
            # Teacher forcing: use real target as next input
            if use_teacher_forcing and t < seq_len - 1:
                current_input = target[:, t:t+1, :]
            else:
                current_input = frame
        
        return torch.cat(outputs, dim=1)


class DanceVAE(nn.Module):
    """Variational Autoencoder for dance sequences"""
    def __init__(self):
        super().__init__()
        self.encoder = DanceEncoder()
        self.decoder = DanceDecoder()
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, teacher_forcing_ratio=0.5):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return reconstruction, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu
    
    def decode(self, z):
        return self.decoder(z, target=None, teacher_forcing_ratio=0.0)


def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    """VAE loss: reconstruction + KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


# =============================================================================
# --- TRAINING ---
# =============================================================================

def train_autoencoder(model, dataloader, num_epochs=NUM_EPOCHS):
    """Train the VAE"""
    print(f"\nTraining VAE on {DEVICE}...")
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        # Decay teacher forcing over time (start high, end low)
        teacher_forcing_ratio = max(0.2, 1.0 - epoch / num_epochs)
        
        for batch in dataloader:
            batch = batch.to(DEVICE)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Calculate losses (using beta=0.01)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = total_recon / len(dataloader.dataset)
        avg_kl = total_kl / len(dataloader.dataset)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} "
                  f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
    
    print("Training complete!")
    return history


# =============================================================================
# --- GENETIC ALGORITHM ---
# =============================================================================

class LatentGenome:
    """
    Genome is a sequence of latent vectors that will be decoded into dance sequences
    """
    def __init__(self, latent_vectors=None):
        if latent_vectors is None:
            # Initialize with random latent vectors
            self.latent_vectors = [
                np.random.randn(LATENT_DIM).astype(np.float32)
                for _ in range(NUM_SEQUENCES)
            ]
        else:
            self.latent_vectors = latent_vectors
        self.fitness = None
        self.decoded_frames = None


def decode_genome(genome, model, dataset):
    """Decode latent vectors to dance frames"""
    if genome.decoded_frames is not None:
        return genome.decoded_frames
    
    model.eval()
    all_frames = []
    
    with torch.no_grad():
        for latent in genome.latent_vectors:
            z = torch.FloatTensor(latent).unsqueeze(0).to(DEVICE)
            decoded = model.decode(z)
            frames = decoded.cpu().numpy()[0]  # (seq_len, 51)
            all_frames.append(frames)
    
    # Concatenate all sequences
    combined = np.concatenate(all_frames, axis=0)  # (total_frames, 51)
    
    # Denormalize
    combined = combined.reshape(-1, NUM_KEYPOINTS, NUM_COORDS)
    combined = combined * dataset.std.squeeze() + dataset.mean.squeeze()
    
    genome.decoded_frames = combined
    return combined


def calculate_fitness(genome, model, dataset):
    """
    Evaluate fitness based on:
    1. Reconstruction quality (smoothness)
    2. Movement variety
    3. Physical plausibility
    4. Transition smoothness between sequences
    """
    if genome.fitness is not None:
        return genome.fitness
    
    frames = decode_genome(genome, model, dataset)
    
    if len(frames) < 10:
        genome.fitness = -1000
        return genome.fitness
    
    score = 0
    
    # --- 1. Smoothness Score ---
    velocities = np.linalg.norm(np.diff(frames, axis=0), axis=(1, 2))
    mean_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # Reward moderate movement
    if 1.0 < mean_velocity < 10.0:
        score += 100
    elif mean_velocity < 0.5:
        score -= 50  # Too static
    elif mean_velocity > 20:
        score -= 100  # Too jerky
    
    # Penalize high variance (jerky)
    score -= velocity_std * 5
    
    # --- 2. Transition Smoothness ---
    # Check velocity at sequence boundaries
    for i in range(1, NUM_SEQUENCES):
        boundary = i * SEQUENCE_LENGTH
        if boundary < len(frames) - 1:
            transition_vel = np.linalg.norm(frames[boundary] - frames[boundary - 1])
            if transition_vel > 15:
                score -= 30  # Bad transition
            elif transition_vel < 5:
                score += 20  # Smooth transition
    
    # --- 3. Physical Plausibility ---
    # Feet should stay relatively low (ankles are indices 15, 16)
    ankle_heights = frames[:, [15, 16], 1]  # Y is height
    min_ankle = np.min(ankle_heights)
    max_ankle = np.max(ankle_heights)
    
    # Penalize if ankles go too high or negative
    if min_ankle < 0:
        score -= abs(min_ankle) * 50
    
    # --- 4. Head Above Hips ---
    head_y = frames[:, 0, 1]  # Nose Y
    hip_y = np.mean(frames[:, [11, 12], 1], axis=1)  # Average hip Y
    
    upright_ratio = np.mean(head_y > hip_y)
    score += upright_ratio * 80
    
    # --- 5. Movement Variety ---
    # Reward diverse movements (different poses)
    pose_variance = np.var(frames, axis=0).mean()
    score += min(pose_variance * 2, 50)  # Cap bonus
    
    # --- 6. Latent Space Regularization ---
    # Penalize extreme latent values
    for latent in genome.latent_vectors:
        latent_norm = np.linalg.norm(latent)
        if latent_norm > 5:
            score -= (latent_norm - 5) * 10
    
    genome.fitness = score
    return score


def tournament_select(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=lambda g: g.fitness)


def crossover(parent1, parent2):
    """Crossover latent vectors"""
    if random.random() > CROSSOVER_RATE:
        return LatentGenome([v.copy() for v in parent1.latent_vectors])
    
    child_vectors = []
    for i in range(NUM_SEQUENCES):
        if random.random() < 0.5:
            child_vectors.append(parent1.latent_vectors[i].copy())
        else:
            child_vectors.append(parent2.latent_vectors[i].copy())
    
    return LatentGenome(child_vectors)


def mutate(genome):
    """Mutate latent vectors with Gaussian noise"""
    mutated_vectors = []
    
    for latent in genome.latent_vectors:
        if random.random() < MUTATION_RATE:
            # Add Gaussian noise
            noise = np.random.randn(LATENT_DIM).astype(np.float32) * MUTATION_STRENGTH
            mutated = latent + noise
            mutated_vectors.append(mutated)
        else:
            mutated_vectors.append(latent.copy())
    
    child = LatentGenome(mutated_vectors)
    return child


def interpolate_latent(genome):
    """Mutate by interpolating between two latent vectors"""
    if len(genome.latent_vectors) < 2:
        return genome
    
    mutated_vectors = genome.latent_vectors.copy()
    
    # Pick two random vectors to interpolate
    i, j = random.sample(range(len(mutated_vectors)), 2)
    alpha = random.random()
    new_vector = alpha * mutated_vectors[i] + (1 - alpha) * mutated_vectors[j]
    
    # Replace one of them
    mutated_vectors[random.choice([i, j])] = new_vector
    
    return LatentGenome(mutated_vectors)


def run_ga_evolution(model, dataset):
    """Run genetic algorithm to find novel dances"""
    print(f"\nInitializing GA population of {POPULATION_SIZE}...")
    
    # Initialize population
    population = [LatentGenome() for _ in range(POPULATION_SIZE)]
    
    # Evaluate initial fitness
    for genome in population:
        calculate_fitness(genome, model, dataset)
    
    population.sort(key=lambda g: g.fitness, reverse=True)
    best_ever = population[0]
    
    print(f"Generation 0 - Best Fitness: {population[0].fitness:.2f}")
    
    for gen in range(1, NUM_GENERATIONS + 1):
        new_population = []
        
        # Elitism
        new_population.append(population[0])
        new_population.append(population[1])
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            child = crossover(parent1, parent2)
            child = mutate(child)
            
            # Occasionally do interpolation mutation
            if random.random() < 0.1:
                child = interpolate_latent(child)
            
            calculate_fitness(child, model, dataset)
            new_population.append(child)
        
        population = new_population
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        if population[0].fitness > best_ever.fitness:
            best_ever = population[0]
        
        if gen % 10 == 0:
            print(f"Generation {gen} - Best: {population[0].fitness:.2f}, "
                  f"Avg: {np.mean([g.fitness for g in population]):.2f}")
    
    print(f"\n--- GA Evolution Complete ---")
    print(f"Best Fitness: {best_ever.fitness:.2f}")
    
    return best_ever


# =============================================================================
# --- VISUALIZATION ---
# =============================================================================

def get_bone_color(bone_idx):
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


def visualize_dance(frames, title="Generated Dance"):
    """Visualize dance in 3D"""
    print(f"\nPreparing visualization ({len(frames)} frames)...")
    
    x_min, x_max = frames[:,:,0].min(), frames[:,:,0].max()
    y_min, y_max = frames[:,:,2].min(), frames[:,:,2].max()
    z_min, z_max = frames[:,:,1].min(), frames[:,:,1].max()
    
    padding = 20
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim3d([x_min - padding, x_max + padding])
    ax.set_ylim3d([y_min - padding, y_max + padding])
    ax.set_zlim3d([z_min - padding, z_max + padding])
    ax.set_xlabel('X')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    ax.view_init(elev=10, azim=-60)
    
    lines = []
    for i, _ in enumerate(BONES):
        color = get_bone_color(i)
        line, = ax.plot([], [], [], color=color, lw=2, marker='o', markersize=3)
        lines.append(line)
    
    title_text = ax.set_title('', fontsize=14, fontweight='bold')
    info_text = fig.text(0.02, 0.02, '', fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update(frame_idx):
        current_pose = frames[frame_idx]
        
        for line, bone in zip(lines, BONES):
            start, end = bone
            xs = [current_pose[start, 0], current_pose[end, 0]]
            ys = [current_pose[start, 2], current_pose[end, 2]]
            zs = [current_pose[start, 1], current_pose[end, 1]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        
        seq_num = frame_idx // SEQUENCE_LENGTH + 1
        title_text.set_text(f"{title} - Sequence {seq_num}/{NUM_SEQUENCES}")
        info_text.set_text(f"Frame: {frame_idx}/{len(frames)}")
        
        return lines + [title_text, info_text]
    
    print("Starting animation... Close the window to continue.")
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000/SOURCE_FPS, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()


def save_animation(frames, filename):
    """Save animation to file"""
    print(f"\nSaving animation to {filename}...")
    
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
        line, = ax.plot([], [], [], color=get_bone_color(i), lw=2, marker='o', markersize=3)
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
        title_text.set_text(f'VAE+GA Generated Dance - Frame {frame_idx}/{len(frames)}')
        return lines + [title_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/30, blit=False)
    
    if filename.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(filename, writer=writer)
    else:
        ani.save(filename, writer='pillow', fps=30)
    
    plt.close()
    print(f"Saved to {filename}")


def get_next_filename(base_name="vae_ga_dance", ext="mp4"):
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
    print("Autoencoder + GA Dance Generator")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load data
    sequences = load_all_sequences(DATA_FOLDER)
    dataset = DanceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create and train VAE
    model = DanceVAE()
    
    # Check for saved model
    model_path = 'dance_vae_model.pth'
    if os.path.exists(model_path):
        print(f"\nLoading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        # Train the model
        history = train_autoencoder(model, dataloader)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['recon_loss'], label='Reconstruction')
        plt.plot(history['kl_loss'], label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    # Run GA to find novel dances
    best_genome = run_ga_evolution(model, dataset)
    
    # Get the final dance
    frames = decode_genome(best_genome, model, dataset)
    
    # Visualize
    visualize_dance(frames, "VAE + GA Generated Dance")
    
    # Ask to save
    save = input("\nSave animation? (y/n): ").strip().lower()
    if save == 'y':
        filename = get_next_filename("vae_ga_dance", "mp4")
        save_animation(frames, filename)


if __name__ == '__main__':
    main()
