"""
Music-Conditioned Autoencoder Dance Generator with Genetic Algorithm
Extends autoencoderDanceGA.py to generate dances conditioned on music input.

1. Audio feature extraction using librosa (mel spectrograms, beats, tempo)
2. AudioEncoder to process music features
3. Conditional decoder that takes both latent vector and audio features
4. Music-dance synchronisation in fitness function
5. Paired music-dance dataset loading
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import random
import os
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import warnings
warnings.filterwarnings('ignore')

# Try to import librosa for audio processing
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not installed. Run: pip install librosa")
    LIBROSA_AVAILABLE = False

# =============================================================================
# --- SETTINGS ---
# =============================================================================
DATA_FOLDER = 'keypoints3d'
MUSIC_FOLDER = 'AISTmusic'
SOURCE_FPS = 60
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Audio Parameters
AUDIO_SR = 22050  # Audio sample rate
N_MELS = 80  # Number of mel bands
HOP_LENGTH = 512  # Hop length for STFT (gives ~43 fps at 22050 Hz)
AUDIO_FPS = AUDIO_SR / HOP_LENGTH  # ~43 fps

# Autoencoder Parameters
SEQUENCE_LENGTH = 60  # 1 second of dance (60 frames at 60 FPS)
AUDIO_FRAMES = int(SEQUENCE_LENGTH * AUDIO_FPS / SOURCE_FPS)  # Corresponding audio frames
LATENT_DIM = 128
HIDDEN_DIM = 512
AUDIO_HIDDEN_DIM = 256
AUDIO_EMBED_DIM = 64  # Audio embedding dimension
NUM_KEYPOINTS = 17
NUM_COORDS = 3
INPUT_DIM = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
KL_WARMUP_EPOCHS = 20

# GA Parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 5
NUM_SEQUENCES = 4

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
# --- AUDIO PROCESSING ---
# =============================================================================

def extract_music_id(filename):
    """Extract music ID from dance filename (e.g., 'mBR0' from 'gBR_sBM_cAll_d04_mBR0_ch01.pkl')"""
    match = re.search(r'm([A-Z]{2}\d)', filename)
    if match:
        return 'm' + match.group(1)
    return None


def load_audio_features(audio_path, duration=None):
    """
    Load and extract features from an audio file.
    
    Returns:
        mel_spec: Mel spectrogram (n_mels, time_frames)
        beats: Beat frames
        tempo: Estimated tempo
        onset_env: Onset strength envelope
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio processing")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=AUDIO_SR, duration=duration)
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalise mel spectrogram
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_env = (onset_env - onset_env.mean()) / (onset_env.std() + 1e-8)
    
    # Chroma features (pitch content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    return {
        'mel_spec': mel_spec_db,
        'tempo': tempo,
        'beat_frames': beat_frames,
        'onset_env': onset_env,
        'chroma': chroma
    }


def resample_audio_to_dance_fps(audio_features, num_dance_frames):
    """
    Resample audio features to match dance frame rate.
    
    Args:
        audio_features: Dict with mel_spec, onset_env, etc.
        num_dance_frames: Number of dance frames to align to
    
    Returns:
        Resampled features aligned to dance frames
    """
    mel_spec = audio_features['mel_spec']  # (n_mels, audio_frames)
    onset_env = audio_features['onset_env']  # (audio_frames,)
    
    audio_frames = mel_spec.shape[1]
    
    # Create interpolation indices
    audio_indices = np.linspace(0, audio_frames - 1, num_dance_frames).astype(int)
    
    # Resample mel spectrogram
    mel_resampled = mel_spec[:, audio_indices]  # (n_mels, num_dance_frames)
    
    # Resample onset envelope
    onset_resampled = onset_env[audio_indices]
    
    # Create beat mask (1 where beat occurs, 0 otherwise)
    beat_mask = np.zeros(num_dance_frames)
    beat_frames = audio_features['beat_frames']
    for bf in beat_frames:
        # Map audio beat frame to dance frame
        dance_frame = int(bf * num_dance_frames / audio_frames)
        if 0 <= dance_frame < num_dance_frames:
            beat_mask[dance_frame] = 1.0
    
    return {
        'mel_spec': mel_resampled,  # (n_mels, num_dance_frames)
        'onset_env': onset_resampled,  # (num_dance_frames,)
        'beat_mask': beat_mask,  # (num_dance_frames,)
        'tempo': audio_features['tempo']
    }


# =============================================================================
# --- DATA LOADING ---
# =============================================================================

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data.get('keypoints3d_optim', data.get('keypoints3d'))


def load_music_cache(music_folder):
    """Pre-load all music features into a cache"""
    music_cache = {}
    music_path = Path(music_folder)
    
    if not music_path.exists():
        print(f"Warning: Music folder {music_folder} not found")
        return music_cache
    
    print("Loading music features...")
    for mp3_file in music_path.glob('*.mp3'):
        music_id = mp3_file.stem  # e.g., 'mBR0'
        try:
            features = load_audio_features(str(mp3_file))
            music_cache[music_id] = features
            print(f"  Loaded {music_id}")
        except Exception as e:
            print(f"  Error loading {music_id}: {e}")
    
    print(f"Loaded {len(music_cache)} music tracks")
    return music_cache


def load_paired_sequences(data_folder, music_folder, sequence_length=SEQUENCE_LENGTH, max_files=200):
    """
    Load dance sequences paired with their corresponding music features.
    
    Returns:
        dance_sequences: numpy array of shape (N, seq_len, 17, 3)
        audio_sequences: numpy array of shape (N, n_mels + 2, seq_len)
        music_ids: list of music IDs for reference
    """
    dance_sequences = []
    audio_sequences = []
    music_ids = []
    
    data_path = Path(data_folder)
    music_cache = load_music_cache(music_folder)
    
    if not music_cache:
        print("No music loaded - falling back to unconditional mode")
        return load_all_sequences_unconditional(data_folder, sequence_length, max_files)
    
    print("\nLoading paired dance-music sequences...")
    all_files = list(data_path.glob('*.pkl'))
    
    if max_files is not None:
        all_files = random.sample(all_files, min(max_files, len(all_files)))
    
    for pkl_file in all_files:
        # Extract music ID from filename
        music_id = extract_music_id(pkl_file.name)
        
        if music_id is None or music_id not in music_cache:
            continue
        
        # Load dance keypoints
        keypoints = load_pkl(pkl_file)
        num_frames = len(keypoints)
        
        # Get audio features
        audio_features = music_cache[music_id]
        
        # Extract overlapping sequences
        for start in range(0, num_frames - sequence_length, sequence_length // 2):
            end = start + sequence_length
            
            # Dance sequence
            dance_seq = keypoints[start:end]
            
            # Calculate corresponding time in audio
            start_time = start / SOURCE_FPS
            end_time = end / SOURCE_FPS
            
            # Get audio frames for this segment
            audio_start = int(start_time * AUDIO_FPS)
            audio_end = int(end_time * AUDIO_FPS)
            
            mel_spec = audio_features['mel_spec']
            onset_env = audio_features['onset_env']
            
            if audio_end > mel_spec.shape[1]:
                continue
            
            # Extract audio segment and resample to match dance frames
            segment_features = {
                'mel_spec': mel_spec[:, audio_start:audio_end],
                'onset_env': onset_env[audio_start:audio_end],
                'beat_frames': audio_features['beat_frames'] - audio_start,
                'tempo': audio_features['tempo']
            }
            
            resampled = resample_audio_to_dance_fps(segment_features, sequence_length)
            
            # Combine mel + onset + beat into single tensor
            # Shape: (n_mels + 2, seq_len)
            audio_combined = np.vstack([
                resampled['mel_spec'],  # (n_mels, seq_len)
                resampled['onset_env'].reshape(1, -1),  # (1, seq_len)
                resampled['beat_mask'].reshape(1, -1)   # (1, seq_len)
            ])
            
            dance_sequences.append(dance_seq)
            audio_sequences.append(audio_combined)
            music_ids.append(music_id)
    
    print(f"Loaded {len(dance_sequences)} paired sequences")
    
    return np.array(dance_sequences), np.array(audio_sequences), music_ids


def load_all_sequences_unconditional(data_folder, sequence_length=SEQUENCE_LENGTH, max_files=200):
    """Fallback: load dance sequences without music (for compatibility)"""
    sequences = []
    data_path = Path(data_folder)
    
    print("Loading dance sequences (unconditional mode)...")
    all_files = list(data_path.glob('*.pkl'))
    
    if max_files is not None:
        all_files = random.sample(all_files, min(max_files, len(all_files)))
    
    for pkl_file in all_files:
        keypoints = load_pkl(pkl_file)
        num_frames = len(keypoints)
        
        for start in range(0, num_frames - sequence_length, sequence_length // 2):
            seq = keypoints[start:start + sequence_length]
            sequences.append(seq)
    
    print(f"Loaded {len(sequences)} sequences")
    return np.array(sequences), None, None


class MusicDanceDataset(Dataset):
    """PyTorch Dataset for paired music-dance sequences"""
    
    def __init__(self, dance_sequences, audio_sequences=None):
        # Normalise dance data
        self.dance_mean = dance_sequences.mean(axis=(0, 1), keepdims=True)
        self.dance_std = dance_sequences.std(axis=(0, 1), keepdims=True) + 1e-8
        
        # Normalise and flatten dance: (N, seq_len, 17, 3) -> (N, seq_len, 51)
        normalised_dance = (dance_sequences - self.dance_mean) / self.dance_std
        self.dance_data = normalised_dance.reshape(len(dance_sequences), SEQUENCE_LENGTH, -1)
        self.dance_data = torch.FloatTensor(self.dance_data)
        
        # Audio data (already normalised during extraction)
        self.has_audio = audio_sequences is not None
        if self.has_audio:
            self.audio_data = torch.FloatTensor(audio_sequences)  # (N, n_mels+2, seq_len)
        else:
            # Create dummy audio for unconditional mode
            self.audio_data = torch.zeros(len(dance_sequences), N_MELS + 2, SEQUENCE_LENGTH)
        
    def __len__(self):
        return len(self.dance_data)
    
    def __getitem__(self, idx):
        return self.dance_data[idx], self.audio_data[idx]
    
    def denormalise_dance(self, data):
        """Convert normalised dance data back to original scale"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        original_shape = data.shape
        if len(original_shape) == 2:
            data = data.reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.dance_std.squeeze() + self.dance_mean.squeeze()
        else:
            data = data.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.dance_std + self.dance_mean
        
        return data


# =============================================================================
# --- NEURAL NETWORK MODELS ---
# =============================================================================

class AudioEncoder(nn.Module):
    """Encodes audio features into a conditioning vector"""
    
    def __init__(self, n_mels=N_MELS, hidden_dim=AUDIO_HIDDEN_DIM, embed_dim=AUDIO_EMBED_DIM):
        super().__init__()
        
        input_channels = n_mels + 2  # mel + onset + beat
        
        # 1D CNN to process audio features over time
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Project to embedding dimension
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Audio features (batch, n_mels+2, seq_len)
        Returns:
            audio_embed: (batch, seq_len, embed_dim) - per-frame audio embeddings
            audio_global: (batch, embed_dim) - global audio summary
        """
        # Conv expects (batch, channels, seq_len)
        conv_out = self.conv(x)  # (batch, hidden_dim, seq_len)
        
        # Transpose for LSTM: (batch, seq_len, hidden_dim)
        lstm_in = conv_out.transpose(1, 2)
        lstm_out, (h_n, _) = self.lstm(lstm_in)  # (batch, seq_len, hidden_dim*2)
        
        # Per-frame embeddings
        audio_embed = self.fc(lstm_out)  # (batch, seq_len, embed_dim)
        
        # Global summary from last hidden states
        global_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        audio_global = self.fc(global_hidden)  # (batch, embed_dim)
        
        return audio_embed, audio_global


class DanceEncoder(nn.Module):
    """Encodes a dance sequence into a latent vector"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class ConditionalDanceDecoder(nn.Module):
    """
    Decoder conditioned on both latent vector and audio features.
    Generates dance that matches the input music.
    """
    
    def __init__(self, latent_dim=LATENT_DIM, audio_embed_dim=AUDIO_EMBED_DIM, 
                 hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM):
        super().__init__()
        
        self.seq_len = SEQUENCE_LENGTH
        self.output_dim = output_dim
        
        # Combine latent + audio
        combined_dim = latent_dim + audio_embed_dim
        
        # MLP decoder with audio conditioning
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, SEQUENCE_LENGTH * output_dim)
        )
        
        # Additional per-frame refinement with audio
        self.frame_refiner = nn.Sequential(
            nn.Linear(output_dim + audio_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, z, audio_embed, audio_global):
        """
        Args:
            z: Latent vector (batch, latent_dim)
            audio_embed: Per-frame audio (batch, seq_len, audio_embed_dim)
            audio_global: Global audio summary (batch, audio_embed_dim)
        Returns:
            output: Generated dance (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # Combine latent with global audio
        combined = torch.cat([z, audio_global], dim=1)
        
        # Generate initial output
        output = self.decoder(combined)
        output = output.view(batch_size, self.seq_len, self.output_dim)
        
        # Refine each frame with per-frame audio features
        refined_frames = []
        for t in range(self.seq_len):
            frame = output[:, t, :]  # (batch, output_dim)
            audio_t = audio_embed[:, t, :]  # (batch, audio_embed_dim)
            
            # Concatenate and refine
            frame_audio = torch.cat([frame, audio_t], dim=1)
            refined = self.frame_refiner(frame_audio)
            
            # Residual connection
            refined_frames.append(frame + 0.1 * refined)
        
        output = torch.stack(refined_frames, dim=1)
        
        return output


class MusicConditionedVAE(nn.Module):
    """
    Variational Autoencoder for music-conditioned dance generation.
    
    Architecture:
        Encoder: Dance -> Latent (mu, logvar)
        Audio Encoder: Music -> Audio embeddings
        Decoder: (Latent, Audio) -> Dance
    """
    
    def __init__(self):
        super().__init__()
        
        self.audio_encoder = AudioEncoder()
        self.dance_encoder = DanceEncoder()
        self.decoder = ConditionalDanceDecoder()
        
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, dance, audio):
        """
        Args:
            dance: (batch, seq_len, 51) normalised dance sequence
            audio: (batch, n_mels+2, seq_len) audio features
        """
        # Encode audio
        audio_embed, audio_global = self.audio_encoder(audio)
        
        # Encode dance
        mu, logvar = self.dance_encoder(dance)
        z = self.reparameterise(mu, logvar)
        
        # Decode with audio conditioning
        reconstruction = self.decoder(z, audio_embed, audio_global)
        
        return reconstruction, mu, logvar
    
    def encode(self, dance):
        mu, _ = self.dance_encoder(dance)
        return mu
    
    def decode(self, z, audio):
        """Generate dance from latent vector conditioned on audio"""
        audio_embed, audio_global = self.audio_encoder(audio)
        return self.decoder(z, audio_embed, audio_global)
    
    def generate_from_music(self, audio, num_samples=1):
        """Generate random dances for given music"""
        batch_size = audio.size(0)
        
        # Random latent vectors
        z = torch.randn(batch_size * num_samples, LATENT_DIM).to(audio.device)
        
        # Repeat audio for multiple samples
        audio_repeated = audio.repeat(num_samples, 1, 1)
        
        return self.decode(z, audio_repeated)


# =============================================================================
# --- TRAINING ---
# =============================================================================

def train_music_vae(model, dataloader, num_epochs=NUM_EPOCHS):
    """Train the music-conditioned VAE"""
    print(f"\nTraining Music-Conditioned VAE on {DEVICE}...")
    model = model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        # KL annealing
        if epoch < KL_WARMUP_EPOCHS:
            beta = 0.01 * (epoch / KL_WARMUP_EPOCHS)
        else:
            beta = 0.01
        
        for dance_batch, audio_batch in dataloader:
            dance_batch = dance_batch.to(DEVICE)
            audio_batch = audio_batch.to(DEVICE)
            
            optimiser.zero_grad()
            recon, mu, logvar = model(dance_batch, audio_batch)
            
            # Losses
            recon_loss = nn.functional.mse_loss(recon, dance_batch, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
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

class MusicConditionedGenome:
    """Genome for music-conditioned dance generation"""
    
    def __init__(self, latent_vectors=None, audio_features=None):
        if latent_vectors is None:
            self.latent_vectors = [
                np.random.randn(LATENT_DIM).astype(np.float32)
                for _ in range(NUM_SEQUENCES)
            ]
        else:
            self.latent_vectors = latent_vectors
        
        self.audio_features = audio_features  # Audio to condition on
        self.fitness = None
        self.decoded_frames = None


# Global cache for reference poses (populated once)
_reference_poses_cache = None

def get_reference_poses(dataset, num_samples=500):
    """
    Get a set of reference poses from the training data for novelty computation.
    Caches results to avoid recomputation.
    """
    global _reference_poses_cache
    
    if _reference_poses_cache is not None:
        return _reference_poses_cache
    
    # Sample poses from the dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    reference_poses = []
    
    for idx in indices:
        dance, _ = dataset[idx]
        # Denormalise and reshape
        dance_np = dance.numpy().reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
        dance_np = dance_np * dataset.dance_std.squeeze() + dataset.dance_mean.squeeze()
        # Sample a few frames from each sequence
        for frame_idx in range(0, SEQUENCE_LENGTH, 10):  # Every 10th frame
            reference_poses.append(dance_np[frame_idx])
    
    _reference_poses_cache = np.array(reference_poses)  # (N, 17, 3)
    print(f"Cached {len(_reference_poses_cache)} reference poses for novelty computation")
    return _reference_poses_cache


def compute_pose_novelty(frames, reference_poses):
    """
    Compute novelty score for generated frames.
    Novelty = average minimum distance to reference poses.
    Higher distance = more novel.
    
    Returns a normalised novelty score (0-1 range, higher = more novel)
    """
    if reference_poses is None or len(reference_poses) == 0:
        return 0.5  # Neutral if no reference
    
    # Sample frames to reduce computation
    sample_indices = np.linspace(0, len(frames) - 1, min(20, len(frames))).astype(int)
    sampled_frames = frames[sample_indices]  # (20, 17, 3)
    
    min_distances = []
    for frame in sampled_frames:
        # Compute distance to all reference poses
        distances = np.linalg.norm(reference_poses - frame, axis=(1, 2))  # (N,)
        min_dist = np.min(distances)
        min_distances.append(min_dist)
    
    avg_min_distance = np.mean(min_distances)
    
    # Normalise: typical distance range is ~50-300 for AIST++ data
    # Map to 0-1 range with sigmoid-like scaling
    novelty_score = 1 - np.exp(-avg_min_distance / 100)
    
    return novelty_score


def decode_music_genome(genome, model, dataset):
    """Decode genome to dance frames with music conditioning"""
    if genome.decoded_frames is not None:
        return genome.decoded_frames
    
    if genome.audio_features is None:
        # Fall back to unconditional decoding
        return decode_genome_unconditional(genome, model, dataset)
    
    model.eval()
    all_frames = []
    
    with torch.no_grad():
        for i, latent in enumerate(genome.latent_vectors):
            z = torch.FloatTensor(latent).unsqueeze(0).to(DEVICE)
            
            # Get corresponding audio segment
            audio = genome.audio_features[i] if i < len(genome.audio_features) else genome.audio_features[0]
            audio = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)
            
            decoded = model.decode(z, audio)
            frames = decoded.cpu().numpy()[0]
            all_frames.append(frames)
    
    combined = np.concatenate(all_frames, axis=0)
    combined = combined.reshape(-1, NUM_KEYPOINTS, NUM_COORDS)
    combined = combined * dataset.dance_std.squeeze() + dataset.dance_mean.squeeze()
    combined = smooth_frames(combined)
    
    genome.decoded_frames = combined
    return combined


def decode_genome_unconditional(genome, model, dataset):
    """Fallback unconditional decoding"""
    model.eval()
    all_frames = []
    
    # Create dummy audio
    dummy_audio = torch.zeros(1, N_MELS + 2, SEQUENCE_LENGTH).to(DEVICE)
    
    with torch.no_grad():
        for latent in genome.latent_vectors:
            z = torch.FloatTensor(latent).unsqueeze(0).to(DEVICE)
            decoded = model.decode(z, dummy_audio)
            frames = decoded.cpu().numpy()[0]
            all_frames.append(frames)
    
    combined = np.concatenate(all_frames, axis=0)
    combined = combined.reshape(-1, NUM_KEYPOINTS, NUM_COORDS)
    combined = combined * dataset.dance_std.squeeze() + dataset.dance_mean.squeeze()
    combined = smooth_frames(combined)
    
    genome.decoded_frames = combined
    return combined


def smooth_frames(frames, window_size=5):
    """Apply moving average smoothing"""
    if len(frames) <= window_size:
        return frames
    
    smoothed = np.copy(frames)
    half_window = window_size // 2
    
    for i in range(half_window, len(frames) - half_window):
        smoothed[i] = np.mean(frames[i - half_window:i + half_window + 1], axis=0)
    
    return smoothed


def calculate_music_sync_fitness(genome, model, dataset):
    """
    Calculate fitness with music synchronisation metrics.
    
    Balances 80% accuracy (motion quality + music sync) with 20% novelty.
    
    Accuracy metrics:
    - Beat alignment: Movement peaks should align with musical beats
    - Tempo matching: Dance speed should match music tempo
    - Onset responsiveness: Movements should respond to musical onsets
    - Physical plausibility and smoothness
    
    Novelty metric:
    - Distance from training data poses (rewards unseen moves)
    """
    if genome.fitness is not None:
        return genome.fitness
    
    frames = decode_music_genome(genome, model, dataset)
    
    if len(frames) < 10:
        genome.fitness = -1000
        return genome.fitness
    
    score = 0
    
    # === Standard Motion Quality Metrics ===
    
    # Smoothness
    velocities = np.linalg.norm(np.diff(frames, axis=0), axis=(1, 2))
    mean_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    if 10 < mean_velocity < 150:
        score += 100
    elif mean_velocity < 5:
        score -= 50
    elif mean_velocity > 300:
        score -= 100
    
    score -= min(velocity_std * 0.5, 50)
    
    # Acceleration smoothness
    if len(velocities) > 1:
        accelerations = np.abs(np.diff(velocities))
        mean_accel = np.mean(accelerations)
        if mean_accel > 20:
            score -= (mean_accel - 20) * 2
    
    # Physical plausibility
    ankle_heights = frames[:, [15, 16], 1]
    min_ankle = np.min(ankle_heights)
    if min_ankle < -50:
        score -= min(abs(min_ankle + 50) * 0.5, 50)
    
    # Head above hips
    head_y = frames[:, 0, 1]
    hip_y = np.mean(frames[:, [11, 12], 1], axis=1)
    upright_ratio = np.mean(head_y > hip_y)
    score += upright_ratio * 100
    
    # Movement variety
    pose_variance = np.var(frames, axis=0).mean()
    score += min(pose_variance * 0.1, 50)
    
    # === Music Synchronisation Metrics ===
    
    if genome.audio_features is not None and len(genome.audio_features) > 0:
        try:
            # Combine all audio segments
            all_audio = np.concatenate([af for af in genome.audio_features], axis=1)
            
            # Get beat mask (last row of audio features)
            beat_mask = all_audio[-1, :len(velocities)]  # (num_frames,)
            
            # Get onset envelope (second to last row)
            onset_env = all_audio[-2, :len(velocities)]  # (num_frames,)
            
            # Beat alignment score
            # High velocity frames should align with beat frames
            beat_frames = np.where(beat_mask > 0.5)[0]
            if len(beat_frames) > 0:
                beat_velocities = velocities[beat_frames[beat_frames < len(velocities)]]
                if len(beat_velocities) > 0:
                    # Reward if velocity is high at beats
                    beat_sync_score = np.mean(beat_velocities) / (mean_velocity + 1e-8)
                    score += min(beat_sync_score * 30, 50)
            
            # Onset responsiveness
            # Correlation between onset strength and velocity
            min_len = min(len(onset_env), len(velocities))
            if min_len > 10:
                correlation = np.corrcoef(onset_env[:min_len], velocities[:min_len])[0, 1]
                if not np.isnan(correlation):
                    score += correlation * 40  # Can be negative if anti-correlated
            
        except Exception as e:
            pass  # Skip music metrics if there's an issue
    
    # Latent space regularisation
    for latent in genome.latent_vectors:
        latent_norm = np.linalg.norm(latent)
        if latent_norm > 15:
            score -= (latent_norm - 15) * 5
    
    # === Novelty Reward (20% of total fitness) ===
    # Get reference poses for novelty computation
    reference_poses = get_reference_poses(dataset)
    novelty_score = compute_pose_novelty(frames, reference_poses)
    
    # Scale: accuracy metrics above typically sum to ~100-250
    # For 80/20 balance, novelty should contribute ~25-60 points when novel
    # novelty_score is 0-1, so multiply by max novelty reward
    MAX_NOVELTY_REWARD = 75  # ~20% of typical max accuracy score (~300)
    novelty_reward = novelty_score * MAX_NOVELTY_REWARD
    
    # Apply 80/20 weighting
    # accuracy_weight = 0.8, novelty_weight = 0.2 (already baked into MAX_NOVELTY_REWARD)
    accuracy_score = score  # This is the 80% (motion quality + music sync)
    
    # Final score combines both
    final_score = accuracy_score + novelty_reward
    
    genome.fitness = final_score
    return final_score


def tournament_select(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=lambda g: g.fitness)


def crossover_music(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return MusicConditionedGenome(
            [v.copy() for v in parent1.latent_vectors],
            parent1.audio_features
        )
    
    child_vectors = []
    for i in range(NUM_SEQUENCES):
        if random.random() < 0.5:
            child_vectors.append(parent1.latent_vectors[i].copy())
        else:
            child_vectors.append(parent2.latent_vectors[i].copy())
    
    return MusicConditionedGenome(child_vectors, parent1.audio_features)


def mutate_music(genome, real_latents=None):
    mutated_vectors = []
    
    for latent in genome.latent_vectors:
        if random.random() < MUTATION_RATE:
            if real_latents is not None and len(real_latents) > 0:
                target = random.choice(real_latents)
                alpha = random.uniform(0.7, 0.95)
                mutated = alpha * latent + (1 - alpha) * target
                mutated_vectors.append(mutated.astype(np.float32))
            else:
                noise = np.random.randn(LATENT_DIM).astype(np.float32) * 0.05
                mutated_vectors.append(latent + noise)
        else:
            mutated_vectors.append(latent.copy())
    
    return MusicConditionedGenome(mutated_vectors, genome.audio_features)


def get_real_dance_latents(model, dataset, num_samples=10):
    """Encode real dances to get latent vectors"""
    model.eval()
    latents = []
    
    indices = random.sample(range(len(dataset)), min(num_samples * NUM_SEQUENCES, len(dataset)))
    
    with torch.no_grad():
        for idx in indices:
            dance, _ = dataset[idx]
            dance = dance.unsqueeze(0).to(DEVICE)
            mu = model.encode(dance)
            latents.append(mu.cpu().numpy()[0])
    
    return latents


def run_music_ga_evolution(model, dataset, target_music_features=None):
    """
    Run GA to generate dances for specific music.
    
    Args:
        model: Trained MusicConditionedVAE
        dataset: MusicDanceDataset
        target_music_features: Audio features to generate dance for
                              Shape: list of (n_mels+2, seq_len) arrays
    """
    print(f"\nInitialising Music-Conditioned GA (population: {POPULATION_SIZE})...")
    
    # Get real latent vectors for seeding
    real_latents = get_real_dance_latents(model, dataset, num_samples=POPULATION_SIZE)
    
    # Initialise population
    population = []
    for i in range(POPULATION_SIZE):
        latent_vectors = []
        for j in range(NUM_SEQUENCES):
            idx = (i * NUM_SEQUENCES + j) % len(real_latents)
            noisy_latent = real_latents[idx] + np.random.randn(LATENT_DIM).astype(np.float32) * 0.02
            latent_vectors.append(noisy_latent)
        
        genome = MusicConditionedGenome(latent_vectors, target_music_features)
        population.append(genome)
    
    # Evaluate initial fitness
    for genome in population:
        calculate_music_sync_fitness(genome, model, dataset)
    
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
            
            child = crossover_music(parent1, parent2)
            child = mutate_music(child, real_latents)
            
            calculate_music_sync_fitness(child, model, dataset)
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
# --- VISUALISATION ---
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


def visualise_dance_with_music(frames, audio_path=None, title="Generated Dance"):
    """Visualise dance, optionally with synchronised music playback"""
    print(f"\nPreparing visualisation ({len(frames)} frames)...")
    
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


def save_animation(frames, filename, music_path=None):
    """Save animation to file, optionally with music"""
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
        title_text.set_text(f'Music-Conditioned Dance - Frame {frame_idx}/{len(frames)}')
        return lines + [title_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/SOURCE_FPS, blit=False)
    
    if filename.endswith('.mp4') and music_path:
        # Save video without audio first
        temp_video = filename.replace('.mp4', '_temp.mp4')
        writer = animation.FFMpegWriter(fps=SOURCE_FPS, bitrate=2400)
        ani.save(temp_video, writer=writer)
        plt.close()
        
        # Calculate video duration
        duration = len(frames) / SOURCE_FPS
        
        # Combine video and audio using ffmpeg
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', music_path,
            '-t', str(duration),  # Trim to video length
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            filename
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_video)  # Clean up temp file
            print(f"Saved with music to {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not add music (ffmpeg error). Saved video only.")
            os.rename(temp_video, filename)
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Saved video without music.")
            print("Install ffmpeg with: brew install ffmpeg")
            os.rename(temp_video, filename)
    elif filename.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=SOURCE_FPS, bitrate=2400)
        ani.save(filename, writer=writer)
        plt.close()
        print(f"Saved to {filename}")
    else:
        ani.save(filename, writer='pillow', fps=SOURCE_FPS)
        plt.close()
        print(f"Saved to {filename}")


def get_next_filename(base_name="music_dance", ext="mp4"):
    counter = 1
    while True:
        filename = f"{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def select_music_track(music_folder):
    """Interactive music selection"""
    music_path = Path(music_folder)
    tracks = sorted(list(music_path.glob('*.mp3')))
    
    print("\nAvailable music tracks:")
    for i, track in enumerate(tracks):
        genre = track.stem[1:3]
        genre_name = GENRE_NAMES.get(genre, genre)
        print(f"  {i+1}. {track.stem} ({genre_name})")
    
    while True:
        try:
            choice = input("\nSelect track number (or 'random'): ").strip()
            if choice.lower() == 'random':
                return random.choice(tracks)
            idx = int(choice) - 1
            if 0 <= idx < len(tracks):
                return tracks[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


# =============================================================================
# --- MAIN ---
# =============================================================================

def main():
    print("=" * 60)
    print("Music-Conditioned Dance Generator")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    if not LIBROSA_AVAILABLE:
        print("\nERROR: librosa is required for music processing.")
        print("Install with: pip install librosa")
        return
    
    # Load paired data
    dance_seqs, audio_seqs, music_ids = load_paired_sequences(DATA_FOLDER, MUSIC_FOLDER)
    
    if audio_seqs is None:
        print("No paired data available. Please check data folders.")
        return
    
    dataset = MusicDanceDataset(dance_seqs, audio_seqs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    model = MusicConditionedVAE()
    
    # Check for saved model
    model_path = 'music_conditioned_vae.pth'
    if os.path.exists(model_path):
        print(f"\nLoading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        # Train the model
        history = train_music_vae(model, dataloader)
        
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
        plt.savefig('music_vae_training.png')
        plt.close()
    
    # Select music track
    music_track = select_music_track(MUSIC_FOLDER)
    print(f"\nSelected: {music_track.name}")
    
    # Load full music features
    print("Loading music features...")
    full_audio = load_audio_features(str(music_track))
    
    # Create audio segments for each sequence in the genome
    total_dance_frames = NUM_SEQUENCES * SEQUENCE_LENGTH
    target_audio_features = []
    
    for i in range(NUM_SEQUENCES):
        start_frame = i * SEQUENCE_LENGTH
        end_frame = (i + 1) * SEQUENCE_LENGTH
        
        # Calculate audio time range
        start_time = start_frame / SOURCE_FPS
        end_time = end_frame / SOURCE_FPS
        
        audio_start = int(start_time * AUDIO_FPS)
        audio_end = int(end_time * AUDIO_FPS)
        
        if audio_end > full_audio['mel_spec'].shape[1]:
            # Pad or loop if music is shorter
            audio_end = full_audio['mel_spec'].shape[1]
            audio_start = max(0, audio_end - int(SEQUENCE_LENGTH * AUDIO_FPS / SOURCE_FPS))
        
        segment = {
            'mel_spec': full_audio['mel_spec'][:, audio_start:audio_end],
            'onset_env': full_audio['onset_env'][audio_start:audio_end],
            'beat_frames': full_audio['beat_frames'] - audio_start,
            'tempo': full_audio['tempo']
        }
        
        resampled = resample_audio_to_dance_fps(segment, SEQUENCE_LENGTH)
        
        audio_combined = np.vstack([
            resampled['mel_spec'],
            resampled['onset_env'].reshape(1, -1),
            resampled['beat_mask'].reshape(1, -1)
        ])
        
        target_audio_features.append(audio_combined)
    
    # Run GA
    best_genome = run_music_ga_evolution(model, dataset, target_audio_features)
    
    # Decode the best dance
    frames = decode_music_genome(best_genome, model, dataset)
    
    # Visualise
    visualise_dance_with_music(frames, str(music_track), 
                               f"Dance for {music_track.stem}")
    
    # Ask to save
    save = input("\nSave animation? (y/n): ").strip().lower()
    if save == 'y':
        filename = get_next_filename(f"music_dance_{music_track.stem}", "mp4")
        save_animation(frames, filename, music_path=str(music_track))


if __name__ == '__main__':
    main()
