"""
Guided Latent Diffusion Dance Generator
========================================
Replaces the Genetic Algorithm from autoencoderDanceGA.py with a modern
generative pipeline while keeping the proven VAE architecture.

Pipeline:
  Step 1 – Data Normalization & Preprocessing  (kept from autoencoderDanceGA.py)
  Step 2 – Biomechanical VAE                   (kept from autoencoderDanceGA.py)
  Step 3 – Adversarial Critic                  (NEW — replaces GA fitness)
  Step 4 – Audio-Conditioned Latent Diffusion  (NEW — replaces GA search)
  Step 5 – Classifier-Guided Sampling          (NEW — steers creativity)
  Step 6 – Denormalization & Synthesis          (NEW — ffmpeg video + audio)

Requires: torch, numpy, librosa, matplotlib, ffmpeg (system)
"""

import numpy as np
import math
import os
import re
import pickle
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

warnings.filterwarnings("ignore")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed.  Run: pip install librosa")


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FOLDER = "keypoints3d"
MUSIC_FOLDER = "AISTmusic"
SOURCE_FPS = 60
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# --- Skeleton ---
NUM_KEYPOINTS = 17
NUM_COORDS = 3
INPUT_DIM = NUM_KEYPOINTS * NUM_COORDS  # 51

BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 11), (6, 12), (11, 12),        # Torso
    (6, 8), (8, 10),                            # Right arm
    (5, 7), (7, 9),                             # Left arm
    (12, 14), (14, 16),                         # Right leg
    (11, 13), (13, 15),                         # Left leg
]

BONE_COLORS = {
    "head": "#FF6B6B", "torso": "#4ECDC4",
    "right_arm": "#45B7D1", "left_arm": "#96CEB4",
    "right_leg": "#FFEAA7", "left_leg": "#DDA0DD",
}

# --- Sequence ---
SEQUENCE_LENGTH = 60  # 1 second at 60 FPS

# --- Audio ---
AUDIO_SR = 22050
N_MELS = 80
HOP_LENGTH = 512
AUDIO_FPS = AUDIO_SR / HOP_LENGTH  # ~43 fps
AUDIO_EMBED_DIM = 64

# --- VAE (Step 2) — same arch as autoencoderDanceGA.py ---
LATENT_DIM = 256
HIDDEN_DIM = 512
VAE_BATCH = 128
VAE_LR = 0.001
VAE_EPOCHS = 150
KL_WARMUP_EPOCHS = 20

# --- Adversarial Critic (Step 3) ---
CRITIC_HIDDEN = 256
CRITIC_EPOCHS = 100
CRITIC_BATCH = 256
CRITIC_LR = 1e-4

# --- Latent Diffusion (Step 4) ---
DIFF_TIMESTEPS = 1000
DIFF_HIDDEN = 256
DIFF_TIME_EMB = 128
DIFF_EPOCHS = 100
DIFF_BATCH = 256
DIFF_LR = 1e-4

# --- Guided Sampling (Step 5) ---
TARGET_REALISM = 1
GUIDANCE_SCALE = 3.0
DDIM_STEPS = 100
MIN_FINAL_CRITIC_SCORE = 0.88      # Retry a segment if final score falls below this
SEGMENT_MAX_ATTEMPTS = 3           # Max guided sampling retries per segment
TARGET_SCORE_TOLERANCE = 0.03      # Accept segment when |score - target| is below this

# --- Output ---
NUM_SEGMENTS = 4  # how many 1-second segments to generate for one dance

# --- Genres ---
GENRES = ["BR", "PO", "LO", "MH", "LH", "HO", "WA", "KR", "JS", "JB"]
GENRE_NAMES = {
    "BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop",
    "LH": "LA Hip-hop", "HO": "House", "WA": "Waack", "KR": "Krump",
    "JS": "Street Jazz", "JB": "Ballet Jazz",
}


# =============================================================================
# STEP 1 — DATA NORMALIZATION & PREPROCESSING
# =============================================================================
# Kept from autoencoderDanceGA.py: mean/std normalization per joint-coordinate.
# =============================================================================

def load_pkl(filepath):
    """Load AIST++ pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data.get("keypoints3d_optim", data.get("keypoints3d"))


def extract_music_id(filename):
    """Extract music ID from AIST++ filename e.g. 'mBR0'."""
    match = re.search(r"m([A-Z]{2}\d)", filename)
    return "m" + match.group(1) if match else None


# ---- Audio Feature Extraction ----

def load_audio_features(audio_path, duration=None):
    """Extract mel spectrogram, beats, onset envelope from an mp3 file."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio processing")
    y, sr = librosa.load(audio_path, sr=AUDIO_SR, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset = (onset - onset.mean()) / (onset.std() + 1e-8)
    return {"mel": mel_db, "tempo": tempo, "beats": beat_frames, "onset": onset}


def load_music_cache(music_folder):
    """Pre-load all audio features into a dict keyed by music ID (e.g. 'mBR0')."""
    cache = {}
    mpath = Path(music_folder)
    if not mpath.exists():
        print(f"Warning: music folder '{music_folder}' not found")
        return cache
    print("Loading music features...")
    for mp3 in sorted(mpath.glob("*.mp3")):
        mid = mp3.stem
        try:
            cache[mid] = load_audio_features(str(mp3))
            print(f"  {mid}")
        except Exception as e:
            print(f"  {mid} FAILED: {e}")
    print(f"Loaded {len(cache)} tracks")
    return cache


def summarise_audio_segment(audio_feats, start_frame, end_frame, out_frames):
    """
    Slice audio features for a dance segment and resample to out_frames.
    Returns (N_MELS + 2, out_frames) — mel + onset + beat_mask.
    """
    mel = audio_feats["mel"]  # (N_MELS, A)
    onset = audio_feats["onset"]  # (A,)
    A = mel.shape[1]

    # Clamp
    astart = max(0, int(start_frame / SOURCE_FPS * AUDIO_FPS))
    aend = min(A, int(end_frame / SOURCE_FPS * AUDIO_FPS))
    if aend <= astart:
        return np.zeros((N_MELS + 2, out_frames), dtype=np.float32)

    seg_mel = mel[:, astart:aend]
    seg_onset = onset[astart:aend]

    # Resample to out_frames
    seg_len = seg_mel.shape[1]
    idx = np.linspace(0, seg_len - 1, out_frames).astype(int)
    mel_r = seg_mel[:, idx]
    onset_r = seg_onset[idx]

    # Beat mask
    beat_mask = np.zeros(out_frames, dtype=np.float32)
    beats = audio_feats["beats"]
    seg_beats = beats[(beats >= astart) & (beats < aend)] - astart
    for bf in seg_beats:
        df = int(bf * out_frames / seg_len)
        if 0 <= df < out_frames:
            beat_mask[df] = 1.0

    return np.vstack([mel_r, onset_r[None, :], beat_mask[None, :]])  # (N_MELS+2, T)


# ---- Data Loading ----

def load_paired_data(data_folder, music_folder,
                     sequence_length=SEQUENCE_LENGTH, max_files=None):
    """
    Load dance sequences paired with aligned audio features.

    Returns
    -------
    dance_seqs : np.ndarray (N, seq_len, 17, 3)
    audio_seqs : np.ndarray (N, N_MELS+2, seq_len) or None
    music_ids  : list[str|None]
    """
    mcache = load_music_cache(music_folder) if LIBROSA_AVAILABLE else {}
    data_path = Path(data_folder)
    all_files = list(data_path.glob("*.pkl"))
    if max_files:
        all_files = random.sample(all_files, min(max_files, len(all_files)))

    dances, audios, mids = [], [], []
    audio_dim = N_MELS + 2

    print(f"Loading paired dance-music data from {len(all_files)} files...")
    for pkl in all_files:
        mid = extract_music_id(pkl.name)
        try:
            kp = load_pkl(pkl)
        except Exception:
            continue
        n = len(kp)

        has_audio = mid is not None and mid in mcache

        for s in range(0, n - sequence_length, sequence_length // 2):
            e = s + sequence_length
            dances.append(kp[s:e])

            if has_audio:
                aud = summarise_audio_segment(mcache[mid], s, e, sequence_length)
                audios.append(aud)
            else:
                audios.append(np.zeros((audio_dim, sequence_length), dtype=np.float32))
            mids.append(mid)

    raw = np.array(dances, dtype=np.float32)  # (N, T, 17, 3)
    aud = np.array(audios, dtype=np.float32)  # (N, audio_dim, T)
    has_any_audio = any(m is not None for m in mids)

    print(f"  {len(raw)} segments, audio={'yes' if has_any_audio else 'no'}")
    return raw, aud if has_any_audio else None, mids


# ---- PyTorch Dataset ----

class DanceAudioDataset(Dataset):
    """
    Holds normalised dance sequences and optional aligned audio features.
    Normalisation follows autoencoderDanceGA.py: per-joint mean/std.
    """

    def __init__(self, sequences, audio=None):
        """
        Parameters
        ----------
        sequences : np.ndarray (N, seq_len, 17, 3)
        audio     : np.ndarray (N, N_MELS+2, seq_len) or None
        """
        # Compute normalisation stats over the whole dataset
        self.mean = sequences.mean(axis=(0, 1), keepdims=True)   # (1,1,17,3)
        self.std = sequences.std(axis=(0, 1), keepdims=True) + 1e-8

        normalised = (sequences - self.mean) / self.std
        # Flatten joints: (N, T, 51)
        self.dance = torch.tensor(
            normalised.reshape(len(sequences), SEQUENCE_LENGTH, -1),
            dtype=torch.float32,
        )

        if audio is not None:
            self.audio = torch.tensor(audio, dtype=torch.float32)
        else:
            self.audio = torch.zeros(len(sequences), N_MELS + 2, SEQUENCE_LENGTH)

    def __len__(self):
        return len(self.dance)

    def __getitem__(self, idx):
        return self.dance[idx], self.audio[idx]

    # --- Utility for denormalization (Step 6) ---
    def denormalize(self, data):
        """Convert normalised flat data back to original-scale 3D coordinates."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        original = data.shape
        if len(original) == 2:
            data = data.reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.std.squeeze() + self.mean.squeeze()
        else:
            data = data.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDS)
            data = data * self.std + self.mean
        return data


# =============================================================================
# STEP 2 — BIOMECHANICAL VAE
# =============================================================================
# Architecture kept from autoencoderDanceGA.py:
#   Encoder: Bidirectional LSTM → mu, logvar
#   Decoder: Feedforward MLP (generates all frames at once)
# =============================================================================

class DanceEncoder(nn.Module):
    """Bidirectional LSTM → (mu, logvar)."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class DanceDecoder(nn.Module):
    """MLP decoder — outputs all frames simultaneously."""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM):
        super().__init__()
        self.seq_len = SEQUENCE_LENGTH
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, SEQUENCE_LENGTH * output_dim),
        )

    def forward(self, z, target=None, teacher_forcing_ratio=0.0, seq_len=SEQUENCE_LENGTH):
        batch = z.size(0)
        out = self.decoder(z)
        return out.view(batch, self.seq_len, self.output_dim)


class DanceVAE(nn.Module):
    """Variational Autoencoder — guarantees decoded latent codes produce
    structurally valid skeletons because it only ever saw real human data."""

    def __init__(self):
        super().__init__()
        self.encoder = DanceEncoder()
        self.decoder = DanceDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, teacher_forcing_ratio=0.5):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return recon, mu, logvar

    def encode(self, x):
        """Encode to mean of posterior (deterministic)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z, target=None, teacher_forcing_ratio=0.0)


def train_vae(model, dataloader, epochs=VAE_EPOCHS, lr=VAE_LR,
              checkpoint_path="guided_vae_ckpt.pth"):
    """Train the biomechanical VAE with KL warmup (from autoencoderDanceGA.py)."""
    print(f"\n{'='*55}")
    print(f"  [Step 2] Training Biomechanical VAE  ({epochs} epochs)")
    print(f"{'='*55}")
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    history = {"loss": [], "recon": [], "kl": []}

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        print(f"  Resuming VAE from epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        tot_loss = tot_recon = tot_kl = 0.0
        beta = min(0.1, 0.1 * epoch / max(KL_WARMUP_EPOCHS, 1))

        for dance_b, _ in dataloader:
            dance_b = dance_b.to(DEVICE)
            recon, mu, logvar = model(dance_b, teacher_forcing_ratio=0.5)
            recon_loss = F.mse_loss(recon, dance_b, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            tot_recon += recon_loss.item()
            tot_kl += kl_loss.item()

        n = len(dataloader)
        history["loss"].append(tot_loss / n)
        history["recon"].append(tot_recon / n)
        history["kl"].append(tot_kl / n)
        sched.step(tot_loss / n)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={tot_loss/n:.5f}  "
                  f"recon={tot_recon/n:.5f}  kl={tot_kl/n:.5f}  β={beta:.4f}")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "history": history,
            }, checkpoint_path)

    print("  VAE training complete.")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return history


# =============================================================================
# STEP 3 — ADVERSARIAL CRITIC  (replaces GA fitness scoring)
# =============================================================================

class AdversarialCritic(nn.Module):
    """
    Binary classifier grading realism of a *decoded* dance sequence.

    Input : (B, T, 51)  — normalised dance sequence
    Output: (B, 1)      — score in [0, 1]  (1 = perfectly human)

    Training:
        Real samples → label 1.0  (clean AIST++ sequences)
        Fake samples → label 0.0  (VAE-decoded random latent noise)
    """

    def __init__(self, feature_dim=INPUT_DIM, hidden=CRITIC_HIDDEN):
        super().__init__()
        # Temporal encoder (processes the whole sequence)
        self.temporal = nn.GRU(
            feature_dim, hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, T, F) → (B, 1)"""
        out, _ = self.temporal(x)       # (B, T, H*2)
        pooled = out.mean(dim=1)        # (B, H*2)  average-pool over time
        return self.head(pooled)        # (B, 1)


def train_critic(critic, vae, dataloader, epochs=CRITIC_EPOCHS, lr=CRITIC_LR):
    """
        Train the Critic with hard-negative mining so fake data stays challenging.

        Strategy
        --------
        1) Build a large pool of heterogeneous fakes per batch (easy + near-real).
        2) Score the pool with the current critic.
        3) Keep only the hardest negatives (highest "real" scores).
        4) Train on clean real data versus mined hard fakes.

        This prevents the critic from saturating on trivially fake samples.
    """
    print(f"\n{'='*55}")
    print(f"  [Step 3] Training Adversarial Critic  ({epochs} epochs)")
    print(f"{'='*55}")
    critic.to(DEVICE)
    vae.to(DEVICE)
    vae.eval()
    opt = optim.Adam(critic.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        critic.train()
        tot_loss = 0.0
        correct = 0
        total = 0

        # Curriculum: start moderate, then increase fake difficulty.
        hardness = min(1.0, (epoch + 1) / max(int(epochs * 0.4), 1))

        for dance_b, _ in dataloader:
            B = dance_b.size(0)
            dance_b = dance_b.to(DEVICE)

            # Mild augmentation keeps the critic from keying on tiny artefacts.
            real_in = dance_b + 0.01 * torch.randn_like(dance_b)

            # --- Real (label-smoothed: 0.9 instead of 1.0) ---
            real_labels = torch.full((B, 1), 0.9, device=DEVICE)
            pred_real = critic(real_in)
            loss_real = F.binary_cross_entropy(pred_real, real_labels)

            # --- Candidate fake pool ---
            with torch.no_grad():
                z_real = vae.encode(dance_b)

                # Easy negatives: random latent samples.
                z_random = torch.randn(B, LATENT_DIM, device=DEVICE)
                fake_random = vae.decode(z_random)

                # Near-real negatives: small and large latent perturbations.
                low_noise = 0.05 + 0.15 * hardness
                high_noise = 0.25 + 0.45 * hardness
                z_perturbed_low = z_real + low_noise * torch.randn_like(z_real)
                z_perturbed_high = z_real + high_noise * torch.randn_like(z_real)
                fake_perturbed_low = vae.decode(z_perturbed_low)
                fake_perturbed_high = vae.decode(z_perturbed_high)

                # Reconstructions are usually close to real and therefore hard.
                fake_recon = vae.decode(z_real)

                # Temporal negatives: local continuity broken by roll + swap.
                shift = random.randint(1, max(2, dance_b.size(1) // 6))
                fake_rolled = torch.roll(dance_b, shifts=shift, dims=1)
                split = random.randint(dance_b.size(1) // 4, 3 * dance_b.size(1) // 4)
                fake_swapped = torch.cat([dance_b[:, split:, :], dance_b[:, :split, :]], dim=1)

                # Structural negatives: random feature dropout across all frames.
                drop_prob = 0.05 + 0.20 * hardness
                feat_keep = (torch.rand(B, 1, INPUT_DIM, device=DEVICE) > drop_prob).float()
                fake_dropout = dance_b * feat_keep

                candidate_fakes = torch.cat([
                    fake_random,
                    fake_recon,
                    fake_perturbed_low,
                    fake_perturbed_high,
                    fake_rolled,
                    fake_swapped,
                    fake_dropout,
                ], dim=0)

                # Mine hardest negatives: choose fakes critic still thinks are real.
                cand_scores = critic(candidate_fakes).squeeze(1)
                hard_k = min(B, candidate_fakes.size(0))
                hard_idx = torch.topk(cand_scores, k=hard_k, largest=True).indices
                hard_fakes = candidate_fakes[hard_idx]

            # Add noise so the critic focuses on structure/motion, not pixel-perfect cues.
            hard_fakes = hard_fakes + 0.01 * torch.randn_like(hard_fakes)

            fake_labels = torch.full((hard_fakes.size(0), 1), 0.1, device=DEVICE)
            pred_fake = critic(hard_fakes.detach())
            loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)

            loss = (loss_real + loss_fake) / 2.0

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            correct += ((pred_real > 0.5).sum() + (pred_fake < 0.5).sum()).item()
            total += B + hard_fakes.size(0)

        sched.step()
        acc = correct / max(total, 1)
        history["loss"].append(tot_loss / max(len(dataloader), 1))
        history["acc"].append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={history['loss'][-1]:.4f}  acc={acc:.3f}")

    print("  Critic training complete.")
    return history


# =============================================================================
# STEP 4 — AUDIO-CONDITIONED LATENT DIFFUSION
# =============================================================================
# The diffusion process operates in the VAE's latent space (dim = LATENT_DIM).
# This is much more efficient than diffusing raw 3060-dim sequences and
# guarantees outputs decode to physically plausible skeletons.
# =============================================================================

# ---- Cosine noise schedule ----

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


class DiffusionSchedule:
    """Pre-computes all DDPM / DDIM noise schedule constants."""

    def __init__(self, T=DIFF_TIMESTEPS):
        self.T = T
        self.betas = cosine_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_ab = torch.sqrt(self.alpha_bar)
        self.sqrt_1m_ab = torch.sqrt(1.0 - self.alpha_bar)

    def _idx(self, vals, t, shape):
        B = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(B, *((1,) * (len(shape) - 1))).to(t.device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._idx(self.sqrt_ab, t, x0.shape) * x0
            + self._idx(self.sqrt_1m_ab, t, x0.shape) * noise
        ), noise


# ---- Audio Encoder ----

class AudioEncoder(nn.Module):
    """
    CNN + GRU encoder that compresses audio features into a fixed-size
    conditioning vector suitable for the latent denoiser.
    """

    def __init__(self, in_ch=N_MELS + 2, embed=AUDIO_EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, embed, 3, padding=1), nn.BatchNorm1d(embed), nn.ReLU(),
        )
        self.gru = nn.GRU(embed, embed, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(embed * 2, embed)

    def forward(self, x):
        """x: (B, in_ch, T) → global embedding (B, embed)."""
        c = self.conv(x)                      # (B, embed, T)
        c = c.permute(0, 2, 1)                # (B, T, embed)
        _, h = self.gru(c)                     # h: (2, B, embed)
        glob = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2*embed)
        return self.proj(glob)                    # (B, embed)


# ---- Sinusoidal Timestep Embedding ----

class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---- Residual MLP Block ----

class ResMLPBlock(nn.Module):
    """Linear residual block with time + audio conditioning."""

    def __init__(self, dim_in, dim_out, t_dim, cond_dim=0):
        super().__init__()
        self.t_proj = nn.Linear(t_dim, dim_out)
        self.c_proj = nn.Linear(cond_dim, dim_out) if cond_dim > 0 else None
        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.SiLU(),
            nn.Linear(dim_out, dim_out), nn.LayerNorm(dim_out), nn.SiLU(),
        )
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.block[0](x)                        # first linear
        h = self.block[1](h)                         # LayerNorm
        h = self.block[2](h + self.t_proj(t_emb))   # SiLU after adding time
        if c_emb is not None and self.c_proj is not None:
            h = h + self.c_proj(c_emb)
        h = self.block[3](h)                         # second linear
        h = self.block[4](h)                         # LayerNorm
        h = self.block[5](h)                         # SiLU
        return h + self.skip(x)


# ---- Latent Denoiser ----

class LatentDenoiser(nn.Module):
    """
    Predicts noise ε given a noisy latent code z_t, timestep t,
    and (optional) audio conditioning.

    Operates entirely in the VAE's latent space (R^{LATENT_DIM}).
    Temporal structure is delegated to the VAE decoder.
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden=DIFF_HIDDEN,
                 t_dim=DIFF_TIME_EMB, audio_embed=AUDIO_EMBED_DIM):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPE(t_dim),
            nn.Linear(t_dim, t_dim * 2), nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.audio_enc = AudioEncoder(embed=audio_embed)

        self.res1 = ResMLPBlock(latent_dim, hidden, t_dim, cond_dim=audio_embed)
        self.res2 = ResMLPBlock(hidden, hidden * 2, t_dim, cond_dim=audio_embed)
        self.res3 = ResMLPBlock(hidden * 2, hidden * 2, t_dim, cond_dim=audio_embed)
        self.res4 = ResMLPBlock(hidden * 2 + hidden * 2, hidden, t_dim, cond_dim=audio_embed)
        self.res5 = ResMLPBlock(hidden + hidden, latent_dim, t_dim, cond_dim=audio_embed)

        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t, t, audio=None):
        """
        z_t  : (B, LATENT_DIM)  — noisy latent
        t    : (B,)             — timestep indices
        audio: (B, n_mels+2, T) — raw audio features (or None)
        Returns: predicted noise ε  (B, LATENT_DIM)
        """
        t_emb = self.time_mlp(t)  # (B, t_dim)

        if audio is not None:
            a_emb = self.audio_enc(audio)  # (B, audio_embed)
        else:
            a_emb = torch.zeros(z_t.size(0), AUDIO_EMBED_DIM, device=z_t.device)

        # U-Net-style: encoder → bottleneck → decoder with skip connections
        h1 = self.res1(z_t, t_emb, a_emb)      # (B, H)
        h2 = self.res2(h1, t_emb, a_emb)        # (B, 2H)
        h3 = self.res3(h2, t_emb, a_emb)        # (B, 2H)  bottleneck
        d2 = self.res4(torch.cat([h3, h2], dim=-1), t_emb, a_emb)  # (B, H)
        d1 = self.res5(torch.cat([d2, h1], dim=-1), t_emb, a_emb)  # (B, L)

        return self.out_proj(d1)


def encode_dataset_to_latents(vae, dataloader):
    """Encode the whole dataset to latent codes for diffusion training."""
    vae.eval()
    all_z, all_audio = [], []
    with torch.no_grad():
        for dance_b, audio_b in dataloader:
            dance_b = dance_b.to(DEVICE)
            mu = vae.encode(dance_b)
            all_z.append(mu.cpu())
            all_audio.append(audio_b)
    return torch.cat(all_z, dim=0), torch.cat(all_audio, dim=0)


class LatentAudioDataset(Dataset):
    """Dataset of (latent_code, audio_features) pairs for diffusion training."""

    def __init__(self, latents, audio):
        self.latents = latents   # (N, LATENT_DIM)
        self.audio = audio       # (N, N_MELS+2, T)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.audio[idx]


def train_diffusion(denoiser, latent_dataset, schedule,
                    epochs=DIFF_EPOCHS, lr=DIFF_LR, batch_size=DIFF_BATCH):
    """Train the latent denoiser with ε-prediction objective."""
    print(f"\n{'='*55}")
    print(f"  [Step 4] Training Latent Diffusion  ({epochs} epochs)")
    print(f"{'='*55}")
    denoiser.to(DEVICE)
    loader = DataLoader(latent_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    opt = optim.AdamW(denoiser.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    losses = []

    for epoch in range(epochs):
        denoiser.train()
        ep_loss = 0.0
        for z_batch, audio_batch in loader:
            z_batch = z_batch.to(DEVICE)
            audio_batch = audio_batch.to(DEVICE)
            B = z_batch.size(0)

            t = torch.randint(0, schedule.T, (B,), device=DEVICE).long()
            z_t, noise = schedule.q_sample(z_batch, t)

            pred = denoiser(z_t, t, audio_batch)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        lr_sched.step()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.6f}")

    print("  Diffusion training complete.")
    return losses


# =============================================================================
# STEP 5 — CLASSIFIER-GUIDED SAMPLING
# =============================================================================
# At each denoising step:
#   1. Predict clean z0 from noisy z_t.
#   2. Decode z0 through the VAE to get a dance sequence.
#   3. Feed the decoded dance to the Critic → realism score.
#   4. Compute loss = (score − target)².
#   5. Back-propagate through z0 to obtain the gradient.
#   6. Subtract the scaled gradient to bend the trajectory toward the
#      target realism level (e.g. 80% → "transformational creativity").
# =============================================================================

def guided_sample(denoiser, vae, critic, schedule, audio=None,
                  num_samples=1, steps=DDIM_STEPS,
                  target=TARGET_REALISM, guidance_scale=GUIDANCE_SCALE,
                  return_score=False):
    """
    Classifier-guided DDIM sampling in the VAE's latent space.
    Returns decoded dance sequences (num_samples, T, 51) on CPU.
    """
    print(f"\n  [Step 5] Classifier-Guided Sampling")
    print(f"    target={target:.0%}  base scale={guidance_scale}  steps={steps}")
    target = float(target)

    # High targets (>=85%) get progressively safer hyper-parameters so extreme pushes stay stable.
    high_push = max(target - 0.85, 0.0) / 0.15  # 0 at 85%, 1 at 100%
    scale_factor = 1.0 - 0.4 * high_push        # trim up to 40% of the user scale near 100%
    safe_scale = max(0.1, guidance_scale * scale_factor)
    freeze_margin = max(0.02, 0.08 - 0.06 * high_push)  # freeze a tad before target when pushing hard

    # Realistic cap: critic probabilities rarely reach 1.0, so hard 100% targets can
    # keep pushing until late steps and destabilise.
    effective_target = min(target, 0.93)

    print(f"    target-adapted scale={safe_scale:.2f}  freeze margin={freeze_margin:.3f}")
    if effective_target < target:
        print(f"    effective target capped to {effective_target:.0%} for stability")
    denoiser.eval()
    vae.eval()
    # Critic MUST stay in train mode — cuDNN RNN does not support backward in eval mode
    critic.train()

    # Reduce critic stochasticity during guidance.
    old_temporal_dropout = getattr(critic.temporal, "dropout", 0.0)
    critic.temporal.dropout = 0.0
    head_dropouts = [m for m in critic.head.modules() if isinstance(m, nn.Dropout)]
    for m in head_dropouts:
        m.eval()

    T = schedule.T
    times = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)

    # Start from pure noise in latent space
    z = torch.randn(num_samples, LATENT_DIM, device=DEVICE)

    if audio is None:
        audio_in = torch.zeros(num_samples, N_MELS + 2, SEQUENCE_LENGTH, device=DEVICE)
    else:
        audio_in = audio.to(DEVICE)

    # Only start guiding after 30% of steps (early z0 predictions bad)
    guide_start = int(steps * 0.3)

    # Best-z tracking: keep the latent whose critic score is closest to target.
    best_z = z.clone()
    best_score = 0.0
    best_dist = float("inf")
    guidance_frozen = False  # stop guiding once target reached

    for i, t_val in enumerate(times):
        t_batch = t_val.expand(num_samples)

        # 1. Predict clean z0
        with torch.no_grad():
            eps_pred = denoiser(z, t_batch, audio_in)
            ab = schedule.alpha_bar[t_val]
            z0_pred = (z - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab)
            z0_pred = z0_pred.clamp(-3.0, 3.0)

        # 2. Critic-guided steering (only after initial denoising, and only if not frozen)
        if i >= guide_start and not guidance_frozen:
            with torch.inference_mode(False):
                z0_g = z0_pred.clone().detach().requires_grad_(True)
                decoded = vae.decode(z0_g)          # (B, T, 51)
                score = critic(decoded)              # (B, 1)

                # Explicit target matching objective: minimize (score - target)^2.
                target_loss = torch.mean((score - effective_target) ** 2)
                target_loss.backward()
            grad = z0_g.grad.detach()

            # Normalise gradient direction
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad = grad / grad_norm
            mean_score = score.mean().item()

            # Gentle ramp up over the first 40% of guided steps, then hold
            progress = (i - guide_start) / max(steps - guide_start - 1, 1)
            ramp = min(progress / 0.4, 1.0)
            adaptive_scale = safe_scale * ramp

            # Taper guidance near the target to avoid oscillation/overshoot.
            dist_to_target = abs(mean_score - effective_target)
            taper = max(0.15, min(1.0, dist_to_target / max(2 * freeze_margin, 1e-6)))
            adaptive_scale *= taper

            # Gradient descent on target loss moves score toward target from either side.
            z0_pred = z0_pred - adaptive_scale * grad

            # Track best z0 for fallback (closest to target).
            if dist_to_target < best_dist:
                best_dist = dist_to_target
                best_score = mean_score
                best_z = z0_pred.detach().clone()

            # Freeze guidance when we are already close enough to target.
            if dist_to_target <= freeze_margin:
                guidance_frozen = True

        # 3. DDIM update
        ab_prev = (
            schedule.alpha_bar[times[i + 1]] if i + 1 < len(times)
            else torch.tensor(1.0)
        )
        z = torch.sqrt(ab_prev) * z0_pred + torch.sqrt(1 - ab_prev) * eps_pred

        # Progress logging
        if (i + 1) % max(steps // 5, 1) == 0:
            with torch.no_grad():
                s = critic(vae.decode(z0_pred)).mean().item()
            print(f"      step {i+1}/{steps}  critic={s:.3f}")

    # Final decode — fall back to best-target snapshot if final drifted away.
    with torch.no_grad():
        final_dance = vae.decode(z)          # (B, T, 51)
        final_score = critic(final_dance).mean().item()
        final_dist = abs(final_score - effective_target)
        drift_margin = 0.01
        if final_dist > best_dist + drift_margin:
            print(
                f"    Final score {final_score:.3f} drifted from target — "
                f"using best snapshot ({best_score:.3f})"
            )
            final_dance = vae.decode(best_z)
            final_score = best_score

    # Restore critic settings.
    critic.temporal.dropout = old_temporal_dropout
    for m in head_dropouts:
        m.train()
    print(f"    Final Critic score: {final_score:.3f}  (target distance={abs(final_score - effective_target):.3f})")

    final_dance = final_dance.cpu()
    if return_score:
        return final_dance, float(final_score)
    return final_dance


@torch.no_grad()
def unguided_sample(denoiser, vae, schedule, audio=None,
                    num_samples=1, steps=DDIM_STEPS):
    """DDIM sampling without Critic guidance (baseline comparison)."""
    denoiser.eval()
    vae.eval()
    T = schedule.T
    times = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)
    z = torch.randn(num_samples, LATENT_DIM, device=DEVICE)

    if audio is None:
        audio_in = torch.zeros(num_samples, N_MELS + 2, SEQUENCE_LENGTH, device=DEVICE)
    else:
        audio_in = audio.to(DEVICE)

    for i, t_val in enumerate(times):
        t_batch = t_val.expand(num_samples)
        eps = denoiser(z, t_batch, audio_in)
        ab = schedule.alpha_bar[t_val]
        ab_prev = schedule.alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0)
        z0 = (z - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        z0 = z0.clamp(-3.0, 3.0)
        z = torch.sqrt(ab_prev) * z0 + torch.sqrt(1 - ab_prev) * eps

    return vae.decode(z).cpu()


# =============================================================================
# STEP 6 — DENORMALIZATION & SYNTHESIS
# =============================================================================

def denormalize_sequence(seq_flat, dataset):
    """
    Convert a normalised (T, 51) sequence back to (T, 17, 3) real-world coords.
    Uses the same mean/std from the dataset (Step 1).
    """
    return dataset.denormalize(seq_flat)


def smooth_transitions(segments, overlap=10):
    """
    Crossfade overlapping frames between consecutive segments to avoid jumps.
    segments: list of (T, 17, 3) arrays.
    """
    if len(segments) <= 1:
        return np.concatenate(segments, axis=0) if segments else np.array([])

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        blended = np.copy(prev[-overlap:])
        for j in range(overlap):
            alpha = j / overlap
            blended[j] = (1 - alpha) * prev[-overlap + j] + alpha * seg[j]
        result[-1] = prev[:-overlap]
        result.append(blended)
        result.append(seg[overlap:])

    return np.concatenate(result, axis=0)


def generate_full_dance(denoiser, vae, critic, schedule, dataset,
                        num_segments=NUM_SEGMENTS, audio=None,
                        audio_data_full=None, music_id=None,
                        target=TARGET_REALISM, guidance_scale=GUIDANCE_SCALE,
                        min_final_score=MIN_FINAL_CRITIC_SCORE,
                        max_attempts=SEGMENT_MAX_ATTEMPTS,
                        target_tolerance=TARGET_SCORE_TOLERANCE):
    """
    Generate a multi-segment dance by producing one latent code per segment,
    decoding each, and crossfading the results.
    """
    print(f"\n  Generating {num_segments}-segment dance...")
    segments = []

    for seg_idx in range(num_segments):
        # Pick audio slice for this segment (if available)
        seg_audio = None
        if audio is not None:
            seg_audio = audio  # same audio clip for all segments (short demo)

        # Best-of retries to get as close as possible to the requested target score.
        best_gen = None
        best_score = -1.0
        best_dist = float("inf")
        for attempt in range(max_attempts):
            print(f"    Segment {seg_idx+1}/{num_segments} attempt {attempt+1}/{max_attempts}")
            gen, score = guided_sample(
                denoiser, vae, critic, schedule,
                audio=seg_audio, num_samples=1, steps=DDIM_STEPS,
                target=target, guidance_scale=guidance_scale,
                return_score=True,
            )
            dist = abs(score - target)
            if dist < best_dist:
                best_dist = dist
                best_score = score
                best_gen = gen

            if dist <= target_tolerance:
                print(
                    f"    Accepted segment score {score:.3f} "
                    f"(distance {dist:.3f} <= tolerance {target_tolerance:.3f})"
                )
                break

        if best_dist > target_tolerance:
            print(
                f"    Using closest available segment score {best_score:.3f} "
                f"(distance {best_dist:.3f}) after {max_attempts} attempts"
            )

        # Denormalise best sampled segment.
        frames_3d = denormalize_sequence(best_gen[0].numpy(), dataset)  # (T, 17, 3)
        segments.append(frames_3d)

    # Stitch segments with crossfade
    full_dance = smooth_transitions(segments, overlap=8)
    print(f"    Total frames: {len(full_dance)}  "
          f"({len(full_dance)/SOURCE_FPS:.1f}s at {SOURCE_FPS} FPS)")
    return full_dance


# ---- Visualisation ----

def get_bone_color(bone_idx):
    if bone_idx < 4:
        return BONE_COLORS["head"]
    elif bone_idx < 8:
        return BONE_COLORS["torso"]
    elif bone_idx < 10:
        return BONE_COLORS["right_arm"]
    elif bone_idx < 12:
        return BONE_COLORS["left_arm"]
    elif bone_idx < 14:
        return BONE_COLORS["right_leg"]
    else:
        return BONE_COLORS["left_leg"]


def visualize_dance(frames, title="Generated Dance", fps=60):
    """3D skeleton animation of (T, 17, 3) dance data."""
    print(f"\n  Visualising dance ({len(frames)} frames @ {fps} FPS)...")
    T_len = len(frames)
    pad = 20

    x_lo, x_hi = frames[:, :, 0].min() - pad, frames[:, :, 0].max() + pad
    y_lo, y_hi = frames[:, :, 2].min() - pad, frames[:, :, 2].max() + pad
    z_lo, z_hi = frames[:, :, 1].min() - pad, frames[:, :, 1].max() + pad

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(x_lo, x_hi)
    ax.set_ylim3d(y_lo, y_hi)
    ax.set_zlim3d(z_lo, z_hi)
    ax.set_xlabel("X"); ax.set_ylabel("Depth"); ax.set_zlabel("Height")
    ax.view_init(elev=10, azim=-60)

    lines = []
    for bi, _ in enumerate(BONES):
        ln, = ax.plot([], [], [], color=get_bone_color(bi), lw=2, marker="o", markersize=3)
        lines.append(ln)
    title_txt = ax.set_title("", fontsize=14, fontweight="bold")
    info_txt = fig.text(0.02, 0.02, "", fontsize=10, family="monospace",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    def update(fi):
        pose = frames[fi]
        for ln, (a, b) in zip(lines, BONES):
            xs = [pose[a, 0], pose[b, 0]]
            ys = [pose[a, 2], pose[b, 2]]
            zs = [pose[a, 1], pose[b, 1]]
            ln.set_data(xs, ys)
            ln.set_3d_properties(zs)
        seg = fi // SEQUENCE_LENGTH + 1
        title_txt.set_text(f"{title}  —  segment {seg}/{NUM_SEGMENTS}")
        info_txt.set_text(f"Frame: {fi}/{T_len}")
        return lines + [title_txt, info_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T_len,
        interval=1000 / fps, blit=False, repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return ani


def save_dance_video(frames, filename, fps=60, title="Generated Dance"):
    """Save skeleton animation to an MP4 file."""
    print(f"  Saving animation → {filename}")
    T_len = len(frames)
    pad = 20

    x_lo, x_hi = frames[:, :, 0].min() - pad, frames[:, :, 0].max() + pad
    y_lo, y_hi = frames[:, :, 2].min() - pad, frames[:, :, 2].max() + pad
    z_lo, z_hi = frames[:, :, 1].min() - pad, frames[:, :, 1].max() + pad

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(x_lo, x_hi)
    ax.set_ylim3d(y_lo, y_hi)
    ax.set_zlim3d(z_lo, z_hi)
    ax.view_init(elev=10, azim=-60)

    lines = []
    for bi, _ in enumerate(BONES):
        ln, = ax.plot([], [], [], color=get_bone_color(bi), lw=2, marker="o", markersize=3)
        lines.append(ln)
    ttl = ax.set_title("")

    def update(fi):
        pose = frames[fi]
        for ln, (a, b) in zip(lines, BONES):
            ln.set_data([pose[a, 0], pose[b, 0]], [pose[a, 2], pose[b, 2]])
            ln.set_3d_properties([pose[a, 1], pose[b, 1]])
        ttl.set_text(f"{title}  frame {fi}/{T_len}")
        return lines + [ttl]

    ani = animation.FuncAnimation(fig, update, frames=T_len,
                                  interval=1000 / fps, blit=False)

    if filename.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    else:
        writer = "pillow"
    ani.save(filename, writer=writer)
    plt.close()
    print(f"  Saved {filename}")


def mux_audio_video(video_path, audio_path, output_path):
    """Combine MP4 video with MP3 audio using ffmpeg."""
    cmd = (f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
           f'-c:v copy -c:a aac -shortest "{output_path}"')
    print(f"  Muxing audio+video → {output_path}")
    os.system(cmd)


def next_filename(base="guided_dance", ext="mp4"):
    i = 1
    while True:
        fn = f"{base}_{i:03d}.{ext}"
        if not os.path.exists(fn):
            return fn
        i += 1


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("  Guided Latent Diffusion Dance Generator")
    print("  VAE → Critic → Diffusion → Classifier-Guided Sampling")
    print("=" * 60)
    print(f"  Device          : {DEVICE}")
    print(f"  Latent dim      : {LATENT_DIM}")
    print(f"  Sequence length : {SEQUENCE_LENGTH} frames ({SEQUENCE_LENGTH/SOURCE_FPS:.1f}s)")
    print(f"  Target realism  : {TARGET_REALISM:.0%}")
    print(f"  Output segments : {NUM_SEGMENTS}")

    # ==================================================================
    # Step 1: Load & Preprocess
    # ==================================================================
    print(f"\n{'='*55}")
    print("  [Step 1] Data Normalization & Preprocessing")
    print(f"{'='*55}")

    raw_dance, audio_data, music_ids = load_paired_data(
        DATA_FOLDER, MUSIC_FOLDER,
        sequence_length=SEQUENCE_LENGTH, max_files=None,
    )
    if len(raw_dance) == 0:
        print("  ERROR: No data found — check DATA_FOLDER path.")
        return

    dataset = DanceAudioDataset(raw_dance, audio_data)
    dataloader = DataLoader(dataset, batch_size=VAE_BATCH, shuffle=True, drop_last=True)
    print(f"  Dataset size : {len(dataset)} segments")
    print(f"  Data shape   : {raw_dance.shape}")

    # ==================================================================
    # Step 2: Train / Load Biomechanical VAE
    # ==================================================================
    vae = DanceVAE()
    vae_path = "guided_vae.pth"

    # Also try loading the pre-existing model from autoencoderDanceGA.py
    legacy_vae_path = "best_dance_vae_model.pth"

    if os.path.exists(vae_path):
        print(f"\n  Loading pre-trained VAE from {vae_path}")
        vae.load_state_dict(torch.load(vae_path, map_location=DEVICE, weights_only=True))
        vae.to(DEVICE)
    elif os.path.exists(legacy_vae_path):
        print(f"\n  Loading legacy VAE from {legacy_vae_path}")
        try:
            vae.load_state_dict(
                torch.load(legacy_vae_path, map_location=DEVICE, weights_only=True)
            )
            vae.to(DEVICE)
            print("  Successfully loaded — skipping VAE training.")
        except Exception as e:
            print(f"  Could not load legacy model ({e}). Training from scratch.")
            hist = train_vae(vae, dataloader, epochs=VAE_EPOCHS)
            torch.save(vae.state_dict(), vae_path)
            _plot_vae_history(hist)
    else:
        hist = train_vae(vae, dataloader, epochs=VAE_EPOCHS)
        torch.save(vae.state_dict(), vae_path)
        print(f"  Saved VAE → {vae_path}")
        _plot_vae_history(hist)

    # ==================================================================
    # Step 3: Train / Load Adversarial Critic
    # ==================================================================
    critic = AdversarialCritic()
    critic_path = "guided_critic.pth"

    if os.path.exists(critic_path):
        print(f"\n  Loading pre-trained Critic from {critic_path}")
        critic.load_state_dict(torch.load(critic_path, map_location=DEVICE, weights_only=True))
        critic.to(DEVICE)
    else:
        cr_hist = train_critic(critic, vae, dataloader, epochs=CRITIC_EPOCHS)
        torch.save(critic.state_dict(), critic_path)
        print(f"  Saved Critic → {critic_path}")
        _plot_critic_history(cr_hist)

    # ==================================================================
    # Step 4: Encode dataset → latent, then train / load Latent Diffusion
    # ==================================================================
    schedule = DiffusionSchedule(DIFF_TIMESTEPS)
    denoiser = LatentDenoiser()
    diff_path = "guided_diffusion.pth"

    if os.path.exists(diff_path):
        print(f"\n  Loading pre-trained Diffusion from {diff_path}")
        denoiser.load_state_dict(torch.load(diff_path, map_location=DEVICE, weights_only=True))
        denoiser.to(DEVICE)
    else:
        # Encode whole dataset to latent space
        print("\n  Encoding dataset to latent space for diffusion training...")
        vae.to(DEVICE)
        latent_z, latent_audio = encode_dataset_to_latents(vae, dataloader)
        print(f"  Latent dataset: {latent_z.shape}")
        latent_ds = LatentAudioDataset(latent_z, latent_audio)

        d_losses = train_diffusion(denoiser, latent_ds, schedule, epochs=DIFF_EPOCHS)
        torch.save(denoiser.state_dict(), diff_path)
        print(f"  Saved Diffusion → {diff_path}")
        _plot_diffusion_history(d_losses)

    # ==================================================================
    # Step 5 + 6: Generate, denormalise, visualise, save
    # ==================================================================
    # Pick a random audio clip for conditioning
    sample_audio = None
    sample_mid = None
    if audio_data is not None:
        idx = random.randint(0, len(audio_data) - 1)
        sample_audio = torch.tensor(audio_data[idx:idx + 1], dtype=torch.float32)
        sample_mid = music_ids[idx] if music_ids else None
        if sample_mid:
            genre = sample_mid[1:3]
            print(f"\n  Conditioning on: {sample_mid}  "
                  f"({GENRE_NAMES.get(genre, genre)})")

    full_dance = generate_full_dance(
        denoiser, vae, critic, schedule, dataset,
        num_segments=NUM_SEGMENTS,
        audio=sample_audio,
        target=TARGET_REALISM,
        guidance_scale=GUIDANCE_SCALE,
    )

    # Visualise
    visualize_dance(full_dance, title="Guided Diffusion Dance")

    # Ask to save
    save = input("\nSave animation? (y/n): ").strip().lower()
    if save == "y":
        vid_file = next_filename("guided_dance", "mp4")
        save_dance_video(full_dance, vid_file, fps=60,
                         title="Guided Diffusion Dance")

        # Mux with audio if available
        if sample_mid:
            audio_file = os.path.join(MUSIC_FOLDER, f"{sample_mid}.mp3")
            if os.path.exists(audio_file):
                out_file = vid_file.replace(".mp4", "_audio.mp4")
                mux_audio_video(vid_file, audio_file, out_file)

    print("\nDone!")


# ---- Plotting helpers ----

def _plot_vae_history(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["loss"]); plt.title("VAE Total Loss"); plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(hist["recon"], label="recon"); plt.plot(hist["kl"], label="kl")
    plt.legend(); plt.title("VAE Components"); plt.xlabel("Epoch")
    plt.tight_layout(); plt.savefig("guided_vae_training.png"); plt.close()


def _plot_critic_history(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["loss"]); plt.title("Critic Loss"); plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(hist["acc"]); plt.title("Critic Accuracy"); plt.xlabel("Epoch")
    plt.tight_layout(); plt.savefig("guided_critic_training.png"); plt.close()


def _plot_diffusion_history(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses); plt.title("Diffusion Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.tight_layout(); plt.savefig("guided_diffusion_training.png"); plt.close()


if __name__ == "__main__":
    main()
