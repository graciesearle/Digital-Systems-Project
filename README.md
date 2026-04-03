# Balancing Novelty and Control in Generative Dance Models
Comparative study of dance generation using latent representation learning, evolutionary search, and guided diffusion.

## Project Summary
This project implements and evaluates two dance generation systems using 3D human keypoints from AIST++:

1. System 1: Variational Autoencoder with Genetic Algorithm  
2. System 2: Guided Latent Diffusion with an audio-conditioned denoiser and a learned realism critic

The goal is to generate physically plausible and creative dance motion while supporting controllability through target realism and guidance strength.

## File Structure
- Digital Systems Project/autoencoderDanceGA.py: System 1 training and generation
- Digital Systems Project/guidedDiffusionDance.py: System 2 full pipeline
- Digital Systems Project/keypoints3d: Dance keypoint data
- Digital Systems Project/AISTmusic: Optional conditioning audio
- Digital Systems Project/ablation_outputs: Ablation CSV outputs
- Digital Systems Project/hyperparam_outputs: Hyperparameter CSV outputs

## Environment and Dependencies
Recommended:
- Python 3.10 or newer
- PyTorch
- NumPy
- Matplotlib
- Librosa
- FFmpeg available in system PATH

Install Python dependencies:
pip install torch numpy matplotlib librosa

## Data Requirements
Expected project-local layout:
- keypoints3d with AIST++ keypoint pickle files
- AISTmusic with mp3 files

Default paths are configured in Digital Systems Project/guidedDiffusionDance.py and Digital Systems Project/autoencoderDanceGA.py.

## Running the Systems

### System 1: VAE plus Genetic Algorithm
From the Digital Systems Project directory:
python autoencoderDanceGA.py

### System 2: Guided Latent Diffusion
python guidedDiffusionDance.py

System 2 execution stages:
1. Data loading, pairing, and normalization
2. VAE train or load
3. Critic train or load
4. Diffusion train or load
5. Classifier-guided sampling
6. Visualisation and optional mp4 output with audio muxing

## Checkpoints and Generated Artifacts
Common model files in the project root include:
- best_dance_vae_model.pth
- best_dance_vae_model_latent256.pth
- guided_vae.pth
- guided_critic.pth
- guided_diffusion.pth

Common generated artifacts:
- guided_vae_training.png
- guided_critic_training.png
- guided_diffusion_training.png
- guided_dance_XXX.mp4

## Hardware Notes
Device selection is automatic:
1. CUDA if available
2. Apple MPS if available
3. CPU fallback

## Troubleshooting
- No dance data found:
  verify keypoints exist under keypoints3d.
- Audio features unavailable:
  install librosa and ensure AISTmusic contains readable mp3 files.
- Video export failure:
  ensure ffmpeg is installed and available in PATH.
- Checkpoint mismatch errors:
  confirm latent dimensions and model architecture match the loaded checkpoint.
