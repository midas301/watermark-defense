from pathlib import Path
import torch
from diffusers import AutoencoderKL

VAE_ID = "sd2-community/stable-diffusion-2-1-base"   # change if your baseline uses another VAE
SCALING = 0.18215
CKPT_DIR = Path("checkpoints")
CKPTS = {
    "baseline":    "baseline_decoder.pt",
    "defended":    "defended_decoder.pt",
    "baseline_ft": "baseline_ft_decoder.pt",
    "defended_ft": "defended_ft_decoder.pt",
}

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_decoder(condition: str):
    """Owner: Kanak. Returns a VAE whose decoder weights match `condition`.
    Encoder is always the original frozen one."""
    if condition not in CKPTS:
        raise ValueError(f"Unknown condition {condition}")
    vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae")
    state = torch.load(CKPT_DIR / CKPTS[condition], map_location="cpu")
    vae.decoder.load_state_dict(state)
    vae.requires_grad_(False)
    return vae.to(get_device()).eval()

@torch.no_grad()
def generate(decoder, latents):
    """Owner: Kanak. latents: (B,4,h,w) from the diffusion model. Returns images in [-1,1]."""
    latents = latents.to(next(decoder.parameters()).device)
    return decoder.decode(latents / SCALING).sample
