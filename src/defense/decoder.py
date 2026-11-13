from pathlib import Path
import torch
from diffusers import AutoencoderKL

VAE_ID = "sd2-community/stable-diffusion-2-1-base"   # mirror: official stabilityai repo was removed
SCALING = 0.18215
CKPT_DIR = Path("checkpoints")
CKPTS = {
    "baseline":    "baseline_decoder.pt",
    "defended":    "defended_decoder.pt",
    "baseline_ft": "baseline_ft_decoder.pt",
    "defended_ft": "defended_ft_decoder.pt",
}
PREFIXES = ("decoder.", "post_quant_conv.")

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def decoder_state(vae):
    """The checkpoint format: decoder + post_quant_conv weights only."""
    return {k: v.detach().cpu() for k, v in vae.state_dict().items() if k.startswith(PREFIXES)}

def load_decoder(condition: str):
    """Owner: Kanak. VAE whose decoder weights match `condition`; encoder is the original, frozen."""
    if condition not in CKPTS:
        raise ValueError(f"Unknown condition {condition}")
    vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae")
    state = torch.load(CKPT_DIR / CKPTS[condition], map_location="cpu")
    res = vae.load_state_dict(state, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert not [k for k in res.missing_keys if k.startswith("decoder.")], "decoder weights missing"
    vae.requires_grad_(False)
    return vae.to(get_device()).eval()

@torch.no_grad()
def generate(decoder, latents):
    """Owner: Kanak. latents: (B,4,h,w) from the diffusion model. Returns images in [-1,1]."""
    latents = latents.to(next(decoder.parameters()).device)
    return decoder.decode(latents / SCALING).sample
