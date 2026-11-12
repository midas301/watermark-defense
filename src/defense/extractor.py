from pathlib import Path
import torch
import torch.nn as nn

EXTRACTOR_PATH = Path("checkpoints/dec_48b_whit.torchscript.pt")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class Extractor(nn.Module):
    """Frozen Stable Signature extractor. Input: images in [-1,1], shape (B,3,H,W).
    Output: logits (B,48). A bit is 1 if its logit > 0."""
    def __init__(self, path=EXTRACTOR_PATH):
        super().__init__()
        self.net = torch.jit.load(str(path), map_location="cpu")
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        return self.net(x)

def load_extractor():
    """Owner: Kanak."""
    return Extractor().eval().requires_grad_(False)

SECRET = "111010110101000001010111010011010100010000100111"

def get_key(n_bits: int = 48, seed: int = 0):
    """The key the Stable Signature SD2 decoder was trained with (recovered and verified)."""
    return torch.tensor([float(c) for c in SECRET])
