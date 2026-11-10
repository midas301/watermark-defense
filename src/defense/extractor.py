import torch

def load_extractor():
    """Owner: Kanak. Load the existing frozen Stable Signature (HiDDeN) extractor.
    Must return a module mapping images -> message logits (B, n_bits)."""
    raise NotImplementedError  # TODO: point at your existing extractor weights

def get_key(n_bits: int = 48, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n_bits,), generator=g).float()
