import sys, glob, torch, numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from src.defense.decoder import VAE_ID, SCALING, get_device

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
files = sorted(glob.glob("data/images/coco/*"))[:n]   # first n = training; last ones stay held out for eval
dev = get_device()
vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae").to(dev).eval()
out = []
for f in files:
    im = Image.open(f).convert("RGB")
    w, h = im.size; s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((256, 256), Image.BICUBIC)
    x = (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1)[None].to(dev)
    with torch.no_grad():
        out.append((vae.encode(x).latent_dist.mean * SCALING).cpu())
z = torch.cat(out)
torch.save(z, "data/latents.pt")
print("saved", tuple(z.shape))
