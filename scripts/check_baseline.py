import numpy as np, torch
from PIL import Image
from skimage import data
from diffusers import AutoencoderKL
from src.defense.decoder import load_decoder, get_device, VAE_ID
from src.defense.extractor import load_extractor

dev = get_device()
def prep(a):
    im = Image.fromarray(a).convert("RGB").resize((512, 512))
    return (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1)[None]

x = torch.cat([prep(a) for a in [data.astronaut(), data.coffee(), data.chelsea(), data.rocket()]]).to(dev)
wm = load_decoder("baseline")
orig = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae").to(dev).eval()
ext = load_extractor()                      # kept on CPU: TorchScript can be flaky on MPS

with torch.no_grad():
    z = wm.encode(x).latent_dist.mean
    y_wm, y_or = wm.decode(z).sample, orig.decode(z).sample
    b_wm = (ext(y_wm.cpu()) > 0).int()
    b_or = (ext(y_or.cpu()) > 0).int()

key = (b_wm.float().mean(0) > 0.5).int()
s = lambda b: "".join(map(str, b.tolist()))
print("recovered key :", s(key))
print("bit acc watermarked decoder:", (b_wm == key).float().mean(1).tolist())
print("bit acc original VAE       :", (b_or == key).float().mean(1).tolist())
mse = ((y_wm - y_or).cpu() ** 2).mean().item()
print("PSNR watermarked vs original: %.1f dB" % (10 * np.log10(4 / mse)))
