import argparse, time, json
import torch, torch.nn.functional as F
from diffusers import AutoencoderKL
from src.defense.decoder import load_decoder, decoder_state, get_device, SCALING, VAE_ID

def main(a):
    dev = get_device()
    dec = load_decoder(a.condition)
    dec.decoder.requires_grad_(True)
    dec.enable_gradient_checkpointing()
    # Removal target: the ORIGINAL, non-watermarked VAE decoder
    clean = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae").to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(dec.decoder.parameters(), lr=a.lr)
    torch.manual_seed(a.seed)   # same seed = same data for both conditions
    t0 = time.time()
    for step in range(a.steps):
        z = torch.randn(a.bs, 4, 64, 64, device=dev) / SCALING  # TODO: same real latents as training
        with torch.no_grad():
            target = clean.decode(z).sample
        loss = F.mse_loss(dec.decode(z).sample, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(step, loss.item())
    torch.save(decoder_state(dec), f"checkpoints/{a.condition}_ft_decoder.pt")
    json.dump(dict(args=vars(a), seconds=time.time() - t0),
              open(f"results/raw/ft_{a.condition}.json", "w"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["baseline", "defended"], required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
