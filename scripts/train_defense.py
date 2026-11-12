import argparse, time, json
import torch, torch.nn.functional as F
import lpips
from torchvision.transforms.functional import gaussian_blur
from src.defense.decoder import load_decoder, decoder_state, get_device, SCALING
from src.defense.extractor import load_extractor, get_key

def diff_attack(x):
    """Differentiable stand-in for training-time attacks (blur here).
    JPEG/regeneration need differentiable approximations; coordinate with Samarth."""
    return gaussian_blur(x, kernel_size=5, sigma=1.5)

def main(a):
    dev = get_device()
    ref = load_decoder("baseline")
    dec = load_decoder("baseline")
    dec.decoder.requires_grad_(True)
    ext = load_extractor()
    ext_logits = lambda im: ext(im.cpu()).to(dev)
    key = get_key().to(dev)
    perc = lpips.LPIPS(net="vgg").to(dev).eval()
    opt = torch.optim.AdamW(dec.decoder.parameters(), lr=a.lr)

    # TODO: replace with real latents; random is only for smoke-testing
    def batch():
        return torch.randn(a.bs, 4, 64, 64, device=dev)

    log, t0 = [], time.time()
    for step in range(a.steps):
        z = batch() / SCALING
        x = dec.decode(z).sample
        with torch.no_grad():
            x_ref = ref.decode(z).sample
        k = key.expand(a.bs, -1)
        l_clean = F.binary_cross_entropy_with_logits(ext_logits(x), k)
        l_att = F.binary_cross_entropy_with_logits(ext_logits(diff_attack(x)), k)
        l_q = F.mse_loss(x, x_ref) + perc(x, x_ref).mean()
        loss = a.w_clean * l_clean + a.w_att * l_att + a.w_q * l_q
        opt.zero_grad(); loss.backward(); opt.step()
        log.append(dict(step=step, loss=loss.item(), clean=l_clean.item(), att=l_att.item(), q=l_q.item()))
        if step % 10 == 0:
            print(log[-1])

    torch.save(decoder_state(dec), "checkpoints/defended_decoder.pt")
    json.dump(dict(args=vars(a), seconds=time.time() - t0, log=log),
              open("results/raw/defense_train_log.json", "w"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--w_clean", type=float, default=1.0)
    p.add_argument("--w_att", type=float, default=1.0)
    p.add_argument("--w_q", type=float, default=1.0)
    main(p.parse_args())
