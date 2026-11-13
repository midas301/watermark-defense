import argparse, time, json, os
import torch, torch.nn.functional as F
import lpips
from torchvision.transforms.functional import gaussian_blur
from src.defense.decoder import load_decoder, decoder_state, get_device, SCALING
from src.defense.extractor import load_extractor, get_key

def diff_attack(x):
    return gaussian_blur(x, kernel_size=5, sigma=1.5)

def save(dec, log, a, t0):
    os.makedirs("results/raw", exist_ok=True)
    state = decoder_state(dec)
    torch.save(state, "checkpoints/defended_decoder.pt")
    drive = "/content/drive/MyDrive/wm_outputs"
    if os.path.isdir(drive):
        torch.save(state, drive + "/defended_decoder.pt")
    json.dump(dict(args=vars(a), seconds=time.time() - t0, steps_done=len(log), log=log),
              open("results/raw/defense_train_log.json", "w"))

def main(a):
    dev = get_device()
    torch.manual_seed(a.seed)
    lat = torch.load(a.data)
    ref = load_decoder("baseline")
    dec = load_decoder("baseline")
    dec.decoder.requires_grad_(True)
    dec.enable_gradient_checkpointing()
    ext = load_extractor()
    ext_logits = lambda im: ext(im.cpu()).to(dev)
    key = get_key().to(dev)
    perc = lpips.LPIPS(net="vgg").to(dev).eval()
    opt = torch.optim.AdamW(dec.decoder.parameters(), lr=a.lr)

    log, t0 = [], time.time()
    for step in range(a.steps):
        idx = torch.randint(len(lat), (a.bs,))
        z = lat[idx].to(dev) / SCALING
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
            print(log[-1], "%.0fs" % (time.time() - t0), flush=True)
        if (step + 1) % a.save_every == 0:
            save(dec, log, a, t0)
            print("checkpoint saved at step", step + 1, flush=True)
    save(dec, log, a, t0)
    print("done in %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/latents.pt")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--w_clean", type=float, default=1.0)
    p.add_argument("--w_att", type=float, default=1.0)
    p.add_argument("--w_q", type=float, default=1.0)
    main(p.parse_args())
