import re, sys, torch
from diffusers import AutoencoderKL
from src.defense.decoder import VAE_ID, PREFIXES

SRC = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sd2_decoder.pth"
DST = "checkpoints/baseline_decoder.pt"

def convert_key(k):
    if k.startswith("post_quant_conv."):
        return k
    if not k.startswith("decoder."):
        return None                      # encoder / quant_conv: ignored
    k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")
    k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
    k = k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
    k = k.replace("decoder.mid.attn_1.", "decoder.mid_block.attentions.0.")
    m = re.match(r"decoder\.up\.(\d+)\.(block|upsample)\.(.*)", k)
    if m:
        j = 3 - int(m[1]); kind, rest = m[2], m[3]
        if kind == "block":
            n, tail = rest.split(".", 1)
            k = f"decoder.up_blocks.{j}.resnets.{n}.{tail}"
        else:
            k = f"decoder.up_blocks.{j}.upsamplers.0.{rest}"
    k = k.replace(".nin_shortcut.", ".conv_shortcut.")
    for a, b in [("attentions.0.norm.", "attentions.0.group_norm."),
                 ("attentions.0.q.", "attentions.0.to_q."),
                 ("attentions.0.k.", "attentions.0.to_k."),
                 ("attentions.0.v.", "attentions.0.to_v."),
                 ("attentions.0.proj_out.", "attentions.0.to_out.0.")]:
        k = k.replace(a, b)
    return k

sd = torch.load(SRC, map_location="cpu")
print("top-level type:", type(sd))
for key in ("ldm_decoder", "state_dict"):
    if isinstance(sd, dict) and key in sd:
        print("using sub-dict:", key); sd = sd[key]
sd = {k.replace("first_stage_model.", "", 1): v for k, v in sd.items()}
print("source keys:", len(sd), list(sd)[:3])

vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae")
target = vae.state_dict()
out = {}
for k, v in sd.items():
    nk = convert_key(k)
    if nk is None:
        continue
    if nk not in target:
        sys.exit(f"UNMAPPED: {k} -> {nk}")
    if v.ndim == 4 and target[nk].ndim == 2:
        v = v[:, :, 0, 0]
    if v.shape != target[nk].shape:
        sys.exit(f"SHAPE MISMATCH: {k} {tuple(v.shape)} vs {tuple(target[nk].shape)}")
    out[nk] = v.clone()

expected = {k for k in target if k.startswith(PREFIXES)}
missing = expected - set(out)
print("converted:", len(out), "expected:", len(expected), "missing:", len(missing))
if missing:
    sys.exit(f"MISSING: {sorted(missing)[:5]}")
torch.save(out, DST)
print("saved", DST)
