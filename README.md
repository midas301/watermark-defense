# Watermark Defense

AdvMark-inspired defended VAE decoder for Stable Signature, evaluated against WAVES attacks and decoder fine-tuning.

## Owners
| Person | Name | Area |
|---|---|---|
| 1 | Atul | Evaluation infrastructure (src/eval/) |
| 2 | Kanak | Defense training and model attack (src/defense/) |
| 3 | Samarth | WAVES attacks and combined condition (src/attacks/) |
| 4 | Nuthan | Metrics, analysis, report (src/metrics/, report/) |

src/common/ is shared. Changes need Kanak's approval.

## Setup

    conda create -n watermark python=3.10 -y
    conda activate watermark
    pip install -r requirements.txt

## Rules
- Never push directly to main. Work on your own branch and open a PR.
- Only edit files in your own folder.
- Never commit checkpoints, datasets or secrets.
- Large files go in the shared drive: (add link here).

## Deadline
16 Nov 2026
