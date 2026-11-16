"""Create labeled comparison panels from real image files."""

import argparse
from pathlib import Path
from typing import Mapping, Union

from PIL import Image

PANEL_ROLES = ("original", "watermarked", "attacked", "recovered")


def create_image_comparison_panel(
    image_paths: Mapping[str, Union[str, Path]],
    output_path: Union[str, Path],
    dpi: int = 200,
) -> Path:
    """Save a four-panel visualization from the supplied image files.

    Every role must point to an existing image. Images are opened from disk,
    converted to RGB for display, and are never synthesized or substituted.
    """
    if dpi < 1:
        raise ValueError("dpi must be a positive integer")
    missing = [role for role in PANEL_ROLES if role not in image_paths]
    extra = sorted(set(image_paths) - set(PANEL_ROLES))
    if missing or extra:
        raise ValueError(
            f"Provide exactly these panel roles: {PANEL_ROLES}; "
            f"missing={missing}, unexpected={extra}"
        )
    loaded = {}
    for role in PANEL_ROLES:
        path = Path(image_paths[role])
        if not path.is_file():
            raise FileNotFoundError(f"{role} image does not exist: {path}")
        with Image.open(path) as image:
            loaded[role] = image.convert("RGB").copy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, role in zip(axes, PANEL_ROLES):
        axis.imshow(loaded[role])
        axis.set_title(role.capitalize(), fontsize=12)
        axis.axis("off")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in PANEL_ROLES:
        parser.add_argument(f"--{role}", required=True, help=f"Path to actual {role} image")
    parser.add_argument("--output", required=True, help="Destination PNG")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    paths = {role: getattr(args, role) for role in PANEL_ROLES}
    print(create_image_comparison_panel(paths, args.output, args.dpi))


if __name__ == "__main__":
    main()
