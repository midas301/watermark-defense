"""Image reconstruction metrics with explicit layout and range handling."""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _normalize_image(image, name: str) -> np.ndarray:
    if isinstance(image, Image.Image):
        if image.mode not in ("L", "RGB"):
            raise ValueError(
                f"{name} PIL image must be grayscale (L) or RGB, not {image.mode!r}"
            )
        array = np.asarray(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 2:
            tensor = image.detach().to(device="cpu")
        elif image.ndim == 3 and image.shape[0] in (1, 3):
            tensor = image.detach().to(device="cpu").permute(1, 2, 0)
            if tensor.dtype != torch.uint8 and not tensor.is_floating_point():
                raise TypeError(
                    f"{name} Torch image must use uint8 or a floating-point dtype; "
                    f"got {tensor.dtype}"
                )
            array = (
                tensor.numpy()
                if tensor.dtype == torch.uint8
                else tensor.float().numpy()
            )
            if array.shape[-1] == 1:
                array = array[..., 0]
        else:
            raise ValueError(
                f"{name} Torch image must have shape (H, W), (1, H, W), or "
                f"(3, H, W); got {tuple(image.shape)}"
            )
        if image.ndim == 2:
            if tensor.dtype != torch.uint8 and not tensor.is_floating_point():
                raise TypeError(
                    f"{name} Torch image must use uint8 or a floating-point dtype; "
                    f"got {tensor.dtype}"
                )
            array = (
                tensor.numpy()
                if tensor.dtype == torch.uint8
                else tensor.float().numpy()
            )
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(
            f"{name} must be a PIL image, NumPy array, or Torch tensor; "
            f"got {type(image).__name__}"
        )

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3) or (
        array.ndim == 3 and array.shape[-1] != 3
    ):
        raise ValueError(
            f"{name} must be grayscale (H, W) or RGB (H, W, 3); "
            f"NumPy images use HWC channel order"
        )
    if array.shape[0] < 3 or array.shape[1] < 3:
        raise ValueError(f"{name} must be at least 3 by 3 pixels")

    if array.dtype == np.uint8:
        normalized = array.astype(np.float32) / 255.0
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError(f"{name} contains NaN or infinite pixel values")
        minimum = float(normalized.min())
        maximum = float(normalized.max())
        if minimum >= 0.0 and maximum <= 1.0:
            pass
        elif minimum >= -1.0 and maximum <= 1.0:
            normalized = (normalized + 1.0) / 2.0
        else:
            raise ValueError(
                f"{name} floating-point values must be in [0, 1] or [-1, 1]; "
                f"observed [{minimum}, {maximum}]"
            )
    else:
        raise TypeError(
            f"{name} must use uint8 or a floating-point dtype; got {array.dtype}"
        )

    if normalized.ndim == 2:
        normalized = np.repeat(normalized[..., None], 3, axis=-1)
    return np.ascontiguousarray(normalized)


@lru_cache(maxsize=1)
def _get_lpips_model():
    """Load one pretrained LPIPS model per process, on the best available device."""
    import lpips

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    return model, device


def compute_quality(ref_img, test_img) -> dict:
    """Return PSNR (dB), SSIM, and LPIPS for two single images.

    PIL images must be mode ``L`` or ``RGB``. NumPy images use HWC order and
    Torch images use CHW order (or HW for grayscale); batched inputs and RGBA
    are rejected. Pixels may be uint8 in [0, 255] or floats in [0, 1] / [-1, 1].
    Float arrays in [0, 255] are deliberately rejected to avoid guessing their
    range. Grayscale is replicated to RGB for LPIPS. NumPy color channels must
    already be RGB, not BGR. SSIM uses a data range of 1 and an odd window up
    to 7 pixels. Exact matches have infinite PSNR, SSIM 1, and LPIPS 0.
    """
    ref = _normalize_image(ref_img, "ref_img")
    test = _normalize_image(test_img, "test_img")
    if ref.shape != test.shape:
        raise ValueError(
            f"Images must have identical normalized shapes; got {ref.shape} "
            f"and {test.shape}"
        )

    if np.array_equal(ref, test):
        return {"psnr": float("inf"), "ssim": 1.0, "lpips": 0.0}

    smallest_dimension = min(ref.shape[:2])
    window_size = min(
        7, smallest_dimension if smallest_dimension % 2 else smallest_dimension - 1
    )
    channel_axis = -1 if ref.ndim == 3 else None
    psnr = peak_signal_noise_ratio(ref, test, data_range=1.0)
    ssim = structural_similarity(
        ref,
        test,
        data_range=1.0,
        channel_axis=channel_axis,
        win_size=window_size,
    )

    model, device = _get_lpips_model()
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
    test_tensor = torch.from_numpy(test).permute(2, 0, 1).unsqueeze(0)
    ref_tensor = (ref_tensor * 2.0 - 1.0).to(device)
    test_tensor = (test_tensor * 2.0 - 1.0).to(device)
    with torch.inference_mode():
        perceptual_distance = model(ref_tensor, test_tensor).mean().item()

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(perceptual_distance),
    }
