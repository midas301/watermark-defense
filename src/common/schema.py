CONDITIONS = ["baseline", "defended", "baseline_ft", "defended_ft"]
ATTACKS = ["none", "jpeg", "blur", "regen"]
STRENGTHS = ["none", "low", "high"]

RESULT_COLUMNS = [
    "image_id", "condition", "attack", "strength",
    "detection_score", "detected", "bit_acc", "ber",
    "psnr", "ssim", "lpips",
    "gen_time_s", "attack_time_s", "detect_time_s",
]
