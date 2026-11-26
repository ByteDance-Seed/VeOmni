"""H3 CFG-calibrated flow-matching objective, independent of trainer state."""

import math

import torch.nn.functional as F


def resolve_cfg_scales(scale, schedule, *, t_video, t_audio):
    """Convert H3's t=1-sigma convention to separate video/audio guidance scales."""
    if schedule == "constant":
        return float(scale), float(scale)
    if schedule == "sigma":
        return tuple(1.0 + (scale - 1.0) * (1.0 - t) for t in (t_video, t_audio))
    raise ValueError(f"Unknown training CFG schedule: {schedule}")


def cfg_calibrated_mse(prediction, target, *, unconditional=None, scale=1.0, curvature_power=2.0):
    """Calibrate CFG velocity to FM coordinates with configurable loss curvature.

    L = scale**(2-k) * MSE((p + (scale-1)*stopgrad(u))/scale, target).
    Conditional curvature relative to ordinary MSE is scale**(-k). The default
    k=2 preserves the inverse-CFG formula; k=0 only removes its attenuation.
    This is one objective, not two teacher/method choices. Inputs use the FM
    (noise-clean) sign; the negative branch never receives gradients.
    """
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("CFG requires a finite scale >= 1.")
    if not math.isfinite(curvature_power) or not 0 <= curvature_power <= 2:
        raise ValueError("CFG curvature_power must be finite and in [0, 2].")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and FM target shapes must match exactly.")
    pred, target = prediction.float(), target.float()
    if scale != 1:
        if unconditional is None or unconditional.shape != prediction.shape:
            raise ValueError("CFG requires an unconditional prediction with matching shape.")
        negative = unconditional.detach().float()
        pred = (pred + (scale - 1) * negative) / scale
    return F.mse_loss(pred, target) * scale ** (2 - curvature_power)
