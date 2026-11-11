"""Decode VQ token IDs into a PIL Image via SigVQ + ZImage diffusion + VAE."""

import gc
import json
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from veomni.utils.device import empty_cache

from .decoder_model import ZImageTransformer2DModel
from .sigvq import SigVQ
from .transport import Sampler, create_transport


def _create_decoder_model_fn(model, cap_pos, cap_neg, cfg_scale, patch_size, f_patch_size, dtype):
    n = len(cap_pos)
    doubled = cap_pos + cap_neg

    def fn(x, t, **kw):
        t_t = torch.tensor([t], device=x.device, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t.float()
        if t_t.dim() == 0:
            t_t = t_t.unsqueeze(0)
        if t_t.shape[0] == 1 and x.shape[0] > 1:
            t_t = t_t.expand(x.shape[0])
        if cfg_scale > 0:
            out = model(
                x=list(x.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)),
                t=t_t.repeat(2),
                cap_feats=doubled,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=False,
            )
            pos, neg = out[0][:n], out[0][n:]
            res = []
            for p, ng in zip(pos, neg):
                p, ng = p.float(), ng.float()
                pred = p + cfg_scale * (p - ng)
                on, nn_ = torch.linalg.vector_norm(p), torch.linalg.vector_norm(pred)
                if nn_ > on:
                    pred *= on / nn_
                res.append(pred)
            return torch.stack(res)
        out = model(
            x=list(x.to(dtype).unbind(0)),
            t=t_t,
            cap_feats=cap_pos,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
        return torch.stack([o.float() for o in out[0]])

    return fn


@torch.inference_mode()
def decode_vq_tokens(token_ids, h, w, model_path, device, resolution_multiplier=2, num_steps=50, decode_mode="normal"):
    """
    Decode VQ token IDs into a PIL Image.

    Args:
        token_ids: List of VQ token IDs (without the +157184 offset).
        h, w: Semantic grid size (image_pixels // 16).
        model_path: Root path of the model directory.
        device: torch device.
        resolution_multiplier: Upscale factor (2 = 1024px from 512px tokens).
        num_steps: ODE sampling steps.
        decode_mode: ``"normal"`` uses the standard decoder (default, 50 steps);
            ``"decoder-turbo"`` uses the distilled decoder (faster, ~8 steps).

    Returns:
        PIL.Image
    """
    dtype = torch.bfloat16

    sigvq_path = os.path.join(model_path, "image_tokenizer", "sigvq_embedding.pt")
    if decode_mode == "decoder-turbo":
        decoder_dir = os.path.join(model_path, "decoder-turbo")
    else:
        decoder_dir = os.path.join(model_path, "decoder")
    vae_dir = os.path.join(model_path, "vae")

    # ---------- Stage 1: SigVQ  → semantic features ----------
    extractor = SigVQ(vocab_size=16384, inner_dim=4096).to(device, dtype=dtype)
    extractor.load_state_dict(torch.load(sigvq_path, map_location=device, weights_only=True))
    extractor.eval()

    th = h * 16 * resolution_multiplier
    tw = w * 16 * resolution_multiplier
    tok = torch.tensor(token_ids).view(1, 1, h, w).float().to(device)
    up = F.interpolate(tok, scale_factor=2, mode="nearest").long().view(1, -1)
    cap_pos = [extractor(up).squeeze(0)]
    cap_neg = [torch.zeros_like(cap_pos[0])]

    # SigVQ is no longer needed — release immediately
    del extractor
    gc.collect()
    empty_cache()

    # ---------- Stage 2: Diffusion ODE sampling ----------
    config_path = os.path.join(decoder_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["axes_lens"] = [32768, 1024, 1024]
    cfg["cap_feat_dim"] = 4096

    # Build model on meta device, load weights directly to GPU, then tie —
    # avoids the ~12 GB peak from holding both random init + loaded weights.
    with torch.device("meta"):
        diff_model = ZImageTransformer2DModel(**cfg)
    ckpt = os.path.join(decoder_dir, "model.safetensors")
    diff_model.load_state_dict(load_file(ckpt, device=str(device)), assign=True)
    diff_model = diff_model.to(dtype=dtype).eval()

    z = torch.randn([1, 16, 1, 2 * (th // 16), 2 * (tw // 16)], device=device)
    model_fn = _create_decoder_model_fn(
        diff_model,
        cap_pos,
        cap_neg,
        cfg_scale=0.0 if decode_mode == "decoder-turbo" else 1.0,
        patch_size=cfg.get("all_patch_size", (2,))[0],
        f_patch_size=cfg.get("all_f_patch_size", (1,))[0],
        dtype=dtype,
    )

    sampler = Sampler(create_transport("Linear", "velocity", None))
    sample_fn = sampler.sample_ode(
        sampling_method="euler",
        num_steps=num_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        time_shifting_factor=6,
        stochast_ratio=1.0 if decode_mode == "decoder-turbo" else 0.0,
    )

    pbar = tqdm(total=num_steps, desc="Decoding", leave=False)

    def wrapped(x, t, _fn=model_fn, **kw):
        pbar.update(1)
        return _fn(x, t, **kw)

    samples = sample_fn(z, wrapped)[-1].squeeze(2)
    pbar.close()

    # Diffusion model is done — release before loading VAE
    del diff_model, cap_pos, cap_neg, model_fn
    gc.collect()
    empty_cache()

    # ---------- Stage 3: VAE decode ----------
    vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype).to(device).eval()

    s = samples.to(dtype)
    s = (s / vae.config.scaling_factor) + vae.config.shift_factor
    px = ((vae.decode(s, return_dict=False)[0] + 1) / 2).clamp_(0, 1)

    del vae
    gc.collect()
    empty_cache()

    return to_pil_image(px[0].float())
