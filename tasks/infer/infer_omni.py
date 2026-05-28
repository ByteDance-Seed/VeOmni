"""SeedOmni V2 native inference — one launcher YAML, one prompt, one optional image.

Usage
-----
::

    # Text → image (no image arg means infer_gen scenario)
    python tasks/infer/infer_omni.py \\
        --yaml configs/seed_omni/janus_1.3b/veomni_janus.yaml \\
        --prompt "A close-up of Sydney Opera House at night."

    # Image + text → text  (image arg implies infer_und scenario)
    python tasks/infer/infer_omni.py \\
        --yaml configs/seed_omni/janus_1.3b/veomni_janus.yaml \\
        --prompt "Describe this image." \\
        --image path/or/url.png

Defaults
--------
The scenario (``omni_infer_type``) is **inferred** from whether an image
is supplied; the launcher YAML's existing ``omni_infer_type`` is
overridden so the user never has to set it.  Outputs land under
``./output/`` next to the script's working directory: ``reply.txt`` for
the assistant text and ``generated_image_<i>.png`` for each VAE-decoded
image.
"""

from __future__ import annotations

import argparse
import os

import requests
import torch
from PIL import Image

from veomni.models.seed_omni.configuration_seed_omni import load_launcher_model_section
from veomni.trainer.omni_inferencer import OmniInferencer


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeedOmni V2 inference — pass a launcher YAML, a prompt, "
        "and (optionally) an image path. Image present → understanding (I2T); "
        "image absent → text-to-image (T2I).",
    )
    p.add_argument(
        "--yaml",
        required=True,
        help="Launcher YAML, e.g. configs/seed_omni/janus_1.3b/veomni_janus.yaml",
    )
    p.add_argument("--prompt", required=True, help="User text prompt.")
    p.add_argument(
        "--image",
        default=None,
        help="Optional path or http(s) URL to an image. Omit for text-to-image generation.",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help=(
            "Override the launcher YAML's `model.model_path` (the split-checkpoint root "
            "produced by ``scripts/multimodal/convert_model/split_janus.py``).  Handy when "
            "the YAML's default points at an in-repo placeholder and your split lives "
            "elsewhere, e.g. `/tmp/janus_1.3b_split`."
        ),
    )
    p.add_argument(
        "--output_dir",
        default="output",
        help="Directory for reply.txt + generated_image_*.png (created if missing).",
    )
    p.add_argument("--max_new_tokens", type=int, default=1024, help="FSM iteration cap.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help=(
            "Classifier-free guidance weight for text-to-image generation (Janus default = 5.0). "
            "Ignored in I2T scenarios and when <= 1.0. Same shape as `--guidance_scale` on the HF "
            "baseline `scripts/multimodal/infer/janus_hf_infer_gen.py`."
        ),
    )
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=1.0, help="Nucleus-sampling top-p threshold.")
    p.add_argument(
        "--trace",
        action="store_true",
        help="Dump FSM step / transition log to <output_dir>/trace.txt (debugging aid).",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_image(path: str) -> Image.Image:
    if path.startswith(("http://", "https://")):
        return Image.open(requests.get(path, stream=True).raw).convert("RGB")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


def _save_image(tensor: torch.Tensor, out_path: str) -> None:
    """Save a ``(1, H, W, 3)`` or ``(H, W, 3)`` tensor in ``[-1, 1]`` as PNG."""
    img = tensor.detach().to(dtype=torch.float32, device="cpu")
    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
    if img.dim() != 3 or img.size(-1) != 3:
        raise ValueError(f"Cannot save image with shape {tuple(img.shape)} (expected (H, W, 3)).")
    arr = ((img.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).numpy()
    Image.fromarray(arr).save(out_path)


def _extract_reply(ctx: dict) -> str:
    finalize = ctx.get("finalize") or {}
    for payload in finalize.values():
        if isinstance(payload, dict) and "text" in payload:
            return payload["text"]
    return ""


def _select_scenario(has_image: bool, infer_map: dict) -> str:
    """Pick the inference scenario based on whether an image was provided.

    With an image we want image-understanding (``infer_und``); without
    one we want text-to-image (``infer_gen``).  If the launcher YAML
    omits one of those, fall back to whatever scenarios it does declare
    so the script still does *something* useful.
    """
    preferred = "infer_und" if has_image else "infer_gen"
    if preferred in infer_map:
        return preferred
    if infer_map:
        return next(iter(infer_map))
    raise ValueError(
        "Launcher YAML declares no `model.omni_infer_yaml_path` entries — cannot pick an inference scenario."
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    launcher_path = os.path.realpath(args.yaml)
    if not os.path.isfile(launcher_path):
        raise FileNotFoundError(f"Launcher YAML not found: {launcher_path}")

    model_section = load_launcher_model_section(launcher_path)
    infer_map = model_section.get("omni_infer_yaml_path") or {}
    scenario = _select_scenario(has_image=bool(args.image), infer_map=infer_map)
    print(f"[infer_omni] launcher        = {launcher_path}")
    print(f"[infer_omni] model_path      = {model_section.get('model_path')}")
    print(f"[infer_omni] inference type  = {scenario} ({'I2T' if args.image else 'T2I'})")

    os.makedirs(args.output_dir, exist_ok=True)

    inferencer = OmniInferencer.from_launcher(
        launcher_path,
        infer_type=scenario,
        model_path=args.model_path,
        seed=args.seed,
    )

    images: list[Image.Image] = [_load_image(args.image)] if args.image else []
    force_image_gen = not args.image  # T2I scenarios should steer to image-VQ immediately

    # CFG only kicks in for T2I — for I2T we squelch it back to 1.0 so the
    # janus_text_encoder skips building the uncond branch entirely (its
    # construction rule isn't well-defined for prompts containing
    # image_und parts; see `_maybe_build_cfg_uncond_embeds`).
    guidance_scale = float(args.guidance_scale) if force_image_gen else 1.0
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "guidance_scale": guidance_scale,
    }

    trace_buf: list[str] = [] if args.trace else None
    ctx = inferencer.generate(
        prompt=args.prompt,
        images=images,
        force_image_gen=force_image_gen,
        generation_kwargs=generation_kwargs,
        max_new_tokens=args.max_new_tokens,
        trace=trace_buf,
    )

    reply = _extract_reply(ctx)
    reply_path = os.path.join(args.output_dir, "reply.txt")
    # encoding="utf-8" is load-bearing — Janus is multilingual so reply text
    # may carry CJK / emoji.  Default-locale opens crash on `LANG=C` images
    # (the common slim-base case).
    with open(reply_path, "w", encoding="utf-8") as f:
        f.write(reply + ("\n" if reply and not reply.endswith("\n") else ""))
    print(f"[infer_omni] reply ({len(reply)} chars) → {reply_path}")
    if reply:
        print(f"--- reply ---\n{reply}\n-------------")

    images_out: list[torch.Tensor] = list(ctx.get("generated_images_collected") or [])
    for idx, img_tensor in enumerate(images_out):
        out_path = os.path.join(args.output_dir, f"generated_image_{idx}.png")
        _save_image(img_tensor, out_path)
        print(f"[infer_omni] image #{idx} → {out_path}")

    if trace_buf is not None:
        trace_path = os.path.join(args.output_dir, "trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace_buf) + "\n")
        print(f"[infer_omni] FSM trace ({len(trace_buf)} lines) → {trace_path}")

    if not reply and not images_out:
        print("[infer_omni] WARNING — FSM produced no reply and no images.")


if __name__ == "__main__":
    main()
