"""Compare V2 BAGEL input-image VAE-context graph outputs against an official fixture."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.adapter import (  # noqa: E402
    adapt_image_edit_fixture,
    assert_image_edit_fixture_schema,
)
from tests.seed_omni.fixtures.bagel.compare_image_gen import _generated_image_report  # noqa: E402
from tests.seed_omni.fixtures.bagel.compare_text_only_graph import (  # noqa: E402
    _cache_to_cpu,
    _compare_cache,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
    _v2_tolerance,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig  # noqa: E402
from veomni.models.seed_omni.modeling_omni import OmniModel  # noqa: E402
from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector  # noqa: E402
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT  # noqa: E402
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit  # noqa: E402
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder  # noqa: E402
from veomni.models.seed_omni.modules.bagel.vae.modeling import BagelVAE  # noqa: E402


def _load_modules(
    model_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BagelTextEncoder, BagelSiglipNavit, BagelVAE, BagelFlowConnector, BagelQwen2MoT]:
    text_encoder = BagelTextEncoder.from_pretrained(model_root / "bagel_text_encoder", torch_dtype=dtype)
    siglip = BagelSiglipNavit.from_pretrained(model_root / "bagel_siglip_navit", torch_dtype=dtype)
    vae = BagelVAE.from_pretrained(model_root / "bagel_vae", torch_dtype=dtype)
    flow_connector = BagelFlowConnector.from_pretrained(model_root / "bagel_flow_connector", torch_dtype=dtype)
    qwen2_mot = BagelQwen2MoT.from_pretrained(model_root / "bagel_qwen2_mot", torch_dtype=dtype)
    text_encoder.to(device=device, dtype=dtype).eval()
    siglip.to(device=device, dtype=dtype).eval()
    vae.to(device=device, dtype=dtype).eval()
    flow_connector.to(device=device, dtype=dtype).eval()
    # Official Bagel keeps Qwen RoPE frequency buffers in fp32.
    qwen2_mot.to(device=device).eval()
    return text_encoder, siglip, vae, flow_connector, qwen2_mot


def _load_graph_config(config_dir: Path) -> OmniConfig:
    train_config = yaml.safe_load((config_dir / "train.yaml").read_text(encoding="utf-8"))
    infer_config = yaml.safe_load((config_dir / "infer_interleave.yaml").read_text(encoding="utf-8"))
    merged = {
        "modules": train_config["modules"],
        "nodes": train_config["nodes"],
        "edges": train_config["edges"],
        "training_graph": train_config["training_graph"],
        "generation_graph": infer_config["generation_graph"],
        "generation_kwargs": infer_config.get("generation_kwargs", {}),
    }
    return OmniConfig.from_dict(merged)


@torch.no_grad()
def compare_image_edit_graph(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    assert_image_edit_fixture_schema(fixture)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = _v2_tolerance(fixture)

    text_encoder, siglip, vae, flow_connector, qwen2_mot = _load_modules(
        model_root,
        device=torch_device,
        dtype=torch_dtype,
    )
    config = _load_graph_config(config_dir)
    model = OmniModel(
        config,
        {
            "bagel_text_encoder": text_encoder,
            "bagel_siglip_navit": siglip,
            "bagel_vae": vae,
            "bagel_flow_connector": flow_connector,
            "bagel_qwen2_mot": qwen2_mot,
        },
    ).eval()

    conversation = _to_device(adapt_image_edit_fixture(fixture), torch_device)
    trace: list[str] = []
    generation_kwargs = dict(config.generation_kwargs or {})
    generation_kwargs.update(
        {
            "enable_vae_context": True,
            "infer_mode": "gen",
            "max_flow_steps": 1,
            "max_new_tokens": 3,
        }
    )
    _restore_rng_state(fixture.get("rng_state_before_vae"), torch_device)
    ctx = model.generate({"conversation_list": conversation}, trace=trace, generation_kwargs=generation_kwargs)

    vae_context_item = next(
        item for item in ctx["conversation_list"] if item.meta.get("bagel_role") == "image_vae_context"
    )
    vae_latent_metrics = _tensor_metrics(
        vae_context_item.meta["vae_context_latents"].detach().cpu(),
        fixture["vae_context"]["packed_latents"],
    )
    vae_embed_metrics = _tensor_metrics(
        vae_context_item.meta["vae_context_latent_embeds"].detach().cpu(),
        fixture["vae_context"]["latent_embeds"],
    )
    vae_sequence_metrics = _tensor_metrics(
        vae_context_item.meta["vae_context_packed_sequence"].detach().cpu(),
        fixture["vae_context"]["packed_sequence"],
    )
    cache_after_prompt = _compare_cache(
        _cache_to_cpu(ctx["past_key_values"]), fixture["cache_after_prompt"], tolerance
    )
    latent_embed_metrics = _tensor_metrics(
        ctx["bagel_last_latent_embeds"].detach().cpu(),
        fixture["one_step"]["latent_embeds"],
    )
    packed_sequence_metrics = _tensor_metrics(
        ctx["bagel_last_packed_sequence"].detach().cpu(),
        fixture["one_step"]["packed_sequence"],
    )
    hidden_metrics = _tensor_metrics(
        ctx["bagel_last_hidden_state"].detach().cpu(),
        fixture["one_step"]["hidden_state"],
    )
    velocity_metrics = _tensor_metrics(
        ctx["bagel_last_velocity"].detach().cpu(),
        fixture["one_step"]["velocity"],
    )
    x_t1_metrics = _tensor_metrics(
        ctx["bagel_last_x_t"].detach().cpu(),
        fixture["one_step"]["x_t1"],
    )
    for metrics in (
        vae_latent_metrics,
        vae_embed_metrics,
        vae_sequence_metrics,
        latent_embed_metrics,
        packed_sequence_metrics,
        hidden_metrics,
        velocity_metrics,
        x_t1_metrics,
    ):
        metrics["passes"] = _passes(metrics, tolerance)

    all_pass = bool(
        vae_latent_metrics["passes"]
        and vae_embed_metrics["passes"]
        and vae_sequence_metrics["passes"]
        and cache_after_prompt["passes"]
        and latent_embed_metrics["passes"]
        and packed_sequence_metrics["passes"]
        and hidden_metrics["passes"]
        and velocity_metrics["passes"]
        and x_t1_metrics["passes"]
    )
    return {
        "case_id": fixture["metadata"]["case_id"],
        "dtype": dtype,
        "tolerance": tolerance,
        "trace": trace,
        "generated_images": _generated_image_report(model.generated),
        "vae_context_latents": vae_latent_metrics,
        "vae_context_latent_embeds": vae_embed_metrics,
        "vae_context_packed_sequence": vae_sequence_metrics,
        "cache_after_prompt": cache_after_prompt,
        "latent_embeds": latent_embed_metrics,
        "packed_sequence": packed_sequence_metrics,
        "hidden_state": hidden_metrics,
        "velocity": velocity_metrics,
        "x_t1": x_t1_metrics,
        "all_pass": all_pass,
    }


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    if torch.is_tensor(state.get("torch_cpu")):
        torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if device.type == "cuda" and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/seed_omni/bagel_7b_mot"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_image_edit_graph(
        args.fixture,
        args.model_root,
        config_dir=args.config_dir,
        device=args.device,
        dtype=args.dtype,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
