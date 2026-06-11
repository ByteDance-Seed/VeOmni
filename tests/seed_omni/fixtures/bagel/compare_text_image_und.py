"""Compare V2 BAGEL text+image understanding graph outputs against an official fixture."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import adapt_text_image_und_fixture  # noqa: E402

from tests.seed_omni.fixtures.bagel.compare_text_image_und_module import (  # noqa: E402
    assert_text_image_fixture_schema,
)
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
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT  # noqa: E402
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit  # noqa: E402
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder  # noqa: E402


class _UnusedModule(nn.Module):
    pass


def _load_modules(
    model_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BagelTextEncoder, BagelSiglipNavit, BagelQwen2MoT]:
    text_encoder = BagelTextEncoder.from_pretrained(model_root / "bagel_text_encoder", torch_dtype=dtype)
    siglip_navit = BagelSiglipNavit.from_pretrained(model_root / "bagel_siglip_navit", torch_dtype=dtype)
    qwen2_mot = BagelQwen2MoT.from_pretrained(model_root / "bagel_qwen2_mot", torch_dtype=dtype)
    text_encoder.to(device=device, dtype=dtype).eval()
    siglip_navit.to(device=device, dtype=dtype).eval()
    # Official Bagel keeps Qwen RoPE frequency buffers in fp32.
    qwen2_mot.to(device=device).eval()
    return text_encoder, siglip_navit, qwen2_mot


def _load_graph_config(config_dir: Path) -> OmniConfig:
    train_config = yaml.safe_load((config_dir / "train.yaml").read_text(encoding="utf-8"))
    infer_config = yaml.safe_load((config_dir / "infer_und.yaml").read_text(encoding="utf-8"))
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
def compare_text_image_und_graph(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    assert_text_image_fixture_schema(fixture)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = _v2_tolerance(fixture)

    text_encoder, siglip_navit, qwen2_mot = _load_modules(model_root, device=torch_device, dtype=torch_dtype)
    config = _load_graph_config(config_dir)
    model = OmniModel(
        config,
        {
            "bagel_text_encoder": text_encoder,
            "bagel_siglip_navit": siglip_navit,
            "bagel_vae": _UnusedModule(),
            "bagel_flow_connector": _UnusedModule(),
            "bagel_qwen2_mot": qwen2_mot,
        },
    ).eval()

    conversation = _to_device(adapt_text_image_und_fixture(fixture), torch_device)
    trace: list[str] = []
    generation_kwargs = dict(config.generation_kwargs or {})
    generation_kwargs.update({"max_new_tokens": 1, "do_sample": False, "temperature": 1.0, "top_p": 1.0})
    ctx = model.generate({"conversation_list": conversation}, trace=trace, generation_kwargs=generation_kwargs)

    image_embed_metrics = _tensor_metrics(
        ctx["bagel_last_image_embeds"].detach().cpu(),
        fixture["prepared"]["image_embeds"]["image_embeds"],
    )
    hidden_metrics = _tensor_metrics(
        ctx["bagel_last_hidden_state"].detach().cpu(),
        fixture["one_step"]["hidden_state"],
    )
    logits_metrics = _tensor_metrics(
        ctx["bagel_last_logits"].detach().cpu(),
        fixture["one_step"]["logits"],
    )
    image_embed_metrics["passes"] = _passes(image_embed_metrics, tolerance)
    hidden_metrics["passes"] = _passes(hidden_metrics, tolerance)
    logits_metrics["passes"] = _passes(logits_metrics, tolerance)
    greedy_token = ctx["bagel_last_greedy_token"].detach().cpu()
    greedy_match = torch.equal(greedy_token, fixture["one_step"]["greedy_token"])
    cache_after_step = _compare_cache(
        _cache_to_cpu(ctx["past_key_values"]),
        fixture["one_step"]["cache_after_step"],
        tolerance,
    )

    all_pass = bool(
        image_embed_metrics["passes"]
        and hidden_metrics["passes"]
        and logits_metrics["passes"]
        and greedy_match
        and cache_after_step["passes"]
    )
    return {
        "case_id": fixture["metadata"]["case_id"],
        "dtype": dtype,
        "tolerance": tolerance,
        "trace": trace,
        "image_embeds": image_embed_metrics,
        "hidden_state": hidden_metrics,
        "logits": logits_metrics,
        "greedy_token": {
            "expected": fixture["one_step"]["greedy_token"].tolist(),
            "actual": greedy_token.tolist(),
            "passes": greedy_match,
        },
        "cache_after_step": cache_after_step,
        "all_pass": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/seed_omni/bagel_7b_mot"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_text_image_und_graph(
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
