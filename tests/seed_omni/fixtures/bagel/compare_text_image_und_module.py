"""Compare V2 BAGEL text+image understanding modules against an official fixture."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.compare_text_only_graph import (
    _cache_to_cpu,
    _compare_cache,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
    _v2_tolerance,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT, NaiveCache
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder


TEXT_IMAGE_CASE_ID = "text_image_understanding_one_step_logits"


def assert_text_image_fixture_schema(fixture: dict[str, Any]) -> None:
    metadata = fixture["metadata"]
    if metadata["case_id"] != TEXT_IMAGE_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in (
        "raw_input",
        "rng_state",
        "tokenizer",
        "prepared",
        "cache_after_image",
        "cache_after_prompt",
        "one_step",
    ):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

    image_fields = fixture["prepared"]["image"]
    for name in (
        "packed_text_ids",
        "packed_text_indexes",
        "vit_token_seqlens",
        "packed_vit_tokens",
        "packed_vit_position_ids",
        "packed_vit_token_indexes",
        "packed_position_ids",
        "packed_seqlens",
        "packed_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in image_fields:
            raise AssertionError(f"Fixture image section missing tensor: {name}")

    image_embeds = fixture["prepared"]["image_embeds"]
    for name in ("image_embeds", "cu_seqlens", "max_seqlen"):
        if name not in image_embeds:
            raise AssertionError(f"Fixture image_embeds section missing field: {name}")

    prompt_fields = fixture["prepared"]["prompt"]
    for name in (
        "packed_text_ids",
        "packed_text_position_ids",
        "text_token_lens",
        "packed_text_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in prompt_fields:
            raise AssertionError(f"Fixture prompt section missing tensor: {name}")

    for name in ("query_lens", "packed_query_indexes", "packed_key_value_indexes_for_step"):
        if name not in fixture["prepared"]:
            raise AssertionError(f"Fixture prepared section missing tensor: {name}")


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
    qwen2_mot.to(device=device).eval()
    return text_encoder, siglip_navit, qwen2_mot


def _embed_text(text_encoder: BagelTextEncoder, input_ids: torch.Tensor) -> torch.Tensor:
    return text_encoder.embed_tokens(input_ids)


@torch.no_grad()
def compare_text_image_und(
    fixture_path: Path,
    model_root: Path,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    assert_text_image_fixture_schema(fixture)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = _v2_tolerance(fixture)

    text_encoder, siglip_navit, qwen2_mot = _load_modules(model_root, device=torch_device, dtype=torch_dtype)
    image_fields = _to_device(fixture["prepared"]["image"], torch_device)
    prompt_fields = _to_device(fixture["prepared"]["prompt"], torch_device)
    start_fields = _to_device(fixture["prepared"]["start"], torch_device)

    image_output = siglip_navit(
        packed_pixel_values=image_fields["packed_vit_tokens"],
        packed_flattened_position_ids=image_fields["packed_vit_position_ids"],
        cu_seqlens=_to_device(fixture["prepared"]["image_embeds"]["cu_seqlens"], torch_device),
        max_seqlen=int(fixture["prepared"]["image_embeds"]["max_seqlen"]),
    )
    image_embeds = image_output["image_embeds"]
    image_embed_metrics = _tensor_metrics(
        image_embeds.detach().cpu(), fixture["prepared"]["image_embeds"]["image_embeds"]
    )
    image_embed_metrics["passes"] = _passes(image_embed_metrics, tolerance)

    past_key_values = NaiveCache(qwen2_mot.config.num_hidden_layers)
    packed_image_text_embedding = _embed_text(text_encoder, image_fields["packed_text_ids"])
    image_sequence = packed_image_text_embedding.new_zeros(
        (int(image_fields["packed_seqlens"].sum().item()), qwen2_mot.config.hidden_size)
    )
    image_sequence[image_fields["packed_text_indexes"]] = packed_image_text_embedding
    image_sequence[image_fields["packed_vit_token_indexes"]] = image_embeds.to(dtype=image_sequence.dtype)
    image_output = qwen2_mot(
        packed_query_sequence=image_sequence,
        query_lens=image_fields["packed_seqlens"],
        packed_query_position_ids=image_fields["packed_position_ids"],
        packed_query_indexes=image_fields["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=image_fields["key_values_lens"],
        packed_key_value_indexes=image_fields["packed_key_value_indexes"],
        update_past_key_values=True,
        is_causal=False,
        mode="und",
    )
    image_cache_metrics = _compare_cache(
        _cache_to_cpu(image_output["past_key_values"]), fixture["cache_after_image"], tolerance
    )

    prompt_embedding = _embed_text(text_encoder, prompt_fields["packed_text_ids"])
    prompt_output = qwen2_mot(
        packed_query_sequence=prompt_embedding,
        query_lens=prompt_fields["text_token_lens"],
        packed_query_position_ids=prompt_fields["packed_text_position_ids"],
        packed_query_indexes=prompt_fields["packed_text_indexes"],
        past_key_values=image_output["past_key_values"],
        key_values_lens=prompt_fields["key_values_lens"],
        packed_key_value_indexes=prompt_fields["packed_key_value_indexes"],
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    )
    prompt_cache_metrics = _compare_cache(
        _cache_to_cpu(prompt_output["past_key_values"]), fixture["cache_after_prompt"], tolerance
    )

    start_embedding = _embed_text(text_encoder, start_fields["packed_start_tokens"])
    one_step_output = qwen2_mot(
        packed_query_sequence=start_embedding,
        query_lens=_to_device(fixture["prepared"]["query_lens"], torch_device),
        packed_query_position_ids=start_fields["packed_query_position_ids"],
        packed_query_indexes=_to_device(fixture["prepared"]["packed_query_indexes"], torch_device),
        past_key_values=prompt_output["past_key_values"],
        key_values_lens=start_fields["key_values_lens"],
        packed_key_value_indexes=_to_device(fixture["prepared"]["packed_key_value_indexes_for_step"], torch_device),
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    )
    logits = text_encoder.lm_head(one_step_output["hidden_states"])
    hidden_metrics = _tensor_metrics(
        one_step_output["hidden_states"].detach().cpu(), fixture["one_step"]["hidden_state"]
    )
    logits_metrics = _tensor_metrics(logits.detach().cpu(), fixture["one_step"]["logits"])
    hidden_metrics["passes"] = _passes(hidden_metrics, tolerance)
    logits_metrics["passes"] = _passes(logits_metrics, tolerance)
    cache_step_metrics = _compare_cache(
        _cache_to_cpu(one_step_output["past_key_values"]), fixture["one_step"]["cache_after_step"], tolerance
    )

    greedy_token = torch.argmax(logits.detach().cpu(), dim=-1)
    greedy_match = torch.equal(greedy_token, fixture["one_step"]["greedy_token"])

    all_pass = bool(
        image_embed_metrics["passes"]
        and image_cache_metrics["passes"]
        and prompt_cache_metrics["passes"]
        and hidden_metrics["passes"]
        and logits_metrics["passes"]
        and cache_step_metrics["passes"]
        and greedy_match
    )
    return {
        "fixture": str(fixture_path),
        "model_root": str(model_root),
        "device": str(torch_device),
        "dtype": dtype,
        "tolerance": tolerance,
        "image_embeds": image_embed_metrics,
        "cache_after_image": image_cache_metrics,
        "cache_after_prompt": prompt_cache_metrics,
        "one_step_hidden_state": hidden_metrics,
        "one_step_logits": logits_metrics,
        "one_step_greedy": {
            "expected": fixture["one_step"]["greedy_token"].tolist(),
            "actual": greedy_token.tolist(),
            "passes": greedy_match,
        },
        "cache_after_step": cache_step_metrics,
        "all_pass": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_text_image_und(args.fixture, args.model_root, device=args.device, dtype=args.dtype)
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
