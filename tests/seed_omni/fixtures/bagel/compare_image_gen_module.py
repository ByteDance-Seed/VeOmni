"""Compare V2 BAGEL image-generation modules against an official velocity fixture."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.compare_text_only_graph import (  # noqa: E402
    _cache_to_cpu,
    _compare_cache,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
    _v2_tolerance,
)
from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector  # noqa: E402
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT, NaiveCache  # noqa: E402
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder  # noqa: E402


IMAGE_GEN_CASE_ID = "image_generation_one_step_velocity"


def assert_image_gen_fixture_schema(fixture: dict[str, Any]) -> None:
    metadata = fixture["metadata"]
    if metadata["case_id"] != IMAGE_GEN_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in ("raw_input", "rng_state", "tokenizer", "prepared", "cache_after_prompt", "one_step"):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

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

    latent_fields = fixture["prepared"]["latent"]
    for name in (
        "packed_text_ids",
        "packed_text_indexes",
        "packed_init_noises",
        "packed_vae_position_ids",
        "packed_vae_token_indexes",
        "packed_seqlens",
        "packed_position_ids",
        "key_values_lens",
        "packed_indexes",
        "packed_key_value_indexes",
    ):
        if name not in latent_fields:
            raise AssertionError(f"Fixture latent section missing tensor: {name}")

    timestep_fields = fixture["prepared"]["timesteps"]
    for name in ("timesteps", "dts", "timestep", "dt"):
        if name not in timestep_fields:
            raise AssertionError(f"Fixture timesteps section missing tensor: {name}")

    one_step = fixture["one_step"]
    for name in ("x_t0", "packed_sequence", "latent_embeds", "hidden_state", "velocity", "x_t1"):
        if name not in one_step:
            raise AssertionError(f"Fixture one_step section missing tensor: {name}")


def _load_modules(
    model_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BagelTextEncoder, BagelFlowConnector, BagelQwen2MoT]:
    text_encoder = BagelTextEncoder.from_pretrained(model_root / "bagel_text_encoder", torch_dtype=dtype)
    flow_connector = BagelFlowConnector.from_pretrained(model_root / "bagel_flow_connector", torch_dtype=dtype)
    qwen2_mot = BagelQwen2MoT.from_pretrained(model_root / "bagel_qwen2_mot", torch_dtype=dtype)
    text_encoder.to(device=device, dtype=dtype).eval()
    flow_connector.to(device=device, dtype=dtype).eval()
    # Official Bagel keeps Qwen RoPE frequency buffers in fp32.
    qwen2_mot.to(device=device).eval()
    return text_encoder, flow_connector, qwen2_mot


def _embed_text(text_encoder: BagelTextEncoder, input_ids: torch.Tensor) -> torch.Tensor:
    return text_encoder.embed_tokens(input_ids)


@torch.no_grad()
def compare_image_gen_module(
    fixture_path: Path,
    model_root: Path,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    assert_image_gen_fixture_schema(fixture)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = _v2_tolerance(fixture)

    text_encoder, flow_connector, qwen2_mot = _load_modules(model_root, device=torch_device, dtype=torch_dtype)
    prompt_fields = _to_device(fixture["prepared"]["prompt"], torch_device)
    latent_fields = _to_device(fixture["prepared"]["latent"], torch_device)
    timestep_fields = _to_device(fixture["prepared"]["timesteps"], torch_device)
    one_step = _to_device(fixture["one_step"], torch_device)

    past_key_values = NaiveCache(qwen2_mot.config.num_hidden_layers)
    prompt_embedding = _embed_text(text_encoder, prompt_fields["packed_text_ids"])
    prompt_output = qwen2_mot(
        packed_query_sequence=prompt_embedding,
        query_lens=prompt_fields["text_token_lens"],
        packed_query_position_ids=prompt_fields["packed_text_position_ids"],
        packed_query_indexes=prompt_fields["packed_text_indexes"],
        past_key_values=past_key_values,
        key_values_lens=prompt_fields["key_values_lens"],
        packed_key_value_indexes=prompt_fields["packed_key_value_indexes"],
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    )
    prompt_cache_metrics = _compare_cache(
        _cache_to_cpu(prompt_output["past_key_values"]), fixture["cache_after_prompt"], tolerance
    )

    text_embeds = _embed_text(text_encoder, latent_fields["packed_text_ids"])
    packed_sequence = text_embeds.new_zeros(
        (int(latent_fields["packed_seqlens"].sum().item()), qwen2_mot.config.hidden_size)
    )
    packed_sequence[latent_fields["packed_text_indexes"]] = text_embeds
    latent_output = flow_connector.embed_latent(
        latents=one_step["x_t0"],
        position_ids=latent_fields["packed_vae_position_ids"],
        timesteps=timestep_fields["timestep"],
    )
    latent_embeds = latent_output["latent_embeds"]
    packed_sequence[latent_fields["packed_vae_token_indexes"]] = latent_embeds.to(dtype=packed_sequence.dtype)

    latent_embed_metrics = _tensor_metrics(latent_embeds.detach().cpu(), fixture["one_step"]["latent_embeds"])
    latent_embed_metrics["passes"] = _passes(latent_embed_metrics, tolerance)
    packed_sequence_metrics = _tensor_metrics(packed_sequence.detach().cpu(), fixture["one_step"]["packed_sequence"])
    packed_sequence_metrics["passes"] = _passes(packed_sequence_metrics, tolerance)

    flow_output = qwen2_mot(
        packed_query_sequence=packed_sequence,
        query_lens=latent_fields["packed_seqlens"],
        packed_query_position_ids=latent_fields["packed_position_ids"],
        packed_query_indexes=latent_fields["packed_indexes"],
        past_key_values=prompt_output["past_key_values"],
        key_values_lens=latent_fields["key_values_lens"],
        packed_key_value_indexes=latent_fields["packed_key_value_indexes"],
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=latent_fields["packed_vae_token_indexes"],
        packed_text_indexes=latent_fields["packed_text_indexes"],
    )
    hidden_states = flow_output["hidden_states"]
    hidden_metrics = _tensor_metrics(hidden_states.detach().cpu(), fixture["one_step"]["hidden_state"])
    hidden_metrics["passes"] = _passes(hidden_metrics, tolerance)

    velocity_all = flow_connector.decode_velocity(hidden_states)["velocity"]
    velocity = velocity_all[latent_fields["packed_vae_token_indexes"]]
    velocity_metrics = _tensor_metrics(velocity.detach().cpu(), fixture["one_step"]["velocity"])
    velocity_metrics["passes"] = _passes(velocity_metrics, tolerance)

    x_t1 = one_step["x_t0"] - velocity.to(one_step["x_t0"].device) * timestep_fields["dt"][0]
    x_t1_metrics = _tensor_metrics(x_t1.detach().cpu(), fixture["one_step"]["x_t1"])
    x_t1_metrics["passes"] = _passes(x_t1_metrics, tolerance)

    all_pass = bool(
        prompt_cache_metrics["passes"]
        and latent_embed_metrics["passes"]
        and packed_sequence_metrics["passes"]
        and hidden_metrics["passes"]
        and velocity_metrics["passes"]
        and x_t1_metrics["passes"]
    )
    return {
        "fixture": str(fixture_path),
        "model_root": str(model_root),
        "device": str(torch_device),
        "dtype": dtype,
        "tolerance": tolerance,
        "cache_after_prompt": prompt_cache_metrics,
        "latent_embeds": latent_embed_metrics,
        "packed_sequence": packed_sequence_metrics,
        "hidden_state": hidden_metrics,
        "velocity": velocity_metrics,
        "x_t1": x_t1_metrics,
        "all_pass": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_image_gen_module(args.fixture, args.model_root, device=args.device, dtype=args.dtype)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
