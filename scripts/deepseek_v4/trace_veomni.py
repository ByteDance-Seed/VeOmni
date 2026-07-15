"""Trace VeOmni's DeepSeek-V4 TileLang + fused-Triton execution path.

The input token tensor is read from the trace produced by
``trace_official.py`` so both implementations receive bit-identical inputs.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from veomni.arguments.arguments_types import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling_module
from veomni.ops.kernels.deepseek_v4 import act_quant, fp4_act_quant
from veomni.utils import helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logprob-chunk-size", type=int, default=64)
    parser.add_argument("--moe-backend", choices=("eager", "fused_triton"), default="fused_triton")
    parser.add_argument("--indexer-backend", choices=("eager", "tilelang"), default="tilelang")
    parser.add_argument("--attention-backend", choices=("eager", "tilelang_sparse"), default="tilelang_sparse")
    parser.add_argument("--mhc-backend", choices=("eager", "tile_kernels"), default="tile_kernels")
    parser.add_argument("--eager-moe-output", type=Path)
    parser.add_argument("--eager-dsa-output", type=Path)
    parser.add_argument("--eager-mhc-output", type=Path)
    parser.add_argument("--same-input-indexer-reference", action="store_true")
    parser.add_argument("--reference-quantization", action="store_true")
    parser.add_argument(
        "--reference-attention-kv-quantization",
        action="store_true",
        help="Apply only the official FP8 E4M3/UE8M0 fake quantization to attention KV tensors",
    )
    return parser.parse_args()


def hidden_fingerprint(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = hidden_states.detach().float().flatten(start_dim=2)
    return {
        "mean": flat.mean(dim=-1).cpu(),
        "rms": flat.square().mean(dim=-1).sqrt().cpu(),
        "sample": flat[..., :8].cpu(),
    }


def make_ops_config(args: argparse.Namespace) -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation=args.moe_backend,
        cross_entropy_loss_implementation="chunk_loss",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        dsa_indexer_backend=args.indexer_backend,
        dsa_attention_backend=args.attention_backend,
        mhc_backend=args.mhc_backend,
    )


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    helper.enable_high_precision_for_bf16()
    torch.manual_seed(33377335)

    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        extra_parallel_names=("ep",),
        extra_parallel_sizes=(world_size,),
        extra_parallel_placement_innermost=(False,),
    )

    ops = make_ops_config(args)
    model = build_foundation_model(
        config_path=str(args.checkpoint),
        weights_path=str(args.checkpoint),
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=ops,
    )
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=str(args.checkpoint),
        enable_reshard_after_forward=True,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=False,
        basic_modules=list(set(getattr(model, "_no_split_modules", None) or [])),
        # DeepSeek-V4's raw FP4 checkpoint requires key and tensor conversion.
        # The EP streaming loader intentionally rejects that combination today.
        ep_sharded_stream_load=False,
    )
    model.eval()

    def fake_quant_fp8(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        quantized = x.detach().clone()
        act_quant(
            quantized,
            block_size=block_size,
            scale_fmt="ue8m0",
            scale_dtype=torch.float8_e8m0fnu,
            inplace=True,
        )
        return x + (quantized - x).detach()

    quantized_linear_pattern = re.compile(
        r"(?:\.self_attn\.(?:q_a_proj|q_b_proj|kv_proj|o_b_proj)"
        r"|\.self_attn\.compressor\.indexer\.q_b_proj"
        r"|\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))$"
    )
    if args.reference_quantization:
        quantized_linears = 0
        for module_name, module in model.named_modules():
            if quantized_linear_pattern.search(module_name):

                def quantized_linear_pre_hook(_module, inputs):
                    return (fake_quant_fp8(inputs[0]), *inputs[1:])

                module.register_forward_pre_hook(quantized_linear_pre_hook)
                quantized_linears += 1
        for layer in model.model.layers:

            def routed_expert_pre_hook(_module, inputs):
                return (fake_quant_fp8(inputs[0]), *inputs[1:])

            layer.mlp.experts.register_forward_pre_hook(routed_expert_pre_hook)
        if rank == 0:
            print(f"enabled reference activation quantization on {quantized_linears} linear modules")

    official_trace = torch.load(args.official_trace, map_location="cpu", weights_only=True)
    input_ids = official_trace["input_ids"].to(device)

    def make_trace(implementation: str) -> dict[str, object]:
        return {
            "format_version": 1,
            "implementation": implementation,
            "input_ids": input_ids.cpu(),
            "moe_topk": {},
            "indexer_topk": {},
            "indexer_eager_same_input": {},
            "attention_topk": {},
            "hidden": {},
        }

    implementation = (
        f"veomni_moe={args.moe_backend}_indexer={args.indexer_backend}_attention={args.attention_backend}"
        f"_mhc={args.mhc_backend}"
    )
    if args.reference_attention_kv_quantization:
        implementation += "_attention_kv_fp8_simulation"
    trace = make_trace(implementation)

    layers = model.model.layers
    if rank == 0:
        for layer_id, layer in enumerate(layers):

            def gate_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["moe_topk"][layer_id] = output[2].detach().to(torch.int32).cpu()

            def layer_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["hidden"][layer_id] = hidden_fingerprint(output)

            layer.mlp.gate.register_forward_hook(gate_hook)
            layer.register_forward_hook(layer_hook)
            indexer = getattr(getattr(layer.self_attn, "compressor", None), "indexer", None)
            if indexer is not None:

                def indexer_hook(_module, _inputs, output, *, layer_id=layer_id):
                    trace["indexer_topk"][layer_id] = output.detach().to(torch.int32).cpu()

                indexer.register_forward_hook(indexer_hook)

    active_layer = {"id": -1}
    original_indexer = modeling_module.v4_lighting_indexer

    def traced_indexer(
        q,
        k,
        weights,
        compress_ratio,
        topk,
        topk_indices=None,
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
    ):
        if args.reference_quantization:
            from fast_hadamard_transform import hadamard_transform

            q = hadamard_transform(q, scale=q.shape[-1] ** -0.5)
            k = hadamard_transform(k, scale=k.shape[-1] ** -0.5)
            q_quantized = q.detach().clone()
            k_quantized = k.detach().clone()
            fp4_act_quant(q_quantized, inplace=True)
            fp4_act_quant(k_quantized, inplace=True)
            q = q + (q_quantized - q).detach()
            k = k + (k_quantized - k).detach()
        result = original_indexer(
            q,
            k,
            weights,
            compress_ratio,
            topk,
            topk_indices,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
        if rank == 0 and args.same_input_indexer_reference and topk_indices is None:
            reference_chunks = []
            key_positions = torch.arange(k.shape[0], device=k.device)
            for start in range(0, q.shape[0], 64):
                end = min(q.shape[0], start + 64)
                scores = torch.einsum("sbhd,tbd->bsht", q[start:end].float(), k.float())
                scores = (scores.relu() * weights[start:end].permute(1, 0, 2).float().unsqueeze(-1)).sum(dim=2)
                if cu_seqlen_ks is None:
                    valid_start = torch.zeros(end - start, device=k.device, dtype=torch.long)
                    valid_end = (torch.arange(start, end, device=k.device) + 1) // compress_ratio
                else:
                    valid_start = cu_seqlen_ks[start:end].long()
                    valid_end = cu_seqlen_ke[start:end].long()
                valid = (key_positions >= valid_start[:, None]) & (key_positions < valid_end[:, None])
                values, indices = scores.masked_fill(~valid.unsqueeze(0), float("-inf")).topk(topk, dim=-1)
                reference_chunks.append(indices.masked_fill(values == -torch.inf, -1).to(torch.int32).cpu())
            trace["indexer_eager_same_input"][active_layer["id"]] = torch.cat(reference_chunks, dim=1)
        return result

    modeling_module.v4_lighting_indexer = traced_indexer
    original_sparse_attn = modeling_module.sparse_attn_tilelang

    def traced_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale=None):
        if rank == 0:
            trace["attention_topk"][active_layer["id"]] = topk_idxs.detach().to(torch.int32).cpu()
        if args.reference_quantization or args.reference_attention_kv_quantization:
            rope_dim = 64
            kv = torch.cat((fake_quant_fp8(kv[..., :-rope_dim], block_size=64), kv[..., -rope_dim:]), dim=-1)
        return original_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale)

    modeling_module.sparse_attn_tilelang = traced_sparse_attn
    for layer_id, layer in enumerate(layers):

        def attention_pre_hook(_module, _inputs, *, layer_id=layer_id):
            active_layer["id"] = layer_id

        layer.self_attn.register_forward_pre_hook(attention_pre_hook)

    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]

    def run_pass(output_path: Path) -> None:
        # FSDP2's unshard hooks preserve tensor version counters, which are
        # deliberately absent under inference_mode. no_grad is the supported
        # evaluation context for fully_shard models.
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels,
                use_cache=False,
                return_log_probs=True,
                chunk_size=args.logprob_chunk_size,
            )
        if outputs.log_probs is None:
            raise RuntimeError("VeOmni did not return per-token log-probabilities")
        if rank == 0:
            trace["logprobs"] = outputs.log_probs[:, :-1].detach().cpu()
            trace["entropy"] = None if outputs.entropy is None else outputs.entropy[:, :-1].detach().cpu()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trace, output_path)
            print(f"saved VeOmni trace to {output_path}")

    run_pass(args.output)

    if args.eager_dsa_output is not None:
        modeling_module.veomni_dsa_indexer_backend.bind(SimpleNamespace(dsa_indexer_backend="eager"))
        modeling_module.veomni_dsa_attention_backend.bind(SimpleNamespace(dsa_attention_backend="eager"))
        trace = make_trace(
            implementation.replace(f"indexer={args.indexer_backend}", "indexer=eager").replace(
                f"attention={args.attention_backend}", "attention=eager"
            )
        )
        run_pass(args.eager_dsa_output)
        modeling_module.veomni_dsa_indexer_backend.bind(SimpleNamespace(dsa_indexer_backend=args.indexer_backend))
        modeling_module.veomni_dsa_attention_backend.bind(
            SimpleNamespace(dsa_attention_backend=args.attention_backend)
        )

    if args.eager_mhc_output is not None:
        for slot in (modeling_module.veomni_mhc_pre, modeling_module.veomni_mhc_post, modeling_module.veomni_mhc_head):
            slot.bind("eager")
        trace = make_trace(implementation.replace(f"mhc={args.mhc_backend}", "mhc=eager"))
        run_pass(args.eager_mhc_output)

    if args.eager_moe_output is not None:
        # This script always enables EP when world_size > 1. The ordinary
        # eager loop expects all expert weights locally and cannot consume
        # global router IDs with an EP-sharded [E/world_size, ...] tensor.
        if world_size > 1:
            if rank == 0:
                print("skipping eager MoE pass: the eager expert loop is not EP-aware")
        else:
            modeling_module.veomni_moe_experts_forward.bind("eager")
            trace = make_trace(implementation.replace(f"moe={args.moe_backend}", "moe=eager"))
            run_pass(args.eager_moe_output)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
