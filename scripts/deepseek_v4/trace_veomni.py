"""Trace VeOmni's DeepSeek-V4 TileLang + fused-Triton execution path.

The input token tensor is read from the trace produced by
``trace_official.py`` so both implementations receive bit-identical inputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from veomni.arguments.arguments_types import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling_module
from veomni.ops.kernels.deepseek_v4 import linear_bf16_fp32
from veomni.utils import helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logprob-chunk-size", type=int, default=64)
    parser.add_argument(
        "--detail-layers",
        type=int,
        nargs="*",
        default=(),
        help="Store full intermediate tensors for the selected layer IDs",
    )
    parser.add_argument(
        "--force-official-attention-inputs",
        action="store_true",
        help="Replace selected layers' normalized attention inputs with tensors from --official-trace",
    )
    parser.add_argument(
        "--force-official-mlp-inputs",
        action="store_true",
        help="Replace selected layers' normalized MLP inputs with tensors from --official-trace",
    )
    parser.add_argument("--moe-backend", choices=("eager", "fused_triton", "fused_quack"), default="fused_triton")
    parser.add_argument("--indexer-implementation", choices=("eager", "tilelang"), default="tilelang")
    parser.add_argument("--attention-implementation", choices=("eager", "tilelang"), default="tilelang")
    parser.add_argument("--mhc-implementation", choices=("eager", "tilelang"), default="tilelang")
    parser.add_argument("--eager-moe-output", type=Path)
    parser.add_argument("--eager-dsa-output", type=Path)
    parser.add_argument("--eager-mhc-output", type=Path)
    parser.add_argument("--same-input-indexer-reference", action="store_true")
    parser.add_argument(
        "--reference-moe-topk",
        action="store_true",
        help="Replay official MoE expert IDs while retaining VeOmni router weights and expert execution",
    )
    parser.add_argument(
        "--reference-moe-router-weights",
        action="store_true",
        help="Replay official MoE router weights during a top-k reference pass",
    )
    parser.add_argument(
        "--official-moe-topk-output",
        type=Path,
        help="Write a second pass that replays official MoE expert IDs after the primary pass",
    )
    parser.add_argument(
        "--fp8-activation-qat",
        action="store_true",
        help="Enable the Miles-style DeepSeek-V4 FP8 activation QAT forward path",
    )
    parser.add_argument(
        "--reference-compressor-fp32",
        action="store_true",
        help="Match the official compressor's BF16-input/weight GEMMs with FP32 outputs and pooling",
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
        dsa_indexer_implementation=args.indexer_implementation,
        dsa_attention_implementation=args.attention_implementation,
        mhc_implementation=args.mhc_implementation,
        deepseek_v4_fp8_activation_qat=args.fp8_activation_qat,
    )


def main() -> None:
    args = parse_args()
    if args.reference_moe_topk and args.official_moe_topk_output is not None:
        raise ValueError("--reference-moe-topk and --official-moe-topk-output are mutually exclusive")
    if args.reference_moe_router_weights and not (args.reference_moe_topk or args.official_moe_topk_output):
        raise ValueError("--reference-moe-router-weights requires --reference-moe-topk or --official-moe-topk-output")
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

    if args.reference_compressor_fp32:
        projection_count = 0
        for layer in model.model.layers:
            compressor = getattr(layer.self_attn, "compressor", None)
            components = (compressor, getattr(compressor, "indexer", None))
            for component in components:
                if component is None:
                    continue
                for projection_name in ("kv_proj", "gate_proj"):
                    projection = getattr(component, projection_name, None)
                    if projection is None:
                        continue

                    def fp32_projection_hook(module, inputs, _output):
                        weight = module.weight.to_local() if hasattr(module.weight, "to_local") else module.weight
                        return linear_bf16_fp32(inputs[0], weight)

                    projection.register_forward_hook(fp32_projection_hook)
                    projection_count += 1

                kv_norm = getattr(component, "kv_norm", None)
                if kv_norm is not None:

                    def compressor_norm_pre_hook(_module, inputs):
                        return (inputs[0].to(torch.bfloat16), *inputs[1:])

                    kv_norm.register_forward_pre_hook(compressor_norm_pre_hook)
        if rank == 0:
            print(f"enabled FP32-output compressor projections on {projection_count} linear modules")

    official_trace = torch.load(args.official_trace, map_location="cpu", weights_only=True)
    if args.force_official_attention_inputs and not args.detail_layers:
        raise ValueError("--force-official-attention-inputs requires --detail-layers")
    input_ids = official_trace["input_ids"].to(device)

    def make_trace(implementation: str) -> dict[str, object]:
        return {
            "format_version": 1,
            "implementation": implementation,
            "input_ids": input_ids.cpu(),
            "moe_topk": {},
            "moe_weights": {},
            "indexer_topk": {},
            "indexer_eager_same_input": {},
            "attention_topk": {},
            "hidden": {},
            "details": {},
            "terminal": {},
        }

    implementation = (
        f"veomni_moe={args.moe_backend}_indexer={args.indexer_implementation}"
        f"_attention={args.attention_implementation}_mhc={args.mhc_implementation}"
    )
    if args.reference_moe_topk:
        implementation += "_official_moe_topk"
    if args.reference_moe_router_weights and args.reference_moe_topk:
        implementation += "_official_moe_router_weights"
    if args.fp8_activation_qat:
        implementation += "_fp8_activation_qat"
    if args.reference_compressor_fp32:
        implementation += "_compressor_fp32"
    trace = make_trace(implementation)

    layers = model.model.layers
    replay_moe_topk = {"enabled": args.reference_moe_topk}
    replay_moe_router_weights = {"enabled": args.reference_moe_topk and args.reference_moe_router_weights}
    if args.reference_moe_topk or args.official_moe_topk_output is not None:
        missing_layers = set(range(len(layers))) - set(official_trace["moe_topk"])
        if missing_layers:
            raise ValueError(f"Official trace is missing MoE top-k tensors for layers {sorted(missing_layers)}")
        if args.reference_moe_router_weights:
            missing_weight_layers = set(range(len(layers))) - set(official_trace.get("moe_weights", {}))
            if missing_weight_layers:
                raise ValueError(
                    f"Official trace is missing MoE router-weight tensors for layers {sorted(missing_weight_layers)}"
                )
        for layer_id, layer in enumerate(layers):
            official_indices = official_trace["moe_topk"][layer_id].to(device=device, dtype=torch.long)
            official_weights = None
            if args.reference_moe_router_weights:
                official_weights = official_trace["moe_weights"][layer_id].to(device=device)

            def replay_official_topk(
                module,
                _inputs,
                output,
                *,
                layer_id=layer_id,
                official_indices=official_indices,
                official_weights=official_weights,
            ):
                if not replay_moe_topk["enabled"]:
                    return output
                logits, _weights, veomni_indices = output
                if veomni_indices.shape != official_indices.shape:
                    raise ValueError(
                        f"Layer {layer_id} router shape mismatch: VeOmni {tuple(veomni_indices.shape)} "
                        f"vs official {tuple(official_indices.shape)}"
                    )
                if official_weights is not None and replay_moe_router_weights["enabled"]:
                    if _weights.shape != official_weights.shape:
                        raise ValueError(
                            f"Layer {layer_id} router-weight shape mismatch: VeOmni {tuple(_weights.shape)} "
                            f"vs official {tuple(official_weights.shape)}"
                        )
                    weights = official_weights.to(dtype=_weights.dtype)
                else:
                    scores = module.score_fn(logits)
                    weights = scores.gather(1, official_indices)
                    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
                    weights = weights * module.routed_scaling_factor
                return logits, weights, official_indices

            layer.mlp.gate.register_forward_hook(replay_official_topk)
    if rank == 0:

        def terminal_hook(name):
            def hook(_module, _inputs, output):
                trace["terminal"][name] = output.detach().cpu()

            return hook

        model.model.hc_head.register_forward_hook(terminal_hook("collapsed"))
        model.model.norm.register_forward_hook(terminal_hook("normalized"))
    if args.force_official_attention_inputs:
        for layer_id in args.detail_layers:
            official_attention_input = official_trace["details"][layer_id]["attn_input"].to(device)

            def force_attention_input_hook(
                _module,
                inputs,
                *,
                official_attention_input=official_attention_input,
            ):
                return (official_attention_input, *inputs[1:])

            layers[layer_id].self_attn.register_forward_pre_hook(force_attention_input_hook)
    if args.force_official_mlp_inputs:
        for layer_id in args.detail_layers:
            official_mlp_input = official_trace["details"][layer_id]["mlp_input"].to(device)

            def force_mlp_input_hook(
                _module,
                _inputs,
                _output,
                *,
                official_mlp_input=official_mlp_input,
            ):
                return official_mlp_input

            layers[layer_id].post_attention_layernorm.register_forward_hook(force_mlp_input_hook)

    if rank == 0:
        for layer_id, layer in enumerate(layers):

            def gate_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["moe_weights"][layer_id] = output[1].detach().cpu()
                trace["moe_topk"][layer_id] = output[2].detach().to(torch.int32).cpu()

            def layer_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["hidden"][layer_id] = hidden_fingerprint(output)

            layer.mlp.gate.register_forward_hook(gate_hook)
            layer.register_forward_hook(layer_hook)
            if layer_id in args.detail_layers:
                trace["details"][layer_id] = {}

                def save_tensor(name, tensor, *, layer_id=layer_id):
                    trace["details"].setdefault(layer_id, {})[name] = tensor.detach().cpu()

                def layer_detail_hook(_module, inputs, output, *, save_tensor=save_tensor):
                    save_tensor("layer_input", inputs[0])
                    save_tensor("layer_output", output)

                def attn_input_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("attn_input", output)

                def attn_output_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("attn_output", output[0])

                def mlp_input_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("mlp_input", output)

                def mlp_output_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("mlp_output", output)

                def gate_detail_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("router_weights", output[1])

                def tensor_output_hook(name, *, save_tensor=save_tensor, shard_last_dim=False):
                    def hook(_module, _inputs, output):
                        if shard_last_dim:
                            output = output[..., : output.shape[-1] // world_size]
                        save_tensor(name, output)

                    return hook

                def tensor_input_hook(name, *, save_tensor=save_tensor):
                    def hook(_module, inputs):
                        save_tensor(name, inputs[0])

                    return hook

                def o_a_proj_hook(_module, inputs, output, *, save_tensor=save_tensor):
                    if output.shape[-2] % world_size:
                        raise ValueError(f"o_a group count {output.shape[-2]} is not divisible by {world_size=}")
                    local_groups = output.shape[-2] // world_size
                    save_tensor("o_a_input", inputs[0][..., :local_groups, :])
                    save_tensor("o_a_proj", output[..., :local_groups, :])

                layer.register_forward_hook(layer_detail_hook)
                layer.input_layernorm.register_forward_hook(attn_input_hook)
                layer.self_attn.register_forward_hook(attn_output_hook)
                layer.post_attention_layernorm.register_forward_hook(mlp_input_hook)
                layer.mlp.register_forward_hook(mlp_output_hook)
                layer.mlp.gate.register_forward_hook(gate_detail_hook)
                layer.mlp.shared_experts.register_forward_hook(tensor_output_hook("shared_expert_output"))
                layer.mlp.shared_experts.gate_proj.register_forward_hook(tensor_output_hook("shared_gate_proj"))
                layer.mlp.shared_experts.up_proj.register_forward_hook(tensor_output_hook("shared_up_proj"))
                layer.mlp.shared_experts.down_proj.register_forward_pre_hook(tensor_input_hook("shared_down_input"))
                layer.mlp.shared_experts.down_proj.register_forward_hook(tensor_output_hook("shared_down_proj"))
                layer.self_attn.q_a_proj.register_forward_hook(tensor_output_hook("q_a_proj"))
                layer.self_attn.q_a_norm.register_forward_hook(tensor_output_hook("q_a_norm"))
                layer.self_attn.q_b_proj.register_forward_hook(tensor_output_hook("q_b_proj", shard_last_dim=True))
                layer.self_attn.kv_proj.register_forward_hook(tensor_output_hook("kv_proj"))
                layer.self_attn.kv_norm.register_forward_hook(tensor_output_hook("kv_norm"))
                layer.self_attn.o_a_proj.register_forward_hook(o_a_proj_hook)
                layer.self_attn.o_b_proj.register_forward_hook(tensor_output_hook("o_b_proj"))
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
        output = original_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale)
        if rank == 0 and active_layer["id"] in args.detail_layers:
            if output.shape[-2] % world_size:
                raise ValueError(f"attention head count {output.shape[-2]} is not divisible by {world_size=}")
            local_heads = output.shape[-2] // world_size
            trace["details"][active_layer["id"]]["sparse_attn_output"] = output[..., :local_heads, :].detach().cpu()
        return output

    modeling_module.sparse_attn_tilelang = traced_sparse_attn
    for layer_id, layer in enumerate(layers):

        def attention_pre_hook(_module, _inputs, *, layer_id=layer_id):
            active_layer["id"] = layer_id

        layer.self_attn.register_forward_pre_hook(attention_pre_hook)

    # chunk_logprobs_function applies the causal labels[..., 1:] shift.
    # Passing pre-shifted labels here would compare token t against t+2.
    labels = input_ids

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

    if args.official_moe_topk_output is not None:
        replay_moe_topk["enabled"] = True
        replay_moe_router_weights["enabled"] = args.reference_moe_router_weights
        replay_suffix = "_official_moe_routing" if args.reference_moe_router_weights else "_official_moe_topk"
        trace = make_trace(implementation + replay_suffix)
        run_pass(args.official_moe_topk_output)
        replay_moe_topk["enabled"] = False
        replay_moe_router_weights["enabled"] = False

    if args.eager_dsa_output is not None:
        modeling_module.veomni_dsa_indexer_implementation.bind(
            SimpleNamespace(dsa_indexer_implementation="eager")
        )
        modeling_module.veomni_dsa_attention_implementation.bind(
            SimpleNamespace(dsa_attention_implementation="eager")
        )
        trace = make_trace(
            implementation.replace(f"indexer={args.indexer_implementation}", "indexer=eager").replace(
                f"attention={args.attention_implementation}", "attention=eager"
            )
        )
        run_pass(args.eager_dsa_output)
        modeling_module.veomni_dsa_indexer_implementation.bind(
            SimpleNamespace(dsa_indexer_implementation=args.indexer_implementation)
        )
        modeling_module.veomni_dsa_attention_implementation.bind(
            SimpleNamespace(dsa_attention_implementation=args.attention_implementation)
        )

    if args.eager_mhc_output is not None:
        for slot in (modeling_module.veomni_mhc_pre, modeling_module.veomni_mhc_post, modeling_module.veomni_mhc_head):
            slot.bind("eager")
        trace = make_trace(implementation.replace(f"mhc={args.mhc_implementation}", "mhc=eager"))
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
