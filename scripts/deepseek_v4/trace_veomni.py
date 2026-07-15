"""Trace VeOmni's DeepSeek-V4 TileLang + fused-Triton execution path.

The input token tensor is read from the trace produced by
``trace_official.py`` so both implementations receive bit-identical inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from veomni.arguments.arguments_types import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import convert_deepseek_v4_checkpoint_key
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling_module
from veomni.ops.kernels.deepseek_v4 import (
    act_quant,
    fp4_act_quant,
    fp8_gemm,
    fp8_weight_quant_with_scale,
    linear_bf16_fp32,
)
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
    parser.add_argument(
        "--reference-indexer-fp4-quantization",
        action="store_true",
        help="Apply only the official Hadamard rotation and FP4 fake quantization to indexer Q/K",
    )
    parser.add_argument(
        "--reference-compressor-fp32",
        action="store_true",
        help="Match the official compressor's BF16-input/weight GEMMs with FP32 outputs and pooling",
    )
    quantized_linear_group = parser.add_mutually_exclusive_group()
    quantized_linear_group.add_argument(
        "--reference-quantized-linear-fp32",
        action="store_true",
        help="Apply official FP8 activation simulation and FP32 accumulation to dequantized linear weights",
    )
    quantized_linear_group.add_argument(
        "--reference-quantized-linear-fp8",
        action="store_true",
        help="Repack BF16-resident checkpoint linears to FP8 on demand and use official FP8 GEMMs",
    )
    parser.add_argument(
        "--reference-output-projection-partitions",
        type=int,
        default=1,
        help="Split attention o_b FP8 GEMMs along K before FP32 summation, matching official TP accumulation",
    )
    routed_expert_group = parser.add_mutually_exclusive_group()
    routed_expert_group.add_argument(
        "--reference-routed-expert-ondemand-fp4",
        action="store_true",
        help="Quantize BF16 routed-expert weights to FP4 on demand and use official FP8 activation GEMMs",
    )
    routed_expert_group.add_argument(
        "--reference-routed-expert-ondemand-fp8",
        action="store_true",
        help="Apply official in-place FP8 activation quantization before both BF16 routed-expert GEMMs",
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
    specialized_quantization = (
        args.reference_attention_kv_quantization
        or args.reference_indexer_fp4_quantization
        or args.reference_quantized_linear_fp32
        or args.reference_quantized_linear_fp8
        or args.reference_routed_expert_ondemand_fp4
        or args.reference_routed_expert_ondemand_fp8
    )
    if args.reference_quantization and specialized_quantization:
        raise ValueError("--reference-quantization cannot be combined with specialized quantization modes")
    if args.reference_output_projection_partitions < 1:
        raise ValueError("--reference-output-projection-partitions must be positive")
    if args.reference_output_projection_partitions != 1 and not args.reference_quantized_linear_fp8:
        raise ValueError("--reference-output-projection-partitions requires --reference-quantized-linear-fp8")
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

    if args.reference_quantization or args.reference_attention_kv_quantization:
        for layer in model.model.layers:
            layer.self_attn.use_fp8_kv_quantization = True
    if args.reference_routed_expert_ondemand_fp4:
        for layer in model.model.layers:
            layer.mlp.experts.use_ondemand_fp4 = True
            layer.mlp.experts.assume_replicated_ep_inputs = True
    if args.reference_routed_expert_ondemand_fp8:
        for layer in model.model.layers:
            layer.mlp.experts.use_ondemand_fp8 = True
            layer.mlp.experts.assume_replicated_ep_inputs = True

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
                        return linear_bf16_fp32(inputs[0], module.weight)

                    projection.register_forward_hook(fp32_projection_hook)
                    projection_count += 1

                kv_norm = getattr(component, "kv_norm", None)
                if kv_norm is not None:

                    def compressor_norm_pre_hook(_module, inputs):
                        return (inputs[0].to(torch.bfloat16), *inputs[1:])

                    kv_norm.register_forward_pre_hook(compressor_norm_pre_hook)
        if rank == 0:
            print(f"enabled FP32-output compressor projections on {projection_count} linear modules")

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
    if args.reference_quantized_linear_fp8:
        modules = dict(model.named_modules())
        expected_modules = {name for name in modules if quantized_linear_pattern.search(name)}
        index_path = args.checkpoint / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        scale_by_module: dict[str, tuple[str, str]] = {}
        duplicate_modules: set[str] = set()
        for raw_name, filename in weight_map.items():
            if raw_name.startswith("mtp."):
                continue
            converted_name = convert_deepseek_v4_checkpoint_key(raw_name)
            if converted_name.endswith(".scale"):
                module_name = converted_name.removesuffix(".scale")
            elif converted_name.endswith(".weight_scale_inv"):
                module_name = converted_name.removesuffix(".weight_scale_inv")
            else:
                continue
            if module_name not in expected_modules:
                continue
            if module_name in scale_by_module:
                duplicate_modules.add(module_name)
            scale_by_module[module_name] = (filename, raw_name)

        missing_modules = expected_modules - scale_by_module.keys()
        if missing_modules or duplicate_modules:
            problems = []
            if missing_modules:
                problems.append(f"missing scales for {sorted(missing_modules)}")
            if duplicate_modules:
                problems.append(f"duplicate scales for {sorted(duplicate_modules)}")
            raise RuntimeError("Incomplete FP8 checkpoint scale map: " + "; ".join(problems))

        scale_keys_by_file: dict[str, list[tuple[str, str]]] = {}
        for module_name, (filename, raw_name) in scale_by_module.items():
            scale_keys_by_file.setdefault(filename, []).append((raw_name, module_name))

        quantized_linears = 0
        for filename, entries in scale_keys_by_file.items():
            with safe_open(args.checkpoint / filename, framework="pt", device="cpu") as checkpoint_file:
                for raw_name, module_name in entries:
                    module = modules[module_name]
                    weight_scale = checkpoint_file.get_tensor(raw_name).to(
                        device=device,
                        dtype=torch.float8_e8m0fnu,
                    )

                    def quantized_linear_fp8_hook(
                        linear,
                        inputs,
                        _output,
                        *,
                        module_name=module_name,
                        weight_scale=weight_scale,
                    ):
                        weight = linear.weight.to_local() if hasattr(linear.weight, "to_local") else linear.weight
                        partitions = (
                            args.reference_output_projection_partitions
                            if module_name.endswith(".self_attn.o_b_proj")
                            else 1
                        )
                        if weight.shape[-1] % partitions:
                            raise ValueError(f"{module_name} input width is not divisible by {partitions=}")
                        if partitions > 1:
                            input_chunks = inputs[0].chunk(partitions, dim=-1)
                            weight_chunks = weight.chunk(partitions, dim=-1)
                            scale_chunks = weight_scale.chunk(partitions, dim=-1)
                            partials = []
                            for input_chunk, weight_chunk, scale_chunk in zip(
                                input_chunks, weight_chunks, scale_chunks, strict=True
                            ):
                                quantized_weight = fp8_weight_quant_with_scale(
                                    weight_chunk.contiguous(), scale_chunk.contiguous()
                                )
                                quantized_input, input_scale = act_quant(
                                    input_chunk.contiguous(),
                                    block_size=128,
                                    scale_fmt="ue8m0",
                                    scale_dtype=torch.float8_e8m0fnu,
                                )
                                partials.append(
                                    fp8_gemm(
                                        quantized_input,
                                        input_scale,
                                        quantized_weight,
                                        scale_chunk.contiguous(),
                                        torch.float8_e8m0fnu,
                                    )
                                )
                            return torch.stack([partial.float() for partial in partials]).sum(0).to(inputs[0].dtype)
                        quantized_weight = fp8_weight_quant_with_scale(weight.contiguous(), weight_scale)
                        quantized_input, input_scale = act_quant(
                            inputs[0].contiguous(),
                            block_size=128,
                            scale_fmt="ue8m0",
                            scale_dtype=torch.float8_e8m0fnu,
                        )
                        return fp8_gemm(
                            quantized_input,
                            input_scale,
                            quantized_weight,
                            weight_scale,
                            torch.float8_e8m0fnu,
                        )

                    module.register_forward_hook(quantized_linear_fp8_hook)
                    quantized_linears += 1
        if rank == 0:
            print(f"enabled on-demand checkpoint-scale FP8 execution on {quantized_linears} linear modules")

    if args.reference_quantized_linear_fp32:
        quantized_linears = 0
        for module_name, module in model.named_modules():
            if quantized_linear_pattern.search(module_name):

                def quantized_linear_fp32_hook(linear, inputs, output):
                    quantized_input = fake_quant_fp8(inputs[0])
                    if hasattr(linear, "n_groups"):
                        input_shape = quantized_input.shape[:-2]
                        hidden_dim = quantized_input.shape[-1]
                        weight = linear.weight.float().view(linear.n_groups, -1, hidden_dim).transpose(1, 2)
                        grouped_input = (
                            quantized_input.float().reshape(-1, linear.n_groups, hidden_dim).transpose(0, 1)
                        )
                        grouped_output = torch.bmm(grouped_input, weight).transpose(0, 1)
                        return grouped_output.reshape(*input_shape, linear.n_groups, -1).to(output.dtype)
                    return F.linear(quantized_input.float(), linear.weight.float()).to(output.dtype)

                module.register_forward_hook(quantized_linear_fp32_hook)
                quantized_linears += 1
        if rank == 0:
            print(f"enabled FP8-input/FP32-accumulation simulation on {quantized_linears} linear modules")

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
    if args.force_official_attention_inputs and not args.detail_layers:
        raise ValueError("--force-official-attention-inputs requires --detail-layers")
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
            "details": {},
        }

    implementation = (
        f"veomni_moe={args.moe_backend}_indexer={args.indexer_backend}_attention={args.attention_backend}"
        f"_mhc={args.mhc_backend}"
    )
    if args.reference_attention_kv_quantization:
        implementation += "_attention_kv_fp8_simulation"
    if args.reference_indexer_fp4_quantization:
        implementation += "_indexer_fp4_simulation"
    if args.reference_compressor_fp32:
        implementation += "_compressor_fp32"
    if args.reference_quantized_linear_fp32:
        implementation += "_quantized_linear_fp32"
    if args.reference_quantized_linear_fp8:
        implementation += "_quantized_linear_fp8"
        if args.reference_output_projection_partitions != 1:
            implementation += f"_output_projection_partitions_{args.reference_output_projection_partitions}"
    if args.reference_routed_expert_ondemand_fp4:
        implementation += "_routed_expert_ondemand_fp4"
    if args.reference_routed_expert_ondemand_fp8:
        implementation += "_routed_expert_ondemand_fp8"
    trace = make_trace(implementation)

    layers = model.model.layers
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

    if rank == 0:
        for layer_id, layer in enumerate(layers):

            def gate_hook(_module, _inputs, output, *, layer_id=layer_id):
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
        if args.reference_quantization or args.reference_indexer_fp4_quantization:
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
