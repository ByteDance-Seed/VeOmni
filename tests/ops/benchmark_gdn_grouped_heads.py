import argparse
import importlib
import json
import statistics
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch

from veomni.ops.dispatch import OpSlot
from veomni.utils.device import get_device_type, get_gpu_compute_capability, get_torch_device, synchronize


def measure(run, warmup, repeats, iterations):
    for _ in range(warmup):
        run()
    synchronize()
    samples = {False: [], True: []}
    memory = {False: [], True: []}
    device = get_torch_device()
    for repeat in range(repeats):
        for grouped in [False, True] if repeat % 2 == 0 else [True, False]:
            device.reset_peak_memory_stats()
            start = device.Event(enable_timing=True)
            end = device.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                run(grouped)
            end.record()
            synchronize()
            samples[grouped].append(start.elapsed_time(end) / iterations)
            memory[grouped].append(device.max_memory_allocated())
    baseline = statistics.median(samples[False])
    candidate = statistics.median(samples[True])
    return {
        "baseline_ms": baseline,
        "grouped_ms": candidate,
        "speedup": baseline / candidate,
        "latency_reduction_percent": (1 - candidate / baseline) * 100,
        "baseline_samples_ms": samples[False],
        "grouped_samples_ms": samples[True],
        "baseline_peak_allocated_bytes": max(memory[False]),
        "grouped_peak_allocated_bytes": max(memory[True]),
    }


def kernel_runner(sequence, group_size, backward):
    slot = OpSlot("chunk_gated_delta_rule", "standard")
    slot.bind("flash_qla")
    device = get_device_type()
    key_heads, dimension = 4, 128
    value_heads = key_heads * group_size
    query = torch.randn(1, sequence, key_heads, dimension, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(1, sequence, value_heads, dimension, device=device, dtype=torch.bfloat16)
    gate = -torch.rand(1, sequence, value_heads, device=device, dtype=torch.float32) * 0.5
    beta = torch.rand(1, sequence, value_heads, device=device, dtype=torch.bfloat16)
    inputs = (query, key, value, gate, beta)
    for tensor in inputs:
        tensor.requires_grad_(backward)
    boundaries = torch.tensor([0, sequence // 4, sequence], device=device, dtype=torch.int32)
    output_gradient = torch.randn_like(value)

    def run(grouped=None):
        if grouped is None:
            run(False)
            return run(True)
        for tensor in inputs:
            tensor.grad = None
        kernel_query = query if grouped or group_size == 1 else query.repeat_interleave(group_size, dim=2)
        kernel_key = key if grouped or group_size == 1 else key.repeat_interleave(group_size, dim=2)
        with torch.set_grad_enabled(backward):
            output, _ = slot(
                kernel_query,
                kernel_key,
                value,
                g=gate,
                beta=beta,
                cu_seqlens=boundaries,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            if backward:
                output.backward(output_gradient)
                for tensor in inputs:
                    tensor.grad = None

    return run


def layer_runner(model_name, sequence, group_size, backward):
    modeling = importlib.import_module(
        f"veomni.models.transformers.{model_name}.generated.patched_modeling_{model_name}_gpu"
    )
    class_name = {
        "qwen3_5": "Qwen3_5GatedDeltaNet",
        "qwen3_5_moe": "Qwen3_5MoeGatedDeltaNet",
        "qwen4_exp": "Qwen4ExpTextGatedDeltaNet",
    }[model_name]
    modeling.veomni_causal_conv1d.bind("fla")
    modeling.veomni_chunk_gated_delta_rule.bind("flash_qla")
    modeling.veomni_gdn_implementation.bind(SimpleNamespace(chunk_gated_delta_rule_implementation="flash_qla"))
    if model_name != "qwen4_exp":
        modeling.veomni_rms_norm_gated.bind("eager")
    modeling.get_parallel_state = lambda: SimpleNamespace(ulysses_enabled=False)
    config = SimpleNamespace(
        hidden_size=512,
        linear_num_value_heads=4 * group_size,
        linear_num_key_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        output_gate_type=None,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    device = get_device_type()
    with torch.device(device):
        layer = getattr(modeling, class_name)(config, layer_idx=0)
    layer.to(dtype=torch.bfloat16)
    hidden = torch.randn(1, sequence, 512, device=device, dtype=torch.bfloat16, requires_grad=backward)
    output_gradient = torch.randn_like(hidden)
    boundaries = torch.tensor([0, sequence // 4, sequence], device=device, dtype=torch.int32)
    metadata = {"linear_attn_cu_seq_lens_q" if model_name == "qwen4_exp" else "cu_seq_lens_q": boundaries}

    def run(grouped=None):
        if grouped is None:
            run(False)
            return run(True)
        layer.gdn_grouped_heads = grouped
        layer.zero_grad(set_to_none=True)
        hidden.grad = None
        with torch.set_grad_enabled(backward):
            output = layer(hidden, **metadata)
            if backward:
                output.backward(output_gradient)
                layer.zero_grad(set_to_none=True)
                hidden.grad = None

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if get_device_type() != "cuda" or get_gpu_compute_capability() != 90:
        raise RuntimeError("This benchmark requires one Hopper SM90 CUDA GPU.")
    if get_torch_device().device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for the single-device benchmark.")
    if min(args.sequences) < 4 or min(args.group_sizes) < 1 or min(args.warmup, args.repeats, args.iterations) < 1:
        raise ValueError("Positive repetition counts, group sizes and sequences >= 4 are required.")
    torch.manual_seed(20261014)
    result = {
        "scope": "single_gdn_kernel_and_layer_not_full_model_training",
        "gpu": get_torch_device().get_device_name(0),
        "versions": {name: version(name) for name in ["torch", "transformers", "flash-qla", "tilelang", "fla-core"]},
        "protocol": vars(args) | {"output": str(args.output), "order": "alternating_AB_BA", "profiler": False},
        "results": [],
    }
    for sequence in args.sequences:
        for group_size in args.group_sizes:
            for target in ["kernel", "qwen3_5", "qwen3_5_moe", "qwen4_exp"]:
                for backward in [False, True]:
                    runner = (
                        kernel_runner(sequence, group_size, backward)
                        if target == "kernel"
                        else layer_runner(target, sequence, group_size, backward)
                    )
                    row = {"target": target, "sequence": sequence, "group_size": group_size, "backward": backward}
                    row.update(measure(runner, args.warmup, args.repeats, args.iterations))
                    result["results"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    del runner
                    get_torch_device().empty_cache()


if __name__ == "__main__":
    main()
