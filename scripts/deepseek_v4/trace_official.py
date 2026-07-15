"""Trace the official DeepSeek-V4-Flash inference implementation.

Run from the VeOmni environment with four ranks, for example::

    .venv/bin/torchrun --standalone --nproc-per-node=4 \
      scripts/deepseek_v4/trace_official.py \
      --repo /tmp/DeepSeek-V4-Flash \
      --checkpoint /tmp/DeepSeek-V4-Flash-official-mp4 \
      --output /tmp/deepseek-v4-traces/official.pt \
      --seq-len 4096

The script imports, but does not modify, the official ``inference/model.py``
shipped in the safetensors repository. Only rank 0 writes the trace because
the reference model-parallel implementation produces identical routing and
attention indices on every rank.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoTokenizer


_TRACE_TEXT = (
    "Numerical parity in distributed inference requires identical routing, "
    "sparse attention selection, normalization, and quantization semantics. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--logprob-chunk-size", type=int, default=64)
    parser.add_argument(
        "--detail-layers",
        type=int,
        nargs="*",
        default=(),
        help="Store full intermediate tensors for the selected layer IDs",
    )
    return parser.parse_args()


def build_input_ids(tokenizer: AutoTokenizer, seq_len: int, device: torch.device) -> torch.Tensor:
    seed_tokens = tokenizer.encode(_TRACE_TEXT, add_special_tokens=False)
    if not seed_tokens:
        raise RuntimeError("Trace text unexpectedly encoded to zero tokens")
    bos = tokenizer.bos_token_id
    prefix = [] if bos is None else [bos]
    repeats = (seq_len - len(prefix) + len(seed_tokens) - 1) // len(seed_tokens)
    tokens = (prefix + (seed_tokens * repeats))[:seq_len]
    if len(tokens) != seq_len:
        raise RuntimeError(f"Could not construct {seq_len} input tokens")
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)


def hidden_fingerprint(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = hidden_states.detach().float().flatten(start_dim=2)
    return {
        "mean": flat.mean(dim=-1).cpu(),
        "rms": flat.square().mean(dim=-1).sqrt().cpu(),
        "sample": flat[..., :8].cpu(),
    }


def distributed_target_logprobs(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    local_weight: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute exact next-token log-probs without gathering the sharded head."""
    rank = dist.get_rank()
    part_vocab = local_weight.shape[0]
    vocab_start = rank * part_vocab
    pieces: list[torch.Tensor] = []
    for start in range(0, hidden_states.shape[1] - 1, chunk_size):
        end = min(hidden_states.shape[1] - 1, start + chunk_size)
        local_logits = F.linear(hidden_states[:, start:end].float(), local_weight.float())
        local_max = local_logits.amax(dim=-1)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        denominator = (local_logits - local_max.unsqueeze(-1)).exp().sum(dim=-1)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)

        chunk_targets = targets[:, start + 1 : end + 1]
        local_target = torch.zeros_like(local_max)
        owned = (chunk_targets >= vocab_start) & (chunk_targets < vocab_start + part_vocab)
        if owned.any():
            local_ids = (chunk_targets - vocab_start).clamp(0, part_vocab - 1)
            gathered = local_logits.gather(-1, local_ids.unsqueeze(-1)).squeeze(-1)
            local_target = torch.where(owned, gathered, local_target)
        dist.all_reduce(local_target, op=dist.ReduceOp.SUM)
        pieces.append((local_target - local_max - denominator.log()).cpu())
    return torch.cat(pieces, dim=1)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)

    inference_dir = args.repo / "inference"
    sys.path.insert(0, str(inference_dir))
    import model as official_model  # noqa: PLC0415

    with (inference_dir / "config.json").open() as handle:
        model_args = official_model.ModelArgs(**json.load(handle))
    model_args.max_batch_size = 1
    model_args.max_seq_len = args.seq_len

    with torch.device(device):
        model = official_model.Transformer(model_args)
    load_model(model, str(args.checkpoint / f"model{rank}-mp{dist.get_world_size()}.safetensors"), strict=False)
    model.eval()
    # The official generate.py sets this after checkpoint loading. Its cached
    # window/compression index helpers intentionally rely on the default device.
    torch.set_default_device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    input_ids = build_input_ids(tokenizer, args.seq_len, device)
    trace: dict[str, object] = {
        "format_version": 1,
        "implementation": "official",
        "input_ids": input_ids.cpu(),
        "moe_topk": {},
        "indexer_topk": {},
        "attention_topk": {},
        "hidden": {},
        "details": {},
    }

    if rank == 0:
        for layer_id, layer in enumerate(model.layers):

            def gate_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["moe_topk"][layer_id] = output[1].detach().to(torch.int32).cpu()

            def layer_hook(_module, _inputs, output, *, layer_id=layer_id):
                trace["hidden"][layer_id] = hidden_fingerprint(output)

            layer.ffn.gate.register_forward_hook(gate_hook)
            layer.register_forward_hook(layer_hook)
            if layer_id in args.detail_layers:
                trace["details"][layer_id] = {}
                details = trace["details"][layer_id]

                def save_tensor(name, tensor, *, details=details):
                    details[name] = tensor.detach().cpu()

                def layer_detail_hook(_module, inputs, output, *, save_tensor=save_tensor):
                    save_tensor("layer_input", inputs[0])
                    save_tensor("layer_output", output)

                def attn_input_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("attn_input", output)

                def attn_output_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("attn_output", output)

                def mlp_input_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("mlp_input", output)

                def mlp_output_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("mlp_output", output)

                def gate_detail_hook(_module, _inputs, output, *, save_tensor=save_tensor):
                    save_tensor("router_weights", output[0])

                def tensor_output_hook(name, *, save_tensor=save_tensor):
                    def hook(_module, _inputs, output):
                        save_tensor(name, output)

                    return hook

                layer.register_forward_hook(layer_detail_hook)
                layer.attn_norm.register_forward_hook(attn_input_hook)
                layer.attn.register_forward_hook(attn_output_hook)
                layer.ffn_norm.register_forward_hook(mlp_input_hook)
                layer.ffn.register_forward_hook(mlp_output_hook)
                layer.ffn.gate.register_forward_hook(gate_detail_hook)
                layer.attn.wq_a.register_forward_hook(tensor_output_hook("q_a_proj"))
                layer.attn.q_norm.register_forward_hook(tensor_output_hook("q_a_norm"))
                layer.attn.wq_b.register_forward_hook(tensor_output_hook("q_b_proj"))
                layer.attn.wkv.register_forward_hook(tensor_output_hook("kv_proj"))
                layer.attn.kv_norm.register_forward_hook(tensor_output_hook("kv_norm"))
                layer.attn.wo_b.register_forward_hook(tensor_output_hook("o_b_proj"))
            if getattr(layer.attn, "indexer", None) is not None:

                def indexer_hook(_module, _inputs, output, *, layer_id=layer_id):
                    relative = torch.where(output >= 0, output - args.seq_len, output)
                    trace["indexer_topk"][layer_id] = relative.detach().to(torch.int32).cpu()

                layer.attn.indexer.register_forward_hook(indexer_hook)

    active_layer = {"id": -1}
    original_sparse_attn = official_model.sparse_attn

    def traced_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        if rank == 0:
            trace["attention_topk"][active_layer["id"]] = topk_idxs.detach().to(torch.int32).cpu()
        return original_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    official_model.sparse_attn = traced_sparse_attn
    for layer_id, layer in enumerate(model.layers):
        original_forward = layer.attn.forward

        def attention_forward(self, *forward_args, _original=original_forward, _layer_id=layer_id, **forward_kwargs):
            active_layer["id"] = _layer_id
            return _original(*forward_args, **forward_kwargs)

        layer.attn.forward = types.MethodType(attention_forward, layer.attn)

    with torch.inference_mode():
        hidden_states = model.embed(input_ids)
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            hidden_states = layer(hidden_states, 0, input_ids)
        collapsed = model.head.hc_head(
            hidden_states,
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
        )
        normalized = model.norm(collapsed)
        logprobs = distributed_target_logprobs(
            normalized,
            input_ids,
            model.head.weight,
            args.logprob_chunk_size,
        )

    if rank == 0:
        trace["logprobs"] = logprobs
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.output)
        print(f"saved official trace to {args.output}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
