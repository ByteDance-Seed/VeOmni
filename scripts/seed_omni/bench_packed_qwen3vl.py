"""What the packed path saves on Qwen3-VL: the step's device placement.

Neither path pads — `pack_llm_conversations_for_forward` already concatenates every
sample's embed segments into one varlen sequence — so the packed graph's only win
is *where* the per-item work happens:

* **Conversation.** The carrier's tensors are built per item in the dataloader
  worker, and `batch_to_device` deliberately leaves them there: "tensors a module
  keeps inside its conversation items stay where the preprocessor built them;
  placing those is the module's own ``pre_forward`` job". The step therefore pays
  one `to(device)` per `ConversationItem` plus a `torch.cat` over all of them.
* **Packed.** The worker emits one `packed_input_ids` / `packed_position_ids` /
  `packed_attention_mask` triple; `batch_to_device` moves those top-level tensors
  in one transfer.

This script measures exactly that delta against the real checkpoint's shapes, plus
the backbone forward at the same token counts so the delta can be read as a share
of a step.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/seed_omni/bench_packed_qwen3vl.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import veomni.models.seed_omni.modules  # noqa: F401  (registrations)
from veomni.models.seed_omni.modules import OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.utils.hf_layout import (
    load_module_from_hf_source,
    read_hf_source,
    resolve_omni_checkpoint_root,
)
from veomni.trainer.omni.omni_trainer import batch_to_device


def _time(fn, *, warmup: int, iters: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/hdfs/veomni/models/Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--images", type=int, default=4, help="image items per sample")
    ap.add_argument("--grid", type=int, nargs=3, default=[1, 24, 24])
    ap.add_argument("--words", type=int, default=400, help="text words per text item")
    ap.add_argument("--batch", type=int, nargs="+", default=[3, 8], help="sample counts to sweep")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="/tmp/bench_packed_qwen3vl.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    root = args.model_path
    source = read_hf_source(resolve_omni_checkpoint_root(root))
    configs = source.layout.build_module_configs(root)
    hidden = configs["qwen3vl_llm"].text_config.hidden_size
    layers = configs["qwen3vl_llm"].text_config.num_hidden_layers

    t, h, w = args.grid
    merged_rows = t * h * w // 4  # spatial_merge_size == 2

    print(f"checkpoint : {root}")
    print(f"backbone   : {layers} layers, hidden {hidden}")
    print(f"sample     : {args.images} images x {merged_rows} merged rows, 2 text items x ~{args.words} words")

    rows: list[dict] = []
    for samples in args.batch:
        # One `ConversationItem` per text turn and per image, as the carrier has.
        item_elems = [args.words * hidden] + [merged_rows * hidden] * args.images + [args.words * hidden]
        item_shapes = [(args.words, hidden)] + [(merged_rows, hidden)] * args.images + [(args.words, hidden)]
        items = [torch.zeros(*shape, dtype=torch.bfloat16) for shape in item_shapes for _ in range(samples)]
        total_rows = sum(shape[0] for shape in item_shapes) * samples

        def conv_place(placed=items) -> None:
            for item in placed:
                item.to(device, non_blocking=True)

        packed_ids = torch.zeros(1, total_rows, dtype=torch.long)
        packed_pids = torch.zeros(3, 1, total_rows, dtype=torch.long)
        packed_mask = torch.ones(1, total_rows, dtype=torch.long)
        packed_batch = {
            "packed_input_ids": packed_ids,
            "packed_position_ids": packed_pids,
            "packed_attention_mask": packed_mask,
        }

        def packed_place(batch=packed_batch) -> None:
            batch_to_device(batch, device)

        conv_ms, conv_std = _time(conv_place, warmup=args.warmup, iters=args.iters, device=device)
        pack_ms, pack_std = _time(packed_place, warmup=args.warmup, iters=args.iters, device=device)
        rows.append(
            {
                "samples": samples,
                "items": len(items),
                "rows": total_rows,
                "conversation_place_ms": round(conv_ms * 1e3, 3),
                "conversation_place_std_ms": round(conv_std * 1e3, 3),
                "packed_place_ms": round(pack_ms * 1e3, 3),
                "packed_place_std_ms": round(pack_std * 1e3, 3),
                "saved_ms": round((conv_ms - pack_ms) * 1e3, 3),
            }
        )
        del items, packed_ids, packed_pids, packed_mask, packed_batch, item_elems
        torch.cuda.empty_cache()

    # Backbone forward at the same token counts, so the placement delta can be
    # read as a share of a step rather than in isolation.
    llm = (
        load_module_from_hf_source(
            OMNI_MODEL_REGISTRY["qwen3vl_llm"](),
            configs["qwen3vl_llm"],
            source,
            "qwen3vl_llm",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    per_sample_rows = args.words * 2 + merged_rows * args.images
    for row in rows:
        T = per_sample_rows * row["samples"]
        emb = torch.zeros(1, T, hidden, dtype=torch.bfloat16, device=device)
        mask = torch.ones(1, T, dtype=torch.long, device=device)
        pids = torch.arange(T, device=device).unsqueeze(0)

        def fwd(e=emb, m=mask, p=pids) -> None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                llm.forward(inputs_embeds=e, attention_mask=m, position_ids=p)

        fwd_ms, _ = _time(fwd, warmup=args.warmup, iters=max(3, args.iters // 2), device=device)
        row["tokens"] = T
        row["backbone_forward_ms"] = round(fwd_ms * 1e3, 1)
        row["saved_pct_of_forward"] = round(row["saved_ms"] / (fwd_ms * 1e3) * 100, 2)
        del emb, mask, pids
        torch.cuda.empty_cache()

    result = {
        "model_path": root,
        "layers": layers,
        "hidden": hidden,
        "images_per_sample": args.images,
        "merged_rows_per_image": merged_rows,
        "words_per_text_item": args.words,
        "rows": rows,
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
