"""
This script trims hidden layer number for a given safetensor dir
so that we can test weight loading against large models like deepseek conveniently

Example usage:
python scripts/trim_safetensor_layers.py \
  --model_dir /mnt/hdfs/tianle.zhong/models/unsloth-deepseek-v3.1-bf16-merged \
  --out_dir  /mnt/hdfs/tianle.zhong/models/unsloth-deepseek-v3.1-bf16-merged-5-layers \
  --num_layers 5 \
  --write_skipped \
  --asset-include "*.md" \
  --asset-include "*.jinja"
"""

import argparse
import glob
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_ID_RE = re.compile(r"^model\.layers\.(\d+)\..+")

DEFAULT_ASSET_ALLOWLIST = {
    # tokenizer & processors
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "processing_config.json",
    "preprocessor_config.json",
    # generation & prompt template
    "generation_config.json",
    "chat_template.jinja",
    # misc docs / licenses
    "README",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
}


def list_safetensor_files(model_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def copy_model_assets(src_dir: str, dst_dir: str, extra_includes: List[str]):
    """
    Copy non-weight assets from src_dir to dst_dir.
    Skips any *.safetensors and model.safetensors.index.json (we regenerate them),
    and also skips config.json here because we will rewrite it after updating num_layers.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # 1) Copy allowlisted fixed filenames if they exist
    copied = []
    for name in DEFAULT_ASSET_ALLOWLIST:
        src = os.path.join(src_dir, name)
        if os.path.exists(src) and os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
            copied.append(name)

    # 2) Apply extra include globs provided by user (e.g., *.md, *.jinja2)
    for pattern in extra_includes:
        for src in glob.glob(os.path.join(src_dir, pattern)):
            bn = os.path.basename(src)
            # skip known weights / index / config (config handled separately)
            if bn.endswith(".safetensors") or bn == "model.safetensors.index.json" or bn == "config.json":
                continue
            if os.path.isfile(src):
                dst = os.path.join(dst_dir, bn)
                # don’t overwrite existing allowlisted copies needlessly
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied.append(bn)

    # 3) Copy tokenizer folder if exists (some repos pack assets under subfolder)
    tok_dir_candidates = ["tokenizer", "assets", "processor"]
    for sub in tok_dir_candidates:
        src_sub = os.path.join(src_dir, sub)
        if os.path.isdir(src_sub):
            dst_sub = os.path.join(dst_dir, sub)
            if not os.path.exists(dst_sub):
                shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
                copied.append(f"{sub}/")

    if copied:
        print(
            f"🧩 Copied {len(copied)} asset(s): "
            + ", ".join(sorted(set(copied))[:10])
            + (" ..." if len(set(copied)) > 10 else "")
        )
    else:
        print("🧩 No extra assets found to copy (besides config.json which we will rewrite).")


def trim_and_reshard(
    model_dir: str,
    num_layers: int,
    out_dir: str,
    max_shard_size_gb: float = 5.0,
    write_skipped: bool = False,
    preview_per_shard: int = 10,
    asset_includes: List[str] = None,
):
    if asset_includes is None:
        asset_includes = []
    os.makedirs(out_dir, exist_ok=True)

    # Copy auxiliary assets first (except config.json and weights/index)
    copy_model_assets(model_dir, out_dir, asset_includes)

    files = list_safetensor_files(model_dir)
    if not files:
        raise FileNotFoundError(f"No .safetensors found in {model_dir}")

    kept_keys_global: List[str] = []
    skipped_keys_global: List[str] = []
    all_kept: List[Tuple[str, torch.Tensor]] = []

    print(f"Trimming to first {num_layers} layers from {model_dir}")
    for fn in files:
        in_path = os.path.join(model_dir, fn)
        kept: List[str] = []
        skipped: List[str] = []
        with safe_open(in_path, framework="pt") as f:
            for k in f.keys():
                m = LAYER_ID_RE.match(k)
                if m:
                    lid = int(m.group(1))
                    if lid < num_layers:
                        kept.append(k)
                        all_kept.append((k, f.get_tensor(k)))
                    else:
                        skipped.append(k)
                else:
                    kept.append(k)
                    all_kept.append((k, f.get_tensor(k)))

        # per-shard console preview
        if kept:
            print(f"\n📦 {fn}: kept {len(kept)}, skipped {len(skipped)}")
            for kk in kept[:preview_per_shard]:
                print(f"  └─ {kk}")
            if len(kept) > preview_per_shard:
                print(f"  ... ({len(kept) - preview_per_shard} more)")
        else:
            print(f"{fn}: ⚠️ no tensors kept")

        kept_keys_global.extend(kept)
        if write_skipped:
            skipped_keys_global.extend(skipped)

    if not all_kept:
        raise RuntimeError("No tensors kept at all — check layer prefix or num_layers.")

    # Write audit files
    kept_path = os.path.join(out_dir, "kept_keys.txt")
    with open(kept_path, "w") as f:
        for k in kept_keys_global:
            f.write(k + "\n")
    print(f"\n🗒️  Wrote kept tensor list: {kept_path} (total {len(kept_keys_global)})")

    if write_skipped:
        skipped_path = os.path.join(out_dir, "skipped_keys.txt")
        with open(skipped_path, "w") as f:
            for k in skipped_keys_global:
                f.write(k + "\n")
        print(f"🗒️  Wrote skipped tensor list: {skipped_path} (total {len(skipped_keys_global)})")

    # Re-shard compactly
    max_bytes = int(max_shard_size_gb * (1024**3))
    current, current_size, shards = [], 0, []
    for k, t in all_kept:
        sz = tensor_nbytes(t)
        if current and current_size + sz > max_bytes:
            shards.append(current)
            current, current_size = [], 0
        current.append((k, t))
        current_size += sz
    if current:
        shards.append(current)

    pad = max(5, len(str(len(shards))))
    total_size = 0
    weight_map: Dict[str, str] = {}
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:0{pad}d}-of-{len(shards):0{pad}d}.safetensors"
        shard_dict = {}
        for k, t in shard:
            shard_dict[k] = t
            weight_map[k] = fname
            total_size += tensor_nbytes(t)
        save_file(shard_dict, os.path.join(out_dir, fname))
        print(f"📝 wrote {fname} with {len(shard)} tensors")

    index_obj = {"weight_map": weight_map, "metadata": {"total_size": total_size, "format": "safetensors"}}

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_obj, f, indent=2)

    # Update and write config.json last (overrides any copied one)
    cfg_in = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_in):
        cfg = json.load(open(cfg_in))
        for k in ["num_hidden_layers", "n_layer", "num_layers"]:
            if k in cfg:
                print(f"Updating {k}: {cfg[k]} → {num_layers}")
                cfg[k] = num_layers
        json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    else:
        print("Warning: no config.json found to update.")

    print(f"\n✅ Done. Trimmed model with {num_layers} layers saved to {out_dir}")
    print("   Load with from_pretrained(out_dir) — HF will use model.safetensors.index.json.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_layers", type=int, required=True)
    ap.add_argument("--max_shard_size_gb", type=float, default=5.0)
    ap.add_argument("--write_skipped", action="store_true", help="Also write skipped_keys.txt for auditing.")
    ap.add_argument("--preview_per_shard", type=int, default=10, help="How many kept keys to print per shard.")
    ap.add_argument(
        "--asset-include",
        action="append",
        default=[],
        help="Extra glob(s) of assets to copy (e.g., '*.md'). Can be used multiple times.",
    )
    args = ap.parse_args()

    trim_and_reshard(
        args.model_dir,
        args.num_layers,
        args.out_dir,
        max_shard_size_gb=args.max_shard_size_gb,
        write_skipped=args.write_skipped,
        preview_per_shard=args.preview_per_shard,
        asset_includes=args.asset_include,
    )
