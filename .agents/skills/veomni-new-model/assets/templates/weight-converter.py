#!/usr/bin/env python3
"""Strict checkpoint-conversion template for {{MODEL_NAME}}.

Customize NAME_RULES, DROP_RULES, TRANSPOSE_RULES, and custom_transform().
The unchanged template performs an identity conversion. Pass an expected target
manifest to check completeness and strictly load the target model separately.
Cross-source concatenation and initialization need additional accounting logic.
This template reads PyTorch and safetensors checkpoints only. It never uses
``strict=False`` and never overwrites the source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import torch


try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - environment-specific dependency
    raise SystemExit("safetensors is required: install the VeOmni environment") from exc


# Applied in order. Anchor patterns so accidental partial matches cannot rename keys.
NAME_RULES: list[tuple[str, str]] = [
    # (r"^module\.", ""),
]

# Every dropped namespace needs a reviewable reason.
DROP_RULES: list[tuple[str, str]] = [
    # (r"^ema\.", "training consumes non-EMA weights"),
]

# Map the final target key to a complete torch.permute dimension tuple.
TRANSPOSE_RULES: dict[str, tuple[int, ...]] = {
    # "model.layers.0.example.weight": (1, 0),
}


def custom_transform(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return target tensors for a per-source split or layout transformation."""
    return {name: tensor}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_safetensors(path: Path) -> Iterable[tuple[str, torch.Tensor, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            yield key, handle.get_tensor(key), str(path)


def _unwrap_torch_checkpoint(payload):
    if isinstance(payload, dict):
        for wrapper in ("state_dict", "model", "module"):
            candidate = payload.get(wrapper)
            if isinstance(candidate, dict) and all(isinstance(key, str) for key in candidate):
                return candidate
    return payload


def iter_torch_checkpoint(path: Path) -> Iterable[tuple[str, torch.Tensor, str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = _unwrap_torch_checkpoint(payload)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError(f"{path} does not contain a string-keyed state dict")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path}:{key} is {type(tensor).__name__}, not a tensor")
        yield key, tensor, str(path)


def source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ValueError(f"empty weight_map in {indexes[0]}")
        return [path / name for name in names]
    safetensors = sorted(path.glob("*.safetensors"))
    if safetensors:
        return safetensors
    torch_files = sorted(path.glob("*.pth")) + sorted(path.glob("*.pt")) + sorted(path.glob("*.bin"))
    if torch_files:
        return torch_files
    raise FileNotFoundError(f"no supported checkpoint files found under {path}")


def iter_source(path: Path) -> Iterable[tuple[str, torch.Tensor, str]]:
    for file in source_files(path):
        if file.suffix == ".safetensors":
            yield from iter_safetensors(file)
        else:
            yield from iter_torch_checkpoint(file)


def map_name(source_name: str) -> str:
    target = source_name
    for pattern, replacement in NAME_RULES:
        target = re.sub(pattern, replacement, target)
    return target


def drop_reason(source_name: str) -> str | None:
    for pattern, reason in DROP_RULES:
        if re.search(pattern, source_name):
            if not reason.strip():
                raise ValueError(f"drop rule {pattern!r} has no reason")
            return reason
    return None


def load_expected(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tensors" in payload:
        payload = payload["tensors"]
    if not isinstance(payload, dict):
        raise ValueError("expected manifest must be a key -> metadata object")
    return payload


def validate_expected(converted: dict[str, torch.Tensor], expected: dict[str, dict] | None) -> list[str]:
    if expected is None:
        return []
    errors: list[str] = []
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    if missing:
        errors.append(f"missing expected keys: {missing[:20]}")
    if extra:
        errors.append(f"unexpected converted keys: {extra[:20]}")
    for key in sorted(set(expected) & set(converted)):
        metadata = expected[key] if isinstance(expected[key], dict) else {}
        expected_shape = metadata.get("shape")
        if expected_shape is not None and list(converted[key].shape) != list(expected_shape):
            errors.append(f"shape mismatch {key}: got {list(converted[key].shape)}, expected {expected_shape}")
        expected_dtype = metadata.get("dtype")
        actual_dtype = str(converted[key].dtype).removeprefix("torch.")
        if expected_dtype is not None and actual_dtype != str(expected_dtype).removeprefix("torch."):
            errors.append(f"dtype mismatch {key}: got {actual_dtype}, expected {expected_dtype}")
    return errors


def convert(args: argparse.Namespace) -> dict:
    if args.report.resolve() == args.output.resolve():
        raise ValueError("report and output must be different paths")
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    converted: dict[str, torch.Tensor] = {}
    mapping: list[dict] = []
    dropped: list[dict] = []
    fingerprints: dict[str, str] = {}
    source_tensor_count = 0
    for file in source_files(args.source):
        fingerprints[str(file)] = sha256(file)

    for source_name, tensor, source_file in iter_source(args.source):
        source_tensor_count += 1
        reason = drop_reason(source_name)
        if reason is not None:
            dropped.append({"source": source_name, "reason": reason})
            continue
        target_name = map_name(source_name)
        transformed = custom_transform(target_name, tensor)
        if not transformed:
            raise ValueError(f"source tensor was neither converted nor dropped: {source_name}")
        for final_name, final_tensor in transformed.items():
            if final_name in TRANSPOSE_RULES:
                final_tensor = final_tensor.permute(TRANSPOSE_RULES[final_name]).contiguous()
            if final_name in converted:
                raise ValueError(f"mapping collision for target key: {final_name}")
            converted[final_name] = final_tensor.contiguous()
            mapping.append(
                {
                    "source": source_name,
                    "target": final_name,
                    "source_file": source_file,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(final_tensor.shape),
                    "source_dtype": str(tensor.dtype),
                    "target_dtype": str(final_tensor.dtype),
                    "identity": (
                        source_name == final_name
                        and tensor.shape == final_tensor.shape
                        and tensor.dtype == final_tensor.dtype
                    ),
                }
            )

    errors = validate_expected(converted, load_expected(args.expected_manifest))
    if errors:
        raise ValueError("; ".join(errors))
    report = {
        "model": "{{MODEL_NAME}}",
        "upstream_revision": "{{UPSTREAM_REVISION}}",
        "source_fingerprints": fingerprints,
        "source_tensor_count": source_tensor_count,
        "converted_tensor_count": len(converted),
        "dropped_tensor_count": len(dropped),
        "mapping": mapping,
        "dropped": dropped,
        "expected_manifest": str(args.expected_manifest) if args.expected_manifest else None,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(converted, str(args.output), metadata={"format": "pt", "model": "{{MODEL_NAME}}"})
        report["output_sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new .safetensors output path")
    parser.add_argument("--report", required=True, type=Path, help="new JSON report path; never overwritten")
    parser.add_argument("--expected-manifest", type=Path, help="JSON key -> {shape: [...]} target manifest")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.source.exists():
        print(f"error: source does not exist: {args.source}", file=sys.stderr)
        return 2
    try:
        report = convert(args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"converted={report['converted_tensor_count']} dropped={report['dropped_tensor_count']} "
        f"dry_run={report['dry_run']} report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
