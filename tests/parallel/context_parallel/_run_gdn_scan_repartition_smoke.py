#!/usr/bin/env python3
"""Host smoke for CP-A repartition without importing full veomni (hf-hub pin)."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / "veomni" / "distributed" / "context_parallel"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    # Minimal package stubs so relative imports in sharding/gdn_scan_cp work.
    for name, path in [
        ("veomni", ROOT / "veomni"),
        ("veomni.distributed", ROOT / "veomni" / "distributed"),
        ("veomni.distributed.context_parallel", CP),
    ]:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    sharding = _load(
        "veomni.distributed.context_parallel.sharding",
        CP / "sharding.py",
        "veomni.distributed.context_parallel",
    )
    scan = _load(
        "veomni.distributed.context_parallel.gdn_scan_cp",
        CP / "gdn_scan_cp.py",
        "veomni.distributed.context_parallel",
    )
    # gdn_scan_cp imports .sharding — already loaded under that name.

    n_ok = 0
    for cp_size in (2, 4, 8):
        for seq_len in (64, 128, 256):
            full = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len, 1)
            zigzag = [
                sharding.balanced_cp_slice(full, cp_size=cp_size, cp_rank=r, dim=1)
                for r in range(cp_size)
            ]
            blocks = scan.simulate_zigzag_to_block(zigzag, cp_size=cp_size, seq_dim=1)
            local = seq_len // cp_size
            for r, block in enumerate(blocks):
                expected = full[:, r * local : (r + 1) * local, :]
                assert torch.equal(block, expected), (cp_size, r)
            back = scan.simulate_block_to_zigzag(blocks, cp_size=cp_size, seq_dim=1)
            for r in range(cp_size):
                assert torch.equal(back[r], zigzag[r]), (cp_size, r)
            n_ok += 1

    assert scan.gdn_replication_factor_for_impl("state_passing_serial", cp_size=4) == 1
    assert scan.gdn_replication_factor_for_impl("gather_full", cp_size=4) == 4
    print(f"CP_A_REPARTITION_SMOKE_OK cases={n_ok}")


if __name__ == "__main__":
    main()
