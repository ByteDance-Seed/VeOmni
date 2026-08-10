# Cursor Cloud (CPU-only) Environment

Notes for agents running on the Cursor Cloud VM. Not relevant to GPU CI or local
GPU/NPU dev boxes.

The Cursor Cloud VM is **CPU-only (no GPU/NPU), x86_64, Python 3.12**. The
startup update script already runs `uv sync --extra gpu --dev`, so `.venv` is
ready — activate it with `source .venv/bin/activate` before any command. `uv`
is installed under `~/.local/bin` (on `PATH` via `~/.bashrc`); the `gpu` extra's
CUDA torch wheels install fine here and `torch.cuda.is_available()` is `False`.

## What works CPU-only (use these to validate changes without hardware)

- Lint gate: `make quality` (and `make style` to auto-fix) — see `Makefile`.
- Patchgen drift: `make check-patchgen` (i.e. `patchgen --check`; CI equivalent
  of the check_patchgen job).
- Device API check — the `-d` flag takes a single directory, so run it once per dir:
  `for d in veomni tasks tests; do python tests/special_sanity/check_device_api_usage.py -d "$d"; done`.
- The CPU subset of `pytest` (registry/ops-gate/eager/data/lora-unit/converter/
  balance/DPO/checkpoint-callback tests). GPU-only tests self-skip via
  `IS_CUDA_AVAILABLE` / `@pytest.mark.skipif(device_count < N)`, but many files in
  `tests/models`, `tests/distributed`, `tests/e2e`, and most
  `tests/parallel/ulysses` and `tests/lora` integration tests need real GPUs — do
  not expect the full `pytest tests/` (a.k.a. `make test`) to pass here.

## What does NOT work here

Real training (`train.sh` / `tasks/train_*.py`), fused CUDA/Triton kernels
(flash-attn, quack, tilelang, FlashMLA), and any multi-GPU/FSDP2/Ulysses/EP test.
Those require the self-hosted GPU CI runners.

## Gotchas

- To build a model on this box, force all-eager ops and `init_device="cpu"`
  (flash-attn/triton are unavailable). `tasks/infer/*.py` show the eager
  `OpsImplementationConfig` pattern (`is_flash_attn_2_available()` → `eager`).
- `make build` is stale (its `setup.py` target doesn't exist); build the wheel
  with `./build.sh` (which runs `python3 -m build`) instead.
- Re-running `uv sync` is cheap and idempotent (~1s when unchanged); prefer it
  over `pip install`.
