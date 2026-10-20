# GPU CI scheduling

Both GPU workflows run `run_gpu_tests.py` on a single eight-GPU node. The
explicit `gpu_tests.json` manifest is the test selection source of truth. It
keeps each original pytest invocation separate; new multi-GPU files should
reserve eight GPUs until their maximum world size and skip guards are audited.

```sh
uv run --frozen python scripts/ci/run_gpu_tests.py --suite unit --include-conditional
uv run --frozen python scripts/ci/run_gpu_tests.py --suite e2e
```

The smaller unit reservations cover models (2), Qwen4 Ulysses (4), context
parallel (4), Qwen3.5 GDN Ulysses (2), MoE-LoRA trainer (4), MoE-LoRA EP2 (2),
and MoE-LoRA streamed loading (2). The model directory has a priority hint
so its long invocation starts before shorter files. All remaining invocations reserve eight
GPUs. Each pytest process receives its own CUDA visibility before importing
torch. Distributed ports are allocated by the existing launch helpers; the
concurrent files have separate output paths. Each launcher tracks and cleans
only its own descendants before releasing its reservation. A failure stops
new dispatches while active invocations finish. Job-start stale-GPU cleanup
runs before the scheduler; there are no global process sweeps between tasks.

E2e alignment still compares every original SP/EP configuration. Its four-rank
modes may run concurrently on disjoint groups of four GPUs; eight-rank modes
remain exclusive. Every mode starts fresh torchrun workers. CPU BLAS threads
and Inductor compiler threads are capped to avoid concurrent oversubscription.

Results, task wall times, GPU assignments and JUnit XML are written under
`outputs/gpu-ci/{unit,e2e}` and uploaded by CI, including on failure. Use
`--serial` to diagnose scheduling, or `--keep-going` to time every invocation.
Do not compare summed command durations with elapsed wall time when jobs overlap.
The conditional Qwen3.5 test keeps its existing changed-path gate; use
`--include-conditional` for full local acceptance.

## Small integration fixtures

`tests/toy_config/ci/` contains opt-in smaller fixtures for synthetic checkpoint,
LoRA, model parity and e2e tests. `tests/tools/ci_config.py` resolves only existing
alternatives at explicitly opted-in build sites. Shared example configs and
processor/tokenizer integration fixtures remain separate.

These configs preserve model families, expert counts/top-k, backend selection,
parallel world sizes and data shapes. Qwen3.5 retains three linear-attention
layers plus full attention, and DeepSeek-V3 retains dense plus MoE layers.
Multimodal fixtures retain their token IDs, vocabularies and RoPE dimensions;
text-only fixtures use vocabularies of 4096. Qwen3-MoE retains four MoE
layers for the LoRA per-layer EP slicing assertions. Vision/audio widths are matched to
the text connectors. Existing numerical tolerances and checkpoint/optimizer
round-trip assertions are unchanged. Wan is invoked once; the Qwen3-MoE HF-save
checks run in the existing EP1/4/8 saveload cases, alongside independent CLI
DCP-to-HF verification.

Fused checkpoint formats also require distinguishable weight layouts: for
Qwen3-VL-MoE, keep expert intermediate width different from hidden width, and
keep twice the intermediate width different from hidden width. A square
projection deliberately raises instead of guessing whether to transpose it.
