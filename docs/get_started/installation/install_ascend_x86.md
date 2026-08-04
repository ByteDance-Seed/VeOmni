# Installation with Ascend NPU (x86)

## Required Environment

Use a CANN installation compatible with the selected hardware and
`torch-npu==2.10.0`. VeOmni provides repository images for CANN 8.3.RC2 and
9.0.0; see the [NPU version compatibility table](../../hardware_support/get_started_npu.md#version-compatibility).

## Prepare CANN

Choose one of the following methods to use CANN:

1. Install CANN according to the [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Install with uv or pip

### UV

> Recommend to use [uv](https://docs.astral.sh/uv/) for faster and easier installation.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

# use the locked uv env
uv sync --locked --extra npu
source .venv/bin/activate
```

`npu` is a single full superset for x86 Ascend: torch 2.10.0+cpu / torch-npu 2.10.0,
diffusion / audio / video / LoRA deps, and `megatron-energon`. CUDA-only
kernels (FA3 / FA4 / FlashQLA) are intentionally absent. See
[pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml)
for the full list.

> **Note**: video/audio processing also needs ffmpeg installed at the OS level:
> ```bash
> # Ubuntu/Debian/openEuler
> sudo apt-get install ffmpeg
> # or
> sudo yum install ffmpeg
> ```

### Pip

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

pip install -e .[npu]
pip install transformers==5.9.0
pip install datasets==2.21.0
```

### Set up the CANN environment
Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
```bash
source $CANN_path/ascend-toolkit/set_env.sh
```

To enable the NPU chunked cross-entropy loss, set
`model.ops_implementation.cross_entropy_loss_implementation: npu` in your training YAML
(replaces the legacy `VEOMNI_ENABLE_CHUNK_LOSS` environment variable).

> **Note:** The NPU chunked cross-entropy backs both `ForCausalLM` and
> `ForConditionalGeneration` (VLMs) — chunk_loss now does the SP reduction
> itself, so VLMs with Ulysses SP enabled get the correct loss. Only
> `ForSequenceClassification` stays on the eager wrapper: chunk_loss
> hard-codes the causal `labels[..., 1:]` shift, which is incompatible
> with the token-level (no-shift) labels that
> `ForSequenceClassificationLoss` expects. A `warning_rank0` is logged at
> install time; expect eager-level numbers for sequence-classification
> losses during profiling.

The x86 `npu` extra already installs the supported `torchcodec==0.10.0`
wheel. No separate source build is required; install FFmpeg as shown above
when video or audio decoding is needed.
