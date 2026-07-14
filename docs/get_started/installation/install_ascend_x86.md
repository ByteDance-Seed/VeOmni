# Installation with Ascend NPU (x86)

## Supported Hardware and Operating Systems

This installation guide applies to the following Ascend NPU products on x86_64 architecture:

| Hardware | Operating System | CANN Version | Status |
|----------|-----------------|--------------|--------|
| Ascend A2 (910B) | Ubuntu 22.04 | 8.3.RC2 / 9.0.0 | ✅ Supported |
| Ascend 950 (A5) | Ubuntu 22.04 | 9.0.0 | ✅ Supported |

> **Note**: A5 refers to the Ascend 950 series products. All existing unit tests and system tests pass on A5.
> For scenarios pending further validation on A5, please refer to [A5 Features Pending Validation](../../hardware_support/a5_unsupported_features.md).

## Required Environment

CANN == 8.3.RC1 (9.0.0 recommended for A5)

## Prepare CANN

Choose one of the following methods to use CANN:

1. Install CANN according to the [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

> **Note**: For Ascend 950 (A5) products, please use CANN 9.0.0.

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

`npu` is a single full superset for x86 Ascend: torch 2.7.1+cpu / torch-npu,
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

### Set up CANN environment before installing torchcodec
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

### Video/Audio Processing Dependencies (Optional)

For video/audio processing capabilities, you need to install torchcodec separately. Follow these steps:

```bash
# Clone the torchcodec repository
cd ..
git clone https://github.com/meta-pytorch/torchcodec.git
cd torchcodec

# Checkout to a specific version for compatibility
git checkout v0.5.0

# Copy the installation script to the torchcodec source directory
cp ../VeOmni/docs/get_started/installation/install_torchcodec_Ascend.sh .

# Note: Ensure Python is installed as a shared library (required for compiling C++ extensions)
# The installation script will automatically verify this requirement

# Run the installation script (replace with your actual CANN path)
bash install_torchcodec_Ascend.sh $CANN_path/ascend-toolkit/set_env.sh

# Verify installation
pip show torchcodec

# Test torchcodec import
python -c "from torchcodec.decoders import VideoDecoder; print('Success')"
# If the terminal outputs'Success', it indicates that the torchcodec installation was successful. If an error message is output, it indicates that the installation was not successful
```
