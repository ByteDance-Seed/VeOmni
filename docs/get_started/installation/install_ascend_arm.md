# Installation with Ascend NPU (ARM)

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

# use the frozen uv env
uv sync --frozen  --extra npu_aarch64
source .venv/bin/activate
```

`npu_aarch64` installs the Ascend and multimodal dependency superset except
`torchcodec`, which has no compatible aarch64 wheel.

> **Note**: For video/audio processing, you also need to install FFmpeg separately:
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

pip install -e .[npu_aarch64]
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

# TorchCodec 0.10 is the release compatible with the PyTorch 2.10 stack above
git checkout v0.10.0

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

The helper script performs a source build adapted for Ascend hosts. Validate the import and
video decode path on the target NPU image after any PyTorch, TorchCodec, CANN, or FFmpeg update.
