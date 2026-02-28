# Installation with Ascend NPU

## Required Environment

CANN == 8.3.RC1

## Prepare CANN

Choose one of the following methods to use CANN:

1. Install CANN according to the [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Install with pip


```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

# Choose one of the following installation options based on your needs:
# Option 1: Stable version (transformers < 5.0)
pip install -e .[npu_aarch64,transformers-stable]

# Option 2: Experimental version for new models (transformers ≥ 5.0)
# Note: This uses the transformers5-exp extra which includes transformers 5.0+ support
# as specified in pyproject.toml (experimental and under development)
# pip install -e .[npu_aarch64,transformers5-exp]
```

### Set up CANN environment before installing torchcodec
Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
```bash
source $CANN_path/ascend-toolkit/set_env.sh
```

### Video/Audio Processing Dependencies (Optional)

For video/audio processing capabilities, you need to install torchcodec separately. Follow these steps:

```bash
# Clone the torchcodec repository
cd ..
git clone https://github.com/meta-pytorch/torchcodec.git
cd torchcodec

# Checkout to a specific version for compatibility
git checkout v0.5.0

# Install required dependencies
# Note: We use conda to install ffmpeg here to ensure version compatibility (4.2.2)
# This installs ffmpeg in the current conda environment, not system-wide
# No need to run system-level installation (apt-get/yum) before this step
conda install pybind11 ffmpeg=4.2.2

# Set environment variables to ensure compilation finds the correct libraries from conda environment
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install torchcodec in development mode without build isolation
pip install -e . --no-build-isolation
```

## Ascend relevant Environment variables

```bash
# Set additional Ascend environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Add chunkloss feature
export VEOMNI_ENABLE_CHUNK_LOSS=1
```
