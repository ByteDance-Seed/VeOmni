#!/bin/bash
set -e

echo "=========================================="
echo "🚀 TorchCodec One-Click Installation for Ascend NPU (Enhanced Version)"
echo "=========================================="
echo "📍 Working Directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "=========================================="

# --- Accept CANN environment script path from command line ---
if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <CANN set_env.sh path>"
    echo "Example: $0 /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi
ASCEND_ENV="$1"

# --- Self-check ---
if [ ! -f "pyproject.toml" ] || [ ! -d "src/torchcodec" ]; then
    echo "❌ Error: Please run this script in the torchcodec source root directory!"
    exit 1
fi

# ==========================================
# 🔧 Auto install FFmpeg + development packages (ensure compilation success)
# ==========================================
echo "[0/4] Checking and installing FFmpeg development dependencies..."
install_ffmpeg() {
    if command -v yum &> /dev/null; then
        echo "   🔧 Installing with yum..."
        yum install -y ffmpeg ffmpeg-devel --nogpgcheck
    elif command -v apt &> /dev/null; then
        echo "   🔧 Installing with apt..."
        apt update -y
        apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
    else
        echo "   ❌ Error: yum / apt package manager not found!"
        echo "   Please manually install FFmpeg development packages and try again."
        exit 1
    fi
}

# Check if ffmpeg and development libraries are installed
if ! command -v ffmpeg &> /dev/null || ! pkg-config --exists libavcodec 2>/dev/null; then
    echo "   ⚠️  FFmpeg or development packages not found, installing automatically..."
    install_ffmpeg
else
    echo "   ✅ FFmpeg already exists"
fi

# --- 1. Python dependencies ---
echo "[1/4] Installing Python build tools..."
pip install --quiet --upgrade pip
pip install --quiet pybind11 wheel setuptools cmake ninja

# --- 2. Load Ascend environment ---
echo "[2/4] Loading Ascend CANN environment..."
if [ -f "$ASCEND_ENV" ]; then
    source "$ASCEND_ENV"
    echo "   ✅ Loaded: $ASCEND_ENV"
else
    echo "❌ CANN environment not found: $ASCEND_ENV"
    exit 1
fi

# --- 3. Key: Auto search FFmpeg path ---
echo "[3/4] Configuring FFmpeg and compilation environment variables..."

# Auto search for libavcodec.pc
FFMPEG_PC_PATH=$(find /usr /usr/local /opt -name "libavcodec.pc" 2>/dev/null | head -n 1)
if [ -z "$FFMPEG_PC_PATH" ]; then
    echo "   ❌ Error: FFmpeg development files not found!"
    exit 1
fi

PC_DIR=$(dirname "$FFMPEG_PC_PATH")
export PKG_CONFIG_PATH="$PC_DIR:$PKG_CONFIG_PATH"
echo "   ✅ FFmpeg pkg-config path: $PC_DIR"

# Verify FFmpeg
FFMPEG_VER=$(pkg-config --modversion libavcodec)
echo "   ✅ FFmpeg identified successfully (version: $FFMPEG_VER)"

# Compilation environment variables
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())"):$CMAKE_PREFIX_PATH
export LIBRARY_PATH=/usr/local/lib:/usr/lib/$(uname -m)-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64:$LD_LIBRARY_PATH

# --- 4. Compile and install ---
echo "[4/4] Cleaning and compiling installation..."
rm -rf build/ dist/ *.egg-info src/torchcodec.egg-info

pip install -e . --no-build-isolation

echo "=========================================="
echo "🎉 Installation successful!"
echo "=========================================="
echo "   torchcodec Installation Verification   "
echo "source $ASCEND_ENV"
echo "python -c \"from torchcodec.decoders import VideoDecoder; print('install success')\""
