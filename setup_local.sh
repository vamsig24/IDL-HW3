#!/usr/bin/env bash
# Sets up a local GPU environment for hw3_p2_local.ipynb on a Linux PC with an NVIDIA GPU.
#
#   1. Creates a Python 3.12 venv with torch 2.7.1 (CUDA 12.8 build, works on RTX 30/40/50 series)
#   2. Installs requirements-local.txt and registers the venv as a Jupyter kernel
#   3. Downloads the HW3P2 dataset from Kaggle (skip with --no-data)
#   4. Links the dataset into the repo as ./hw3p2_data, which is the root the notebook config uses
#
# Usage (from anywhere):
#   ./setup_local.sh                 # full setup
#   ./setup_local.sh --no-data       # skip the dataset download
#
# Overridable with environment variables:
#   VENV_DIR     venv location               (default: ~/venvs/hw3-cu128)
#   DATA_DIR     where the dataset is put    (default: ~/hw3p2-data)
#   KERNEL_NAME  Jupyter kernel name         (default: hw3-cu128, the kernel the notebook expects)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/hw3-cu128}"
DATA_DIR="${DATA_DIR:-$HOME/hw3p2-data}"
KERNEL_NAME="${KERNEL_NAME:-hw3-cu128}"
TORCH_VERSION="2.7.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
DATA_URL="https://www.kaggle.com/api/v1/datasets/download/cmu11785/f26-hw3p2-data"
DOWNLOAD_DATA=1

for arg in "$@"; do
    case "$arg" in
        --no-data) DOWNLOAD_DATA=0 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg (see --help)"; exit 1 ;;
    esac
done

step() { echo; echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------------------------------
step "Checking the NVIDIA driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Install the NVIDIA driver first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
# CUDA 12.8 wheels need driver >= 570
[ "$DRIVER_MAJOR" -ge 570 ] || die "Driver $DRIVER_MAJOR is too old for CUDA 12.8 builds of torch; update to 570 or newer."

# --------------------------------------------------------------------------------------------
step "Finding Python 3.12"
PYTHON=""
if command -v python3.12 >/dev/null; then
    PYTHON="$(command -v python3.12)"
else
    # No system 3.12: get a standalone one through uv
    if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "python3.12 not found; installing uv (https://astral.sh/uv) to fetch it"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    "$UV" python install 3.12
    PYTHON="$("$UV" python find 3.12)"
fi
echo "Using $PYTHON ($("$PYTHON" --version))"

# --------------------------------------------------------------------------------------------
step "Creating venv at $VENV_DIR"
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Already exists, reusing it"
else
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON" -m venv "$VENV_DIR"
fi
PIP="$VENV_DIR/bin/pip"
VPY="$VENV_DIR/bin/python"
"$PIP" install --quiet --upgrade pip

# --------------------------------------------------------------------------------------------
step "Installing torch $TORCH_VERSION (CUDA 12.8)"
"$PIP" install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" --index-url "$TORCH_INDEX"

step "Installing requirements-local.txt"
# The constraint stops any requirement from swapping out the CUDA build of torch
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
printf 'torch==%s\ntorchaudio==%s\n' "$TORCH_VERSION" "$TORCH_VERSION" > "$CONSTRAINTS"
"$PIP" install -r "$REPO_DIR/requirements-local.txt" -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX"

step "Registering Jupyter kernel '$KERNEL_NAME'"
"$VPY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

step "Verifying torch on the GPU"
"$VPY" - <<'EOF'
import torch, torchaudio
assert torch.cuda.is_available(), "torch cannot see the GPU"
x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize(); (x @ x).sum().item()
print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__}, GPU {torch.cuda.get_device_name(0)}: OK")
EOF

# --------------------------------------------------------------------------------------------
find_data_root() { find "$DATA_DIR" -maxdepth 3 -type d -name hw3p2_data 2>/dev/null | head -1; }

if [ "$DOWNLOAD_DATA" -eq 1 ]; then
    step "Dataset"
    if [ -n "$(find_data_root)" ]; then
        echo "Found existing dataset at $(find_data_root), skipping download"
    else
        mkdir -p "$DATA_DIR"
        FREE_GB="$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')"
        # ~13 GB zip plus ~16 GB unpacked, both on disk until the zip is removed
        [ "$FREE_GB" -ge 32 ] || die "Need ~32 GB free in $DATA_DIR, only ${FREE_GB} GB available."
        echo "Downloading ~13 GB to $DATA_DIR (this takes a while)"
        curl -L --fail -o "$DATA_DIR/f26-hw3p2-data.zip" "$DATA_URL"
        echo "Unzipping"
        unzip -q -o "$DATA_DIR/f26-hw3p2-data.zip" -d "$DATA_DIR"
        rm -f "$DATA_DIR/f26-hw3p2-data.zip"
    fi
fi

DATA_ROOT="$(find_data_root)"
if [ -n "$DATA_ROOT" ]; then
    step "Linking $REPO_DIR/hw3p2_data -> $DATA_ROOT"
    ln -sfn "$DATA_ROOT" "$REPO_DIR/hw3p2_data"
    for split in train-clean-100 dev-clean test-clean; do
        printf '  %-16s %6s files\n' "$split" "$(ls "$DATA_ROOT/$split/fbank" | wc -l)"
    done
else
    echo
    echo "No dataset found in $DATA_DIR. Run again without --no-data, or link it yourself:"
    echo "  ln -sfn /path/to/hw3p2_data $REPO_DIR/hw3p2_data"
fi

# --------------------------------------------------------------------------------------------
step "Done"
cat <<EOF
Open hw3_p2_local.ipynb, select the "Python ($KERNEL_NAME)" kernel, and run from the top.
Run tests from a terminal with:  $VPY -m tests.test_attention   (from $REPO_DIR)
EOF
