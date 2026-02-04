#!/bin/bash
#
# BBScore Installation Script
#
# This script sets up the BBScore environment for students.
# It handles conda installation, environment creation, and dependency installation.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Options:
#   --no-conda    Skip conda installation check
#   --cpu-only    Install CPU-only PyTorch (smaller download)
#   --data-dir    Set custom data directory
#   --help        Show help message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENV_NAME="bbscore"
PYTHON_VERSION="3.11"
CPU_ONLY=false
SKIP_CONDA=false
DATA_DIR=""

# Print colored output
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Show help
show_help() {
    echo "BBScore Installation Script"
    echo ""
    echo "Usage: ./install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-conda        Skip conda installation check"
    echo "  --cpu-only        Install CPU-only PyTorch (no CUDA)"
    echo "  --data-dir DIR    Set custom data directory"
    echo "  --env-name NAME   Set environment name (default: bbscore)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./install.sh                           # Full installation"
    echo "  ./install.sh --cpu-only                # CPU-only installation"
    echo "  ./install.sh --data-dir ~/bbscore_data # Custom data directory"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-conda)
            SKIP_CONDA=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_header "BBScore Installation"

echo "This script will:"
echo "  1. Check/install conda (if needed)"
echo "  2. Create a Python $PYTHON_VERSION environment named '$ENV_NAME'"
echo "  3. Install PyTorch and dependencies"
echo "  4. Configure environment variables"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi
print_info "Detected OS: $OS"

# Check for conda
print_header "Step 1: Checking Conda"

if command -v conda &> /dev/null; then
    print_success "Conda is installed"
    CONDA_PATH=$(which conda)
    print_info "Path: $CONDA_PATH"

    # Initialize conda for this shell
    eval "$(conda shell.bash hook 2>/dev/null)" || true
else
    if [ "$SKIP_CONDA" = true ]; then
        print_warning "Conda not found, but --no-conda specified. Using pip only."
    else
        print_warning "Conda not found. Installing Miniconda..."

        # Download Miniconda
        if [ "$OS" = "linux" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [ "$OS" = "macos" ]; then
            # Check for ARM or Intel
            if [[ $(uname -m) == "arm64" ]]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
        else
            print_error "Automatic conda installation not supported on Windows."
            print_info "Please install Miniconda manually from: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi

        print_info "Downloading Miniconda..."
        curl -fsSL -o /tmp/miniconda.sh "$MINICONDA_URL"

        print_info "Installing Miniconda..."
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        rm /tmp/miniconda.sh

        # Initialize conda
        export PATH="$HOME/miniconda3/bin:$PATH"
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash 2>/dev/null || true

        print_success "Miniconda installed successfully"
        print_warning "Please restart your terminal after installation completes"
    fi
fi

# Create conda environment
print_header "Step 2: Creating Python Environment"

if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_info "Using existing environment"
    fi
fi

if ! conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    print_info "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate environment
print_info "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

print_success "Environment '$ENV_NAME' is active"
print_info "Python: $(python --version)"

# Install PyTorch
print_header "Step 3: Installing PyTorch"

if [ "$CPU_ONLY" = true ]; then
    print_info "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$CUDA_VERSION" ]; then
            print_info "NVIDIA GPU detected (Driver: $CUDA_VERSION)"
            print_info "Installing PyTorch with CUDA support..."

            # Try to detect CUDA version for appropriate PyTorch install
            if command -v nvcc &> /dev/null; then
                NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
                print_info "CUDA version: $NVCC_VERSION"
            fi

            # Install with CUDA 12.1 by default (most compatible with recent drivers)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            print_warning "NVIDIA driver found but no GPU detected. Installing CPU version."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_warning "No NVIDIA GPU detected. Installing CPU-only PyTorch."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')" && print_success "PyTorch installed successfully"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install other dependencies
print_header "Step 4: Installing Dependencies"

if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_warning "requirements.txt not found. Installing essential packages..."
    pip install numpy scipy scikit-learn pillow opencv-python tqdm h5py transformers timm wandb boto3 gdown google-cloud-storage
fi

# Configure environment variables
print_header "Step 5: Configuring Environment"

# Determine data directory
if [ -z "$DATA_DIR" ]; then
    # Default locations based on OS
    if [ "$OS" = "macos" ]; then
        DEFAULT_DATA_DIR="$HOME/bbscore_data"
    else
        DEFAULT_DATA_DIR="$HOME/bbscore_data"
    fi

    echo ""
    echo "BBScore needs a directory to store datasets and model weights."
    echo "This directory should have at least 50GB of free space."
    echo ""
    read -p "Enter data directory [$DEFAULT_DATA_DIR]: " DATA_DIR
    DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
fi

# Create data directory
mkdir -p "$DATA_DIR"
print_success "Data directory: $DATA_DIR"

# Add to shell config
SHELL_CONFIG=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
fi

if [ -n "$SHELL_CONFIG" ]; then
    # Check if already configured
    if ! grep -q "SCIKIT_LEARN_DATA" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# BBScore configuration" >> "$SHELL_CONFIG"
        echo "export SCIKIT_LEARN_DATA=\"$DATA_DIR\"" >> "$SHELL_CONFIG"
        print_success "Added SCIKIT_LEARN_DATA to $SHELL_CONFIG"
    else
        print_info "SCIKIT_LEARN_DATA already configured in $SHELL_CONFIG"
    fi
fi

# Set for current session
export SCIKIT_LEARN_DATA="$DATA_DIR"

# Create activation script
ACTIVATE_SCRIPT="activate_bbscore.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Activate BBScore environment
# Usage: source activate_bbscore.sh

# Activate conda environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Set environment variables
export SCIKIT_LEARN_DATA="$DATA_DIR"

echo "BBScore environment activated!"
echo "Data directory: \$SCIKIT_LEARN_DATA"
echo ""
echo "Quick start:"
echo "  python check_system.py --quick"
echo "  python run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge"
EOF
chmod +x "$ACTIVATE_SCRIPT"
print_success "Created $ACTIVATE_SCRIPT"

# Run system check
print_header "Step 6: System Check"

print_info "Running system diagnostic..."
python check_system.py --quick 2>/dev/null || print_warning "System check failed (non-critical)"

# Final summary
print_header "Installation Complete!"

echo "To use BBScore:"
echo ""
echo "  1. Activate the environment:"
echo -e "     ${GREEN}source activate_bbscore.sh${NC}"
echo "     or"
echo -e "     ${GREEN}conda activate $ENV_NAME${NC}"
echo ""
echo "  2. Run a quick test:"
echo -e "     ${GREEN}python run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge${NC}"
echo ""
echo "  3. Check your system:"
echo -e "     ${GREEN}python check_system.py${NC}"
echo ""
echo "Data directory: $DATA_DIR"
echo ""

if [ "$OS" = "linux" ] || [ "$OS" = "macos" ]; then
    print_warning "Remember to restart your terminal or run: source $SHELL_CONFIG"
fi

echo ""
print_success "Happy benchmarking!"
