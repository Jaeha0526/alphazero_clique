#!/bin/bash

# AlphaZero Clique Setup Script
# This script sets up the environment for both PyTorch and JAX implementations

set -e  # Exit on error

echo "======================================"
echo "AlphaZero Clique Environment Setup"
echo "======================================"

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the alphazero_clique root directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch requirements
echo ""
echo "Installing PyTorch requirements..."
pip install -r requirements.txt
echo "✓ PyTorch dependencies installed"

# Install JAX requirements
echo ""
echo "Installing JAX requirements..."

# Detect system and install appropriate JAX version
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Checking CUDA version..."
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    
    if [ "$CUDA_VERSION" = "12" ]; then
        echo "CUDA 12 detected. Installing JAX with CUDA 12 support..."
        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    elif [ "$CUDA_VERSION" = "11" ]; then
        echo "CUDA 11 detected. Installing JAX with CUDA 11 support..."
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Unknown CUDA version. Installing JAX with CUDA 11 support (most compatible)..."
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi
else
    echo "No NVIDIA GPU detected. Installing JAX with CPU support..."
    pip install --upgrade jax jaxlib
fi

# Install other JAX dependencies
echo "Installing JAX ecosystem packages..."
pip install --upgrade flax optax dm-haiku chex

# Install additional JAX requirements if requirements_jax.txt exists
if [ -f "jax_full_src/requirements_jax.txt" ]; then
    echo "Installing additional JAX requirements..."
    pip install -r jax_full_src/requirements_jax.txt
fi

# Verify installations
echo ""
echo "======================================"
echo "Verifying installations..."
echo "======================================"

# Test PyTorch
echo ""
echo "Testing PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test PyTorch Geometric
python3 -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"

# Test JAX
echo ""
echo "Testing JAX installation..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"

# Test Flax
python3 -c "import flax; print(f'Flax version: {flax.__version__}')"

# Test Optax
python3 -c "import optax; print(f'Optax version: {optax.__version__}')"

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p experiments
mkdir -p logs
mkdir -p playable_models
echo "✓ Directories created"

# Final message
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the PyTorch pipeline:"
echo "  python src/pipeline_clique.py --experiment-name my_experiment"
echo ""
echo "To run the JAX pipeline:"
echo "  python jax_full_src/run_jax_optimized.py --experiment-name my_experiment"
echo ""
echo "For JAX GPU usage:"
if command -v nvidia-smi &> /dev/null; then
    # Create GPU activation script
    cat > activate_gpu_env.sh << 'EOF'
#!/bin/bash
# Activate GPU environment for JAX

# Detect CUDA installation
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d "/usr/local/cuda-11.8" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "Warning: CUDA installation not found in standard locations"
fi

if [ ! -z "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "GPU Environment Activated:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  XLA memory preallocation: disabled"
echo "  XLA memory fraction: 0.8"
EOF
    chmod +x activate_gpu_env.sh
    echo "  GPU detected! To use GPU with JAX, run:"
    echo "    source activate_gpu_env.sh"
    echo "  This script has been created for you."
else
    echo "  No GPU detected. JAX will run on CPU."
fi
echo ""