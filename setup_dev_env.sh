#!/bin/bash
set -e

echo "Setting up vLLM development environment with uv..."

# Create/activate virtual environment with uv
echo "Creating virtual environment with uv..."\
uv python install 3.11
uv venv --python 3.11

echo "Activating virtual environment..."
source .venv/bin/activate

# Install precompiled vLLM wheel for C++ extensions
echo "Installing precompiled vLLM wheel from v0.10.1.1 release..."
uv pip install https://github.com/vllm-project/vllm/releases/download/v0.10.1.1/vllm-0.10.1.1-cp38-abi3-manylinux1_x86_64.whl

# Instead of installing in editable mode (which would compile C++), 
# we'll add the current directory to Python path
echo "Setting up Python path for local development..."
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> .venv/bin/activate

# Re-source to apply the PYTHONPATH
source .venv/bin/activate

echo "Setup complete!"
echo "The precompiled C++ extensions are installed from the wheel,"
echo "and Python code changes will be picked up from the local directory."
echo "To activate the environment, run: source .venv/bin/activate"