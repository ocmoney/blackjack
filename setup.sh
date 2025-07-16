#!/bin/bash

# Exit on error
set -e

echo "Starting blackjack project setup..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    wget \
    curl \
    tmux \
    git

# Install Miniconda
echo "Installing Miniconda..."
if [ ! -d "/opt/conda" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/conda
    rm miniconda.sh
else
    echo "Miniconda already installed at /opt/conda"
fi

# Add conda to PATH for current session
export PATH="/opt/conda/bin:$PATH"

# Initialize conda for current shell
eval "$(/opt/conda/bin/conda shell.bash hook)"

# Add conda to PATH permanently
if [[ ":$PATH:" != *":/opt/conda/bin:"* ]]; then
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
fi

# Add conda environment activation to .bashrc
echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate blackjack' >> ~/.bashrc

# Activate conda environment
echo "Activating conda environment..."
conda activate base

# Create and activate conda environment for your project
echo "Creating blackjack conda environment..."
conda create -n blackjack python=3.9 -y
conda activate blackjack

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    # Use conda to install packages, fallback to pip if conda doesn't have them
    while IFS= read -r package; do
        # Skip comments and empty lines
        [[ $package =~ ^[[:space:]]*# ]] && continue
        [[ -z "${package// }" ]] && continue
        
        echo "Installing $package..."
        if conda install -y "$package" 2>/dev/null; then
            echo "✅ Installed $package via conda"
        else
            echo "Installing $package via pip..."
            pip install "$package"
        fi
    done < requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Configure tmux for smooth scrolling
echo "Configuring tmux..."
cat > ~/.tmux.conf << EOL
set -g mouse on
set -g terminal-overrides 'xterm*:smcup@:rmcup@'
EOL

# Add auto-tmux to .bashrc (if not already there)
if ! grep -q "auto_\$(date +%s%N)" ~/.bashrc; then
    echo "Setting up auto-tmux per terminal in ~/.bashrc..."
    cat >> ~/.bashrc << 'EOL'

# Automatically create a new tmux session for each new terminal
if command -v tmux &> /dev/null && [ -z "$TMUX" ]; then
  SESSION_NAME="auto_$(date +%s%N)"
  tmux new-session -s "$SESSION_NAME"
fi
EOL
fi

# Configure Git
echo "Configuring Git..."
git config --global user.name "blackjack-user"
git config --global user.email "user@blackjack.local"

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << EOL
# Miniconda
miniconda/
miniconda.sh
/opt/conda/

# Models
models/
*.pt
*.pth
*.h5
*.ckpt
*.bin
*.onnx

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
results/
checkpoints/
EOL

# Verify the environment is working
echo "Verifying conda environment..."
echo "Python location: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

echo "Setup completed successfully!"
echo "To activate the environment, run: source /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME"