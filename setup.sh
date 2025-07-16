#!/bin/bash

set -e  # Exit immediately on error

echo "🔧 Updating package lists..."
apt update

echo "🐍 Installing Miniconda and tmux..."
apt install -y wget curl tmux

# Download and install Miniconda
echo "📦 Installing Miniconda..."
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"

wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"
bash "$MINICONDA_INSTALLER" -b -p /root/miniconda3

# Add conda to PATH
echo "🛠 Adding conda to PATH..."
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> /root/.bashrc

# Initialize conda for bash
/root/miniconda3/bin/conda init bash

# Source the updated bashrc to make conda available in current session
source /root/.bashrc

# ✅ Add tmux scroll config
echo "🛠 Configuring tmux for smooth scrolling..."
cat <<EOF > /root/.tmux.conf
set -g mouse on
set -g terminal-overrides 'xterm*:smcup@:rmcup@'
EOF

# ✅ Add auto-tmux to .bashrc (if not already there)
if ! grep -q "auto_\$(date +%s%N)" /root/.bashrc; then
    echo "🛠 Setting up auto-tmux per terminal in ~/.bashrc..."
    cat << 'EOF' >> /root/.bashrc

# Automatically create a new tmux session for each new terminal
if command -v tmux &> /dev/null && [ -z "$TMUX" ]; then
  SESSION_NAME="auto_$(date +%s%N)"
  tmux new-session -s "$SESSION_NAME"
fi
EOF
fi


# 🔁 Reload tmux config if already inside tmux
if [ -n "$TMUX" ]; then
    echo "🔁 Reloading tmux config..."
    tmux source-file /root/.tmux.conf
fi

# ✅ Start tmux session if not already inside one
if [ -z "$TMUX" ]; then
    echo "🟢 Starting new tmux session..."
    exec tmux new-session -A -s setup
fi

# 📁 Go to project directory
PROJECT_DIR="/root/blackjack"
echo "📂 Changing directory to $PROJECT_DIR..."
cd "$PROJECT_DIR"

# 📦 Install Python packages
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
if [ -f "$REQUIREMENTS" ]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    # Use conda to install packages, fallback to pip if conda doesn't have them
    while IFS= read -r package; do
        # Skip comments and empty lines
        [[ $package =~ ^[[:space:]]*# ]] && continue
        [[ -z "${package// }" ]] && continue
        
        echo "📦 Installing $package..."
        if conda install -y "$package" 2>/dev/null; then
            echo "✅ Installed $package via conda"
        else
            echo "📦 Installing $package via pip..."
            pip install "$package"
        fi
    done < "$REQUIREMENTS"
else
    echo "ℹ️ No requirements.txt found, skipping package installation"
fi

echo "✅ Setup complete. Scrolling and independent terminals are ready!"


#run this:

# Make it executable:
# chmod +x setup.sh

# Run it:
# ./setup.sh