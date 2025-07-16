#!/bin/bash

set -e  # Exit immediately on error

echo "🔧 Updating package lists..."
apt update

echo "🐍 Installing Python 3 and pip..."
apt install -y python3 python3-pip

echo "🧰 Installing tmux..."
apt install -y tmux

# ✅ Add tmux scroll config
echo "🛠 Configuring tmux for smooth scrolling..."
cat <<EOF > /root/.tmux.conf
set -g mouse on
set -g terminal-overrides 'xterm*:smcup@:rmcup@'
EOF

# Reload config if already inside tmux
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
if [ ! -f "$REQUIREMENTS" ]; then
    echo "❌ requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

echo "📦 Installing Python dependencies..."
python3 -m pip install -r "$REQUIREMENTS"

echo "✅ Setup complete. Scrolling and dependencies are ready!"


# Save it:
# nano setup.sh

# Make it executable:
# chmod +x setup.sh

# Run it:
# ./setup.sh