#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Function to check if a command exists ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- Install uv if not present ---
if ! command_exists uv; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to the PATH for the current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi

# --- Create a virtual environment and install packages ---
echo "Creating virtual environment and syncing dependencies..."
uv venv
uv sync

echo "Setup complete. Virtual environment is ready and dependencies are installed."
echo "To activate the virtual environment, run: source .venv/bin/activate"