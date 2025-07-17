# Stop executing the script if an error occurs
$ErrorActionPreference = "Stop"

# --- Function to check if a command exists ---
function Command-Exists {
    param (
        [string]$Command
    )
    $Found = Get-Command $Command -ErrorAction SilentlyContinue
    return $null -ne $Found
}

# --- Install uv if not present ---
if (-not (Command-Exists "uv")) {
    Write-Host "uv not found. Installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
} else {
    Write-Host "uv is already installed."
}

# --- Create a virtual environment and install packages ---
Write-Host "Creating virtual environment and syncing dependencies..."
# uv venv
# uv sync

python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "Setup complete. Virtual environment is ready and dependencies are installed."
Write-Host "To activate the virtual environment, run: .\.venv\Scripts\activate"