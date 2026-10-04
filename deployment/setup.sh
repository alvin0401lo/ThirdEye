#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3.12}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Install it first, then rerun this script."
  exit 1
fi
"$PYTHON" - <<'PYTHON_CHECK'
import platform, sys
assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version
print(f"Python {sys.version.split()[0]} on {platform.machine()}")
PYTHON_CHECK

sudo apt update
sudo apt install -y python3.12-venv python3-dev build-essential \
  libgl1 libglib2.0-0 libportaudio2 portaudio19-dev espeak-ng

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -c constraints.txt -r requirements.txt
.venv/bin/python -m pip install --no-deps mediapipe==1.0.1
.venv/bin/python -m pip install --no-deps -e .

mkdir -p models
.venv/bin/python - <<'PYTHON_MODEL'
from pathlib import Path
from urllib.request import urlopen

model = Path("models/hand_landmarker.task")
if not model.exists():
    temporary = model.with_suffix(".task.part")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            output.write(response.read())
        temporary.replace(model)
    finally:
        temporary.unlink(missing_ok=True)
PYTHON_MODEL
if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo
echo "Install finished. Edit .env and set OPENAI_API_KEY and model IDs."
echo "Set the device address and matching token in firmware/thirdeye_ai_device/thirdeye_ai_device.ino."
echo "Run ./deployment/run.sh --device-test before starting voice control."
