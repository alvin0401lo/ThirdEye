#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
PYTHON="${PYTHON:-python3.12}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Install it first, then rerun this script."
  exit 1
fi
"$PYTHON" - <<'PY'
import platform, sys
assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version
print(f"Python {sys.version.split()[0]} on {platform.machine()}")
PY

sudo apt update
sudo apt install -y python3.12-venv python3-dev build-essential \
  libgl1 libglib2.0-0 libportaudio2 portaudio19-dev espeak-ng

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -c constraints-orangepi.txt -r requirements-orangepi.txt
.venv/bin/python -m pip install --no-deps mediapipe==1.0.1
.venv/bin/python -m pip install --no-deps -e .

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo
echo "Install finished. Edit .env, set OPENAI_API_KEY and the API model IDs,"
echo "then set the Orange Pi LAN IP and matching device token in firmware/thirdeye_ai_device/thirdeye_ai_device.ino."
echo "Run ./run_orangepi.sh --device-test before starting voice control."
