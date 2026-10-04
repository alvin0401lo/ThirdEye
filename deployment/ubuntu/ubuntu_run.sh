#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "First launch: preparing the environment..."
  bash ./deployment/ubuntu/ubuntu_setup.sh
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Environment setup did not create .venv/bin/python."
  exit 1
fi
if [[ " $* " == *" --voice-control "* ]]; then
  if [[ ! -f .env ]]; then
    cp .env.example .env
  fi
  prompt_env() {
    local name="$1" label="$2" secret="${3:-}" value
    if [[ "$secret" == "secret" ]]; then
      read -r -s -p "$label: " value
      printf '\n'
    else
      read -r -p "$label: " value
    fi
    if [[ -z "$value" ]]; then
      echo "A value is required for $name."
      return 1
    fi
    THIRDEYE_SETUP_VALUE="$value" .venv/bin/python - "$name" <<'PYTHON_SET_ENV'
import os
import sys
from dotenv import set_key
set_key(".env", sys.argv[1], os.environ["THIRDEYE_SETUP_VALUE"])
PYTHON_SET_ENV
  }
  if ! grep -Eq '^OPENAI_API_KEY=.+$' .env; then
    prompt_env OPENAI_API_KEY "OpenAI API key" secret || exit 2
  fi
  if grep -Eq '^THIRDEYE_OPENAI_MODEL=(|YOUR_ROUTING_MODEL)$' .env; then
    prompt_env THIRDEYE_OPENAI_MODEL "OpenAI command model ID" || exit 2
  fi
  if grep -Eq '^THIRDEYE_VISION_MODEL=(|YOUR_VISION_MODEL)$' .env; then
    prompt_env THIRDEYE_VISION_MODEL "OpenAI vision model ID" || exit 2
  fi
  if grep -Eq '^THIRDEYE_CAMERA_TOKEN=(|YOUR_PRIVATE_TOKEN)$' .env; then
    prompt_env THIRDEYE_CAMERA_TOKEN "Device token (must match firmware)" secret || exit 2
  fi
  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    if systemctl is-active --quiet thirdeye.service; then
      echo "ThirdEye voice control is already running."
      exit 0
    fi
    if ! systemctl is-enabled --quiet thirdeye.service; then
      service_user="${SUDO_USER:-$(id -un)}"
      project_root="$PWD"
      THIRDEYE_SERVICE_USER="$service_user" THIRDEYE_PROJECT_ROOT="$project_root" \
        .venv/bin/python - <<'PYTHON_SERVICE' | sudo tee /etc/systemd/system/thirdeye.service >/dev/null
import os
from pathlib import Path
text = Path("deployment/ubuntu/thirdeye.service").read_text(encoding="utf-8")
text = text.replace("@SERVICE_USER@", os.environ["THIRDEYE_SERVICE_USER"])
text = text.replace("@PROJECT_ROOT@", os.environ["THIRDEYE_PROJECT_ROOT"])
print(text, end="")
PYTHON_SERVICE
      sudo systemctl daemon-reload
      sudo systemctl enable --now thirdeye.service
      echo "ThirdEye is running and will start automatically after reboot."
      exit 0
    fi
    sudo systemctl start thirdeye.service
    echo "ThirdEye is running as a system service."
    exit 0
  fi
  echo "No system service manager is available; starting voice control in this terminal."
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec .venv/bin/python -m thirdeye "$@"
