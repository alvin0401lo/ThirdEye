#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Environment missing. Run ./setup_orangepi.sh first."
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec .venv/bin/python -m thirdeye "$@"
