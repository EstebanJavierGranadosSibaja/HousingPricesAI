#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "No existe .venv. Ejecuta primero: ./scripts/setup_env.sh"
  exit 1
fi

if [[ ! -f "models/linear_regression.joblib" || ! -f "models/random_forest.joblib" ]]; then
  if [[ ! -f "data/raw/train.csv" ]]; then
    echo "No existe data/raw/train.csv. Incluye el dataset en el repo o copialo antes de ejecutar la app."
    exit 1
  fi
  echo "[run] Modelos no encontrados. Entrenando..."
  .venv/bin/python -m src.train
fi

.venv/bin/python -m streamlit run app/streamlit_app.py "$@"
