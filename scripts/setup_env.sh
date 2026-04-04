#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[setup] Creando entorno virtual .venv ..."
  python3 -m venv .venv
fi

echo "[setup] Instalando dependencias ..."
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

echo "[setup] Listo. Ejecuta: ./scripts/run_app.sh"
