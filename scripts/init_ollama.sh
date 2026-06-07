#!/usr/bin/env bash
# HousingPricesAI - Ollama initialization script (Linux/macOS)
# Usage: bash scripts/init_ollama.sh [qwen3:8b|llama3.2:3b]
set -e

MODEL="${1:-${OLLAMA_MODEL:-qwen3:8b}}"

echo "=== HousingPricesAI - Ollama Setup ==="
echo "Modelo objetivo: $MODEL"

# Install Ollama if not present
if ! command -v ollama &>/dev/null; then
    echo "Ollama no encontrado. Instalando desde ollama.com..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Ollama: $(ollama --version)"

# Start server if not running
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Iniciando servidor Ollama..."
    ollama serve &
    sleep 4
    echo "Servidor iniciado."
else
    echo "Servidor Ollama ya en ejecucion."
fi

# Pull model
echo ""
echo "Descargando/verificando modelo '$MODEL'..."
echo "(La primera descarga puede tardar varios minutos)"
ollama pull "$MODEL"

echo ""
echo "Configuracion completada."
echo ""
echo "Proximos pasos:"
echo "  1. cp .env.example .env"
echo "  2. streamlit run app/streamlit_app.py"
echo "  3. Calcula un precio y haz clic en 'Explicacion IA'"
