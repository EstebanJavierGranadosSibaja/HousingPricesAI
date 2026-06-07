# HousingPricesAI - Ollama initialization script (Windows PowerShell)
# Usage: .\scripts\init_ollama.ps1  [-Model qwen3:8b]
param(
    [string]$Model = $env:OLLAMA_MODEL
)

# Asegurar que Ollama esté en PATH para esta sesión
$ollamaDir = "$env:LOCALAPPDATA\Programs\Ollama"
if ((Test-Path "$ollamaDir\ollama.exe") -and ($env:PATH -notlike "*$ollamaDir*")) {
    $env:PATH = "$ollamaDir;$env:PATH"
}

if (-not $Model) { $Model = "qwen3:8b" }

Write-Host "=== HousingPricesAI - Ollama Setup ===" -ForegroundColor Cyan
Write-Host "Modelo objetivo: $Model" -ForegroundColor White

# Verify Ollama is installed
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Ollama no encontrado en PATH." -ForegroundColor Red
    Write-Host "Descarga e instala Ollama desde: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "Luego ejecuta este script nuevamente."
    exit 1
}

Write-Host "Ollama encontrado: $(ollama --version)" -ForegroundColor Green

# Check if server is already running
$running = $false
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    $running = ($resp.StatusCode -eq 200)
} catch { }

if (-not $running) {
    Write-Host "Iniciando servidor Ollama en segundo plano..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
    Write-Host "Servidor iniciado." -ForegroundColor Green
} else {
    Write-Host "Servidor Ollama ya en ejecucion." -ForegroundColor Green
}

# Pull the model (skips download if already present)
Write-Host ""
Write-Host "Descargando/verificando modelo '$Model'..." -ForegroundColor Yellow
Write-Host "(La primera descarga puede tardar varios minutos segun la conexion)"
ollama pull $Model

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al descargar el modelo '$Model'." -ForegroundColor Red
    Write-Host "Alternativa ligera: .\scripts\init_ollama.ps1 -Model llama3.2:3b" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Configuracion completada." -ForegroundColor Green
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Cyan
Write-Host "  1. Copia .env.example a .env  (ya configurado para $Model)"
Write-Host "  2. Inicia la app:  streamlit run app/streamlit_app.py"
Write-Host "  3. Calcula un precio y haz clic en 'Explicacion IA'"