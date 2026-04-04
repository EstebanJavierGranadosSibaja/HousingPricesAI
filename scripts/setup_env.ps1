$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] Creando entorno virtual .venv ..."
    python -m venv .venv
}

Write-Host "[setup] Instalando dependencias ..."
& $venvPython -m pip install -U pip
& $venvPython -m pip install -r requirements.txt

Write-Host "[setup] Listo. Python activo recomendado: $venvPython"
Write-Host "[setup] Ejecuta luego: .\\scripts\\run_app.ps1"
