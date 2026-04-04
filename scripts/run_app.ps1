param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StreamlitArgs
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "No existe .venv. Ejecuta primero: .\\scripts\\setup_env.ps1"
}

$trainCsv = Join-Path $root "data\raw\train.csv"
$linearModel = Join-Path $root "models\linear_regression.joblib"
$rfModel = Join-Path $root "models\random_forest.joblib"

if ((-not (Test-Path $linearModel)) -or (-not (Test-Path $rfModel))) {
    if (-not (Test-Path $trainCsv)) {
        throw "No existe data/raw/train.csv. Incluye el dataset en el repo o copialo antes de ejecutar la app."
    }
    Write-Host "[run] Modelos no encontrados. Entrenando..."
    & $venvPython -m src.train
}

$argsToUse = @("run", "app/streamlit_app.py")
if ($StreamlitArgs) {
    $argsToUse += $StreamlitArgs
}

& $venvPython -m streamlit @argsToUse
