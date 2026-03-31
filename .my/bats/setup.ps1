Set-Alias clear Clear-Host
Clear-Host

if (-Not (Test-Path ".venv")) {
  Write-Host "Crate .venv..." -ForegroundColor Cyan
  python -m venv .venv
}

Write-Host "Activate .venv..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

$pipToolsInstalled = pip show pip-tools -q
if (-Not $pipToolsInstalled) {
  Write-Host "Install pip-tools..."  -ForegroundColor Cyan
  pip install pip-tools
}

if (Test-Path "requirements.txt") {
  Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Blue
  pip install -r requirements.txt
}
else {
  Write-Host "Not found requirements.txt" -ForegroundColor Red
}

Write-Host "Venv is already" -ForegroundColor Green
