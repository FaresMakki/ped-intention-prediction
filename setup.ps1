$ErrorActionPreference = "Stop"

$deploymentRoot = $PSScriptRoot
$environmentPython = Join-Path $deploymentRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $environmentPython)) {
    python -m venv (Join-Path $deploymentRoot ".venv")
}

& $environmentPython -m pip install --upgrade pip
& $environmentPython -m pip install -r (Join-Path $deploymentRoot "backend\requirements-dev.txt")

Push-Location (Join-Path $deploymentRoot "frontend")
try {
    npm ci
    npm run build
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Run .\run.ps1 and open http://127.0.0.1:8000"

