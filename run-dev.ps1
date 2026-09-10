$ErrorActionPreference = "Stop"

$deploymentRoot = $PSScriptRoot
$localPython = Join-Path $deploymentRoot ".venv\Scripts\python.exe"
$workspacePython = Join-Path (Split-Path $deploymentRoot -Parent) ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $localPython) { $localPython } else { $workspacePython }

if (-not (Test-Path -LiteralPath $python)) {
    throw "No compatible virtual environment found. Run .\setup.ps1 first."
}

$backend = Start-Process `
    -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory $deploymentRoot `
    -PassThru `
    -WindowStyle Hidden

Push-Location (Join-Path $deploymentRoot "frontend")
try {
    npm run dev
}
finally {
    Pop-Location
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id
    }
}

