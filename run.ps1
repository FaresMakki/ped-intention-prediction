$ErrorActionPreference = "Stop"

$deploymentRoot = $PSScriptRoot
$localPython = Join-Path $deploymentRoot ".venv\Scripts\python.exe"
$workspacePython = Join-Path (Split-Path $deploymentRoot -Parent) ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $localPython) {
    $python = $localPython
}
elseif (Test-Path -LiteralPath $workspacePython) {
    $python = $workspacePython
}
else {
    throw "No compatible virtual environment found. Run .\setup.ps1 first."
}

Push-Location $deploymentRoot
try {
    & $python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}

