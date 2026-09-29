$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $projectPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
}
& $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
