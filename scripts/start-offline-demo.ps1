param([ValidateRange(1024,65535)][int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$taskPreviousOffline = [Environment]::GetEnvironmentVariable('FORCE_OFFLINE_MODE', 'Process')
$taskPreviousDotenv = [Environment]::GetEnvironmentVariable('PYTHON_DOTENV_DISABLED', 'Process')
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $env:FORCE_OFFLINE_MODE = 'true'
    $env:PYTHON_DOTENV_DISABLED = '1'
    $taskPython = if (Test-Path -LiteralPath '.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
    & $taskPython -m uvicorn app:app --host 127.0.0.1 --port $Port
} finally {
    [Environment]::SetEnvironmentVariable('FORCE_OFFLINE_MODE', $taskPreviousOffline, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHON_DOTENV_DISABLED', $taskPreviousDotenv, 'Process')
    Pop-Location
}
