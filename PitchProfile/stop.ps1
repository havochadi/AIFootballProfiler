$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '.runtime/server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host 'No recorded background server. Use Ctrl+C for a foreground server.'
    exit 0
}
$recordedPid = [int](Get-Content -LiteralPath $pidFile)
$serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$recordedPid"
if ($serverProcess) {
    if ($serverProcess.CommandLine -notlike "*$PSScriptRoot*.venv*python.exe*run.py*") {
        throw 'The recorded process is not this project server. It has been left untouched.'
    }
    # The Windows venv launcher starts a Python child; stop the owned tree.
    & taskkill /PID $recordedPid /T /F
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the background server.' }
}
Remove-Item -LiteralPath $pidFile
Write-Host 'The recorded background server is stopped.'
