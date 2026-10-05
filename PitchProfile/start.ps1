param([int]$Port = 8000, [switch]$Background)
$ErrorActionPreference = "Stop"
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Run .\setup.ps1 first." }
if ($Background) {
    $runtimePath = Join-Path $PSScriptRoot '.runtime'
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    $pidFile = Join-Path $runtimePath 'server.pid'
    if (Test-Path -LiteralPath $pidFile) {
        $recordedPid = [int](Get-Content -LiteralPath $pidFile)
        $recordedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$recordedPid"
        if ($recordedProcess -and $recordedProcess.CommandLine -like "*$PSScriptRoot*run.py*") {
            throw 'A background server is already recorded. Run .\stop.ps1 before restarting it.'
        }
    }
    $serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('run.py', '--port', $Port) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'server.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'server.stderr.log') -PassThru
    $serverProcess.Id | Set-Content -LiteralPath $pidFile
    Write-Host "Started background server at http://127.0.0.1:$Port (launcher PID $($serverProcess.Id)). Stop with .\stop.ps1."
    exit 0
}
& $pythonPath (Join-Path $PSScriptRoot 'run.py') --port $Port
exit $LASTEXITCODE
