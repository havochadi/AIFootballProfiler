param([string]$Python = "python", [switch]$Dev)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & $Python -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] <= (3, 13) else 'Use Python 3.12 or 3.13 for the pinned PyTorch wheels.')"
    if ($LASTEXITCODE -ne 0) { throw "Select Python 3.12 or 3.13 using -Python path/to/python.exe." }
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }
    & ./.venv/Scripts/python.exe -m pip install --upgrade torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
    $requirements = if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' }
    & ./.venv/Scripts/python.exe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    # PARSeq jersey reader, pinned; --no-deps keeps its CPU torch pin from replacing CUDA torch.
    & ./.venv/Scripts/python.exe -m pip install --no-deps "git+https://github.com/baudm/parseq.git@1902db043c029a7e03a3818c616c06600af574be"
    if ($LASTEXITCODE -ne 0) { throw "PARSeq installation failed (git is required)." }
    & ./.venv/Scripts/python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed." }
    & ./.venv/Scripts/python.exe scripts/doctor.py
    if ($LASTEXITCODE -ne 0) { throw "Runtime checks failed. See the diagnostics above." }
    Write-Host "Ready. Run .\start.ps1 and open http://127.0.0.1:8000"
} finally {
    Pop-Location
}
