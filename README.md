# AIFootballProfiler / PitchProfile

Turns football broadcast video into player statistics. The app tracks every
visible player and produces per-player statistics (heatmap, distance, passes,
dribbles, shots, defensive actions and more) and a percentile profile against
players in the same position. Detection, calibration and training use the NVIDIA
GPU.

On Windows, from this repository:

```powershell
cd PitchProfile
.\setup.ps1
.\.venv\Scripts\python.exe scripts\fetch_football_models.py
.\start.ps1
```

Open **http://127.0.0.1:8000**. Setup uses Python 3.12 or 3.13; pass
`-Python 'C:\path\to\python.exe'` to select an interpreter. No environment
activation or paid service is required.

See [the operating guide](PitchProfile/README.md) and
[the full-match analysis guide](PitchProfile/FULL_MATCH_ANALYSIS.md).
