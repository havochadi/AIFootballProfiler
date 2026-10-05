# Run PitchProfile

Use the existing **PitchProfile** folder. The original ZIP (archived on D:) is an
incomplete early snapshot, so do not re-extract it over the working copy.

**What is ready:**

- **Full-match analysis from broadcast video alone.** The pipeline detects and
  tracks players, calibrates the pitch automatically, clusters kits into teams,
  reconstructs the ball, reads shirt numbers and detects events.
- **Per-player statistics:** heatmap, distance, speeds and sprints, time on the
  ball, passes, progressive and key passes, crosses, carries, dribbles, shots,
  tackles, interceptions, recoveries, clearances and pressures, as totals and
  per 90 minutes.
- **Archetype labelling as percentages** (several of the 42 roles per player),
  with a semi-supervised model that estimates percentages for unlabelled players.
- **A local library** of 23 SoccerNet matches in 720p on `D:\CVDL Football Data`.
- **A redesigned scouting desk:** three main screens, video / event-map views,
  outcome filters, evidence bookmarks, recoverable drafts and Save & next.
- **Historical match context for all 23 fixtures:** sourced results and line-ups,
  kit-group confirmation, match-day shirt-number lookup and identity correction.
  Available portraits and badges are cached on D: for local display.

**What remains:**

- Label varied player appearances across matches and position groups, then fit
  and evaluate the archetype model. There is no fixed quota that ensures accuracy.
- Review automatic identities before labelling.
- Shots and crosses are the least accurate events; see the measured accuracy
  and limitations in the full-match guide.

Read [PitchProfile/FULL_MATCH_ANALYSIS.md](PitchProfile/FULL_MATCH_ANALYSIS.md)
for how it works and how accurate it is,
[PitchProfile/README.md](PitchProfile/README.md) for setup and commands, and
[PitchProfile/annotation/ANNOTATION_GUIDE.md](PitchProfile/annotation/ANNOTATION_GUIDE.md)
before labelling. [PitchProfile/LOCAL_VERIFICATION.md](PitchProfile/LOCAL_VERIFICATION.md)
records what was run on this machine.

## Launch (Windows, NVIDIA GPU)

From the `PitchProfile` folder, with Python 3.12 or 3.13:

```powershell
.\setup.ps1 -Dev
.\.venv\Scripts\python.exe scripts\fetch_football_models.py
.\start.ps1
```

Open **http://127.0.0.1:8000** and stop the server with Ctrl+C. Analyse matches
with `.\.venv\Scripts\python.exe scripts\analyse_matches.py` (about 30 minutes
per half on the RTX 3090), or from **Matches** in the app. Open a ready half with
**Label players**. Watch footage, bookmark evidence and use **Save & next** to
work through the queue. **Models** holds fitting and evaluation; **More tools**
holds the 42-role guide and independent interval reviews.

Keep a backup of `D:\CVDL Football Data\PitchProfile\data`, especially
`annotations.sqlite`, which holds your labels.
