# PitchProfile

PitchProfile watches football broadcast video and profiles players by playing
style. For every player it can see in a half, it produces a heatmap, distance
and speeds, time on the ball, passes, receptions, carries, take-ons, shots,
tackles, blocks, headers, interceptions, recoveries, pressures, off-ball runs
(in behind, into the box, overlaps, pressing) and more, plus a style profile
ranking each statistic against players in the same position. It does this from
the video alone. You rate some players' archetypes as percentages (a player can
be several archetypes at once), and a semi-supervised model estimates the
percentages for the players you did not label.

- **[FULL_MATCH_ANALYSIS.md](FULL_MATCH_ANALYSIS.md)**: how a half is analysed,
  every statistic's definition, measured accuracy and limitations.
- **[DETECTION_RESEARCH.md](DETECTION_RESEARCH.md)**: why the statistics were
  sparse, what the research literature does about it, what changed, and what remains.
- **[annotation/ANNOTATION_GUIDE.md](annotation/ANNOTATION_GUIDE.md)**: how to
  label archetype percentages.
- **[LOCAL_VERIFICATION.md](LOCAL_VERIFICATION.md)**: what was executed and measured on this machine.

Inference and training run on an NVIDIA GPU (`cuda:0`).

## Windows setup

Use Python **3.12 or 3.13** and an NVIDIA GPU with a CUDA-compatible driver.
From this directory:

```powershell
.\setup.ps1 -Dev
.\.venv\Scripts\python.exe scripts\fetch_football_models.py
.\start.ps1
```

Open **http://127.0.0.1:8000**. Stop with Ctrl+C. The setup script installs into
`.venv`; activation is unnecessary. Use `-Python 'C:\path\to\python.exe'` if
`python` resolves to a different version. `-Dev` includes tests and browser
checks. The CUDA wheel download is approximately 3.5 GB. Setup never installs
into your global Python environment. `fetch_football_models.py` downloads the
detector, pitch-keypoint, shirt-number and action-spotting weights to the data
drive and checks their SHA-256. [MODELS.md](MODELS.md) lists every model.

For a background server, run `.\start.ps1 -Background`; stop it with
`.\stop.ps1`. Logs are in `.runtime/server.stdout.log` and
`.runtime/server.stderr.log`.

If PowerShell script execution is restricted, run these commands directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps "git+https://github.com/baudm/parseq.git@1902db043c029a7e03a3818c616c06600af574be"
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe run.py
```

`imageio-ffmpeg` supplies FFmpeg when it is absent from PATH. A deliberate
`PITCHPROFILE_DEVICE=cpu` override exists for other machines; it is never
selected automatically.

## Analyse matches

The local SoccerNet library is 23 complete matches (46 halves) of 720p broadcast
video in `D:\CVDL Football Data\SoccerNet\videos-720p`, across the Premier
League, La Liga, Serie A, Bundesliga, Ligue 1 and the Champions League. The
**working set** is half of them: 11 matches (22 halves) listed in
`D:\CVDL Football Data\SoccerNet\working_set.json`, keeping the official
train/validation/test split and league spread. Only these are analysed, trained
on and shown. Results for the other halves are parked in
`D:\CVDL Football Data\Archives\soccernet-outside-working-set`.

```powershell
.\.venv\Scripts\python.exe scripts\analyse_matches.py                 # every working-set half not yet analysed
.\.venv\Scripts\python.exe scripts\extract_crops.py                   # more player thumbnails per tracklet
.\.venv\Scripts\python.exe scripts\spot_actions.py                    # video model for tackles, blocks, headers, ...
.\.venv\Scripts\python.exe scripts\analyse_matches.py --postprocess-only   # re-run CPU stages only
```

A half takes about 30 minutes on the RTX 3090. The same analysis can be started
from **Matches** in the app. Open a ready half with **Label players**. The review
desk keeps your player queue, footage and role ratings together. Match context
contains team naming; identity corrections, statistics and events are available
beside the footage. Use **Identify player** to compare footage with searchable player cards.
Advanced track merging is kept in a separate collapsed section. **Stats & maps** brings the style profile (percentiles against same-position players), movement, physical statistics and event maps together; player comparisons are in **Matches**.

To add matches, `scripts\download_soccernet_720p.py --sample N --download`
fetches N more games (720p halves only) with your Hugging Face login.

## Profile archetypes

1. In **Label players**, select a match and use the team / review-status filters
   to work through the queue. Watch the highlighted player, use the green
   appearance timeline or event timestamps to seek, and bookmark useful moments.
2. Choose a position group, then expand a role. All 42 roles have two examples
   beneath the definition: sourced player comparisons and illustrative sequences.
   One role expands at a time. Use the evidence cues, quick 0 / 50 / 100 buttons
   or the precise slider. **Unknown**
   means unobserved; **0** is a deliberate rating. The full catalogue has 42 roles.
3. **Save & next** saves the ratings, notes and timestamped evidence together.
   Edits survive player changes and reloads as drafts on this browser. Drafts
   enter the training dataset only when you save. Review diverse appearances
   across matches and position groups; use evaluation to judge label sufficiency.
4. **Models → Fit on all analysed players** spreads the labels to
   the unlabelled players (label spreading over a nearest-neighbour graph of
   their statistics). It reports grouped-by-match cross-validation against a
   labelled-only baseline.
5. Estimates appear in **Models → Estimated profiles** and behind **Reveal model
   estimate** in the review desk. Rate independently before revealing suggestions.

**More tools → Independent interval reviews** is available for stricter work: two
independent reviewers rate a fixed 20-minute interval, disagreements go to
adjudication, and a supervised model trains on agreed cases (see the annotation
guide). The catalogue is under **More tools → 42-role guide**. Movement, source
corrections and interval-model evaluation are also in **More tools**.

Keyboard shortcuts outside form fields: **Space/K** play or pause, **J/L** seek
five seconds, **B** bookmark. **Ctrl/⌘+Enter** saves and advances. On smaller
screens, the assessment stacks below the footage. **Matches → Exports & tracking
data import** exports saved player labels with their evidence bookmarks.

UI verification: `python scripts/check_labelling_ui.py` exercises real match
footage with disposable annotation data, including drafts, failed saves, rapid
player changes, bookmarks, pitch event maps, historical line-ups, shirt-number
corrections, locally cached portraits and mobile layouts. It does not start model inference.

## Explore events and identify players

In **Stats & maps**, choose **Movement** for the movement heatmap, physical statistics
and positioning. Choose **Passes**, **Shots** or another event type for its statistics
and map. Whole-half statistics and filtered event counts are labelled separately.

The event views provide outcome-coloured origins / pass routes and origin
heatmaps for ten event types. Filter by outcome and video time, select a point,
watch the footage, and save a reviewed outcome. Goals, saved shots, blocks and
misses remain unknown until confirmed; the detector currently recognises attempts.

**Match centre** connects historical ESPN results and line-ups. Confirm which
actual team wears each detected kit, then shirt numbers resolve to match-day
names and available portraits. Use **Player identity** to correct a misread number.
Corrections preserve original track IDs and annotations and survive reprocessing.

All 23 local fixtures have cached historical context. Run
`python scripts/sync_match_context.py` to fetch context and available images for
newly downloaded fixtures. The data remains on D:. ESPN is an undocumented public
feed; cached data remains usable if the service is unavailable. These sourced
records assist review and are not inputs to the video-only models.

## Other data sources

**SoccerTrack v2** match 117092 (panoramic video with annotated positions, ball
and ball actions) is imported and serves as ground truth for the event logic
(`scripts\evaluate_events_soccertrack.py`). The other nine SoccerTrack v2
matches (about 132 GB) can be fetched with
`scripts\download_soccertrack.py --match <id> --download` after
`scripts\download_soccertrack.py --login`. They have not been downloaded. Enter
the token only at the terminal prompt. It is stored in the ignored
`.runtime/huggingface` folder.

**PFF FC 2022 World Cup** tracking is no longer used, because the project does
not use provider metadata. The source archives were deleted. Datasets derived
from them earlier are parked in `D:\CVDL Football Data\Archives\pff-wc2022`
(see its README to restore or delete them). The importer
(`scripts\import_pff.py`) remains in the code.

**Short clips** can still be uploaded under **Data and exports → Analyse a
video**, with manual calibration and identity confirmation. This is the earlier
workflow, used before automatic full-match analysis existed.

## Verify and maintain

```powershell
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_browser.py
.\.venv\Scripts\python.exe scripts/check_repository_ui.py       # needs the app running and one analysed half
.\.venv\Scripts\python.exe scripts/evaluate_events_soccertrack.py
.\.venv\Scripts\python.exe scripts/evaluate_jersey_reader.py
.\.venv\Scripts\python.exe scripts/train_shot_model.py
```

Tests use disposable data directories. The local `.data-location` file points
the app to `D:\CVDL Football Data\PitchProfile\data`. `PITCHPROFILE_DATA`
overrides it for a one-off run. Back up that folder for real work, especially
`annotations.sqlite`, which holds archetype labels and reviews. Reports and
screenshots are in `D:\CVDL Football Data\PitchProfile\evidence`. The service
binds to localhost and has no authentication.

## Import tracking data

Upload an uncompressed CSV and a JSON manifest in **Data and exports**. Export an
existing dataset's metadata as a starting example. Decompress exported
`tracks.csv.gz` before importing through the CSV input.

Required CSV columns: `frame,time_s,player_id,x,y,detected`.
Frames are nonnegative integers; timestamps are finite nonnegative seconds.
There may be only one row per player/frame. `detected` is `1` for observed,
`0` for estimated, or `-1` for unknown. Missing coordinates may be blank;
infinite coordinates are invalid. Only observed, calibrated, in-pitch points
enter heatmaps. Optional columns include `track_id`, `period`, and
`calibration_valid` (0 or 1).

Minimal metadata example (replace values with your actual source and interval):

```json
{
  "title": "Reviewed clip",
  "match_id": "your-fixture-id",
  "date": null,
  "source": "Your tracking provider or reviewed annotation source",
  "sampling_hz": 5,
  "duration_seconds": 60,
  "players": [{
    "player_id": "p1",
    "name": "Unconfirmed player",
    "team": "Unconfirmed",
    "role": "Center Forward",
    "eligible_frames": 300,
    "identity_verified": false,
    "direction_known": false
  }]
}
```

Coordinates use a 105 by 68 metre pitch. Set `direction_known: true` only when
coordinates have already been normalised so that the player attacks right.
The importer validates data but does not infer identities or rotate coordinates.

### Identify a player while watching

1. Click **Identify player** below the footage. The picker shows the paused player
   image and example crops alongside the choices.
2. Choose the actual team, then search by **shirt number or name**. Click the
   correct player card and **Confirm**. Photos, names and numbers come from that
   match’s line-up; only players who took part are selectable. Kit A / B setup is
   not required and confirming an individual does not change the whole team's mapping.
3. If you do not know their name, use **I only know their number or role**. Choose
   outfield player / goalkeeper and optionally enter a number. **This is not the
   goalkeeper** opens this quick correction directly.

Your reviewer name is remembered, and names / notes are optional in quick mode.
The picker closes after a successful save. **Change player** lets you revise the
choice or **Restore model identity**. Cancel and Escape leave the identity unchanged.

The correction applies to all appearances grouped under the selected track in this
half; it does not redraw boxes or split a track that contains multiple people.
Footage position, bookmarks and ratings are retained. Conflicting goalkeeper /
outfield ratings need review before they can be saved or used by the next model fit.
