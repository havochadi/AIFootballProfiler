# SoccerNet and football-player-tracking integration

> **Update, 8 October 2026.** The short-clip path described in parts of this document was removed: the upload dialog, the short-interval analysis, manual calibration and the YOLO11n detector. The app analyses whole halves and whole matches only, with the YOLOv8x pipeline. This document is kept as the record of what was built and checked at the time.

Implemented and checked on 24 September 2026. Open the running app at
http://127.0.0.1:8000. The new controls are in **Data and exports → SoccerNet
video library** and **Movement and tactics**.

## What the repositories provide

| Repository/component | Contents | Training status |
| --- | --- | --- |
| [SoccerNet](https://github.com/SoccerNet/SoccerNet/tree/26e8e46f8258e306fdbf019540e1eda4221a863d) | Dataset downloaders, official match splits, labels/features and task evaluators, including tracking, action spotting, calibration, re-identification and jersey recognition | The core package is a data/evaluation toolkit, not a bundled end-to-end player or archetype model. Downloaded reference annotations and pre-extracted features can be used immediately; training a task model is a separate step. |
| [football-player-tracking](https://github.com/DA-Shaurya/football-player-tracking/tree/d25079981c9a063be8e5bf32933a3bee864a7e1e) | YOLO fine-tuning, MOT conversion, BoT-SORT experiments, ORB camera compensation, jersey clustering, pitch mapping, kinematics, team geometry, heatmaps, visualisation and evaluation scripts | Its author describes a YOLOv8s detector fine-tuned for 20 epochs. The checked repository's `models` directory contains only `.gitkeep`; the football checkpoint is not distributed there. Reproducing that detector requires training or obtaining the checkpoint. |
| BoT-SORT configuration | Motion association with ORB camera compensation; appearance ReID disabled | No additional supervised training needed for this configuration. |
| Kit grouping | Torso colours, grass masking and clustering with temporal voting | Fits unsupervised clusters to the current footage. It does not know team names, player identities or referee/goalkeeper roles. |
| PitchProfile's active detector | Existing local YOLO11n COCO weights | Already pretrained and usable now; not SoccerNet-fine-tuned. |
| PitchProfile's 42-role classifier | Existing reviewed-interval training pipeline | Still needs real archetype reviews. Neither repository supplies the project's 42-role labels or a compatible trained classifier. |

The two upstream revisions are pinned above. Adapted configuration and torso
sampling are attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Upstream benchmark scores are not PitchProfile results.

## Implemented in this checkout

- **External SoccerNet library:** scans the six downloaded 720p halves on D:.
  Analyse a chosen start and duration without uploading or copying a full half.
  Supports 0.5–15 samples/second and up to 3,600 seconds per job.
- **Official annotations:** downloaded action and camera labels for all three
  local matches. Event buttons choose an analysis start ten seconds before the
  event. Match events remain separate from player-attributed evidence.
- **Provenance and splits:** all three current fixtures are official training
  matches. Their halves and derived clips share a match ID and keep that split.
  Validation/test partitions cannot be manufactured by splitting these clips.
- **Tracker choice:** BoT-SORT with ORB is the new default; ByteTrack remains
  selectable for library comparisons. Known SoccerNet cuts and detected image
  changes reset identities. No identity is claimed across a cut.
- **Kit suggestions:** masked torso CIELAB observations, CUDA K-means and a
  per-track vote. Anonymous groups and vote shares appear in Movement and
  tactics. Identity and team confirmation remain explicit review steps.
- **Movement:** observed distance, mean/peak segment speed, peak absolute
  acceleration, intensity durations and a speed chart. Gaps, track changes,
  period changes, invalid pitch positions and speeds above 12 m/s are excluded.
  Unavailable physical measurements remain unknown.
- **Tactics:** common-pitch radar, team centroids, visible outfield width,
  length and convex hull. A conservative instantaneous line-grouping heuristic
  requires ten outfield players with position groups and one known direction;
  it is not a verified formation classifier.
- **Occupancy:** CUDA histograms and Gaussian smoothing for each confirmed
  team, plus a difference map when two teams are present. These show observed
  presence, not possession or pitch control. Older PFF manifests were augmented
  with directions recovered from original metadata; their coordinates were
  not changed.
- **Review overlays:** optional regeneration with kit colours, short image
  trails and calibrated speed labels. NVENC creates the browser MP4 on D:.
  Image trails include camera motion and must not be read as metric paths.
- **Interoperability:** MOT export preserves source frame numbers, confidence,
  track-ID mapping and sampling metadata. A standalone adapter runs official
  TrackEval HOTA, CLEAR/MOTA and Identity/IDF1 on matching local ground truth.
- **Detector preparation/training:** explicit MOT-person to YOLO conversion
  preserves separate input sequence splits and empty frames. Images use hard
  links where supported. CUDA fine-tuning saves new weights on D: and never
  silently replaces the active detector.

The existing landmark/ORB pitch calibration is retained. The upstream demo's
hard-coded Liverpool homographies are not transferable to Chelsea–Burnley or
other matches. Raw-video speed, distance and pitch maps therefore require
reviewed calibration. Existing calibrated provider/reference data can display
these metrics immediately.

SoccerNet's other task datasets and evaluators are research infrastructure:
this change does not claim to implement automatic jersey-number reading,
named-player recognition, action recognition, captioning or end-to-end game
state reconstruction. It does not assign those capabilities to the pretrained
person detector. The acquired action labels are manual annotations, not model
predictions.

## Data and clocks

| Location | Contents |
| --- | --- |
| `D:\CVDL Football Data\SoccerNet\videos-720p` | Three source matches, six halves |
| `D:\CVDL Football Data\SoccerNet\labels` | Matching official action/camera JSON |
| `D:\CVDL Football Data\SoccerNet\official_splits.json` | Official SoccerNet partitions |
| `D:\CVDL Football Data\PitchProfile\data` | Prepared sources, reviews, derived tracks/overlays and model outputs |
| `D:\CVDL Football Data\PitchProfile\evidence\repository-integration` | Browser screenshots and report |
| `D:\CVDL Football Data\Archives\PitchProfile_Project.zip` | Preserved original project snapshot |

The video library uses half-relative source seconds. A processed interval uses
a clock starting at zero, with `source_offset_s` recording its source start.
Frame retrieval, calibration and independent-review playback apply this
offset. MOT files retain the original frame counter and include a metadata
file explaining sparse sampling. For benchmark evaluation, ground truth must
use the same frames, coordinate convention and person classes.

Override the library root with `PITCHPROFILE_SOCCERNET` if moving it later.
Application storage still uses `.data-location`/`PITCHPROFILE_DATA`.
Source weights and code remain in the project; football data live on D:.

Cleanup removed **1,908 byte-identical runtime data files (753,374,964 bytes)**
after SHA-256 comparison with retained D: files. The **16,200,228-byte** original
project ZIP was copied to D:, hash-verified and then removed from the workspace.
Unique outputs, code and credentials were preserved. The full deletion audit
is `D:\CVDL Football Data\PitchProfile\evidence\duplicate_cleanup.json`.

## Commands

Run these from the `PitchProfile` directory; activation is unnecessary.

```powershell
# Refresh official labels for downloaded matches.
.\.venv\Scripts\python.exe scripts/sync_soccernet_labels.py

# Prepare an acquired MOT person-annotation dataset. Class IDs must match its documentation.
.\.venv\Scripts\python.exe scripts/prepare_detector_data.py --help

# Fine-tune after preparing reviewed, separate training/validation sequences.
.\.venv\Scripts\python.exe scripts/train_detector.py --data "D:\CVDL Football Data\PitchProfile\data\detector-training\data.yaml" --epochs 20

# Optional evaluator dependency, already installed in this environment.
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-tracking-eval.txt
.\.venv\Scripts\python.exe scripts/evaluate_tracking.py --help
```

Training raw video alone is insufficient: detection training needs matching
image boxes; tracking evaluation needs ground-truth identities. The three raw
matches and their action labels do not provide those annotations. The detector
training command is available but no new football detector has been trained
or activated in this integration. Set `PITCHPROFILE_DETECTOR_WEIGHTS` only after
evaluating an appropriate class-0 person checkpoint on separate matches.

CUDA is used for detector inference, neural training, colour clustering and
occupancy smoothing; NVENC handles output encoding. OpenCV decoding/drawing,
ORB/association, geometry, file operations and the official NumPy/SciPy tracking
evaluator use CPU operations. No neural inference silently falls back to CPU.

## Executed verification

- Full suite: **96 passed**; the subsequent validation-split regression also
  passed. One existing AnyIO deprecation warning remains.
- Existing browser suite: **11 checks passed**, including GPU upload, interval
  reviews, all 42 roles and the 390-pixel layout.
- New integration browser suite: **7 checks passed**, including six library
  halves, event selection, kit suggestions, missing-calibration handling,
  MOT download, enhanced overlay playback, original MKV playback, PFF tactics,
  GPU occupancy and mobile layouts. No JavaScript errors or HTTP 500 responses.
- Real Chelsea–Burnley first-half excerpt, source seconds **45–75**: **150
  sampled frames** at 5 Hz on the **RTX 3090**, BoT-SORT + ORB, two annotated
  cut resets and an NVENC overlay. The 138 output tracklets are detection
  sequences, not 138 unique football players; fragmentation remains visible.
- Official tracking evaluator checked on explicitly artificial perfect tracks.
  This validates the adapter, not real-match HOTA/IDF1. Its legacy NumPy aliases
  are adapted only within TrackEval modules; metric algorithms are unchanged.

No detection fine-tuning or real-match tracking-accuracy benchmark is claimed.
Calibration quality, identity continuity and archetype reliability still need
appropriate reviewed evidence. Existing real reviews were not modified.
