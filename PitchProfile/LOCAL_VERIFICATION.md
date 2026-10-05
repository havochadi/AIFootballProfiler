# Local verification

## 30 September 2026: who is who (identity model, grouping, legible-thumbnail naming)

Measurements and reasoning: [DETECTION_RESEARCH.md](DETECTION_RESEARCH.md), section 6.

| Check | Result |
| --- | --- |
| Held-out FOOTPASS games (3 games, 6 halves, true boxes and numbers), whole pipeline | Right player for **49.8%** of visible player time (26.2% on 29 Sept), **94.9%** of given identities right (49.2%), goalkeepers 89% (45%); 34.4% of visible time in consistent unnamed players (90% pure); only 2.8% detected but in no player (36.6%) (`evidence/identity_footpass_final.json`) |
| Actions credited to the right named player (same games) | Passes 17.1% → **31.6%**, carries 17.2% → **32.1%**; to the right true player's track (named or not) 58–59% |
| SoccerNet tracking clips (49 clips, true numbers) | Right number for 71.8% of player time, 6.8% wrong (previous rule 49.0% / 3.7%) (`scripts/simulate_identity_sn_tracking.py`) |
| Why players were split | 94% of a player's split-off time was blocked by the "never on screen together" rule (tracker carry-overs); tolerating 15 s of shared time: main-group share 63% → 89%, purity unchanged |
| Why numbers were wrong | Number heads trained on every thumbnail guessed numbers from unreadable views; legible-thumbnail votes 54% → 76% right; retrained on legible thumbnails only 76% → 88% (hardest kit 42–53% → 73%) |
| Identity model retrain (`scripts/train_identity_models.py --legible-only --jersey`) | 6,000 steps, about 40 minutes plus 1.5 h scoring the legibility of 2.25 M training thumbnails; appearance 98.2% of held-out tracklets, goalkeepers 90% recall / 99% precision; previous weights kept as `model_v2_all_crops.pt` |
| Goalkeeper calls on SoccerNet kits | Yellow kits (Villarreal, Barcelona) were called goalkeepers for most of their players (up to 95% of a team's detections). Safeguards in `match_pipeline.override_keepers`; now 1–7% per team on 43 of 44 team-halves (13% on one) |
| Crediting spotted passes | Wider windows, looser boxes, possession fallback: +1–3 points each at a similar precision loss; not adopted (no ball found near the moment is the limit) |
| Legacy reader cache | Per-thumbnail legibility and PARSeq readings cached (`legacy_readings.npz`): re-running post-processing on a half 10 min → 1 min |
| All 22 working-set halves post-processed with the identity model (now the default reader) | Detected outfield time in a player (named or consistent unnamed) 49.8% → **87.5%**; named players' share of tracked time median 59% (18–75% per half; striped kits lowest), 21 named players per half; per team per half: passes 168 → 249, carries 85 → 175; players with no header 46% → 18%, no interception 24% → 16%, no block 19% → 13% (`evidence/coverage_diagnostics.json`, snapshot `identity_v3`) |
| FOOTPASS full-HD training archives 03–05 | Downloaded to D: (106 GB); unpacking needs the NDA password at a terminal |
| Name players panel (match view) | Unnamed players listed by screen time with pictures and the number reader's best guess; typed numbers or "not a player" marks applied in one re-analysis (`/match/identities/names`), stored per footage segment so they survive re-analysis, undoable (`/match/identities/undo`). On Ajax v Barcelona H1: 19 unnamed players, the largest read as #16 in 84% of 1,207 readable views |
| Windows file sharing | A status request during a re-analysis failed once (a manifest being replaced); JSON reads and replaces now retry sharing violations |
| Tests | `pytest`: **162 passed** (new: overlap tolerance, motion links, legacy-guarded and vote naming, jersey aggregation, goalkeeper safeguards, reviewer names and not-a-player marks) |
| Browser checks (app restarted on the new code) | Full-match repository: 10 checks passed (new: name players, with a number and a not-a-player mark applied and undone on real data), no page or server errors; labelling desk: 18 checks passed |

## 29 September 2026: fine-tuned action spotter, 1080p thumbnails, identity linking

Measurements and reasoning: [DETECTION_RESEARCH.md](DETECTION_RESEARCH.md), section 5.

| Check | Result |
| --- | --- |
| SoccerNet NDA data | SN-BAS-2025 (7 labelled games, 720p) and FOOTPASS (51 games, 640×352 video + tactical labels) unpacked on D: with the user's password entered at a terminal prompt; packed at 12.5 frames/s into `SoccerNet\spotter_frames` (200 GB; the zips and unpacked videos take another 86 GB, the 1080p broadcasts 32 GB; 523 GB of D: still free) |
| Label conventions checked on data | FOOTPASS class ids 4–6 are Throw-in, Shot, Header (not the order on its page); `left_to_right = 0` is the team defending the left goal (goalkeeper at x ≈ 0.1); the pipeline's own side mapping matched possession for 96% of confident passes and carries |
| Published spotter, held-out labelled games | BAS test: tackles 0.09 / 0.14 precision / recall, blocks 0.28 / 0.40, team right about half the time for both |
| Fine-tuned spotter (`scripts/finetune_action_spotter.py`, 3,000 steps, about 3 h) | Mean average precision 0.61 → 0.63 (BAS test), 0.48 → 0.60 (FOOTPASS validation); blocks 0.18 → 0.34, headers 0.64 → 0.76, tackles 0.02 → 0.10. On the 22 halves against SoccerNet labels: shot sums 411 → 261 (275 labelled). Now the pipeline's spotter |
| Tackle and block team | Opposite of the last confident touch: blocks right 81% / 97% (BAS / FOOTPASS with the published model; 78% / 97% with the fine-tune) against the model's own 48–78% |
| Free kicks | No longer counted: recall 4–20% against SoccerNet labels with either model |
| 1080p thumbnails | 3 of 11 games are true 1080p. Frames verified by picture matching (counted from the start of each file; one file's seek landed 7 frames off). Kept for Dortmund and Villarreal (identified share up 0.7–3.1 points per half); Chelsea v Burnley reverted to 720p (fewer numbers read on its interlaced broadcast) |
| Identity linking (`scripts/evaluate_linking.py`) | GTA-style appearance clustering 6–13% accurate on hidden numbers; current rule 91%. Rule unchanged |
| All 22 halves re-spotted and post-processed | Blocks 3.2 → 11.8 per team per half (reference 12.2), shots 5 → 6 (6.2), lofted passes 6.9 → 14.4, headers 3.2 → 14.0, tackles 1.5 → 2.2; identified player-time 49.1% → 49.8% |
| Tests | `pytest`: **147 passed** (new: spotter labels and partial-label loss, vectorised head identical to the devkit, won-ball side) |
| Browser checks | Labelling desk passed; full-match repository 9 checks passed, no page or server errors |

## 28–29 September 2026: working set, video action spotting, re-identification, more statistics

The research and reasoning are in [DETECTION_RESEARCH.md](DETECTION_RESEARCH.md); definitions are in
[FULL_MATCH_ANALYSIS.md](FULL_MATCH_ANALYSIS.md).

| Check | Result |
| --- | --- |
| Working set | 11 of 23 matches (22 of 46 halves) in `SoccerNet\working_set.json`: 7 train, 2 validation and 2 test matches, 6 competitions. The other 24 analysed halves were moved to `Archives\soccernet-outside-working-set` (no labels referenced them; README there to restore) |
| Shot model retrained on the working set | Cross-validation over 9 matches 0.40 precision / 0.40 recall (was 0.32 / 0.32 on 5); held-out test 0.36 / 0.47 |
| Tackle/block rules on SoccerTrack ground truth | Tackles 0.06–0.14 precision, 0.08–0.12 recall; blocks ≤ 0.10. A contact-based redesign was tried and **reverted**: annotated tackles and blocks (about 1% of contacts) could not be separated from other contacts |
| Action spotter (SoccerNet 2025 baseline, `evidence/action_spotter.json`) | Held-out test halves against SoccerNet labels: shots 0.78 / 0.63 (shot model 0.35 / 0.56), throw-ins 0.83 / 0.69, ball out of play 0.61 / 0.86. About 4.5 minutes per half on the RTX 3090 |
| Re-identification head (`evidence/reid_head.json`) | Held-out matches, hidden numbers: 72% correct (frozen CLIP 65%, role rule 43%); attachment thresholds give about 90% |
| Thumbnails re-cut | About 60,000 per half (was about 15,000), 0.6–0.7 minutes per half |
| Coverage (`evidence/coverage_diagnostics.json`) | Attributed player-time 32% → 49%; appearances with ≥ 10 identified minutes 100 → 218; statistics per player 39 → 90 |
| Tests | `pytest`: **140 passed** (new: learned-shot selection, appearance attachment, off-ball runs, style-profile percentiles) |
| Browser, labelling desk (`scripts/check_labelling_ui.py`) | **18 checks passed**, including the new style-profile view; test writes used isolated copies |
| Browser, full-match repository (`scripts/check_repository_ui.py`) | **9 checks passed**, no page errors or HTTP 5xx; the library lists the 22 working-set halves |

Honest limits: spotted tackles, blocks and crosses are expected counts well below real-match rates
(tackles about 1.5 vs 12 per team per half), and their player attribution has no local ground truth.
Take-ons are calibrated to plausible rates (7 attempts, 3 kept per team per half), not validated.

## 25 September 2026: guided player identification

- Replaced the two correction forms and method dropdown with **Identify player**:
  choose an actual team, search by shirt number / name, choose a photo card and
  confirm. The picker shows an enlarged paused player when a box is available,
  with example crops alongside. Only players who took part can be selected.
- Direct individual assignments work without kit binding and leave the global
  team mapping unchanged. Explicit individual assignments survive later kit
  remapping; legacy assignments retain their original mapping-dependent behaviour.
- The quick fallback asks for outfield / goalkeeper and an optional shirt number.
  Reviewer identity is remembered; extra names and notes are collapsed. Cancel,
  Escape, failed-save recovery and restoring the model identity are supported.
  Advanced track merging stays outside the identification picker.
- **28 backend tests passed**, plus **18 isolated browser scenarios** in
  `.runtime/labelling-20260925-044145/report.json` and **6 live playback checks**.
  Additional read-only live checks covered the picker at four widths, visible
  confirmation controls, Rooney's searchable card, enlarged paused footage,
  cancellation, missing line-ups and simulated provider failures.
- The website was restarted. Tests did not change real player annotations or run
  detector inference. Disposable browser-test evidence was removed; reports and
  screenshots remain in `.runtime`.

## 25 September 2026: integrated statistics and footage identity correction

- **Stats & maps** now contains movement, physical / positional statistics and
  all event categories. Passing, shooting and other actions show their related
  statistics above the event map. Whole-half statistics are distinguished from
  filtered event counts; repeated third-of-pitch statistics were removed.
- **Wrong player?** and **This is not the goalkeeper** open one correction panel
  beside the footage. A reviewer can correct goalkeeper / outfield status, number
  and name without an online line-up, or select an actual match-day player.
  The correction applies to the selected track throughout this half; it does not
  split mixed identities or redraw detections. Restore model identity provides undo.
- Corrections are stored separately in the annotation database with reviewer,
  timestamp and history. Original evidence and measurements are unchanged. Saved
  labels / drafts survive correction; conflicting goalkeeper / outfield ratings
  require review and are excluded from the next fit. Outdated individual model
  estimates are withheld until a new fit. Line-up confirmation also updates role.
- **50 backend tests passed across focused runs**, plus **17 isolated browser
  scenarios** (`.runtime/labelling-20260925-041412/report.json`) and **6 live playback
  checks** (`.runtime/video-playback-chrome.json`). Live layout checks also covered
  1440, 1024, 768 and 390 px widths, passing details and correction controls.
- Website restarted with the new API. Real annotations were not changed by testing.
  Disposable browser-test evidence was removed after checks. The two experimental
  MP4 copies left from the earlier seek investigation were also removed from D:.

## 25 September 2026: archetype examples and interface cleanup

- All 42 roles now have two examples immediately after the definition, shared
  between the labelling desk, catalogue and interval reviews. Player comparisons
  include context and source links; the rare inverted-wing-back entry uses two
  explicitly illustrative sequences. These are teaching material, not labels or
  claims about events in the selected footage. Role IDs, compatibility and training
  eligibility are unchanged.
- One role card expands at a time. Removed nested rating-panel scrolling, duplicate
  play buttons, duplicate identity sections and the raw event list; reviewed events
  remain in Event maps. Player comparisons moved to Matches. Playback recovery
  controls remain available under Playback help and open automatically on errors.
- The hidden source-profile video now loads only when that screen opens; hidden
  videos pause on navigation. Failed match/player/evidence loads have an accessible
  retry action. Successful recovery clears the stale failure notification.
- Reproduced Chrome hanging on an exact 8.56-second seek in Ajax/Barcelona H1,
  including after an MP4 stream-copy experiment. A stalled seek now retries once
  1 ms later after 2.5 seconds, retaining the selected video frame. The final fix
  uses the original footage and needs no video conversion or new GPU inference.
- **46 automated tests passed** (cases, API, match analysis and match context), plus
  **15 browser scenarios** in `.runtime/labelling-20260924-233917/report.json` and
  **6 playback checks** in `.runtime/video-playback-chrome.json`. Browser test writes
  used isolated copies. The live catalogue serves 42 roles and 84 examples.
- Website restarted to load the catalogue. The separate GPU analysis batch remained
  running; no real annotations or original match evidence were altered.

## 25 September 2026: footage visibility and playback recovery

- All 30 currently analysed SoccerNet halves have source files on the data drive
  and decoded moving frames in Chrome through the live media endpoint. The reported
  playback failure could not be reproduced in that browser; the user's browser and
  selected half were not supplied.
- Added a top-level **Watch footage** button that switches from event maps, brings
  the video into view and starts playback. Loading and decoding errors now have
  **Reload footage** and **Open video** controls. Seeking before metadata arrives
  preserves the intended timestamp instead of losing it during source loading.
- `scripts/check_video_playback.py` passed five read-only browser scenarios,
  including a simulated media 404, recovery at the original timestamp, decoded
  playback and desktop/mobile layout. Evidence: `.runtime/video-playback-chrome.json`
  and `.runtime/video-audit.json`. No annotations or source footage were changed;
  the separate match-analysis process was left running.

## 25 September 2026: scouting desk, event maps and historical match context

- The main navigation is now **Label players**, **Matches** and **Models**.
  The darker scouting-room design retains the full 42-role catalogue and secondary
  review / correction tools. The browser suite passed its original nine labelling
  scenarios plus three integrated event-map / historical-identity scenarios.
- Final combined match-analysis, match-context, API, storage and interval-review
  regression run: **48 passed**, with one upstream AnyIO deprecation warning.
  Test writes use isolated annotation databases.
- Historical ESPN results and complete match-day squads fetched for **23/23**
  downloaded fixtures. Context and available portraits / badges are cached on D:.
- Browser evidence: `.runtime/labelling-20260924-165903/report.json` (timestamps
  in artifact directory names are UTC), including correct attack-direction mapping,
  filtered heatmaps, persistent reviewed outcomes, the 1–1 Chelsea–United fixture,
  historical #10 Rooney with a successfully loaded local portrait, and identity
  correction. No page errors or server 5xx.
- Test identity assignments and outcome labels use disposable copies; the
  user's real player labels, raw detections and match statistics were not rewritten.
- Shot outcomes remain unknown unless reviewed. Provider match-clock events are
  reference material rather than automatically aligned ground-truth video events.
- The separate `scripts/analyse_matches.py` batch was left running. This UI work
  does not require new detector inference or model training.

## 24 September 2026, evening: full-match analysis from video

What each component does is described in
[FULL_MATCH_ANALYSIS.md](FULL_MATCH_ANALYSIS.md). Everything below ran on the
RTX 3090 with the data on `D:\CVDL Football Data`.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | **114 passed** (one upstream AnyIO deprecation warning) |
| `scripts/check_browser.py` | **11 checks passed** |
| `scripts/check_repository_ui.py` (running app, real data) | **9 checks passed**, no page errors or HTTP 5xx: 46-half library, match statistics, identity box on video with event seeking, team naming, archetype label saved and reverted, learning tab, 390 px layouts |
| Batch analysis | 13 of 46 halves finished at the time of writing, at 15–25 sampled frames per second (about 30 minutes per half; slower while other jobs share the CPU). The batch continues in the background |
| Event logic, SoccerTrack v2 117092 ground truth | Passes precision 0.69/0.66, recall 0.70/0.69 (halves 1/2); per-player pass counts r = 0.89 in both halves (unchanged by this session's edits) |
| Shot classifier (`evidence/shot_model.json`) | Cross-validation over 5 matches: 105 shots called for 105 labelled, 34 correct (rule: 19 at 0.17 precision). Held-out test match: 27 called for 27 labelled, 11 correct (rule: 6). After re-processing, the 13 finished halves have 154 shots called for 154 labelled |
| Shirt-number reader | 80.3% tracklet accuracy on the SoccerNet Jersey 2023 test split |
| Attack direction | Correct in all 8 halves with a calibrated kick-off view (team positions at the kick-off label) |

Findings that changed the pipeline in this session:

- **Shots:** the fixed rule found 18–22% of labelled shots. It was replaced by
  a classifier with flight-only candidates. The threshold is set so that called
  shots equal labelled shots, because the F1 optimum called about 40% too many
  in-sample. Training matches are analysed by the cross-validation model that
  never saw them. The candidate check showed 25 of 47 unreachable labels had a
  visible ball flight but no detected touch, which is why flight candidates
  were added.
- **Identity:** attaching segments without a readable number by tactical role
  was tested with hidden known numbers. It was right for about 43% of the
  attached time, and no threshold setting got much above 55%. It was removed.
  Per-player statistics now come only from reliably numbered segments (45–51%
  of outfield player time in four halves checked). Rates per 90 minutes on
  screen and per 100 touches were added, because they do not depend on that
  coverage.
- **Labels:** SoccerNet's 500-game Labels-v2 archive on D: has the same events
  as the SN-Labels files, with the team given as screen side. It now supplies
  shot, goal and kick-off labels for all 23 local games, for training and
  evaluation only. No new download was needed.

Pending when the batch finishes: retrain `scripts/train_shot_model.py` on all
halves, then run `scripts/analyse_matches.py --postprocess-only`. Earlier
halves were post-processed by the batch process, which still runs the code it
started with.

## 24 September 2026: repository integration

The full suite passed **96 tests**, followed by a passing regression for the
SoccerNet validation-split name. The existing browser suite passed **11 checks**;
the new real-data integration suite passed **7 checks**, with no page errors or
HTTP 500 responses. Detector inference, kit clustering and occupancy ran on the
RTX 3090; overlays used NVENC. The real 30-second Chelsea–Burnley example is
available as `soccernet-chelsea-burnley-h1-demo` among the app's 133 sources.

See [REPOSITORY_INTEGRATION.md](REPOSITORY_INTEGRATION.md) for the inspected
revisions, model-training status, feature boundaries and reproduction commands.
New screenshots and the browser report are on the D: data drive under
`PitchProfile/evidence/repository-integration`. Older results below document
earlier versions and are superseded where their counts or default tracker differ.

## 22 September 2026 baseline

The updated working directory was installed and exercised on Windows using
Python 3.13.7, an NVIDIA GeForce RTX 3090 (24 GB), driver 616.64,
PyTorch 2.8.0+cu128, and torchvision 0.23.0+cu128. CUDA operations executed
successfully on `cuda:0`. Installed versions are recorded in
`evidence/environment_windows_python313.txt`.

## Executed checks

| Check | Result |
| --- | --- |
| Dependency installation and `pip check` | Passed; no broken requirements |
| `scripts/doctor.py` | Passed; prepared sources, weights, GPU and FFmpeg available |
| `python -m pytest -q` | **84 passed**, no failures or skips, 24.72 seconds |
| `python scripts/check_browser.py` | **10 checks passed**, no JavaScript errors, HTTP 500s or server tracebacks |
| Live app HTTP check | Homepage HTTP 200 and `/api/status` reporting RTX 3090, CUDA available |
| Windows background lifecycle | `start.ps1 -Background` and `stop.ps1` exercised successfully; final app started on port 8000 |

The two rows above were re-run after this session's percentage-rating rewrite and the
PFF FC import (below); the earlier **76 passed / 11 checks** figures reflect the prior
binary-label system and are superseded.

The automated suite additionally checks interval feature boundaries, all-position
role compatibility, review privacy/history, source-change invalidation, the
released SoccerTrack schema, direction changes and provider benchmark partitions.
A separate version 2 GPU training round-trip verifies the 42 output heads and
unknown-label masking using disposable artificial fixtures.

The automated suite covers observed-only features, unknown labels, independent
reviews, adjudication invalidation, imports, exports, chronological history,
calibration geometry, orientation after recalibration, camera-cut rejection,
GPU selection, training eligibility, disjoint splits, GPU CNN training and model
loading, and database commit/rollback/connection closure. One upstream AnyIO
deprecation warning appeared in Starlette's test client; it did not cause failure.

Training tests use artificial fixtures in temporary directories. They verify
execution and GPU placement, not real archetype accuracy. The real data folder
retained zero team reviews and no archetype model during verification.

The browser checks include creating a 20-minute interval, two independent
reviews with supporting sequences, assessment visibility, and the new form at
a 390-pixel mobile width without page overflow.

The browser checks cover the real profile/history, training gate and missing
experiment disclosure, two independent disposable reviews, identity selection,
SoccerNet reference imagery, actual video upload, calibration frame loading and
reset, empty-result handling, and a 390-pixel mobile viewport without overflow.

## Actual GPU video run

The bundled already annotated preview was uploaded through the browser:

- **17 sampled frames, eight track IDs**, YOLO11n and ByteTrack on `cuda:0`.
- Recorded tracking-loop time: **0.785 seconds**. This excludes general startup
  and final encoding; it is not an end-to-end benchmark.
- Overlay encoded using **`h264_nvenc`** on the GPU.
- Chrome decoded and played the **3.4-second, 800 × 450** output.

CPU operations remain for frame handling, tracking bookkeeping, data processing,
and the scikit-learn logistic baseline. Detector and CNN computation uses CUDA;
video overlay encoding uses NVENC. The app does not silently switch neural
processing to CPU when CUDA is unavailable.

## Reproduction and artifacts

From `PitchProfile` after setup:

```powershell
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --junitxml=.runtime/pytest.xml
.\.venv\Scripts\python.exe scripts/check_browser.py
```

The passing browser run saved its report, server log, and desktop/mobile
screenshots under `.runtime/browser-20260922-035533-768820/` (the folder timestamp
is UTC; local time was 22 September). The re-run against the percentage-rating
system and full PFF dataset saved its report under
`.runtime/browser-20260922-063718-396623/`. Generated `.runtime` files are
ignored by Git. The compact results are also preserved in
`evidence/local_verification.json`.
The bundled detector file's locally recorded SHA-256 is
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`;
this records the supplied file, not an independently verified upstream checksum.

## SoccerTrack acquisition and paired-video checks

Approved Hugging Face access was used to download eight files for match 117092,
totalling **19.09 GB**, pinned to revision
`eae5179377d3b189d9f15c6b4c3a6a6c41a63eaa`. Both halves are imported locally,
with **23 first-half tracks and 33 second-half tracks**. Both share one match ID;
these counts must not be reported as 56 distinct players or two matches.

The first half has 67,375 video/reference frames. The second half initially
failed the alignment check: it has 67,550 video frames against 67,576 reference
frames. FFmpeg shows that its first video frame has PTS **1.00 second**; its
available span is [1.00, 2703.00) seconds. The importer now checks presentation
timestamps and excludes 132 sampled reference rows outside this span. It retains
the source clock, disallows reviews before the video starts, and adjusts OpenCV
frame seeking to the source timestamp. A separate test still rejects unexplained
frame-count differences. The recovered second-half download matches the
provider SHA-256 recorded in `evidence/soccertrack_local_verification.json`.

Chrome decoded and sought both **3840 × 1906** videos at source second 60. The
frame endpoint returned each corresponding frame. The UI sets the correct half
and initial review time. These were read-only checks; no real labels were saved.
A YOLO11n execution check on one real panoramic frame used `cuda:0` and detected
six people at input size 960. This is not detection accuracy or proof of complete
player coverage; panoramic detection quality still requires evaluation.

BAS events are downloaded but remain outside features pending event-clock
verification. Persistent identities and attack directions require review. The
real database retains zero reviews and no trained archetype model. The provider's
benchmark split assigns this match to training, so additional designated matches
are required for held-out evaluation.

## Percentage-based archetype ratings and PFF FC 2022 World Cup data

This session replaced the binary present/absent/unknown review rating with an
independent **0-100 percentage per role** (a player can rate high on several
roles at once), with tolerance-band consensus (`AGREEMENT_TOLERANCE=20` points)
and auto-derived primary/secondary roles (`SECONDARY_FLOOR=30`), and retired the
zero-review legacy three-label tab from the interface (backend routes untouched).
Setting a player's **archetype position group** in the identity panel now shows
the trained model's percentage/mixture suggestion directly on their profile —
`position_group`/`taxonomy_version` flow through `build_profiles()` into
`profiles.json` for every dataset, not only inside formal interval cases. This
was confirmed live: `GET /api/datasets/pff-wc2022-3812-h1/players/pff-7988`
returns `"position_group": "centre_forward"`, `"taxonomy_version": 2.0`, and a
`prediction` block (`"status": "not_trained"` until real reviews exist).

`football_profiler/pff.py` (new) streams PFF FC's released 2022 World Cup
broadcast tracking directly from the provider's delivered zip archives — no
bulk extraction — via `scripts/import_pff.py --archive-dir <path>`. All **64
matches** were imported as **128 half-datasets** (one per period, matching the
SoccerTrack precedent, since PFF's clock resets at half time):

```
Done: 63 imported, 1 already present, 0 failed
```

(The one "already present" match, 3812, was imported earlier in the session as
a manual correctness check before the bulk run.) Two real bugs were caught and
fixed against the live data before the bulk run: a `total_sampled_frames`
double-count from PFF's whole-match (not per-period) `frameNum` counter, and a
backwards attack-direction flip, corrected and verified against England's
goalkeeper's real coordinates (consistent ~16-18 m from their own goal line in
both halves after normalisation). The importer's background run predated a
same-session fix to `features.py` that passes `position_group`/`taxonomy_version`
through profile-building, so every PFF dataset's `profiles.json` was rebuilt
afterward (`F.build_profiles()` per dataset; 128/128 succeeded) — confirmed by
spot-checking that the rebuilt file contains both fields. Live dataset count
after the import: `GET /api/status` reports **132 datasets** (128 PFF halves +
the SoccerTrack 117092 pair + the bundled SkillCorner and SoccerNet fixtures).
PFF tracking has no accompanying video, so it broadens the statistically
trained cohort but is not a source for human interval review; see
`annotation/ANNOTATION_GUIDE.md` for sourcing notes on further datasets
researched but not yet built (SkillCorner opendata, IDSSE-data, Metrica,
Alfheim), out of scope for now per the user's own instruction.

## Remaining project work

The original archive supplied **one SkillCorner match with five centre forwards**,
plus SoccerNet reference trajectories. The four additional matches and auxiliary
CNN experiment described in the historical implementation report are absent.
Installation files, tests, and operating documentation have now been added here.

The app is ready for local use. The match cohort is now substantially broader
(64 PFF FC matches plus SoccerTrack 117092), but real archetype research still
requires a frozen rubric, independent team labels against that cohort, and
held-out evaluation — the real database retains **zero reviews and no trained
archetype model**. No matching SkillCorner footage or real archetype labels
were invented. Browser execution does not establish detection/tracking accuracy,
camera-localisation accuracy, or useful archetype predictions.
