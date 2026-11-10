# PitchProfile, Group D (Events): working notes

Owner: Hasif. Branch: `dev-hasif`. This branch only holds the Group D files. The full project is on `main`.

## What Group D does

Group D works out what happened in the match.

**Event rules** (`match_events.py`) are hand-written rules that work on the player and ball positions from Groups A and B. They decide who had the ball and when. Where the ball goes next decides the event:

| What happens | Event |
|---|---|
| next touch is by a teammate | completed pass |
| next touch is by an opponent | intercepted pass or clearance, then a recovery |
| one player moves the ball 2 m or more | carry |

**The T-DEED action spotter** (`action_spotting.py`) is the trained model. It watches 8-second clips of the video. It finds actions the rules cannot see: shots, tackles, blocks, headers, crosses and throw-ins. Each spotted action goes to the player at the ball at that moment.

**Fine-tuning** (`spotter_data.py`, `scripts/finetune_action_spotter.py`) starts from the published SoccerNet checkpoint. It trains on SoccerNet BAS and FOOTPASS games.

FOOTPASS labels are less detailed than the spotter's 12 classes. A FOOTPASS "Pass" could be a pass, a high pass or a free kick. The partial-label loss lets the model learn from all of these possibilities without guessing one.

Current results on held-out games:

| Measure | Result |
|---|---|
| mAP, FOOTPASS | 0.478 → 0.604 |
| mAP, BAS | 0.606 → 0.628 |
| Passes and carries | about 0.9 AP |
| Tackles | 0.06–0.10 AP (weak) |
| Credited to the right player | only about 34% of passes |

## Files on this branch

| File | What it is |
|---|---|
| `PitchProfile/football_profiler/match_events.py` | event rules |
| `PitchProfile/football_profiler/action_spotting.py` | T-DEED inference and crediting to players |
| `PitchProfile/football_profiler/spotter_data.py` | training data and the partial-label loss |
| `PitchProfile/scripts/finetune_action_spotter.py` | fine-tuning |
| `PitchProfile/scripts/run_loss_ablation.py` | one-command partial-label vs normal-label comparison |
| `PitchProfile/scripts/spot_actions.py`, `pack_spotter_frames.py`, `download_bas_training.py`, `extract_bas_training.py` | running the spotter and preparing training data |
| `PitchProfile/scripts/evaluate_action_spotter.py`, `evaluate_spotter_labelled.py`, `evaluate_events_soccertrack.py`, `measure_shot_credit.py` | evaluation |
| `PitchProfile/tests/test_spot_actions.py`, `test_spotter_data.py`, `test_match_events.py`, `test_loss_ablation.py` | tests |

## Checklist

Week 13 submission is about 23–27 Nov. Confirm the date on xSITe.

| # | Task | Who | Status |
|---|---|---|---|
| 0.1 | Branch `dev-hasif` and baseline commit of the Group D files | Hasif | Done, `c084508` |
| 0.2 | Test setup on the Mac (`PitchProfile/.venv`) | Hasif | Done |
| 0.3 | Ask Hadi for GPU time and how branches merge at the end; tell Danish about the Dispossessed bug | Hasif | To do |
| 1.1 | Keep the rejected tackle rule out of the stats | Hasif | Done, `f50f8ec` |
| 1.2 | Remove the duplicate `available()` | Hasif | Done, `24b34ab` |
| 1.3 | Remove or reuse the unused `contacts()` | Hasif | With 3.1 |
| 2.1 | Switch to train with normal labels, plus a one-command runner | Hasif | Done, `8da14ad`, `e62fa33` |
| 2.2 | Overnight comparison run, result `evidence/loss_ablation.json` | Hadi's PC | Waiting |
| 3.1 | Crediting fix: look wider in time or use the player in possession when the ball is not seen; throw-ins go to the team that did not put the ball out | Hasif | Next |
| 3.2 | Measure crediting before and after (no GPU needed) | Hadi's PC | After 3.1 |
| 4 | Optional: faster T-DEED and a cloud demo | Team decision | Not started |
| 5.1 | Group D section in the project README | Hasif | After 2.2 and 3.2 |
| 5.2 | Improvement commits copied into `main` | Hasif and Hadi | End |
| 6 | Slides, poster box, demo minute, Q&A prep, backup video | Hasif | Wed 18–Fri 20 Nov |

## What changed and why

| Commit | Change | Why |
|---|---|---|
| `24b34ab` | `available()` was defined twice; one copy removed | The second copy silently replaced the first. Behaviour is unchanged. |
| `f50f8ec` | The tackle rule is off by default; only `evaluate_events_soccertrack.py` turns it on | The rule had 6–14% precision and was replaced by the spotter. If the spotter did not run, its tackles still reached the stats. |
| `8da14ad` | `--hard-labels` switch in fine-tuning | Trains the same model with ordinary labels, to show what the partial-label loss adds. It refuses to overwrite the adopted checkpoint. |
| `e62fa33` | `run_loss_ablation.py` | Trains both versions with identical settings. Scores them on the held-out games and writes one comparison table. |

## Checking the code (Mac, no GPU)

```zsh
cd PitchProfile
.venv/bin/python -m pytest -q tests/test_spot_actions.py tests/test_spotter_data.py tests/test_match_events.py tests/test_loss_ablation.py
```

Expected: 20 passed, 1 skipped (the skipped one needs the GPU).

## Run on Hadi's PC (task 2.2)

```
git fetch origin dev-hasif
git switch -c hasif-ablation main
git checkout origin/dev-hasif -- PitchProfile
cd PitchProfile
.\.venv\Scripts\python.exe scripts\run_loss_ablation.py --dry-run
.\.venv\Scripts\python.exe scripts\run_loss_ablation.py
```

The dry run only prints the commands. The real run takes about 7–8 hours. The file needed back is `evidence/loss_ablation.json`.

## Open items

`match_stats.py` (Danish) counts Dispossessed only from rule events, so it shows 0 whenever the spotter ran. That needs Danish's fix.

`dev-hasif` has no shared history with `main`. Commits will be copied across at the end.
