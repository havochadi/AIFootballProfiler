# Models used in PitchProfile

Every model that touches match footage runs locally, on the machine's NVIDIA
GPU. In this file a *model* is something with weights learned from data; hand-written
rules and other non-learned methods are listed separately below. No cloud AI service or language model is used anywhere in the pipeline.
Weights live on the data drive (`D:\CVDL Football Data\PitchProfile\weights`
and `D:\CVDL Football Data\third_party\No-Bells-Just-Whistles\weights`), not
inside the project, and are fetched by `scripts\fetch_football_models.py`
(SHA-256 checked). Licences differ per model — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before any non-research use.

## Pretrained models (full-match video analysis)

These run on every analysed half and are downloaded as-is, except the action
spotter, which this project fine-tunes (next section).

| Stage | Model | Architecture | Source | Licence |
| --- | --- | --- | --- | --- |
| Player/keeper/referee/ball detection | Roboflow football detector | YOLOv8x, 1280 px input | [roboflow/sports](https://github.com/roboflow/sports) | AGPL-3.0 (Ultralytics) |
| Ball detection (fallback) | Roboflow ball-only detector | YOLOv8x | roboflow/sports | AGPL-3.0 |
| Pitch calibration | NBJW keypoint network | HRNetV2-W48, 57 pitch keypoints, half precision | [No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) | GPL-2.0 |
| Shirt-number legibility | Legibility classifier | ResNet-34 | [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) (Koshkina & Elder) | CC BY-NC 3.0 |
| Shirt-number reading | Jersey PARSeq | PARSeq scene-text recogniser, SoccerNet fine-tune | Same pipeline; base code from [baudm/parseq](https://github.com/baudm/parseq) | CC BY-NC 3.0 weights, Apache-2.0 code |
| Ball-action spotting (tackles, blocks, headers, crosses, lofted passes, throw-ins, shots, passes, drives) with the acting team | T-DEED team ball-action baseline, used as the starting point for this project's fine-tune (below) | RegNet-Y 200MF with gate-shift fusion + temporal encoder-decoder, 100-frame clips at 796×448, 12.5 frames/s | [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) (SoccerNet 2025 challenge baseline) | GPL-3.0 |

**Tracking** (BoT-SORT, `config/botsort_match.yaml`) is not a learned model in
this configuration: appearance re-identification is disabled, and motion
association uses sparse optical-flow camera-motion compensation only.

## Models trained by this project

| Model | What it does | Method | Status as of this session |
| --- | --- | --- | --- |
| Ball-action spotter (fine-tuned) | The spotter the pipeline uses (`weights\tdeed_team_bas_ft\checkpoint_best.pt`; the published baseline is used if it is missing) | Fine-tuned from the published checkpoint for 3,000 steps on SoccerNet Ball Action Spotting (5 games) and FOOTPASS (48 games) video under the SoccerNet NDA; FOOTPASS classes as partial labels; tackles and blocks weighted; early backbone stages frozen (`scripts\finetune_action_spotter.py`) | Trained 29 Sept. On held-out games, mean average precision 0.61 → 0.63 (BAS test) and 0.48 → 0.60 (FOOTPASS validation); blocks 0.18 → 0.34, headers 0.64 → 0.76, tackles 0.02 → 0.10 (still weak). Free kicks are not used (`evidence/action_spotter_*.json`, DETECTION_RESEARCH.md section 5) |
| Player identity model | Reads shirt numbers, gives an appearance embedding that tells teammates apart, and spots goalkeepers, per player thumbnail (`weights\identity_vitb16\model.pt`, `football_profiler/identity_model.py`) | CLIP ViT-B/16 fine-tuned end to end: tens and units digit heads, a 256-d supervised-contrastive embedding (teammates as negatives) and a goalkeeper head; trained on FOOTPASS true identities (all 48 training games, 6.0 million thumbnails of 1,485 players, SoccerNet NDA) plus SN-Jersey-2023 numbers; numbers learned only from thumbnails the legibility classifier finds readable (`scripts\train_identity_models.py --legible-only --jersey`) | Retrained 1 Oct on 48 games (10,000 steps); earlier versions kept as `model_legible19.pt` (19 games, 30 Sept) and `model_v2_all_crops.pt` (every thumbnail, 29 Sept). 48 games against 19: right name 48.9% → 53.0% of player time on FOOTPASS validation, 71.8% → 75.7% on SoccerNet tracking clips. On the 3 held-out FOOTPASS games: appearance matching 98.2% of true tracklets, goalkeepers 90% recall at 99% precision, legible-thumbnail vote right for 88% of grouped player time (74% before). Whole pipeline: right player for 50% of visible player time with 95% of names right (FOOTPASS), 72% with 91% right on SoccerNet tracking clips (DETECTION_RESEARCH.md section 6, `evidence/identity_footpass_final.json`) |

## Non-learned components worth knowing about

- **Teams**: kit colours clustered with K-means, separately per half (no
  cross-match team classifier).
- **Ball path**: global shortest-path search over detections, not a model.
- **Events** (passes, carries, take-ons, interceptions, recoveries, pressures,
  clearances): hand-written rules over tracked positions and ball path. Tackles,
  blocks and headers come from the action spotter instead: rules could not tell
  them apart from ordinary contacts, even on ground-truth positions.
- **Off-ball movement** (runs in behind, into the box, overlaps, pressing and
  recovery runs, height and width): rules over tracked positions.
- **Percentile profiles**: percentile ranks against same-position players.
- **Attack direction**: rule from average team position and (optionally)
  kick-off formation.
- **Historical context** (line-ups, scores, portraits): fetched from ESPN's
  public feed as reference data — not a model, and not an input to any of the
  models above.

## Retired on 8 October 2026

Removed after measuring each against the model that replaced it. Their weights are archived under
`weights\_archive` on the data drive, and the evidence files they produced are kept.

| Retired | Replaced by | Why |
| --- | --- | --- |
| Shot classifier (gradient-boosted trees) and the hand-written shot rule | The video action spotter (T-DEED) | Same four held-out SoccerNet halves, same labels and matching: spotter F1 0.70 against 0.41 (0.71 against 0.45 at each model's best threshold). The classifier's release made no difference to who was credited with a shot, and removing it left every end-to-end score unchanged (`scripts/compare_shot_vs_spotter.py` is in git history; `evidence/shot_vs_tdeed.json`). |
| Frozen CLIP encoder and re-identification head | The appearance embedding of the identity model | On SoccerNet tracking clips (6,473 pieces) the old head matched 71.6% against 90.1% for the identity model (`evidence/identity_sn_tracking_mixed_model.json`). It only ran when the identity weights were missing. |
| Archetype profiler (label spreading over 42 styles), interval-review CNN, archetype ratings, interval reviews and the 42-role catalogue | Nothing: the project does not rate playing styles | No players were ever labelled, so neither model was fitted. The code, screens and tests were removed; position groups stay for percentile profiles. Existing `annotations.sqlite` files are left untouched. |
| Short-clip analysis: upload dialog, short-interval analysis, manual calibration and the YOLO11n (COCO) detector | Full-match and full-half analysis with the YOLOv8x pipeline | The models are built and tested on whole halves and matches. The short-clip path used a different detector and none of the identity, event or action models. Its weights file (`models/yolo11n.pt`) and `scripts/train_detector.py` were deleted from the repository. |
| Spotter fine-tune `tdeed_team_bas_ft2` and five older identity-model checkpoints | `tdeed_team_bas_ft`, `identity_vitb16/model.pt` | `ft2` was lower on FOOTPASS validation (0.585 against 0.604 mean average precision). |

Shots, tackles, blocks and headers now come from the action spotter alone, and `analyse()` runs it
as part of the analysis job (`match_pipeline.spot_actions`). Without the spotter's devkit and
checkpoint a half has no shots, tackles, blocks or headers; `scripts/doctor.py` reports whether it is available.

The hand-written tackle rules stay in `match_events.py`: they share the logic that tells an
interception from a pass, and the spotter's tackles replace them whenever it has run.

## Where to look in the code

- `football_profiler/football_models.py` — detector/keypoint model loading.
- `football_profiler/jersey.py` — shirt-number legibility + PARSeq.
- `football_profiler/action_spotting.py` / `scripts/spot_actions.py` / `scripts/evaluate_action_spotter.py` — ball-action spotter.
- `football_profiler/spotter_data.py` / `scripts/pack_spotter_frames.py` / `scripts/finetune_action_spotter.py` /
  `scripts/evaluate_spotter_labelled.py` — spotter fine-tuning data, training and held-out measurement.
- `football_profiler/match_movement.py`, `match_profiles.py` — off-ball runs, style profiles.
- `football_profiler/vision.py` — video helpers only (ffmpeg lookup, overlay encoding, video metadata, frame reads).
