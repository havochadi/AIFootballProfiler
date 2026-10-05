# Better player detection and action statistics: research and changes

September 2026. This report covers why the per-player statistics were sparse and hard to tell apart,
what the research literature does about it, what was changed in PitchProfile, and what remains.
All numbers were measured on this project's footage or on annotated ground truth; sources are listed
at the end.

## 1. What was wrong, measured

A baseline was taken on the 22 working-set halves (`scripts/diagnose_coverage.py`,
`evidence/coverage_diagnostics.json`, label `baseline`) and against SoccerTrack v2 match 117092,
the one local match where every action of every player is annotated.

| Problem | Evidence |
| --- | --- |
| Detection is **not** the bottleneck | About 14 outfield players are detected per frame |
| Players lose their identity at camera cuts | The broadcast cuts every 3.5 s (median). Within a view tracking is reasonable (66% of detections are in stitched segments of 5 s or more), but across cuts only a readable shirt number links a player |
| Only about a third of player time was attributed to a player | 32% of outfield detections belonged to an identified player |
| Defensive events were almost absent | Median per team per half: 2 tackles and 1 dribble, against 12 tackles in a fully annotated real half; 89% of players had no tackle and 94% no dribble |
| The tackle and block rules were wrong, not only starved of data | On SoccerTrack **ground-truth positions**, rule-based tackles had 0.06–0.14 precision and 0.08–0.12 recall, and blocks 0–0.10, while passes reached 0.66–0.69 precision and 0.69–0.70 recall |

A follow-up test explains the last row. A looser contact detector finds a ball contact by the
annotated player for 33 of 36 tackles and 27 of 38 blocks, but real tackles and blocks (about 1% of
contacts) have the same distances, velocity changes and timings as ordinary contacts. From 2D
positions and a 2D ball, a tackle cannot be told from a touch next to a defender. This matches the
literature (below), so tackles, blocks and headers need a model that looks at the video.

## 2. What the research says

**Action events from tracking data.** Vidal-Codina et al. (MIT Sports Lab and FIFA, 2022) generate
passes, receptions, interceptions, shots and goalkeeper events from player and ball tracking with
rules, reaching 90%+ detection for most categories. They state that tackles, duels and dribbles need
additional streams: ball height, limb tracking, or video.

**Learned ball-action spotting from broadcast video.** SoccerNet runs yearly benchmarks on
broadcast footage. The 2025 Team Ball Action Spotting task has 12 classes: pass, drive, header,
high pass, out, cross, throw-in, shot, block, successful tackle, free kick and goal. Each is tagged
with the team (left or right). The baseline is T-DEED, the 2024 winner (Team-mAP@1 51.7; best
entry 60.0). Its checkpoint is public. Tackles and blocks are the hardest classes for every entry.

**Player-centric action spotting.** SoccerNet 2026 added a task that credits each action to a
player by team and shirt number (FOOTPASS dataset: 54 matches; best macro-F1 58.9). Its baselines
combine a track-aware visual detector (TAAD) with a tactical sequence model (DST). The paper finds
that tactical context lifts precision from 25% to 68%. No pretrained weights are published, and
the videos require the SoccerNet NDA.

**Identity across cuts.** The 2025 Game State Reconstruction winners all combine appearance
re-identification embeddings (OSNet, CLIP-ReIdent, PRTReid) with global tracklet association
(GTA-Link). They read shirt numbers from pose-guided torso crops with PARSeq or with
vision-language models. SoccerNet's own PRTReid weights are public on Zenodo.

**Ball detection.** Dedicated sports-ball detectors (WASB, TrackNet-style heatmaps) and tiled
high-resolution inference (SAHI) raise ball recall. The winning spotters also crop around the ball.

## 3. What was changed

| Change | Result |
| --- | --- |
| **Video action spotter** (`football_profiler/action_spotting.py`, `scripts/spot_actions.py`): the SoccerNet 2025 baseline runs on every half (about 4.5 minutes per half) | Tackles, blocks, headers, crosses, lofted passes, throw-ins and free kicks are now measured. Against SoccerNet labels on the two held-out test matches: throw-ins 0.83 precision / 0.69 recall, ball out of play 0.61 / 0.86, **shots 0.78 / 0.63**. Our shot model managed 0.35 / 0.56, so **spotted shots now replace it** (train+validation halves: 0.71 / 0.70 against 0.36 / 0.47) |
| **Crediting spotted actions to players**: the player of the spotted team nearest the ball at that moment (within 3 m) | Statistics add the spotter's probabilities (**expected counts**). Its scores behave like probabilities on this footage: summed throw-in scores 166 against 171 labelled throw-ins |
| **More thumbnails** (`scripts/extract_crops.py`): up to 24 per tracklet, re-cut from the stored boxes without re-running detection | About 60,000 per half instead of about 15,000 |
| **Appearance re-identification** (`football_profiler/reid.py`, `scripts/train_reid_head.py`): frozen CLIP features plus a projection head trained on this project's own shirt-number readings (semi-supervised, no external identity labels) | Hiding the number of a known segment on held-out matches: **72%** correct overall (frozen CLIP 65%, the old role-based rule 43%). Attached segments are about **90%** correct: 17+ thumbnails with a margin of at least 0.02, 9–16 with at least 0.025, never with fewer |
| Identity coverage (all 22 halves) | Detected player-time attributed to a player rose from **32% to 49%**. Appearances with at least 10 identified minutes in a half rose from **100 to 218** |
| **Take-ons** redefined (`match_events._dribbles`): the carrier drives forward at 2 m/s or more at an opponent within 2 m in front; it is successful when his team keeps the ball, as in data-provider definitions | 7–9 attempts per team per half with 43–45% kept, close to real-world rates. The previous rule (opponent ahead, then behind) flipped on about a metre of position noise and found 5% "successes" |
| **Off-ball movement** (`football_profiler/match_movement.py`): high-intensity runs in and out of possession, runs in behind the last defender, runs into the box, overlaps and underlaps, pressing and recovery runs, height relative to team-mates, width, time between the opposition lines | These use positions only, the most reliable evidence, and describe roles such as runners in behind, pressing forwards and overlapping full-backs |
| **More on-ball statistics** (`match_stats.py`): receptions (progressive, in the box, in behind), touches in the final third and box, passes and carries into the final third and box, switches, balls played in behind, turnovers, pass length and direction mix | 90 statistics per player instead of 39 (56 on the ball, 18 off-ball movement, 5 physical, 11 positional) |
| **Style profiles** (`match_profiles.py`, *Stats & maps → Style profile*): every statistic as a percentile against players in the same position group | Players are compared with peers instead of raw counts |
| Rates per 90 minutes of **identified** screen time and per 100 touches | Players identified for longer are not favoured |

The archetype model's features now include all of the above (`semisupervised.py`).

### Before and after, all 22 working-set halves

`evidence/coverage_diagnostics.json`, snapshots `baseline` and `improved`. Medians per team per half.
Spotter statistics are expected counts; the reference is the fully annotated SoccerTrack v2 match.

| | Before | After | Real match (reference) |
| --- | --- | --- | --- |
| Detected player-time attributed to a player | 32% | **49%** | — |
| Player appearances with ≥ 10 identified minutes | 100 | **218** | — |
| Take-on attempts / successful | — / 1 | **7 / 3** | — |
| Tackles | 2 (rules, precision 0.06–0.14) | 1.5 (video model) | 12 |
| Blocks | — | 3.2 | 12 |
| Headers | — | 3.2 | — |
| Crosses | — | 3.8 | 12 |
| Lofted passes | — | 6.9 | 29 |
| Shots | 7 (precision 0.35) | **5 (precision 0.78)** | 6 |
| Players with no tackle | 89% | 54% | — |
| Players with no dribble | 94% | 78% | — |

**Tackles, blocks and crosses are still far below real rates.** The spotter reports about one tackle
for every eight in a fully annotated match. Its spots are credited to the team it predicts and that
team's player nearest the ball. Whether those players are right cannot be checked on this footage:
no local labels name the player for these actions. Use these statistics to compare players, not as
totals. Fine-tuning the spotter (section 4) is the way to raise them.

## 4. What remains, in order of expected value (updated 1 October)

1. **Better ball detection.** A quarter of the passes and carries the spotter finds cannot be given
   to a player, mostly because no ball was found at that moment (section 6). On SoccerNet tracking
   clips at 720p the pipeline finds the ball in 44% of frames where it is visible (62% of its balls
   right), and never when it is under 10 pixels. A ball detector fine-tuned on SoccerNet tracking
   clips (`scripts/prepare_ball_training.py`, `scripts/train_ball_detector.py`) is being measured
   (`scripts/evaluate_ball_detection.py`); if it is better, `match_analysis.redetect_ball` replaces
   only the ball candidates of each half (player tracks and identities stay).
2. **Naming the rest in the review screen.** About a third of visible player time is in
   consistent unnamed players; the Name players panel names them in one step each.
3. **Estimating off-screen positions** (Continuous tracking from broadcast data, 2023) to fill in
   physical statistics while a player is off camera.
4. **Tackles.** Neither available signal finds them (section 6); they need many more labelled
   tackles than exist (about 230 across FOOTPASS and BAS) or a model of the two players' contact.

Done since the first version of this list: appearance-based grouping across camera cuts, legible-
thumbnail number reading and a retrain on all 48 FOOTPASS training games (section 6), goalkeeper
selection, whole matches, and the spotter fine-tune (section 5).

## 5. Measured on labelled video (29 September)

With the SoccerNet NDA data the spotter can be measured on games it never saw, not only against
match-level annotations. `scripts/evaluate_spotter_labelled.py`; reports in `evidence/action_spotter_*.json`.

**Published spotter, two held-out SoccerNet Ball Action Spotting test games** (3,980 labelled
actions; best threshold per class; a spot counts within 1 s):

| Action | Precision | Recall | Right team | Summed scores vs labels |
| --- | --- | --- | --- | --- |
| Pass | 0.87 | 0.87 | 90% | 1,634 / 1,721 |
| Drive (carry) | 0.85 | 0.83 | 90% | 1,416 / 1,449 |
| High pass | 0.83 | 0.81 | 97% | 180 / 181 |
| Throw-in | 0.96 | 0.74 | 81% | 81 / 95 |
| Cross | 0.82 | 0.60 | 97% | 51 / 60 |
| Header | 0.68 | 0.69 | 77% | 146 / 182 |
| Shot | 0.63 | 0.66 | 100% | 47 / 44 |
| Block | 0.28 | 0.40 | 48% | 29 / 67 |
| Successful tackle | 0.09 | 0.14 | 50% | 12 / 28 |

Passes, carries, lofted passes, throw-ins, crosses and shots are reliable, and their summed scores
match the true counts, which is what the statistics add up. **Tackles and blocks from the published
model are close to noise**, and its team for them is a coin flip. It was trained on 34 tackles and
128 blocks.

**Findings that change how the data must be read**

- **FOOTPASS class ids.** The dataset page lists classes 4–6 as Shot, Header, Throw-in. On the
  validation games, class 4 coincides with the spotter's throw-ins (66%), 5 with its shots (70%)
  and 6 with its headers: the ids are Throw-in, Shot, Header (`scripts/pack_spotter_frames.py`).
- **Team side.** On this project's 22 analysed halves the spotter's side matched the team in
  possession for 96% of confident passes and carries, so the pipeline's "left = team defending
  the left goal" is right. In FOOTPASS that team has `left_to_right = 0` (its goalkeeper stands
  at x ≈ 0.1).
- **Who is credited with a tackle or block.** In both labelled datasets the tackler's or
  blocker's team is the other team from the last touch before it (99–100%). Taking the side
  opposite the last confident on-ball spot gave blocks the right team **81%** (BAS test) and
  **91%** (FOOTPASS) of the time, against 48% and 31% from the spotter's own output, so the
  pipeline now does that (`action_spotting.WON_BALL_SIDE`). Before this, a tackle credited to the
  wrong team went to the nearest player of the team that had the ball, often the player tackled.
- **1080p.** Only 3 of the 11 working-set games (Chelsea v Burnley, Dortmund v Wolfsburg,
  Villarreal v Real Madrid) have 1920×1080 broadcasts; the other "HQ" files are 1280×720. Rather
  than re-running detection, player thumbnails are re-cut from the 1080p frames for those halves
  (`scripts/extract_crops.py --hq`; median thumbnail height 69 → 104 px). SoccerNet's `video.ini`
  gives where each half starts in the broadcast in whole seconds, and for Villarreal v Real Madrid
  it was 0.6–0.8 s late; in one HQ file OpenCV's seek (which follows timestamps) also landed 7
  frames away from the counted frames, which shifted every thumbnail of Chelsea v Burnley's second
  half by 0.28 s (moving players fell outside their boxes). The start is therefore measured on
  the pictures with frames counted from the start of the file, the file is read without seeking,
  and the finished thumbnails are compared with their 720p versions (median correlation must be
  at least 0.8, else the 720p ones are kept).
- **Linking unnumbered segments** (`scripts/evaluate_linking.py`, `evidence/identity_linking.json`;
  8 held-out halves, known numbers hidden in 5 folds). Today's rule attaches hidden segments with
  91% accuracy and covers 26% of genuinely unnumbered player time. A GTA-style connector that
  first clusters unnumbered segments by appearance was **6–13% accurate**: team-mates in the same
  kit look alike to the appearance features, so clusters chain different players. Repeating the
  rule with updated player prototypes covers 44% but at 83% accuracy; with a stricter margin that
  keeps 90% it adds only 1 point. The rule is unchanged. More identified time has to come from
  reading more shirt numbers (sharper thumbnails, better torso crops), not from linking.

**Fine-tuning** (`scripts/finetune_action_spotter.py`, `evidence/action_spotter_finetune.json`):
3,000 steps (about 3 hours on one RTX 3090) from the published checkpoint on 5 BAS games and the
48 FOOTPASS training games (about 200 tackles and 1,150 blocks; games joined as they were packed),
FOOTPASS classes as partial labels, tackles and blocks weighted and over-sampled. The early
full-resolution backbone stages (16k of 3.2M backbone parameters) are frozen to fit 24 GB.
Held-out class loss on FOOTPASS validation clips fell from 3.39 to 1.80.

Average precision on games it never saw (published → fine-tuned):

| Action | BAS test (2 games) | FOOTPASS validation (3 games) |
| --- | --- | --- |
| Pass | 0.92 → 0.93 | 0.73 → **0.91** |
| Drive | 0.86 → 0.90 | 0.70 → **0.89** |
| Header | 0.64 → **0.76** | 0.33 → **0.64** |
| Cross | 0.66 → 0.69 | 0.53 → **0.70** |
| Throw-in | 0.86 → 0.86 | 0.73 → 0.78 |
| Shot | 0.62 → 0.66 | 0.58 → 0.51 |
| Block | 0.18 → **0.34** | 0.20 → **0.35** |
| Successful tackle | 0.02 → 0.10 | 0.01 → 0.06 |
| Mean (all classes) | 0.61 → 0.63 | 0.48 → **0.60** |

Blocks' summed scores now match the labelled count on BAS (72 against 67; 132 against 78 on
FOOTPASS). **Tackles remain weak**: best precision 0.11–0.23 at recall 0.25–0.31. The team of a
block from the previous-touch rule was right 78% (BAS) and 97% (FOOTPASS) of the time against
65% and 78% from the fine-tuned model's own output, so the rule stays; for tackles the two are
within the noise of 15 matched tackles. Free kicks are over-predicted against BAS labels (40
against 2), but BAS labels very few free kicks; SoccerNet's own free-kick labels on this project's
halves are the fair check (`scripts/evaluate_action_spotter.py`).

**On this project's 22 halves, against SoccerNet's own labels** (`evidence/action_spotter_labels_v2_*.json`;
4 held-out test halves): shots 0.72/0.67 → 0.68/0.74 precision/recall and summed scores 411 → 261
against 275 labelled shots (the published model counted 1.5 times too many); throw-ins 0.80/0.71 →
0.86/0.66; goals 0.23/0.42 → 0.39/0.58. Free kicks were poor with both models (recall 4–20%) and the
fine-tune sums to 2.8 times their number, so free kicks are no longer counted.

**Effect on the statistics** (`evidence/coverage_diagnostics.json`, snapshot `finetuned`; medians
per team per half; reference = the fully annotated SoccerTrack v2 match):

| | Before (published spotter) | Now (fine-tuned) | Reference |
| --- | --- | --- | --- |
| Blocks | 3.2 | **11.8** | 12.2 |
| Shots | 5.0 | **6.0** | 6.2 |
| Lofted passes | 6.9 | **14.4** | 29 |
| Headers | 3.2 | **14.0** | 1.75 (BAS test games: about 22) |
| Throw-ins | 8.2 | 10.1 | — |
| Tackles | 1.5 | 2.2 | 12 (successful tackles in BAS test games: about 3.5) |
| Crosses | 3.8 | 3.7 | 11.8 |
| Players with no block / no tackle | 70% / 55% | **19% / 45%** | — |
| Detected player-time attributed to a player | 49.1% | 49.8% | — |

The identity gain comes from the 1080p thumbnails of Dortmund v Wolfsburg (43.1 → 43.8% and
48.0 → 49.4% per half) and Villarreal v Real Madrid (56.1 → 59.2% and 55.4 → 56.4%). On Chelsea v
Burnley, an interlaced 2015 broadcast, 1080p thumbnails read fewer shirt numbers (identified share
54.9 → 50.2% and 56.7 → 53.1%), so that game keeps its 720p thumbnails.

## 6. Who is who, measured against true identities (30 September)

FOOTPASS gives every player's true box and shirt number in every frame, so identity can be
scored directly. The three held-out validation games (6 halves, 2.0 million true player-frames)
are run through the whole pipeline (`scripts/analyse_footpass.py`) and scored by
`scripts/evaluate_identity_footpass.py` (boxes matched at IoU ≥ 0.5; reports in
`evidence/identity_footpass_*.json`). SoccerNet tracking clips (49 clips of 30 s,
`scripts/simulate_identity_sn_tracking.py`) check the same steps on SoccerNet footage.

**Where the time went** (29 September, before these changes): 89.7% of visible player time was
detected, 53.1% was given an identity, and that identity was right only 49.2% of the time, so
26.2% of visible player time had the right player. Goalkeepers were found 45% of the time.

**What was wrong, found step by step**

1. **The identity model.** A CLIP ViT-B/16 was fine-tuned on FOOTPASS training crops (19 games,
   2.25 million crops, true identities) plus SN-Jersey-2023 (`scripts/train_identity_models.py`,
   `football_profiler/identity_model.py`). It reads the tens and units digits, gives a 256-d
   appearance embedding (supervised-contrastive, teammates as negatives) and a goalkeeper
   probability. Within a half its embedding separates teammates almost perfectly (segment pairs:
   AUC 0.995; the closest teammates' similarity is about 0.86 at the 99th percentile, the same
   player's about 0.96), and it finds goalkeepers 94% of the time.
2. **Grouping, not appearance, split players.** Segments are grouped per team by appearance
   (`match_identity.resolve_clusters`), never joining two that were on screen together. A player's
   main group held only 63% of his time. 94% of the split-off time was blocked by that rule: a
   segment the tracker carried onto another player for a while overlaps that player's own
   segments. Tolerating up to 15 s of shared screen time raised the main group's share to 89%
   with purity unchanged (94%); with no limit purity fell to 84%.
3. **Linking by pitch position.** Pitch coordinates carry across camera cuts, so a segment that
   starts where a teammate's segment was heading is linked to it (within 5 s; 3 m plus 2 m per
   second of gap; appearance must not contradict). Such links are 91–95% right. They add little
   on their own but reduce wrong names (2.2% → 1.4% of player time).
4. **Reading the number from legible thumbnails only.** The identity model's number heads were
   trained on every thumbnail of a player, shirt visible or not, so turned or blurred thumbnails
   still produce a guessed number, and summed over a group the guesses drown the few legible
   views. Only a quarter of thumbnails are legible (the legibility classifier of
   `football_profiler/jersey.py`). Letting only those vote (certainty-weighted) picked the right
   number for 76% of grouped time against 54% (97–98% on two of the three games). The legacy
   reader (legibility classifier + PARSeq) was right 44% of the time when the two disagreed and
   drops or swaps digits on some kits (18 → 8, 81 → 8, 11 → 10).
5. **Retraining the number heads on legible thumbnails only.** The identity model was retrained
   with number labels kept only on the 28% of training thumbnails the legibility classifier finds
   readable (appearance and goalkeeper heads still learn from all;
   `train_identity_models.py --legible-only --jersey`, 6,000 steps, about 40 minutes). The legible
   vote is now right for 88% of grouped time (74% before); on the hardest validation game 73%
   (42–53% before). The previous weights are kept as `model_v2_all_crops.pt`.
6. **Naming rule.** A group is named when its legible votes give one number at least 85% of the
   votes, or at least 30% with the legacy reader agreeing (`match_identity.NAMING = 'votes'`),
   after small groups have joined and groups have been merged to team size, so the name is read
   from all of a merged player's thumbnails (`NAME_AFTER_MERGE`).

**Result on the held-out FOOTPASS games** (whole pipeline, 30 September):

| | 29 Sept | 30 Sept |
| --- | --- | --- |
| Visible player time with the right identity | 26.2% | **49.8%** |
| Identity right when one is given | 49.2% | **94.9%** |
| Goalkeepers identified | 45% | **89%** |
| Visible time in consistent unnamed players (90% pure) | 0% | 34.4% |
| Visible time in no player at all (detected) | 36.6% | 2.8% |
| Passes credited to the right named player | 17.1% | **31.6%** |
| Carries credited to the right named player | 17.2% | **32.1%** |
| Headers credited to the right track | 0.6% | 14.8% |

In other words, of all visible player time 52% now belongs to a named player or goalkeeper (95%
of it to the right one), 34% to a consistent unnamed player, 3% is detected but in no player and
10% is not detected.

Unnamed players are real, consistent players (`<team>-X<k>`, shown as "Team · position ·
unnamed k"); their statistics accumulate like anyone else's and one rename in the review screen
names them. On SoccerNet tracking clips the same steps name 71.8% of player time right and 6.8%
wrong (the previous consensus rule with the old weights: 49.0% / 3.7%).

**On this project's 22 SoccerNet halves** (no true identities; `evidence/coverage_diagnostics.json`,
snapshot `identity_v3` against `finetuned`): detected outfield time in a player (named or
consistent unnamed) rose from 49.8% to 87.5%; named players hold a median 59% of tracked player
time (18–75% per half), with 21 named players per half. Striped kits are read worst (Inter v
Juventus 26–27%, Paris SG v Toulouse 19–26%, Real Madrid v Athletic 34–42%), as on FOOTPASS.
With the spotter's passes and carries now filled in, a team makes a median 249 passes and 175
carries per half (168 and 85 before), and far fewer players end a half without a header (46% →
18%), interception (24% → 16%) or block (19% → 13%).

**Goalkeepers on other kits.** On this project's SoccerNet halves the identity model's goalkeeper
output called most players of Villarreal's and Barcelona's yellow kits goalkeepers (median player
0.9, against about 0 for other teams), which merged half of those players into "the goalkeeper".
Its calls are now ignored for a team whose typical player scores above 0.3 or whose calls cover
over 15% of its detections, and a called goalkeeper must stay within 16.5 m of a goal line
(`match_pipeline.override_keepers`). On FOOTPASS this costs 3 points of goalkeepers found.

**Goalkeeper = one person, in one place (1 October).** Every goalkeeper-role segment of a team
used to join "the goalkeeper", so referees, other players and background detections did too; on
the worst halves a second "goalkeeper" box of the same team was on screen for over 5 minutes of a
half and the highlighted box jumped. Now a segment joins only if it looks like the team's keeper
(similarity at least 0.5 to his most confident, longest segments) and is not on screen with him
elsewhere on the pitch for more than a second; a second box on the keeper himself (same place:
the detector sometimes reports him as goalkeeper and as player) stays his, and segments without
thumbnails join no player (`match_identity.select_keeper`). Time with two goalkeeper boxes of one
team more than 5 m apart fell to 0–26 s per team-half; on FOOTPASS the keeper is found for 85.6%
of his time (88.5% before) at 99.4% purity.

**Number reader retrained on all 48 FOOTPASS training games (1 October).** With the other 29
training games unpacked (6.0 million thumbnails of 1,485 players; 10,000 steps, legible thumbnails only), the right
name went from 48.9% to 53.0% of player time on FOOTPASS validation (wrong 2.9% → 3.0%) and from
71.8% to 75.7% on SoccerNet tracking clips (wrong 6.8% → 6.5%). Adopted; the 19-game weights are
kept as `model_legible19.pt`.

**Whole matches (1 October).** Both halves of a match form one dataset (`match_merge`, built by
`scripts/build_whole_matches.py`): the two half videos joined without re-encoding, second-half
times shifted, kit groups matched by colour, the second half mirrored so each team attacks one
way, and statistics recomputed over the match. Named players and goalkeepers are one player when
team and number agree; an unnamed second-half player is `<team>-Y<k>` until a reviewer names him
or confirms him as someone ("same person as"), which is written to the half he comes from so it
survives re-analysis. Ratings follow players through renames and re-analysis, and ratings given
on a half are carried into its whole match (the latest wins).

**Names carried across halves (1 October).** In a whole match an unnamed player takes the name of
a player named in the other half when their appearance is clearly the same (similarity at least
0.9, 0.05 clear of the runner-up, never on screen with that player's own segments of his half;
`match_merge.carry_names`). Measured on the FOOTPASS validation games (each game's two halves):
+7.6% of visible player time named right, +1.2% wrong.

**The match line-up, when the reviewer supplies it (1 October).** With the user's approval, the
shirt numbers that played (from the app's match centre) check the naming, but only for a match
whose line-ups the reviewer connected and whose kit groups they confirmed. A number nobody in the
line-up wore is a misreading; an unnamed group whose legible votes, restricted to line-up numbers
not yet taken, give one number at least 90% of them (and at least 20% of all its votes) is named
with it (`match_identity._apply_roster`). On the FOOTPASS validation halves, with each team's true
outfield squad as the line-up: right names 53.0% → 64.2% of visible player time, wrong 3.0% →
2.7%. Without the 20% guard wrong names rose to 5.1%.

**Tackles, measured again (1 October).** On the 26 labelled tackles of the FOOTPASS validation
halves: the tracking rule (possession changing team within 0.6 s with the two players within
2.2 m) found none of them (40 candidates); the spotter at its best threshold 27% at 8% precision;
the two combined, none. A spotter fine-tune weighted towards tackles and blocks did not help
either (tackle average precision 0.06 → 0.08 on FOOTPASS, 0.10 → 0.09 on BAS; crosses and headers
worse) and was not adopted. Tackle counts remain rough expected counts.

**What limits it now**

- **Reading digits on hard kits.** Before retraining, one validation game (red-and-black stripes;
  thin, pale numbers on white) kept 70% of its player time unnamed although those groups had over
  100 legible thumbnails each: the digits were misread, not unseen. Retraining on legible
  thumbnails raised its reading to 73%, and the 48-game retrain lifted names further (above).
  Striped kits are still read worst on this project's matches.
- **Crediting actions to the player on the ball.** Even with perfect identities, 59% of labelled
  passes land on the right player's track. Rule-based passes are credited correctly 86% of the
  time; passes the spotter adds, 35%. A quarter of spotted passes and carries have no player:
  no ball found near that moment (38%), a ball found outside every box of that team (36%), or no
  player of that team on screen (22%). Wider search windows, looser boxes and falling back on the
  player in possession gained 1–3 points each at a similar loss of precision, so they were not
  adopted; better ball detection is the lever.
- **Detection.** 10% of visible player time is never detected; boxes over 150 px (close-ups) are
  detected only 64% of the time.

## Sources

- SoccerNet 2025 Challenges Results — https://arxiv.org/abs/2508.19182
- SoccerNet 2026 Challenges Results — https://arxiv.org/abs/2607.07320
- SoccerNet Team Ball Action Spotting devkit and baseline — https://github.com/SoccerNet/sn-teamspotting
- T-DEED (Xarles et al., CVsports 2024) — https://github.com/arturxe2/T-DEED
- FOOTPASS (Ochin et al.) — https://arxiv.org/abs/2511.16183, https://github.com/JeremieOchin/FOOTPASS
- Entity-aware sequence transduction for player-centric ball action spotting — https://arxiv.org/abs/2608.01696
- Vidal-Codina et al., Automatic event detection in football using tracking data — https://arxiv.org/abs/2202.00804
- From Broadcast to Minimap (SoccerNet GSR) — https://arxiv.org/abs/2504.06357
- SoccerNet Game State Reconstruction baseline and PRTReid — https://github.com/SoccerNet/sn-gamestate
- GTA: Global Tracklet Association — https://arxiv.org/abs/2411.08216
- CLIP-ReIdent — https://arxiv.org/abs/2303.11855
- WASB sports ball detection — https://arxiv.org/abs/2311.05237
- Continuous football player tracking from discrete broadcast data — https://arxiv.org/abs/2311.14642
