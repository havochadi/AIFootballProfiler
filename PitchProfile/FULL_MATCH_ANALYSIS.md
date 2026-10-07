# Full-match analysis: broadcast video to player statistics

PitchProfile watches one broadcast half and produces statistics for every player
it can see: heatmap, distance, speeds, time on the ball, passes, carries,
dribbles, shots, tackles, interceptions, recoveries, pressures and more. The
analysis reads **only the video**. It uses no tracking feeds, event data,
line-ups or calibration files. SoccerNet and SoccerTrack annotations serve only
to measure accuracy (they once also trained a shot classifier, retired on 8 October 2026:
shots now come from the video action spotter alone; see MODELS.md).

The statistics feed the archetype workflow. You label some players with
archetype percentages, and a semi-supervised model estimates percentages for the
rest (see [Archetype profiling](#archetype-profiling)).

The review desk can also connect historical line-ups and results from ESPN.
These records identify players for the reviewer; they are **separate from the
video-only detection, event logic and archetype-model features**.

## Running it

Run these from `PitchProfile/`. Activating the environment is unnecessary.

```powershell
# One-off: detector, ball, pitch-keypoint, jersey-number and action-spotter weights (SHA-256 checked)
.\.venv\Scripts\python.exe scripts\fetch_football_models.py

# Analyse every working-set half not yet done (about 30 minutes per half on the RTX 3090)
.\.venv\Scripts\python.exe scripts\analyse_matches.py
.\.venv\Scripts\python.exe scripts\analyse_matches.py --match Chelsea --half 1

# Video passes that add evidence to analysed halves (about 1 and 5 minutes per half)
.\.venv\Scripts\python.exe scripts\extract_crops.py        # up to 24 thumbnails per tracklet
.\.venv\Scripts\python.exe scripts\spot_actions.py         # tackles, blocks, headers, crosses, ...

# Re-run only the CPU stages from saved evidence (a few minutes per half),
# e.g. after changing event logic or retraining a model
.\.venv\Scripts\python.exe scripts\analyse_matches.py --postprocess-only

# Models trained or calibrated on this project's own analyses
.\.venv\Scripts\python.exe scripts\train_shot_model.py
.\.venv\Scripts\python.exe scripts\train_reid_head.py
.\.venv\Scripts\python.exe scripts\evaluate_action_spotter.py
.\.venv\Scripts\python.exe scripts\diagnose_coverage.py --label <name>   # coverage snapshot
```

Only the matches in the working set (`D:\CVDL Football Data\SoccerNet\working_set.json`,
11 of the 23 local matches) are analysed; `--all-videos` overrides this.

In the app, **Matches** lists local halves and their analysis status. Open a
ready half with **Label players**. The review desk puts a filtered player queue,
the source video with a player highlight, and role ratings on one screen.
Click an appearance on the green timeline or an event timestamp to seek.
**Next appearance** skips off-screen gaps. Movement, physical statistics and event maps share
**Stats & maps**; the comparison table is in **Matches**.

Expand an archetype to read its definition and two examples. Sourced player
comparisons describe behaviours in a stated context; illustrative sequences are
teaching examples, not verified events in your footage. The rating panel uses
the main page scroll and keeps one role expanded at a time.

**Identify player** below the footage opens a visual picker. Choose the actual
team, search by shirt number or name, then confirm a player card. The paused
player image and example crops stay alongside the choices. This does not require
Kit A / B setup and does not change the whole team's mapping.

Use **I only know their number or role** for a quick manual correction, or
**This is not the goalkeeper** to go directly to the outfield option. The reviewer
name is remembered, and extra names / notes are optional. After saving, use
**Change player** to revise the choice or **Restore model identity** to undo it.
Advanced merging is outside this everyday flow.
Corrections apply to all appearances grouped under this track in this half. They
do not redraw boxes or separate a track containing several different people.
The original detections, statistics, saved ratings and bookmarks are preserved.
Conflicting goalkeeper / outfield ratings are flagged and excluded from the next
model fit until reviewed. Corrections persist independently of analysis rebuilds.

In **Stats & maps**, **Movement** shows physical and positional statistics alongside
the movement heatmap; passes, shots and other actions have their own statistics
and maps in the same panel. Whole-half values are separate from filtered counts.
**Playback help** contains reload and direct-video controls.
If Chrome stalls at a selected timestamp, the player automatically retries the
seek once after 2.5 seconds. Original footage remains on the data drive.

More SoccerNet 720p matches can be added with
`scripts\download_soccernet_720p.py --sample N --download` (it samples leagues
and seasons evenly and downloads only the 720p halves to
`D:\CVDL Football Data\SoccerNet\videos-720p`).

## Event maps and outcome review

In **Label players**, switch the evidence panel from **Match footage** to
**Stats & maps**. Choose passes, shots, tackles, interceptions, carries, dribbles,
recoveries, clearances, pressures or touches.

- **Events & pass routes** shows origins and available destinations, coloured by
  outcome. Select a marker or an event in the list, then **Watch moment** to seek
  the video. The inspector lets you save a reviewed outcome with your name and note.
- **Origin heatmap** counts event origins in spatial bins. Outcome and time-range
  filters apply to both displays. The default rotates known attack directions to
  the right; **Stadium view** shows the original coordinates. Missing positions or
  directions are reported and excluded from the relevant map, not invented.
- Completed and intercepted passes come from the video logic. Detected tackles
  and interceptions represent successful possession changes, not every attempted
  challenge. **The shot model does not classify goals, on-target, blocked or
  off-target outcomes.** These remain **Unknown** until you review them.
- The inspector can show reported shot outcomes for the linked player from ESPN.
  Those timestamps use the match clock, which may differ from the video's clock.
  Confirm the corresponding moment before applying an outcome. Provider shot
  coordinates are not mixed into the calibrated video maps.

Reviewed outcomes live in `annotations.sqlite` alongside their change history.
The original `events.json` and model-derived statistics are unchanged. If a source
event changes during reprocessing, its old review is not applied to a different event.

## Historical line-ups, player names and portraits

**Match centre** shows the sourced final score, venue and match-day squads.
Historical records for all **23 downloaded fixtures** have been cached in
`D:\CVDL Football Data\PitchProfile\data\match_context`. Both halves reuse the same
fixture record. Available portraits and team badges are in the sibling
`match_assets` folder and are served by the local website.

1. Open **Match centre · line-ups, score & kit groups**. **Connect match data**
   searches by competition, date and teams; a specific ESPN match link or event ID
   can be supplied when the fixture cannot be matched unambiguously.
2. Optionally, check the footage and assign **Kit A** and **Kit B** to the actual teams.
   These clusters do not inherently mean home and away. Save your confirmation.
3. Detected shirt numbers resolve against that **historical match roster**. A
   line-up match is a suggestion until visually confirmed. Missing squad numbers
   and unused substitutes are flagged for review.
4. Use **Identify player** to choose a team, search the match-day players and
   confirm the correct person. The tracking ID, footage, statistics and archetype labels remain
   attached. Original number readings are retained, and duplicate identity links
   are shown so that you can review them before merging tracks.

For Chelsea–Manchester United on **7 February 2016**, the source reports **1–1**,
United **#10 Wayne Rooney** and **#35 Jesse Lingard**. Rooney's portrait is available.
[ESPN match source](https://www.espn.com/soccer/match/_/gameId/422419).

Portrait coverage varies; the UI shows a shirt-number placeholder when no image
is available. Photos may be current rather than from the historical season.
ESPN's public website feed is undocumented and can change. Failed refreshes keep
the previously cached match, and the source link remains available for checking.

To cache new downloaded fixtures and available images:

```powershell
.\.venv\Scripts\python.exe scripts\sync_match_context.py
.\.venv\Scripts\python.exe scripts\sync_match_context.py --match Chelsea --refresh
```

## Pipeline

Detection is the only GPU-heavy stage. Its raw evidence is saved, so every later
stage can be re-run with `--postprocess-only`.

| Stage | What happens | Method / model |
| --- | --- | --- |
| Sampling | 12.5 samples per second from 25 fps video | — |
| Detection | Players, goalkeepers, referees, ball | Roboflow football YOLOv8x at 1280 px; dedicated ball detector on samples where the player model finds no ball |
| Camera cuts | Shot changes reset tracking; no identity is carried across a cut | HSV-histogram change (Bhattacharyya distance > 0.35) |
| Tracking | Tracklets within each camera shot | BoT-SORT with sparse optical-flow camera-motion compensation (`config/botsort_match.yaml`) |
| Pitch calibration | Image-to-pitch homography on every second sample, then outlier rejection, interpolation inside shots and smoothing | No-Bells-Just-Whistles HRNet pitch keypoints (57 points), RANSAC fit, white-line support check |
| Teams | Two kit clusters per half; referees by detector class | Torso CIELAB colour K-means (unsupervised, per half) |
| Attack direction | The team whose players average further left defends the left goal; goalkeepers join the team whose goal they stand near | Positional rule |
| Ball path | One ball per sample, chosen over the whole half; short gaps (≤ 0.8 s) interpolated | Global shortest-path search over ball detections |
| Possession | Touches are ball-velocity changes at a player (with the ball on the player in the image) or sustained close control; consecutive touches form spells; brief opposing contacts count as deflections | Rules, validated on ground truth (below) |
| Events | Passes, carries, take-ons, interceptions, recoveries, clearances, pressures, crosses, key passes, shots | Rules on tracking and ball; shots by a learned classifier |
| Action spotting | Tackles, blocks, headers, crosses, lofted passes, throw-ins and free kicks, with the team that made them; credited to that team's player nearest the ball | SoccerNet 2025 team ball-action baseline (T-DEED) on 8-s clips at 12.5 frames/s (`action_spotting.py`) |
| Thumbnails | Up to 24 per tracklet, tallest boxes spread in time | Re-cut from stored boxes (`scripts/extract_crops.py`) |
| Identity | Tracklets stitched inside a view; shirt numbers read per segment; players linked across cuts by number, and unnumbered segments by a clear appearance match | SoccerNet legibility ResNet-34 + fine-tuned PARSeq; CLIP ViT-B/16 with a head trained on this project's shirt-number readings (`reid.py`) |
| Off-ball movement | High-intensity runs and their type, position relative to team-mates and the opposition lines | Positions only (`match_movement.py`) |
| Statistics | Per identity and per team; style profiles as percentiles against same-position players | `match_stats.py`, `match_profiles.py` |

## Statistics per player

Counts cover what the broadcast showed of each identified player (see
[Identity](#identity)). Rates come on three bases, selectable in the match view:

- **Per 90 minutes on screen** (the default): counts divided by the player's
  own identified screen time. This basis does not depend on how much of the
  player could be identified. The camera follows the ball, though, so it
  describes involvement while the player is in the picture.
- **Per 90 minutes of team live play**: counts divided by the team's observed
  live time (calibrated pitch view), one denominator for the whole team.
  Players who were identified for more of the half score higher.
- **Per 100 touches**: the mix of on-ball actions (passes, progressive passes,
  crosses, carries, dribbles, shots and so on), which is the most direct
  reading of style and is independent of time and coverage.

Physical measures use only the player's own visible time. Coordinates are
normalised so every player attacks to the right.

| Group | Statistic | Definition |
| --- | --- | --- |
| Position | Heatmap | 20 × 32 grid of visible positions over the 105 × 68 m pitch |
| | Mean position, spread | Mean and standard deviation of x (length) and y (width) |
| | Thirds, lanes, box share | Share of visible time in the defensive/middle/attacking third, left/central/right lane and opponent penalty area |
| Physical | Distance, distance per minute | Positions smoothed over 1 s, measured over 0.4 s steps; steps above 11 m/s discarded |
| | Top speed | 98th percentile of step speeds (resists single-step spikes) |
| | Speed zones, sprints | Seconds walking (< 7.2 km/h), jogging, running, high speed (19.8–25.2) and sprinting (≥ 25.2); a sprint is ≥ 1 s above 25.2 km/h |
| On the ball | Touches, time on ball | Possession spells and their duration |
| | Passes, completion | Release followed by a teammate's touch (complete) or an opponent's (intercepted), within 4 s in the same view |
| | Progressive / long / forward share | Pass gains ≥ 10 m towards goal / length ≥ 30 m / share of passes going forward |
| | Crosses | Pass from the wide attacking zone whose ball path or reception enters the penalty area |
| | Key passes | Completed pass whose receiver shoots within 5 s |
| | Carries, progressive carries | Ball moved ≥ 2 m during one spell / ≥ 10 m towards goal |
| | Take-ons, dribbles | Take-on: the carrier drives towards goal (≥ 2 m/s) at an opponent within 2 m in front and 1.2 m across. A dribble is a take-on after which his team keeps the ball |
| | Shots | Release classified as a shot by the shot model (below); mean shot distance |
| | Receptions | Completed passes received; progressive, in the final third, in the box, and in behind the last defender |
| | Touches in zones | Touches in the final third and in the opponent penalty area |
| | Final-third and box entries | Completed passes and carries that start outside and end inside the final third or the box |
| | Switches, balls in behind | Completed passes moving the ball ≥ 30 m across; completed forward passes received beyond the last opposing outfielder |
| | Pass mix | Shares of short (< 15 m) and long (≥ 30 m) passes, and of backward and sideways passes |
| | Lofted passes, crosses | Spotted by the video model (high pass, cross) and credited to the passer |
| | Turnovers | Dispossessed plus intercepted passes |
| | Dispossessed | Lost the ball to a spotted tackle |
| Defending | Tackles | Successful tackles spotted by the video model, credited to the tackling team's player at the ball |
| | Blocks, headers | Spotted by the video model, credited the same way |
| | Interceptions | Cut out a moving opposing pass (ball ≥ 6 m/s, flight ≥ 3 m) |
| | Recoveries | First touch of the ball after the opponents had it |
| | Clearances | Long release (≥ 20 m) from the own defensive third to the opponents |
| | Pressures | Closed an opposing ball carrier to within 3 m |
| Off the ball | High-intensity runs | ≥ 19.8 km/h for ≥ 1 s; split into runs with and without possession |
| | Runs in behind, runs into the box | In possession, forward runs from level with or behind the last opposing outfielder to beyond him / into the box |
| | Overlaps, underlaps | Forward runs past a team-mate on the ball, outside / inside him |
| | Pressing runs, recovery runs | Out of possession, runs that close the ball carrier to 3 m / run back ≥ 5 m towards the own goal |
| | Height, width, between the lines | Mean position relative to visible team-mates; distance from the centre line; share of possession time between the opposition's defensive and midfield lines |

Statistics from the video model (tackles, blocks, headers, lofted passes, crosses, throw-ins,
free kicks) are **expected counts**: the sum of the model's probabilities for the actions credited
to the player. Only spots above each class's evaluated threshold appear as events on the maps.

**Style profile** (Label players → Stats &amp; maps) shows each statistic as a percentile
against same-position players with at least 10 identified minutes in a half, across all analysed
halves (all outfield players when a group has fewer than 8).

Team statistics include possession share (time on ball). Every event carries
its time, pitch position and camera shot, and is listed under the player in
**Matches → Compare all players’ statistics**.

## Shots

**When a half has action spots (`scripts/spot_actions.py`), shots come from the video
spotter.** It found SoccerNet-labelled shots at 0.71 precision and 0.70 recall, against 0.36
and 0.47 for the shot model below (held-out test halves: 0.78 / 0.63 against 0.35 / 0.56).
Confident spots (score ≥ 0.575, the F1-optimal threshold, which also calls about as many
shots as were labelled) are counted. The shooter is taken from the shot model's release by
that team within 1.5 s when there is one, otherwise the player nearest the ball. The shot
model remains the fallback for footage without spots.

A shot is often not seen as a touch: the ball blurs at the strike, and the
camera cuts to a close-up or replay straight afterwards. A fixed rule found
about a fifth of SoccerNet-labelled shots. Shots are therefore decided by a
gradient-boosted classifier over candidate releases near the goal a team
attacks. Candidates are the ends of possession spells, plus fast ball flights
towards goal that no detected touch explains. The latter are credited to the
nearest attacking player. Features cover geometry (distance and angle to goal),
ball flight (speed, heading, how close it gets to the goal line), the image
(ball height on the player's box, which separates headers), opponents in the
shooting cone and keeper distance. They also include what happened next and the
broadcast editing: whether the view leaves the pitch (close-ups, replays) and
how soon it cuts. The strongest features were the view leaving the pitch after
the release and the ball reaching the goal line.

`scripts\train_shot_model.py` trains the classifier. It takes SoccerNet
'Shots on target', 'Shots off target', 'Goal' and 'Penalty' labels, matched to
releases by time and by the labelled team's side of the pitch. Official
SoccerNet test games are held out. Leave-one-match-out cross-validation on
training and validation games measures accuracy and sets the threshold. Shot
counts are statistics, so the threshold makes the number of shots called equal
the number labelled, rather than maximising F1 (which over-counts). The test
games are scored once. Each cross-validation model is kept and analyses the
match it did not see. No half's shot counts therefore come from a model trained
on that match's labels, and new footage uses the model fitted on all training
games. Without a trained model file, the earlier flight/keeper rule is used.

## Identity

Tracklets are stitched within a camera view. Shirt numbers are read from the
thumbnails of each stitched segment. Up to 14 well-read numbers per team (≥ 3
readings) become the team's players, shown as `Team A #7`. Only segments with
a reliable reading of that number are attributed to the player. In the halves
checked, these segments hold 45–51% of outfield player time.

Segments whose number could not be read join a player only on a clear
**appearance match**. Each segment's thumbnails are embedded with frozen CLIP
ViT-B/16 features and a small projection head. The head is trained on this
project's own analyses (`scripts/train_reid_head.py`): segments whose number was
read are the labels, and team-mates in the same kit are the negatives. This is
the semi-supervised idea applied to identity, with no external identity labels.
An unnumbered segment joins the player whose segments it most resembles when the
similarity beats the runner-up by a margin. The margin is at least 0.02 with 17 or
more thumbnails, and at least 0.025 with 9–16. Segments with fewer thumbnails
are never attached: none of those settings reached 90%. The thresholds come from
held-out matches, where hiding the number of known segments showed these
attachments right about nine times in ten (`evidence/reid_head.json`).
`scripts/extract_crops.py` keeps up to 24 thumbnails per tracklet (about 60,000
per half) so that more segments qualify.

Attaching segments by tactical role was tried first and measured the same way.
It picked the right player for only about 43% of the attached time, and was
removed. Segments that fit no player clearly stay unattributed; their events
still count in the team totals. The rates per 90 minutes on screen and per 100
touches do not depend on how much of a player was identified. If too few
numbers can be read in a half, the fallback groups segments into ten tactical
slots per team, shown by role (for example "left defender").

Kit groups are anonymous: set real team names under **Match context & team names**
in **Label players**. If an identity is wrong, **Correct an identity** merges it into another or renames
it. The CPU stages then re-run, and the correction is kept for later re-runs.
Review identities before labelling players.

## Archetype profiling

1. Open a player in **Label players** and check the identity against footage and
   thumbnails. Set their position group and rate compatible roles from 0 to 100.
   Definitions and evidence cues expand beside each role. Several roles can score
   highly; **Unknown** leaves a role unrated. The mixture shows shares of ratings.
2. Bookmark useful video moments and describe the behaviour. **Save & next**
   stores ratings, notes and bookmarks, then advances through the filtered queue.
   Unsaved drafts survive navigation and reloads on this browser; only saved
   labels are used by the model. Label varied appearances across matches and
   position groups, then use evaluation to assess whether more labels are needed.
3. **Models → Fit on all analysed players.** Each player appearance
   (one player in one half, at least 2 minutes visible) becomes a node. Its
   features are rates per 90 minutes on screen, the per-100-touch action mix,
   positional shares, a 4 × 4 heatmap and physical measures. These rates do not
   depend on how much of the player was identified. Within each position group,
   label spreading (Zhou et al., 2004) over an 8-nearest-neighbour graph gives
   every unlabelled appearance a percentage per archetype and a confidence. An unrated role counts as missing, not as 0.
   Grouped-by-match cross-validation compares the result with a labelled-only
   nearest-neighbour baseline.
4. Estimated profiles appear behind **Reveal model estimate** in the review desk
   and in **Models → Estimated profiles**. Suggestions stay collapsed while you
   review. The 42-role guide and independent interval reviews are in **More tools**.

See [annotation/ANNOTATION_GUIDE.md](annotation/ANNOTATION_GUIDE.md) for the
labelling rules.

## Measured accuracy

| Component | Test | Result |
| --- | --- | --- |
| Pass/carry logic | SoccerTrack v2 match 117092, annotated player positions and ball track, 1 s tolerance | Passes: precision 0.69/0.66, recall 0.70/0.69 (halves 1/2); any passer 0.76/0.74 precision. Per-player pass counts correlate 0.89 (Pearson) with the annotations. Carries vs 'Drive': 0.62/0.67 precision, 0.57/0.55 recall |
| Shots | SoccerNet shot labels, ±2 s; 13 train/validation halves and 4 held-out test halves | **Video spotter (used): 0.71 precision / 0.70 recall; test 0.78 / 0.63.** Shot model (fallback when no spots exist): 0.36 / 0.47; test 0.35 / 0.56 |
| Action spotter on other labelled classes | Same halves and tolerance, held-out test | Throw-ins 0.83 / 0.69, ball out of play 0.61 / 0.86. Summed throw-in scores 306 vs 324 labelled (well calibrated). Goals and free kicks are not comparable (the labels mark the goal line and the award) (`evidence/action_spotter.json`) |
| Tackle and block rules | SoccerTrack ground-truth positions and ball | Tackles 0.06–0.14 precision, 0.08–0.12 recall; blocks ≤ 0.10. Replaced by the spotter |
| Appearance re-identification | Held-out validation/test matches, numbers of known segments hidden | 72% correct overall (frozen CLIP 65%, role rule 43%); attached segments about 90% (`evidence/reid_head.json`) |
| Identity coverage | All 22 working-set halves (`evidence/coverage_diagnostics.json`) | 32% → 49% of detected player-time attributed; 100 → 218 appearances with ≥ 10 identified minutes |
| Shirt-number reader | SoccerNet Jersey 2023 test split, 1,211 tracklets | 80.3% of tracklets read correctly, including "no number visible" |
| Attack direction | Team positions at SoccerNet kick-off labels, 8 halves with a calibrated kick-off view | 8 of 8 correct |
| Calibration stability | Chelsea v Burnley H1: jitter of fixed image points projected to the pitch (second difference between samples, which cancels smooth camera pans) | 90th percentile 1.1 m → 0.18 m and 99th 9–18 m → 0.5 m after outlier rejection and smoothing |
| Throughput | RTX 3090, 720p | About 22–25 sampled frames per second: 25–30 minutes per half plus 1–1.5 minutes of CPU stages |

The pass and carry figures test the event logic on annotated positions. They do
not test the full chain from video. Reports are in
`D:\CVDL Football Data\PitchProfile\evidence` (`event_logic_soccertrack.json`,
`shot_model.json`, `jersey_number_reader.json`, `action_spotter.json`, `reid_head.json`,
`coverage_diagnostics.json`). [DETECTION_RESEARCH.md](DETECTION_RESEARCH.md) explains
the research behind these changes and what remains.

## Limitations

- **Only what the camera shows.** Players off screen get no events or distance.
  Replays that show the pitch from a calibratable angle are analysed like live
  play, so an action can occasionally count twice.
- **Identities come from shirt numbers and appearance.** About half of detected
  player-time is attributed (49% over the working set). 720p broadcast players
  are small, many are seen from the front or far away, and segments with fewer
  than 9 thumbnails are never matched by appearance. Counts are therefore
  partial. Compare players with the per-screen-time and per-touch rates, or the
  style profile. Each team keeps at most 14
  numbers per half, and in practice two or three of them are often misreads
  (for example a "1" that is really "10"). Such identities have little evidence
  and short visible time; merge them into the right player in the match view.
  Review before labelling.
- **The ball has no height.** A lofted ball projects onto the pitch where it is
  not, so aerial passes and headers are the least reliable events. Crosses have
  low recall (4 of 23 in SoccerTrack half 1).
- **Tackles, blocks and crosses from the video model run low.** It reports about
  one tackle for every eight in a fully annotated match (blocks and crosses about
  one in three or four), and which player it credits cannot be checked on this footage.
  Use them to compare players, not as totals. Take-ons are a tracking
  definition calibrated to plausible rates, with no local ground truth.
- **Shots:** the spotter misses about a third of labelled shots.
- **Direction and teams** come from positions and kit colours. Kit clusters are
  anonymous until you name them.
- Event definitions are video approximations of data-provider definitions. They
  do not reproduce any provider's event feed.

## Data and licences

Weights and the calibration code live on the data drive:
`D:\CVDL Football Data\PitchProfile\weights` and
`D:\CVDL Football Data\third_party\No-Bells-Just-Whistles`. Their licences differ
from this project's. The Roboflow detectors are AGPL-3.0 (via Ultralytics),
No-Bells-Just-Whistles is GPL-2.0, and the jersey-number weights are CC BY-NC 3.0
(non-commercial). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Analysed halves are stored as
`D:\CVDL Football Data\PitchProfile\data\datasets\sn-<date>-<home>-<away>-h<half>`.
