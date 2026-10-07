# PitchProfile demo site

A self-contained website for the 10-minute demo. It walks through the product the way a user meets it:
**upload a match video → the analysis runs → match report with per-player statistics, plus the system's
tracking drawn over the real footage.**

Demo match: **Chelsea v Manchester United, 7 Feb 2016, the full match** (both halves, 95:20 of SoccerNet
broadcast video, Premier League). Manchester United is the red kit (group A), Chelsea the blue kit (group B).
The pipeline itself only knows "kit group A / B"; the names were confirmed by eye on frames from both halves
showing the broadcast scoreboard (CHE v MU). **It finished 1-1.** It is the only Manchester United match in the
local library; there is no United win among the 23 downloaded matches.

## Run it

1. **Connect the D: drive.** The Live tracking overlay plays the app's joined video of both halves from
   `D:\CVDL Football Data\PitchProfile\data\datasets\sn-20160207-chelsea-manchester-united\sn-20160207-chelsea-manchester-united.mkv`
   (about 2 GB). Nothing is copied: no footage is stored in this folder. If the drive is missing, the page says so
   and offers **Locate ...mkv** to pick the file by hand.
2. Double-click [index.html](index.html) in Chrome or Edge, full screen. No install, no server, no internet.

## What is real and what is simulated

| Part | Status |
| --- | --- |
| Statistics, heatmaps, event maps, the boxes on the footage and the pitch positions | **Real.** Stored output of the actual pipeline on this match (both halves, merged by the app; the second half is mirrored so each team attacks one way) |
| The video itself | **Real**, played from the drive |
| The processing screen | **Simulated timing.** A real analysis of this match takes about 70 min on the RTX 3090 (51 min detection and tracking for both halves, plus thumbnails, action spotting and CPU stages), so the demo plays a 36 s replay. The stages' totals are the real run's. The boxes appear in the order the pipeline finds things: white boxes, then team colours, the ball, shirt numbers |
| Uploading your own video | The file is only read in the browser for its length, size and resolution. Any upload of 40 minutes or more leads to the **sample match's** report, and the page says so; shorter files are blocked. A file whose name contains "manchester" is also used as the footage source |

The main app today analyses the SoccerNet library halves from the **Matches** screen. Arbitrary file upload into the
full-match pipeline does not exist there yet; this demo shows the intended flow.

## Tabs of the report

- **Match overview**: team comparison, standout players, touches per 5 minutes with the half-time line, where each team had the ball.
- **Live tracking**: six 75-second clips of the real footage (three per half) with boxes, shirt numbers and the ball, beside the same players mapped onto the pitch. Click a player on the video or the pitch to follow them. Layer toggles, speed and scrubber.
- **Players**: the 22 players seen for 10+ minutes (11 per team); a checkbox adds the 34 shorter appearances, which are mostly fragments of the same people. Sortable table (totals or per-90 on screen), search, CSV download, full per-player statistics.
- **Event map**: 12 event types, routes or heatmap, per team or player. Opens on the top passer's passes.
- **About this analysis**: pipeline, measured accuracy, limitations.

## Footage length

The models are built and tested on whole halves and whole matches, so the upload step accepts a full half (about
45 minutes) or a full match (about 90 minutes). Anything under 40 minutes is blocked with a message that says so.
Measured over **all 22 analysed halves**, a half typically gives 34 analysed players (range 26–43), 18 of them on
screen for at least 10 minutes. Identities were decided on whole halves, which is why shorter footage is not offered.

## Suggested 10-minute flow

1. **Upload (1 min).** Press *Use the sample match*. Show the checks.
2. **Analyse (2 min).** Press *Analyse match*. Narrate as the stages tick: white boxes (detection), red and blue (teams), yellow ring (ball), numbers appear (shirt reading). The pitch on the right fills in as positions are mapped.
3. **Report (6 min).**
   - *Live tracking*: play a first-half and a second-half clip (the "Model-detected shot" ones are good), click a player, show the pitch view.
   - *Match overview* and *Players*: sort by distance, switch to per-90, open a player.
   - *Event map*: Shots and the Heatmap view.
   - *About this analysis*: say the limits out loud: tackles run low (1 of 26 found on unseen games), about half of player time is named (52%, with 95% of the names right), counts describe only what the camera showed, and edge players are sometimes missed.
4. **Close (1 min).** Hand back to the slides.

## Rebuild the data

```powershell
PitchProfile\.venv\Scripts\python.exe PitchProfileDemo\build_demo_data.py
```

Reads the analysed match from `D:\CVDL Football Data\PitchProfile\data\datasets` and writes `data.js`
(numbers only, about 2.6 MB). To use another match, edit the constants at the top of the script
(dataset id, team names, video path, clip start times).
