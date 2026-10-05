# PitchProfile demo site

A self-contained website for the 10-minute demo. It walks through the product the way a user meets it:
**upload a match video → the analysis runs → match report with per-player statistics, plus the system's
tracking drawn over the real footage.** Archetype profiling is deliberately left out.

Demo match: **Chelsea v Manchester United, 7 Feb 2016, 1st half** (SoccerNet broadcast, Premier League).
Manchester United is the red kit (group A), Chelsea the blue kit (group B). The pipeline itself only knows
"kit group A / B"; the names were confirmed by eye on a frame showing the broadcast scoreboard (CHE v MU).
**It finished 1-1.** It is the only Manchester United match in the local library; there is no United win
among the 23 downloaded matches.

## Run it

1. **Connect the D: drive.** The Live tracking overlay plays the original video from
   `D:\CVDL Football Data\SoccerNet\videos-720p\england_epl\2015-2016\2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United\1_720p.mkv`.
   Nothing is copied: no footage is stored in this folder. If the drive is missing, the page says so and offers
   **Locate 1_720p.mkv** to pick the file by hand.
2. Double-click [index.html](index.html) in Chrome or Edge, full screen. No install, no server, no internet.

## What is real and what is simulated

| Part | Status |
| --- | --- |
| Statistics, heatmaps, event maps, the boxes on the footage and the pitch positions | **Real.** Stored output of the actual pipeline on this half |
| The video itself | **Real**, played from the drive |
| The processing screen | **Simulated timing.** A real analysis of this half took about 33 min on the RTX 3090 (23 min detection and tracking, plus thumbnails, action spotting and CPU stages), so the demo plays a 36 s replay. The stages' totals are the real run's. The boxes appear in the order the pipeline finds things: white boxes, then team colours, the ball, shirt numbers |
| Uploading your own video | The file is only read in the browser for its length, size and resolution. Any upload leads to the **sample match's** report, and the page says so. A file named like `1_720p.mkv` is also used as the footage source |

The main app today analyses the SoccerNet library halves from the **Matches** screen. Arbitrary file upload into the
full-match pipeline does not exist there yet; this demo shows the intended flow.

## Tabs of the report

- **Match overview**: team comparison, standout players, touches per 5 minutes, where each team had the ball.
- **Live tracking**: four 75-second clips of the real footage with boxes, shirt numbers and the ball, beside the same players mapped onto the pitch. Click a player on the video or the pitch to follow them. Layer toggles, speed and scrubber.
- **Players**: sortable table (totals or per-90 on screen), search, CSV download, full per-player profile.
- **Event map**: 12 event types, routes or heatmap, per team or player. Opens on the top passer's passes.
- **About this analysis**: pipeline, measured accuracy, limitations.

## Minimum footage length

Measured over **all 22 analysed halves**, on the first N minutes of each (players on screen inside the calibrated
pitch view, using the app's own 2-minute threshold for profiling):

| Footage | Players with ≥ 2 min on screen, median (range) |
| --- | --- |
| 2 min | 0 (0–1) |
| 5 min | 6 (0–19) |
| **10 min (minimum)** | 20 (13–24) |
| 15 min | 23 (14–31) |
| 45 min (full half) | 34 (26–43); 18 of them have ≥ 10 min |

So: **10 minutes minimum, a full half recommended.** Five minutes is risky (one half had no player past the
threshold). The demo blocks clips under 3 minutes and warns under 10. Caveat: identities were decided on whole
halves, so a short clip on its own would name fewer players.

## Suggested 10-minute flow

1. **Upload (1 min).** Press *Use the sample match*. Show the checks and the footage-length table.
2. **Analyse (2 min).** Press *Analyse match*. Narrate as the stages tick: white boxes (detection), red and blue (teams), yellow ring (ball), numbers appear (shirt reading). The pitch on the right fills in as positions are mapped.
3. **Report (6 min).**
   - *Live tracking*: play the third clip (it contains a model-detected shot), click a player, show the pitch view.
   - *Match overview* and *Players*: sort by distance, switch to per-90, open a player.
   - *Event map*: Shots and the Heatmap view.
   - *About this analysis*: say the limits out loud: tackles run low (†), about half of player time is named, counts describe only what the camera showed, and edge players are sometimes missed.
4. **Close (1 min).** Hand over to the archetype slides.

## Rebuild the data

```powershell
PitchProfile\.venv\Scripts\python.exe PitchProfileDemo\build_demo_data.py
```

Reads the analysed half from `D:\CVDL Football Data\PitchProfile\data\datasets` and writes `data.js`
(numbers only, about 1.6 MB). To use another match, edit the constants at the top of the script
(dataset id, team names, video path, clip start times).
