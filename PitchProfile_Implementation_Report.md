# PitchProfile — implementation and evidence report

> **Checkout audit, 22 September 2026:** this supplied report describes a later
> package than the ZIP present in this repository. The actual original archive
> contains one SkillCorner match (32 players, five centre forwards), SoccerNet
> reference tracks, and a short video preview. Four extra matches, the auxiliary
> reconstruction experiment, tests, and operating files described below were
> absent. The working directory now adds setup, GPU support, runtime fixes,
> and a new test suite. The proposal's 42-role catalogue, interval-aligned reviews,
> version 2 training path and authorised SoccerTrack importer have also been
> added. This does not supply real archetype labels or validation results.
> Use `PitchProfile/README.md` and
> `PitchProfile/LOCAL_VERIFICATION.md` for current instructions and executed
> results. The historical claims below are retained for context and are not
> verification of this checkout.

Status: **working local prototype; final archetype validation incomplete**. Prepared for the AAI3001 football player profiling project. The team's archetype assignments remain the source of reference labels.

## Delivered implementation

| Component | Implemented behaviour | Practical boundary |
|---|---|---|
| Video analysis | Upload, CPU YOLO11n person detection, ByteTrack, saved overlay and progress | Short recorded clips; person IDs need review; no automatic named-player recognition |
| Pitch mapping | Manual landmarks, homography, optional ORB camera alignment, coverage flags | Geometry checks and frame-loading tested; no independent moving-camera accuracy benchmark |
| Profiles | Observed-only heatmaps, thirds, coverage, source attribution and matched earlier event history | Off-screen movement is not reconstructed in displayed profiles |
| Team labels | Three overlapping centre-forward archetypes, independent reviews, unknowns, third-reviewer adjudication, history and exports | Matching evidence and real team reviews are still needed |
| Supervised pipeline | Logistic baseline, CNN plus numeric features, CNN with observation gaps, grouped splits and saved reports | Save/load and evaluation paths tested using disposable artificial labels; no real archetype classifier trained |
| Auxiliary deep learning | Residual CNN trained on masked real reference heatmaps | One-match preparatory experiment; not archetype accuracy or complete off-screen recovery |
| Interface | Player profiles, reviews, experiments, imports, exports, identity and fixture correction | Local app; one worker and a three-job queue; no shared-service authentication |
| Reproducibility | Pinned dependencies, weights/checksum, prepared data, code, tests, source manifests and operating guide | Linux/Python 3.12 installation tested; Docker and other OS instructions not executed |

## Data actually acquired

The package contains **30 centre-forward appearances, 23 distinct centre forwards and 5 matches**. All 30 pass the provisional 30-observed-second/20%-coverage check. Across all included SkillCorner players there are 57 player/match cases and 1,572,485 supplied coordinate records. Four extra raw tracking files were actually downloaded and checked against their Git LFS hashes; they are not just links to potential data.

The five fixtures are Auckland–Newcastle (30 November 2024), Melbourne Victory–Western United (10 January 2025), Melbourne City–Macarthur (7 March 2025), Western United–Auckland (3 May 2025), and Melbourne Victory–Auckland (17 May 2025). Extra matches retain centre forwards only to keep the runnable package small. The original fixture retains 32 participating player records.

There are **zero real archetype reviews**, and no real archetype model checkpoint. Matching raw footage for those five fixtures is not included. This remains the central evidence gap: team members need appropriate same-match evidence before assigning labels. Twenty-match event histories are useful supporting data but do not replace temporal/player alignment or human review. Transfermarkt is not used as archetype ground truth or an event/video source.

For Guillermo May in the 17 May fixture, the playing-period denominator is 58,449 samples. The provider supplies 40,404 coordinate samples (69.1%), of which 22,569 are flagged as detected (38.6%). Twelve detected points fall outside the standard pitch. The displayed heatmap uses **22,557 valid observed points**, or 38.6% of eligible samples. The 17,835 estimated coordinate samples are excluded from the heatmap.

## Executed model evidence

### Detector sanity check

On the first SoccerNet SNGS-060 image, the pinned pretrained detector matched **17 of 17 reference people** with 17 predictions at IoU 0.5 and confidence 0.25. This yields precision and recall of 1.0 **on that single image only**. The reference includes players, goalkeepers and referees. This is not mAP, a tracking benchmark, general football detection accuracy or archetype accuracy. The check can be rerun using the bundled image and first-frame reference boxes.

### Recorded-video runtime check

The public publisher preview is already annotated. After conversion it is 3.4 seconds long. The pipeline processed 17 sampled frames at five samples/second and produced 8 track IDs. The recorded tracking-loop time was approximately 1.73 seconds on the available CPU, excluding video transcoding and general application startup. This demonstrates execution, not accurate identity continuity or sustained real-time operation. The browser test additionally exercised an actual upload and background analysis job.

### Auxiliary CNN experiment

The network learned a correction to a smoothing baseline using **2,161 30-second windows from 32 players in one real match**. Windows require at least 100 observed positions. There were 1,210 training windows from 18 players, 576 validation windows from seven players, and **375 test windows from seven different players**. Player groups do not overlap. The smoothing parameter and CNN checkpoint were selected on validation players; test players were evaluated afterwards.

A contiguous fraction of the observed-record sequence was removed. The target is the original observed heatmap, which is itself incomplete. Jensen–Shannon divergence uses natural logarithms; smaller is better, with zero meaning identical distributions.

| Removed observed records | Observed remainder | Gaussian smoothing | Residual CNN |
|---|---:|---:|---:|
| 20% | 0.0426 | 0.0494 | 0.0441 |
| 40% | 0.1104 | 0.1123 | 0.1072 |
| 60% | 0.2018 | 0.1980 | 0.1947 |

The CNN improved over both baselines at 40% and 60% removal, but the observed-only baseline was better at 20%. The gains are modest. Correlated windows, one match and seven test players limit inference. This is preliminary evidence for studying observation-gap robustness; it does not establish a novel architecture, correct tactical archetypes, recovered off-screen paths or generalisation to new matches. The app never substitutes these reconstructions into a player's observed heatmap.

The prior broad-position diagnostic remains in `evidence/diagnostic_baseline.json`. It predicts broad position groups, not the three archetypes, and must not be reported as archetype-classification performance.

## Verification

- **15 automated tests passed**, zero failures, in a clean isolated Python 3.12 installation. They cover observed/estimated separation, gap masking, unknown-target gradients, disjoint group splits, independent reviewer identities, adjudication invalidation, invalid imports, homography geometry, recalibration orientation, camera-cut rejection, chronological history, API/export behaviour and model save/load. The supervised integration test uses explicitly artificial fixtures in a temporary directory.
- **8 browser workflows passed**, with no JavaScript page errors or server errors: real profile display, review agreement, experiment display/training gate, reference/model distinction, calibration frame loading, video upload/processing, fixture confirmation and a 390-pixel mobile viewport without page overflow.
- Official YOLO11n weights are pinned and checksum-verifiable. Additional public source downloads include their hashes. No real-team review database was populated by the tests.
- Two upstream deprecation warnings appeared in the test client. They did not cause failures; dependencies are pinned in the delivered requirements.

This verifies the implemented prototype's behaviour. It does **not** measure HOTA/IDF1, camera localisation error against independent pitch points, event-recognition accuracy, expert agreement on archetypes, or live-stream reliability.

## Alignment with the briefing

| Briefing requirement | Evidence now available | Remaining work |
|---|---|---|
| Slide 3: curated dataset and own annotation | Real transformed provider/reference data, provenance, blank cohort worksheet, independent-review application | Team must obtain matching evidence and create the actual reference labels |
| Slide 3: model contribution | Baselines, custom small CNN, gap augmentation implementation, executed auxiliary comparison | Real-label archetype experiments, ablations and broader evaluation; no claim of established research novelty |
| Slide 3: deployment | Executed local application and clean installation; Docker recipe | Cloud deployment, operational controls and scalability testing |
| Slide 3: usable interface | Functional desktop/mobile interface and eight browser checks | User testing with team members and workflow refinement |
| Week 3 compulsory hurdle | Earlier proposal and project flow diagram remain the planning submission | Team details and submission via the course channel are the team's responsibility |
| Week 6 and Week 13 | Runnable demo, reproducible source, evidence and five-minute demo script | Final research results, poster, recorded backup presentation, team repository publication and course submission |

## Next concrete team work

1. Acquire suitable matching footage or other sufficiently informative independent evidence for the selected player/match intervals. If an interval differs from the bundled whole-match features, prepare aligned interval features too.
2. Freeze the rubric and cohort; have two team members review each case independently, then adjudicate disagreements. Leave unsupported labels unknown. Review the full 30-case cohort before making model-performance claims.
3. Check label balance and group coverage. Run the predeclared match holdout, then a separate verified-player holdout if adequate. Report per-label results and all baseline comparisons, including negative findings.
4. Evaluate the video pipeline on several independently annotated clips: person detection, identity switches, coverage and pitch localisation. Provider-trajectory success is not automatically uploaded-video success.
5. Extend only after those results: more matched cases, robust identity correction, action recognition and deployment. Five matches and 23 centre forwards are enough to run a small pilot, not enough to guarantee a successful or general-purpose scouting model.

The strongest defensible statement at this stage is that the project is **technically implementable as a bounded prototype, and a functioning implementation now exists**. Whether its archetype predictions are useful remains an empirical question requiring the team's evidence and labels.
