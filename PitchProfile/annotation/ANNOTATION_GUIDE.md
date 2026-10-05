# Archetype labelling

The 42 archetype definitions, evidence cues, position compatibility and source
links come from `AAI3001_Week_3_Project_Proposal_Final.docx`. They are kept in
[archetypes.v2.json](archetypes.v2.json) and shown under **Archetypes (42)** in
the app. Catalogue membership is not proof that a role can be learned
reliably. Pilot and freeze the rubric before collecting the final cohort.

There are two ways to label. The first feeds the semi-supervised model and is
the main workflow. The second is a stricter two-reviewer protocol.

## 1. Labelling analysed matches (semi-supervised)

Each label is one player's appearance in one analysed half.

1. Open the half in **Match statistics**. Name the two kit groups first.
2. Open a player. Check the identity: the thumbnails, and the box on the video
   when you click events. If two identities are the same person, or one mixes
   two people, use **Correct this identity** before labelling.
3. Watch enough of the player: their events (click to seek the video) and some
   stretches without the ball. Use the statistics and heatmap as support, not
   as the answer. **Stats & maps → Style profile** ranks each statistic
   against players in the same position group. It points to what stands out,
   such as many runs in behind or few passes, which you can then check in the
   footage. Tackles, blocks and headers there are expected counts from a video
   model and run low in absolute terms, so compare players rather than reading
   them as exact totals.
4. Set the **position group**. It decides which archetypes are offered. The
   group suggested from average position is only a starting point.
5. Rate **every compatible archetype from 0 to 100, independently**. This is not
   a budget: a centre forward can be 80 Poacher and 70 Pressing forward at once.
   Suggested anchors:
   - **0–20**: the role's behaviours are absent or incidental.
   - **40–60**: clearly present for part of the half.
   - **80–100**: the defining pattern of this player's half.
   Tick **insufficient evidence** when the footage cannot show a role (for
   example, too little time on screen in the relevant phase). An unrated role is
   left out of learning, not treated as 0. The mixture view shows the same
   ratings rescaled to shares summing to 100%.
6. Enter your labeller ID (the same one every time) and a short note on the
   decisive evidence, then save. Saving again replaces your label. The history
   keeps every version.

Rate what this half shows, not the player's reputation, name or usual
position. The same player in another match is a separate appearance, so label
it from that footage.

**How many.** Label about half of the appearances, spread across matches, teams
and all eight position groups. The model needs at least 2 labelled appearances
in a group to estimate that group, and at least 6 for cross-validation.
Players visible for less than 2 minutes in a half are left out of learning.

**Fit.** **Archetype learning → Fit on all analysed players** runs label
spreading over the statistics of all appearances in each position group,
labelled and unlabelled. It estimates percentages and a support value for every
unlabelled appearance. The report hides the labels of whole matches in turn.
It compares the mean absolute error (percentage points) and top-role agreement
with a nearest-neighbour model that uses only the labelled players. Report those
held-out numbers, not agreement on players you labelled yourself.

## 2. Independent interval reviews (stricter protocol)

**Interval reviews** supports a formal study with two independent reviewers.
Create a case for an identified player, match, period, exact start/end times and
reviewed position group. Features use that interval only. Cases are immutable,
and overlapping cases for the same player and source are rejected. Source
corrections invalidate earlier cases and their consensus. After correcting
identity or direction, create and review a replacement.

For the proposal pilot, review at least **20 minutes** and record at least
**three distinct supporting sequences** per reviewer, each with a start, end and
behavioural note. Short cases can be saved for preparation but cannot enter
training.

Ratings use the same 0–100 scale per role. Two reviewers' ratings for a role
**agree** when they are within 20 points (`taxonomy.AGREEMENT_TOLERANCE`), and
the settled value is their mean. A wider spread needs a third reviewer to
adjudicate with a numeric decision and a reason. Primary and secondary roles are
derived from the highest settled ratings at or above 30. They are not chosen
directly. Reviewers submit under different IDs, and the form loads only the
named review. Editing a review invalidates its adjudications. Original decisions
remain in the audit history. The local app does not enforce reviewer identity
or access control. Export cases, rubric, reviews, history and adjudications from
the interval screen.

Pilot **16–24 cases across eight position groups**. Keep rubric-development
examples out of the final test cohort. The proposal's initial target is 30
adjudicated cases, ten players and five matches per learned role, with positive
and negative examples. The 12-case training gate only verifies that training
runs; it is not evidence of sufficient support for 42 roles.

## Footage

The local SoccerNet library holds 23 complete broadcast matches (46 halves)
across six competitions. Each is analysed automatically, and the source video
is watched in the match view. SoccerTrack v2 match 117092 (panoramic video)
is also available. Provider tracking without video (the earlier PFF World Cup
import) is no longer used; see the README.

`cohort_review_template.csv` lists only the five centre-forward appearances of
the originally bundled package. It is a historical planning aid, not an import
format. The retired three-role **Legacy reviews** workflow recorded no reviews.
Its routes remain in the backend but are not linked from the interface.
