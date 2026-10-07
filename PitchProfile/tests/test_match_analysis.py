"""Full-match video analysis stages on synthetic evidence (no GPU models or match data)."""
import re
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from football_profiler import fieldcal as FC
from football_profiler import match_events as ME
from football_profiler import match_identity as MI
from football_profiler import match_pipeline as MPL
from football_profiler import match_post as MP
from football_profiler import match_stats as MS
from football_profiler import storage as S
from football_profiler.match_analysis import CutDetector

SHAPE = (720, 1280)


def camera():
    """World (pitch metres) -> image homography of a plausible broadcast view of the left half."""
    world = np.float32([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]])
    image = np.float32([[150, 260], [620, 250], [760, 560], [40, 640]])
    return cv2.getPerspectiveTransform(world, image)


def project(points, H):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def test_fit_recovers_pitch_positions_and_rejects_an_outlier_keypoint():
    H = camera()
    world = FC.KEYPOINTS[FC.GROUND]
    image = project(world, H)
    visible = (image[:, 0] > 0) & (image[:, 0] < 1280) & (image[:, 1] > 0) & (image[:, 1] < 720)
    world, image = world[visible], image[visible] + np.random.default_rng(0).normal(0, .4, (visible.sum(), 2))
    image[0] += [60, -45]  # one badly localised keypoint
    result = FC.fit(image, world, SHAPE)
    assert result is not None and result['valid']
    assert result['inliers'] == len(world) - 1
    probe = np.array([[8.0, 30.0], [14.0, 45.0], [3.0, 20.0]])
    assert np.abs(FC.to_pitch(project(probe, H), result['H']) - probe).max() < .3


def test_fit_rejects_keypoints_on_one_line():
    H = camera()
    world = np.array([[0, 13.84], [0, 24.84], [0, 30.34], [0, 37.66], [0, 43.16], [0, 54.16]])
    result = FC.fit(project(world, H), world, SHAPE)
    assert result is None or not result['valid']


def test_heatmap_decoding_is_sub_pixel_and_thresholded():
    import torch
    hm = torch.zeros(1, 58, 270, 480)
    yy, xx = torch.meshgrid(torch.arange(270.), torch.arange(480.), indexing='ij')
    hm[0, 4] = torch.exp(-((xx - 100.4) ** 2 + (yy - 50.6) ** 2) / 2)
    peaks = FC.decode(hm)
    assert peaks[0, 4, 0] == pytest.approx(200.8, abs=.35)
    assert peaks[0, 4, 1] == pytest.approx(101.2, abs=.35)
    assert np.isnan(peaks[0, 5, 0])


def test_ball_path_prefers_the_consistent_trajectory_and_bridges_short_gaps():
    samples, xy, conf = [], [], []
    for s in range(30):
        if s not in (14, 15, 16):
            samples.append(s); xy.append((100 + 10 * s, 400)); conf.append(.5)
        if s % 3 == 0:
            samples.append(s); xy.append((1100, 120)); conf.append(.3)  # static distractor
    picked = MP.select_ball(np.array(samples), np.array(xy, float), np.array(conf))
    chosen = np.array(xy)[picked]
    assert len(picked) == 27
    assert (chosen[:, 1] == 400).all()


def test_calibration_is_filled_inside_pitch_shots_only():
    H = np.linalg.inv(camera())
    H = H / H[2, 2]
    rows = []
    for s in range(20):
        shot = 0 if s < 10 else 1
        attempted = s % 2 == 0
        valid = attempted and shot == 0
        row = {'sample': s, 'source_frame': 2 * s, 'time_s': s / 12.5, 'shot': shot,
               'calibration_attempted': attempted, 'calibration_valid': valid, 'inliers': 8 if valid else 0,
               'rmse_px': 1.0 if valid else np.nan}
        row.update({f'h{k}': (H.ravel()[k] if valid else np.nan) for k in range(9)})
        rows.append(row)
    f = MP.fill_calibration(pd.DataFrame(rows), SHAPE)
    assert f.loc[f['sample'] < 10, 'pitch_view'].all()
    assert not f.loc[f['sample'] >= 10, 'pitch_view'].any()
    assert f.loc[1, 'calibration_source'] == 'interpolated'
    assert f.loc[9, 'calibration_source'] == 'held'
    assert f.loc[f['sample'] >= 10, 'h0'].isna().all()


def tracklet_table():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(16):
        team_colour = (150, 120, 180) if i < 8 else (80, 150, 110)
        rows.append({'tracklet': f't{i}', 'role': 'player', 'observations': 40, 'on_pitch_share': 1.0,
                     'kit_samples': 10, 'kit_L': team_colour[0] + rng.normal(0, 3), 'kit_a': team_colour[1] + rng.normal(0, 2),
                     'kit_b': team_colour[2] + rng.normal(0, 2), 'mean_x': (40 if i < 8 else 62) + rng.normal(0, 4)})
    rows.append({'tracklet': 'gk', 'role': 'goalkeeper', 'observations': 30, 'on_pitch_share': 1.0, 'kit_samples': 0,
                 'kit_L': np.nan, 'kit_a': np.nan, 'kit_b': np.nan, 'mean_x': 6.0})
    rows.append({'tracklet': 'ref', 'role': 'referee', 'observations': 30, 'on_pitch_share': 1.0, 'kit_samples': 0,
                 'kit_L': np.nan, 'kit_a': np.nan, 'kit_b': np.nan, 'mean_x': 50.0})
    return pd.DataFrame(rows).set_index('tracklet')


def test_kits_split_teams_and_goal_side_positioning_sets_direction_and_keeper():
    t, info = MP.assign_teams(tracklet_table())
    assert info['status'] == 'suggestions'
    first = t.loc['t0', 'team']
    assert (t.loc[[f't{i}' for i in range(8)], 'team'] == first).all()
    assert (t.loc[[f't{i}' for i in range(8, 16)], 'team'] != first).all()
    assert t.loc['ref', 'team'] == 'referee'
    directions = MP.attack_directions(t)
    assert directions['status'] == 'inferred' and directions['defends_left'] == first
    t = MP.assign_keepers(t, directions)
    assert t.loc['gk', 'team'] == first


def scenario():
    """Team A attacks right: A1 passes to A2, A2's pass is intercepted by B1, then A3 shoots."""
    hz = 12.5
    players = {'A1': ('A', (40, 34)), 'A2': ('A', (55, 34)), 'B1': ('B', (66, 34)), 'A3': ('A', (93, 34)),
               'B9': ('B', (80, 20))}
    ball = []
    for s in range(10):
        ball.append((s, 40.4, 34.0))                         # A1 controls
    for s in range(10, 20):
        ball.append((s, 40.4 + 1.5 * (s - 9), 34.0))        # pass to A2 (18.75 m/s)
    for s in range(20, 30):
        ball.append((s, 55.4, 34.0))                         # A2 controls
    for s in range(30, 36):
        ball.append((s, 55.4 + 1.8 * (s - 29), 34.0))       # A2 passes forward
    for s in range(36, 46):
        ball.append((s, 66.4, 34.0))                         # B1 intercepts and controls
    for s in range(50, 60):
        ball.append((s, 93.4, 34.0))                         # after a camera cut, A3 controls near goal
    for s in range(60, 64):
        ball.append((s, 93.4 + 2.0 * (s - 59), 34.0))       # shot at 25 m/s
    people = []
    for s in range(64):
        for name, (team, (x, y)) in players.items():
            people.append({'tracklet': f'{name}-v{int(s >= 50)}', 'track': name, 'sample': s, 'time_s': s / hz,
                           'view_shot': int(s >= 50),
                           'team': team, 'role': 'player', 'x': float(x), 'y': float(y),
                           'bbox_x': np.nan, 'bbox_y': np.nan, 'bbox_w': np.nan, 'bbox_h': np.nan, 'conf': .9})
    b = pd.DataFrame(ball, columns=['sample', 'x', 'y'])
    b['time_s'] = b['sample'] / hz
    b['view_shot'] = (b['sample'] >= 50).astype(int)
    b['cx'], b['cy'], b['interpolated'] = np.nan, np.nan, False
    directions = {'status': 'inferred', 'attacks_right': 'A', 'defends_left': 'A'}
    return pd.DataFrame(people), b, directions, hz


def test_touch_based_events_pass_and_interception():
    people, ball, directions, hz = scenario()
    ev, spells, pos, pp, seg = ME.events(people, ball, directions, hz)
    who = lambda s: re.sub(r'^seg-|-v\d+$', '', s) if isinstance(s, str) else s
    passes = ev[ev.type == 'pass']
    complete = passes[passes.outcome == 'complete']
    assert [(who(r.segment), who(r.receiver)) for r in complete.itertuples()] == [('A1', 'A2')]
    assert complete.length_m.iat[0] == pytest.approx(15, abs=1.5)
    intercepted = passes[passes.outcome == 'intercepted']
    assert [who(s) for s in intercepted.segment] == ['A2']
    assert [who(s) for s in ev[ev.type == 'interception'].segment] == ['B1']
    assert 'B1' in [who(s) for s in ev[ev.type == 'recovery'].segment]
    assert not (ev.type == 'shot').any()             # shots come from the video spotter, not from tracking rules
    # A ball rolling past B9 without changing course is not a touch.
    assert 'B9' not in {who(s) for s in spells.segment}


def test_spotted_shot_removes_the_rule_events_of_its_transition():
    ev = pd.DataFrame([
        {'type': 'pass', 'team': 'A', 'segment': 'A3', 'time_s': 10.0, 'outcome': 'intercepted'},     # the release itself
        {'type': 'recovery', 'team': 'B', 'segment': 'B1', 'time_s': 11.0, 'opponent': 'A3'},          # the keeper's recovery
        {'type': 'pass', 'team': 'A', 'segment': 'A2', 'time_s': 9.5, 'outcome': 'complete'},          # the set-up pass stays
        {'type': 'pass', 'team': 'A', 'segment': 'A1', 'time_s': 30.0, 'outcome': 'intercepted'},      # another moment
        {'type': 'interception', 'team': 'B', 'segment': 'B2', 'time_s': 12.0, 'opponent': 'A9'},      # another transition
        {'type': 'clearance', 'team': 'B', 'segment': 'B4', 'time_s': 10.1, 'outcome': 'cleared'}])    # the other team's
    shots = pd.DataFrame([{'team': 'A', 'time_s': 10.2}])
    assert list(MPL.drop_shot_transitions(ev, shots).segment) == ['A2', 'A1', 'B2', 'B4']
    assert len(MPL.drop_shot_transitions(ev, shots.iloc[0:0])) == len(ev)             # no shot, nothing removed


def test_stats_count_passes_and_scale_per_90():
    people, ball, directions, hz = scenario()
    ev, spells, pos, pp, seg = ME.events(people, ball, directions, hz)
    identity = {s: re.sub(r'^seg-|-v\d+$', '', s) for s in pp.segment.unique()}
    stats = MS.build(pp, ev, spells, identity, directions, hz, live_seconds=45 * 60)
    a2 = next(p for p in stats['players'] if p['identity'] == 'A2')
    assert a2['on_ball']['passes'] == 1 and a2['on_ball']['passes_completed'] == 0
    a1 = next(p for p in stats['players'] if p['identity'] == 'A1')
    assert a1['on_ball']['pass_completion'] == 1.0
    assert a1['per90']['passes'] == pytest.approx(2.0)
    assert a1['per100_touches']['passes'] == pytest.approx(100.0)   # one touch spell, one pass
    assert a1['per90_visible'] == {}                                  # under a minute on screen
    assert a1['positional']['mean_x'] == pytest.approx(40)


def test_shirt_numbers_anchor_identities_and_leave_unread_segments_out():
    rows, numbers = [], {}

    def add(name, start, yn, depth, number=None, votes=.6):
        rows.append({'segment': name, 'team': 'A', 'role': 'player', 'start_s': start, 'end_s': start + 10, 'n': 50,
                     'yn': yn, 'depth': depth, 'duration_s': 10.0})
        if number is not None:
            numbers[name] = {'number': number, 'confidence': 1.0, 'votes': votes}

    for k, start in enumerate((0, 100, 200)):
        add(f's7{k}', start, 10, .8, 7)          # left forward
        add(f's10{k}', start + 20, 34, .5, 10)   # central midfielder
        add(f's4{k}', start + 20, 58, .2, 4)     # right defender
    add('s7x', 5, 11, .8, 7, votes=.35)          # a weaker simultaneous #7 reading
    add('s36a', 40, 10.5, .79, 36, votes=.9)     # weak misread (26 -> 36 style)
    add('s36b', 140, 10.2, .81, 36, votes=.9)
    add('unread', 60, 34, .52)
    seg = pd.DataFrame(rows).set_index('segment')
    identity, rows_out = MI.resolve_numbers(seg, numbers, 'A')
    assert {r['number'] for r in rows_out} == {7, 10, 4}
    assert identity['s70'] == 'A-7' and identity['s100'] == 'A-10' and identity['s40'] == 'A-4'
    assert identity.get('s7x') != 'A-7'               # cannot wear the same shirt at the same time
    # Weak readings and unread segments are not attributed by role (measured to be mostly wrong).
    assert 's36a' not in identity and 's36b' not in identity and 'unread' not in identity


def test_off_ball_runs_in_behind_and_pressing():
    from football_profiler import match_movement as MV
    hz = 12.5
    rows = []
    # Team A attacks right. B's back line stands at x=80; A9 sprints from x=70 to x=92 while A holds the ball.
    mates = (('A2', 'A', 30.0, 20.0), ('A3', 'A', 30.0, 48.0))
    for s in range(60):
        t = s / hz
        for name, team, x, y in mates + (('A9', 'A', min(70 + 7 * t, 92), 34.0), ('A6', 'A', 50.0, 30.0),
                                 ('B2', 'B', 80.0, 20.0), ('B3', 'B', 80.0, 30.0), ('B4', 'B', 80.0, 40.0),
                                 ('B5', 'B', 80.0, 50.0)):
            rows.append({'sample': s, 'time_s': t, 'segment': name, 'team': team, 'role': 'player', 'x': x, 'y': y})
    # Then B wins the ball; A6 sprints about 13 m to the B carrier, B3.
    for s in range(60, 110):
        t = (s - 60) / hz
        for name, team, x, y in mates + (('A9', 'A', 92.0, 34.0), ('A6', 'A', min(64 + 6 * t, 77.5), 30.0),
                                 ('B2', 'B', 80.0, 20.0), ('B3', 'B', 80.0, 30.0), ('B4', 'B', 80.0, 40.0),
                                 ('B5', 'B', 80.0, 50.0)):
            rows.append({'sample': s, 'time_s': s / hz, 'segment': name, 'team': team, 'role': 'player', 'x': x, 'y': y})
    people = pd.DataFrame(rows)
    spells = pd.DataFrame([{'segment': 'A6', 'team': 'A', 'start_sample': 0, 'end_sample': 55},
                           {'segment': 'B3', 'team': 'B', 'start_sample': 60, 'end_sample': 109}])
    directions = {'status': 'inferred', 'attacks_right': 'A', 'defends_left': 'A'}
    out = MV.movement_stats(people, spells, {'A9': 'A-9', 'A6': 'A-6'}, directions, hz)
    assert out['A-9']['runs_in_behind'] == 1 and out['A-9']['forward_runs'] == 1
    assert out['A-6']['pressing_runs'] == 1
    assert out['A-9']['mean_x_vs_team_m'] > out['A-6']['mean_x_vs_team_m']   # the striker plays higher
    assert not any(k.startswith('_') for k in out['A-9'])        # no internal accumulators leak out


def test_percentile_profile_ranks_against_same_position_peers(monkeypatch):
    from football_profiler import match_profiles as PF
    peers = pd.DataFrame([{'dataset_id': 'd', 'identity': f'p{i}', 'group': 'centre_forward', 'shots': float(i)}
                          for i in range(10)] + [{'dataset_id': 'd', 'identity': 'cb', 'group': 'centre_back', 'shots': 0.0}])
    monkeypatch.setattr(PF, 'peer_table', lambda: peers)
    player = {'role': 'player', 'visible_seconds': 900, 'per90_visible': {'shots': 7.0}}
    prof = PF.profile('d', player, 'centre_forward')
    shots = next(s for sec in prof['sections'] for s in sec['stats'] if s['key'] == 'shots')
    assert prof['peer_basis'] == 'same position group' and prof['peers'] == 10
    assert shots['percentile'] == pytest.approx(70.0)            # 7 of 10 peers below
    assert PF.profile('d', player, 'full_back')['peer_basis'] == 'all outfield players'   # too few full-backs
    assert PF.profile('d', {'role': 'goalkeeper'}, 'goalkeeper') is None


def test_appearance_attaches_unread_segments_only_when_clearly_decided():
    rows, numbers, emb = [], {}, {}
    axes = {7: np.array([1.0, 0, 0]), 10: np.array([0, 1.0, 0]), 4: np.array([0, 0, 1.0])}

    def add(name, start, number=None, vec=None, crops=20):
        rows.append({'segment': name, 'team': 'A', 'role': 'player', 'start_s': start, 'end_s': start + 10, 'n': 50,
                     'yn': 30.0, 'depth': .5, 'duration_s': 10.0})
        if number is not None:
            numbers[name] = {'number': number, 'confidence': 1.0, 'votes': .6}
        if vec is not None:
            emb[name] = (vec / np.linalg.norm(vec), crops)

    for k, start in enumerate((0, 100, 200)):
        for n, offset in ((7, 0), (10, 20), (4, 40)):
            add(f's{n}{k}', start + offset, n, axes[n])
    add('looks7', 60, vec=np.array([.95, .2, .1]))              # clear match, 20 thumbnails
    add('ambiguous', 300, vec=np.array([.7, .69, 0]))            # between #7 and #10
    add('few_crops', 400, vec=np.array([1.0, 0, 0]), crops=6)   # too few thumbnails to trust
    add('during7', 5, vec=np.array([1.0, 0, 0]))                 # overlaps #7's own segment
    seg = pd.DataFrame(rows).set_index('segment')
    identity, rows_out = MI.resolve_numbers(seg, numbers, 'A', emb)
    assert identity['looks7'] == 'A-7'
    assert 'ambiguous' not in identity and 'few_crops' not in identity
    assert identity.get('during7') != 'A-7'                      # one player cannot be in two places
    assert next(r for r in rows_out if r['number'] == 7)['appearance_segments'] == 1


def analysed_dataset(identifier='sn-20150101-home-away-h1'):
    players = []
    stats_players = []
    rng = np.random.default_rng(3)
    for team in ('A', 'B'):
        for n in range(1, 12):
            pid = f'{team}-{n}'
            players.append({'player_id': pid, 'name': f'Team {team} #{n}', 'team': f'Team {team}', 'team_key': team,
                            'role': 'Outfield player', 'position_group': 'central_midfield' if n % 2 else 'centre_forward',
                            'jersey': n, 'direction': 'right', 'direction_known': True, 'eligible_frames': 100,
                            'playing_seconds': 1800})
            heat = rng.random((20, 32))
            stats_players.append({
                'identity': pid, 'team': team, 'role': 'player', 'segments': 10, 'visible_seconds': 600.0,
                'physical': {'distance_m': 1500.0, 'moving_seconds': 500.0, 'distance_per_min_m': 110.0,
                             'top_speed_kmh': 28.0, 'sprints': 3, 'zone_seconds': {'high_speed': 20.0, 'sprint': 5.0}},
                'positional': {'heatmap': (heat / heat.sum()).tolist(), 'mean_x': 40 + 3 * n, 'mean_y': 34.0,
                               'spread_x': 8.0, 'spread_y': 9.0, 'defensive_third': .3, 'middle_third': .4,
                               'attacking_third': .3, 'central_lane': .5, 'box_share': .05 * (n % 3)},
                'on_ball': {'touches': 20 + n, 'passes': 10 + n, 'passes_completed': 8, 'pass_completion': .8,
                            'time_on_ball_s': 30.0, 'mean_pass_length_m': 15.0, 'forward_pass_share': .5},
                'per90': {'touches': 40.0 + n, 'passes': 30.0 + n, 'shots': .5 * (n % 4), 'dribbles': 1.0}})
    manifest = {'id': identifier, 'title': 'Home v Away · 2015-01-01 · H1', 'match_id': 'soccernet:x', 'analysis': 'full-match-1',
                'source': 'test', 'source_kind': 'model_predictions', 'sampling_hz': 12.5, 'duration_seconds': 2700,
                'total_sampled_frames': 100, 'players': players, 'video': None,
                'teams': {'A': {'name': 'Team A', 'colour': '#ff0000', 'attacks': 'right'},
                          'B': {'name': 'Team B', 'colour': '#0000ff', 'attacks': 'left'}}}
    d = S.dataset_dir(identifier, create=True)
    S.write_json(d / 'manifest.json', manifest)
    S.write_json(d / 'match_stats.json', {'players': stats_players, 'teams': [], 'per90_factor': 3.0, 'live_seconds': 1800})
    S.write_json(d / 'events.json', {'events': [{'type': 'pass', 'identity': 'A-3', 'time_s': 12.0, 'outcome': 'complete'},
                                                {'type': 'touch', 'identity': 'A-3', 'time_s': 11.0}]})
    S.write_json(d / 'identities.json', {'segments': {'seg-1': {'identity': 'A-3'}}, 'segment_tracks': {'seg-1': ['s1-t4']}})
    ok, jpg = cv2.imencode('.jpg', np.zeros((60, 30, 3), np.uint8))
    with zipfile.ZipFile(d / 'crops.zip', 'w') as z:
        z.writestr('s1-t4/12.jpg', jpg.tobytes())
    S.write_json(d / 'analysis.json', {'postprocess': {'version': 'full-match-1'}})
    S.save_tracks(identifier, pd.DataFrame({'frame': [0, 1] * 22, 'time_s': [0, .08] * 22,
                                            'player_id': [p['player_id'] for p in players for _ in range(2)],
                                            'x': 50.0, 'y': 34.0, 'detected': 1, 'calibration_valid': 1}))
    return identifier


def test_match_api_statistics_and_crops(client):
    identifier = analysed_dataset()
    base = f'/api/datasets/{identifier}'
    summary = client.get(base + '/match').json()
    assert len(summary['players']) == 22 and summary['teams']['A']['name'] == 'Team A'
    assert 'heatmap' not in summary['players'][0]['positional']
    events = client.get(base + '/match/events?player=A-3').json()['events']
    assert [e['type'] for e in events] == ['pass']
    crops = client.get(base + '/match/crops/A-3').json()['crops']
    assert len(crops) == 1 and client.get(crops[0]).headers['content-type'] == 'image/jpeg'
    assert 'label' not in summary['players'][0]


def test_team_names_rename_players(client):
    identifier = analysed_dataset()
    r = client.post(f'/api/datasets/{identifier}/match/team-names', json={'A': 'Chelsea', 'B': 'Burnley'})
    assert r.status_code == 200
    manifest = S.read_json(S.dataset_dir(identifier) / 'manifest.json')
    assert manifest['teams']['A']['name'] == 'Chelsea'
    assert next(p for p in manifest['players'] if p['player_id'] == 'B-4')['name'] == 'Burnley #4'


def test_missing_image_boxes_is_a_valid_empty_evidence_response(client):
    identifier = analysed_dataset()
    r = client.get(f'/api/datasets/{identifier}/match/boxes/A-3')
    assert r.status_code == 200
    assert r.json()['boxes'] == []


def test_cut_detector_fires_on_shot_changes_only():
    detect = CutDetector()
    grass = np.zeros((360, 640, 3), np.uint8); grass[:] = (40, 140, 40)
    crowd = np.zeros((360, 640, 3), np.uint8); crowd[:] = (30, 30, 160)
    assert detect(grass) is False
    assert not detect(grass.copy())
    assert detect(crowd)


def test_soccernet_naming_and_fixture_parsing():
    game = 'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'
    assert MPL.soccernet_identifier(game, 1) == 'sn-20150221-chelsea-burnley-h1'
    assert MPL.fixture_teams('2016-09-13 - 21-45 Barcelona 7 - 0 Celtic') == ('Barcelona', 'Celtic')
    assert MPL.suggest_position_group('goalkeeper', {}) == 'goalkeeper'
    assert MPL.suggest_position_group('player', {'mean_x': 25, 'mean_y': 34}) == 'centre_back'
    assert MPL.suggest_position_group('player', {'mean_x': 75, 'mean_y': 60}) == 'wide_attacker'


def test_batch_naming_stores_names_per_segment_and_undo_removes_them(client, monkeypatch):
    from football_profiler import app as APP
    jobs = []
    monkeypatch.setattr(APP, 'start_job', lambda kind, func, **kw: jobs.append((kind, kw)) or {'job_id': 'j1'})
    identifier = analysed_dataset()
    d = S.dataset_dir(identifier)
    S.write_json(d / 'identities.json', {'segments': {'seg-1': {'identity': 'A-X2'}, 'seg-2': {'identity': 'A-X2'},
                                                      'seg-3': {'identity': 'B-X1'}}, 'segment_tracks': {}})
    base = f'/api/datasets/{identifier}/match/identities'
    r = client.post(base + '/names', json={'names': [{'source': 'A-X2', 'number': 8}, {'source': 'B-X1', 'number': 11}]})
    assert r.status_code == 200 and r.json()['job_id'] == 'j1' and len(jobs) == 1
    assert S.read_json(d / 'confirmed_identities.json') == {'seg-1': 'A-8', 'seg-2': 'A-8', 'seg-3': 'B-11'}
    assert client.post(base + '/names', json={'names': [{'source': 'A-X9', 'number': 8}]}).status_code == 400
    assert client.post(base + '/names', json={'names': [{'source': 'A-X2', 'number': 100}]}).status_code == 422
    assert client.post(base + '/undo', json={'identity': 'A-8'}).status_code == 200
    assert S.read_json(d / 'confirmed_identities.json') == {'seg-3': 'B-11'}
    assert client.post(base + '/undo', json={'identity': 'A-8'}).status_code == 400


def test_marking_not_a_player_removes_its_segments_from_every_player(client, monkeypatch):
    from football_profiler import app as APP, match_identity as MI
    monkeypatch.setattr(APP, 'start_job', lambda kind, func, **kw: {'job_id': 'j2'})
    identifier = analysed_dataset()
    d = S.dataset_dir(identifier)
    S.write_json(d / 'identities.json', {'segments': {'seg-1': {'identity': 'A-X2'}}, 'segment_tracks': {}})
    base = f'/api/datasets/{identifier}/match/identities'
    assert client.post(base + '/names', json={'names': [{'source': 'A-X2', 'not_player': True}]}).status_code == 200
    assert S.read_json(d / 'confirmed_identities.json') == {'seg-1': 'NONE'}
    assert client.get(f'/api/datasets/{identifier}/match').json()['not_player_segments'] == 1
    assert client.post(base + '/names', json={'names': [{'source': 'A-X2', 'number': 5, 'not_player': True}]}).status_code == 422
    assert client.post(base + '/names', json={'names': [{'source': 'A-X2'}]}).status_code == 422
    seg = pd.DataFrame({'team': ['A'], 'n': [5], 'yn': [30.0], 'depth': [.5], 'duration_s': [3.0]}, index=['seg-1'])
    out, _ = MI.apply_confirmed({'seg-1': 'A-X2'}, pd.DataFrame([{'identity': 'A-X2'}]), {'seg-1': 'NONE'}, seg)
    assert out == {}
    assert client.post(base + '/undo', json={'identity': 'NONE'}).status_code == 200
    assert S.read_json(d / 'confirmed_identities.json') == {}


def test_corrections_follow_players_through_renames_swaps_and_conflicts(client):
    from football_profiler import match_pipeline as MPL
    identifier = analysed_dataset()
    with S.db() as c:
        for pid, text in (('A-X1', 'one'), ('A-X2', 'two'), ('A-X3', 'three'), ('A-16', 'sixteen')):
            c.execute('INSERT INTO player_identity_links(dataset_id,player_id,payload) VALUES(?,?,?)', (identifier, pid, text))
    old = {'A-X1': ['s1', 's2'], 'A-X2': ['s3'], 'A-X3': ['s4'], 'A-16': ['s5']}
    # X1 was named 16 (joining the existing #16), X2 and X3 swapped their unnamed numbers.
    identity = {'s1': 'A-16', 's2': 'A-16', 's3': 'A-X3', 's4': 'A-X2', 's5': 'A-16'}
    moves = MPL.player_moves(old, identity, {'s1': 10, 's2': 5, 's3': 7, 's4': 9, 's5': 30})
    assert moves == {'A-X1': 'A-16', 'A-X2': 'A-X3', 'A-X3': 'A-X2'}
    S.move_player_records(identifier, moves)
    with S.db() as c:
        def value(pid):
            row = c.execute('SELECT payload FROM player_identity_links WHERE dataset_id=? AND player_id=?', (identifier, pid)).fetchone()
            return row['payload'] if row else None
        assert value('A-X3') == 'two' and value('A-X2') == 'three'
        assert value('A-16') == 'sixteen'                # the player's own record is never replaced...
        assert 'one' in (value('A-X1~kept'), value('A-X1'))      # ...and the moved one is kept, not lost
        assert c.execute('SELECT COUNT(*) FROM player_identity_links WHERE dataset_id=?', (identifier,)).fetchone()[0] == 4
