"""Our own labelled clip is matched to the app's events fairly: one event per label, same action, within 1 s."""
import csv
import sys

import pytest

from football_profiler import storage as S
from scripts import own_labels as OL

TEAMS = {'A': {'name': 'Alpha'}, 'B': {'name': 'Beta'}}
PLAYERS = [{'player_id': 'A-7', 'team_key': 'A', 'jersey': 7}, {'player_id': 'A-9', 'team_key': 'A', 'jersey': 9},
           {'player_id': 'B-4', 'team_key': 'B', 'jersey': 4}]


def sheet(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(OL.COLUMNS)
        w.writerows(rows)
    return path


def test_times_actions_teams_and_numbers_are_read_leniently():
    assert OL.parse_time('1:05') == 65 and OL.parse_time('1:05.5') == 65.5 and OL.parse_time('1:00:00') == 3600
    assert OL.parse_time(' 42 ') == 42
    with pytest.raises(ValueError):
        OL.parse_time('1m05')
    assert OL.action_of('Throw In') == 'throw-in' and OL.action_of('drive') == 'carry'
    with pytest.raises(ValueError):
        OL.action_of('goal kick')
    assert OL.team_of('beta', TEAMS) == 'B' and OL.team_of('a', TEAMS) == 'A'
    assert OL.team_of('unknown', TEAMS) is None and OL.team_of('Gamma', TEAMS) is None
    assert OL.jersey_of('#10') == 10 and OL.jersey_of('?') is None and OL.jersey_of('') is None


def test_clip_start_moves_labels_onto_the_video_clock(tmp_path):
    path = sheet(tmp_path / 'l.csv', [['0:10', '2', 'pass', 'Alpha', '7', '', ''], ['0:05', '2', 'shot', 'B', '', '', '']])
    rows = OL.read_labels(path, {'half': 2, 'teams': TEAMS}, clip_start_s=600)
    assert [(r['time_s'], r['action'], r['team'], r['jersey']) for r in rows] == \
        [(605.0, 'shot', 'B', None), (610.0, 'pass', 'A', 7)]
    whole = {'halves': ['h1', 'h2'], 'second_half_offset_s': 2700, 'teams': TEAMS}
    assert OL.read_labels(path, whole, clip_start_s=3300)[0]['time_s'] == 3305.0    # whole match: its own clock
    with pytest.raises(ValueError, match='row 2'):                                # a half-2 label on a first half
        OL.read_labels(path, {'half': 1, 'teams': TEAMS})


def test_each_label_takes_the_nearest_event_of_its_action_once():
    labels = [{'time_s': 10.0, 'action': 'pass'}, {'time_s': 10.4, 'action': 'pass'}, {'time_s': 20, 'action': 'cross'}]
    events = [{'type': 'pass', 'time_s': 10.3}, {'type': 'shot', 'time_s': 10.0},
              {'type': 'pass', 'time_s': 20.2, 'cross': True}]
    pairs, used = OL.match(labels, events)
    assert [e and e['time_s'] for _, e in pairs] == [10.3, None, 20.2]         # the second pass has nothing left
    assert used == {0: 'pass', 2: 'cross'}


def test_scores_found_team_player_and_precision():
    labels = [{'row': 2, 'time_s': 10.0, 'action': 'pass', 'team': 'A', 'jersey': 7},
              {'row': 3, 'time_s': 15.0, 'action': 'pass', 'team': 'A', 'jersey': 9},
              {'row': 4, 'time_s': 30.0, 'action': 'shot', 'team': 'B', 'jersey': None},
              {'row': 5, 'time_s': 40.0, 'action': 'tackle', 'team': 'B', 'jersey': 4}]
    events = [{'type': 'pass', 'source': 'rules', 'time_s': 10.2, 'team': 'A', 'identity': 'A-7'},
              {'type': 'pass', 'source': 'rules', 'time_s': 15.5, 'team': 'A', 'identity': 'A-7'},    # wrong player
              {'type': 'pass', 'source': 'rules', 'time_s': 25.0, 'team': 'A', 'identity': 'A-9'},    # not labelled
              {'type': 'shot', 'source': 'action_spotter', 'confident': True, 'time_s': 30.5, 'team': 'A'},
              {'type': 'tackle', 'source': 'action_spotter', 'confident': False, 'time_s': 40.0, 'team': 'B'},
              {'type': 'pass', 'source': 'rules', 'time_s': 99.0, 'team': 'A', 'identity': 'A-9'}]    # outside clip
    r = OL.score(labels, events, PLAYERS, window=(9, 41))
    p = r['actions']['pass']
    assert (p['labels'], p['found'], p['events_in_clip'], p['precision']) == (2, 1.0, 3, .667)
    assert p['right_player'] == .5 and p['team_right_when_found'] == 1.0 and p['credited_when_found'] == 1.0
    s = r['actions']['shot']
    assert s['found'] == 1.0 and s['team_right_when_found'] == 0.0 and s['labels_with_jersey'] == 0 and s['right_player'] is None
    assert r['actions']['tackle']['found'] == 0.0                     # the low-score spot is not counted
    o = r['overall']
    assert (o['labels'], o['found'], o['events_in_clip'], o['precision']) == (4, .75, 4, .75)
    assert [x['event_model'] for x in r['rows']] == ['rules', 'rules', 'video', None]


def test_check_writes_a_report_the_performance_page_reads(isolated_data, tmp_path, monkeypatch, capsys):
    from football_profiler import model_performance as MPF
    evidence = tmp_path / 'evidence'
    monkeypatch.setattr(S, 'EVIDENCE', evidence)
    d = S.dataset_dir('sn-test-h1', create=True)
    S.write_json(d / 'manifest.json', {'id': 'sn-test-h1', 'title': 'Alpha v Beta', 'half': 1, 'teams': TEAMS,
                                       'players': PLAYERS})
    S.write_json(d / 'events.json', {'events': [
        {'type': 'pass', 'source': 'rules', 'time_s': 610.2, 'team': 'A', 'identity': 'A-7'},
        {'type': 'throw_in', 'source': 'action_spotter', 'confident': True, 'time_s': 640.0, 'team': 'B', 'identity': 'B-4'}]})
    labels = sheet(tmp_path / 'clip.csv', [['0:10', '1', 'pass', 'Alpha', '7', 'Mount', ''],
                                           ['0:40', '1', 'throw-in', 'Beta', '4', '', ''],
                                           ['0:50', '1', 'header', 'Gamma', '?', '', 'kit unclear']])
    monkeypatch.setattr(sys, 'argv', ['own_labels.py', 'check', '--dataset', 'sn-test-h1', '--labels', str(labels),
                                      '--clip-start', '10:00', '--clip-length', '1:00', '--tag', 'demo'])
    OL.main()
    out = capsys.readouterr().out
    assert 'Gamma' in out                                              # unknown team names are reported
    report = S.read_json(evidence / 'own_labels_demo.json')
    assert report['window_s'] == [600.0, 660.0] and report['overall']['found'] == .667
    assert report['actions']['throw-in']['right_player'] == 1.0
    assert MPF.gather(evidence)['own_labels']['labels_file'] == 'clip.csv'


def test_template_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / 'labels' / 'sheet.csv'
    monkeypatch.setattr(sys, 'argv', ['own_labels.py', 'template', str(path)])
    OL.main()
    assert path.read_text().strip() == ','.join(OL.COLUMNS)
    with pytest.raises(SystemExit):
        OL.main()


def test_clip_is_cut_from_the_dataset_video_at_the_exact_start(isolated_data, tmp_path, monkeypatch, capsys):
    cv2 = pytest.importorskip('cv2')
    pytest.importorskip('imageio_ffmpeg')
    import numpy as np
    d = S.dataset_dir('sn-clip-h1', create=True)
    writer = cv2.VideoWriter(str(d / 'v.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    for i in range(100):                                      # 4 s; frame i is i * 2 grey
        writer.write(np.full((48, 64, 3), i * 2, np.uint8))
    writer.release()
    S.write_json(d / 'manifest.json', {'id': 'sn-clip-h1', 'title': 'Clip test', 'video': 'v.avi', 'half': 1})
    out = tmp_path / 'clip.mp4'
    monkeypatch.setattr(sys, 'argv', ['own_labels.py', 'clip', '--dataset', 'sn-clip-h1', '--start', '2',
                                      '--length', '1', '--out', str(out)])
    OL.main()
    assert '--clip-start 2 --clip-length 1' in capsys.readouterr().out
    video = cv2.VideoCapture(str(out))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = video.read()
    assert ok and 24 <= frames <= 26
    assert abs(float(first.mean()) - 100) < 6                 # frame 50, not a keyframe before it
