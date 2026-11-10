"""evaluate_spotter_labelled.py reports the throw-in team rule from cached spots (no model run)."""
import json
from types import SimpleNamespace

from scripts.evaluate_spotter_labelled import measure


def test_throw_in_team_is_reported_for_each_window(tmp_path):
    # The right team throws at 20 s; the spotter says left; the left team's last touch was 12 s before.
    spots = [{'label': 'PASS', 'side': 'left', 'frame': 100, 'time_s': 8.0, 'score': .9},
             {'label': 'THROW IN', 'side': 'left', 'frame': 250, 'time_s': 20.0, 'score': .9}]
    (tmp_path / 'spots_test.json').write_text(json.dumps({'spots': spots}), encoding='utf-8')
    video = SimpleNamespace(dataset='bas', name='game', folder=tmp_path,
                            events=[{'label': 'THROW IN', 'frame': 250, 'side': 'right'},
                                    {'label': 'PASS', 'frame': 100, 'side': 'left'}])
    report = measure([video], 'test')
    row = report['THROW IN']['by_threshold']['0.05']
    assert row['matched'] == 1 and row['team_correct'] == 0.0
    assert row['team_correct_previous_touch_rule'] == 0.0                    # 4 s: no touch found, spotter's side
    assert row['team_correct_previous_touch_within_s'] == {'10': 0.0, '20': 1.0, '30': 1.0}
    assert 'team_correct_previous_touch_within_s' not in report['PASS']['by_threshold']['0.05']
