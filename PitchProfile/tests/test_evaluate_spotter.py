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


def test_check_spotter_changes_compares_half_overlap_and_the_throw_in_team():
    from scripts.check_spotter_changes import compare, show
    throw = {'labels': 9, 'average_precision': .8, 'best_f1_threshold': .3,
             'by_threshold': {'0.3': {'team_correct': .81, 'team_correct_previous_touch_rule': .82,
                                      'team_correct_previous_touch_within_s': {'10': .85, '20': .9, '30': .88}}}}
    report = lambda m, pass_ap: {'mean_average_precision': m, 'classes': {
        'PASS': {'labels': 50, 'average_precision': pass_ap}, 'THROW IN': throw}}
    out = compare({'bas_test': {'finetuned': report(.628, .93), 'finetuned_half_overlap': report(.621, .92)}},
                  'finetuned', 'finetuned_half_overlap')
    bas = out['bas_test']
    assert bas['mean_average_precision'] == {'current': .628, 'half_overlap': .621, 'change': -.007}
    assert bas['average_precision_change'] == {'PASS': -.01, 'THROW IN': 0.0}
    assert bas['throw_in_team'] == {'spotter': .81, 'last_touch_4s': .82, 'last_touch_10s': .85,
                                    'last_touch_20s': .9, 'last_touch_30s': .88}
    assert out['footpass_val'] == {}                                       # no reports yet
    text = show({'held_out': out})
    assert 'mean AP 0.628 -> 0.621' in text and 'no reports yet' in text
