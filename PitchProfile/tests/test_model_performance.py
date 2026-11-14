"""The model performance page reads whichever reports exist and applies the README's decision rules."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from football_profiler import event_live as EL
from football_profiler import model_performance as MPF
from football_profiler import storage as S


def spotter_report(mean, classes):
    return {'mean_average_precision': mean,
            'classes': {c: {'labels': n, 'average_precision': ap} for c, (n, ap) in classes.items()}}


def write_reports(evidence, ablation=True):
    for tag, fp, bas in (('published', .478, .606), ('finetuned', .604, .628),
                         *((('tdeed_ablation_partial', .600, .630), ('tdeed_ablation_hard', .570, .610)) if ablation else ())):
        S.write_json(evidence / f'action_spotter_footpass_val_{tag}.json',
                     spotter_report(fp, {'Pass': (900, fp + .3), 'Tackle': (40, .08), 'Goal': (0, 0.0)}))
        S.write_json(evidence / f'action_spotter_bas_test_{tag}.json', spotter_report(bas, {'PASS': (500, .9)}))
    actions = lambda right: {'Pass': {'labels': 900, 'found': .9, 'credited_correctly': .34, 'credited_to_right_track': .59,
                                      'accuracy_when_credited': .5},
                             'Throw-in': {'labels': 60, 'found': .68, 'credited_correctly': right,
                                          'credited_to_right_track': .2, 'accuracy_when_credited': .3}}
    S.write_json(evidence / 'identity_footpass_variant_current.json', {'overall': {'actions': actions(.062)}})
    S.write_json(evidence / 'identity_footpass_variant_throwinline.json', {'overall': {'actions': actions(.15)}})
    S.write_json(evidence / 'credit_reasons.json', {
        'datasets': ['a', 'b'], 'reasons': ['no team', 'team not on screen', 'no ball', 'ball away from players'],
        'overall': {'rule_off': {'pass': {'spots': 100, 'credited': .75, 'picture': .5, 'pitch': .25, 'touchline': 0,
                                          'no ball': .095, 'ball away from players': .09, 'team not on screen': .055,
                                          'no team': .01}}}})
    S.write_json(evidence / 'spotter_changes.json', {
        'timing': {'settings': {'before': {'minutes_per_half': 4.5, 'speed_up': 1.0, 'largest_score_change': None},
                                'read_ahead': {'minutes_per_half': 3.0, 'speed_up': 1.5, 'largest_score_change': 0.0},
                                'read_ahead_half_overlap': {'minutes_per_half': 1.6, 'speed_up': 2.8,
                                                            'largest_score_change': .2}}},
        'held_out': {'footpass_val': {'mean_average_precision': {'current': .604, 'half_overlap': .601, 'change': -.003},
                                      'throw_in_team': {'spotter': .6, 'last_touch_4s': .55, 'last_touch_30s': .8}},
                     'bas_test': {'mean_average_precision': {'current': .628, 'half_overlap': .626, 'change': -.002},
                                  'throw_in_team': {'spotter': .62, 'last_touch_30s': .84}}}})
    S.write_json(evidence / 'event_logic_soccertrack.json', {'halves': {
        '1': {'passes_any_player_1s': {'matched': 70, 'predicted': 100, 'annotated': 100}, 'per_player_pass_counts': {'pearson': .9}},
        '2': {'passes_any_player_1s': {'matched': 68, 'predicted': 100, 'annotated': 100}, 'per_player_pass_counts': {'pearson': .88}}}})


def test_everything_measured_gives_scores_and_verdicts(tmp_path):
    write_reports(tmp_path)
    g = MPF.gather(tmp_path)
    fp = g['spotter']['sets']['footpass_val']
    assert fp['mean_average_precision'] == {'published': .478, 'finetuned': .604, 'tdeed_ablation_partial': .6,
                                            'tdeed_ablation_hard': .57}
    assert set(fp['average_precision']) == {'Pass', 'Tackle'}                 # classes without labels left out
    assert g['rules']['passes_any_player_1s']['precision'] == .69 and g['rules']['pass_count_correlation'] == .89
    assert g['headline'] == {'spotter_map': {'value': .604, 'from': .478}, 'pass_found': .9, 'pass_right_player': .34,
                             'pass_right_track': .59, 'pass_count_correlation': .89}
    c = {x['id']: x for x in g['changes']}
    assert c['C3']['state'] == 'measured' and c['C3']['verdict'] == 'the trick helps'
    assert c['C4a']['result'].startswith('Uncredited passes: no ball 38%, ball away from players 36%')
    assert c['C4b']['verdict'] == 'switch on' and '6% -> 15%' in c['C4b']['result']
    assert c['C5a']['verdict'] == 'keep on'
    assert c['C5b']['verdict'] == 'switch on'                  # mean AP moved under 0.005 on both sets
    assert c['C6']['verdict'] == 'switch on'                   # the 30 s look-back beats the spotter on both
    assert c['C6']['result'] == 'FOOTPASS: spotter 60%, last touch (30 s) 80%; SoccerNet BAS: spotter 62%, last touch (30 s) 84%'
    assert g['missing'] == ['own_labels_<tag>.json']


def test_decision_rules_say_no_when_the_numbers_say_no(tmp_path):
    write_reports(tmp_path)
    worse = S.read_json(tmp_path / 'identity_footpass_variant_current.json')
    worse['overall']['actions']['Throw-in']['accuracy_when_credited'] = .9         # the rule would lower it
    S.write_json(tmp_path / 'identity_footpass_variant_current.json', worse)
    changes = S.read_json(tmp_path / 'spotter_changes.json')
    changes['held_out']['bas_test']['mean_average_precision']['change'] = -.02
    changes['held_out']['bas_test']['throw_in_team'] = {'spotter': .9, 'last_touch_30s': .8}
    changes['timing']['settings']['read_ahead']['largest_score_change'] = .01
    S.write_json(tmp_path / 'spotter_changes.json', changes)
    hard = spotter_report(.60, {'Pass': (900, .9)})
    S.write_json(tmp_path / 'action_spotter_footpass_val_tdeed_ablation_hard.json', hard)
    S.write_json(tmp_path / 'action_spotter_bas_test_tdeed_ablation_hard.json', spotter_report(.63, {'PASS': (500, .9)}))
    c = {x['id']: x for x in MPF.gather(tmp_path)['changes']}
    assert c['C3']['verdict'] == 'too close to call with one run each'
    assert c['C4b']['verdict'] == 'leave off'
    assert c['C5a']['verdict'].startswith('scores changed')
    assert c['C5b']['verdict'].startswith('team decides')
    assert c['C6']['verdict'] == 'leave off'


def test_nothing_measured_still_lists_the_changes(tmp_path):
    g = MPF.gather(tmp_path)
    assert [c['id'] for c in g['changes']] == ['C1', 'C2', 'C3', 'C4a', 'C4b', 'C5a', 'C5b', 'C6']
    assert {c['id'] for c in g['changes'] if c['state'] == 'waiting'} == {'C3', 'C4b', 'C5b', 'C6'}
    assert g['headline'] == {} and g['sources'] == []
    assert 'loss' not in ' '.join(g['missing']) and 'action_spotter_footpass_val_tdeed_ablation_hard.json' in g['missing']


def test_without_the_ablation_the_page_shows_what_exists(tmp_path):
    write_reports(tmp_path, ablation=False)
    g = MPF.gather(tmp_path)
    assert set(g['spotter']['sets']['bas_test']['mean_average_precision']) == {'published', 'finetuned'}
    assert {x['id']: x for x in g['changes']}['C3']['state'] == 'waiting'


def test_newest_own_labels_report_is_used(tmp_path):
    S.write_json(tmp_path / 'own_labels_a.json', {'created': '2026-11-14T10:00', 'overall': {'labels': 10, 'found': .5}})
    S.write_json(tmp_path / 'own_labels_b.json', {'created': '2026-11-15T10:00', 'overall': {'labels': 12, 'found': .75}})
    S.write_json(tmp_path / 'own_labels_bad.json', {})
    g = MPF.gather(tmp_path)
    assert g['own_labels']['overall']['found'] == .75 and g['headline']['own_labels_found'] == .75


def test_api_reads_the_evidence_folder(tmp_path, monkeypatch):
    write_reports(tmp_path)
    monkeypatch.setattr(S, 'EVIDENCE', tmp_path)
    app = FastAPI()
    app.include_router(EL.router)
    data = TestClient(app).get('/api/model-performance').json()
    assert data['headline']['pass_found'] == .9 and len(data['changes']) == 8


def test_gather_script_saves_one_file(tmp_path, capsys, monkeypatch):
    import sys
    from scripts import gather_model_performance as GMP
    write_reports(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['gather_model_performance.py', '--evidence', str(tmp_path)])
    GMP.main()
    out = capsys.readouterr().out
    assert 'Ours (in the app)' in out and 'C4b' in out and 'switch on' in out
    assert S.read_json(tmp_path / 'model_performance.json')['headline']['pass_right_player'] == .34
