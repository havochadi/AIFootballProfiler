import numpy as np
import pytest

from football_profiler import spotter_data as SD


def bit(name):
    return 1 << (0 if name == 'background' else SD.INDEX[name])


def test_bas_targets_spread_label_side_and_displacement():
    events = [{'frame': 10, 'label': 'PLAYER SUCCESSFUL TACKLE', 'side': 'right'}]
    allow, displ, team = SD.targets(events, 30, 'bas')
    assert all(allow[i] == bit('PLAYER SUCCESSFUL TACKLE') for i in range(6, 15))
    assert allow[5] == allow[15] == bit('background')
    assert displ[6] == -4 and displ[10] == 0 and displ[14] == 4
    assert team[10] == 1 and team[5] == -1


def test_later_event_wins_where_windows_overlap():
    events = [{'frame': 10, 'label': 'PASS', 'side': 'left'}, {'frame': 13, 'label': 'DRIVE', 'side': 'right'}]
    allow, displ, team = SD.targets(events, 30, 'bas')
    assert allow[11] == bit('DRIVE') and displ[11] == -2 and team[11] == 1
    assert allow[8] == bit('PASS') and team[8] == 0


def test_footpass_labels_are_partial():
    events = [{'frame': 20, 'label': 'Pass', 'left_to_right': 0}, {'frame': 50, 'label': 'Block', 'left_to_right': 1}]
    allow, _, team = SD.targets(events, 80, 'footpass')
    assert allow[0] == bit('background') | bit('OUT') | bit('GOAL')
    assert allow[20] == bit('PASS') | bit('HIGH PASS') | bit('FREE KICK')
    assert allow[50] == bit('BALL PLAYER BLOCK')
    assert team[20] == 0 and team[50] == 1          # left_to_right 0 = the team defending the left goal


def test_partial_label_loss_is_weighted_cross_entropy_for_exact_labels():
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    logits = torch.randn(2, 5, SD.NUM_CLASSES)
    labels = torch.randint(0, SD.NUM_CLASSES, (2, 5))
    weights = torch.full((SD.NUM_CLASSES,), 5.0)
    weights[0] = 1.0
    ours = SD.partial_label_loss(logits, 1 << labels, weights)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, SD.NUM_CLASSES), labels.reshape(-1), reduction='none')
    assert torch.allclose(ours.reshape(-1), ce * weights[labels.reshape(-1)], atol=1e-5)


def test_vectorised_double_head_matches_devkit(monkeypatch):
    torch = pytest.importorskip('torch')
    from pathlib import Path
    from football_profiler import action_spotting as AS
    devkit = Path(r'D:\CVDL Football Data\third_party\sn-teamspotting')     # tests redirect the data folder
    if not (devkit / 'model' / 'modules.py').is_file():
        pytest.skip('sn-teamspotting devkit not on this machine')
    monkeypatch.setattr(AS, 'devkit_dir', lambda: devkit)
    kit = AS._import_devkit()
    torch.manual_seed(1)
    pred = torch.randn(3, 40, 31)
    displ = torch.randn(3, 40) * 3
    displ[0, :5] = torch.tensor([.5, 1.5, -.5, 2.5, -2.5])      # rounding ties
    ours = AS.double_head(pred, displ, num_classes=13)
    theirs = kit.process_double_head(pred, displ, num_classes=13)
    assert torch.allclose(ours, theirs)


def test_won_ball_side_is_opposite_of_last_confident_touch():
    from football_profiler import action_spotting as AS
    spots = [{'label': 'PASS', 'side': 'left', 'time_s': 10.0, 'score': .9},
             {'label': 'SHOT', 'side': 'right', 'time_s': 11.0, 'score': .8},
             {'label': 'DRIVE', 'side': 'left', 'time_s': 11.5, 'score': .1},     # not confident
             {'label': 'BALL PLAYER BLOCK', 'side': 'right', 'time_s': 12.0, 'score': .4}]
    assert AS.won_ball_side(spots, spots[-1]) == 'left'
    assert AS.won_ball_side(spots, {'time_s': 30.0}) is None                   # nothing within 4 s


def test_partial_label_loss_sums_allowed_probabilities():
    torch = pytest.importorskip('torch')
    logits = torch.zeros(1, 1, SD.NUM_CLASSES)
    allow = torch.tensor([[bit('PASS') | bit('HIGH PASS')]])
    loss = SD.partial_label_loss(logits, allow, torch.ones(SD.NUM_CLASSES))
    assert float(loss) == pytest.approx(-np.log(2 / SD.NUM_CLASSES), abs=1e-5)


def test_spotter_fills_only_passes_the_rules_missed():
    import pandas as pd
    from football_profiler import match_pipeline as MP
    rules = pd.DataFrame([{'type': 'pass', 'team': 'A', 'time_s': 10.0}, {'type': 'carry', 'team': 'B', 'time_s': 30.0}])
    spot = lambda kind, team, t, score=.9, confident=True: {
        'type': kind, 'label': kind.upper(), 'team': team, 'time_s': t, 'score': score, 'confident': confident,
        'segment': 's1', 'view_shot': 1, 'x': 50.0, 'y': 30.0, 'attack_sign': 1}
    found = pd.DataFrame([spot('pass', 'A', 10.4),            # the rules already have it
                          spot('pass', 'B', 10.4),            # other team: new
                          spot('high_pass', 'A', 20.0),       # missed by the rules: new
                          spot('pass', 'A', 20.3),            # same moment as the high pass: not twice
                          spot('pass', 'A', 40.0, .1, False),  # not confident
                          spot('drive', 'B', 30.5),           # the rules have the carry
                          spot('drive', 'A', 50.0)])          # new carry
    filled = MP.fill_from_spotter(found, rules)
    got = sorted((r.type, r.team, r.time_s) for r in filled.itertuples())
    assert got == [('carry', 'A', 50.0), ('pass', 'A', 20.0), ('pass', 'B', 10.4)]
    assert filled[filled.type.eq('pass')].outcome.eq('unknown').all()
