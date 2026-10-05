"""Interval reviews use artificial evidence and temporary storage only."""
import numpy as np
import pandas as pd
import pytest

from football_profiler import cases as C, features as F, learning as L, storage as S, taxonomy as T


def create_case(sample_dataset, **changes):
    identifier, _, _, _ = sample_dataset
    return C.create(**dict(dataset_id=identifier, player_id='p1', period=1, start_s=0,
                           end_s=1, position_group='centre_forward', **changes))


def review(case, reviewer, value=100, **changes):
    labels = dict.fromkeys(case['labels'])
    labels['target_man'] = value
    payload = dict(reviewer=reviewer, labels=labels, evidence='Artificial evidence only',
                   sequences=[{'start_s': case['start_s'] + i * .1,
                               'end_s': case['start_s'] + i * .1 + .05,
                               'note': 'Artificial sequence'} for i in range(3)])
    payload.update(changes)
    return C.save_review(case['id'], **payload)


def test_catalogue_has_proposal_roles_and_compatibility():
    assert len(T.ROLES) == 42 and len(T.GROUPS) == 8
    assert not T.ROLES['complete_forward']['trainable']
    assert 'ball_winner' in T.compatible('central_midfield')
    assert 'ball_winner' in T.compatible('defensive_midfield')
    assert 'sweeper_keeper' not in T.compatible('centre_forward')
    # Every reviewable role has teaching examples; they are never reference labels.
    for role in T.ROLES.values():
        assert len(role['examples']) >= 2, role['id']
        assert any(e['kind'] == 'scenario' for e in role['examples'])
        for example in role['examples']:
            assert example['title'] and example['description'] and example['context']
            if example['kind'] == 'player_comparison':
                assert example['source'].startswith('https://')


def test_interval_features_and_overlap_are_exact(sample_dataset):
    identifier, player, manifest, rows = sample_dataset
    case = C.create(identifier, 'p1', 1, .2, .6, 'centre_forward')
    assert case['profile']['eligible_frames'] == 4
    assert case['profile']['valid_position_frames'] == 4
    assert case['profile']['events'] is None
    assert case['profile']['feature_vector'][13:] == [0] * 5
    assert not case['pilot_duration_met']
    assert C.create(identifier, 'p1', 1, .2, .6, 'centre_forward')['id'] == case['id']
    with pytest.raises(ValueError, match='overlapping'):
        C.create(identifier, 'p1', 1, .5, .8, 'centre_forward')
    with pytest.raises(ValueError, match='beyond'):
        C.create(identifier, 'p1', 1, 0, 2, 'centre_forward')
    assert L.predict(case['profile'])['status'] == 'insufficient_evidence'


def test_independent_reviews_history_adjudication_and_staleness(sample_dataset):
    case = create_case(sample_dataset)
    review(case, 'Alice')
    result = review(case, 'Bob', 0)
    assert result['labels']['target_man']['status'] == 'disagreement'
    with pytest.raises(ValueError, match='third reviewer'):
        C.adjudicate(case['id'], 'target_man', 100, 'ALICE', 'Artificial reasoning')
    C.adjudicate(case['id'], 'target_man', 100, 'Carol', 'Artificial reasoning')
    assert C.consensus(case['id'])['labels']['target_man']['status'] == 'adjudicated'
    review(case, 'Bob')
    assert C.consensus(case['id'])['primary_role'] == 'target_man'
    assert C.export()['adjudications'] == []
    assert len(C.export()['review_history']) == 3
    assert S.reviews() == []
    assert C.training_profiles() == []  # One-second case is not a training case.
    identifier, _, manifest, _ = sample_dataset
    manifest['players'][0]['name'] = 'Revised identity'
    S.write_json(S.dataset_dir(identifier) / 'manifest.json', manifest)
    assert C.get(case['id'])['stale']
    assert C.consensus(case['id'])['labels']['target_man']['value'] is None
    with pytest.raises(ValueError, match='Source data changed'):
        review(case, 'Alice')


def test_review_validation_and_private_api(client, sample_dataset):
    case = create_case(sample_dataset)
    with pytest.raises(ValueError, match='insufficient'):
        review(case, 'Alice', labels=dict.fromkeys(case['labels']))
    with pytest.raises(ValueError, match='inside'):
        review(case, 'Alice', sequences=[{'start_s': 0, 'end_s': 2, 'note': 'Outside interval'}])
    with pytest.raises(ValueError):
        review(case, 'Alice', labels={'sweeper_keeper': 1})
    review(case, 'Alice')
    result = client.get(f"/api/cases/{case['id']}/reviews?reviewer=Bob").json()
    assert result == {'own_review': None, 'reviewer_count': 1}
    assert len(client.get('/api/taxonomy').json()['roles']) == 42
    assert client.get('/api/cases/summary').json()['review_rows'] == 1
    assert client.get('/api/cases/export').json()['rubric']['version'] == '2.0'


def test_percentage_consensus_tolerance_band_and_range_validation(sample_dataset):
    case = create_case(sample_dataset)
    with pytest.raises(ValueError, match='percentage'):
        review(case, 'Alice', 101)
    with pytest.raises(ValueError, match='percentage'):
        review(case, 'Alice', -1)
    review(case, 'Alice', 40)
    within_tolerance = review(case, 'Bob', 55)  # spread 15 <= AGREEMENT_TOLERANCE (20)
    status = within_tolerance['labels']['target_man']
    assert status['status'] == 'agreed' and status['spread'] == 15 and status['value'] == pytest.approx(47.5)
    assert within_tolerance['primary_role'] == 'target_man'
    beyond_tolerance = review(case, 'Bob', 85)  # Bob revises the same case; spread now 45 > tolerance
    status = beyond_tolerance['labels']['target_man']
    assert status['status'] == 'disagreement' and status['spread'] == 45 and status['value'] is None
    assert beyond_tolerance['primary_role'] is None
    with pytest.raises(ValueError, match='percentage'):
        C.adjudicate(case['id'], 'target_man', 150, 'Carol', 'Artificial reasoning')
    resolved = C.adjudicate(case['id'], 'target_man', 62, 'Carol', 'Artificial reasoning')
    assert resolved['labels']['target_man'] == {'value': 62, 'spread': None, 'status': 'adjudicated', 'reviewer_count': 2}


def test_provider_partitions_cannot_be_reassigned():
    cases = [{'benchmark_split': s} for s in ['train', 'train', 'validation', 'test']]
    groups = ['a', 'a', 'b', 'c']
    splits = L.case_splits(cases, groups, 42, 'match')
    assert [x.tolist() for x in splits] == [[0, 1], [2], [3]]
    with pytest.raises(ValueError, match='all three'):
        L.case_splits(cases[:2], groups[:2], 42, 'match')
    with pytest.raises(ValueError, match='multiple'):
        L.case_splits(cases, ['a'] * 4, 42, 'match')
    with pytest.raises(ValueError, match='separate match'):
        L.case_splits(cases, groups, 42, 'player')


def test_interval_training_roundtrip_on_configured_gpu(isolated_data):
    from football_profiler.runtime import torch_device
    for group in range(3):
        identifier = f'artificial-interval-{group}'
        players, rows = [], []
        for index in range(6):
            pid = f'p{index}'
            players.append({'player_id': pid, 'name': 'Artificial player', 'eligible_frames': 120,
                            'direction_known': True, 'global_id': f'{identifier}:{pid}',
                            'identity_verified': True})
            rows.extend((frame, frame * 10, pid, 1, 20 + (index % 2) * 60, 30, 1, 1)
                        for frame in range(120))
        manifest = {'id': identifier, 'match_id': identifier, 'source': 'Artificial fixture',
                    'sampling_hz': .1, 'duration_seconds': 1200, 'players': players}
        S.write_json(S.dataset_dir(identifier, True) / 'manifest.json', manifest)
        S.save_tracks(identifier, pd.DataFrame(rows, columns=[
            'frame', 'time_s', 'player_id', 'period', 'x', 'y', 'detected', 'calibration_valid']))
        for index, player in enumerate(players):
            case = C.create(identifier, player['player_id'], 1, 0, 1200, 'centre_forward')
            for reviewer in ('Artificial Alice', 'Artificial Bob'):
                review(case, reviewer, (index % 2) * 100)
    profiles = C.training_profiles()
    assert len(profiles) == 18 and len(profiles[0]['y']) == 42
    assert np.isnan(profiles[0]['y'][T.LABELS.index('complete_forward')])
    report = L.train_models(epochs=1, schema_version='2.0')
    assert report['device'] == str(torch_device())
    assert report['label_names'] == list(T.LABELS)
    assert sum(report['active_labels']) == 1
    assert (isolated_data / 'models/latest-v2.json').exists()
    assert not (isolated_data / 'models/latest.json').exists()
    for model in ('logistic', 'cnn', 'cnn_with_gaps'):
        report['selected_model'] = model
        S.write_json(isolated_data / 'models' / report['run_id'] / 'report.json', report)
        prediction = L.predict(profiles[0])
        assert prediction['status'] == 'model_suggestion'
        assert {x['label'] for x in prediction['labels']} == set(T.compatible('centre_forward'))
        assert all(x['percentage'] is None for x in prediction['labels'] if x['label'] != 'target_man')
