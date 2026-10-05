import json
import pytest
from football_profiler import soccertrack as ST, storage as S


@pytest.fixture
def released_gsr(tmp_path):
    data = {'info': {'frame_rate': 25, 'clip_start': 0, 'clip_stop': 30},
            'images': [{'image_id': str(3000001 + i), 'is_labeled': True} for i in range(50)],
            'annotations': [{'image_id': str(3000001 + i), 'track_id': 7,
                             'attributes': {'role': 'player', 'team': 'left', 'jersey': 9, 'player_id': 52},
                             'bbox_pitch': {'x_bottom_middle': 10 if i else 60, 'y_bottom_middle': 0}}
                            for i in range(50)]}
    path = tmp_path / 'released.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_released_object_frame_clock_orientation_and_identity(released_gsr):
    rows, players, info = ST.parse_half(released_gsr, '117092', 2, left_attacks='left')
    assert info['duration_seconds'] == 2  # Ignore misleading old clip_stop.
    assert len(rows) == 10 and info['source_frames'] == 50
    assert rows.time_s.tolist() == [i / 5 for i in range(10)]
    assert rows.iloc[0].x < 0  # Preserve out-of-pitch evidence for quality accounting.
    assert rows.iloc[1].x == 42.5
    assert players[0]['player_id'] == 'h2-t7'
    assert not players[0]['identity_verified'] and players[0]['global_id'] is None
    assert players[0]['provider_player_id'] == '52'


def test_import_half_and_media_reference(released_gsr):
    m = ST.import_half(released_gsr, '117092', 1)
    assert m['benchmark_split'] == 'train'
    assert m['match_id'] == 'soccertrack:117092'
    assert m['source_kind'] == 'soccertrack_reference'
    profiles = S.read_json(S.dataset_dir(m['id']) / 'profiles.json')
    assert profiles[0]['position_coverage'] == .9
    assert not profiles[0]['direction_known']
    assert profiles[0]['events'] is None
    with pytest.raises(ValueError, match='already present'):
        ST.import_half(released_gsr, '117092', 1)


def test_confirming_reference_direction_rotates_once_and_invalidates_cases(client, released_gsr):
    from football_profiler import cases as C
    m = ST.import_half(released_gsr, '117092', 1)
    identifier, player = m['id'], m['players'][0]
    case = C.create(identifier, player['player_id'], 1, 0, 2, 'centre_forward')
    endpoint = f"/api/datasets/{identifier}/players/{player['player_id']}/identity"
    before = S.load_tracks(identifier)
    payload = {'name': player['name'], 'direction': 'left'}
    assert client.post(endpoint, json=payload).status_code == 200
    after = S.load_tracks(identifier)
    assert after.x.tolist() == (105 - before.x).tolist()
    assert C.get(case['id'])['stale']
    assert client.post(endpoint, json=payload).status_code == 200
    assert S.load_tracks(identifier).x.tolist() == after.x.tolist()
    assert S.read_json(S.dataset_dir(identifier) / 'profiles.json')[0]['direction_known']


def test_delayed_video_pts_preserves_source_clock_and_bounds(client, released_gsr, tmp_path, monkeypatch):
    import numpy as np
    from football_profiler import cases as C, vision as V
    video = tmp_path / 'delayed.mp4'
    video.write_bytes(b'artificial fixture')
    monkeypatch.setattr(V, 'video_info', lambda path: {'fps': 25, 'frames': 25, 'width': 100, 'height': 60})
    monkeypatch.setattr(ST, 'video_start_time', lambda path: 1.)
    manifest = ST.import_half(released_gsr, '117092', 2, video=video)
    assert manifest['video_start_s'] == 1
    assert manifest['duration_seconds'] == 2
    assert manifest['excluded_reference_rows_outside_video'] == 5
    tracks = S.load_tracks(manifest['id'])
    assert len(tracks) == 5 and tracks.time_s.min() == 1
    assert manifest['players'][0]['eligible_frames'] == 5
    with pytest.raises(ValueError, match='video starts'):
        C.create(manifest['id'], 'h2-t7', 2, 0, 2, 'centre_forward')
    assert C.create(manifest['id'], 'h2-t7', 2, 1, 2, 'centre_forward')['profile']['position_coverage'] == 1
    seen = []
    monkeypatch.setattr(V, 'read_frame', lambda path, seconds: seen.append(seconds) or np.zeros((60,100,3), dtype=np.uint8))
    assert client.get(f"/api/datasets/{manifest['id']}/frame?seconds=1.5").status_code == 200
    assert seen == [.5]
    assert client.get(f"/api/datasets/{manifest['id']}/frame?seconds=0").status_code == 400


def test_video_mismatch_is_not_silently_trimmed(released_gsr, tmp_path, monkeypatch):
    from football_profiler import vision as V
    video = tmp_path / 'wrong.mp4'
    video.write_bytes(b'artificial fixture')
    monkeypatch.setattr(V, 'video_info', lambda path: {'fps': 25, 'frames': 20, 'width': 100, 'height': 60})
    monkeypatch.setattr(ST, 'video_start_time', lambda path: 1.)
    with pytest.raises(ValueError, match='does not align'):
        ST.import_half(released_gsr, '117092', 2, video=video)
    assert not (S.DATA / 'datasets/soccertrack-117092-h2').exists()


def test_bas_clock_requires_explicit_half_and_actor_match(released_gsr, tmp_path):
    _, players, _ = ST.parse_half(released_gsr, '117092', 2)
    event = {'gameTime': '2 - 45:01', 'position': 2701000, 'label': 'PASS', 'player_id': 52, 'team': 'left'}
    path = tmp_path / 'bas.json'
    path.write_text(json.dumps({'UrlLocal': '117092', 'actions': [event,
                    {**event, 'gameTime': '90:01'}, {**event, 'team': 'right'},
                    {**event, 'gameTime': '1 - 45:01'}]}), encoding='utf-8')
    result = ST.parse_events(path, '117092', 2, players, 2, 2700000)
    assert len(result['events']) == 1
    assert result['events'][0]['time_s'] == 1
    assert result['events'][0]['kind'] == 'Pass'
    assert result['excluded'] == {'other_period': 1, 'ambiguous_period': 1, 'outside_half': 0, 'unmatched_actor': 1}
    with pytest.raises(ValueError, match='offset was verified'):
        ST.import_half(released_gsr, '117092', 2, bas=path, event_offset_ms=2700000)
    path.write_text(json.dumps({'match_id': '117093', 'actions': [event]}), encoding='utf-8')
    with pytest.raises(ValueError, match='identity'):
        ST.parse_events(path, '117092', 2, players, 2, 2700000)


def test_unknown_images_duplicate_rows_and_wrong_format_fail(released_gsr):
    raw = json.loads(released_gsr.read_text(encoding='utf-8'))
    raw['annotations'].append(raw['annotations'][0])
    released_gsr.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate player'):
        ST.parse_half(released_gsr, '117092', 1)
    raw['annotations'][-1] = {**raw['annotations'][0], 'image_id': '9999999'}
    released_gsr.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='unknown GSR image'):
        ST.parse_half(released_gsr, '117092', 1)
    released_gsr.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='released GSR object'):
        ST.parse_half(released_gsr, '117092', 1)
