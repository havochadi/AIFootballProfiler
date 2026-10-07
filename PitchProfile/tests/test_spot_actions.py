"""The analysis job runs the video action spotter and stores its spots on the half's own time axis."""

from football_profiler import action_spotting as AS
from football_profiler import match_pipeline as MP
from football_profiler import storage as S

DETECTION = {'source_fps': 25.0, 'start_frame': 75, 'stop_frame': 1250, 'time_origin_s': 3.0}
SPOTS = [{'label': 'PASS', 'side': 'left', 'frame': 12, 'time_s': 1.0, 'score': .9},      # before the analysed interval
         {'label': 'SHOT', 'side': 'right', 'frame': 50, 'time_s': 4.0, 'score': .8},
         {'label': 'PASS', 'side': 'left', 'frame': 1250, 'time_s': 100.0, 'score': .7}]  # after it


def test_shift_spots_keeps_the_analysed_interval_on_the_half_time_axis():
    out = MP.shift_spots(SPOTS, 25.0, 75, 1250, 3.0, AS.FRAME_STRIDE)
    assert [s['label'] for s in out] == ['SHOT']
    assert out[0]['time_s'] == 1.0                      # 4.0 s of video minus the 3.0 s origin
    assert out[0]['frame'] == 50 - 38                   # frames shift by origin * 12.5 frames/s
    assert SPOTS[1]['time_s'] == 4.0                    # the input is not modified


def test_spot_actions_writes_shifted_spots(monkeypatch):
    monkeypatch.setattr(AS, 'available', lambda: True)
    seen = {}

    def fake(video, progress=None, max_frames=None):
        seen['max_frames'] = max_frames
        return {'model': 'fake spotter', 'spots': list(SPOTS)}
    monkeypatch.setattr(AS, 'spot_video', fake)
    S.dataset_dir('half-1', create=True)
    status = MP.spot_actions('half-1', 'video.mp4', DETECTION)
    saved = S.read_json(S.dataset_dir('half-1') / 'action_spots.json')
    assert status['status'] == 'run' and status['spots'] == 1
    assert [s['label'] for s in saved['spots']] == ['SHOT'] and saved['time_origin_s'] == 3.0
    assert seen['max_frames'] == 1250 // AS.FRAME_STRIDE + 1      # nothing beyond the analysed interval is read


def test_spot_actions_keeps_existing_spots_unless_forced(monkeypatch):
    d = S.dataset_dir('half-2', create=True)
    S.write_json(d / 'action_spots.json', {'spots': [{'label': 'PASS'}]})
    monkeypatch.setattr(AS, 'available', lambda: True)
    monkeypatch.setattr(AS, 'spot_video', lambda *a, **k: {'spots': []})
    assert MP.spot_actions('half-2', 'video.mp4', DETECTION)['status'] == 'kept existing spots'
    assert S.read_json(d / 'action_spots.json')['spots'] == [{'label': 'PASS'}]
    assert MP.spot_actions('half-2', 'video.mp4', DETECTION, force=True)['status'] == 'run'


def test_spot_actions_reports_a_missing_checkpoint_without_failing(monkeypatch):
    monkeypatch.setattr(AS, 'available', lambda: False)
    S.dataset_dir('half-3', create=True)
    status = MP.spot_actions('half-3', 'video.mp4', DETECTION)
    assert status['status'].startswith('not run')
    assert not (S.dataset_dir('half-3') / 'action_spots.json').exists()


def test_spot_actions_survives_a_spotter_failure(monkeypatch):
    monkeypatch.setattr(AS, 'available', lambda: True)

    def broken(*a, **k):
        raise RuntimeError('out of memory')
    monkeypatch.setattr(AS, 'spot_video', broken)
    S.dataset_dir('half-4', create=True)
    status = MP.spot_actions('half-4', 'video.mp4', DETECTION)
    assert status['status'].startswith('failed: RuntimeError')
    assert not (S.dataset_dir('half-4') / 'action_spots.json').exists()
