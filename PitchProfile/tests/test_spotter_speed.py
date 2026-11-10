"""Spotter speed: reading frames ahead changes nothing but the time (action_spotting.read_ahead)."""
import threading

import cv2
import numpy as np
import pytest
import torch

from football_profiler import action_spotting as AS


class FakeNet:
    """Stands in for T-DEED: one number per frame (its mean pixel) spread over the output heads."""

    def __init__(self):
        self.clips = 0

    def __call__(self, x, inference=True):
        self.clips += x.shape[0]
        m = x.mean(dim=(2, 3, 4)) / 255                                  # (clips, frames)
        im = m[..., None] * torch.linspace(-3, 3, len(AS.CLASSES) + 1)
        return {'im_feat': im, 'displ_feat': torch.zeros(m.shape), 'team_feat': torch.stack([m, -m], -1)}, None


class FakeKit:
    @staticmethod
    def process_predictionTeam(pred, team):
        side = torch.softmax(team, -1)                                    # left / right
        return torch.cat([pred[..., :1], (pred[..., 1:, None] * side[..., None, :]).flatten(-2)], -1)


@pytest.fixture
def fake_model(monkeypatch):
    net = FakeNet()
    monkeypatch.setattr(AS, 'model', lambda: (net, FakeKit(), torch.device('cpu')))
    monkeypatch.setattr(AS, 'CLIP_LEN', 8)                                # small clips, same overlap: step = len / 4
    monkeypatch.setattr(AS, 'CLIP_STEP', 2)
    monkeypatch.setattr(AS, 'BATCH', 2)
    return net


@pytest.fixture
def video(tmp_path):
    """120 frames at 25/s (60 seen by the spotter), each a different shade."""
    path = tmp_path / 'clip.avi'
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    for i in range(120):
        frame = np.full((48, 64, 3), (i * 2) % 256, np.uint8)
        frame[:, : 1 + i % 60] = 255 - frame[:, : 1 + i % 60]
        out.write(frame)
    out.release()
    return path


def test_read_ahead_keeps_every_item_in_order():
    assert list(AS.read_ahead(iter(range(1000)), size=7)) == list(range(1000))
    assert list(AS.read_ahead(iter(range(10)), size=0)) == list(range(10))           # 0: read in turn


def test_read_ahead_raises_a_reading_error_after_the_items_before_it():
    def broken():
        yield from range(5)
        raise ValueError('bad frame')
    got = []
    with pytest.raises(ValueError, match='bad frame'):
        for item in AS.read_ahead(broken(), size=3):
            got.append(item)
    assert got == [0, 1, 2, 3, 4]


def test_stopping_early_stops_the_reader_and_closes_the_source():
    closed = threading.Event()

    def endless():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            closed.set()                                                  # where the video file is released
    reader = AS.read_ahead(endless(), size=4)
    assert [next(reader) for _ in range(3)] == [0, 1, 2]
    reader.close()
    assert closed.wait(5)
    assert not any(t.name == 'spotter-read-ahead' and t.is_alive() for t in threading.enumerate())


def test_scores_from_a_video_are_identical_with_and_without_read_ahead(fake_model, video, monkeypatch):
    monkeypatch.setattr(AS, 'PROGRESS_EVERY', 20)
    runs = {}
    for ahead in (0, AS.READ_AHEAD):
        monkeypatch.setattr(AS, 'READ_AHEAD', ahead)
        calls = []
        progress = lambda f, m: calls.append((f, m, threading.current_thread() is threading.main_thread()))
        runs[ahead] = AS.frame_scores(video, progress), calls
    (before, calls_before), (after, calls_after) = runs.values()
    assert before.shape == (60, 25)
    assert np.array_equal(before, after)
    assert calls_before == calls_after == [(20 / 60, 'Action spotting 20/60 frames', True),
                                           (40 / 60, 'Action spotting 40/60 frames', True),
                                           (.95, 'Action spotting 60/60 frames', True)]


def test_scores_from_packed_frames_are_identical_with_and_without_read_ahead(fake_model, monkeypatch):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (6, 8, 3), dtype=np.uint8) for _ in range(37)]
    runs = []
    for ahead in (0, 5):
        monkeypatch.setattr(AS, 'READ_AHEAD', ahead)
        runs.append(AS.frame_scores(None, frames=enumerate(frames)))
    assert runs[0].shape == (37, 25) and np.array_equal(*runs)
    assert (np.abs(runs[0]).sum(1) > 0).all()                             # every frame is scored
