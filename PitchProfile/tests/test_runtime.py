import pytest

from football_profiler.runtime import runtime_info, torch_device


def test_default_device_requires_cuda(monkeypatch):
    import torch

    monkeypatch.delenv("PITCHPROFILE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA processing was requested"):
        torch_device()
    info = runtime_info()
    assert info["requested_device"] == "cuda:0"
    assert info["device"] is None
    assert info["cuda_available"] is False
    assert "PITCHPROFILE_DEVICE=cpu" in info["error"]


def test_default_device_selects_first_gpu(monkeypatch):
    import torch

    monkeypatch.delenv("PITCHPROFILE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Artificial CUDA test device")
    assert str(torch_device()) == "cuda:0"
    info = runtime_info()
    assert info["device"] == "cuda:0"
    assert info["gpu_name"] == "Artificial CUDA test device"
    assert info["error"] is None


def test_cpu_is_available_only_by_explicit_configuration(monkeypatch):
    import torch

    monkeypatch.setenv("PITCHPROFILE_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(torch_device()) == "cpu"
    assert runtime_info()["error"] is None


def test_selected_cuda_index_must_exist(monkeypatch):
    import torch

    monkeypatch.setenv("PITCHPROFILE_DEVICE", "cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 CUDA device"):
        torch_device()


@pytest.mark.parametrize("value", ["", "auto", "mps", "cuda:-1", "gpu", "0"])
def test_device_selection_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("PITCHPROFILE_DEVICE", value)
    with pytest.raises(ValueError, match="Set PITCHPROFILE_DEVICE"):
        torch_device()
