"""Shared, explicit device selection for video and neural-network processing."""
from __future__ import annotations

import os


def torch_device():
    """Require the configured GPU, unless the user explicitly selects CPU."""
    import torch

    requested = os.environ.get("PITCHPROFILE_DEVICE", "cuda:0").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda:") or not requested[5:].isdigit():
        raise ValueError("Set PITCHPROFILE_DEVICE to cuda:0 (or another CUDA index), or explicitly to cpu.")
    device = torch.device(requested)
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA processing was requested but PyTorch cannot access an NVIDIA GPU. "
            "Install the CUDA PyTorch build with setup.ps1 and check the NVIDIA driver. "
            "CPU processing requires an explicit PITCHPROFILE_DEVICE=cpu override."
        )
    if device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available. "
            "Set PITCHPROFILE_DEVICE to an available CUDA index."
        )
    return device


def runtime_info():
    """Return diagnostics without preventing profiles from opening if CUDA is unavailable."""
    info = {
        "requested_device": os.environ.get("PITCHPROFILE_DEVICE", "cuda:0"),
        "device": None, "cuda_available": False, "gpu_name": None,
        "torch_version": None, "cuda_version": None, "error": None,
    }
    try:
        import torch

        info.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    cuda_available=torch.cuda.is_available())
        device = torch_device()
        info["device"] = str(device)
        if device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(device)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        info["error"] = str(exc)
    return info
