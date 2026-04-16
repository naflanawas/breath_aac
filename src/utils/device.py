"""
Shared device selection utility for MURMUR training and inference scripts.
"""
import torch

def pick_device() -> torch.device:
    """Return the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")