"""Windows/NVIDIA native adapters; imports never query or modify a GPU."""
from .backend import WindowsNvidiaBackend

__all__ = ["WindowsNvidiaBackend"]
