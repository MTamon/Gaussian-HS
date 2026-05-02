import torch


def load_trusted_checkpoint(*args, **kwargs):
    """Load project checkpoints saved with torch.save.

    PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects
    checkpoints containing optimizer states and other Python objects.
    """
    kwargs.setdefault("weights_only", False)
    return torch.load(*args, **kwargs)
