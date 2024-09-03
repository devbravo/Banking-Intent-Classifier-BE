import torch


def get_device() -> str:
    """
    Determines the appropriate device to use for PyTorch operations.

    Returns:
        str: 'cuda' if a GPU is available, 'mps' if an Apple M1/M2 GPU is 
              available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'