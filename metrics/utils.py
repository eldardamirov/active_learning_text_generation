from torch.cuda import is_available


def get_device(device: str = None):
    if device is None:
        device = "cuda" if is_available() else "cpu"
    return device
