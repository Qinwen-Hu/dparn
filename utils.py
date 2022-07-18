import torch


def is_pytorch_1_8():
    """ Returns True if PyTorch is >= 1.8.
    """
    return int(str(torch.__version__).split(".")[1]) >= 8
