from typing import Any

from torch import nn
from torchvision.datasets import VisionDataset


def augment_label(label: int, new_dim: int, perturb_fn: nn.Identity()):
    vector_label =