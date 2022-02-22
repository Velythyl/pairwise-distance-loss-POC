import functools
import os
from typing import Any

import numpy as np
import torch
from numpy.random import SeedSequence, default_rng
from torch.utils.data import random_split
from torchvision.datasets import MNIST, VisionDataset
from torchvision.transforms import transforms
from tqdm import tqdm


class VectorTargetDataset(VisionDataset):
    _repr_indent = 4

    def __init__(
            self,
            vision_dataset, dataset_seed, vector_width, gaussian_instead_of_uniform, scale=0.1
    ) -> None:
        self.vision_dataset = vision_dataset
        self.seed = dataset_seed
        self.vector_width = vector_width
        self.gaussian_instead_of_uniform = gaussian_instead_of_uniform
        self.scale = scale

        self.classes = self.vision_dataset.targets
        self.targets = self.noise_targets()
        self.data = self.vision_dataset.data.float()

    def noise_targets(self):
        targets = np.array(self.classes)
        target_matrix = np.array([targets] * self.vector_width).T

        noise_shape = (len(targets), self.vector_width)
        rng = default_rng(self.seed)
        if self.gaussian_instead_of_uniform:
            deterministic_noise_vectors = rng.normal(loc=0.0, scale=self.scale, size=noise_shape)
        else:
            deterministic_noise_vectors = rng.uniform(low=-self.scale, high=self.scale, size=noise_shape)

        noised_targets = target_matrix + deterministic_noise_vectors

        noised_targets = torch.from_numpy(noised_targets).float()

        highest_class = torch.max(self.classes)
        lowest_class = 0
        assert torch.min(self.classes) == 0

        max_noised = highest_class + self.scale
        min_noised = lowest_class - self.scale
        noised_targets = (torch.clip(noised_targets, min_noised, max_noised) - min_noised) / (
                max_noised - min_noised)

        return noised_targets

    def __getitem__(self, index: int) -> Any:
        item, target = self.vision_dataset.__getitem__(index)

        return item.float(), self.targets[index].float(), self.classes[index]

    def __len__(self) -> int:
        return len(self.vision_dataset)

    def __repr__(self) -> str:
        return self.vision_dataset.__repr__()

    def extra_repr(self) -> str:
        return self.vision_dataset.extra_repr()


def assert_determinism():
    dataset_1 = VectorTargetDataset(
        MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=False,
        scale=0.8
    )

    dataset_2 = VectorTargetDataset(
        MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=False,
        scale=0.8
    )

    for i in tqdm(range(len(dataset_1))):
        one = dataset_1[i]
        two = dataset_2[i]
        assert torch.equal(one[0], two[0])
        assert torch.equal(one[1], two[1])


if __name__ == "__main__":
    assert_determinism()
