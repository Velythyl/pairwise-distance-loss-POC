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
            vision_dataset, dataset_seed, vector_width, gaussian_instead_of_uniform, scale=0.8
    ) -> None:
        self.vision_dataset = vision_dataset
        self.seed = dataset_seed
        self.vector_width = vector_width
        self.gaussian_instead_of_uniform = gaussian_instead_of_uniform
        self.scale = scale

        highest_class = torch.max(vision_dataset.targets)
        lowest_class = 0
        assert torch.min(vision_dataset.targets) == 0

        max_noised = highest_class + scale
        min_noised = lowest_class - scale

        self.scale_target = lambda target: (torch.clip(target, min_noised, max_noised) - min_noised) / (
                    max_noised - min_noised)

    def noise_target(self, target, index):
        rng = default_rng(self.seed + index)

        if self.gaussian_instead_of_uniform:
            deterministic_noise_vector = rng.normal(loc=0.0, scale=self.scale, size=self.vector_width)
        else:
            deterministic_noise_vector = rng.uniform(low=-self.scale, high=self.scale, size=self.vector_width)

        target_vector = np.array([target] * self.vector_width)

        noised_target_vector = deterministic_noise_vector + target_vector

        noised_target_vector = torch.from_numpy(noised_target_vector).float()

        return self.scale_target(noised_target_vector)

    def __getitem__(self, index: int) -> Any:
        item, target = self.vision_dataset.__getitem__(index)

        target = self.noise_target(target, index)
        return item, target

    def __len__(self) -> int:
        return len(self.vision_dataset)

    def __repr__(self) -> str:
        return self.vision_dataset.__repr__()

    def extra_repr(self) -> str:
        return self.vision_dataset.extra_repr()


def get_vectortarget_dataset(dataset_partial, dataset_seed, vector_width, gaussian_instead_of_uniform, scale=0.4):
    rng = default_rng(dataset_seed)

    temp_dataset = dataset_partial()

    highest_class = torch.max(temp_dataset.targets)
    lowest_class = 0
    assert torch.min(temp_dataset.targets) == 0

    max_noised = highest_class + scale
    min_noised = lowest_class - scale

    def target_transform(target):
        if gaussian_instead_of_uniform:
            deterministic_noise_vector = rng.normal(loc=0.0, scale=scale, size=vector_width)
        else:
            deterministic_noise_vector = rng.uniform(low=-scale, high=scale, size=vector_width)

        target_vector = np.array([target] * vector_width)

        noised_target_vector = deterministic_noise_vector + target_vector

        noised_target_vector = torch.from_numpy(noised_target_vector).float()

        clipped = torch.clip(noised_target_vector, min=min_noised, max=max_noised)  # forces bounds
        rescaled = (clipped - min_noised) / (max_noised - min_noised)  # goes from bounds to 0-1
        return rescaled

    final_dataset = dataset_partial(target_transform=target_transform)

    return final_dataset


def assert_determinism():
    dataset_1 = get_vectortarget_dataset(
        functools.partial(MNIST, os.getcwd(), download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=False,
        scale=0.5
    )

    dataset_2 = get_vectortarget_dataset(
        functools.partial(MNIST, os.getcwd(), download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=False,
        scale=0.5
    )

    for i in tqdm(range(len(dataset_1))):
        one = dataset_1[i]
        two = dataset_2[i]
        assert torch.equal(one[0], two[0])
        assert torch.equal(one[1], two[1])

def assert_determinism2():
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
    assert_determinism2()

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    dataset = get_vectortarget_dataset(
        functools.partial(MNIST, os.getcwd(), download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=False,
        scale=0.5
    )
    train2, val2 = random_split(dataset, [55000, 5000])

    target1 = train[train.indices[0]]
    target2 = train2[train2.indices[0]]
    x = 0
