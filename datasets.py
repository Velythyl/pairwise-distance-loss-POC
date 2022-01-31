import os

import numpy as np
from numpy.random import SeedSequence, default_rng
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def get_vectortarget_dataset(dataset_partial, seed, gaussian_scale, vector_width):
    rng = default_rng(seed)

    def target_transform(target):
        deterministic_noise_vector = rng.normal(loc=0.0, scale=gaussian_scale, size=vector_width)
        target_vector = np.array([target] * vector_width)

        noised_target_vectpr = deterministic_noise_vector + target_vector
        return noised_target_vectpr

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), target_transform=)
train, val = random_split(dataset, [55000, 5000])

x=0