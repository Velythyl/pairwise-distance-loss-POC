import functools
import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

import eval_model
from datasets import VectorTargetDataset


def pairwise_cosine_embedding(mat, margin=0.5):
    gram = mat @ mat.T

    zeros = torch.zeros(gram.shape).to(GPU)

    cosine_loss = torch.maximum(zeros, gram-margin)

    #cosine_loss = 1 - gram
    return 1-cosine_loss

def pairwise_distance_loss(embeddings, targets, no_loss=False):
    #embeddings = F.normalize(embeddings)

    target_gram = targets @ targets.T
    embed_gram = embeddings @ embeddings.T

    return F.mse_loss(embed_gram, target_gram)


class NormalizerModule(nn.Module):
    def forward(self, input):
        return F.normalize(input, dim=1)


class Encoder(pl.LightningModule):
    def __init__(self, loss_function):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU(),
        )
        self.loss_function = loss_function

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = x.view(x.size(0), -1)
        embedding = self.encoder(x)
        #embedding = F.normalize(embedding)
        return embedding

class Relationer(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, input):

class RelationNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(pairwise_distance_loss)
        self.relationer = Relationer()

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
GPU = 0
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# Init our model
mnist_model = Encoder(pairwise_distance_loss)

# Init DataLoader from MNIST Dataset
train_ds = VectorTargetDataset(
    MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=True,
    scale=0.1,
    recenter=True
)  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)


eval_ds = VectorTargetDataset(
    MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=True,
    scale=0.1,
    recenter=True
)  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=10,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
eval_model.main(mnist_model.cuda(GPU), eval_ds, GPU)

