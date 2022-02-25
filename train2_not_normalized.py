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


def pairwise_distance_loss(embeddings, targets, no_loss=False):
   # embeddings = F.normalize(embeddings)

    target_gram = targets @ targets.T   # Gram matrix
    embed_gram = embeddings @ embeddings.T

    return F.mse_loss(embed_gram, target_gram)

    loss = torch.cdist(target_gram, embed_gram)
    return loss.sum()


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
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, z = batch
        x = x.view(x.size(0), -1)
        embeds = self.encoder(x)

        loss = self.loss_function(embeds,y)

        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    #def validation_step(self, *args, **kwargs):
    #    pass
    #def on_validation_end(self) -> None:
    #    eval_model.main(self, eval_ds, GPU)


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
    scale=0.1
)  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

eval_ds = VectorTargetDataset(
    MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=True,
    scale=0.1
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

