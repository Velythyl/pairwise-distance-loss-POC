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

from datasets import VectorTargetDataset


def pairwise_distance_loss(embeddings, targets, no_loss=False):
    target_gram = targets @ targets.T
    embed_gram = embeddings @ embeddings.T

    loss = torch.cdist(target_gram, embed_gram)
    return loss.mean()


class NormalizerModule(nn.Module):
    def forward(self, input):
        return F.normalize(input, dim=1)


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU()
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = x.view(x.size(0), -1)
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        loss = pairwise_distance_loss(z, y)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# Init our model
mnist_model = Encoder()

# Init DataLoader from MNIST Dataset
train_ds = VectorTargetDataset(
    MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=True,
    scale=0.5
)  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

eval_ds = VectorTargetDataset(
    MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=True,
    scale=0.5
)  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())


def compute_centroids(embeddings, classes):
    centroids = torch.zeros((int(torch.max(classes)), embeddings.shape[1]))

    for i, clas in enumerate(range(classes.max())):
        class_embeds = embeddings[classes == clas]
        centroid = torch.mean(class_embeds, dim=0)
        centroids[i] = centroid

    return centroids


def compute_centroids_dataset(dataset):
    targets = dataset.targets
    classes = dataset.classes
    return compute_centroids(targets, classes)


def compute_centroids_model(model, dataset):
    with torch.no_grad():
        embeddings = model(dataset.data)
        classes = dataset.classes
        return compute_centroids(embeddings, classes)


# true_centroids = compute_centroids_dataset(train_ds)
# true_centroids = eval_ds.classes
pred_centroids = compute_centroids_model(mnist_model, eval_ds)
final_distance = (torch.cdist(pred_centroids, pred_centroids) * 100).int()
print(final_distance)
# print(pairwise_distance_loss(true_centroids, pred_centroids, no_loss=True))

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=10,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

pred_centroids = compute_centroids_model(mnist_model, eval_ds)
final_distance = (torch.cdist(pred_centroids, pred_centroids) * 100).int()
print(final_distance)

pred_centroids = compute_centroids_dataset(eval_ds)
final_distance = (torch.cdist(pred_centroids, pred_centroids) * 100).int()
print(final_distance)
