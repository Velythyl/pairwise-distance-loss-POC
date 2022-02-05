import functools
import os
import torch
from pytorch_lightning import Trainer
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from datasets import get_vectortarget_dataset


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax()
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        loss = self.pairwise_distance_loss(z, y)

        self.log("train_loss", loss)
        return loss

    def pairwise_distance_loss(self, embeddings, targets):
        target_distances = torch.cdist(targets, targets)
        embedding_distances = torch.cdist(embeddings, embeddings)

        loss = F.mse_loss(embedding_distances, target_distances)
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
train_ds = get_vectortarget_dataset(
    functools.partial(MNIST, os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    dataset_seed=0,
    vector_width=2,
    gaussian_instead_of_uniform=False,
    scale=0.5
)#MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

def compute_centroids(targets):
compute_centroids()