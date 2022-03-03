import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from datasets import VectorTargetDataset
import pytorch_lightning as pl
import torch.nn.functional as F

from utils import tensor2numpy

colors = ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'pink', 'orange', 'brown', "magenta"]



def plot_2d(targets, classes):
    targets = tensor2numpy(targets)
    classes = tensor2numpy(classes)

    x = targets[:,0]
    y = targets[:,1]

    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=classes, cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar()
    loc = np.arange(0,max(classes),max(classes)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(list(map(str, np.unique(classes))))

    plt.show()

def plot_3d(targets, classes, angled):
    targets = tensor2numpy(targets)
    classes = tensor2numpy(classes)

    x = targets[:,0]
    y = targets[:,1]
    z = targets[:,2]

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, c=classes, cmap=matplotlib.colors.ListedColormap(colors))
    if angled:
        ax.view_init(30, 60)


    #plt.scatter(x, y, c=classes, cmap=))
    #cb = plt.colorbar()
    #loc = np.arange(0,max(classes),max(classes)/float(len(colors)))
    #cb.set_ticks(loc)
    #cb.set_ticklabels(list(map(str, np.unique(classes))))

    plt.show()

def main(embeddings, targets, classes):
    plot_2d(targets, classes)
    plot_3d(embeddings, classes, angled=True)
    plot_3d(embeddings, classes, angled=False)

if __name__ == "__main__":
    def pairwise_distance_loss(embeddings, targets, no_loss=False):
        # embeddings = F.normalize(embeddings)

        target_gram = torch.cdist(targets, targets)
        embed_gram = torch.cdist(embeddings, embeddings)

        return F.mse_loss(target_gram, embed_gram)


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

            loss = self.loss_function(embeds, y)

            self.log("train_loss", loss.item())
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer
        # def validation_step(self, *args, **kwargs):
        #    pass
        # def on_validation_end(self) -> None:
        #    eval_model.main(self, eval_ds, GPU)


    train_ds = VectorTargetDataset(
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
        dataset_seed=0,
        vector_width=2,
        gaussian_instead_of_uniform=True,
        scale=0.1
    )  # MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    plot_2d(train_ds.targets, train_ds.classes)

    with torch.no_grad():
        mnist_model = Encoder(pairwise_distance_loss)
        plot_3d(mnist_model(train_ds.data), train_ds.classes)
