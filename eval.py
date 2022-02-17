import torch


def compute_centroids(embeddings, classes):
    centroids = torch.zeros((int(torch.max(classes)), embeddings.shape[1]))
    centroids_std = torch.clone(centroids)

    for i, clas in enumerate(range(classes.max())):
        class_embeds = embeddings[classes == clas]

        centroid = torch.mean(class_embeds, dim=0)
        centroids[i] = centroid

        std = torch.std(class_embeds, dim=0)
        centroids_std[i] = std

    return centroids, centroids_std

def eval(model, dataset):
