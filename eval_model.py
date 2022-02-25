import numpy as np
import sklearn.metrics
import torch
from torch.nn.functional import mse_loss


def tensor2numpy(tensor):
    try:
        tensor = tensor.cpu().numpy()
    except:
        pass
    return tensor

def compute_centroids(embeddings, classes):
    embeddings = tensor2numpy(embeddings)
    classes = tensor2numpy(classes)

    embeddings = embeddings
    centroids = np.zeros((int(np.max(classes)), embeddings.shape[1]))
    centroids_std = np.copy(centroids)

    for i, clas in enumerate(range(classes.max())):
        class_embeds = embeddings[classes == clas]

        centroid = np.mean(class_embeds, axis=0)
        centroids[i] = centroid

        std = np.std(class_embeds, axis=0)
        centroids_std[i] = std

    return centroids, centroids_std

def main_(embeddings, targets, classes):
    pred_centroids, pred_centroid_std = compute_centroids(embeddings, classes)
    true_centroids, true_centroid_std = compute_centroids(targets, classes)

    #print("Pred centroids")
    #print(pred_centroids)
    #print("True centroids")
    #print(true_centroids)

    print("Silhouette score")
    print(sklearn.metrics.silhouette_score(embeddings, classes))

    pred_centroids = torch.from_numpy(pred_centroids)
    true_centroids = torch.from_numpy(true_centroids)

    #print("Pred pairwise distance")
    pred_centroids = torch.nn.functional.normalize(pred_centroids)
    pred_distances = torch.cdist(pred_centroids, pred_centroids)
    #print(pred_distances)

    #print("True pairwise distance")
    true_distances = torch.cdist(true_centroids, true_centroids)
    #print(true_distances)

    loss = mse_loss(pred_distances, true_distances)
    print("Pairwise distance MSE")
    print(loss.item())


def main(model, eval_dataset, GPU):
    with torch.no_grad():
        data = torch.clone(eval_dataset.data).cuda(GPU)
        targets = torch.clone(eval_dataset.targets).cpu().numpy()
        classes = torch.clone(eval_dataset.classes).cpu().numpy()

        embeddings = model(data).cpu().numpy()

        return main_(embeddings, targets, classes)
