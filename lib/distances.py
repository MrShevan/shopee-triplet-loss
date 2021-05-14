import torch


def knn(vectors, n_neighbors, device):
    if device.type == 'cuda':
        vectors = vectors.to(device)

    # Compute distances
    distance_matrix = torch.cdist(vectors, vectors, p=2.0)

    # Compute top neighbors
    distances, indices = torch.topk(distance_matrix, k=n_neighbors, dim=1,
                                    largest=False, sorted=True)

    del distance_matrix

    vectors = vectors.detach().cpu()
    indices = indices.detach().cpu().numpy()
    distances = distances.detach().cpu().numpy()

    return distances, indices
