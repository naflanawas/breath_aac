import torch

def compute_prototypes(embeddings, labels, n_classes):
    """
    embeddings: [N, D]
    labels: [N]
    """
    prototypes = []
    for c in range(n_classes):
        proto = embeddings[labels == c].mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)  # [C, D]


def prototypical_predict(query_embeddings, prototypes):
    """
    query_embeddings: [Q, D]
    prototypes: [C, D]
    """
    dists = torch.cdist(query_embeddings, prototypes)
    return dists.argmin(dim=1)
