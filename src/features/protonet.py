import torch
import torch.nn.functional as F

def compute_prototypes(embeddings, labels, n_classes):
    """
    embeddings: [N, D]
    labels:     [N]
    Returns:    [C, D] - mean embedding per class (L2-normalised for cosine)
    """
    prototypes = []
    for c in range(n_classes):
        proto = embeddings[labels == c].mean(dim=0)
        prototypes.append(proto)
    protos = torch.stack(prototypes)           # [C, D]
    return F.normalize(protos, dim=1)          # L2-normalise for cosine


def prototypical_predict(query_embeddings, prototypes):
    """
    query_embeddings: [Q, D]
    prototypes:       [C, D]  (should be L2-normalised)
    Returns:          [Q]  - predicted class indices
    """
    q = F.normalize(query_embeddings, dim=1)   # [Q, D]
    sims = q @ prototypes.T                    # [Q, C] cosine similarity
    return sims.argmax(dim=1)