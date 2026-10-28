"""Geometry probes on latent prompts z: identifiability, family classification, parameter regression."""

from __future__ import annotations

import torch


def pooled(z: torch.Tensor) -> torch.Tensor:
    return z.mean(dim=0)  # [n_latent, hidden] -> [hidden]


def flat(z: torch.Tensor) -> torch.Tensor:
    return z.reshape(-1)


def ridge_fit_predict(X_train, y_train, X_test, alpha=10.0):
    """Closed-form ridge with intercept, in dual form so it works when dim >> n."""
    X = torch.stack(X_train).double()
    Xt = torch.stack(X_test).double().to(X.device)
    y = torch.tensor(y_train, dtype=torch.double, device=X.device)
    x_mean, y_mean = X.mean(dim=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    K = Xc @ Xc.T
    alpha_vec = torch.linalg.solve(K + alpha * torch.eye(K.shape[0], dtype=torch.double, device=X.device), yc)
    return ((Xt - x_mean) @ (Xc.T @ alpha_vec) + y_mean).tolist()


def nearest_centroid_accuracy(train_vecs, train_labels, test_vecs, test_labels) -> float:
    labels = sorted(set(train_labels))
    centroids = torch.stack(
        [torch.stack([v for v, l in zip(train_vecs, train_labels) if l == lab]).mean(dim=0) for lab in labels]
    )
    test = torch.stack(test_vecs)
    preds = torch.cdist(test.float(), centroids.float()).argmin(dim=1).tolist()
    return sum(labels[p] == t for p, t in zip(preds, test_labels)) / len(test_labels)


def consistency_report(z_by_task: dict[str, list[torch.Tensor]], rep=flat) -> dict:
    """Same task / different support draws vs different tasks.

    ratio = mean same-task distance / mean different-task distance (lower is better;
    Phase 1 free-z was ~1.0). nn_retrieval = fraction of z whose nearest other z comes
    from the same task (chance = (k-1)/(N-1)).
    """
    names, vecs = [], []
    for name, zs in z_by_task.items():
        for z in zs:
            names.append(name)
            vecs.append(rep(z).float())
    X = torch.stack(vecs)
    D = torch.cdist(X, X)
    same = torch.tensor([[a == b for b in names] for a in names], device=D.device)
    eye = torch.eye(len(names), dtype=torch.bool, device=D.device)
    same_off = same & ~eye
    diff = ~same
    nn_idx = D.masked_fill(eye, float("inf")).argmin(dim=1).tolist()
    k = len(next(iter(z_by_task.values())))
    return {
        "same_task_mean_dist": D[same_off].mean().item(),
        "diff_task_mean_dist": D[diff].mean().item(),
        "ratio": (D[same_off].mean() / D[diff].mean()).item(),
        "nn_retrieval": sum(names[j] == names[i] for i, j in enumerate(nn_idx)) / len(names),
        "nn_chance": (k - 1) / (len(names) - 1),
        "n_tasks": len(z_by_task),
        "draws_per_task": k,
    }
