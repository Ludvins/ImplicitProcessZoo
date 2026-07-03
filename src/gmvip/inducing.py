import torch


def initialize_inducing_points(
    X_train: torch.Tensor,
    num_inducing: int,
    method: str = "kmeans",
    seed: int | None = None,
) -> torch.Tensor:
    if X_train.ndim != 2:
        raise ValueError("X_train must have shape [N, D].")
    if num_inducing <= 0:
        raise ValueError("num_inducing must be positive.")
    if num_inducing > X_train.shape[0] and method in {"random_subset", "train_quantiles"}:
        raise ValueError("num_inducing cannot exceed the number of training points for this method.")

    method = str(method)
    if method == "grid_1d":
        if X_train.shape[1] != 1:
            raise ValueError("grid_1d requires one-dimensional inputs.")
        z = torch.linspace(
            float(X_train[:, 0].min()),
            float(X_train[:, 0].max()),
            int(num_inducing),
            dtype=X_train.dtype,
            device=X_train.device,
        )
        return z.unsqueeze(-1)

    if method == "train_quantiles":
        if X_train.shape[1] != 1:
            raise ValueError("train_quantiles requires one-dimensional inputs.")
        sorted_x = X_train[:, 0].sort().values
        positions = torch.linspace(
            0,
            sorted_x.shape[0] - 1,
            int(num_inducing),
            dtype=X_train.dtype,
            device=X_train.device,
        ).round().long()
        return sorted_x[positions].unsqueeze(-1)

    if method == "random_subset":
        generator = torch.Generator(device=X_train.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        perm = torch.randperm(X_train.shape[0], generator=generator, device=X_train.device)
        selected = X_train[perm[: int(num_inducing)]].clone()
        if selected.shape[1] == 1:
            selected = selected[torch.argsort(selected[:, 0])]
        return selected

    if method == "kmeans":
        # Small deterministic PyTorch fallback; enough for inducing initialization.
        generator = torch.Generator(device=X_train.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        if X_train.shape[0] <= num_inducing:
            return X_train.clone()
        perm = torch.randperm(X_train.shape[0], generator=generator, device=X_train.device)
        centers = X_train[perm[: int(num_inducing)]].clone()
        for _ in range(25):
            distances = torch.cdist(X_train, centers)
            labels = distances.argmin(dim=1)
            new_centers = centers.clone()
            for idx in range(int(num_inducing)):
                mask = labels == idx
                if torch.any(mask):
                    new_centers[idx] = X_train[mask].mean(dim=0)
            if torch.allclose(new_centers, centers):
                break
            centers = new_centers
        if centers.shape[1] == 1:
            centers = centers[torch.argsort(centers[:, 0])]
        return centers

    raise ValueError("method must be 'random_subset', 'kmeans', 'grid_1d', or 'train_quantiles'.")
