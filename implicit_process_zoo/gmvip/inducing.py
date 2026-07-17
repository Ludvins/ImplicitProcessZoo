import math

import torch


def _farthest_point_subset(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if num_points >= points.shape[0]:
        return points.clone()

    center = points.mean(dim=0, keepdim=True)
    first = torch.cdist(points, center).squeeze(-1).argmin()
    selected = [first]
    min_sqdist = (points - points[first]).square().sum(dim=1)

    for _ in range(1, int(num_points)):
        next_idx = min_sqdist.argmax()
        selected.append(next_idx)
        dist = (points - points[next_idx]).square().sum(dim=1)
        min_sqdist = torch.minimum(min_sqdist, dist)

    return points[torch.stack(selected)].clone()


def _latin_hypercube_grid(
    mins: torch.Tensor,
    maxs: torch.Tensor,
    num_points: int,
    seed: int | None,
) -> torch.Tensor:
    device = mins.device
    dtype = mins.dtype
    num_points = int(num_points)
    positions = (
        torch.arange(num_points, dtype=dtype, device=device) + 0.5
    ) / float(num_points)
    generator = torch.Generator(device=device)
    generator.manual_seed(0 if seed is None else int(seed))
    coords = []
    for _ in range(int(mins.numel())):
        perm = torch.randperm(num_points, generator=generator, device=device)
        coords.append(positions[perm])
    unit = torch.stack(coords, dim=1)
    return mins.unsqueeze(0) + unit * (maxs - mins).unsqueeze(0)


def _grid_inducing_points(
    X_train: torch.Tensor,
    num_inducing: int,
    seed: int | None,
) -> torch.Tensor:
    num_inducing = int(num_inducing)
    input_dim = int(X_train.shape[1])
    mins = X_train.min(dim=0).values
    maxs = X_train.max(dim=0).values

    if input_dim == 1:
        z = torch.linspace(
            float(mins[0]),
            float(maxs[0]),
            num_inducing,
            dtype=X_train.dtype,
            device=X_train.device,
        )
        return z.unsqueeze(-1)

    levels = max(2, math.ceil(num_inducing ** (1.0 / input_dim)))
    grid_size = levels ** input_dim
    max_grid_size = max(10_000, num_inducing * 64)

    if grid_size <= max_grid_size:
        axes = [
            torch.linspace(
                float(mins[d]),
                float(maxs[d]),
                levels,
                dtype=X_train.dtype,
                device=X_train.device,
            )
            for d in range(input_dim)
        ]
        grid = torch.cartesian_prod(*axes)
        return _farthest_point_subset(grid, num_inducing)

    return _latin_hypercube_grid(mins, maxs, num_inducing, seed)


def initialize_inducing_points(
    X_train: torch.Tensor,
    num_inducing: int,
    method: str = "kmeans",
    seed: int | None = None,
) -> torch.Tensor:
    """Select inducing inputs from training data.

    Parameters
    ----------
    X_train : torch.Tensor
        Training inputs with shape ``[N, D]``.
    num_inducing : int
        Number of inducing inputs.
    method : {"kmeans", "random_subset", "grid_1d", "train_quantiles"}
        Selection strategy.
    seed : int, optional
        Local seed for stochastic strategies.

    Returns
    -------
    torch.Tensor
        Inducing inputs with shape ``[num_inducing, D]``.

    Raises
    ------
    ValueError
        If shapes, counts, or the requested method are invalid.
    """
    if X_train.ndim != 2:
        raise ValueError("X_train must have shape [N, D].")
    if num_inducing <= 0:
        raise ValueError("num_inducing must be positive.")
    method = str(method)
    if num_inducing > X_train.shape[0] and method in {"random_subset", "train_quantiles"}:
        raise ValueError(
            "num_inducing cannot exceed the number of training points for this method."
        )

    if method == "grid":
        return _grid_inducing_points(X_train, num_inducing, seed)

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
        positions = (
            torch.linspace(
                0,
                sorted_x.shape[0] - 1,
                int(num_inducing),
                dtype=X_train.dtype,
                device=X_train.device,
            )
            .round()
            .long()
        )
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

    raise ValueError(
        "method must be 'grid', 'random_subset', 'kmeans', 'grid_1d', or 'train_quantiles'."
    )
