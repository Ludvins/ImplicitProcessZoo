import torch

from .utils.likelihood import gaussian_logp
from .utils.utils import infinite_loader


class DeterministicMAP(torch.nn.Module):
    """Deterministic MLP MAP baseline with learned isotropic Gaussian noise."""

    def __init__(
        self,
        input_dim,
        output_dim,
        structure,
        activation,
        num_data,
        l2=1e-4,
        y_mean=0.0,
        y_std=1.0,
        log_variance_init=-5.0,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if l2 < 0:
            raise ValueError("l2 must be non-negative.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_data = num_data
        self.l2 = l2
        self.activation = activation
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

        dims = [input_dim] + list(structure) + [output_dim]
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_dim, out_dim, dtype=dtype, device=device)
                for in_dim, out_dim in zip(dims, dims[1:])
            ]
        )
        log_variance_value = torch.as_tensor(log_variance_init, dtype=dtype, device=device)
        if log_variance_value.ndim > 1:
            raise ValueError("log_variance_init must be scalar or one-dimensional.")
        self.log_variance = torch.nn.Parameter(log_variance_value.clone())
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))

        self.data_terms = []
        self.l2_terms = []
        self.KLs = []

    def predict_f(self, X):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        x = X
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def predict_f_samples(self, X, S=1):
        F = self.predict_f(X)
        return F.unsqueeze(0).expand(S, *F.shape)

    def forward(self, X):
        F = self.predict_f(X).unsqueeze(0)
        mean = F * self.y_std + self.y_mean
        sigma = torch.exp(0.5 * self.log_variance)
        if sigma.ndim == 0:
            sigma = sigma.view(1, 1, 1)
        else:
            sigma = sigma.view(1, 1, -1)
        sigma = sigma * self.y_std
        return mean, sigma.expand_as(mean)

    def predict(self, X, S=1):
        self.eval()
        with torch.no_grad():
            mean, std = self(X)
            return mean.expand(S, *mean.shape[1:]), std.expand(S, *std.shape[1:])

    def predict_y_samples(self, X, S=1):
        mean, std = self.predict(X, S)
        return mean + std * torch.randn_like(mean)

    def regularizer(self):
        l2 = torch.zeros((), dtype=self.dtype, device=self.log_variance.device)
        for param in self.layers.parameters():
            l2 = l2 + param.square().sum()
        return 0.5 * self.l2 * l2

    def nelbo(self, X, y):
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        target_device = self.device if self.device is not None else X.device
        X = X.to(dtype=self.dtype, device=target_device)
        y = y.to(dtype=self.dtype, device=target_device)

        F = self.predict_f(X)
        logpdf = gaussian_logp(F, y, self.log_variance).sum()
        data_term = -(self.num_data / X.shape[0]) * logpdf
        l2_term = self.regularizer()
        loss = data_term + l2_term

        self.data_terms.append(data_term.detach())
        self.l2_terms.append(l2_term.detach())
        self.KLs.append(l2_term.detach())
        return loss

    def _train_step(self, optimizer, X, y):
        self.train()
        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(
        self,
        train_loader,
        optimizer=None,
        lr=1e-3,
        epochs=None,
        iterations=None,
        use_tqdm=False,
        return_loss=False,
        cosine_annealing=False,
    ):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            total = epochs if epochs is not None else max(1, iterations // len(train_loader))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total, eta_min=lr / 100
            )

        losses = []
        if iterations is not None:
            stream = infinite_loader(train_loader)
            for i in range(iterations):
                X, y = next(stream)
                loss = self._train_step(optimizer, X, y)
                if return_loss:
                    losses.append(loss.detach().cpu().numpy())
                if scheduler is not None and (i + 1) % len(train_loader) == 0:
                    scheduler.step()
            return losses

        for _ in range(epochs):
            for X, y in train_loader:
                loss = self._train_step(optimizer, X, y)
                if return_loss:
                    losses.append(loss.detach().cpu().numpy())
            if scheduler is not None:
                scheduler.step()
        return losses
