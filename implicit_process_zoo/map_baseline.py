import torch

from .utils.likelihood import gaussian_logp
from .utils.random import fork_torch_rng, preserve_constructor_rng
from .utils.training import fit_loop, make_cosine_scheduler, validate_fit_mode


@preserve_constructor_rng
class DeterministicMAP(torch.nn.Module):
    """Deterministic MLP MAP baseline with learned Gaussian noise.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of outputs.
    structure : sequence of int
        Width of each hidden layer.
    activation : callable
        Activation applied after every hidden layer.
    num_data : int
        Number of observations in the complete training set.
    l2 : float, default=1e-4
        Weight-decay coefficient for the MAP objective.
    y_mean : float or torch.Tensor, default=0.0
        Training-target mean used to restore the original scale.
    y_std : float or torch.Tensor, default=1.0
        Training-target standard deviation used to restore the original scale.
    log_variance_init : float or torch.Tensor, default=-5.0
        Initial observation log variance.
    device : torch.device or str, optional
        Device on which to create parameters.
    dtype : torch.dtype, default=torch.float64
        Parameter and computation data type.
    seed : int, default=2147483647
        Local seed used to initialize the network.
    """

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
        with fork_torch_rng(seed):
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

    def predict_f_samples(self, X, num_samples, *, seed=None):
        """Repeat latent MAP predictions along a sample axis.

        Parameters
        ----------
        X : torch.Tensor
            Inputs with shape ``[N, input_dim]``.
        num_samples : int
            Number of repeated function samples.
        seed : int, optional
            Accepted for compatibility; deterministic samples do not use it.

        Returns
        -------
        torch.Tensor
            Function samples with shape ``[S, N, D]``.
        """
        F = self.predict_f(X)
        return F.unsqueeze(0).expand(num_samples, *F.shape)

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

    def predict(self, X, num_samples, *, seed=None):
        self.eval()
        with torch.no_grad():
            mean, std = self(X)
            return mean.expand(num_samples, *mean.shape[1:]), std.expand(
                num_samples, *std.shape[1:]
            )

    def predict_y_samples(self, X, num_samples, *, seed=None):
        """Draw Gaussian observation samples.

        Parameters
        ----------
        X : torch.Tensor
            Inputs with shape ``[N, input_dim]``.
        num_samples : int
            Number of observation samples.
        seed : int, optional
            Local seed for observation noise.

        Returns
        -------
        torch.Tensor
            Observation samples with shape ``[S, N, D]``.
        """
        with fork_torch_rng(seed):
            mean, std = self.predict(X, num_samples)
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
        """Fit for exactly one epoch- or iteration-based duration.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Minibatches of input and target tensors.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use; defaults to Adam.
        lr : float, default=1e-3
            Learning rate used when creating the default optimizer.
        epochs : int, optional
            Number of complete passes over ``train_loader``.
        iterations : int, optional
            Number of optimizer steps. Mutually exclusive with ``epochs``.
        use_tqdm : bool, default=False
            Whether to display a progress bar.
        return_loss : bool, default=False
            Whether to return the per-step loss history.
        cosine_annealing : bool, default=False
            Whether to apply cosine learning-rate annealing.

        Returns
        -------
        list of float or None
            Loss history when ``return_loss`` is true, otherwise ``None``.
        """
        validate_fit_mode(epochs=epochs, iterations=iterations)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = (
            make_cosine_scheduler(optimizer, train_loader, epochs=epochs, iterations=iterations)
            if cosine_annealing
            else None
        )
        return fit_loop(
            self,
            train_loader,
            optimizer,
            epochs=epochs,
            iterations=iterations,
            use_tqdm=use_tqdm,
            return_loss=return_loss,
            scheduler=scheduler,
        )
