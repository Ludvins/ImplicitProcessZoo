import numpy as np
import torch

from ..utils.random import preserve_constructor_rng
from .noise_samplers import GaussianSampler, UniformSampler


class GenerativeFunction(torch.nn.Module):
    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=True,
        seed=2147483647,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic function using sampled
        noise values and input values.

        Parameters
        ----------
        num_samples : int
                      Number of samples to generate.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__()
        self.num_samples = num_samples
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fix_random_noise = fix_random_noise
        self.device = device
        self.seed = seed
        self.dtype = dtype

    def freeze_parameters(self):
        """Makes the model parameters non-trainable."""
        for param in self.parameters():
            param.requires_grad = False

    def defreeze_parameters(self):
        """Set the model parameters as trainable."""
        for param in self.parameters():
            param.requires_grad = True

    def regularizer(self):
        """Override in subclasses to add a regularization term to the loss."""
        return 0

    def forward(self):
        raise NotImplementedError


@preserve_constructor_rng
class BayesLinear(GenerativeFunction):
    """Gaussian-weight Bayesian linear layer producing coherent samples."""

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=True,
        zero_mean_prior=False,
        weight_log_sigma_init=0.0,
        seed=0,
        dtype=torch.float64,
    ):
        """
        Generates samples from a stochastic Bayesian Linear function
        f(x) = w^T x + b,   where w and b follow a Gaussian distribution,
        parameterized by their mean and log standard deviation.

        Parameters:
        -----------
        num_samples : int
                      Number of samples to generate.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        zero_mean_prior : boolean
                          wether to consider 0 mean prior or not, i. e, to
                          create variables for the mean values of the gaussian
                          distributions or fix these to 0.
        weight_log_sigma_init : float
                          Initial value for ``weight_log_sigma`` and
                          ``bias_log_sigma``. Default 0.0 (sigma=1, matching
                          the prior). Set to a smaller value (e.g. -3 or -5)
                          to start MFVI/FBNN closer to a deterministic net,
                          which reduces gradient noise and helps escape
                          posterior collapse on harder tasks.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.zero_mean_prior = zero_mean_prior
        # Instantiate Standard Gaussian sampler
        self.gaussian_sampler = GaussianSampler(seed, device, dtype=dtype)

        # If the BNN has zero mean, no parameters are considered for the
        # mean values the weights and bias variable

        if zero_mean_prior:
            self.weight_mu = 0
            self.bias_mu = 0
        else:
            self.weight_mu = torch.nn.Parameter(
                torch.zeros([input_dim, output_dim], dtype=dtype, device=device)
            )
            self.bias_mu = torch.nn.Parameter(
                torch.zeros([1, output_dim], dtype=dtype, device=device)
            )

        self.weight_log_sigma = torch.nn.Parameter(
            torch.full([input_dim, output_dim], weight_log_sigma_init, dtype=dtype, device=device)
        )
        self.bias_log_sigma = torch.nn.Parameter(
            torch.full([1, output_dim], weight_log_sigma_init, dtype=dtype, device=device)
        )

        # Reset the generator's seed if fixed noise.
        self.gaussian_sampler.reset_seed()
        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        else:
            # Compute the shape of the noise to generate
            z_w_shape = (self.num_samples, self.input_dim, self.output_dim)
            z_b_shape = (self.num_samples, 1, self.output_dim)

            # Generate Gaussian values
            z_w = self.gaussian_sampler(z_w_shape)
            z_b = self.gaussian_sampler(z_b_shape)

            return (z_w, z_b)

    def forward(self, inputs):
        """Forwards the given input through the Bayesian Linear layer.

        Arguments
        ---------

        inputs : torch tensor of shape (S, N, D)
                 Input tensor where the last two dimensions are batch and
                 data dimensionality.
        """
        if inputs.shape[-1] != self.input_dim:
            raise RuntimeError("Input shape does not match stored data dimension")

        z_w, z_b = self.get_noise()
        w = self.weight_mu + z_w * torch.exp(self.weight_log_sigma)
        b = self.bias_mu + z_b * torch.exp(self.bias_log_sigma)
        return inputs @ w + b

    def KL(self):
        """
        Computes the KL divergence of w and b to their prior distribution,
        a standard Gaussian N(0, I).

        Returns
        -------
        KL : int
             The addition of the 2 KL terms computed
        """
        # Compute covariance diagonal matrixes
        w_Sigma = torch.square(torch.exp(self.weight_log_sigma))
        b_Sigma = torch.square(torch.exp(self.bias_log_sigma))

        # Compute the 2*KL divergence of w
        KL = -self.input_dim * self.output_dim
        KL += torch.sum(w_Sigma)
        KL += torch.sum(self.weight_mu**2)
        KL -= 2 * torch.sum(self.weight_log_sigma)

        # Compute the 2*KL divergence of b
        KL -= self.output_dim
        KL += torch.sum(b_Sigma)
        KL += torch.sum(self.bias_mu**2)
        KL -= 2 * torch.sum(self.bias_log_sigma)

        # Re-escale
        return KL / 2

    def regularizer(self):
        return self.KL()


@preserve_constructor_rng
class SimplerBayesLinear(BayesLinear):
    """Bayesian linear layer with scalar (shared) mean and log-sigma per layer.

    This matches the DVIP paper (Ortega et al., ICLR 2023) parameterization:
    all weights in a layer share a single mean and a single log-std, and
    likewise for biases.  This drastically reduces the number of prior
    hyper-parameters compared to ``BayesLinear`` (per-weight parameters).
    """

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        device=None,
        fix_random_noise=True,
        zero_mean_prior=False,
        weight_log_sigma_init=0.0,
        seed=0,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            zero_mean_prior=zero_mean_prior,
            weight_log_sigma_init=weight_log_sigma_init,
            seed=seed,
            dtype=dtype,
        )
        # Override the per-weight parameters initialized by BayesLinear with
        # scalar shared parameters (on the correct device/dtype).
        if zero_mean_prior:
            self.weight_mu = 0
            self.bias_mu = 0
        else:
            self.weight_mu = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
            self.bias_mu = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))

        self.weight_log_sigma = torch.nn.Parameter(
            torch.tensor(weight_log_sigma_init, dtype=dtype, device=device)
        )
        self.bias_log_sigma = torch.nn.Parameter(
            torch.tensor(weight_log_sigma_init, dtype=dtype, device=device)
        )

    def KL(self):
        d_w = self.input_dim * self.output_dim
        d_b = self.output_dim

        w_var = torch.exp(self.weight_log_sigma) ** 2
        b_var = torch.exp(self.bias_log_sigma) ** 2

        KL = -d_w + d_w * w_var + d_w * self.weight_mu**2 - 2 * d_w * self.weight_log_sigma
        KL += -d_b + d_b * b_var + d_b * self.bias_mu**2 - 2 * d_b * self.bias_log_sigma

        return KL / 2


@preserve_constructor_rng
class BayesianNN(GenerativeFunction):
    """Multilayer Bayesian neural-network prior function generator."""

    def __init__(
        self,
        structure,
        activation,
        num_samples,
        input_dim,
        output_dim,
        layer_model,
        dropout=0.0,
        seed=2147483647,
        fix_random_noise=True,
        zero_mean_prior=False,
        weight_log_sigma_init=0.0,
        device=None,
        dtype=torch.float64,
    ):
        """
        Defines a Bayesian Neural Network with multiple layers.

        Parameters:
        -----------
        structure : array-like
                    Contains the inner dimensions of the Bayesian Neural
                    network. For example, [10, 10] symbolizes a Bayesian
                    network with 2 inner layers of width 10.
        activation : function
                     Activation function to use between inner layers.
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        layer_model :

        dropout : float between 0 and 1
                  The degree of dropout used after each activation layer
        device : torch.device
                 The device in which the computations are made.
        fix_random_noise : boolean
                           Wether to reset the Random Generator's seed in each
                           iteration.
        zero_mean_prior : boolean
                          Wether to consider zero mean layers.
        seed : int
               Initial seed for the random number generator.
        dtype : data-type
                The dtype of the layer's computations and weights.

        """
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.input_dim = input_dim

        # Store parameters
        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.dropout = torch.nn.Dropout(dropout)
        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [self.input_dim] + structure + [output_dim]
        layers = []

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):
            # Append the Bayesian linear layer to the array of layers
            layers.append(
                layer_model(
                    self.num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    zero_mean_prior=zero_mean_prior,
                    weight_log_sigma_init=weight_log_sigma_init,
                    seed=seed,
                    dtype=dtype,
                )
            )
        # Store the layers as ModuleList so that pytorch can handle
        # training/evaluation modes and parameters.
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        """Forward pass over each inner layer, activation is applied on every
        but the last level.

        Parameters
        ----------
        inputs : torch tensor of shape (N, D)
                 Contains the minibatch of N points with dimensionality D.

        Returns
        -------
        samples : torch tensor of shape (num_samples, N, D)
                  All the results of propagaring the input
                  num_samples times over the BNN.
        """

        # Replicate the input on the first dimension as many times as
        #  desired samples. expand() is a zero-copy view unlike tile().
        x = inputs.unsqueeze(0).expand(self.num_samples, *inputs.shape)

        for layer in self.layers[:-1]:
            # Apply BNN layer
            x = self.activation(layer(x))
            # Pytorch internally handles when the dropout layer is in
            # training mode. Moreover, if p = 0, no bernoully samples
            # are taken, so there is no additional computational cost
            # in calling this function in evaluation or p=0.
            x = self.dropout(x)

        # Last layer has identity activation function
        return self.layers[-1](x)

    def KL(self):
        """Computes the Kl divergence of the model as the addition of the
        KL divergences of its sub-models."""
        return torch.stack([layer.KL() for layer in self.layers]).sum()

    def regularizer(self):
        return self.KL()


@preserve_constructor_rng
class BayesianCNN(GenerativeFunction):
    """LeNet-style CNN feature extractor (deterministic) + Bayesian linear head.

    Architecture (LeNet-5 variant):
        Conv2d(in, 6, 5) -> ReLU -> AvgPool2d(2)
        Conv2d(6, 16, 5) -> ReLU -> AvgPool2d(2)
        Flatten -> BayesLinear(feat_dim, 120) -> ReLU
        BayesLinear(120, 84) -> ReLU
        BayesLinear(84, output_dim)

    The conv layers are deterministic; the three FC layers are Bayesian,
    producing ``num_samples`` stochastic outputs.

    Supports MNIST (1x28x28, input_dim=784) and CIFAR10 (3x32x32, input_dim=3072).
    """

    _SHAPE_MAP = {
        784: (1, 28, 28),  # MNIST
        3072: (3, 32, 32),  # CIFAR10
    }

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        layer_model=BayesLinear,
        head_dims=None,
        dropout=0.0,
        device=None,
        fix_random_noise=True,
        weight_log_sigma_init=0.0,
        seed=2147483647,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        if input_dim not in self._SHAPE_MAP:
            raise ValueError(
                f"BayesianCNN: unsupported input_dim={input_dim}. "
                f"Supported: {list(self._SHAPE_MAP.keys())}"
            )
        self.image_shape = self._SHAPE_MAP[input_dim]  # (C, H, W)
        in_channels = self.image_shape[0]

        # --- LeNet conv feature extractor (deterministic) ---
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        ).to(device=device, dtype=dtype)

        # Compute flattened feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, *self.image_shape, dtype=dtype, device=device)
            feat_dim = self.features(dummy).reshape(1, -1).shape[1]

        # --- Bayesian FC head (LeNet-5: 120 -> 84 -> output_dim) ---
        if head_dims is None:
            head_dims = [120, 84]
        self.dropout = torch.nn.Dropout(dropout)

        dims = [feat_dim] + list(head_dims) + [output_dim]
        layers = []
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                layer_model(
                    num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    weight_log_sigma_init=weight_log_sigma_init,
                    seed=seed,
                    dtype=dtype,
                )
            )
        self.head = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : (N, D) flat tensor

        Returns
        -------
        (num_samples, N, output_dim)
        """
        N = inputs.shape[0]
        # Reshape to image and extract features (deterministic, shared across samples)
        x_img = inputs.reshape(N, *self.image_shape)
        feat = self.features(x_img)  # (N, 16, H', W')
        feat = feat.reshape(N, -1)  # (N, feat_dim)

        # Expand for num_samples and pass through Bayesian head
        x = feat.unsqueeze(0).expand(self.num_samples, N, -1)  # (S, N, feat_dim)

        for layer in self.head[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.head[-1](x)  # (S, N, output_dim)

    def KL(self):
        return torch.stack([layer.KL() for layer in self.head]).sum()

    def regularizer(self):
        return self.KL()


@preserve_constructor_rng
class BayesConv2d(GenerativeFunction):
    """Bayesian 2-D convolution with mean-field Gaussian weight posterior.

    Per-forward, draws ``num_samples`` independent weight tensors from
    ``q(w) = N(weight_mu, exp(2*weight_log_sigma))`` (and similarly for bias),
    and applies all S convolutions in a single ``F.conv2d`` call via the
    grouped-conv trick (``groups=S``).

    Input  : ``(S, N, C_in, H, W)``
    Output : ``(S, N, C_out, H_out, W_out)``

    With ``fix_random_noise=True`` the noise tensor is sampled once at init
    and reused on every forward (consistent with ``BayesLinear``); in that
    mode the layer behaves as a fixed-noise basis rather than a fresh-noise
    MFVI sampler.
    """

    def __init__(
        self,
        num_samples,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        device=None,
        fix_random_noise=True,
        weight_log_sigma_init=0.0,
        seed=0,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim=in_channels,
            output_dim=out_channels,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )
        ksz = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.kernel_size = ksz
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gaussian_sampler = GaussianSampler(seed, device, dtype=dtype)

        # weight_mu shape: (out, in, kH, kW); broadcasts over S.
        self.weight_mu = torch.nn.Parameter(
            torch.zeros(out_channels, in_channels, *ksz, dtype=dtype, device=device)
        )
        self.weight_log_sigma = torch.nn.Parameter(
            torch.full(
                (out_channels, in_channels, *ksz), weight_log_sigma_init, dtype=dtype, device=device
            )
        )
        # bias_mu shape: (out,); broadcasts over S.
        self.bias_mu = torch.nn.Parameter(torch.zeros(out_channels, dtype=dtype, device=device))
        self.bias_log_sigma = torch.nn.Parameter(
            torch.full((out_channels,), weight_log_sigma_init, dtype=dtype, device=device)
        )

        self.gaussian_sampler.reset_seed()
        if self.fix_random_noise:
            self.noise = self.get_noise(first_call=True)

    def get_noise(self, first_call=False):
        if self.fix_random_noise and not first_call:
            return self.noise
        z_w = self.gaussian_sampler(
            (self.num_samples, self.out_channels, self.in_channels, *self.kernel_size)
        )
        z_b = self.gaussian_sampler((self.num_samples, self.out_channels))
        return (z_w, z_b)

    def forward(self, x):
        """Forward pass over a (S, N, C_in, H, W) tensor using grouped conv."""
        if x.dim() != 5:
            raise RuntimeError(f"BayesConv2d expects (S, N, C, H, W); got shape {tuple(x.shape)}")
        S, N, C_in, H, W = x.shape
        if self.num_samples != S or C_in != self.in_channels:
            raise RuntimeError(
                f"BayesConv2d expected S={self.num_samples} C_in={self.in_channels}, "
                f"got S={S} C_in={C_in}"
            )

        z_w, z_b = self.get_noise()
        sigma_w = torch.exp(self.weight_log_sigma)
        sigma_b = torch.exp(self.bias_log_sigma)
        # Broadcast (S, out, in, kH, kW) and (S, out)
        w = self.weight_mu.unsqueeze(0) + z_w * sigma_w.unsqueeze(0)
        b = self.bias_mu.unsqueeze(0) + z_b * sigma_b.unsqueeze(0)

        # Grouped-conv batching: rearrange so each MC sample s acts as a group.
        # x_g: (N, S*C_in, H, W); w_g: (S*C_out, C_in, kH, kW); b_g: (S*C_out,)
        x_g = x.permute(1, 0, 2, 3, 4).reshape(N, S * C_in, H, W)
        w_g = w.reshape(S * self.out_channels, C_in, *self.kernel_size)
        b_g = b.reshape(S * self.out_channels)
        y_g = torch.nn.functional.conv2d(
            x_g,
            w_g,
            b_g,
            stride=self.stride,
            padding=self.padding,
            groups=S,
        )
        H_o, W_o = y_g.shape[-2:]
        # Reshape back to (S, N, C_out, H', W')
        return y_g.reshape(N, S, self.out_channels, H_o, W_o).permute(1, 0, 2, 3, 4).contiguous()

    def KL(self):
        """KL( q(w,b) || N(0, I) ) — mean-field Gaussian, summed over weights+bias."""
        w_var = torch.exp(self.weight_log_sigma) ** 2
        b_var = torch.exp(self.bias_log_sigma) ** 2
        d_w = self.weight_mu.numel()
        d_b = self.bias_mu.numel()
        KL = -d_w
        KL = KL + torch.sum(w_var) + torch.sum(self.weight_mu**2)
        KL = KL - 2 * torch.sum(self.weight_log_sigma)
        KL = KL - d_b
        KL = KL + torch.sum(b_var) + torch.sum(self.bias_mu**2)
        KL = KL - 2 * torch.sum(self.bias_log_sigma)
        return KL / 2


@preserve_constructor_rng
class BayesianCNNFull(GenerativeFunction):
    """Fully Bayesian LeNet-5 variant: Bayesian conv stack AND Bayesian head.

    Architecture (mirrors :class:`BayesianCNN` but with stochastic conv):
        BayesConv2d(in, 6, 5)  -> ReLU -> AvgPool2d(2)
        BayesConv2d(6, 16, 5)  -> ReLU -> AvgPool2d(2)
        Flatten -> BayesLinear(feat_dim, 120) -> ReLU
        BayesLinear(120, 84) -> ReLU
        BayesLinear(84, output_dim)

    Each forward draws ``num_samples`` samples of every Bayesian weight (conv
    and FC). With ``weight_log_sigma_init=0.0`` (default) every layer adds
    noise of std 1 — typically too noisy for SGD; pass ``-3`` (matching the
    MFVI/FBNN classification defaults) when training MFVI/FBNN with this
    backbone.

    Supports the same ``input_dim`` values as :class:`BayesianCNN`.
    """

    _SHAPE_MAP = {
        784: (1, 28, 28),  # MNIST / FashionMNIST
        3072: (3, 32, 32),  # CIFAR10
    }

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        layer_model=None,
        head_dims=None,
        dropout=0.0,
        device=None,
        fix_random_noise=True,
        weight_log_sigma_init=0.0,
        seed=2147483647,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        if input_dim not in self._SHAPE_MAP:
            raise ValueError(
                f"BayesianCNNFull: unsupported input_dim={input_dim}. "
                f"Supported: {list(self._SHAPE_MAP.keys())}"
            )
        if layer_model is None:
            layer_model = BayesLinear

        self.image_shape = self._SHAPE_MAP[input_dim]  # (C, H, W)
        in_channels = self.image_shape[0]

        # --- Bayesian conv feature extractor ---
        self.conv1 = BayesConv2d(
            num_samples,
            in_channels,
            6,
            kernel_size=5,
            padding=2,
            device=device,
            fix_random_noise=fix_random_noise,
            weight_log_sigma_init=weight_log_sigma_init,
            seed=seed,
            dtype=dtype,
        )
        self.conv2 = BayesConv2d(
            num_samples,
            6,
            16,
            kernel_size=5,
            padding=0,
            device=device,
            fix_random_noise=fix_random_noise,
            weight_log_sigma_init=weight_log_sigma_init,
            seed=seed + 1,
            dtype=dtype,
        )

        # Compute flattened feature dim by running the deterministic-shape conv
        # arithmetic on a dummy single sample (no need to allocate (S, ...)).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.image_shape, dtype=dtype, device=device)
            # Use F.conv2d directly to avoid the (S, ...) layout for shape calc.
            tmp = torch.nn.functional.conv2d(
                dummy.squeeze(0),
                torch.zeros(6, in_channels, 5, 5, dtype=dtype, device=device),
                padding=2,
            )
            tmp = torch.nn.functional.avg_pool2d(tmp, 2, 2)
            tmp = torch.nn.functional.conv2d(
                tmp,
                torch.zeros(16, 6, 5, 5, dtype=dtype, device=device),
            )
            tmp = torch.nn.functional.avg_pool2d(tmp, 2, 2)
            feat_dim = tmp.reshape(1, -1).shape[1]

        # --- Bayesian FC head ---
        if head_dims is None:
            head_dims = [120, 84]
        self.dropout = torch.nn.Dropout(dropout)

        dims = [feat_dim] + list(head_dims) + [output_dim]
        layers = []
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                layer_model(
                    num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    weight_log_sigma_init=weight_log_sigma_init,
                    seed=seed,
                    dtype=dtype,
                )
            )
        self.head = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu

    @staticmethod
    def _avg_pool_5d(x, kernel_size, stride):
        """Apply avg_pool2d to (S, N, C, H, W) by flattening S and N."""
        S, N, C, H, W = x.shape
        x4 = x.reshape(S * N, C, H, W)
        y4 = torch.nn.functional.avg_pool2d(x4, kernel_size=kernel_size, stride=stride)
        H_o, W_o = y4.shape[-2:]
        return y4.reshape(S, N, C, H_o, W_o)

    def forward(self, inputs):
        """Returns (num_samples, N, output_dim)."""
        N = inputs.shape[0]
        x = inputs.reshape(N, *self.image_shape)
        # Expand to (S, N, C, H, W). Inputs are shared across MC samples.
        x = x.unsqueeze(0).expand(self.num_samples, *x.shape).contiguous()

        # Bayesian conv block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self._avg_pool_5d(x, kernel_size=2, stride=2)
        # Bayesian conv block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self._avg_pool_5d(x, kernel_size=2, stride=2)
        # Flatten to (S, N, feat_dim)
        x = x.reshape(self.num_samples, N, -1)

        # Bayesian FC head
        for layer in self.head[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.head[-1](x)  # (S, N, output_dim)

    def KL(self):
        return (
            self.conv1.KL()
            + self.conv2.KL()
            + torch.stack([layer.KL() for layer in self.head]).sum()
        )

    def regularizer(self):
        return self.KL()


@preserve_constructor_rng
class BayesianResNet(GenerativeFunction):
    """torchvision ResNet feature extractor (deterministic) + Bayesian linear head.

    Backbone is built via ``torchvision.models.<backbone>(weights=None)``; its
    final ``.fc`` is replaced with ``nn.Identity()`` to extract pooled features,
    which are then fed into a Bayesian FC head:

        BayesLinear(feat_dim, output_dim)            (head_dims=[], default)
      or BayesLinear(feat_dim, h1) -> ReLU -> ... -> BayesLinear(h_k, output_dim)

    For CIFAR10 (32×32), the default ImageNet stem (7×7 stride-2 conv + maxpool)
    down-samples too aggressively. Set ``cifar_stem=True`` to swap it for a
    3×3 stride-1 conv with no maxpool.

    Only ``input_dim=3072`` (CIFAR10) is validated; other sizes will work as long
    as the torchvision backbone accepts them.
    """

    _SHAPE_MAP = {3072: (3, 32, 32)}  # CIFAR10

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        layer_model=BayesLinear,
        head_dims=None,
        dropout=0.0,
        backbone="resnet18",
        cifar_stem=True,
        device=None,
        fix_random_noise=True,
        weight_log_sigma_init=0.0,
        seed=2147483647,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )
        if input_dim not in self._SHAPE_MAP:
            raise ValueError(
                f"BayesianResNet: unsupported input_dim={input_dim}. "
                f"Supported: {list(self._SHAPE_MAP.keys())}."
            )
        self.image_shape = self._SHAPE_MAP[input_dim]

        import torchvision.models as tvm

        if not hasattr(tvm, backbone):
            raise ValueError(f"Unknown torchvision backbone: {backbone!r}")
        net = getattr(tvm, backbone)(weights=None)

        if cifar_stem:
            # Swap ImageNet stem for CIFAR-friendly stem (3×3 s1, no maxpool).
            net.conv1 = torch.nn.Conv2d(
                3, net.conv1.out_channels, 3, stride=1, padding=1, bias=False
            )
            net.maxpool = torch.nn.Identity()

        feat_dim = net.fc.in_features
        net.fc = torch.nn.Identity()
        self.features = net.to(device=device, dtype=dtype)

        if head_dims is None:
            head_dims = []
        self.dropout = torch.nn.Dropout(dropout)
        dims = [feat_dim] + list(head_dims) + [output_dim]
        layers = []
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                layer_model(
                    num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    weight_log_sigma_init=weight_log_sigma_init,
                    seed=seed,
                    dtype=dtype,
                )
            )
        self.head = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu

    def forward(self, inputs):
        N = inputs.shape[0]
        x_img = inputs.reshape(N, *self.image_shape)
        feat = self.features(x_img)  # (N, feat_dim) — fc replaced by Identity
        x = feat.unsqueeze(0).expand(self.num_samples, N, -1)
        for layer in self.head[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.head[-1](x)

    def KL(self):
        return torch.stack([layer.KL() for layer in self.head]).sum()

    def regularizer(self):
        return self.KL()


@preserve_constructor_rng
class BayesianLSTM(GenerativeFunction):
    """LSTM encoder (deterministic) + Bayesian MLP head.

    Architecture:
        LSTM(feature_dim, lstm_hidden, lstm_layers) -> take final hidden
        BayesLinear(lstm_hidden, head_dims[0]) -> ReLU
        ...
        BayesLinear(head_dims[-1], output_dim)

    The LSTM layers are deterministic; the FC head layers are Bayesian,
    producing ``num_samples`` stochastic outputs.

    The input is a flat (N, t_obs * feature_dim) tensor which is reshaped
    internally to (N, t_obs, feature_dim) before feeding the LSTM — same
    pattern as BayesianCNN reshaping flat pixels to images.
    """

    def __init__(
        self,
        num_samples,
        input_dim,
        output_dim,
        t_obs=8,
        feature_dim=2,
        lstm_hidden=64,
        lstm_layers=1,
        head_dims=None,
        layer_model=BayesLinear,
        dropout=0.0,
        device=None,
        fix_random_noise=True,
        seed=2147483647,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples,
            input_dim,
            output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        self.t_obs = t_obs
        self.feature_dim = feature_dim

        # --- Deterministic LSTM encoder ---
        self.encoder = torch.nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        ).to(device=device, dtype=dtype)

        # --- Bayesian FC head ---
        if head_dims is None:
            head_dims = [64, 32]
        self.dropout = torch.nn.Dropout(dropout)

        dims = [lstm_hidden] + list(head_dims) + [output_dim]
        layers = []
        for _in, _out in zip(dims, dims[1:]):
            layers.append(
                layer_model(
                    num_samples,
                    _in,
                    _out,
                    device=device,
                    fix_random_noise=fix_random_noise,
                    seed=seed,
                    dtype=dtype,
                )
            )
        self.head = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : (N, t_obs * feature_dim) flat tensor

        Returns
        -------
        (num_samples, N, output_dim)
        """
        N = inputs.shape[0]
        # Reshape to sequence and encode (deterministic, shared across samples)
        x_seq = inputs.reshape(N, self.t_obs, self.feature_dim)
        _, (h_n, _) = self.encoder(x_seq)  # h_n: (lstm_layers, N, lstm_hidden)
        feat = h_n[-1]  # (N, lstm_hidden) — last layer's final hidden state

        # Expand for num_samples and pass through Bayesian head
        x = feat.unsqueeze(0).expand(self.num_samples, N, -1)  # (S, N, lstm_hidden)

        for layer in self.head[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.head[-1](x)  # (S, N, output_dim)

    def KL(self):
        return torch.stack([layer.KL() for layer in self.head]).sum()

    def regularizer(self):
        return self.KL()

    @property
    def layers(self):
        # Expose the Bayesian head as `.layers` so MFVI/FBNN/TFSVI helpers
        # (which iterate `gen_fn.layers` to mutate num_samples / regenerate
        # noise) work transparently with this LSTM-headed BNN. Properties
        # don't double-register the underlying ModuleList in `_modules`.
        return self.head


@preserve_constructor_rng
class GP(GenerativeFunction):
    """Gaussian-process function generator with configurable kernel."""

    def __init__(
        self,
        num_samples=1,
        input_dim=1,
        output_dim=1,
        inner_layer_dim=10,
        kernel_amp=1,
        kernel_length=1,
        fix_random_noise=True,
        seed=2147483647,
        device=None,
        dtype=torch.float64,
        **kwargs,
    ):
        """Generates samples from a Bayesian Neural Network that
        approximates a GP with 0 mean and RBF kernel. More precisely,
        the RBF kernel is approximated by
        RBF(x1, x2) = E_w,b [(cos wx_1 + b)cos(wx_2 + b)]
        where w ~ N(0, 1) and b ~ U(0, 2pi). This implies that
        phi(x) = cos(wx + b)
        can be used as kernel function to aproximate the kernel
        RBF(x1, x2) ~ phi(x1) phi(x2)
        Samples from the process can be drawn using the
        reparameterization trick, samples = phi * N(0, 1).
        Using this information, a network with two layers is used,
        the inner one computes phi and the last one the samples.
        Source: Random Features for Large-Scale Kernel Machines
                by Ali Rahimi and Ben Recht
        https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        Parameters
        ----------
        input_dim : int
                    Dimensionality of the input data.
        output_dim : int
                     Dimensionality of the output targets.
        inner_layer_dim : int
                          Dimensionality of the hidden layer, i.e,
                          number of samples of w and b to aproximate
                          the expectation.
        kernel_amp : float
                     Amplitude of the RBF kernel that is being
                     approximated.
        kernel_length : float
                        Length of the RBF kernel that is being
                        approximated.
        seed : int
               Random number seed.
        fix_random_noise : Wether to fix the sampled noise to be
                           the same on each call.
        device : torch device
                 Device in which computations are made.
        dtype : data dtype
                Data type to use (precision).
        """
        super().__init__(
            num_samples=num_samples,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )

        # Initialize variables and noise generators
        self.inner_layer_dim = inner_layer_dim
        self.inner_layer_dim_inv = 1 / self.inner_layer_dim
        self.gaussian_sampler = GaussianSampler(seed, device, dtype=dtype)
        self.uniform_sampler = UniformSampler(seed, device, dtype=dtype)

        # Initialize parameters, logarithms are used in order to avoid
        #  constraining to positive values.
        self.log_kernel_amp = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_amp, dtype=self.dtype))
        )
        self.log_kernel_length = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_length, dtype=self.dtype))
        )
        self.rng = np.random.default_rng(0)
        self.z_mean = torch.tensor(
            self.rng.normal(size=(self.input_dim, self.inner_layer_dim)) * 1,
            dtype=self.dtype,
            device=self.device,
        )
        self.b_mean = torch.tensor(
            self.rng.uniform(size=(1, self.inner_layer_dim)) * 2 * np.pi,
            dtype=self.dtype,
            device=self.device,
        )
        self.w_mean = torch.tensor(
            self.rng.normal(size=(self.inner_layer_dim, output_dim)) * 0.01,
            dtype=self.dtype,
            device=self.device,
        )

    def get_noise(self, num_samples):

        z = self.z_mean + self.gaussian_sampler((self.input_dim, self.inner_layer_dim))
        b = self.b_mean + 2 * np.pi * self.uniform_sampler((1, self.inner_layer_dim)) - np.pi
        # Compute the shape of the noise to generate
        w = self.w_mean + self.gaussian_sampler(
            (num_samples, self.inner_layer_dim, self.output_dim)
        )

        return z, b, w

    def forward(self, inputs, num_samples=None):
        """Approximated samples of a zero-mean GP with RBF kernel.

        Parameters
        ----------
        inputs : torch tensor of shape (..., N, D)
        num_samples : int, optional
            Number of prior samples. Defaults to ``self.num_samples`` so
            this generative function is a drop-in replacement for
            ``BayesianNN(x)``.
        """
        if num_samples is None:
            num_samples = self.num_samples
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(2.0 * torch.exp(self.log_kernel_amp) / self.output_dim)
        z, b, w = self.get_noise(num_samples)

        phi = scale_factor * torch.cos(x @ z + b)

        return phi @ w

    def KL(self):
        """Zero: the RFF GP has no Bayesian weight posterior to regularize.

        Only the kernel hyperparameters (amplitude, length-scale) are
        learned point-estimates, so there is no KL term to contribute.
        Returned as a tensor so it composes with the ELBO loss.
        """
        return torch.zeros((), dtype=self.dtype, device=self.device)

    def forward_mean(self, inputs):
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(2.0 * torch.exp(self.log_kernel_amp) / self.output_dim)
        phi = scale_factor * torch.cos(x @ self.z_mean + self.b_mean)
        return phi @ self.w_mean

    def forward_weights(self, inputs, weights):
        x = inputs / torch.exp(self.log_kernel_length)
        scale_factor = torch.sqrt(2.0 * torch.exp(self.log_kernel_amp) / self.output_dim)
        phi = scale_factor * torch.cos(x @ weights[0] + weights[1])

        return phi @ weights[2]

    def get_weights(self):
        return tuple([self.z_mean, self.b_mean, self.w_mean])

    def get_std_params(self):
        z = torch.ones((self.input_dim, self.inner_layer_dim))
        b = torch.tensor(2 * np.pi / (np.sqrt(12)), dtype=self.dtype) + torch.zeros(
            (1, self.inner_layer_dim)
        )
        w = torch.ones((self.inner_layer_dim, self.output_dim))
        return torch.cat([torch.log(z.flatten()), torch.log(b.flatten()), torch.log(w.flatten())])


@preserve_constructor_rng
class ExactGP(GenerativeFunction):
    """Exact zero-mean Gaussian-process prior with RBF kernel.

    Samples f(X) ~ N(0, K(X, X)) where
        K(x, x') = amp^2 * exp(-||x - x'||^2 / (2 * length^2)).
    The N x N kernel matrix is built and Cholesky-decomposed at every
    forward pass; practical only for small-to-moderate minibatches.
    Only the kernel amplitude and length-scale are learned (point
    estimates via type-II MLE through the ELBO).

    API matches ``BayesianNN``: ``forward(x)`` returns (num_samples, N, D).
    """

    def __init__(
        self,
        num_samples=1,
        input_dim=1,
        output_dim=1,
        kernel_amp=1.0,
        kernel_length=1.0,
        jitter=1e-6,
        fix_random_noise=True,
        seed=2147483647,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_samples=num_samples,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            fix_random_noise=fix_random_noise,
            seed=seed,
            dtype=dtype,
        )
        self.log_kernel_amp = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_amp, dtype=dtype, device=device))
        )
        self.log_kernel_length = torch.nn.Parameter(
            torch.log(torch.tensor(kernel_length, dtype=dtype, device=device))
        )
        self.jitter = jitter
        self.gaussian_sampler = GaussianSampler(seed, device, dtype=dtype)
        self.gaussian_sampler.reset_seed()
        self._cached_z = None
        self._cached_z_shape = None
        self._cached_cholesky_key = None
        self._cached_cholesky = None

    def _rbf(self, X):
        """Pairwise RBF kernel matrix on a single batch of points."""
        length = torch.exp(self.log_kernel_length)
        amp = torch.exp(self.log_kernel_amp)
        Xs = (X**2).sum(-1, keepdim=True)  # (N, 1)
        sq = Xs + Xs.transpose(-2, -1) - 2 * X @ X.transpose(-2, -1)
        sq = torch.clamp(sq, min=0.0)
        return amp**2 * torch.exp(-0.5 * sq / (length**2))

    def _cholesky_cache_key(self, X):
        """Return a safe cache key for repeated, non-adaptive GP inputs."""
        if X.requires_grad:
            return None
        if self.log_kernel_amp.requires_grad or self.log_kernel_length.requires_grad:
            return None
        return (
            id(X),
            X.data_ptr(),
            tuple(X.shape),
            tuple(X.stride()),
            X.storage_offset(),
            X.dtype,
            X.device.type,
            X.device.index,
            getattr(X, "_version", 0),
            getattr(self.log_kernel_amp, "_version", 0),
            getattr(self.log_kernel_length, "_version", 0),
            self.jitter,
        )

    def _cholesky(self, X):
        key = self._cholesky_cache_key(X)
        if key is not None and key == self._cached_cholesky_key:
            return self._cached_cholesky
        K = self._rbf(X)
        eye = torch.eye(X.shape[0], dtype=self.dtype, device=K.device)
        jitter = float(self.jitter)
        L = None
        for _ in range(6):
            L, info = torch.linalg.cholesky_ex(K + jitter * eye)
            if int(info.max().detach().cpu()) == 0:
                break
            jitter *= 10.0
        else:
            L = torch.linalg.cholesky(K + jitter * eye)
        if key is not None:
            self._cached_cholesky_key = key
            self._cached_cholesky = L
        else:
            self._cached_cholesky_key = None
            self._cached_cholesky = None
        return L

    def forward(self, inputs, num_samples=None):
        """Exact GP samples at ``inputs``.

        Parameters
        ----------
        inputs : (N, D) or (S, N, D) tensor
        num_samples : int, optional
            Defaults to ``self.num_samples``.

        Returns
        -------
        (num_samples, N, output_dim) tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        # Collapse any leading sample dimension — the GP only needs the
        # distinct input points. Caller's extra leading dim is ignored.
        if inputs.ndim == 3:
            inputs = inputs[0]
        N = inputs.shape[0]
        shape = (num_samples, N, self.output_dim)
        # Always recompute K and L so gradients flow into the kernel
        # hyperparameters. Only the base noise z is cached under
        # fix_random_noise=True, which is what keeps the prior sample
        # functions f_s = L z_s the "same draw" across iterations —
        # even as the kernel hyperparameters are learned, the sample
        # paths move only through L, not through z.
        L = self._cholesky(inputs)
        if self.fix_random_noise:
            if self._cached_z is None or self._cached_z_shape != shape:
                self._cached_z = self.gaussian_sampler(shape)
                self._cached_z_shape = shape
            z = self._cached_z
        else:
            z = self.gaussian_sampler(shape)
        return torch.einsum("nm, smd -> snd", L, z)

    def KL(self):
        """Zero: only kernel hyperparameters are learned (point estimates)."""
        return torch.zeros((), dtype=self.dtype, device=self.device)
