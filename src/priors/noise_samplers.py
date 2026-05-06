import torch


class GaussianSampler:
    """Seeded standard normal noise sampler."""

    def __init__(self, seed=2147483647, device=None, dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device)
        self.seed = seed
        self.generator.manual_seed(seed)

    def __call__(self, shape):
        return torch.randn(
            shape, generator=self.generator, device=self.device, dtype=self.dtype
        )

    def reset_seed(self):
        self.generator.manual_seed(self.seed)


class UniformSampler:
    """Seeded uniform [0, 1) noise sampler."""

    def __init__(self, seed=2147483647, device=None, dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

    def __call__(self, shape):
        return torch.rand(
            shape, generator=self.generator, device=self.device, dtype=self.dtype
        )
