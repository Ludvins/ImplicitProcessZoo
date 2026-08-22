import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..data.downloads import download_source, fetch_and_extract
from ..data.sources import get_data_source

NORMALIZATION_EPSILON = 1e-6


class Training_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        verbose=True,
        normalize_inputs=True,
        normalize_targets=True,
    ):

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"inputs and targets must both be rank-2, got {inputs.shape} and {targets.shape}."
            )
        if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
            raise ValueError(
                f"inputs and targets must have the same nonzero row count, got {inputs.shape} and {targets.shape}."
            )
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("inputs and targets must contain only finite values.")
        self.inputs = inputs
        if normalize_targets:
            self.targets_mean = np.mean(targets, axis=0, keepdims=True)
            self.targets_std = np.maximum(
                np.std(targets, axis=0, keepdims=True), NORMALIZATION_EPSILON
            )
        else:
            self.targets_mean = 0
            self.targets_std = 1
        self.targets = (targets - self.targets_mean) / self.targets_std

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = targets.shape[1]

        # Normalize inputs
        if normalize_inputs:
            self.inputs_std = np.std(self.inputs, axis=0, keepdims=True) + 1e-6
            self.inputs_mean = np.mean(self.inputs, axis=0, keepdims=True)
        else:
            self.inputs_mean = 0
            self.inputs_std = 1

        self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std
        if verbose:
            print("Number of samples: ", self.n_samples)
            print("Input dimension: ", self.input_dim)
            print("Label dimension: ", self.output_dim)
            print("Labels mean value: ", self.targets_mean)
            print("Labels standard deviation: ", self.targets_std)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Test_Dataset(Dataset):
    def __init__(self, inputs, targets=None, inputs_mean=0.0, inputs_std=1.0):
        inputs = np.asarray(inputs)
        if inputs.ndim != 2 or inputs.shape[0] == 0:
            raise ValueError(f"inputs must be a nonempty rank-2 array, got {inputs.shape}.")
        if targets is not None:
            targets = np.asarray(targets)
            if targets.ndim != 2 or targets.shape[0] != inputs.shape[0]:
                raise ValueError(
                    f"targets must be rank-2 and align with inputs, got {targets.shape}."
                )
        self.inputs = (inputs - inputs_mean) / inputs_std
        self.targets = targets
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        if self.targets is not None:
            self.output_dim = targets.shape[1]

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


data_base = "./data"
airline_csv_url = (
    "https://media.githubusercontent.com/media/Ludvins/Variational-LLA/main/data/airline.csv"
)


def _downloaded_source_path(source_name: str) -> Path:
    source = get_data_source(source_name)
    return download_source(source, Path(data_base) / "downloads" / source.filename)


def _extracted_source_member(source_name: str, member: str) -> Path:
    outputs = fetch_and_extract(get_data_source(source_name), Path(data_base))
    return outputs[member]


def _torchvision_datasets_transforms():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for MNIST/FashionMNIST/CIFAR10 datasets. "
            "Install torchvision or use a non-image dataset."
        ) from exc
    return datasets, transforms


class CustomDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def split_data(self, data):
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
            raise ValueError(f"Expected a nonempty [N, features+target] matrix, got {data.shape}.")
        if not np.isfinite(data).all():
            raise ValueError("Dataset matrix contains NaN or infinite values.")
        self.inputs = data[:, :-1]
        self.targets = data[:, -1]

        if len(self.inputs.shape) == 1:
            self.inputs = self.inputs[..., np.newaxis]
        if len(self.targets.shape) == 1:
            self.targets = self.targets[..., np.newaxis]

    def get_split(self, test_size, seed):
        train_indexes, test_indexes = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=seed
        )

        train_dataset = Training_Dataset(
            self.inputs[train_indexes],
            self.targets[train_indexes],
            normalize_targets=self.type == "regression",
        )
        train_test_dataset = Test_Dataset(
            self.inputs[train_indexes],
            self.targets[train_indexes],
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )
        test_dataset = Test_Dataset(
            self.inputs[test_indexes],
            self.targets[test_indexes],
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )

        return train_dataset, train_test_dataset, test_dataset

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def len_train(self, test_size):
        train_indexes, test_indexes = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=0
        )
        return len(train_indexes)


class Constant_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=1234)

        n = rng.poisson(3, 1)[0]
        locations = rng.uniform(0, 1, n)
        values = rng.uniform(0, 1, n + 1)

        sort = np.argsort(locations)
        locations = locations[sort]

        def f(x):
            y = x * 0 + values[0]
            for i in range(n):
                a = np.where(x > locations[i])
                y[a] = values[i + 1]

            return y

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(0, 0.2, 50)
        inputs = np.concatenate([inputs, np.linspace(0.8, 1, 50)], axis=-1)
        targets = f(inputs) + rng.standard_normal(inputs.shape) * 0.01

        test_inputs = np.linspace(0, 1.0, 300)
        test_targets = f(test_inputs) + rng.standard_normal(test_inputs.shape) * 0.01

        targets = targets[..., np.newaxis]
        inputs = inputs[..., np.newaxis]
        test_inputs = test_inputs[..., np.newaxis]
        test_targets = test_targets[..., np.newaxis]

        self.train = Training_Dataset(
            inputs,
            targets,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            inputs,
            targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_inputs,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 100

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 100


class Linear_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        n = rng.poisson(3, 1)[0]
        locations = np.append(rng.uniform(0, 1, n), [0, 1])
        values = rng.uniform(0, 1, n + 2)

        sort = np.argsort(locations)
        locations = locations[sort]

        def f(x):
            y = x * 0 + values[0]
            a = (x.reshape(-1, 1) > locations.reshape(1, -1)).sum(axis=1) - 1
            y = (values[a + 1] - values[a]) / (locations[a + 1] - locations[a]) * (
                x - locations[a]
            ) + values[a]
            return y

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(0 + 1e-6, 0.2, 50)
        inputs = np.concatenate([inputs, np.linspace(0.8, 1 - 1e-6, 50)], axis=-1)
        targets = f(inputs) + rng.standard_normal(inputs.shape) * 0.01

        test_inputs = np.linspace(0, 1.0, 300)
        test_targets = f(test_inputs) + rng.standard_normal(test_inputs.shape) * 0.01

        targets = targets[..., np.newaxis]
        inputs = inputs[..., np.newaxis]
        test_inputs = test_inputs[..., np.newaxis]
        test_targets = test_targets[..., np.newaxis]

        self.train = Training_Dataset(
            inputs,
            targets,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            inputs,
            targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_inputs,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 100

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 100


class Bimodal_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        rng = np.random.default_rng(0)

        n = 2000
        x = rng.uniform(size=n) * 8 - 4
        epsilon = rng.normal(size=n)

        y = np.zeros_like(x)
        y[: n // 2] = 20 * np.cos(x[: n // 2] - 0.5) + epsilon[: n // 2]

        y[n // 2 :] = 20 * np.sin(x[n // 2 :] - 0.5) + epsilon[n // 2 :]
        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


class Heterocedastic_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        rng = np.random.default_rng(0)

        x = rng.uniform(size=2000) * 8 - 4
        epsilon = rng.normal(size=2000) * 2

        sin = np.sin(x)

        y = 7 * sin + epsilon * sin + 10

        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


Heteroscedastic_Dataset = Heterocedastic_Dataset


class Gap_Dataset(CustomDataset):
    """1D regression with a missing central interval.

    Observations live on [-4, -1] union [1, 4], while plots/evaluation grids can
    inspect the unobserved gap [-1, 1]. This is useful for visually comparing
    whether posterior predictive uncertainty expands where there is no data.
    """

    gap_bounds = (-1.0, 1.0)
    plot_domain = (-4.5, 4.5)

    @staticmethod
    def true_function(x):
        return 2.5 * np.sin(1.35 * x) + 0.35 * x + 0.35 * np.cos(3.0 * x)

    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        rng = np.random.default_rng(0)

        n_left = 1000
        n_right = 1000
        left = rng.uniform(-4.0, self.gap_bounds[0], size=n_left)
        right = rng.uniform(self.gap_bounds[1], 4.0, size=n_right)
        x = np.concatenate([left, right])
        rng.shuffle(x)

        noise = 0.25 * rng.normal(size=x.shape[0])
        y = self.true_function(x) + noise

        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


class SPGPSnelson_Dataset(CustomDataset):
    """Snelson/SPGP 1D regression data from ``data/spgp_snelson``.

    The public data bundle provides 200 labelled training observations and a
    301-point input grid used for plotting. It does not include labelled test
    targets for that grid, so ``get_split`` returns the training observations
    for evaluation while exposing the grid as ``plot_inputs``.
    """

    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data_dir = os.path.join(data_base, "spgp_snelson")
        train_inputs_path = os.path.join(data_dir, "snelson_train_inputs.dat")
        train_outputs_path = os.path.join(data_dir, "snelson_train_outputs.dat")
        test_inputs_path = os.path.join(data_dir, "snelson_test_inputs.dat")

        missing = [
            path
            for path in (train_inputs_path, train_outputs_path, test_inputs_path)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Missing SPGP Snelson data files. Expected files under "
                f"{data_dir!r}; missing: {missing}"
            )

        inputs = np.loadtxt(train_inputs_path, dtype=np.float64).reshape(-1, 1)
        targets = np.loadtxt(train_outputs_path, dtype=np.float64).reshape(-1, 1)
        self.plot_inputs = np.loadtxt(test_inputs_path, dtype=np.float64).reshape(-1, 1)
        self.plot_domain = (
            float(np.min(self.plot_inputs)),
            float(np.max(self.plot_inputs)),
        )

        self.inputs = inputs
        self.targets = targets

        self.train = Training_Dataset(inputs, targets)
        self.train_test = Test_Dataset(
            inputs,
            targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs,
            targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.inputs)

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return len(self.train)


class Synthetic_Dataset(CustomDataset):
    """Fixed 400-point scalar synthetic regression dataset."""

    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data_path = os.path.join(data_base, "synthetic", "data.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Missing synthetic regression data file. Expected "
                f"{data_path!r}. Download it from "
                "https://raw.githubusercontent.com/Ludvins/Variational-LLA/"
                "898ec63da3c1aa9c79f3c7d9d802453f83f80466/data/data.npy"
            )

        data = np.load(data_path).astype(np.float64)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Expected synthetic data with shape [N, 2], got {data.shape}")

        order = np.argsort(data[:, 0])
        data = data[order]
        self.inputs = data[:, :1]
        self.targets = data[:, 1:2]
        self.plot_domain = (
            float(np.min(self.inputs)),
            float(np.max(self.inputs)),
        )

        self.train = Training_Dataset(self.inputs, self.targets)
        self.train_test = Test_Dataset(
            self.inputs,
            self.targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            self.inputs,
            self.targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.inputs)

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return len(self.train)


class Skewed_Dataset(CustomDataset):
    """1D regression whose conditional p(y|x) is skewed (not Gaussian).

    y(x) = 5 sin(x) + s(x) * ( exp(sigma z) - exp(sigma^2 / 2) ),  z ~ N(0, 1)

    The additive noise is a centered lognormal (zero mean, heavy right tail).
    The tail direction is flipped at x = 0 via s(x) = sign(x), so the left
    half of the domain is left-skewed and the right half is right-skewed.
    A Gaussian posterior predictive must over-disperse symmetrically to
    cover either tail, which is exactly the failure mode FTIP's flow-based
    posterior should fix.
    """

    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        rng = np.random.default_rng(0)

        n = 20000
        x = rng.uniform(size=n) * 8 - 4
        sigma = 0.8
        z = norm.ppf(np.linspace(0.02, 0.98, n)) * 2
        rng.shuffle(z)

        eps = np.exp(sigma * z) - np.exp(sigma**2 / 2) + 1
        s = np.where(x >= 0, 1.0, -1.0)
        y = 5 * np.sin(x) + s * eps

        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


class Boston_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        raw_df = pd.read_fwf(_downloaded_source_path("boston"), header=None).to_numpy()
        self.split_data(raw_df)


class Energy_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_excel(_downloaded_source_path("energy")).values
        data = data[:, :9]
        self.split_data(data)


class Concrete_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_excel(_downloaded_source_path("concrete")).values
        self.split_data(data)


class Naval_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        path = _extracted_source_member("naval", "UCI CBM Dataset/data.txt")
        data = pd.read_fwf(path, header=None).values
        data = data[:, :-1]
        self.split_data(data)


class Power_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        path = _extracted_source_member("power", "CCPP/Folds5x2_pp.xlsx")
        data = pd.read_excel(path).values
        self.split_data(data)


class Protein_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_csv(_downloaded_source_path("protein")).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
        self.split_data(data)


class Protein_Bimodal_Dataset(CustomDataset):
    """Protein dataset with feature 2 dropped to induce conditional bimodality.

    The original protein (CASP) dataset has 9 input features. Feature 2 is a
    key disambiguator: removing it makes the residual distribution bimodal,
    creating a genuine (non-synthetic) multimodal regression problem.
    """

    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_csv(_downloaded_source_path("protein")).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
        # Drop feature 2 (0-indexed) from the 9 input columns
        data = np.delete(data, 2, axis=1)
        self.split_data(data)


class Kin8nm_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_csv(_downloaded_source_path("kin8nm"), dtype=float).values
        self.split_data(data)


class Yatch_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        # File is whitespace-delimited, not CSV; default pd.read_csv puts
        # every row into one string column and breaks downstream splitting.
        data = pd.read_csv(_downloaded_source_path("yacht"), header=None, sep=r"\s+").values
        self.split_data(data)


Yacht_Dataset = Yatch_Dataset


class WineRed_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data = pd.read_csv(_downloaded_source_path("winered"), delimiter=";").values
        self.split_data(data)


class CO2_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        path = Path(data_base) / "monthly_in_situ_co2_mlo.csv"
        if not path.exists():
            raise FileNotFoundError(
                "The historical Scripps CO2 URL used by this project is no longer available. "
                f"Place a reviewed source file at {path}."
            )
        data = (
            pd.read_csv(path, comment='"', header=None, on_bad_lines="skip")
            .apply(pd.to_numeric, errors="coerce")
            .dropna(axis=0, how="all")
            .values[:, [3, 4]]
        )
        mask = (data == -99.99).any(1)
        self.data = data[~mask]
        self.len_data = self.data.shape[0]

        split = self.len_data // 5
        test_indexes = np.concatenate(
            (np.arange(split, 2 * split), np.arange(3 * split, 4 * split)), axis=0
        )
        train_indexes = np.setdiff1d(np.arange(self.len_data), test_indexes)

        train_data = self.data[train_indexes, :1]
        test_data = self.data[test_indexes, :1]
        train_targets = self.data[train_indexes, 1:]
        test_targets = self.data[test_indexes, 1:]

        self.train = Training_Dataset(train_data, train_targets)

        self.train_test = Test_Dataset(
            self.data[:, :1],
            self.data[:, 1:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 778

    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, seed, *args):
        return self.train, self.train_test, self.test


class MNIST_Dataset(CustomDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        datasets, transforms = _torchvision_datasets_transforms()
        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        train_data = train.data.reshape(60000, -1) / 255.0
        test_data = test.data.reshape(10000, -1) / 255.0

        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 70000

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 60000


class FashionMNIST_Dataset(CustomDataset):
    """Flat-vector FashionMNIST wrapper (mirrors :class:`MNIST_Dataset`).

    Same shape as MNIST — 28×28 grayscale, 60K train / 10K test, 10 classes —
    so the same models and configs work without modification. Useful as an
    ID dataset whose OOD counterpart is plain MNIST.
    """

    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        datasets, transforms = _torchvision_datasets_transforms()
        train = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

        train_data = train.data.reshape(60000, -1) / 255.0
        test_data = test.data.reshape(10000, -1) / 255.0

        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            normalize_targets=False,
            normalize_inputs=False,
        )
        self.train_test = Test_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 70000

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 60000


class CIFAR10_Dataset(CustomDataset):
    """Flat-vector CIFAR10 wrapper (mirrors :class:`MNIST_Dataset`).

    Each example is a flattened ``3 × 32 × 32 = 3072`` float vector in
    ``[0, 1]``. :class:`implicit_process_zoo.priors.generative_functions.BayesianCNN` reshapes
    it back to ``(C, H, W)`` internally when used as the generative function.
    """

    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        datasets, transforms = _torchvision_datasets_transforms()
        train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

        # CIFAR10.data is (N, 32, 32, 3) uint8; permute to (N, C, H, W), flatten.
        import torch

        train_data = torch.tensor(train.data).float().permute(0, 3, 1, 2).reshape(50000, -1) / 255.0
        test_data = torch.tensor(test.data).float().permute(0, 3, 1, 2).reshape(10000, -1) / 255.0
        train_targets = torch.tensor(train.targets).reshape(-1, 1)
        test_targets = torch.tensor(test.targets).reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            normalize_targets=False,
            normalize_inputs=False,
        )
        self.train_test = Test_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 60000

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 50000


class MNIST_regression_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        datasets, transforms = _torchvision_datasets_transforms()
        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        train_data = train.data / 255.0
        test_data = test.data / 255.0

        # train_data = train.data.reshape(60000, -1)
        # test_data = test.data.reshape(10000, -1)

        train_targets = train_data
        test_targets = test_data

        self.train = Training_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 70000

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 60000


class Rectangles_Dataset(CustomDataset):
    def __init__(self):
        self.type = "binaryclass"
        self.classes = 2
        self.output_dim = 1

        train_data = np.loadtxt("data/rectangles/rectangles_im_train.amat")
        test_data = np.loadtxt("data/rectangles/rectangles_im_test.amat")

        self.len_data = train_data.shape[0] + test_data.shape[0]
        self.train = Training_Dataset(
            train_data[:, :-1],
            train_data[:, -1].reshape(-1, 1),
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            train_data[:, :-1],
            train_data[:, -1].reshape(-1, 1),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data[:, :-1],
            test_data[:, -1].reshape(-1, 1),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class Year_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        path = Path(data_base) / "YearPredictionMSD.txt"
        if not path.exists():
            raise FileNotFoundError(
                "YearPredictionMSD is too large for an implicit benchmark download. "
                f"Download it from the official UCI record, verify it, and place it at {path}."
            )
        data = pd.read_csv(path, header=None, delimiter=",").values

        self.len_data = data.shape[0]

        X = data[:, 1:]
        Y = data[:, 0].reshape(-1, 1)

        self.n_train = 463715
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train])
        self.train_test = Test_Dataset(
            X[: self.n_train],
            Y[: self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train :],
            Y[self.n_train :],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self, test_size):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class Airline_Dataset(CustomDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data_path = os.path.join(data_base, "airline.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Airline is too large for an implicit unverified download. "
                f"Review {airline_csv_url} and place the CSV at {data_path}."
            )

        data = pd.read_csv(data_path)
        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

        # Pick out the data
        Y = data["ArrDelay"].values[:800000].reshape(-1, 1)
        names = [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "plane_age",
            "AirTime",
            "Distance",
            "ArrTime",
            "DepTime",
        ]
        X = data[names].values[:800000]

        self.len_data = X.shape[0]
        self.n_train = min(700000, self.len_data)
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train])

        self.train_test = Test_Dataset(
            X[: self.n_train],
            Y[: self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train :],
            Y[self.n_train :],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self, test_size):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class HIGGS_Dataset(CustomDataset):
    def __init__(self):
        self.type = "binaryclass"
        self.classes = 2
        self.output_dim = 1

        if os.path.exists("data/HIGGS.csv"):
            print("HIGGS csv file found.")
            data = pd.read_csv("data/HIGGS.csv", header=None).values
        elif os.path.exists("data/HIGGS.csv.gz"):
            print("HIGGS gzip file found.")
            data = pd.read_csv("data/HIGGS.csv.gz", header=None, compression="gzip").values
        else:
            raise FileNotFoundError(
                "HIGGS is too large for an implicit benchmark download. Place the verified "
                "official HIGGS.csv or HIGGS.csv.gz file under data/."
            )

        print("HIGGS data loaded. Data shape: ", end="")
        print(data.shape)
        X = data[:, 1:]
        Y = data[:, 0].reshape(-1, 1)

        self.len_data = 11000000
        test_size = 500000
        self.train = Training_Dataset(
            X[: self.len_data - test_size],
            Y[: self.len_data - test_size],
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            X[: self.len_data - test_size],
            Y[: self.len_data - test_size],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.len_data - test_size :],
            Y[self.len_data - test_size :],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class SUSY_Dataset(CustomDataset):
    def __init__(self):
        self.type = "binaryclass"
        self.classes = 2
        self.output_dim = 1

        if os.path.exists("data/SUSY.csv"):
            print("SUSY csv file found.")
            data = pd.read_csv("data/SUSY.csv", header=None).values
        elif os.path.exists("data/SUSY.csv.gz"):
            print("SUSY gzip file found.")
            data = pd.read_csv("data/SUSY.csv.gz", header=None, compression="gzip").values
        else:
            raise FileNotFoundError(
                "SUSY is too large for an implicit benchmark download. Place the verified "
                "official SUSY.csv or SUSY.csv.gz file under data/."
            )

        print("SUSY data loaded. Data shape: ", end="")
        print(data.shape)
        X = data[:, 1:]
        Y = data[:, 0].reshape(-1, 1)

        self.len_data = 5000000
        test_size = 500000
        self.train = Training_Dataset(
            X[: self.len_data - test_size],
            Y[: self.len_data - test_size],
            normalize_targets=False,
            normalize_inputs=True,
        )

        self.train_test = Test_Dataset(
            X[: self.len_data - test_size],
            Y[: self.len_data - test_size],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.len_data - test_size :],
            Y[self.len_data - test_size :],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class Taxi_Dataset(CustomDataset):
    def __init__(self, year=2020):
        self.type = "regression"
        self.output_dim = 1

        if os.path.exists("data/taxi.csv"):
            print("Taxi csv file found.")
            data = pd.read_csv("data/taxi.csv")
        elif os.path.exists("data/taxi.zip"):
            print("Taxi zip file found.")
            data = pd.read_csv("data/taxi.zip", compression="zip", dtype=object)
        else:
            print("Downloading Taxi Dataset...")
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-01.parquet"
            os.makedirs("data", exist_ok=True)
            try:
                data = pd.read_parquet(url)
            except ImportError as exc:
                raise ImportError(
                    "Taxi dataset requires pyarrow: pip install pyarrow\n"
                    "Or download manually and place as data/taxi.csv"
                ) from exc
            data.to_csv("data/taxi.csv", index=False)

        data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
        data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["day_of_month"] = data["tpep_pickup_datetime"].dt.day
        data["month"] = data["tpep_pickup_datetime"].dt.month
        data["time_of_day"] = (
            data["tpep_pickup_datetime"] - data["tpep_pickup_datetime"].dt.normalize()
        ) / pd.Timedelta(seconds=1)
        data["trip_duration"] = (
            data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
        ).dt.total_seconds()
        # Use lat/lon if available (old CSV), otherwise use location IDs (new parquet)
        if "pickup_latitude" in data.columns:
            location_cols = [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ]
        else:
            location_cols = ["PULocationID", "DOLocationID"]
        feature_cols = (
            [
                "time_of_day",
                "day_of_week",
                "day_of_month",
                "month",
            ]
            + location_cols
            + ["trip_distance"]
        )
        data = data[feature_cols + ["trip_duration"]]
        data = data[data["trip_duration"] >= 10]
        data = data[data["trip_duration"] <= 5 * 3600]
        data = data.astype(float)
        data = data.values
        self.split_data(data)


class Arm2D_IK_Dataset(CustomDataset):
    """
    2D 2-link planar arm inverse kinematics dataset.

    Inputs:  end-effector position (x, y)  -> shape (N, 2)
    Targets: (sin(theta1), cos(theta1), sin(theta2), cos(theta2)) -> shape (N, 4)

    This problem is intrinsically multimodal because most (x,y) have two IK solutions.
    """

    def __init__(
        self,
        n=20000,
        l1=1.0,
        l2=1.0,
        input_noise_std=0.0,
        seed=0,
    ):
        self.type = "regression"
        self.output_dim = 4  # sin/cos for two angles

        rng = np.random.default_rng(seed=seed)

        # Sample joint angles uniformly
        theta1 = rng.uniform(-np.pi, np.pi, size=n)
        theta2 = rng.uniform(-np.pi, np.pi, size=n)

        # Forward kinematics
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

        inputs = np.stack([x, y], axis=1)

        if input_noise_std and input_noise_std > 0:
            inputs = inputs + rng.normal(0.0, input_noise_std, size=inputs.shape)

        # Stable target encoding (avoids angle wrap discontinuities)
        targets = np.stack(
            [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)],
            axis=1,
        )

        self.inputs = inputs.astype(np.float64)
        self.targets = targets.astype(np.float64)


class Pedestrian_Dataset(Dataset):
    """ETH / UCY pedestrian trajectory dataset, EigenTrajectory layout.

    Two supported layouts:

    1. Benchmark split (preferred, matches EigenTrajectory exactly):
       ``data_dir/<benchmark>/<split>/<scene>[_train|_val].txt``.
       Use with ``Pedestrian_Dataset(benchmark="eth", split="train")``.

    2. All-scenes layout (legacy / custom splitting in code):
       ``data_dir/raw/all_data/<sequence>.txt``.
       Use with ``Pedestrian_Dataset(sequences=[...])``.

    Each file is 4-column (tab- or space-separated):
    ``frame, ped_id, x, y``. Velocities are not available and are not
    returned.

    Each item is a dict with ``obs_traj``, ``pred_traj``, ``obs_mask``,
    ``seq_name``, ``ped_id``, ``start_frame``.
    """

    DEFAULT_SEQUENCES = (
        "biwi_eth",
        "biwi_hotel",
        "crowds_zara01",
        "crowds_zara02",
        "crowds_zara03",
        "students001",
        "students003",
        "uni_examples",
    )

    # Mapping for the 5 standard ETH/UCY leave-one-out benchmarks
    BENCHMARK_TEST_SCENES = {
        "eth": ["biwi_eth"],
        "hotel": ["biwi_hotel"],
        "univ": ["students001", "students003"],
        "zara1": ["crowds_zara01"],
        "zara2": ["crowds_zara02"],
    }

    def __init__(
        self,
        data_dir="./data/pedestrian",
        sequences=None,
        t_obs=8,
        t_pred=12,
        benchmark=None,
        split=None,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.window = t_obs + t_pred
        self.samples = []
        self.frame_steps = {}  # per-scene frame step
        self.benchmark = benchmark
        self.split = split

        if benchmark is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(
                    f"benchmark={benchmark!r} requires split in "
                    f"('train', 'val', 'test'); got split={split!r}"
                )
            if benchmark not in self.BENCHMARK_TEST_SCENES:
                raise ValueError(
                    f"Unknown benchmark={benchmark!r}; expected one of "
                    f"{list(self.BENCHMARK_TEST_SCENES)}"
                )
            # Collect the files present in data_dir/<benchmark>/<split>/.
            split_dir = os.path.join(data_dir, benchmark, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Expected directory {split_dir}; download with "
                    "an ETH/UCY data preparation helper."
                )
            files = sorted(f for f in os.listdir(split_dir) if f.endswith(".txt"))
            # Extract canonical scene name by stripping _train/_val suffix.
            file_info = []
            for fname in files:
                stem = fname[:-4]  # drop ".txt"
                if stem.endswith("_train"):
                    scene = stem[:-6]
                elif stem.endswith("_val"):
                    scene = stem[:-4]
                else:
                    scene = stem  # test files have no suffix
                file_info.append((os.path.join(split_dir, fname), scene))
        else:
            if sequences is None:
                sequences = self.DEFAULT_SEQUENCES
            file_info = [
                (os.path.join(data_dir, "raw", "all_data", s + ".txt"), s) for s in sequences
            ]

        for path, seq_name in file_info:
            raw = np.loadtxt(path)
            # columns: frame, ped_id, x, y
            frames = raw[:, 0].astype(int)
            ped_ids = raw[:, 1].astype(int)
            pos_xy = raw[:, 2:4]

            # Detect the frame step for this sequence
            unique_frames = np.sort(np.unique(frames))
            frame_step = int(np.median(np.diff(unique_frames)))
            self.frame_steps[seq_name] = frame_step

            for pid in np.unique(ped_ids):
                mask = ped_ids == pid
                p_frames = frames[mask]
                order = np.argsort(p_frames)
                p_frames = p_frames[order]
                p_pos = pos_xy[mask][order]

                # Split into contiguous segments
                gaps = np.where(np.diff(p_frames) != frame_step)[0] + 1
                seg_starts = np.concatenate([[0], gaps])
                seg_ends = np.concatenate([gaps, [len(p_frames)]])

                for s, e in zip(seg_starts, seg_ends):
                    seg_len = e - s
                    if seg_len < self.window:
                        continue
                    # Sliding window
                    for i in range(seg_len - self.window + 1):
                        self.samples.append(
                            {
                                "pos": p_pos[s + i : s + i + self.window],
                                "seq_name": seq_name,
                                "ped_id": int(pid),
                                "start_frame": int(p_frames[s + i]),
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        s = self.samples[index]
        pos = torch.tensor(s["pos"], dtype=torch.float32)
        return {
            "obs_traj": pos[: self.t_obs],
            "pred_traj": pos[self.t_obs :],
            "obs_mask": torch.ones(self.t_obs, dtype=torch.bool),
            "seq_name": s["seq_name"],
            "ped_id": s["ped_id"],
            "start_frame": s["start_frame"],
        }


def get_dataset(dataset_name):
    aliases = {
        "yatch": "yacht",
        "heterocedastic": "heteroscedastic",
    }
    if dataset_name in aliases:
        canonical = aliases[dataset_name]
        warnings.warn(
            f"Dataset name {dataset_name!r} is deprecated; use {canonical!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset_name = canonical
    d = {
        "bimodal": Bimodal_Dataset,
        "synthetic": Synthetic_Dataset,
        "constant": Constant_Dataset,
        "linear": Linear_Dataset,
        "gap": Gap_Dataset,
        "spgp_snelson": SPGPSnelson_Dataset,
        "snelson": SPGPSnelson_Dataset,
        "heteroscedastic": Heteroscedastic_Dataset,
        "skewed": Skewed_Dataset,
        "boston": Boston_Dataset,
        "energy": Energy_Dataset,
        "concrete": Concrete_Dataset,
        "naval": Naval_Dataset,
        "kin8nm": Kin8nm_Dataset,
        "yacht": Yacht_Dataset,
        "power": Power_Dataset,
        "protein": Protein_Dataset,
        "ProteinBimodal": Protein_Bimodal_Dataset,
        "winered": WineRed_Dataset,
        "CO2": CO2_Dataset,
        "MNIST": MNIST_Dataset,
        "FashionMNIST": FashionMNIST_Dataset,
        "CIFAR10": CIFAR10_Dataset,
        "MNIST2": MNIST_regression_Dataset,
        "Rectangles": Rectangles_Dataset,
        "Year": Year_Dataset,
        "year": Year_Dataset,
        "yearpredictionmsd": Year_Dataset,
        "Airline": Airline_Dataset,
        "airline": Airline_Dataset,
        "HIGGS": HIGGS_Dataset,
        "SUSY": SUSY_Dataset,
        "taxi": Taxi_Dataset,
        "arm2d": Arm2D_IK_Dataset,
        "pedestrian": Pedestrian_Dataset,
    }

    return d[dataset_name]()
