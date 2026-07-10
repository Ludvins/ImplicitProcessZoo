import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.classification import benchmark as classification_benchmark
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP


class TinyClassificationDataset:
    def __init__(self):
        rng = np.random.RandomState(0)
        self.inputs = rng.randn(8, 784).astype(np.float64)
        self.targets = rng.randint(0, 3, size=(8, 1)).astype(np.float64)
        self.input_dim = 784
        self.output_dim = 3
        self.classes = 3
        self.n_samples = self.inputs.shape[0]

    def __len__(self):
        return self.n_samples


def _args(model):
    return classification_benchmark.parse_args(
        [
            "--model",
            model,
            "--dataset",
            "FashionMNIST",
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--head_dims",
            "4",
            "--num_samples",
            "4",
            "--eval_samples",
            "4",
            "--fbnn_num_prior_samples",
            "4",
            "--fbnn_num_measurement",
            "4",
            "--fbnn_num_context",
            "4",
            "--tfsvi_S_ctx",
            "1",
            "--tfsvi_K_ctx",
            "2",
            "--sip_num_inducing",
            "3",
            "--sip_num_prior_samples",
            "4",
            "--sip_num_train_samples",
            "4",
            "--sip_critic_steps",
            "0",
            "--gmvip_num_inducing",
            "3",
            "--gmvip_num_operator_bank_samples",
            "4",
            "--gmvip_num_train_samples",
            "4",
            "--gmvip_num_eval_samples",
            "4",
            "--flow_depth",
            "1",
        ]
    )


def test_classification_parser_accepts_all_methods():
    for model in classification_benchmark.CLASSIFICATION_MODELS:
        args = _args(model)
        assert args.model == model


@pytest.mark.parametrize(
    "model_type",
    ["map", "vip", "ftip", "mfvi", "fbnn", "tfsvi", "gmvip", "sip"],
)
def test_classification_methods_build_and_predict_logits(model_type):
    dataset = TinyClassificationDataset()
    args = _args(model_type)

    model = classification_benchmark.build_model(args, dataset, model_type)
    if model_type == "gmvip":
        assert isinstance(model, GeneralizedMatheronVIP)
        assert model.likelihood_type == "multiclass"
    xb = torch.as_tensor(dataset.inputs[:2], dtype=torch.float64)
    samples = classification_benchmark.predict_logits_samples(
        model,
        xb,
        args,
        model_type,
    )

    assert samples.shape[-2:] == (2, 3)
    assert samples.ndim == 3
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "model_type",
    ["map", "vip", "ftip", "mfvi", "fbnn", "tfsvi", "gmvip", "sip"],
)
def test_classification_methods_run_one_train_step(model_type):
    dataset = TinyClassificationDataset()
    args = _args(model_type)
    model = classification_benchmark.build_model(args, dataset, model_type)
    X = torch.as_tensor(dataset.inputs[:4], dtype=torch.float64)
    y = torch.as_tensor(dataset.targets[:4], dtype=torch.float64)
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    classification_benchmark.initialize_function_context(model, model_type, loader)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-3,
    )

    loss = model._train_step(optimizer, X, y)

    assert torch.isfinite(loss.detach())


def test_classification_variant_filenames_do_not_collide():
    learn_args = _args("ftip")
    fixed_args = _args("ftip")
    fixed_args.ftip_learn_prior = False

    learn_name = classification_benchmark.result_file_name(
        "FashionMNIST",
        "ftip",
        learn_args,
        None,
    )
    fixed_name = classification_benchmark.result_file_name(
        "FashionMNIST",
        "ftip",
        fixed_args,
        None,
    )

    assert learn_name != fixed_name
