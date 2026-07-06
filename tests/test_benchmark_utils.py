import pytest
import sys

from experiments.benchmark_utils import training_decomposition, wandb_log_result


class DummyModel:
    pass


def test_training_decomposition_for_kl_model():
    model = DummyModel()
    model.bb_alphas = [10.0, 8.0]
    model.KLs = [1.0, 1.5]
    model.prior_regularizers = [0.25]

    values = training_decomposition(model, model_type="vip")

    assert values["train/data_fit"] == 8.0
    assert values["train/kl"] == 1.5
    assert values["train/elbo_kl"] == 1.5
    assert values["train/prior_regularizer"] == 0.25
    assert values["train/evidence_regularizer"] == 0.25
    assert values["train/regularizer"] == 1.75
    assert values["train/objective_regularizer"] == 1.75
    assert values["train/regularizer_total"] == 1.75
    assert values["train/reconstructed_loss"] == 9.75


def test_training_decomposition_for_gmvip():
    model = DummyModel()
    model.data_terms = [4.0]
    model.KLs = [3.0]
    model.function_terms = [3.0]
    model.betas = [0.2]

    values = training_decomposition(model, model_type="gmvip")

    assert values["train/data_fit"] == 4.0
    assert values["train/gmvip_kl"] == 3.0
    assert values["train/gmvip_beta"] == 0.2
    assert values["train/regularizer"] == pytest.approx(0.6)
    assert values["train/reconstructed_loss"] == pytest.approx(4.6)


def test_training_decomposition_for_ftip_flow_components():
    model = DummyModel()
    model.bb_alphas = [2.0]
    model.KLs = [0.7]
    model.base_KLs = [0.9]
    model.flow_ldj = [-0.2]

    values = training_decomposition(model, model_type="ftip")

    assert values["train/data_fit"] == 2.0
    assert values["train/kl"] == 0.7
    assert values["train/ftip_base_kl"] == 0.9
    assert values["train/ftip_flow_ldj"] == -0.2


def test_wandb_log_result_uses_explicit_step(monkeypatch):
    class DummyRun:
        def __init__(self):
            self.summary = {}

    class DummyWandb:
        def __init__(self):
            self.run = DummyRun()
            self.logged = []

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    dummy = DummyWandb()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    wandb_log_result(
        {
            "train_time_s": 1.5,
            "train": {"RMSE": 0.1},
            "test": {"RMSE": 0.2},
            "prior": {},
        },
        step=30000,
    )

    assert dummy.logged[0][1] == 30000
    assert dummy.run.summary["final/test/RMSE"] == 0.2
