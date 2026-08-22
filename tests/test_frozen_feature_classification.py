import pytest
import torch

from experiments.common import frozen_feature_classification as frozen

EXPECTED_CANDIDATE_IDS = {
    "map": ["c00_efaa9a3601", "c01_4d33b85496"],
    "mfvi": [
        "c00_c2cb92729c",
        "c01_5376639569",
        "c02_20b5fdef8c",
        "c03_3db1a95624",
        "c04_2762b3d3d9",
        "c05_a23558e779",
    ],
    "vip": [
        "c00_5234421dc0",
        "c01_506c36e757",
        "c02_d2eec652d7",
        "c03_553efb1919",
        "c04_44a782db30",
        "c05_2ed6171c22",
    ],
    "ftip": [
        "c00_9a068ad7e4",
        "c01_44b3613936",
        "c02_b17729f7ac",
        "c03_e322242b8f",
        "c04_043dee884f",
        "c05_78effc5f29",
    ],
    "fbnn": [
        "c00_fb53dab1ad",
        "c01_621fa551dd",
        "c02_f2f9d079b9",
        "c03_f8325a269c",
        "c04_97a3669833",
        "c05_099c19a2c8",
    ],
    "tfsvi": [
        "c00_45d5c154a9",
        "c01_1031e46398",
        "c02_02f90d2ba1",
        "c03_3ad4e6d9b4",
        "c04_c9374f7136",
        "c05_de5b9f91f9",
    ],
    "gmvip": [
        "c00_4722bcccbf",
        "c01_9fc1ef6778",
        "c02_acbbf4ef9c",
        "c03_b9a455b1bf",
        "c04_67e33a8d0f",
        "c05_1f9bdc5e70",
    ],
    "sip": [
        "c00_7db9b50d46",
        "c01_97821892c5",
        "c02_d2aec08e2d",
        "c03_cebe6c3d20",
        "c04_de0ef33574",
        "c05_544b106459",
    ],
}


def split(feature_dim=4, num_classes=3, per_class=4):
    targets = torch.arange(num_classes).repeat_interleave(per_class)
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(len(targets), feature_dim, generator=generator)
    return frozen.Split(features, targets)


def test_candidate_grids_keep_stable_ids():
    grids = frozen.candidate_grids()
    assert set(grids) == set(frozen.METHODS)
    assert {
        method: [candidate["candidate_id"] for candidate in candidates]
        for method, candidates in grids.items()
    } == EXPECTED_CANDIDATE_IDS


@pytest.mark.parametrize("num_classes", [3, 10])
def test_map_uses_spec_dimensions_and_sample_first_predictions(num_classes):
    spec = frozen.FrozenFeatureSpec(512, num_classes)
    data = split(feature_dim=512, num_classes=num_classes, per_class=1)
    candidate = frozen.candidate_grids()["map"][0]
    model = frozen.build_model(spec, "map", candidate, data, 0, torch.device("cpu"))

    assert model.weight.shape == (num_classes, 512)
    logp, shape = frozen.predictive_log_probabilities(
        spec,
        "map",
        model,
        data,
        torch.device("cpu"),
        batch_size=32,
        num_samples=4,
    )
    assert shape == (4, num_classes, num_classes)
    assert logp.shape == (num_classes, num_classes)
    assert torch.isfinite(logp).all()


@pytest.mark.parametrize("train_prior", [False, True])
def test_fbnn_prior_policy(train_prior):
    spec = frozen.FrozenFeatureSpec(4, 3, train_fbnn_prior=train_prior)
    candidate = frozen.candidate_grids()["fbnn"][0]
    model = frozen.build_model(spec, "fbnn", candidate, split(), 0, torch.device("cpu"))

    assert frozen.trainable_prior(model, "fbnn") is train_prior
    frozen.check_model(spec, "fbnn", model)


@pytest.mark.parametrize("method", ["vip", "ftip", "gmvip", "sip"])
def test_function_priors_and_inducing_locations_are_trainable(method):
    spec = frozen.FrozenFeatureSpec(4, 3, train_fbnn_prior=False)
    candidate = frozen.candidate_grids()[method][0].copy()
    if "basis_samples" in candidate:
        candidate["basis_samples"] = 4
    if "prior_samples" in candidate:
        candidate["prior_samples"] = 4
    if "train_samples" in candidate:
        candidate["train_samples"] = 2
    if "num_inducing" in candidate:
        candidate["num_inducing"] = 4

    model = frozen.build_model(spec, method, candidate, split(), 0, torch.device("cpu"))

    assert frozen.trainable_prior(model, method)
    if method in ("gmvip", "sip"):
        assert frozen.trainable_inducing_locations(model, method)
    frozen.check_model(spec, method, model)


def test_temperature_scaling_preserves_classes_and_balanced_folds():
    spec = frozen.FrozenFeatureSpec(4, 3)
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(30, 3, generator=generator)
    targets = torch.arange(3).repeat(10)
    logp = torch.log_softmax(logits, dim=1)
    temperature = frozen.fit_temperature(logp, targets, iterations=10)
    scaled = frozen.apply_temperature(logp, temperature)

    assert temperature > 0
    assert torch.equal(logp.argmax(1), scaled.argmax(1))
    folds = frozen.balanced_folds(spec, targets, num_folds=5, seed=0)
    assert sorted(torch.cat(folds).tolist()) == list(range(len(targets)))
    assert all(
        torch.bincount(targets[indices], minlength=3).tolist() == [2, 2, 2] for indices in folds
    )


def test_spec_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="feature_dim"):
        frozen.FrozenFeatureSpec(0, 3)
    with pytest.raises(ValueError, match="num_classes"):
        frozen.FrozenFeatureSpec(4, 1)
