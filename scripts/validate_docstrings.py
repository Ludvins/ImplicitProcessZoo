"""Validate the published Python API against the NumPy docstring standard."""

from __future__ import annotations

import importlib
import inspect

from numpydoc.validate import validate

# Keep this manifest aligned with ``docs/api``. The project deliberately
# publishes a curated API instead of promising every legacy convenience method.
PUBLIC_API = {
    "implicit_process_zoo.data.canonical_dataset_name": (),
    "implicit_process_zoo.data.get_dataset": (),
    "implicit_process_zoo.flows.rq_spline_forward": (),
    "implicit_process_zoo.flows.flows.AffineLayer": ("forward",),
    "implicit_process_zoo.flows.flows.CouplingLayer": ("forward",),
    "implicit_process_zoo.flows.flows.CouplingFlow": ("set_affine", "forward"),
    "implicit_process_zoo.flows.spline_coupling.SplineCouplingLayer": ("forward",),
    "implicit_process_zoo.flows.spline_coupling.SplineCouplingFlow": (
        "set_affine",
        "forward",
    ),
    "implicit_process_zoo.flows.glow_mixing.InvertibleConv1x1LU": ("forward",),
    "implicit_process_zoo.flows.glow_mixing.SplineCoupling1x1Flow": (
        "set_affine",
        "forward",
    ),
    "implicit_process_zoo.map_baseline.DeterministicMAP": (
        "predict_f_samples",
        "predict_y_samples",
        "fit",
    ),
    "implicit_process_zoo.mfvi.MFVI": ("predict_f_samples", "predict_y_samples", "fit"),
    "implicit_process_zoo.fbnn.FBNN": ("predict_f_samples", "predict_y_samples", "fit"),
    "implicit_process_zoo.tfsvi.TFSVI": (
        "predict_f_samples",
        "predict_y_samples",
        "fit",
    ),
    "implicit_process_zoo.vip.VIP": ("predict_f_samples", "predict_y_samples", "fit"),
    "implicit_process_zoo.ftip.FTIP": (
        "predict_f_samples",
        "predict_y_samples",
        "warm_start_from_vip",
        "fit",
    ),
    "implicit_process_zoo.ftip.UnifiedFTIP": (
        "predict_f_samples",
        "predict_y_samples",
        "fit",
    ),
    "implicit_process_zoo.gmvip.GeneralizedMatheronVIP": (
        "predict_f_samples",
        "predict_y_samples",
        "predict_summary",
    ),
    "implicit_process_zoo.sip.SIP": ("predict_f_samples", "predict_y_samples", "fit"),
    "implicit_process_zoo.gmvip.RBFKernel": ("forward", "diag"),
    "implicit_process_zoo.gmvip.initialize_inducing_points": (),
    "implicit_process_zoo.gmvip.BaseMatheronOperator": (
        "psi",
        "inducing_mean",
        "inducing_scale_matrix",
        "apply",
    ),
    "implicit_process_zoo.gmvip.EmpiricalCovarianceMatheronOperator": (
        "psi",
        "inducing_mean",
        "inducing_scale_matrix",
        "mean_at",
    ),
    "implicit_process_zoo.gmvip.RBFCardinalMatheronOperator": (
        "psi",
        "inducing_mean",
        "inducing_scale_matrix",
        "mean_at",
    ),
    "implicit_process_zoo.gmvip.CholeskyGaussianCoefficientPosterior": (
        "rsample",
        "rsample_with_kl",
        "sample_prior",
        "kl_to_standard_normal",
    ),
    "implicit_process_zoo.gmvip.RealNVPCoefficientPosterior": (
        "rsample",
        "rsample_with_kl",
        "sample_prior",
        "kl_to_standard_normal",
        "log_prob",
    ),
    "implicit_process_zoo.gmvip.GaussianRegressionLikelihood": ("log_prob",),
    "implicit_process_zoo.priors.generative_functions.BayesLinear": ("forward",),
    "implicit_process_zoo.priors.generative_functions.BayesianNN": (
        "forward",
        "freeze_parameters",
        "defreeze_parameters",
    ),
    "implicit_process_zoo.priors.generative_functions.GP": ("forward",),
    "implicit_process_zoo.priors.function_bank.CoherentPriorFunctionSampler": (
        "sample_latents",
        "evaluate_latents",
        "sample_values",
    ),
    "implicit_process_zoo.priors.function_bank.PriorFunctionBank": ("evaluate",),
    "implicit_process_zoo.utils.prediction.batched_predict_samples": (),
    "implicit_process_zoo.utils.random.standard_normal_samples": (),
    "implicit_process_zoo.utils.training.validate_fit_mode": (),
    "implicit_process_zoo.utils.training.make_cosine_scheduler": (),
    "implicit_process_zoo.utils.training.fit_loop": (),
    "implicit_process_zoo.utils.checkpoints.capture_rng_state": (),
    "implicit_process_zoo.utils.checkpoints.restore_rng_state": (),
    "implicit_process_zoo.utils.checkpoints.build_training_checkpoint": (),
    "implicit_process_zoo.utils.checkpoints.save_training_checkpoint": (),
    "implicit_process_zoo.utils.checkpoints.load_training_checkpoint": (),
    "implicit_process_zoo.utils.checkpoints.restore_training_checkpoint": (),
    "implicit_process_zoo.utils.checkpoints.load_warm_start_state": (),
}

# NumPy-style structural and signature checks. Extended summaries, examples,
# and See Also sections are valuable when they add information, but are not
# mandatory for every small accessor or tensor transform.
ENFORCED_CHECKS = {
    "GL06",
    "GL07",
    "GL08",
    "GL10",
    "PR01",
    "PR02",
    "PR03",
    "PR04",
    "PR05",
    "PR06",
    "PR07",
    "PR08",
    "PR09",
    "PR10",
    "RT01",
    "RT02",
    "RT03",
    "RT04",
    "RT05",
    "SS01",
    "SS02",
    "SS03",
    "SS05",
    "SS06",
    "YD01",
}


def _resolve(path: str):
    """Resolve an import path to its Python object.

    Parameters
    ----------
    path : str
        Fully qualified import path.

    Returns
    -------
    object
        Imported module, class, function, or method.
    """
    parts = path.split(".")
    for stop in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        for part in parts[stop:]:
            value = getattr(value, part)
        return value
    raise ImportError(f"Cannot resolve public API object {path!r}.")


def public_api_paths() -> list[str]:
    """Collect the objects rendered by the curated API reference.

    Returns
    -------
    list of str
        Sorted, duplicate-free paths checked by the documentation gate.
    """
    paths = set(PUBLIC_API)
    for path, members in PUBLIC_API.items():
        value = _resolve(path)
        if members and not inspect.isclass(value):
            raise TypeError(f"Public API members require a class, got {path!r}.")
        paths.update(f"{path}.{member}" for member in members)
    return sorted(paths)


def main() -> int:
    """Validate the supported API and print actionable failures.

    Returns
    -------
    int
        Zero when all enforced checks pass, otherwise one.
    """
    failures: list[tuple[str, str, str]] = []
    paths = public_api_paths()
    for path in paths:
        value = _resolve(path)
        result = validate(path)
        failures.extend(
            (path, code, message)
            for code, message in result["errors"]
            if code in ENFORCED_CHECKS
            # Mkdocstrings merges constructor documentation into the class
            # page. Numpydoc otherwise reports constructor parameters as
            # missing from the class-level narrative docstring.
            and not (inspect.isclass(value) and code.startswith("PR"))
        )

    if failures:
        for path, code, message in failures:
            print(f"{path}: {code}: {message}")
        print(f"\n{len(failures)} NumPy-style errors across {len(paths)} public objects.")
        return 1

    print(f"Validated {len(paths)} public objects with NumPy-style docstrings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
