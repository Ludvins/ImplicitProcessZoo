#!/usr/bin/env python3
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _crps_A(mu, sigma_2, eps=1e-12):
    """Helper for the CRPS kernel: A(mu, sigma^2) = 2*sigma*phi(z) + mu*(2*Phi(z) - 1)."""
    sigma_2 = sigma_2.clamp_min(eps)
    sigma = torch.sqrt(sigma_2)
    z = mu / sigma
    norm = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    return 2 * sigma * torch.exp(norm.log_prob(z)) + mu * (2 * norm.cdf(z) - 1)


class Metrics:
    def __init__(self, num_data=-1, device=None):
        """Defines a class that encapsulates all considered metrics.

        Arguments
        ---------
        num_data : int
                   Number of data samples in the dataset used. This is used
                   to scale the metrics from each batch by the proportion of
                   data they contain.
                   This scale is never greater than 1.
        device : torch device
                 Device in which to make the computations.
        """

        self.num_data = num_data
        self.device = device
        self.reset()

    def reset(self):
        """Ressets all the metrics to zero."""
        self.loss = torch.tensor(0.0, device=self.device)
        self.nll = torch.tensor(0.0, device=self.device)

    def update(self, y, loss, mean_pred, std_pred, light=True):
        raise NotImplementedError

    def compute_nll(self, y, mean_pred, std_pred):
        l = self.logdensity(mean_pred, std_pred**2, y)
        l = torch.sum(l, -1)
        log_num_samples = torch.log(torch.tensor(mean_pred.shape[0]))
        lse = torch.logsumexp(l, axis=0) - log_num_samples
        return -torch.mean(lse)


class MetricsRegression(Metrics):
    def __init__(self, num_data=-1, device=None):
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.mse = torch.tensor(0.0, device=self.device)
        self.crps = torch.tensor(0.0, device=self.device)
        self.cqm = torch.tensor(0.0, device=self.device)
        self._pit_values = []

    def logdensity(self, mu, var, x):
        """Computes the log density of a one dimensional Gaussian distribution
        of mean mu and variance var, evaluated on x.
        """
        logp = -0.5 * (np.log(2 * np.pi) + var.log() + (mu - x).square() / var)
        return logp

    def update(self, y, loss, mean_pred, std_pred, light=True):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        mean_pred : torch tensor of shape (S, batch_size, output_dim)
                    Contains the mean predictions for each sample
                    in the batch.
        std_pred : torch tensor of shape (S, batch_size, output_dim)
                   Contains the std predictions for each sample
                   in the batch.
        light : boolean
                Wether to compute only the lighter (computationally) metrics.
        """
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
        # Update light metrics
        self.loss += scale * loss
        # The RMSE is computed using the mean prediction of the Gaussian
        #  Mixture, that is, the mean of the mean predictions.
        self.mse += scale * self.compute_mse(y, mean_pred.mean(0))
        self.nll += scale * self.compute_nll(y, mean_pred, std_pred)
        # Update heavy metrics
        if not light:
            self.crps += scale * self.compute_crps(y, mean_pred, std_pred)
            self._pit_values.append(self._compute_pit(y, mean_pred, std_pred))

    def compute_mse(self, y, prediction):
        """Computes the root mean squared error for the given predictions."""
        return torch.nn.functional.mse_loss(prediction, y)

    def compute_crps(self, y, mean_pred, std_pred, max_crps_samples=100):

        if mean_pred.shape[-1] != 1:
            raise NotImplementedError("CRPS only implemented for output_dim=1.")

        mean_pred = mean_pred.squeeze(-1)  # (S, B)
        std_pred = std_pred.squeeze(-1)  # (S, B)

        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)  # (B,)

        S = mean_pred.shape[0]

        # Subsample if S is large to keep the S×S pairwise term tractable
        if max_crps_samples < S:
            idx = torch.randperm(S, device=mean_pred.device)[:max_crps_samples]
            mean_sub = mean_pred[idx]  # (S', B)
            std_sub = std_pred[idx]
        else:
            mean_sub = mean_pred
            std_sub = std_pred

        var_sub = std_sub**2  # (S', B)

        # Data term: E_s[A(y - mu_s, sigma_s^2)]  ->  mean over S' -> (B,)
        dev_mean = y.unsqueeze(0) - mean_sub  # (S', B)
        data_term = _crps_A(dev_mean, var_sub).mean(0)  # (B,)

        # Fixed term: 0.5 * E_{s,s'}[A(mu_s - mu_s', sigma_s^2 + sigma_s'^2)]
        means_diff = mean_sub.unsqueeze(1) - mean_sub.unsqueeze(0)  # (S', S', B)
        vars_sum = var_sub.unsqueeze(1) + var_sub.unsqueeze(0)  # (S', S', B)
        fixed_term = 0.5 * _crps_A(means_diff, vars_sum).mean(dim=(0, 1))  # (B,)

        return (data_term - fixed_term).mean()

    def _compute_pit(self, y, mean_pred, std_pred):
        eps = 1e-12
        z = (y.unsqueeze(0) - mean_pred) / std_pred.clamp_min(eps)
        norm = torch.distributions.Normal(
            torch.zeros(1, device=self.device, dtype=z.dtype),
            torch.ones(1, device=self.device, dtype=z.dtype),
        )
        cdf_vals = norm.cdf(z)
        pit = cdf_vals.mean(dim=0)
        return pit.squeeze(-1)

    def compute_cqm(self, num_alpha=200):
        """Centered Quantile Metric.

        Using PIT values u_i, the coverage at level alpha is
            gamma(alpha) = (1/N) sum_i  1(|2*u_i - 1| <= alpha)
        and CQM = integral_0^1 |gamma(alpha) - alpha| d_alpha.
        """
        if not self._pit_values:
            return 0.0
        pit = torch.cat(self._pit_values, dim=0)  # (N,)
        centered = torch.abs(2.0 * pit - 1.0)  # (N,)
        alphas = torch.linspace(0, 1, num_alpha, device=pit.device, dtype=pit.dtype)
        # gamma(alpha) = fraction of |2u-1| <= alpha
        gamma = (centered.unsqueeze(-1) <= alphas.unsqueeze(0)).float().mean(dim=0)
        # Trapezoidal integration of |gamma - alpha|
        integrand = torch.abs(gamma - alphas)
        cqm = torch.trapezoid(integrand, alphas)
        return float(cqm.cpu().numpy())

    def update_density(self, y, nll_batch, samples, light=True):
        """Metrics update for sample-based density estimators.

        Parameters
        ----------
        y : (B, D) targets, denormalized.
        nll_batch : scalar tensor
            Mean NLL on this batch, computed exactly from the model's density
            (already on the denormalized y scale).
        samples : (S, B, D) samples drawn from the predictive p(y|x).
        light : bool
            When False, also accumulate CRPS/PIT.
        """
        batch_size = y.shape[0]
        scale = 1 if self.num_data == -1 else batch_size / self.num_data

        # Loss mirrors the convention used by :meth:`update` (pass-through).
        self.loss += scale * nll_batch.detach()
        self.nll += scale * nll_batch.detach()

        mean_pred = samples.mean(0)  # (B, D)
        self.mse += scale * self.compute_mse(y, mean_pred)

        if not light:
            self.crps += scale * self._compute_crps_samples(y, samples)
            self._pit_values.append(self._compute_pit_samples(y, samples))

    @staticmethod
    def _compute_crps_samples(y, samples, max_crps_samples=100):
        """Sample-based CRPS: mean_s|y - Y_s| - 0.5 * mean_{s,s'} |Y_s - Y_s'|."""
        if samples.shape[-1] != 1:
            raise NotImplementedError("CRPS only implemented for output_dim=1.")
        samples = samples.squeeze(-1)  # (S, B)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)  # (B,)

        S = samples.shape[0]
        if max_crps_samples < S:
            idx = torch.randperm(S, device=samples.device)[:max_crps_samples]
            samples = samples[idx]
            S = max_crps_samples

        data_term = (samples - y.unsqueeze(0)).abs().mean(0)  # (B,)
        diff = samples.unsqueeze(1) - samples.unsqueeze(0)  # (S, S, B)
        pair_term = 0.5 * diff.abs().mean(dim=(0, 1))  # (B,)
        return (data_term - pair_term).mean()

    @staticmethod
    def _compute_pit_samples(y, samples):
        """PIT via empirical CDF: fraction of samples <= y."""
        if samples.shape[-1] == 1:
            samples = samples.squeeze(-1)  # (S, B)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)  # (B,)
        return (samples <= y.unsqueeze(0)).float().mean(0)

    def get_pit_values(self):
        """Return all PIT values as a numpy array."""
        if not self._pit_values:
            return np.array([])
        return torch.cat(self._pit_values, dim=0).detach().cpu().numpy()

    def get_calibration_curve(self, num_alpha=200):
        """Return (alphas, gamma) arrays for plotting the calibration curve."""
        if not self._pit_values:
            return np.array([]), np.array([])
        pit = torch.cat(self._pit_values, dim=0)
        centered = torch.abs(2.0 * pit - 1.0)
        alphas = torch.linspace(0, 1, num_alpha, device=pit.device, dtype=pit.dtype)
        gamma = (centered.unsqueeze(-1) <= alphas.unsqueeze(0)).float().mean(dim=0)
        return alphas.cpu().numpy(), gamma.cpu().numpy()

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "RMSE": float(np.sqrt(self.mse.detach().cpu().numpy())),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "CRPS": float(self.crps.detach().cpu().numpy()),
            "CQM": self.compute_cqm(),
        }


class TrajectoryMetrics:
    """Trajectory prediction metrics: ADE, FDE, and best-of-K variants.

    All inputs should be in **denormalized** (original) coordinates.

    Parameters
    ----------
    t_pred : int
        Number of prediction timesteps.
    K : int
        K for minADE_K / minFDE_K (best-of-K evaluation).
    """

    def __init__(self, t_pred=12, K=20):
        self.t_pred = t_pred
        self.K = K
        self.reset()

    def reset(self):
        self._ade_sum = 0.0
        self._fde_sum = 0.0
        self._ade_var_sum = 0.0
        self._fde_var_sum = 0.0
        self._min_ade_sum = 0.0
        self._min_fde_sum = 0.0
        self._n = 0

    def update(self, pred_samples, gt):
        """
        Parameters
        ----------
        pred_samples : (S, N, T_pred*2) tensor — S sampled trajectories
        gt : (N, T_pred*2) tensor — ground truth trajectory
        """
        S, N, _ = pred_samples.shape
        pred = pred_samples.reshape(S, N, self.t_pred, 2)
        gt_r = gt.reshape(N, self.t_pred, 2)

        # Per-sample, per-timestep L2 displacement
        disp = torch.norm(pred - gt_r.unsqueeze(0), dim=-1)  # (S, N, T)

        ade_per_sample = disp.mean(dim=-1)  # (S, N)
        fde_per_sample = disp[:, :, -1]  # (S, N)

        # Mean over samples (marginal ADE/FDE)
        self._ade_sum += ade_per_sample.mean(0).sum().item()
        self._fde_sum += fde_per_sample.mean(0).sum().item()

        # Variance over samples (per data point, then averaged)
        self._ade_var_sum += ade_per_sample.var(0).sum().item()
        self._fde_var_sum += fde_per_sample.var(0).sum().item()

        # Best-of-K
        K = min(self.K, S)
        self._min_ade_sum += ade_per_sample[:K].min(0).values.sum().item()
        self._min_fde_sum += fde_per_sample[:K].min(0).values.sum().item()
        self._n += N

    def get_dict(self):
        n = max(self._n, 1)
        return {
            "ADE": self._ade_sum / n,
            "FDE": self._fde_sum / n,
            "ADE_var": self._ade_var_sum / n,
            "FDE_var": self._fde_var_sum / n,
            f"minADE_{self.K}": self._min_ade_sum / n,
            f"minFDE_{self.K}": self._min_fde_sum / n,
        }


class ECE(torch.nn.Module):
    """Expected Calibration Error with binning."""

    def __init__(self, n_bins=15):
        super().__init__()
        self.confidences = []
        self.predictions = []
        self.labels = []
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def reset(self):
        self.confidences = []
        self.predictions = []
        self.labels = []

    def update(self, labels, F):
        probs = F.softmax(-1)
        conf, pred = torch.max(probs, -1)
        self.confidences.append(conf)
        self.predictions.append(pred)
        self.labels.append(labels.squeeze(-1))

    def compute(self):
        if not self.confidences:
            return 0.0
        predictions = torch.cat(self.predictions, -1)
        labels = torch.cat(self.labels, -1)
        confidences = torch.cat(self.confidences, -1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece.item())


class MetricsClassification:
    """Classification metrics for sample-based predictions (FTIP/VIP).

    Expects posterior sample logits of shape (S, N, K) where S is the number
    of samples, N is batch size, and K is the number of classes.  Follows the
    BayesiPy SoftmaxClassificationSamples pattern: softmax -> average probs ->
    log -> cross-entropy / accuracy / Brier / ECE.
    """

    def __init__(self, num_data=-1, device=None):
        self.num_data = num_data
        self.device = device
        self.reset()

    def reset(self):
        self.nll = torch.tensor(0.0, device=self.device)
        self.acc = torch.tensor(0.0, device=self.device)
        self.brier = torch.tensor(0.0, device=self.device)
        self.ece = ECE()
        self._n_seen = 0

    def update(self, y, loss, mean_pred, std_pred=None, light=True):
        """Update metrics with a batch of posterior sample predictions.

        Parameters
        ----------
        y : (N,) or (N, 1) class indices
        loss : scalar loss tensor (stored but not used for metrics)
        mean_pred : (S, N, K) posterior sample logits/values
        std_pred : ignored
        light : ignored (all metrics are cheap for classification)
        """
        if mean_pred.dim() == 2:
            mean_pred = mean_pred.unsqueeze(0)

        N = y.shape[0]
        self._n_seen += N

        # Softmax -> average probs -> log = averaged predictive logits
        prob_samples = mean_pred.softmax(-1)  # (S, N, K)
        logits = prob_samples.mean(0).clamp(min=1e-8).log()  # (N, K)

        y_long = y.long().squeeze(-1)  # (N,)

        self.acc += (torch.argmax(logits, -1) == y_long).float().sum()
        self.nll += torch.nn.functional.cross_entropy(logits, y_long, reduction="sum")
        self.brier += self._compute_brier(y_long, logits)
        self.ece.update(y, logits)

    def _compute_brier(self, y, logits):
        probs = logits.softmax(-1)
        oh = torch.nn.functional.one_hot(y, num_classes=probs.shape[-1]).to(probs.dtype)
        return ((probs - oh) ** 2).sum()

    def get_dict(self):
        n = max(self._n_seen, 1)
        return {
            "NLL": float(self.nll.item()) / n,
            "Error": 1.0 - float(self.acc.item()) / n,
            "ECE": self.ece.compute(),
            "Brier": float(self.brier.item()) / n,
        }


class MetricsBinary:
    """Binary classification metrics for sample-based predictions.

    Expects posterior **probability** samples of shape (S, N, 1), each entry
    p(y=1 | x). For VIP this is one probit-marginalized probability (S=1);
    for sample-based models (FTIP/MFVI/FBNN/TFSVI) it is S Monte Carlo draws.
    The predictive mean p* = mean over S is used for Error, NLL (Bernoulli),
    Brier, and top-label ECE. AUC uses p* and is updated lazily.
    """

    def __init__(self, num_data=-1, device=None, n_bins=15):
        self.num_data = num_data
        self.device = device
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self.nll = torch.tensor(0.0, device=self.device)
        self.acc = torch.tensor(0.0, device=self.device)
        self.brier = torch.tensor(0.0, device=self.device)
        self._p_buf = []
        self._y_buf = []
        self._n_seen = 0

    def update(self, y, loss, mean_pred, std_pred=None, light=True):
        """Update metrics with a batch of posterior probability samples.

        Parameters
        ----------
        y : (N,) or (N, 1) labels in {0, 1}
        loss : scalar loss tensor (stored but unused)
        mean_pred : (S, N, 1) probability samples in [0, 1], or (N, 1) /
            (1, N, 1) for a point predictive (e.g. VIP).
        std_pred, light : ignored (kept for API parity with regression).
        """
        if mean_pred.dim() == 2:
            mean_pred = mean_pred.unsqueeze(0)

        # Average over posterior samples -> predictive probability of y=1.
        p = mean_pred.mean(0).reshape(-1).clamp(1e-7, 1 - 1e-7)  # (N,)
        y_flat = y.reshape(-1).to(dtype=p.dtype)
        N = y_flat.shape[0]
        self._n_seen += N

        pred = (p >= 0.5).to(p.dtype)
        self.acc += (pred == y_flat).sum()
        self.nll += -(y_flat * torch.log(p) + (1 - y_flat) * torch.log(1 - p)).sum()
        self.brier += ((p - y_flat) ** 2).sum()
        self._p_buf.append(p.detach())
        self._y_buf.append(y_flat.detach())

    def _compute_ece(self, p, y):
        """Top-label ECE: confidence = max(p, 1-p), accuracy = pred == y."""
        confidences = torch.maximum(p, 1 - p)
        predictions = (p >= 0.5).to(p.dtype)
        accuracies = (predictions == y).to(p.dtype)
        bin_boundaries = torch.linspace(0.0, 1.0, self.n_bins + 1, device=p.device)
        ece = torch.tensor(0.0, device=p.device)
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > lo) & (confidences <= hi)
            prop = in_bin.to(p.dtype).mean()
            if prop > 0:
                acc_bin = accuracies[in_bin].mean()
                conf_bin = confidences[in_bin].mean()
                ece = ece + (acc_bin - conf_bin).abs() * prop
        return float(ece.item())

    def _compute_auc(self, p, y):
        if y.unique().numel() < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y.cpu().numpy(), p.cpu().numpy()))
        except ValueError:
            return float("nan")

    def get_dict(self):
        n = max(self._n_seen, 1)
        if self._p_buf:
            p_all = torch.cat(self._p_buf, dim=0)
            y_all = torch.cat(self._y_buf, dim=0)
            ece = self._compute_ece(p_all, y_all)
            auc = self._compute_auc(p_all, y_all)
        else:
            ece = 0.0
            auc = float("nan")
        return {
            "NLL": float(self.nll.item()) / n,
            "Error": 1.0 - float(self.acc.item()) / n,
            "ECE": ece,
            "Brier": float(self.brier.item()) / n,
            "AUC": auc,
        }
