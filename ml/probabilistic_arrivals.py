"""
ml/probabilistic_arrivals.py
Probabilistic count forecasting engine for EV charging station arrivals.
Implements:
1. Poisson GLM (baseline)
2. Negative Binomial NB2 (overdispersion)
3. Hurdle Negative Binomial (two-stage zero-inflated / hurdle model)
4. XGBoost-as-Poisson baseline (existing point forecast -> Poisson PMF)
5. Empirical station-hour-of-week frequency baseline

Provides full PMF generation, probability of zero arrivals, tail probabilities,
and Monte Carlo / mixture rate sampling for Erlang-B queueing propagation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.special as sp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler


def compute_pmf_metrics(pmf: np.ndarray, max_k: int = 12) -> Dict[str, Union[float, int, List[float]]]:
    """
    Computes statistical and decision-level summary metrics from a discrete PMF vector over k = 0..max_k.
    Guarantees sum(pmf) == 1.0 via normalization.
    """
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    k_vals = np.arange(len(pmf), dtype=np.float64)

    # Normalize to ensure exact probability axioms
    s = pmf.sum()
    if s > 0:
        pmf = pmf / s
    else:
        pmf = np.zeros_like(pmf)
        pmf[0] = 1.0

    mean_val = float(np.sum(k_vals * pmf))
    var_val = float(np.sum(((k_vals - mean_val) ** 2) * pmf))

    p_zero = float(pmf[0])
    p_ge_1 = float(1.0 - p_zero)
    p_ge_2 = float(np.sum(pmf[2:])) if len(pmf) > 2 else 0.0
    p_ge_3 = float(np.sum(pmf[3:])) if len(pmf) > 3 else 0.0

    cdf = np.cumsum(pmf)
    q05 = int(k_vals[np.searchsorted(cdf, 0.05)])
    q50 = int(k_vals[np.searchsorted(cdf, 0.50)])
    q95 = int(k_vals[np.searchsorted(cdf, 0.95)])

    return {
        "mean": round(mean_val, 4),
        "variance": round(var_val, 4),
        "p_zero": round(p_zero, 4),
        "p_ge_1": round(p_ge_1, 4),
        "p_ge_2": round(p_ge_2, 4),
        "p_ge_3": round(p_ge_3, 4),
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "pmf": [round(float(p), 5) for p in pmf],
    }


def poisson_pmf_batch(mu: np.ndarray, max_k: int = 12) -> np.ndarray:
    """Vectorized Poisson PMF over k=0..max_k for rates mu."""
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-7, 50.0)[:, None]
    k = np.arange(max_k + 1, dtype=np.float64)[None, :]
    log_pmf = k * np.log(mu) - mu - sp.gammaln(k + 1.0)
    pmf = np.exp(log_pmf)
    # Re-normalize row-wise to absorb negligible tail mass
    row_sums = pmf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pmf / row_sums


def negative_binomial_pmf_batch(mu: np.ndarray, alpha: float, max_k: int = 12) -> np.ndarray:
    """
    Vectorized Negative Binomial (NB2) PMF over k=0..max_k.
    Parameterization: Var(Y) = mu + alpha * mu^2.
    r = 1 / alpha, p = 1 / (1 + alpha * mu).
    """
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-7, 50.0)[:, None]
    alpha = max(float(alpha), 1e-5)
    r = 1.0 / alpha
    k = np.arange(max_k + 1, dtype=np.float64)[None, :]

    p = 1.0 / (1.0 + alpha * mu)
    log_pmf = (
        sp.gammaln(k + r)
        - sp.gammaln(r)
        - sp.gammaln(k + 1.0)
        + r * np.log(np.clip(p, 1e-12, 1.0))
        + k * np.log(np.clip(1.0 - p, 1e-12, 1.0))
    )
    pmf = np.exp(log_pmf)
    row_sums = pmf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pmf / row_sums


class PoissonGLMModel:
    """
    Poisson GLM baseline predicting count rate mu = exp(X * beta).
    """

    def __init__(self, alpha: float = 1e-4, max_iter: int = 300):
        self.alpha = alpha
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PoissonGLMModel":
        y = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_rate(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        rates = self.model.predict(X_scaled)
        return np.clip(rates, 1e-6, None)

    def predict_pmf(self, X: np.ndarray, max_k: int = 12) -> np.ndarray:
        rates = self.predict_rate(X)
        return poisson_pmf_batch(rates, max_k=max_k)


class NegativeBinomialModel:
    """
    Negative Binomial (NB2) model with ML dispersion parameter alpha.
    Var(Y) = mu + alpha * mu^2.
    """

    def __init__(self, alpha_reg: float = 1e-4, max_iter: int = 300):
        self.alpha_reg = alpha_reg
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.mean_model = PoissonRegressor(alpha=alpha_reg, max_iter=max_iter)
        self.dispersion_alpha: float = 0.1
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NegativeBinomialModel":
        y = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
        X_scaled = self.scaler.fit_transform(X)
        self.mean_model.fit(X_scaled, y)
        mu = np.clip(self.mean_model.predict(X_scaled), 1e-6, 50.0)

        # Estimate dispersion alpha by maximum profile log-likelihood
        # Sample subset if dataset is very large (> 100k) for swift convergence
        if len(y) > 100000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y), size=100000, replace=False)
            y_sub, mu_sub = y[idx], mu[idx]
        else:
            y_sub, mu_sub = y, mu

        def neg_log_lik(log_a):
            a = np.exp(log_a)
            r = 1.0 / a
            p = 1.0 / (1.0 + a * mu_sub)
            ll = (
                sp.gammaln(y_sub + r)
                - sp.gammaln(r)
                - sp.gammaln(y_sub + 1.0)
                + r * np.log(np.clip(p, 1e-12, 1.0))
                + y_sub * np.log(np.clip(1.0 - p, 1e-12, 1.0))
            )
            return -float(np.mean(ll))

        res = minimize_scalar(neg_log_lik, bounds=(-7.0, 3.0), method="bounded")
        self.dispersion_alpha = float(np.exp(res.x))
        self.is_fitted = True
        return self

    def predict_rate(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        rates = self.mean_model.predict(X_scaled)
        return np.clip(rates, 1e-6, None)

    def predict_pmf(self, X: np.ndarray, max_k: int = 12) -> np.ndarray:
        rates = self.predict_rate(X)
        return negative_binomial_pmf_batch(rates, self.dispersion_alpha, max_k=max_k)


class HurdleNegativeBinomialModel:
    """
    Two-stage Hurdle Negative Binomial model:
    Stage 1: Logistic Regression modeling P(Y = 0) vs P(Y >= 1).
    Stage 2: Truncated Negative Binomial modeling count distribution given Y >= 1.
    Guarantees exact handling of zero-inflation and overdispersed positive arrivals.
    """

    def __init__(self, C_zero: float = 1.0, alpha_reg: float = 1e-4, max_iter: int = 300):
        self.C_zero = C_zero
        self.alpha_reg = alpha_reg
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.zero_hurdle_model = LogisticRegression(C=C_zero, max_iter=max_iter, solver="lbfgs")
        self.count_model = PoissonRegressor(alpha=alpha_reg, max_iter=max_iter)
        self.dispersion_alpha: float = 0.1
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HurdleNegativeBinomialModel":
        y = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
        X_scaled = self.scaler.fit_transform(X)

        # Stage 1: Zero hurdle (1 if Y == 0, 0 if Y >= 1)
        is_zero = (y == 0).astype(np.int32)
        self.zero_hurdle_model.fit(X_scaled, is_zero)

        # Stage 2: Positive count model on Y > 0
        pos_mask = y > 0
        if pos_mask.sum() > 0:
            X_pos = X_scaled[pos_mask]
            y_pos = y[pos_mask]
            self.count_model.fit(X_pos, y_pos)
            mu_pos = np.clip(self.count_model.predict(X_pos), 1e-6, 50.0)

            # Fit dispersion on positive counts
            if len(y_pos) > 50000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(y_pos), size=50000, replace=False)
                y_sub, mu_sub = y_pos[idx], mu_pos[idx]
            else:
                y_sub, mu_sub = y_pos, mu_pos

            def neg_log_lik(log_a):
                a = np.exp(log_a)
                r = 1.0 / a
                p = 1.0 / (1.0 + a * mu_sub)
                p_0 = (1.0 + a * mu_sub) ** (-r)
                # Truncated NB log likelihood: log(P(Y=y)) - log(1 - P(Y=0))
                ll_untr = (
                    sp.gammaln(y_sub + r)
                    - sp.gammaln(r)
                    - sp.gammaln(y_sub + 1.0)
                    + r * np.log(np.clip(p, 1e-12, 1.0))
                    + y_sub * np.log(np.clip(1.0 - p, 1e-12, 1.0))
                )
                ll_tr = ll_untr - np.log(np.clip(1.0 - p_0, 1e-12, 1.0))
                return -float(np.mean(ll_tr))

            res = minimize_scalar(neg_log_lik, bounds=(-7.0, 3.0), method="bounded")
            self.dispersion_alpha = float(np.exp(res.x))
        else:
            self.count_model.fit(X_scaled, y)
            self.dispersion_alpha = 0.1

        self.is_fitted = True
        return self

    def predict_p_zero(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        # Class 1 corresponds to is_zero == 1
        classes = self.zero_hurdle_model.classes_
        zero_col = 1 if classes[1] == 1 else 0
        probs = self.zero_hurdle_model.predict_proba(X_scaled)[:, zero_col]
        return np.clip(probs, 1e-5, 1.0 - 1e-5)

    def predict_rate(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        p_zero = self.predict_p_zero(X)
        mu_pos = np.clip(self.count_model.predict(X_scaled), 1e-6, 50.0)
        # Hurdle expected value = (1 - p_zero) * E[Y | Y > 0]
        # For truncated NB: E[Y | Y > 0] = mu / (1 - P(0))
        r = 1.0 / self.dispersion_alpha
        p_0_nb = (1.0 + self.dispersion_alpha * mu_pos) ** (-r)
        mu_trunc = mu_pos / np.clip(1.0 - p_0_nb, 1e-7, 1.0)
        return (1.0 - p_zero) * mu_trunc

    def predict_pmf(self, X: np.ndarray, max_k: int = 12) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        p_zero = self.predict_p_zero(X)
        mu_pos = np.clip(self.count_model.predict(X_scaled), 1e-6, 50.0)

        # Raw untruncated NB PMF
        nb_pmf = negative_binomial_pmf_batch(mu_pos, self.dispersion_alpha, max_k=max_k)
        p_0_nb = nb_pmf[:, 0:1]
        denom = np.clip(1.0 - p_0_nb, 1e-7, 1.0)

        # Truncated positive probabilities
        pos_pmf = nb_pmf[:, 1:] / denom
        final_pmf = np.empty((len(X), max_k + 1), dtype=np.float64)
        final_pmf[:, 0] = p_zero
        final_pmf[:, 1:] = (1.0 - p_zero[:, None]) * pos_pmf

        # Normalize
        row_sums = final_pmf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return final_pmf / row_sums


class XGBoostPoissonBaseline:
    """
    Evaluates existing Phase 5 XGBoost point forecasts as a Poisson arrival distribution.
    lambda = max(1e-6, y_pred_xgb).
    """

    def __init__(self, base_predictor):
        self.base_predictor = base_predictor

    def predict_pmf_from_preds(self, y_pred_point: np.ndarray, max_k: int = 12) -> np.ndarray:
        rates = np.clip(np.asarray(y_pred_point, dtype=np.float64), 1e-6, 50.0)
        return poisson_pmf_batch(rates, max_k=max_k)


class EmpiricalFrequencyBaseline:
    """
    Empirical station-hour-of-week arrival distribution.
    Computes empirical frequency of k=0..max_k across training data.
    """

    def __init__(self, max_k: int = 12):
        self.max_k = max_k
        self.global_pmf_: np.ndarray = np.zeros(max_k + 1, dtype=np.float64)
        self.st_hw_pmf_: Dict[Tuple[str, int, int], np.ndarray] = {}
        self.hw_pmf_: Dict[Tuple[int, int], np.ndarray] = {}

    def fit(self, df_train) -> "EmpiricalFrequencyBaseline":
        y = np.clip(df_train["sessions_count"].to_numpy(dtype=np.int32), 0, self.max_k)
        # Global PMF
        counts = np.bincount(y, minlength=self.max_k + 1)[: self.max_k + 1]
        self.global_pmf_ = counts.astype(np.float64) / max(len(y), 1)

        # Hour x Weekend PMF
        for (h, wknd), group in df_train.groupby(["hour", "is_weekend"]):
            g_y = np.clip(group["sessions_count"].to_numpy(dtype=np.int32), 0, self.max_k)
            c = np.bincount(g_y, minlength=self.max_k + 1)[: self.max_k + 1]
            self.hw_pmf_[(int(h), int(wknd))] = c.astype(np.float64) / max(len(g_y), 1)

        # Station x Hour x Weekend PMF
        for (st, h, wknd), group in df_train.groupby(["station_id", "hour", "is_weekend"]):
            g_y = np.clip(group["sessions_count"].to_numpy(dtype=np.int32), 0, self.max_k)
            c = np.bincount(g_y, minlength=self.max_k + 1)[: self.max_k + 1]
            self.st_hw_pmf_[(str(st), int(h), int(wknd))] = c.astype(np.float64) / max(len(g_y), 1)

        return self

    def predict_pmf(self, df) -> np.ndarray:
        n = len(df)
        pmf_out = np.zeros((n, self.max_k + 1), dtype=np.float64)
        sts = df["station_id"].values
        hrs = df["hour"].values
        wknds = df["is_weekend"].values

        for i in range(n):
            key_st = (str(sts[i]), int(hrs[i]), int(wknds[i]))
            key_hw = (int(hrs[i]), int(wknds[i]))
            if key_st in self.st_hw_pmf_:
                pmf_out[i] = self.st_hw_pmf_[key_st]
            elif key_hw in self.hw_pmf_:
                pmf_out[i] = self.hw_pmf_[key_hw]
            else:
                pmf_out[i] = self.global_pmf_
        return pmf_out


def sample_arrival_rates(
    pmf: np.ndarray,
    n_samples: int = 500,
    seed: int = 42,
    mean_val: Optional[float] = None,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Draws plausible arrival rate realizations (lambda) from predictive distribution:
    - If alpha is provided (Negative Binomial gamma-Poisson mixture), draws continuous
      lambda ~ Gamma(shape=1/alpha, scale=alpha * mean_val).
    - Otherwise, samples discrete arrival realizations k ~ Categorical(pmf) with additive jitter.
    """
    rng = np.random.RandomState(seed)
    if alpha is not None and mean_val is not None and alpha > 1e-4 and mean_val > 1e-6:
        shape = 1.0 / alpha
        scale = alpha * mean_val
        rates = rng.gamma(shape=shape, scale=scale, size=n_samples)
        return np.clip(rates, 0.0, 50.0)

    # Categorical sampling over PMF
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    pmf = pmf / max(pmf.sum(), 1e-12)
    k_vals = np.arange(len(pmf), dtype=np.float64)
    samples = rng.choice(k_vals, size=n_samples, p=pmf)
    return samples
