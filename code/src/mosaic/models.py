"""Branch classifiers with calibrated probabilities and explicit uncertainty.

Each L3 branch (visual, audio, audiovisual) produces not just a score but a
:class:`BranchPrediction` carrying four things fusion needs:

``p_fake``    a *calibrated* probability, so that 0.7 means 0.7 across branches;
``llr``       the calibrated log-odds, the quantity that composes additively in fusion;
``vacuity``   how much the branch does not know, in [0, 1];
``ood_score`` how far this input sits from anything the branch was fitted on.

Why uncertainty is modelled the way it is
-----------------------------------------
Two different things can make a branch unreliable, and conflating them would be wrong.

*Disagreement* — the branch is being asked about a genuinely borderline case. Captured by
fitting a bootstrap ensemble and reading the spread of its members: wide spread means the
decision boundary is not well determined in this region. Converted to a subjective-logic
vacuity by moment-matching a Beta distribution to the ensemble's mean and variance.

*Unfamiliarity* — the input is unlike anything in the training distribution, in which case
the ensemble may agree confidently and still be meaningless. Captured by a Mahalanobis
distance to the training distribution, calibrated against the training set's own distance
percentiles. A high-distance input has its vacuity raised regardless of ensemble agreement.

The distinction matters for the five-way verdict: an out-of-distribution clip should
produce UNKNOWN, not a confident wrong answer, and only an OOD term can produce that.

Models are deliberately small — L2-regularised logistic regression ensembles, plus an
optional ~25k-parameter CNN for audio. On a few hundred clips, anything larger would fit
noise, and the whole pipeline has to run on a laptop.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------------------
# Prediction container
# --------------------------------------------------------------------------------------


@dataclass
class FeatureContribution:
    name: str
    value: float           # raw feature value
    z: float               # standardised value
    weight: float          # mean model coefficient
    contribution: float    # z * weight, in log-odds toward "fake"


@dataclass
class BranchPrediction:
    """One branch's evidence about one clip."""

    p_fake: float
    llr: float
    vacuity: float
    ood_score: float = 0.0
    ood_flag: bool = False
    available: bool = True
    reason_unavailable: str | None = None
    ensemble_std: float = 0.0
    contributions: list[FeatureContribution] = field(default_factory=list)

    @staticmethod
    def unavailable(reason: str) -> "BranchPrediction":
        """A branch that could not run contributes *no evidence* and total vacuity.

        It must never be recorded as p_fake=0 — "we could not check" is not "it is real".
        """
        return BranchPrediction(
            p_fake=0.5, llr=0.0, vacuity=1.0, available=False, reason_unavailable=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_fake": round(self.p_fake, 6),
            "llr": round(self.llr, 6),
            "vacuity": round(self.vacuity, 6),
            "ood_score": round(self.ood_score, 4),
            "ood_flag": self.ood_flag,
            "available": self.available,
            "reason_unavailable": self.reason_unavailable,
            "ensemble_std": round(self.ensemble_std, 6),
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _logit(p: np.ndarray | float, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# --------------------------------------------------------------------------------------
# Branch model
# --------------------------------------------------------------------------------------


class BranchModel:
    """Bootstrap ensemble of L2-regularised logistic regressions, with calibration + OOD."""

    def __init__(self, name: str, feature_names: list[str], *, n_bootstrap: int = 15,
                 C: float = 0.25, seed: int = 1337, max_vacuity_evidence: float = 60.0):
        self.name = name
        self.feature_names = list(feature_names)
        self.n_bootstrap = n_bootstrap
        self.C = C
        self.seed = seed
        self.max_evidence = max_vacuity_evidence

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.models_: list[LogisticRegression] = []
        self.platt_: LogisticRegression | None = None
        self.ood_mean_: np.ndarray | None = None
        self.ood_prec_: np.ndarray | None = None
        self.ood_ref_: np.ndarray | None = None   # training distance percentiles
        self.fitted_ = False
        self.train_stats_: dict[str, dict[str, float]] = {}

    # -- fitting ------------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "BranchModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"{self.name}: bad shapes X={X.shape} y={y.shape}")
        if len(np.unique(y)) < 2:
            raise ValueError(f"{self.name}: training labels contain a single class")

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Constant features carry no information; a unit std keeps them at z=0 rather
        # than exploding to +/-inf.
        self.std_[self.std_ < 1e-9] = 1.0
        Z = (X - self.mean_) / self.std_

        rng = np.random.default_rng(self.seed)
        self.models_ = []
        n = Z.shape[0]
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:      # degenerate resample
                idx = np.arange(n)
            # L2 is the lbfgs default; passing penalty="l2" explicitly is deprecated
            # from scikit-learn 1.8 onward.
            clf = LogisticRegression(
                C=self.C, solver="lbfgs", max_iter=2000,
                class_weight="balanced", random_state=self.seed + b,
            )
            clf.fit(Z[idx], y[idx])
            self.models_.append(clf)

        # Platt calibration on held-out data when available; otherwise on train, which is
        # recorded as a limitation rather than presented as proper calibration.
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            cal_X, cal_y = np.asarray(X_val, dtype=np.float64), np.asarray(y_val).astype(int)
            self.calibrated_on_ = "validation"
        else:
            cal_X, cal_y = X, y
            self.calibrated_on_ = "train (no validation split supplied)"
        p_raw = self._ensemble_mean(cal_X)
        self.platt_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.platt_.fit(_logit(p_raw).reshape(-1, 1), cal_y)

        # Mahalanobis reference on the training distribution.
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov)
        # Ledoit-Wolf-style shrinkage toward a diagonal: with a few hundred samples and
        # dozens of features the raw covariance is near-singular.
        shrink = 0.20
        cov = (1 - shrink) * cov + shrink * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        self.ood_mean_ = Z.mean(axis=0)
        self.ood_prec_ = np.linalg.pinv(cov)
        d_train = self._mahalanobis(Z)
        self.ood_ref_ = np.percentile(d_train, np.arange(0, 101))

        self.train_stats_ = {
            name: {
                "real_mean": float(X[y == 0, i].mean()) if (y == 0).any() else 0.0,
                "real_std": float(X[y == 0, i].std()) if (y == 0).any() else 0.0,
                "fake_mean": float(X[y == 1, i].mean()) if (y == 1).any() else 0.0,
            }
            for i, name in enumerate(self.feature_names)
        }
        self.fitted_ = True
        return self

    # -- inference ----------------------------------------------------------------------

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def _ensemble_probs(self, X: np.ndarray) -> np.ndarray:
        Z = self._standardise(X)
        return np.stack([m.predict_proba(Z)[:, 1] for m in self.models_])  # (B, N)

    def _ensemble_mean(self, X: np.ndarray) -> np.ndarray:
        return self._ensemble_probs(X).mean(axis=0)

    def _mahalanobis(self, Z: np.ndarray) -> np.ndarray:
        d = Z - self.ood_mean_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.ood_prec_, d), 0.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(fake) for a batch."""
        self._check_fitted()
        p_raw = self._ensemble_mean(X)
        return self.platt_.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]

    def predict(self, x: np.ndarray, *, ood_percentile: float = 99.0,
                extra_probs: list[float] | None = None) -> BranchPrediction:
        """Full prediction for a single feature vector.

        ``extra_probs`` lets an additional model (e.g. the audio CNN) join the ensemble
        for the purpose of both the mean probability and the disagreement estimate.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            return BranchPrediction.unavailable(
                f"feature vector has {x.shape[1]} values, model expects {len(self.feature_names)}"
            )
        if not np.isfinite(x).all():
            return BranchPrediction.unavailable("feature vector contains non-finite values")

        member_probs = list(self._ensemble_probs(x)[:, 0])
        if extra_probs:
            member_probs.extend(float(p) for p in extra_probs)
        members = np.asarray(member_probs, dtype=np.float64)

        p_raw = float(members.mean())
        p_cal = float(self.platt_.predict_proba(np.array([[_logit(p_raw)]]))[0, 1])
        spread = float(members.std())

        # Ensemble disagreement -> Beta evidence -> subjective-logic vacuity.
        var = max(spread**2, 1e-9)
        bound = p_raw * (1 - p_raw)
        if var >= bound - 1e-9:
            evidence = 0.0            # spread as wide as a Bernoulli: no information
        else:
            evidence = max(bound / var - 1.0, 0.0)
        evidence = min(evidence, self.max_evidence)
        vacuity = 2.0 / (evidence + 2.0)

        # Unfamiliarity: distance relative to the training distribution's own percentiles.
        Z = self._standardise(x)
        dist = float(self._mahalanobis(Z)[0])
        ref = self.ood_ref_
        pct = float(np.searchsorted(ref, dist) )
        pct = float(np.clip(pct, 0, 100))
        ood_flag = pct >= ood_percentile
        if ood_flag:
            # Beyond the training envelope the calibrated probability is not trustworthy;
            # push vacuity up so fusion discounts this branch instead of believing it.
            overshoot = (pct - ood_percentile) / max(100.0 - ood_percentile, 1e-6)
            vacuity = float(np.clip(max(vacuity, 0.5 + 0.5 * overshoot), 0.0, 1.0))

        contributions = self._contributions(x[0], Z[0])
        return BranchPrediction(
            p_fake=p_cal,
            llr=float(_logit(p_cal)),
            vacuity=float(np.clip(vacuity, 0.0, 1.0)),
            ood_score=pct,
            ood_flag=ood_flag,
            available=True,
            ensemble_std=spread,
            contributions=contributions,
        )

    def _contributions(self, raw: np.ndarray, z: np.ndarray) -> list[FeatureContribution]:
        coefs = np.mean([m.coef_[0] for m in self.models_], axis=0)
        out = [
            FeatureContribution(
                name=self.feature_names[i], value=float(raw[i]), z=float(z[i]),
                weight=float(coefs[i]), contribution=float(coefs[i] * z[i]),
            )
            for i in range(len(self.feature_names))
        ]
        out.sort(key=lambda c: abs(c.contribution), reverse=True)
        return out

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name}: model is not fitted")

    # -- persistence --------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "BranchModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------------------
# Small audio CNN (optional ensemble member)
# --------------------------------------------------------------------------------------


class AudioCNN:
    """~25k-parameter CNN over log-mel patches.

    Present as a second, architecturally different ensemble member for the audio branch:
    it sees the spectrogram directly rather than the hand-designed features, so where it
    disagrees with the feature model the disagreement is informative rather than redundant.
    Sized to train in seconds on CPU or MPS; it is not a pretrained spoof embedding and is
    not presented as one.
    """

    def __init__(self, n_mels: int = 64, seed: int = 1337, device: str | None = None):
        self.n_mels = n_mels
        self.seed = seed
        self.device_str = device
        self.net = None
        self.fitted_ = False
        self.norm_: tuple[float, float] = (0.0, 1.0)

    def _build(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(24, 1),
        )

    def _device(self):
        import torch

        if self.device_str:
            return torch.device(self.device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, specs: list[np.ndarray], y: np.ndarray, *, epochs: int = 40,
            lr: float = 3e-3, batch_size: int = 32) -> "AudioCNN":
        import torch
        import torch.nn as nn

        X = self._stack(specs)
        self.norm_ = (float(X.mean()), float(X.std()) or 1.0)
        X = (X - self.norm_[0]) / self.norm_[1]
        y = np.asarray(y).astype(np.float32)

        dev = self._device()
        self.net = self._build().to(dev)
        xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(dev)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(dev)

        pos = float(y.sum())
        neg = float(len(y) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=dev)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)

        n = xt.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        self.fitted_ = True
        return self

    def predict_proba(self, specs: list[np.ndarray]) -> np.ndarray:
        import torch

        if not self.fitted_:
            raise RuntimeError("AudioCNN is not fitted")
        X = (self._stack(specs) - self.norm_[0]) / self.norm_[1]
        dev = self._device()
        # A model restored from disk lands on CPU (map_location="cpu"), while inputs are
        # placed on the accelerator. Move the network every call so load-then-predict works
        # regardless of where the weights currently live.
        self.net.to(dev)
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(dev))
            return torch.sigmoid(logits).squeeze(1).cpu().numpy()

    def _stack(self, specs: list[np.ndarray], n_time: int = 128) -> np.ndarray:
        """Pad/crop each spectrogram to a fixed (n_mels, n_time) patch."""
        out = np.zeros((len(specs), self.n_mels, n_time), dtype=np.float32)
        for i, s in enumerate(specs):
            s = np.asarray(s, dtype=np.float32)
            m = min(self.n_mels, s.shape[0])
            t = min(n_time, s.shape[1])
            out[i, :m, :t] = s[:m, :t]
        return out

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict() if self.net else None,
                    "norm": self.norm_, "n_mels": self.n_mels, "seed": self.seed}, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "AudioCNN":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        cnn = AudioCNN(n_mels=blob["n_mels"], seed=blob["seed"])
        cnn.net = cnn._build()
        if blob["state"] is not None:
            cnn.net.load_state_dict(blob["state"])
            cnn.net.eval()
            cnn.fitted_ = True
        cnn.norm_ = tuple(blob["norm"])
        return cnn
