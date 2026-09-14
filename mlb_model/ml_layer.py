"""Residual ML layer on top of the structural model.

The GBM gets the structural logit as an offset (init_score/base_margin), so
with zero trees it reproduces the structural probability exactly - it can
only learn corrections. Validation is walk-forward by date, never random
k-fold (adjacent games share context; random folds leak and flatter ML).
Ship gate: if OOS log loss doesn't beat the structural baseline, predict()
just returns the structural number.

Backend: lightgbm > xgboost > sklearn, whichever imports.

    python -m mlb_model.ml_layer --train backtest_results/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger("mlb_model.ml_layer")


# --------------------------------------------------------------- backends
def _load_backend():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except ImportError:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb
    except ImportError:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa
        return "sklearn", None
    except ImportError:
        return None, None


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------ model
@dataclass
class ValidationReport:
    n_folds: int
    n_train_total: int
    n_test_total: int
    structural_log_loss: float
    ml_log_loss: float
    shipped: bool
    fold_details: list[dict]

    def summary(self) -> str:
        verdict = ("SHIPPED: ML beats structural out-of-sample"
                   if self.shipped else
                   "GATED OFF: ML does NOT beat structural out-of-sample - "
                   "predictions will use the structural probability")
        lines = [f"Walk-forward validation ({self.n_folds} folds, "
                 f"{self.n_test_total} OOS games):",
                 f"  structural log loss: {self.structural_log_loss:.4f}",
                 f"  ML-blended log loss: {self.ml_log_loss:.4f}",
                 f"  {verdict}"]
        for f in self.fold_details:
            lines.append(f"    fold {f['fold']}: train {f['n_train']} -> "
                         f"test {f['n_test']}  struct {f['ll_struct']:.4f} "
                         f"ml {f['ll_ml']:.4f}")
        return "\n".join(lines)


class ResidualModel:
    """GBM correction on top of the structural win probability."""

    def __init__(self):
        self.backend, self._lib = _load_backend()
        if self.backend is None:
            log.error("No GBM backend (pip install lightgbm) - ML layer "
                      "will pass structural probabilities through")
        self.model = None
        self.shipped = False
        self.report: ValidationReport | None = None
        self.features = list(config.ML_FEATURES)

    # ----------------------------------------------------------- internals
    def _fit_one(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray):
        if self.backend == "lightgbm":
            lgb = self._lib
            ds = lgb.Dataset(X, label=y, init_score=offset,
                             feature_name=self.features)
            params = dict(objective="binary", verbosity=-1,
                          learning_rate=config.ML_PARAMS["learning_rate"],
                          max_depth=config.ML_PARAMS["max_depth"],
                          num_leaves=config.ML_PARAMS.get("num_leaves", 31),
                          min_child_samples=config.ML_PARAMS["min_child_samples"],
                          bagging_fraction=config.ML_PARAMS["subsample"],
                          bagging_freq=1,
                          feature_fraction=config.ML_PARAMS["colsample_bytree"],
                          lambda_l1=config.ML_PARAMS.get("reg_alpha", 0.0),
                          lambda_l2=config.ML_PARAMS["reg_lambda"])
            return lgb.train(params, ds,
                             num_boost_round=config.ML_PARAMS["n_estimators"])
        if self.backend == "xgboost":
            xgb = self._lib
            dm = xgb.DMatrix(X, label=y, base_margin=offset,
                             feature_names=self.features)
            params = dict(objective="binary:logistic",
                          eta=config.ML_PARAMS["learning_rate"],
                          max_depth=config.ML_PARAMS["max_depth"],
                          subsample=config.ML_PARAMS["subsample"],
                          colsample_bytree=config.ML_PARAMS["colsample_bytree"],
                          reg_alpha=config.ML_PARAMS.get("reg_alpha", 0.0),
                          reg_lambda=config.ML_PARAMS["reg_lambda"])
            return xgb.train(params, dm,
                             num_boost_round=config.ML_PARAMS["n_estimators"])
        # sklearn fallback: no native offset -> feed structural logit as a
        # feature instead (weaker but honest; noted in the tag).
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=config.ML_PARAMS["n_estimators"],
            learning_rate=config.ML_PARAMS["learning_rate"],
            max_depth=config.ML_PARAMS["max_depth"],
            l2_regularization=config.ML_PARAMS["reg_lambda"])
        m.fit(np.column_stack([X, offset]), y)
        return m

    def _predict_one(self, model, X: np.ndarray,
                     offset: np.ndarray) -> np.ndarray:
        if self.backend == "lightgbm":
            raw = model.predict(X, raw_score=True)
            return _sigmoid(raw + offset)
        if self.backend == "xgboost":
            xgb = self._lib
            dm = xgb.DMatrix(X, base_margin=offset,
                             feature_names=self.features)
            return model.predict(dm)
        return model.predict_proba(np.column_stack([X, offset]))[:, 1]

    # ---------------------------------------------------------------- API
    def fit(self, rows: list[dict], n_folds: int = 4) -> ValidationReport:
        """rows: date-ordered dicts with ML_FEATURES + 'home_won'.
        Walk-forward: fold k trains on the first k blocks, tests on block
        k+1. Never shuffled, never random."""
        rows = sorted(rows, key=lambda r: r["date"])
        X = np.array([[float(r.get(f, 0) or 0) for f in self.features]
                      for r in rows])
        y = np.array([int(r["home_won"]) for r in rows])
        p_struct = np.array([float(r["p_home"]) for r in rows])
        offset = _logit(p_struct)
        n = len(rows)

        if self.backend is None or n < config.ML_MIN_TRAIN_GAMES:
            why = ("no GBM backend" if self.backend is None else
                   f"only {n} games < {config.ML_MIN_TRAIN_GAMES} minimum")
            log.warning("ML layer gated off before training: %s", why)
            self.report = ValidationReport(0, n, 0, _log_loss(p_struct, y),
                                           _log_loss(p_struct, y), False, [])
            self.shipped = False
            return self.report

        # Expanding-window folds over the date-ordered rows.
        edges = np.linspace(0, n, n_folds + 2, dtype=int)  # first block = seed
        folds, oos_ml, oos_struct, oos_y = [], [], [], []
        for k in range(1, n_folds + 1):
            tr = slice(0, edges[k])
            te = slice(edges[k], edges[k + 1])
            if edges[k + 1] - edges[k] < 20:
                continue
            m = self._fit_one(X[tr], y[tr], offset[tr])
            p_ml = self._predict_one(m, X[te], offset[te])
            folds.append({"fold": k, "n_train": edges[k],
                          "n_test": int(edges[k + 1] - edges[k]),
                          "ll_struct": _log_loss(p_struct[te], y[te]),
                          "ll_ml": _log_loss(p_ml, y[te])})
            oos_ml.extend(p_ml)
            oos_struct.extend(p_struct[te])
            oos_y.extend(y[te])

        ll_s = _log_loss(oos_struct, oos_y)
        ll_m = _log_loss(oos_ml, oos_y)
        self.shipped = ll_m < ll_s  # the gate
        self.report = ValidationReport(len(folds), n, len(oos_y),
                                       ll_s, ll_m, self.shipped, folds)
        # Final model refit on ALL data (only used if the gate passed).
        self.model = self._fit_one(X, y, offset)
        log.warning("ML walk-forward: struct %.4f vs ml %.4f -> %s",
                    ll_s, ll_m, "SHIP" if self.shipped else "GATE OFF")
        return self.report

    def predict(self, feature_row: dict) -> float:
        """Blended home win probability for one game. Falls back to the
        structural probability whenever the gate is closed or inputs are
        missing - the ML layer can only ever be an upgrade, never a crash."""
        p_struct = float(feature_row["p_home"])
        if not self.shipped or self.model is None:
            return p_struct
        try:
            X = np.array([[float(feature_row.get(f, 0) or 0)
                           for f in self.features]])
            off = _logit(np.array([p_struct]))
            return float(self._predict_one(self.model, X, off)[0])
        except Exception as e:  # noqa: BLE001 - any ML failure -> structural
            log.warning("ML predict failed (%s); using structural", e)
            return p_struct

    # -------------------------------------------------------- persistence
    def save(self, path: str | Path = config.ML_MODEL_FILE):
        # report saved as plain text, not as the dataclass: pickling a class
        # defined in this module breaks unpickling when training ran via
        # `python -m` (the class pickles under __main__ and other entry
        # points can't resolve it - bit us on 2026-07-21).
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model,
                         "shipped": self.shipped, "features": self.features,
                         "report_text": self.report.summary() if self.report
                         else ""}, f)
        log.info("Saved ML model (%s, shipped=%s) to %s",
                 self.backend, self.shipped, path)

    @classmethod
    def load(cls, path: str | Path = config.ML_MODEL_FILE) -> "ResidualModel | None":
        path = Path(path)
        if not path.exists():
            log.info("No ML model at %s - structural only", path)
            return None
        m = cls()
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            if d["backend"] != m.backend:
                log.warning("ML model was trained with %s but %s is "
                            "installed - ignoring saved model",
                            d["backend"], m.backend)
                return None
            m.model, m.shipped = d["model"], d["shipped"]
            m.features = d["features"]
            m.report = None  # summary text lives in d["report_text"]
            return m
        except Exception as e:  # noqa: BLE001
            log.error("Failed to load ML model: %s", e)
            return None


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True,
                    help="predictions.csv produced by backtest.py")
    ap.add_argument("--out", type=Path, default=Path(config.ML_MODEL_FILE))
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with open(args.train, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Empty training file.")
    missing = [c for c in ("date", "p_home", "home_won") if c not in rows[0]]
    if missing:
        raise SystemExit(f"Training file missing columns: {missing} "
                         "(re-run backtest.py to regenerate)")
    m = ResidualModel()
    report = m.fit(rows, n_folds=args.folds)
    print("\n" + report.summary())
    m.save(args.out)
    print(f"\nModel saved to {args.out} (gate {'OPEN' if m.shipped else 'CLOSED'})")


if __name__ == "__main__":
    main()
