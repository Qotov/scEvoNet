"""Resampling and negative controls for feature importance and batch robustness."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scevonet.net import EvoManager, SampleConfig, fit_ovr_model

logger = logging.getLogger(__name__)


def bootstrap_importance_stability(
    matrix: pd.DataFrame,
    labels: Sequence[str] | pd.Series,
    cell_type: str,
    *,
    config: SampleConfig | None = None,
    n_bootstrap: int = 20,
    top_k: int = 100,
    stability_threshold: float = 0.5,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    **Bootstrap resampling of cells** — refit the one-vs-rest model for ``cell_type``
    repeatedly and measure how often each gene appears in the top-``top_k`` by importance.

    Returns
    -------
    stability_table
        Columns: ``gene``, ``frequency`` (proportion of boots in top_k), ``stable``
        (True if frequency ≥ ``stability_threshold``).
    observed_ranking
        Importances from a single fit on the full data (reference ranking).

    Notes
    -----
    Computationally heavy (``n_bootstrap`` full LightGBM fits). Reduce ``n_bootstrap``
    or ``top_k`` for exploration.
    """
    cfg = config or SampleConfig()
    labels_arr = np.asarray(pd.Series(labels).astype(str))
    n = len(labels_arr)
    rng = np.random.default_rng(random_seed)

    # Reference fit on full data
    y_full = pd.Series((labels_arr == cell_type).astype(int), index=matrix.index)
    ref = fit_ovr_model(matrix, y_full, cfg)
    ref_genes = ref["features"]["gene"].head(top_k).tolist()
    ref_imp = ref["features"].set_index("gene")["importance"].reindex(ref_genes)

    counts: Counter[str] = Counter()
    n_ok = 0
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        matrix_b = matrix.iloc[idx]
        labels_b = labels_arr[idx]
        y_b = pd.Series((labels_b == cell_type).astype(int), index=matrix_b.index)
        # Skip degenerate bootstrap samples
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        n_ok += 1
        boot_cfg = SampleConfig(
            top_features_limit=cfg.top_features_limit,
            n_estimators=cfg.n_estimators,
            test_size=cfg.test_size,
            random_state=cfg.random_state + b + 1,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        out = fit_ovr_model(matrix_b, y_b, boot_cfg)
        top = set(out["features"]["gene"].head(top_k).astype(str))
        counts.update(top)

    denom = max(n_ok, 1)
    freq_rows = []
    for gene in sorted(counts.keys(), key=lambda g: -counts[g]):
        freq_rows.append(
            {
                "gene": gene,
                "frequency": counts[gene] / denom,
                "stable": (counts[gene] / denom) >= stability_threshold,
            }
        )
    stability_df = pd.DataFrame(freq_rows).sort_values("frequency", ascending=False)
    return stability_df, ref_imp


def permutation_importance_null(
    matrix: pd.DataFrame,
    labels: Sequence[str] | pd.Series,
    cell_type: str,
    *,
    config: SampleConfig | None = None,
    n_perm: int = 50,
    summary_statistic: str = "top10_mean",
    random_seed: int = 0,
) -> dict:
    """
    **Label permutation null:** shuffle cell-type labels, refit the same OVR model,
    and build a null distribution of a scalar summary of importance (e.g. mean of top 10).

    Compare the **observed** statistic from the real labels to this null to sanity-check
    that the classifier is not matching noise.

    Parameters
    ----------
    summary_statistic
        ``top10_mean`` — mean importance of top 10 genes; ``top1`` — max importance.

    Returns
    -------
    dict with keys ``observed``, ``null_values``, ``p_value`` (one-sided: how often
    null ≥ observed; low p suggests structure beyond chance).
    """
    cfg = config or SampleConfig()
    labels_arr = np.asarray(pd.Series(labels).astype(str))
    rng = np.random.default_rng(random_seed)

    def stat_from_features(feat_df: pd.DataFrame) -> float:
        imp = feat_df["importance"].values
        if summary_statistic == "top1":
            return float(np.nanmax(imp))
        if summary_statistic == "top10_mean":
            k = min(10, len(imp))
            top = np.sort(imp)[-k:]
            return float(np.mean(top))
        raise ValueError("summary_statistic must be 'top1' or 'top10_mean'")

    y_real = pd.Series((labels_arr == cell_type).astype(int), index=matrix.index)
    obs_fit = fit_ovr_model(matrix, y_real, cfg)
    observed = stat_from_features(obs_fit["features"])

    null_vals: list[float] = []
    for p in range(n_perm):
        perm = rng.permutation(labels_arr)
        y_perm = pd.Series((perm == cell_type).astype(int), index=matrix.index)
        if y_perm.sum() == 0 or y_perm.sum() == len(y_perm):
            continue
        perm_cfg = SampleConfig(
            top_features_limit=cfg.top_features_limit,
            n_estimators=cfg.n_estimators,
            test_size=cfg.test_size,
            random_state=cfg.random_state + p + 1000,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        fit_p = fit_ovr_model(matrix, y_perm, perm_cfg)
        null_vals.append(stat_from_features(fit_p["features"]))

    if not null_vals:
        return {"observed": observed, "null_values": [], "p_value": np.nan}

    null_arr = np.asarray(null_vals)
    p_value = float(np.mean(null_arr >= observed))
    return {"observed": observed, "null_values": null_vals, "p_value": p_value}


def leave_batch_out_auc(
    matrix: pd.DataFrame,
    labels: Sequence[str] | pd.Series,
    batch_labels: Sequence[str] | pd.Series,
    cell_type: str,
    *,
    config: SampleConfig | None = None,
) -> pd.DataFrame:
    """
    **Cross-batch robustness:** for each batch ID, train the OVR model on all other
    batches and score ROC-AUC on held-out cells from that batch.

    Rows with only one class in the test batch receive NaN AUC.

    Parameters
    ----------
    batch_labels
        Batch / donor / plate ID per cell (aligned with ``matrix`` rows).
    """
    cfg = config or SampleConfig()
    lab = pd.Series(labels, index=matrix.index).astype(str)
    batches = pd.Series(batch_labels, index=matrix.index).astype(str)

    rows = []
    for holdout in sorted(batches.unique()):
        train_mask = batches != holdout
        test_mask = batches == holdout
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        m_tr = matrix.loc[train_mask]
        y_tr = pd.Series((lab.loc[train_mask] == cell_type).astype(int), index=m_tr.index)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            logger.warning("Skipping batch %s: single class in training portion.", holdout)
            continue
        fit = fit_ovr_model(m_tr, y_tr, cfg)
        feats = fit["features"]["gene"].tolist()
        m_te = matrix.loc[test_mask]
        y_te = (lab.loc[test_mask] == cell_type).astype(int).values
        if len(np.unique(y_te)) < 2:
            auc = np.nan
        else:
            x_prep = EvoManager.prepare_input_df(m_te, feats)
            pred = fit["model"].predict(x_prep)
            auc = float(roc_auc_score(y_te, pred))
        rows.append({"held_out_batch": holdout, "n_test_cells": int(test_mask.sum()), "auc": auc})
    return pd.DataFrame(rows)
