"""Cluster-level feature-importance tests (the 10-mark section).

The synthetic features (``f1_a``/``f1_b``/``f2_vol``) are NOT in the real redundancy file, so the
identity / mapping / joint-MDA checks build a small SYNTHETIC ``clusters`` dict; one separate light
test exercises the real ``results/feature_redundancy.json`` loader. The load-bearing assertion is
the **decomposition identity**: cluster MDI sums + the ``"dummies"`` row reconstruct the model's
total native importance exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.model.cv import CombinatorialPurgedCV
from stml.model.importance import (
    clustered_mda,
    clustered_mdi,
    clustered_shap,
    iterative_cluster_drop,
    load_feature_clusters,
    map_model_features_to_clusters,
    native_tree_importance,
    noise_injection_check,
    single_feature_importance,
)
from stml.model.mlp import MLPModel
from stml.model.trees import RFModel, XGBModel

SEED = 42
FEATS = ["f1_a", "f1_b", "f2_vol"]
SYN_CLUSTERS = {0: ["f1_a", "f1_b"], 1: ["f2_vol"]}
RF_PARAMS = {"n_estimators": 60, "max_depth": 4}
XGB_PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}
MLP_PARAMS = {"hidden": (8,), "epochs": 30}


def _xy(synthetic_panel):
    X = synthetic_panel[FEATS].reset_index(drop=True)
    y = synthetic_panel["bin"].to_numpy()
    return X, y


# -- 1. decomposition identity ------------------------------------------------
def test_clustered_mdi_decomposition_identity(synthetic_panel):
    X, y = _xy(synthetic_panel)
    model = XGBModel(XGB_PARAMS, SEED).fit(X, y)
    native = native_tree_importance(model)
    cmdi = clustered_mdi(model, SYN_CLUSTERS)

    # cluster 0 == native[f1_a] + native[f1_b]; cluster 1 == native[f2_vol]
    assert cmdi[0] == np.float64(native[["f1_a", "f1_b"]].sum())
    assert abs(cmdi[1] - float(native["f2_vol"])) < 1e-9
    assert abs(cmdi[0] - float(native[["f1_a", "f1_b"]].sum())) < 1e-9
    # no model column is a dummy here
    assert cmdi["dummies"] == 0.0
    # total over clusters + dummies == total native importance
    # (XGBoost importances are float32; the per-cluster grouping above is exact -- only the
    # full-vector resummation differs at the float32 epsilon, so tolerance is float32-sized.)
    assert abs(cmdi.sum() - float(native.sum())) < 1e-6


# -- 2. mapping (every feature -> exactly one id|dummies; missing surfaced) ----
def test_map_model_features_to_clusters():
    feats = ["f1_x", "f4_clust_3", "inst_es1s", "f2_y"]
    clusters = {10: ["f1_x"], 20: ["f2_y"], 30: ["f3_z"]}
    assignment, missing = map_model_features_to_clusters(feats, clusters)

    assert set(assignment) == set(feats)  # every feature mapped exactly once
    assert assignment["f1_x"] == 10
    assert assignment["f2_y"] == 20
    assert assignment["f4_clust_3"] == "dummies"
    assert assignment["inst_es1s"] == "dummies"
    # cluster 30's representative (f3_z) is pruned -> reported, not silent
    assert missing == [30]


def test_mapping_covers_dummies_and_clusters_only():
    feats = ["f1_a", "f1_b", "f2_vol", "f4_clust_1", "inst_aaa"]
    assignment, missing = map_model_features_to_clusters(feats, SYN_CLUSTERS)
    assert assignment["f4_clust_1"] == "dummies"
    assert assignment["inst_aaa"] == "dummies"
    assert assignment["f1_a"] == 0 and assignment["f2_vol"] == 1
    assert missing == []  # both synthetic clusters represented


# -- 3. joint MDA over a multi-member cluster, tree + torch -------------------
def test_clustered_mda_joint_tree(synthetic_panel):
    X, y = _xy(synthetic_panel)
    model = XGBModel(XGB_PARAMS, SEED).fit(X, y)
    mda = clustered_mda(model, X, y, SYN_CLUSTERS, n_repeats=3, seed=SEED)
    assert 0 in mda.index and 1 in mda.index  # 2-member cluster 0 row present
    assert np.isfinite(mda[0]) and np.isfinite(mda[1])


def test_clustered_mda_joint_rf(synthetic_panel):
    X, y = _xy(synthetic_panel)
    model = RFModel(RF_PARAMS, SEED).fit(X, y)
    mda = clustered_mda(model, X, y, SYN_CLUSTERS, n_repeats=3, seed=SEED)
    assert 0 in mda.index and np.isfinite(mda[0])


def test_clustered_mda_joint_torch(synthetic_panel):
    X, y = _xy(synthetic_panel)
    model = MLPModel(MLP_PARAMS, SEED).fit(X, y)
    mda = clustered_mda(model, X, y, SYN_CLUSTERS, n_repeats=3, seed=SEED)
    assert 0 in mda.index and np.isfinite(mda[0]) and np.isfinite(mda[1])


def test_clustered_shap_groups_to_clusters(synthetic_panel):
    X, y = _xy(synthetic_panel)
    model = XGBModel(XGB_PARAMS, SEED).fit(X, y)
    cshap = clustered_shap(model, X, SYN_CLUSTERS, max_samples=100, seed=SEED)
    assert set(cshap.index) >= {0, 1, "dummies"}
    assert np.isfinite(cshap[0]) and np.isfinite(cshap[1])


# -- 4. noise injection: real feature outranks injected noise -----------------
def test_noise_injection_ranks_below_real(synthetic_panel):
    X, y = _xy(synthetic_panel)
    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    frame = noise_injection_check(RFModel, RF_PARAMS, X, y, synthetic_panel, cv,
                                  n_noise=5, seed=SEED)
    assert frame["is_noise"].sum() == 5
    # f1_a drives the label -> it must rank above every injected noise column
    f1a_imp = frame.loc["f1_a", "perm_auc_drop"]
    max_noise = frame.loc[frame["is_noise"], "perm_auc_drop"].max()
    assert f1a_imp > max_noise


# -- 5. SFI + iterative cluster drop run + shapes -----------------------------
def test_single_feature_importance_shape(synthetic_panel):
    X, y = _xy(synthetic_panel)
    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    sfi = single_feature_importance(RFModel, RF_PARAMS, X, y, synthetic_panel, cv, seed=SEED)
    assert isinstance(sfi, pd.Series) and not sfi.empty
    assert set(sfi.index) <= set(FEATS)
    assert sfi.is_monotonic_decreasing
    assert sfi.notna().all()


def test_iterative_cluster_drop_shape(synthetic_panel):
    X, y = _xy(synthetic_panel)
    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    frame = iterative_cluster_drop(
        XGBModel, XGB_PARAMS, X, y, synthetic_panel, cv, SYN_CLUSTERS,
        importance_fn=clustered_mdi, seed=SEED, min_clusters=1, max_iters=2,
    )
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["dropped_cluster", "n_features_left", "cv_auc", "delta_auc"]
    assert not frame.empty
    # features strictly shrink as clusters are dropped
    assert frame["n_features_left"].is_monotonic_decreasing


# -- real redundancy file: members non-empty, total to 175 feature names ------
def test_load_feature_clusters_real_file():
    clusters = load_feature_clusters()
    assert isinstance(clusters, dict)
    assert all(len(members) >= 1 for members in clusters.values())
    all_feats = [f for members in clusters.values() for f in members]
    assert len(all_feats) == 175
    assert len(set(all_feats)) == 175  # no feature in two clusters
