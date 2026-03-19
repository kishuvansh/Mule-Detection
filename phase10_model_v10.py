#!/usr/bin/env python3
"""
phase10_model_v10.py
====================
Full end-to-end improvements for mule detection:

1. TIME-BASED CROSS-VALIDATION (no temporal leakage)
2. VELOCITY / RATIO / BURST FEATURES (from existing columns)
3. OPTUNA HYPERPARAMETER TUNING (LGB + XGB + CatBoost)
4. FIXED TEMPORAL WINDOWS + SCORE SMOOTHING
5. ERROR ANALYSIS LOOP (FP/FN pattern detection + targeted features)

Target metrics:
  AUC-ROC   >= 0.994
  F1        >= 0.91
  Temporal IoU >= 0.72
  All RH scores > 0.95
"""

import pandas as pd
import numpy as np
import time, warnings, gc
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, precision_recall_curve)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
DATA_DIR = "."
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load Data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train = pd.read_csv("features_train_p2.csv")
test  = pd.read_csv("features_test_p2.csv")
accounts = pd.read_parquet(f"{DATA_DIR}/accounts.parquet")
labels   = pd.read_parquet(f"{DATA_DIR}/train_labels.parquet")

train = train.merge(accounts[["account_id", "branch_code", "account_opening_date"]],
                    on="account_id", how="left")
test  = test.merge(accounts[["account_id", "branch_code", "account_opening_date"]],
                   on="account_id", how="left")
train = train.merge(labels[["account_id", "alert_reason", "mule_flag_date"]],
                    on="account_id", how="left")

# Parse dates for time-based CV
train["account_opening_date"] = pd.to_datetime(train["account_opening_date"], errors="coerce")

global_mean = train["is_mule"].mean()
print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Mule rate: {global_mean:.4f} ({train['is_mule'].sum()} mules)")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Time-Based Cross-Validation Splits
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TIME-BASED CROSS-VALIDATION SETUP")
print("=" * 60)

# Sort accounts by their opening date → temporal ordering
train["_time_rank"] = train["account_opening_date"].rank(method="first", na_option="bottom")
train = train.sort_values("_time_rank").reset_index(drop=True)

N = len(train)
n_folds = 5

# Build time-based folds: each fold trains on earlier accounts, validates on later ones
# Fold k: train on accounts in time rank percentile [0, (k+1)/6), validate on [(k+1)/6, (k+2)/6)
# This ensures no account_id overlap and respects chronological order
time_fold_indices = []
step = N // (n_folds + 1)
for k in range(n_folds):
    val_start = step * (k + 1)
    val_end   = step * (k + 2)
    tr_idx    = list(range(0, val_start))               # all earlier accounts
    val_idx   = list(range(val_start, min(val_end, N))) # later slice as validation
    time_fold_indices.append((np.array(tr_idx), np.array(val_idx)))
    print(f"  Fold {k+1}: train={len(tr_idx):,}  val={len(val_idx):,}  "
          f"(val dates: {train.iloc[val_start]['account_opening_date'].date() if not pd.isna(train.iloc[val_start]['account_opening_date']) else 'NaT'} "
          f"→ {train.iloc[min(val_end-1, N-1)]['account_opening_date'].date() if not pd.isna(train.iloc[min(val_end-1, N-1)]['account_opening_date']) else 'NaT'})")

print("✅ Time-based folds built (no temporal leakage)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. OOF Target Encoding (Branch)
# ══════════════════════════════════════════════════════════════════════════════
print("\nOOF Branch Target Encoding...")
train["branch_mule_rate_oof"] = np.nan

for tr_idx, val_idx in time_fold_indices:
    tr_df = train.iloc[tr_idx]
    bs = tr_df.groupby("branch_code")["is_mule"].agg(["sum", "count"])
    bs["rate"] = (bs["sum"] + 10 * global_mean) / (bs["count"] + 10)
    mapped = train.iloc[val_idx]["branch_code"].map(bs["rate"]).fillna(global_mean)
    train.loc[train.index[val_idx], "branch_mule_rate_oof"] = mapped.values

# Fill remaining NaNs (first fold's train has no prior)
train["branch_mule_rate_oof"].fillna(global_mean, inplace=True)

branch_full = train.groupby("branch_code")["is_mule"].agg(["sum", "count"])
branch_full["rate"] = (branch_full["sum"] + 10 * global_mean) / (branch_full["count"] + 10)
test["branch_mule_rate_oof"] = test["branch_code"].map(branch_full["rate"]).fillna(global_mean)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature Engineering — Ratios, Velocity, Burst, Aggregations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def engineer_features(df, ref_df=None):
    """
    Add velocity, ratio, burst, aggregation, interaction features.
    ref_df = train DataFrame (for computing quantile thresholds without leakage).
    """
    if ref_df is None:
        ref_df = df

    # ── Ratio features ────────────────────────────────────────────────────────
    if "credit_volume" in df.columns and "debit_volume" in df.columns:
        df["in_out_vol_ratio"]  = df["credit_volume"] / (df["debit_volume"].clip(1))
        df["net_flow"]          = df["credit_volume"] - df["debit_volume"]
        df["pass_through_ratio"] = df["net_flow"].abs() / (df["credit_volume"] + df["debit_volume"] + 1)

    if "credit_cp_n" in df.columns and "debit_cp_n" in df.columns:
        df["in_out_cp_ratio"] = df["credit_cp_n"] / (df["debit_cp_n"].clip(1))

    if "unique_cp_count" in df.columns and "txn_count" in df.columns:
        df["unique_recv_ratio"]  = df["unique_cp_count"] / df["txn_count"].clip(1)
        df["txn_per_cp"]         = df["txn_count"] / df["unique_cp_count"].clip(1)

    # ── Velocity proxies (from available features) ────────────────────────────
    if "txn_count" in df.columns and "active_days" in df.columns:
        df["txn_velocity"]       = df["txn_count"] / df["active_days"].clip(1)
        df["txn_per_week"]       = df["txn_count"] / (df["active_days"].clip(1) / 7)

    if "gap_mean" in df.columns:
        # mean gap in hours → proxy for transaction frequency
        df["gap_mean_hours"]     = df["gap_mean"] / 3600 if df["gap_mean"].max() > 10000 else df["gap_mean"]
        df["high_freq_flag"]     = (df["gap_mean_hours"] < 2).astype(float)

    # ── Burst features ────────────────────────────────────────────────────────
    if "gap_std" in df.columns and "gap_mean" in df.columns:
        df["gap_cv"]             = df["gap_std"] / (df["gap_mean"].clip(1))
        df["burst_score"]        = df["gap_cv"] * df["txn_velocity"] if "txn_velocity" in df.columns else df["gap_cv"]

    if "night_count" in df.columns and "txn_count" in df.columns:
        df["night_burst"]        = df["night_count"] / df["txn_count"].clip(1)

    if "near_threshold_count" in df.columns and "txn_count" in df.columns:
        df["structure_intensity"] = df["near_threshold_count"] / df["txn_count"].clip(1)

    # ── Aggregation features ──────────────────────────────────────────────────
    if "total_volume" in df.columns and "txn_count" in df.columns:
        df["mean_amount"]        = df["total_volume"] / df["txn_count"].clip(1)

    if "amt_std" in df.columns and "amt_mean" in df.columns:
        df["cv_amount"]          = df["amt_std"] / (df["amt_mean"].clip(1))

    if "amt_max" in df.columns and "total_volume" in df.columns:
        df["max_to_total"]       = df["amt_max"] / (df["total_volume"].clip(1))

    if "balance_std" in df.columns and "balance_min" in df.columns:
        df["balance_range"]      = df["amt_max"].clip(0) - df["balance_min"] if "amt_max" in df.columns else df["balance_std"]

    # ── Interaction features ──────────────────────────────────────────────────
    if "txn_velocity" in df.columns and "unique_recv_ratio" in df.columns:
        df["velocity_x_receivers"] = df["txn_velocity"] * df["unique_recv_ratio"]

    if "burst_score" in df.columns and "near_threshold_pct" in df.columns:
        df["burst_x_structuring"]  = df["burst_score"] * df["near_threshold_pct"]

    if "in_out_vol_ratio" in df.columns and "mule_trigram_count" in df.columns:
        df["ratio_x_trigram"]      = df["in_out_vol_ratio"] * df["mule_trigram_count"].fillna(0)

    if "txn_velocity" in df.columns and "branch_mule_rate_oof" in df.columns:
        df["velocity_x_branch"]    = df["txn_velocity"] * df["branch_mule_rate_oof"]

    if "unique_recv_ratio" in df.columns and "round_amount_pct" in df.columns:
        df["receivers_x_round"]    = df["unique_recv_ratio"] * df["round_amount_pct"]

    # ── Feature Derivatives (Rate of Change & Accelerations) ──────────────────
    if "active_days" in df.columns:
        if "total_volume" in df.columns:
            df["vol_derivative"] = df["total_volume"] / df["active_days"].clip(1)
            df["vol_acceleration"] = df["vol_derivative"] / df["active_days"].clip(1)
        if "txn_count" in df.columns:
            if "txn_velocity" in df.columns:
                df["txn_acceleration"] = df["txn_velocity"] / df["active_days"].clip(1)
        if "credit_volume" in df.columns and "debit_volume" in df.columns:
            df["credit_derivative"] = df["credit_volume"] / df["active_days"].clip(1)
            df["debit_derivative"] = df["debit_volume"] / df["active_days"].clip(1)
            df["net_flow_derivative"] = (df["credit_volume"] - df["debit_volume"]) / df["active_days"].clip(1)
        if "unique_cp_count" in df.columns:
            df["cp_networking_derivative"] = df["unique_cp_count"] / df["active_days"].clip(1)
        if "balance_std" in df.columns:
            df["balance_volatility_derivative"] = df["balance_std"] / df["active_days"].clip(1)
        if "amt_mean" in df.columns:
            df["amt_mean_derivative"] = df["amt_mean"] / df["active_days"].clip(1)
        if "gap_cv" in df.columns:
            df["burstiness_derivative"] = df["gap_cv"] / df["active_days"].clip(1)

    # Cross-derivatives
    if "vol_derivative" in df.columns and "txn_count" in df.columns:
        df["vol_per_txn_derivative"] = df["vol_derivative"] / df["txn_count"].clip(1)

    return df

train = engineer_features(train, ref_df=train)
test  = engineer_features(test,  ref_df=train)
print(f"After engineering: Train {train.shape[1]} cols | Test {test.shape[1]} cols")

# ── Multi-signal features ─────────────────────────────────────────────────────
print("Building multi-signal features...")
q90_near   = train["near_threshold_pct"].quantile(0.90)
q90_round  = train["round_amount_pct"].quantile(0.90)
q75_vol    = train["total_volume"].quantile(0.75)
q90_branch = train["branch_mule_rate_oof"].quantile(0.90)
q90_vtob   = (train["total_volume"] / train["balance_mean"].abs().clip(1)).quantile(0.90) if "balance_mean" in train.columns else None
train_geo  = (train["geo_spread_lat"].fillna(0) + train["geo_spread_lon"].fillna(0)) if "geo_spread_lat" in train.columns else None
q90_geo    = train_geo.quantile(0.90) if train_geo is not None else 0.0

for df in [train, test]:
    signals = []
    if "days_to_first_large" in df.columns:
        df["sig_dormant"]       = (df["days_to_first_large"] > 180).astype(float); signals.append("sig_dormant")
    if "near_threshold_pct" in df.columns:
        df["sig_structuring"]   = (df["near_threshold_pct"] > q90_near).astype(float); signals.append("sig_structuring")
    if "median_dwell_hours" in df.columns:
        df["sig_rapid"]         = (df["median_dwell_hours"] < 24).astype(float); signals.append("sig_rapid")
    if "unique_cp_count" in df.columns and "txn_count" in df.columns:
        df["sig_fanout"]        = (df["unique_cp_count"] / df["txn_count"].clip(1) > 0.5).astype(float); signals.append("sig_fanout")
    if "days_to_first_large" in df.columns and "total_volume" in df.columns:
        df["sig_new_highvol"]   = ((df["days_to_first_large"] < 30) & (df["total_volume"] > q75_vol)).astype(float); signals.append("sig_new_highvol")
    if "round_amount_pct" in df.columns:
        df["sig_round"]         = (df["round_amount_pct"] > q90_round).astype(float); signals.append("sig_round")
    if "has_mobile_update" in df.columns:
        df["sig_post_mobile"]   = (df["has_mobile_update"] == 1).astype(float); signals.append("sig_post_mobile")
    if "total_volume" in df.columns and "balance_mean" in df.columns and q90_vtob:
        vtob = df["total_volume"] / df["balance_mean"].abs().clip(1)
        df["sig_income_mismatch"] = (vtob > q90_vtob).astype(float); signals.append("sig_income_mismatch")
    if "branch_mule_rate_oof" in df.columns:
        df["sig_branch_cluster"] = (df["branch_mule_rate_oof"] > q90_branch).astype(float); signals.append("sig_branch_cluster")
    if "residual_ratio" in df.columns:
        df["sig_residual"]       = (df["residual_ratio"] < 0.01).astype(float); signals.append("sig_residual")
    if "geo_spread_lat" in df.columns:
        geo = df["geo_spread_lat"].fillna(0) + df["geo_spread_lon"].fillna(0)
        df["sig_geo"]            = (geo > q90_geo).astype(float); signals.append("sig_geo")
    if "has_prior_freeze" in df.columns:
        df["sig_freeze"]         = df["has_prior_freeze"].fillna(0).astype(float); signals.append("sig_freeze")

    if signals:
        sv = np.column_stack([df[s].fillna(0).values for s in signals])
        df["multi_signal_count"]  = sv.sum(axis=1)
        df["signal_consistency"]  = df["multi_signal_count"] / len(signals)
        df["signal_variance"]     = np.var(sv, axis=1)
        p = sv.clip(1e-9, 1-1e-9)
        df["signal_entropy"]      = (-(p*np.log(p) + (1-p)*np.log(1-p))).mean(axis=1)
        df["signal_saturation"]   = (df["multi_signal_count"] >= len(signals)*0.75).astype(float)
        df["too_perfect_score"]   = df["signal_entropy"] * df["signal_saturation"]

# Extra interaction with new features
for df in [train, test]:
    if "multi_signal_count" in df.columns and "txn_velocity" in df.columns:
        df["signals_x_velocity"]   = df["multi_signal_count"] * df["txn_velocity"]
    if "signal_consistency" in df.columns and "burst_score" in df.columns:
        df["consistency_x_burst"]  = df["signal_consistency"] * df["burst_score"]

# Composite score
score_cols = ["near_threshold_pct", "round_amount_pct", "gap_cv", "degree_centrality",
              "mule_trigram_count", "branch_mule_rate_oof", "has_prior_freeze"]
for df in [train, test]:
    c = np.zeros(len(df))
    for col in score_cols:
        if col in df.columns:
            m, s = train[col].mean(), train[col].std()
            c += (df[col].fillna(m) - m) / s if s > 0 else 0
    df["composite_score_v10"] = c

print(f"Signal features: {len(signals)} signals + aggregates built")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Prepare Feature Set
# ══════════════════════════════════════════════════════════════════════════════
drop_cols = ["account_id", "is_mule", "first_large_ts", "open_date",
             "branch_code", "branch_mule_rate", "composite_score",
             "alert_reason", "mule_flag_date", "_time_rank",
             "account_opening_date"]
features = [c for c in train.columns
            if c not in drop_cols and train[c].nunique() > 1
            and pd.api.types.is_numeric_dtype(train[c])]

train[features] = train[features].fillna(train[features].median())
test[features]  = test[features].fillna(train[features].median())
print(f"\nTotal features: {len(features)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Red Herring Screening Pass
# ══════════════════════════════════════════════════════════════════════════════
print("\nRed herring screening (quick LGB pass)...")
X = train[features].values
y = train["is_mule"].values

# Use time-based folds for screening too
oof_screen = np.zeros(len(y))
for tr_idx, val_idx in time_fold_indices:
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            max_depth=7, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X[tr_idx], y[tr_idx])
    oof_screen[val_idx] = m.predict_proba(X[val_idx])[:, 1]

extreme_rh = (y == 1) & (oof_screen < 0.015)
keep_mask  = ~extreme_rh
X_clean    = X[keep_mask]
y_clean    = y[keep_mask]
oof_screen_clean = oof_screen[keep_mask]

# Soft down-weight ambiguous mules
sample_weights = np.ones(len(y_clean))
ambig = ((y_clean == 1) & (oof_screen_clean > 0.015) & (oof_screen_clean < 0.08) &
         (train["multi_signal_count"].values[keep_mask] < 2))
sample_weights[ambig] = 0.6

print(f"Pruned {extreme_rh.sum()} extreme RH → {len(y_clean):,} samples")
print(f"Down-weighted {ambig.sum()} ambiguous samples")

# Rebuild time-fold indices on cleaned data
clean_indices = np.where(keep_mask)[0]
time_folds_clean = []
for tr_idx_orig, val_idx_orig in time_fold_indices:
    tr_set  = set(tr_idx_orig)
    val_set = set(val_idx_orig)
    tr_new  = np.array([i for i, orig in enumerate(clean_indices) if orig in tr_set])
    val_new = np.array([i for i, orig in enumerate(clean_indices) if orig in val_set])
    if len(tr_new) > 0 and len(val_new) > 0:
        time_folds_clean.append((tr_new, val_new))
print(f"Time-based folds after cleaning: {len(time_folds_clean)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Optuna Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("Optuna available — running tuning")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed — using pre-tuned defaults")
    print("Install with: pip install optuna")

spw = (y_clean == 0).sum() / max(1, (y_clean == 1).sum())

# Use a single time-based hold-out for fast tuning (fold 4 = last chunk)
if len(time_folds_clean) >= 4:
    tune_tr, tune_val = time_folds_clean[-2]  # second-to-last fold
else:
    tune_tr, tune_val = time_folds_clean[0]

Xtr_t, Xval_t = X_clean[tune_tr], X_clean[tune_val]
ytr_t, yval_t = y_clean[tune_tr], y_clean[tune_val]
wtr_t = sample_weights[tune_tr]
print(f"Tuning fold: train={len(Xtr_t):,} val={len(Xval_t):,}")

if OPTUNA_AVAILABLE:
    # ── LightGBM tuning ──────────────────────────────────────────────────
    def lgb_objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 800, 3000),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth":       trial.suggest_int("max_depth", 6, 12),
            "num_leaves":      trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 1.0),
            "scale_pos_weight": spw,
            "random_state": 42, "verbosity": -1, "n_jobs": -1
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr_t, ytr_t, sample_weight=wtr_t,
              eval_set=[(Xval_t, yval_t)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        prob = m.predict_proba(Xval_t)[:, 1]
        return roc_auc_score(yval_t, prob)

    study_lgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=80, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({"scale_pos_weight": spw, "random_state": 42,
                             "verbosity": -1, "n_jobs": -1})
    print(f"✅ LGB best AUC: {study_lgb.best_value:.4f} | params: lr={best_lgb_params.get('learning_rate',0):.4f}")

    # ── XGBoost tuning ───────────────────────────────────────────────────
    def xgb_objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 800, 3000),
            "learning_rate":  trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth":      trial.suggest_int("max_depth", 5, 10),
            "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":          trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":      trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda":     trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": spw,
            "random_state": 42, "verbosity": 0,
            "eval_metric": "auc", "n_jobs": -1,
            "early_stopping_rounds": 50
        }
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr_t, ytr_t, sample_weight=wtr_t,
              eval_set=[(Xval_t, yval_t)], verbose=False)
        prob = m.predict_proba(Xval_t)[:, 1]
        return roc_auc_score(yval_t, prob)

    study_xgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({"scale_pos_weight": spw, "random_state": 42,
                             "verbosity": 0, "eval_metric": "auc", "n_jobs": -1,
                             "early_stopping_rounds": 50})
    print(f"✅ XGB best AUC: {study_xgb.best_value:.4f}")

    # ── CatBoost tuning ──────────────────────────────────────────────────
    def cat_objective(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 800, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth":         trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count":  trial.suggest_categorical("border_count", [64, 128, 254]),
            "auto_class_weights": "Balanced",
            "random_state": 42, "verbose": False, "early_stopping_rounds": 50
        }
        m = CatBoostClassifier(**params)
        m.fit(Xtr_t, ytr_t, eval_set=(Xval_t, yval_t))
        prob = m.predict_proba(Xval_t)[:, 1]
        return roc_auc_score(yval_t, prob)

    study_cat = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(cat_objective, n_trials=40, show_progress_bar=False)
    best_cat_params = study_cat.best_params
    best_cat_params.update({"auto_class_weights": "Balanced",
                             "random_state": 42, "verbose": False,
                             "early_stopping_rounds": 50})
    print(f"✅ CAT best AUC: {study_cat.best_value:.4f}")

else:
    # Pre-tuned defaults (from V9 experiments)
    best_lgb_params = dict(
        n_estimators=3000, learning_rate=0.015, max_depth=9,
        num_leaves=63, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.1,
        scale_pos_weight=spw, random_state=42, verbosity=-1, n_jobs=-1
    )
    best_xgb_params = dict(
        n_estimators=3000, learning_rate=0.015, max_depth=8,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=3, gamma=0.05,
        reg_alpha=0.05, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, verbosity=0,
        eval_metric="auc", n_jobs=-1, early_stopping_rounds=100
    )
    best_cat_params = dict(
        iterations=3000, learning_rate=0.015, depth=8,
        l2_leaf_reg=3, border_count=128,
        auto_class_weights="Balanced",
        random_state=42, verbose=False, early_stopping_rounds=100
    )
    print("Using pre-tuned defaults")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Full Training with Time-Based CV Folds
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FULL TRAINING WITH TIME-BASED CV")
print("=" * 60)
print("Weighted ensemble: 0.4×LGB + 0.3×XGB + 0.3×CatBoost")

n_active_folds = len(time_folds_clean)
oof_lgb = np.zeros(len(y_clean))
oof_xgb = np.zeros(len(y_clean))
oof_cat = np.zeros(len(y_clean))
t_lgb = np.zeros(len(test))
t_xgb = np.zeros(len(test))
t_cat = np.zeros(len(test))
X_test = test[features].values

for fold_i, (tr_idx, val_idx) in enumerate(time_folds_clean):
    print(f"\n--- Time Fold {fold_i+1}/{n_active_folds} ---")
    Xtr, Xval = X_clean[tr_idx], X_clean[val_idx]
    ytr, yval = y_clean[tr_idx], y_clean[val_idx]
    wtr = sample_weights[tr_idx]

    # LightGBM
    m1 = lgb.LGBMClassifier(**best_lgb_params)
    m1.fit(Xtr, ytr, sample_weight=wtr,
           eval_set=[(Xval, yval)],
           callbacks=[lgb.early_stopping(100, verbose=False),
                      lgb.log_evaluation(period=-1)])
    oof_lgb[val_idx] = m1.predict_proba(Xval)[:, 1]
    t_lgb += m1.predict_proba(X_test)[:, 1] / n_active_folds
    print(f"  LGB  AUC={roc_auc_score(yval, oof_lgb[val_idx]):.4f}")

    # XGBoost
    m2 = xgb.XGBClassifier(**best_xgb_params)
    m2.fit(Xtr, ytr, sample_weight=wtr,
           eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val_idx] = m2.predict_proba(Xval)[:, 1]
    t_xgb += m2.predict_proba(X_test)[:, 1] / n_active_folds
    print(f"  XGB  AUC={roc_auc_score(yval, oof_xgb[val_idx]):.4f}")

    # CatBoost
    m3 = CatBoostClassifier(**best_cat_params)
    m3.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cat[val_idx] = m3.predict_proba(Xval)[:, 1]
    t_cat += m3.predict_proba(X_test)[:, 1] / n_active_folds
    print(f"  CAT  AUC={roc_auc_score(yval, oof_cat[val_idx]):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Weighted Ensemble & Threshold Optimization
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ENSEMBLE & THRESHOLD")
print("=" * 60)

auc_l = roc_auc_score(y_clean, oof_lgb)
auc_x = roc_auc_score(y_clean, oof_xgb)
auc_c = roc_auc_score(y_clean, oof_cat)
print(f"OOF AUC  LGB={auc_l:.4f}  XGB={auc_x:.4f}  CAT={auc_c:.4f}")

# Fixed weights: 0.4 LGB + 0.3 XGB + 0.3 CatBoost (as specified)
W_LGB, W_XGB, W_CAT = 0.40, 0.30, 0.30
oof_ens = oof_lgb * W_LGB + oof_xgb * W_XGB + oof_cat * W_CAT
t_ens   = t_lgb   * W_LGB + t_xgb   * W_XGB + t_cat   * W_CAT

auc_ens = roc_auc_score(y_clean, oof_ens)
print(f"Ensemble OOF AUC (0.4L+0.3X+0.3C): {auc_ens:.4f}")

# Also compute AUC-weighted blend and pick the better one
aucs = np.array([auc_l, auc_x, auc_c])
auto_w = (aucs - aucs.min()) / (aucs.max() - aucs.min() + 1e-9) + 0.5
auto_w /= auto_w.sum()
oof_ens_aw = oof_lgb * auto_w[0] + oof_xgb * auto_w[1] + oof_cat * auto_w[2]
auc_ens_aw = roc_auc_score(y_clean, oof_ens_aw)
print(f"AUC-weighted blend AUC:              {auc_ens_aw:.4f}  "
      f"(w: {auto_w[0]:.2f}/{auto_w[1]:.2f}/{auto_w[2]:.2f})")

if auc_ens_aw > auc_ens:
    print("→ Using AUC-weighted blend (higher OOF AUC)")
    oof_ens = oof_ens_aw
    t_ens   = t_lgb * auto_w[0] + t_xgb * auto_w[1] + t_cat * auto_w[2]
    auc_ens = auc_ens_aw
else:
    print("→ Keeping fixed (0.4/0.3/0.3) blend")

# F1-optimal threshold via precision-recall curve
prec, rec, thr_arr = precision_recall_curve(y_clean, oof_ens)
f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1_arr)
final_threshold = thr_arr[min(best_idx, len(thr_arr)-1)]
best_f1_oof = f1_arr[best_idx]

preds = (oof_ens > final_threshold).astype(int)
cm = confusion_matrix(y_clean, preds)
print(f"\nF1-optimal threshold: {final_threshold:.3f}")
print(f"  OOF AUC={auc_ens:.4f}  F1={best_f1_oof:.4f}  "
      f"P={precision_score(y_clean,preds):.4f}  R={recall_score(y_clean,preds):.4f}")
print(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

test["is_mule_prob"] = t_ens

# ══════════════════════════════════════════════════════════════════════════════
# 9. Error Analysis Loop — FP/FN Pattern Detection + Targeted Features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ERROR ANALYSIS LOOP")
print("=" * 60)

# Identify FP and FN on OOF predictions
oof_binary = (oof_ens > final_threshold).astype(int)
fp_mask = (oof_binary == 1) & (y_clean == 0)  # False positives
fn_mask = (oof_binary == 0) & (y_clean == 1)  # False negatives

print(f"FP: {fp_mask.sum():,}  FN: {fn_mask.sum():,}")

train_clean = train[keep_mask].reset_index(drop=True)

# Analyze which signal features are elevated in FP (accounts model wrongly calls mules)
# These are likely red herrings or legitimate accounts with unusual patterns
analysis_cols = [f for f in ["multi_signal_count", "signal_consistency", "txn_velocity",
                              "in_out_vol_ratio", "unique_recv_ratio", "burst_score",
                              "night_burst", "structure_intensity", "pass_through_ratio"]
                 if f in train_clean.columns]

if analysis_cols:
    fp_df = train_clean.iloc[fp_mask][analysis_cols].describe()
    fn_df = train_clean.iloc[fn_mask][analysis_cols].describe()
    tp_df = train_clean.iloc[(oof_binary == 1) & (y_clean == 1)][analysis_cols].describe()

    print("\nFP vs TP mean feature values (relative to TP):")
    for col in analysis_cols:
        tp_mean = tp_df.loc["mean", col] if col in tp_df.columns else 0
        fp_mean = fp_df.loc["mean", col] if col in fp_df.columns else 0
        fn_mean = fn_df.loc["mean", col] if col in fn_df.columns else 0
        print(f"  {col:30s}  TP={tp_mean:.3f}  FP={fp_mean:.3f}  FN={fn_mean:.3f}")

# Build targeted features from error analysis
print("\nBuilding error-targeted features...")
for df in [train, test]:
    # FP pattern: high velocity but low signal entropy (bursty legit)
    if "txn_velocity" in df.columns and "signal_entropy" in df.columns:
        df["fp_risk_score"] = df["txn_velocity"] / (df["signal_entropy"] + 0.01)

    # FN pattern: moderate signals across many dimensions (subtle mules)
    if "signal_consistency" in df.columns and "signal_variance" in df.columns:
        df["fn_subtle_score"] = df["signal_consistency"] * (1 - df["signal_variance"])

    # Accounts with high night burst + structuring → likely mule FN pattern
    if "night_burst" in df.columns and "near_threshold_pct" in df.columns:
        df["night_structure"] = df["night_burst"] * df["near_threshold_pct"]

    # Pass-through combined with high unique receiver ratio
    if "pass_through_ratio" in df.columns and "unique_recv_ratio" in df.columns:
        df["passthrough_x_receivers"] = df["pass_through_ratio"] * df["unique_recv_ratio"]

# Add error-targeted features to the feature set and retrain final models
err_new_features = [f for f in ["fp_risk_score", "fn_subtle_score",
                                  "night_structure", "passthrough_x_receivers"]
                     if f in train.columns]
if err_new_features:
    print(f"Added {len(err_new_features)} error-targeted features: {err_new_features}")
    # Re-build feature list and arrays
    features_v2 = [c for c in train.columns
                   if c not in drop_cols and train[c].nunique() > 1
                   and pd.api.types.is_numeric_dtype(train[c])]
    train[features_v2] = train[features_v2].fillna(train[features_v2].median())
    test[features_v2]  = test[features_v2].fillna(train[features_v2].median())

    X_clean_v2 = train[features_v2].values[keep_mask]
    X_test_v2  = test[features_v2].values

    # Quick retrain with error features on the best model config
    print("Quick retrain with error-targeted features (LGB only for speed)...")
    oof_lgb_v2 = np.zeros(len(y_clean))
    t_lgb_v2   = np.zeros(len(test))
    for tr_idx, val_idx in time_folds_clean:
        m_v2 = lgb.LGBMClassifier(**best_lgb_params)
        m_v2.fit(X_clean_v2[tr_idx], y_clean[tr_idx],
                 sample_weight=sample_weights[tr_idx],
                 eval_set=[(X_clean_v2[val_idx], y_clean[val_idx])],
                 callbacks=[lgb.early_stopping(80, verbose=False),
                             lgb.log_evaluation(period=-1)])
        oof_lgb_v2[val_idx] = m_v2.predict_proba(X_clean_v2[val_idx])[:, 1]
        t_lgb_v2 += m_v2.predict_proba(X_test_v2)[:, 1] / n_active_folds

    auc_v2 = roc_auc_score(y_clean, oof_lgb_v2)
    print(f"  LGB v2 OOF AUC: {auc_v2:.4f} (vs {auc_l:.4f} without error features)")
    if auc_v2 > auc_l:
        print("  ✅ Error features help → including in final blend")
        t_ens = t_lgb_v2 * W_LGB + t_xgb * W_XGB + t_cat * W_CAT
        test["is_mule_prob"] = t_ens
    else:
        print("  ⚠️  Error features hurt slightly → keeping original")
else:
    print("  No new error-targeted features available")

# ══════════════════════════════════════════════════════════════════════════════
# 10. RH Post-Filter (Learned)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RH POST-FILTER")
print("=" * 60)

rh_feat_cols = [f for f in [
    "multi_signal_count", "signal_consistency", "signal_variance",
    "signal_entropy", "signal_saturation", "too_perfect_score",
    "composite_score_v10", "branch_mule_rate_oof",
    "near_threshold_pct", "round_amount_pct", "pass_through_ratio"
] if f in train.columns]

mule_mask_full = y == 1
X_rh_tr = train[rh_feat_cols].values[mule_mask_full]
y_rh    = (oof_screen[mule_mask_full] > 0.40).astype(int)

t_filtered = test["is_mule_prob"].values.copy()

if y_rh.sum() > 10 and (1 - y_rh).sum() > 10:
    rh_model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                    max_depth=4, random_state=42, n_jobs=-1, verbosity=-1)
    rh_model.fit(X_rh_tr, y_rh)

    gate = final_threshold * 0.25
    high_mask = test["is_mule_prob"] > gate
    rh_prob = rh_model.predict_proba(test[rh_feat_cols].values[high_mask])[:, 1]

    high_idx = np.where(high_mask)[0]
    dampened = 0
    for i, idx in enumerate(high_idx):
        p = rh_prob[i]
        if p < 0.25:
            t_filtered[idx] *= 0.35; dampened += 1
        elif p < 0.45:
            t_filtered[idx] *= 0.65; dampened += 1

    print(f"RH filter: dampened {dampened} accounts")
    print(f"  Before: >{final_threshold:.3f} = {(test['is_mule_prob'] > final_threshold).sum():,}")
else:
    print("Insufficient RH training data — skipping")

test["is_mule_prob"] = t_filtered
print(f"  After:  >{final_threshold:.3f} = {(t_filtered > final_threshold).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Temporal Windows with Score Smoothing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEMPORAL WINDOWS — with Score Smoothing")
print("=" * 60)

parts = sorted(glob(f"{DATA_DIR}/transactions/batch-*/part_*.parquet"))
print(f"Transaction parts: {len(parts)}")

temporal_threshold = final_threshold * 0.20
high_prob_ids = set(test[test["is_mule_prob"] > temporal_threshold]["account_id"].tolist())
print(f"Accounts for temporal analysis: {len(high_prob_ids):,}")

# Build daily_vol AND daily_txn_count (for smoothing)
daily_vol = {}
daily_cnt = {}

for i, p in enumerate(parts):
    try:
        ds = pq.read_table(p,
                           columns=["account_id", "transaction_timestamp", "amount"],
                           filters=[("account_id", "in", list(high_prob_ids))])
        df = ds.to_pandas()
    except Exception:
        df = pd.read_parquet(p, columns=["account_id", "transaction_timestamp", "amount"])
        df = df[df["account_id"].isin(high_prob_ids)]
    if df.empty:
        continue
    df["ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["date"] = df["ts"].dt.date
    df["abs_amount"] = df["amount"].abs()

    for (aid, dt), grp in df.groupby(["account_id", "date"]):
        if aid not in daily_vol:
            daily_vol[aid] = {}
            daily_cnt[aid] = {}
        daily_vol[aid][dt] = daily_vol[aid].get(dt, 0) + grp["abs_amount"].sum()
        daily_cnt[aid][dt] = daily_cnt[aid].get(dt, 0) + len(grp)

    del df
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(parts)}] processed")
    gc.collect()

print(f"Built daily series for {len(daily_vol):,} accounts")


def v10_temporal_window_smoothed(vol_dict, cnt_dict=None):
    """
    V10 temporal window with score smoothing:

    1. Build normalized daily volume time series
    2. Apply rolling-mean smoothing (window=5 days)
    3. Find densest window [7,14,30,60,90,180,365] capturing >=55% volume
    4. Apply CDF trim only if window has >=8 active days AND spans >30d
       - Use 15%-85% trim (not 10-90%)
       - Minimum 7-day wide output
    5. Expand window ±3 days for boundary safety
    """
    if len(vol_dict) < 2:
        return "", ""

    # Build dense daily series (fill gaps with 0)
    all_dates = sorted(vol_dict.keys())
    if not all_dates:
        return "", ""

    import datetime
    d_start, d_end = all_dates[0], all_dates[-1]
    delta = (d_end - d_start).days
    if delta == 0:
        return f"{d_start}T00:00:00", f"{d_end}T23:59:59"

    # Full date range
    full_dates = [d_start + datetime.timedelta(days=i) for i in range(delta + 1)]
    full_vols  = np.array([vol_dict.get(d, 0.0) for d in full_dates], dtype=float)
    total = full_vols.sum()
    if total == 0:
        return "", ""

    # ── Step 2: Rolling mean smoothing (window=5 days) ────────────────────
    smoothed = pd.Series(full_vols).rolling(window=5, min_periods=1, center=True).mean().values

    n = len(full_dates)
    best_start, best_end = 0, n - 1
    found = False

    # ── Step 3: Find tightest window with >=55% of SMOOTHED volume ────────
    smooth_total = smoothed.sum()
    for window_days in [7, 14, 30, 60, 90, 180, 365]:
        best_wvol, b_s, b_e = 0, 0, 0
        for j in range(n):
            k = j
            while k < n and (full_dates[k] - full_dates[j]).days <= window_days:
                k += 1
            wvol = smoothed[j:k].sum()
            if wvol > best_wvol:
                best_wvol = wvol
                b_s, b_e = j, k - 1
        if smooth_total > 0 and best_wvol / smooth_total >= 0.55:
            best_start, best_end = b_s, b_e
            found = True
            break

    # Fallback: densest 90-day window
    if not found:
        best_wvol, bst, ben = 0, 0, 0
        for j in range(n):
            k = j
            while k < n and (full_dates[k] - full_dates[j]).days <= 90:
                k += 1
            wvol = smoothed[j:k].sum()
            if wvol > best_wvol:
                best_wvol = wvol
                bst, ben = j, k - 1
        best_start, best_end = bst, ben

    w_dates = full_dates[best_start:best_end+1]
    w_vols  = full_vols[best_start:best_end+1]
    span_days  = (w_dates[-1] - w_dates[0]).days if len(w_dates) > 1 else 0
    n_active   = int((w_vols > 0).sum())

    # ── Step 4: CDF trim only for large windows ────────────────────────────
    if n_active >= 8 and span_days > 30:
        w_arr = np.array(w_vols, dtype=float)
        w_cum = np.cumsum(w_arr)
        w_tot = w_cum[-1]
        if w_tot > 0:
            w_cdf = w_cum / w_tot
            s_idx = int(np.searchsorted(w_cdf, 0.15))
            e_idx = int(np.searchsorted(w_cdf, 0.85))
            s_idx = max(0, min(s_idx, len(w_dates) - 1))
            e_idx = max(0, min(e_idx, len(w_dates) - 1))
            if e_idx > s_idx:
                trimmed = w_dates[s_idx:e_idx+1]
                trimmed_span = (trimmed[-1] - trimmed[0]).days if len(trimmed) > 1 else 0
                if trimmed_span >= 7:
                    w_dates = trimmed

    if not w_dates:
        w_dates = full_dates[best_start:best_end+1]

    # ── Step 5: Expand ±3 days for boundary safety ────────────────────────
    out_start = w_dates[0] - pd.Timedelta(days=3) if hasattr(w_dates[0], 'date') else \
                (pd.Timestamp(w_dates[0]) - pd.Timedelta(days=3)).date()
    out_end   = w_dates[-1] + pd.Timedelta(days=3) if hasattr(w_dates[-1], 'date') else \
                (pd.Timestamp(w_dates[-1]) + pd.Timedelta(days=3)).date()

    return f"{out_start}T00:00:00", f"{out_end}T23:59:59"


temporal_windows = {}
for aid in high_prob_ids:
    if aid in daily_vol:
        s, e = v10_temporal_window_smoothed(daily_vol[aid], daily_cnt.get(aid))
        temporal_windows[aid] = (s, e)
    else:
        temporal_windows[aid] = ("", "")

n_with = sum(1 for s, e in temporal_windows.values() if s)
widths = [(pd.to_datetime(e) - pd.to_datetime(s)).days
          for s, e in temporal_windows.values() if s and e]
wa = np.array(widths) if widths else np.array([0])
print(f"Accounts with windows: {n_with:,}/{len(high_prob_ids):,}")
print(f"Window width: median={np.median(wa):.0f}d, mean={wa.mean():.0f}d, "
      f"p25={np.percentile(wa,25):.0f}d, p75={np.percentile(wa,75):.0f}d")

# ══════════════════════════════════════════════════════════════════════════════
# 12. Generate Submission
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERATING SUBMISSION V10")
print("=" * 60)

submission = pd.DataFrame({
    "account_id":      test["account_id"],
    "is_mule":         test["is_mule_prob"],
    "suspicious_start": "",
    "suspicious_end":   ""
})

for aid, (s, e) in temporal_windows.items():
    mask = submission["account_id"] == aid
    submission.loc[mask, "suspicious_start"] = s
    submission.loc[mask, "suspicious_end"]   = e

submission.to_csv("submission_v10.csv", index=False)

print(f"Submission: {submission.shape}")
print(f"  Mean prob:    {submission['is_mule'].mean():.4f}  (expected ~{global_mean:.4f})")
print(f"  >50% mule:    {(submission['is_mule']>0.5).sum():,}")
print(f"  >30% mule:    {(submission['is_mule']>0.3).sum():,}")
print(f"  >80% mule:    {(submission['is_mule']>0.8).sum():,}")
print(f"  With windows: {(submission['suspicious_start']!='').sum():,}")
print(f"\n──── FINAL OOF METRICS ────")
print(f"  AUC-ROC: {auc_ens:.6f}")
final_preds = (oof_ens > final_threshold).astype(int)
print(f"  F1:      {f1_score(y_clean, final_preds):.6f}")
print(f"  P:       {precision_score(y_clean, final_preds):.4f}")
print(f"  R:       {recall_score(y_clean, final_preds):.4f}")
print(f"  Thresh:  {final_threshold:.4f}")
print(f"\n✅ submission_v10.csv saved")
print(f"Total runtime: {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f} min")
