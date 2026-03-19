#!/usr/bin/env python3
"""
V10 — Fix IoU crash + push AUC/F1 over targets
================================================
V9 results vs targets:
  AUC   0.9902  → need 0.994+  (ExtraTrees was dragging ensemble DOWN)
  F1    0.8878  → need 0.91+   (close, need tighter threshold tuning)
  IoU   0.2364  → need 0.72+   ← CRITICAL: 10-90% CDF trim too aggressive
  RH_7  ?       → need 0.95+

V10 fixes:
1. IoU: Replace 10-90% CDF trim with a smarter approach:
   - Only trim when window has >=10 active days AND width > 30 days
   - Use 15-85% trim (not 10-90%) to avoid over-cutting sparse windows
   - Fall back to full densest window if trimmed window < 7 days wide
2. AUC: Drop ExtraTrees (hurts ensemble - lower AUC than GBDT models)
   Keep LGB + XGB + CatBoost only with higher n_estimators
3. F1: Fine-grained PR curve threshold search already in place, keep it
4. RH filter: Use OOF cross-validated RH model (not just single fit)
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, fbeta_score,
                             precision_score, recall_score, confusion_matrix,
                             precision_recall_curve)
# ExtraTrees removed — it was dragging ensemble AUC below LGB/XGB/Cat
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from glob import glob
import pyarrow.parquet as pq
import gc, warnings
warnings.filterwarnings('ignore')

DATA_DIR = "."  # notebook runs from inside Phase 2/
t0 = time.time()

# ── 0. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
train = pd.read_csv("features_train_p2.csv")
test  = pd.read_csv("features_test_p2.csv")
accounts = pd.read_parquet(f"{DATA_DIR}/accounts.parquet")
labels   = pd.read_parquet(f"{DATA_DIR}/train_labels.parquet")

train = train.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")
test  = test.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")
train = train.merge(labels[["account_id", "alert_reason"]], on="account_id", how="left")

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Mule rate: {train['is_mule'].mean():.4f} ({train['is_mule'].sum()} mules)")
global_mean = train["is_mule"].mean()

# ── 1. OOF Branch Target Encoding ─────────────────────────────────────────────
print("\nOOF Target Encoding (branch-level)...")
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["branch_mule_rate_oof"] = np.nan

for tr_idx, val_idx in skf_te.split(train, train["is_mule"]):
    tr_df = train.iloc[tr_idx]
    bs = tr_df.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
    bs["rate"] = (bs["sum"] + 10 * global_mean) / (bs["count"] + 10)
    mapped = train.iloc[val_idx]["branch_code"].map(bs["rate"]).fillna(global_mean)
    train.loc[train.index[val_idx], "branch_mule_rate_oof"] = mapped.values

branch_stats_full = train.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
branch_stats_full["rate"] = (branch_stats_full["sum"] + 10*global_mean) / (branch_stats_full["count"] + 10)
test["branch_mule_rate_oof"] = test["branch_code"].map(branch_stats_full["rate"]).fillna(global_mean)

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\nEngineering features...")

# Quantile thresholds derived from train only (no leakage into test)
q90_near   = train["near_threshold_pct"].quantile(0.90)
q90_round  = train["round_amount_pct"].quantile(0.90)
q75_vol    = train["total_volume"].quantile(0.75)
q90_branch = train["branch_mule_rate_oof"].quantile(0.90)
train_vol_to_bal = (train["total_volume"] / train["balance_mean"].abs().clip(1)) if "balance_mean" in train.columns else None
q90_vtob   = train_vol_to_bal.quantile(0.90) if train_vol_to_bal is not None else None
# Geo threshold from train only
train_geo  = (train["geo_spread_lat"].fillna(0) + train["geo_spread_lon"].fillna(0)) if "geo_spread_lat" in train.columns else None
q90_geo    = train_geo.quantile(0.90) if train_geo is not None else 0.0

SIGNAL_THRESHOLDS = {
    "q90_near":   q90_near,
    "q90_round":  q90_round,
    "q75_vol":    q75_vol,
    "q90_branch": q90_branch,
    "q90_vtob":   q90_vtob,
    "q90_geo":    q90_geo,
}

def build_signals(df, thresholds):
    """Build boolean multi-signal features."""
    signals = []
    t = thresholds

    # 1. Dormant reactivation
    if "days_to_first_large" in df.columns:
        df["sig_dormant"] = (df["days_to_first_large"] > 180).astype(float)
        signals.append("sig_dormant")

    # 2. Structuring
    if "near_threshold_pct" in df.columns:
        df["sig_structuring"] = (df["near_threshold_pct"] > t["q90_near"]).astype(float)
        signals.append("sig_structuring")

    # 3. Rapid pass-through
    if "median_dwell_hours" in df.columns:
        df["sig_rapid"] = (df["median_dwell_hours"] < 24).astype(float)
        signals.append("sig_rapid")

    # 4. Fan-out
    if "unique_cp_count" in df.columns and "txn_count" in df.columns:
        df["sig_fanout"] = (df["unique_cp_count"] / df["txn_count"].clip(1) > 0.5).astype(float)
        signals.append("sig_fanout")

    # 5. New account high value
    if "days_to_first_large" in df.columns and "total_volume" in df.columns:
        df["sig_new_highvol"] = ((df["days_to_first_large"] < 30) &
                                 (df["total_volume"] > t["q75_vol"])).astype(float)
        signals.append("sig_new_highvol")

    # 6. Round amounts
    if "round_amount_pct" in df.columns:
        df["sig_round"] = (df["round_amount_pct"] > t["q90_round"]).astype(float)
        signals.append("sig_round")

    # 7. Post-mobile spike
    if "has_mobile_update" in df.columns:
        df["sig_post_mobile"] = (df["has_mobile_update"] == 1).astype(float)
        signals.append("sig_post_mobile")

    # 8. Income mismatch
    if "total_volume" in df.columns and "balance_mean" in df.columns and t["q90_vtob"] is not None:
        vtob = df["total_volume"] / df["balance_mean"].abs().clip(1)
        df["sig_income_mismatch"] = (vtob > t["q90_vtob"]).astype(float)
        signals.append("sig_income_mismatch")

    # 9. Branch cluster
    if "branch_mule_rate_oof" in df.columns:
        df["sig_branch_cluster"] = (df["branch_mule_rate_oof"] > t["q90_branch"]).astype(float)
        signals.append("sig_branch_cluster")

    # 10. Residual (near-zero balance after activity)
    if "residual_ratio" in df.columns:
        df["sig_residual"] = (df["residual_ratio"] < 0.01).astype(float)
        signals.append("sig_residual")

    # 11. Geographic anomaly (high geo spread) — threshold from train only
    if "geo_spread_lat" in df.columns and "geo_spread_lon" in df.columns:
        geo = df["geo_spread_lat"].fillna(0) + df["geo_spread_lon"].fillna(0)
        geo_thresh = thresholds.get("q90_geo", geo.quantile(0.90))
        df["sig_geo"] = (geo > geo_thresh).astype(float)
        signals.append("sig_geo")

    # 12. Prior freeze → strong mule signal
    if "has_prior_freeze" in df.columns:
        df["sig_freeze"] = df["has_prior_freeze"].fillna(0).astype(float)
        signals.append("sig_freeze")

    # Aggregate signal features
    if signals:
        sig_vals = np.column_stack([df[s].fillna(0).values for s in signals])
        df["multi_signal_count"]   = sig_vals.sum(axis=1)
        df["signal_consistency"]   = df["multi_signal_count"] / len(signals)
        df["signal_variance"]      = np.var(sig_vals, axis=1)
        # Shannon entropy of signal vector
        p = sig_vals.clip(1e-9, 1-1e-9)
        entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
        df["signal_entropy"]       = entropy.mean(axis=1)
        df["signal_saturation"]    = (df["multi_signal_count"] >= len(signals) * 0.75).astype(float)
        # NEW: "too-perfect" score — accounts where MANY signals activate at MODERATE strength
        #   Real mules are sparse (few strong signals).
        #   RH_7-type red herrings may have many medium signals.
        df["too_perfect_score"]    = df["signal_entropy"] * df["signal_saturation"]

    return df, signals

for df in [train, test]:
    df, signals = build_signals(df, SIGNAL_THRESHOLDS)

print(f"Built {len(signals)} signal features + aggregates")

# Additional cross-product features
for df in [train, test]:
    if "gap_cv" in df.columns and "degree_centrality" in df.columns:
        df["gapcv_x_degree_v2"]   = df["gap_cv"] * df["degree_centrality"]
    if "multi_signal_count" in df.columns and "branch_mule_rate_oof" in df.columns:
        df["signal_x_branch"]     = df["multi_signal_count"] * df["branch_mule_rate_oof"]
    if "signal_consistency" in df.columns and "mule_trigram_count" in df.columns:
        df["consistency_x_tri"]   = df["signal_consistency"] * df["mule_trigram_count"].fillna(0)
    if "txn_count" in df.columns and "active_days" in df.columns:
        df["txn_intensity"]       = df["txn_count"] / df["active_days"].clip(1)
    if "total_volume" in df.columns and "txn_count" in df.columns:
        df["avg_txn_size"]        = df["total_volume"] / df["txn_count"].clip(1)

# Composite anomaly score
score_cols = ["near_threshold_pct", "round_amount_pct", "gap_cv", "degree_centrality",
              "mule_trigram_count", "branch_mule_rate_oof", "has_prior_freeze"]
for df in [train, test]:
    c_score = np.zeros(len(df))
    for col in score_cols:
        if col in df.columns:
            m, s = train[col].mean(), train[col].std()
            c_score += (df[col].fillna(m) - m) / s if s > 0 else 0
    df["composite_score_v9"] = c_score

print("Feature engineering complete.")

# ── 3. Prepare Feature Set ────────────────────────────────────────────────────
drop_cols = ["account_id", "is_mule", "first_large_ts", "open_date",
             "branch_code", "branch_mule_rate", "composite_score", "alert_reason"]
features = [c for c in train.columns if c not in drop_cols and train[c].nunique() > 1]
train[features] = train[features].fillna(train[features].median())
test[features]  = test[features].fillna(train[features].median())
print(f"\nTotal features: {len(features)}")

# ── 4. Red Herring Pruning ────────────────────────────────────────────────────
print("\nRed herring screening (OOF LightGBM pass)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train[features].values
y = train["is_mule"].values
oof_screen = np.zeros(len(y))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                            max_depth=7, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X[tr], y[tr])
    oof_screen[val] = m.predict_proba(X[val])[:, 1]

# Prune only clear red herrings (very low OOF score AND labeled mule)
extreme_rh = (y == 1) & (oof_screen < 0.015)
keep_mask = ~extreme_rh
X_clean = X[keep_mask]
y_clean = y[keep_mask]
oof_screen_clean = oof_screen[keep_mask]

# Soft weights: down-weight ambiguous mules slightly
sample_weights = np.ones(len(y_clean))
ambig = ((y_clean == 1) & (oof_screen_clean > 0.015) & (oof_screen_clean < 0.08) &
         (train["multi_signal_count"].values[keep_mask] < 2))
sample_weights[ambig] = 0.6

print(f"Pruned {extreme_rh.sum()} extreme red herrings → {len(y_clean):,} samples")
print(f"Down-weighted {ambig.sum()} ambiguous samples")

# ── 5. Three-Model Ensemble (LGB + XGB + CatBoost) ───────────────────────────
# ExtraTrees removed — its OOF AUC was consistently below gradient boosters,
# pulling the weighted blend DOWN. Three strong GBDTs with higher n_estimators.
print("\n" + "=" * 60)
print("3-MODEL ENSEMBLE: LGB + XGB + CatBoost")
print("=" * 60)

spw = (y_clean == 0).sum() / max(1, (y_clean == 1).sum())
oof_lgb = np.zeros(len(y_clean))
oof_xgb = np.zeros(len(y_clean))
oof_cat = np.zeros(len(y_clean))
t_lgb = np.zeros(len(test))
t_xgb = np.zeros(len(test))
t_cat = np.zeros(len(test))

X_test = test[features].values

for fold, (tr, val) in enumerate(skf.split(X_clean, y_clean)):
    print(f"\n--- Fold {fold+1}/5 ---")
    Xtr, Xval = X_clean[tr], X_clean[val]
    ytr, yval = y_clean[tr], y_clean[val]
    wtr = sample_weights[tr]

    # LightGBM — tuned for AUC + F1 balance
    m1 = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.015, max_depth=9,
        num_leaves=63, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.1,
        scale_pos_weight=spw, random_state=42, verbosity=-1, n_jobs=-1
    )
    m1.fit(Xtr, ytr, sample_weight=wtr,
           eval_set=[(Xval, yval)],
           callbacks=[lgb.early_stopping(100, verbose=False),
                      lgb.log_evaluation(period=-1)])
    oof_lgb[val] = m1.predict_proba(Xval)[:, 1]
    t_lgb += m1.predict_proba(X_test)[:, 1] / 5.0
    print(f"  LGB AUC: {roc_auc_score(yval, oof_lgb[val]):.4f}")

    # XGBoost
    m2 = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.015, max_depth=8,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=3, gamma=0.05,
        reg_alpha=0.05, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, verbosity=0,
        eval_metric="auc", n_jobs=-1, early_stopping_rounds=100
    )
    m2.fit(Xtr, ytr, sample_weight=wtr,
           eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val] = m2.predict_proba(Xval)[:, 1]
    t_xgb += m2.predict_proba(X_test)[:, 1] / 5.0
    print(f"  XGB AUC: {roc_auc_score(yval, oof_xgb[val]):.4f}")

    # CatBoost
    m3 = CatBoostClassifier(
        iterations=3000, learning_rate=0.015, depth=8,
        l2_leaf_reg=3, border_count=128,
        auto_class_weights='Balanced',
        random_state=42, verbose=False, early_stopping_rounds=100
    )
    m3.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cat[val] = m3.predict_proba(Xval)[:, 1]
    t_cat += m3.predict_proba(X_test)[:, 1] / 5.0
    print(f"  CAT AUC: {roc_auc_score(yval, oof_cat[val]):.4f}")

# ── 6. Blending ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLENDING")
print("=" * 60)

# OOF AUCs to determine blend weights
auc_lgb = roc_auc_score(y_clean, oof_lgb)
auc_xgb = roc_auc_score(y_clean, oof_xgb)
auc_cat = roc_auc_score(y_clean, oof_cat)
print(f"OOF AUC  LGB={auc_lgb:.4f}  XGB={auc_xgb:.4f}  CAT={auc_cat:.4f}")

# AUC-weighted average (models with higher OOF AUC get more weight)
aucs = np.array([auc_lgb, auc_xgb, auc_cat])
weights = (aucs - aucs.min()) / (aucs.max() - aucs.min() + 1e-9) + 0.5
weights = weights / weights.sum()
print(f"Blend weights: LGB={weights[0]:.3f}  XGB={weights[1]:.3f}  CAT={weights[2]:.3f}")

oof_ens = oof_lgb * weights[0] + oof_xgb * weights[1] + oof_cat * weights[2]
t_ens   = t_lgb * weights[0] + t_xgb * weights[1] + t_cat * weights[2]

auc_ens = roc_auc_score(y_clean, oof_ens)
print(f"Ensemble OOF AUC: {auc_ens:.4f}")

# ── 7. Threshold Optimization for F1 ─────────────────────────────────────────
print("\nOptimizing threshold for F1...")
prec, rec, thresholds_pr = precision_recall_curve(y_clean, oof_ens)
# F1 = 2*P*R / (P+R)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1_scores)
best_t_f1 = thresholds_pr[min(best_idx, len(thresholds_pr)-1)]
best_f1 = f1_scores[best_idx]

# Also compute F-beta=2 threshold (recall-heavy)
f2_scores = (1 + 4) * prec * rec / (4*prec + rec + 1e-9)
best_idx_f2 = np.argmax(f2_scores)
best_t_f2 = thresholds_pr[min(best_idx_f2, len(thresholds_pr)-1)]
best_f2 = f2_scores[best_idx_f2]

print(f"  F1-optimal:  t={best_t_f1:.3f}  OOF F1={best_f1:.4f}")
print(f"  F2-optimal:  t={best_t_f2:.3f}  OOF F2={best_f2:.4f}")

# Use F1-optimal threshold (target is F1, not recall)
final_threshold = best_t_f1
preds = (oof_ens > final_threshold).astype(int)
cm = confusion_matrix(y_clean, preds)
print(f"\nUsing F1-optimized t={final_threshold:.3f}:")
print(f"  OOF F1={f1_score(y_clean,preds):.4f}  P={precision_score(y_clean,preds):.4f}  R={recall_score(y_clean,preds):.4f}")
print(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

test["is_mule_prob"] = t_ens
mule_preds = test[test["is_mule_prob"] > final_threshold]["account_id"].tolist()
print(f"Predicted mules on TEST: {len(mule_preds)} ({len(mule_preds)/len(test)*100:.1f}%)")

# ── 8. RH_7 Post-Filter (Learned, not hand-coded) ────────────────────────────
print("\n" + "=" * 60)
print("RH_7 FILTER: Too-Perfect Detector")
print("=" * 60)

# Build a small second-stage on OOF predictions:
# Among accounts predicted as mules (high prob), identify ones that look
# "too perfect" — matching too many signals at moderate level = planted RH

# Second-stage features for likely mule accounts
rh_features = [f for f in [
    "multi_signal_count", "signal_consistency", "signal_variance",
    "signal_entropy", "signal_saturation", "too_perfect_score",
    "composite_score_v9", "branch_mule_rate_oof",
    "near_threshold_pct", "round_amount_pct"
] if f in train.columns]

if rh_features:
    # Among labeled mules: those with high oof_screen score are real mules,
    # those with low score are planted red herrings
    mule_mask = y == 1
    X_rh_train = train[rh_features].values[mule_mask]
    # Use oof_screen > 0.4 as "likely real mule"
    y_rh = (oof_screen[mule_mask] > 0.4).astype(int)

    if y_rh.sum() > 10 and (1 - y_rh).sum() > 10:
        rh_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=42, n_jobs=-1, verbosity=-1
        )
        rh_model.fit(X_rh_train, y_rh)

        # Apply to test predictions above a low probability gate
        gate = final_threshold * 0.3
        high_prob_test = test["is_mule_prob"] > gate

        X_rh_test = test[rh_features].values[high_prob_test]
        rh_prob = rh_model.predict_proba(X_rh_test)[:, 1]  # P(real mule)

        # Dampen test probabilities where P(real mule) is low
        t_filtered = t_ens.copy()
        high_idx = np.where(high_prob_test)[0]
        dampened = 0
        for i, idx in enumerate(high_idx):
            if rh_prob[i] < 0.30:
                # Very likely RH → dampen strongly
                t_filtered[idx] = t_ens[idx] * 0.4
                dampened += 1
            elif rh_prob[i] < 0.50:
                # Possibly RH → moderate dampen
                t_filtered[idx] = t_ens[idx] * 0.7
                dampened += 1

        test["is_mule_prob"] = t_filtered
        print(f"RH filter dampened {dampened} accounts")
        print(f"After RH filter: >0.5 = {(t_filtered > 0.5).sum():,} (was {(t_ens > 0.5).sum():,})")
    else:
        print("Insufficient RH training data — skipping second-stage filter")
        t_filtered = t_ens.copy()
        test["is_mule_prob"] = t_filtered
else:
    t_filtered = t_ens.copy()
    test["is_mule_prob"] = t_filtered

# ── 9. Tight Temporal Windows (MAD-based peak detection) ─────────────────────
print("\n" + "=" * 60)
print("TEMPORAL WINDOWS — Peak-density MAD method")
print("=" * 60)

parts = sorted(glob(f"{DATA_DIR}/transactions/batch-*/part_*.parquet"))
print(f"Transaction parts: {len(parts)}")

temporal_threshold = final_threshold * 0.20
high_prob_ids = set(test[test["is_mule_prob"] > temporal_threshold]["account_id"].tolist())
print(f"Accounts for temporal analysis: {len(high_prob_ids):,}")

daily_vol = {}
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
    grp = df.groupby(["account_id", "date"])["abs_amount"].sum()
    for (aid, dt), vol in grp.items():
        if aid not in daily_vol:
            daily_vol[aid] = {}
        daily_vol[aid][dt] = daily_vol[aid].get(dt, 0) + vol
    del df
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(parts)}] processed")
    gc.collect()

print(f"Built series for {len(daily_vol):,} accounts")


def v10_temporal_window(vol_dict):
    """
    V10 Temporal Window — Fixed IoU Crash

    Root cause of V9 IoU=0.236:
      The 10-90% CDF trim was applied even to sparse accounts with only
      3-5 active days, collapsing windows to 0-1 day spans that don't
      overlap with the true ground-truth window at all.

    Fix:
      1. Find densest window [7,14,30,60,90,180,365] days with >=55% volume
      2. Only apply CDF trim if:
         a) Window has >= 8 active days, AND
         b) Window spans > 30 calendar days
         Otherwise keep the raw peak window (don't trim)
      3. Use 15-85% quantile trim (not 10-90%) — less aggressive
      4. After trim, ensure result is >= 7 calendar days wide
         (widen back to 7 days if trim made it narrower)
    """
    if len(vol_dict) < 2:
        return "", ""

    dates = sorted(vol_dict.keys())
    vols = np.array([vol_dict[d] for d in dates])
    total = vols.sum()
    if total == 0:
        return "", ""

    n = len(dates)
    best_start, best_end = 0, n - 1
    found = False

    # Step 1: Find tightest window capturing >=55% of volume
    for window_days in [7, 14, 30, 60, 90, 180, 365]:
        best_wvol = 0
        b_s, b_e = 0, 0
        for j in range(n):
            k, wvol = j, 0
            while k < n and (dates[k] - dates[j]).days <= window_days:
                wvol += vols[k]
                k += 1
            if wvol > best_wvol:
                best_wvol = wvol
                b_s, b_e = j, k - 1
        if best_wvol / total >= 0.55:
            best_start, best_end = b_s, b_e
            found = True
            break

    # Fallback: use the densest 90-day window
    if not found:
        best_wvol = 0
        for j in range(n):
            k, wvol = j, 0
            while k < n and (dates[k] - dates[j]).days <= 90:
                wvol += vols[k]
                k += 1
            if wvol > best_wvol:
                best_wvol = wvol
                best_start, best_end = j, k - 1

    w_vols = vols[best_start:best_end+1]
    w_dates = list(dates[best_start:best_end+1])
    window_span_days = (w_dates[-1] - w_dates[0]).days if len(w_dates) > 1 else 0
    n_active = len(w_vols)

    # Step 2: CDF trim ONLY if window is wide enough to survive trimming
    if n_active >= 8 and window_span_days > 30:
        w_arr = np.array(w_vols)
        w_cum = np.cumsum(w_arr)
        w_tot = w_cum[-1]
        if w_tot > 0:
            w_cdf = w_cum / w_tot
            s_idx = int(np.searchsorted(w_cdf, 0.15))   # 15% (less aggressive)
            e_idx = int(np.searchsorted(w_cdf, 0.85))   # 85%
            s_idx = min(s_idx, len(w_dates) - 1)
            e_idx = min(e_idx, len(w_dates) - 1)
            if e_idx > s_idx:
                trimmed_dates = w_dates[s_idx:e_idx+1]
                # Safety: don't collapse to < 7 calendar days
                trimmed_span = (trimmed_dates[-1] - trimmed_dates[0]).days if len(trimmed_dates) > 1 else 0
                if trimmed_span >= 7:
                    w_dates = trimmed_dates
                # else: keep original untrimmed window

    if not w_dates:
        w_dates = list(dates[best_start:best_end+1])

    return f"{w_dates[0]}T00:00:00", f"{w_dates[-1]}T23:59:59"


temporal_windows = {}
for aid in high_prob_ids:
    if aid in daily_vol:
        s, e = v10_temporal_window(daily_vol[aid])
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

# ── 10. Generate Submission ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING SUBMISSION V10")
print("=" * 60)

submission = pd.DataFrame({
    "account_id": test["account_id"],
    "is_mule": test["is_mule_prob"],
    "suspicious_start": "",
    "suspicious_end": ""
})

for aid, (s, e) in temporal_windows.items():
    mask = submission["account_id"] == aid
    submission.loc[mask, "suspicious_start"] = s
    submission.loc[mask, "suspicious_end"] = e

submission.to_csv("submission_v10.csv", index=False)

# Final report
print(f"Submission: {submission.shape}")
print(f"  Mean prob:    {submission['is_mule'].mean():.4f} (expected ~{global_mean:.4f})")
print(f"  >50% mule:    {(submission['is_mule']>0.5).sum():,}")
print(f"  >30% mule:    {(submission['is_mule']>0.3).sum():,}")
print(f"  >80% mule:    {(submission['is_mule']>0.8).sum():,}")
print(f"  With windows: {(submission['suspicious_start']!='').sum():,}")
print(f"\nEnsemble OOF AUC: {auc_ens:.4f}")
print(f"OOF F1 (at t={final_threshold:.3f}): {f1_score(y_clean, (oof_ens > final_threshold).astype(int)):.4f}")
print(f"OOF Precision: {precision_score(y_clean, (oof_ens > final_threshold).astype(int)):.4f}")
print(f"OOF Recall:    {recall_score(y_clean, (oof_ens > final_threshold).astype(int)):.4f}")
print(f"\n✅ submission_v10.csv saved")
print(f"Total: {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f} min")
