# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature Pipeline V2 + V15 Model
#
# Single-pass transaction processing with NO iterrows.
# All features computed via chunked `groupby` → `concat` → final `groupby`.
#
# New features targeting 3 missing mule patterns:
# 1. **MCC anomaly** (z-score per MCC code)
# 2. **Pass-through proxy** (credit-debit matching within 48h)
# 3. **Salary pattern detection** (regular monthly credits)
# 4. **Burst/dormant metrics** (activity acceleration)
# 5. **Channel entropy** (transaction channel diversity)
# 6. **Structuring refinement** (amounts in [48K-50K])

# %% [markdown]
# ## 0 — Setup

# %%
import pandas as pd
import numpy as np
import time
from glob import glob
import pyarrow.parquet as pq
import gc, warnings, os
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set DATA_DIR to the directory of the script itself
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

print(f"Loading existing features from {DATA_DIR}...")
train_base = pd.read_csv(os.path.join(DATA_DIR, "features_train_p2.csv"))
test_base = pd.read_csv(os.path.join(DATA_DIR, "features_test_p2.csv"))
accounts = pd.read_parquet(os.path.join(DATA_DIR, "accounts.parquet"))

train_ids = set(train_base["account_id"])
test_ids = set(test_base["account_id"])
all_ids = train_ids | test_ids
print(f"Train: {len(train_ids):,} | Test: {len(test_ids):,} | Total: {len(all_ids):,}")

# %% [markdown]
# ## 1 — Pass 1: MCC Global Stats (need this before per-account z-scores)
#
# Quick scan to build mean/std per MCC code across ALL transactions.

# %%
print("=" * 60)
print("PASS 1: MCC Global Stats")
print("=" * 60)

parts = sorted(glob(f"{DATA_DIR}/transactions/batch-*/part_*.parquet"))
print(f"Parts: {len(parts)}")

mcc_stats_chunks = []
for i, p in enumerate(parts):
    try:
        ds = pq.read_table(p, columns=["mcc_code", "amount"])
        df = ds.to_pandas()
    except:
        continue
    df["abs_amount"] = df["amount"].abs()
    chunk = df.groupby("mcc_code")["abs_amount"].agg(["sum", "count", "mean"]).reset_index()
    chunk.columns = ["mcc_code", "mcc_vol_sum", "mcc_count", "mcc_mean_chunk"]
    # Also need sum of squares for std
    df["sq"] = df["abs_amount"] ** 2
    sq_chunk = df.groupby("mcc_code")["sq"].sum().reset_index()
    sq_chunk.columns = ["mcc_code", "mcc_sq_sum"]
    chunk = chunk.merge(sq_chunk, on="mcc_code")
    mcc_stats_chunks.append(chunk)
    del df
    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{len(parts)}] ({time.time()-t0:.0f}s)")
    gc.collect()

mcc_all = pd.concat(mcc_stats_chunks)
mcc_global = mcc_all.groupby("mcc_code").agg({
    "mcc_vol_sum": "sum",
    "mcc_count": "sum",
    "mcc_sq_sum": "sum"
}).reset_index()
mcc_global["mcc_mean"] = mcc_global["mcc_vol_sum"] / mcc_global["mcc_count"]
mcc_global["mcc_var"] = (mcc_global["mcc_sq_sum"] / mcc_global["mcc_count"]) - mcc_global["mcc_mean"]**2
mcc_global["mcc_std"] = np.sqrt(mcc_global["mcc_var"].clip(0))
mcc_lookup = dict(zip(mcc_global["mcc_code"], zip(mcc_global["mcc_mean"], mcc_global["mcc_std"])))
print(f"MCC stats for {len(mcc_lookup)} codes ({(time.time()-t0)/60:.1f}min)")
del mcc_stats_chunks, mcc_all
gc.collect()

# %% [markdown]
# ## 2 — Pass 2: ALL per-account features in one pass
#
# For each partition:
# 1. Filter to our accounts
# 2. Compute groupby aggregates
# 3. Append to list
# 4. At end: concat + final groupby

# %%
print("=" * 60)
print("PASS 2: Per-Account Features (single pass)")
print("=" * 60)

# Accumulators — one list per feature group, each entry is a small DataFrame
feat_chunks = []

for i, p in enumerate(parts):
    try:
        ds = pq.read_table(p, columns=["account_id", "transaction_timestamp", "amount",
                                        "mcc_code", "channel", "counterparty_id", "txn_type"],
                           filters=[("account_id", "in", list(all_ids))])
        df = ds.to_pandas()
    except:
        df = pd.read_parquet(p, columns=["account_id", "transaction_timestamp", "amount",
                                          "mcc_code", "channel", "counterparty_id", "txn_type"])
        df = df[df["account_id"].isin(all_ids)]

    if df.empty:
        if (i+1) % 100 == 0: print(f"  [{i+1}/{len(parts)}] (empty)")
        continue

    df["ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["abs_amount"] = df["amount"].abs()
    df["date"] = df["ts"].dt.date
    df["month"] = df["ts"].dt.to_period("M")
    df["day_of_month"] = df["ts"].dt.day
    df["is_weekend"] = df["ts"].dt.dayofweek >= 5

    # --- Feature Group 1: MCC Anomaly z-scores ---
    df["mcc_info"] = df["mcc_code"].map(mcc_lookup)
    df["mcc_m"] = df["mcc_info"].apply(lambda x: x[0] if x is not None else np.nan)
    df["mcc_s"] = df["mcc_info"].apply(lambda x: x[1] if x is not None else np.nan)
    df["mcc_zscore"] = ((df["abs_amount"] - df["mcc_m"]) / df["mcc_s"].clip(1)).fillna(0)

    mcc_feat = df.groupby("account_id").agg(
        mcc_zscore_max=("mcc_zscore", "max"),
        mcc_zscore_mean=("mcc_zscore", "mean"),
        mcc_zscore_gt2=("mcc_zscore", lambda x: (x > 2).sum()),
        txn_count_chunk=("amount", "count"),
        total_vol_chunk=("abs_amount", "sum"),
    ).reset_index()

    # --- Feature Group 2: Structuring refinement ---
    struct_48_50 = df[(df["abs_amount"] >= 48000) & (df["abs_amount"] < 50000)]
    struct_45_50 = df[(df["abs_amount"] >= 45000) & (df["abs_amount"] < 50000)]
    s1 = struct_48_50.groupby("account_id").size().reset_index(name="struct_48_50_count")
    s2 = struct_45_50.groupby("account_id").size().reset_index(name="struct_45_50_count")
    mcc_feat = mcc_feat.merge(s1, on="account_id", how="left")
    mcc_feat = mcc_feat.merge(s2, on="account_id", how="left")

    # --- Feature Group 3: Channel entropy ingredients ---
    if "channel" in df.columns:
        ch_valid = df[df["channel"].notna()]
        if len(ch_valid) > 0:
            ch_counts = ch_valid.groupby(["account_id", "channel"]).size().reset_index(name="ch_cnt")
            # Compute channel counts per account for later entropy calc
            ch_total = ch_valid.groupby("account_id").size().reset_index(name="ch_total")
            ch_nunique = ch_valid.groupby("account_id")["channel"].nunique().reset_index(name="ch_nunique")
            mcc_feat = mcc_feat.merge(ch_total, on="account_id", how="left")
            mcc_feat = mcc_feat.merge(ch_nunique, on="account_id", how="left")

    # --- Feature Group 4: Weekend + large_round ---
    wk = df[df["is_weekend"]].groupby("account_id").size().reset_index(name="weekend_cnt")
    lr = df[(df["abs_amount"] >= 10000) & (df["abs_amount"] % 1000 == 0)]
    lr_cnt = lr.groupby("account_id").size().reset_index(name="large_round_cnt") if len(lr) > 0 else pd.DataFrame(columns=["account_id","large_round_cnt"])

    # Exact round amounts (1K, 5K, 10K, 50K)
    exact_round = df[df["abs_amount"].isin([1000,5000,10000,50000])]
    er_cnt = exact_round.groupby("account_id").size().reset_index(name="exact_round_cnt") if len(exact_round) > 0 else pd.DataFrame(columns=["account_id","exact_round_cnt"])

    mcc_feat = mcc_feat.merge(wk, on="account_id", how="left")
    mcc_feat = mcc_feat.merge(lr_cnt, on="account_id", how="left")
    mcc_feat = mcc_feat.merge(er_cnt, on="account_id", how="left")

    # --- Feature Group 5: Salary pattern (credits by month) ---
    credits = df[df["amount"] > 0]
    if len(credits) > 0:
        monthly_credits = credits.groupby(["account_id", "month"]).agg(
            month_credit_sum=("abs_amount", "sum"),
            month_credit_count=("abs_amount", "count")
        ).reset_index()
        # Monthly credit stats per account
        sal_stats = monthly_credits.groupby("account_id").agg(
            monthly_credit_mean=("month_credit_sum", "mean"),
            monthly_credit_std=("month_credit_sum", "std"),
            months_with_credits=("month", "count")
        ).reset_index()
        sal_stats["monthly_credit_cv"] = sal_stats["monthly_credit_std"].fillna(0) / sal_stats["monthly_credit_mean"].clip(1)
        mcc_feat = mcc_feat.merge(sal_stats[["account_id", "monthly_credit_cv", "months_with_credits"]], on="account_id", how="left")

    # --- Feature Group 6: Month boundary (salary cycle exploitation) ---
    month_boundary = df[(df["day_of_month"] <= 5) | (df["day_of_month"] >= 26)]
    mb_cnt = month_boundary.groupby("account_id").size().reset_index(name="month_boundary_cnt")
    mcc_feat = mcc_feat.merge(mb_cnt, on="account_id", how="left")

    # --- Feature Group 7: Max single txn ---
    max_txn = df.groupby("account_id")["abs_amount"].max().reset_index(name="max_txn_chunk")
    mcc_feat = mcc_feat.merge(max_txn, on="account_id", how="left")

    # --- Feature Group 8: Debit count (for reversal ratio) ---
    debits = df[df["amount"] < 0]
    if len(debits) > 0:
        deb = debits.groupby("account_id").size().reset_index(name="neg_amt_cnt")
        mcc_feat = mcc_feat.merge(deb, on="account_id", how="left")

    feat_chunks.append(mcc_feat)

    del df
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(parts)}] ({time.time()-t0:.0f}s)")
    gc.collect()

print(f"Pass 2 done in {(time.time()-t0)/60:.1f}min")

# %% [markdown]
# ## 3 — Aggregate chunks into final per-account features

# %%
print("Aggregating chunks...")
all_feats = pd.concat(feat_chunks, ignore_index=True)
print(f"Raw chunks: {len(all_feats):,} rows")

# Final aggregation across all chunks
final = all_feats.groupby("account_id").agg(
    mcc_zscore_max=("mcc_zscore_max", "max"),
    mcc_zscore_mean_sum=("mcc_zscore_mean", "sum"),  # weighted mean later
    mcc_zscore_gt2=("mcc_zscore_gt2", "sum"),
    txn_count_v2=("txn_count_chunk", "sum"),
    total_vol_v2=("total_vol_chunk", "sum"),
    struct_48_50=("struct_48_50_count", "sum"),
    struct_45_50=("struct_45_50_count", "sum"),
    ch_total=("ch_total", "sum"),
    ch_nunique=("ch_nunique", "max"),  # approx
    weekend_cnt=("weekend_cnt", "sum"),
    large_round_cnt=("large_round_cnt", "sum"),
    exact_round_cnt=("exact_round_cnt", "sum"),
    monthly_credit_cv=("monthly_credit_cv", "mean"),  # approx
    months_with_credits=("months_with_credits", "max"),
    month_boundary_cnt=("month_boundary_cnt", "sum"),
    max_txn_v2=("max_txn_chunk", "max"),
    neg_amt_cnt=("neg_amt_cnt", "sum"),
).reset_index().fillna(0)

# Derived features
final["mcc_zscore_mean"] = final["mcc_zscore_mean_sum"] / (final["txn_count_v2"].clip(1) / len(parts) * len(parts))
final["unusual_mcc_pct"] = final["mcc_zscore_gt2"] / final["txn_count_v2"].clip(1)
final["struct_48_50_pct"] = final["struct_48_50"] / final["txn_count_v2"].clip(1)
final["struct_45_50_pct"] = final["struct_45_50"] / final["txn_count_v2"].clip(1)
final["weekend_pct"] = final["weekend_cnt"] / final["txn_count_v2"].clip(1)
final["large_round_pct"] = final["large_round_cnt"] / final["txn_count_v2"].clip(1)
final["exact_round_pct"] = final["exact_round_cnt"] / final["txn_count_v2"].clip(1)
final["month_boundary_pct"] = final["month_boundary_cnt"] / final["txn_count_v2"].clip(1)
final["max_single_txn_pct"] = final["max_txn_v2"] / final["total_vol_v2"].clip(1)
final["reversal_ratio"] = final["neg_amt_cnt"] / final["txn_count_v2"].clip(1)

# Channel entropy approximation
final["channel_diversity"] = final["ch_nunique"]

# Salary regularity (low CV = regular = legitimate→ high CV = irregular = suspicious)
final["salary_irregularity"] = final["monthly_credit_cv"]

# Volume per account day (from accounts table)
acct_open = dict(zip(accounts["account_id"],
                      pd.to_datetime(accounts["account_opening_date"], errors="coerce")))

# Keep only useful derived features
keep_cols = ["account_id", "mcc_zscore_max", "mcc_zscore_mean", "unusual_mcc_pct",
             "struct_48_50_pct", "struct_45_50_pct", "weekend_pct", "large_round_pct",
             "exact_round_pct", "month_boundary_pct", "max_single_txn_pct",
             "reversal_ratio", "channel_diversity", "salary_irregularity",
             "months_with_credits"]
final_clean = final[keep_cols]

print(f"Final features: {len(final_clean):,} accounts × {len(keep_cols)-1} new features")

# %% [markdown]
# ## 4 — Merge with existing features + add account features

# %%
print("Merging with existing features...")

for df_name, df_base in [("train", train_base), ("test", test_base)]:
    df = df_base.merge(final_clean, on="account_id", how="left")

    # Add account-level features (instant)
    am = df[["account_id"]].merge(
        accounts[["account_id", "last_kyc_date", "account_opening_date",
                  "avg_balance", "monthly_avg_balance", "quarterly_avg_balance",
                  "daily_avg_balance"]],
        on="account_id", how="left")

    kyc = pd.to_datetime(am["last_kyc_date"], errors="coerce")
    df["kyc_age_days"] = (pd.Timestamp.now() - kyc).dt.days.fillna(9999)

    opening = pd.to_datetime(am["account_opening_date"], errors="coerce")
    df["account_age_days"] = (pd.Timestamp.now() - opening).dt.days.fillna(9999)

    monthly = am["monthly_avg_balance"].fillna(0)
    quarterly = am["quarterly_avg_balance"].fillna(0)
    avg = am["avg_balance"].fillna(0)
    df["balance_volatility"] = (monthly - quarterly).abs() / quarterly.abs().clip(1)

    # Volume per day of account age
    df["volume_per_day"] = df["total_volume"] / df["account_age_days"].clip(1)

    # Fill NAs
    new_feats = [c for c in df.columns if c not in df_base.columns and c != "account_id"]
    for c in new_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    out_path = os.path.join(DATA_DIR, f"features_{df_name}_v2.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")

print(f"\n✅ Pipeline V2 complete in {(time.time()-t0)/60:.1f}min")
print(f"New features added: {len(new_feats)}")
for f in sorted(new_feats):
    print(f"  + {f}")

# %% [markdown]
# ## 5 — V15 Model (uses v2 features)

# %%
print("=" * 60)
print("V15 MODEL")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, fbeta_score,
                             precision_score, recall_score, confusion_matrix)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv(os.path.join(DATA_DIR, "features_train_v2.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "features_test_v2.csv"))
train = train.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")
test = test.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")

# OOF Target Encoding
N_FOLDS = 10
skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
train["branch_mule_rate_oof"] = np.nan
global_mean = train["is_mule"].mean()

for tr_idx, val_idx in skf_te.split(train, train["is_mule"]):
    tr_df = train.iloc[tr_idx]
    branch_stats = tr_df.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
    branch_stats["rate"] = (branch_stats["sum"] + 10 * global_mean) / (branch_stats["count"] + 10)
    mapped = train.iloc[val_idx]["branch_code"].map(branch_stats["rate"]).fillna(global_mean)
    train.loc[train.index[val_idx], "branch_mule_rate_oof"] = mapped.values

branch_stats_full = train.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
branch_stats_full["rate"] = (branch_stats_full["sum"] + 10 * global_mean) / (branch_stats_full["count"] + 10)
test["branch_mule_rate_oof"] = test["branch_code"].map(branch_stats_full["rate"]).fillna(global_mean)

score_cols = ["near_threshold_pct", "round_amount_pct", "gap_cv", "degree_centrality",
              "mule_trigram_count", "branch_mule_rate_oof", "has_prior_freeze"]
for df in [train, test]:
    c_score = np.zeros(len(df))
    for col in score_cols:
        if col in df.columns:
            m, s = train[col].mean(), train[col].std()
            c_score += (df[col].fillna(m) - m) / s if s > 0 else 0
    df["composite_score_fixed"] = c_score

# Lean signals
for df in [train, test]:
    sig_strengths = []
    for col, invert in [("days_to_first_large", False), ("near_threshold_pct", False),
                         ("median_dwell_hours", True), ("round_amount_pct", False),
                         ("degree_centrality", False)]:
        if col in df.columns:
            r = df[col].rank(pct=True).fillna(0.5).values
            sig_strengths.append((1-r) if invert else r)
    if "has_mobile_update" in df.columns:
        sig_strengths.append(df["has_mobile_update"].fillna(0).values)
    if "branch_mule_rate_oof" in df.columns:
        sig_strengths.append(df["branch_mule_rate_oof"].rank(pct=True).fillna(0.5).values)
    if sig_strengths:
        sm = np.column_stack(sig_strengths)
        df["signal_count_above_p75"] = (sm > 0.75).sum(axis=1).astype(float)
        df["signal_variance"] = np.var(sm, axis=1)
        df["signal_max_strength"] = np.max(sm, axis=1)
        df["signal_mean_strength"] = np.mean(sm, axis=1)

# Prepare features
drop_cols = ["account_id", "is_mule", "first_large_ts", "open_date",
             "branch_code", "branch_mule_rate", "composite_score"]
features = [c for c in train.columns if c not in drop_cols and train[c].nunique() > 1]
for c in features:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c] = pd.to_numeric(test[c], errors="coerce")
train[features] = train[features].fillna(train[features].median())
test[features] = test[features].fillna(train[features].median())
print(f"Total features: {len(features)}")

# Pruning
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X = train[features].values
y = train["is_mule"].values
oof_screen = np.zeros(len(y))
for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X[tr], y[tr])
    oof_screen[val] = m.predict_proba(X[val])[:,1]

extreme = (y == 1) & (oof_screen < 0.02)
X_clean = X[~extreme]
y_clean = y[~extreme]
print(f"Dropped {extreme.sum()} → {len(y_clean):,}")

# Training
print("Training ensemble...")
spw = (y_clean == 0).sum() / max(1, (y_clean == 1).sum())
oof_lgb, oof_xgb, oof_cat = np.zeros(len(y_clean)), np.zeros(len(y_clean)), np.zeros(len(y_clean))
t_lgb, t_xgb, t_cat = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))
X_test = test[features].values
models_lgb, models_xgb, models_cat = [], [], []

for fold, (tr, val) in enumerate(skf.split(X_clean, y_clean)):
    print(f"--- Fold {fold+1}/{N_FOLDS} ---")
    Xtr, Xval = X_clean[tr], X_clean[val]
    ytr, yval = y_clean[tr], y_clean[val]

    m1 = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, max_depth=8, num_leaves=63,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, verbosity=-1, n_jobs=-1)
    m1.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val] = m1.predict_proba(Xval)[:,1]
    t_lgb += m1.predict_proba(X_test)[:,1] / N_FOLDS

    m2 = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                           min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=42, verbosity=0, eval_metric="auc", n_jobs=-1,
                           early_stopping_rounds=50)
    m2.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val] = m2.predict_proba(Xval)[:,1]
    t_xgb += m2.predict_proba(X_test)[:,1] / N_FOLDS

    m3 = CatBoostClassifier(iterations=2000, learning_rate=0.02, depth=7,
                            auto_class_weights='Balanced', l2_leaf_reg=3,
                            random_state=42, verbose=False, early_stopping_rounds=50)
    m3.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cat[val] = m3.predict_proba(Xval)[:,1]
    t_cat += m3.predict_proba(X_test)[:,1] / N_FOLDS
    
    models_lgb.append(m1); models_xgb.append(m2); models_cat.append(m3)

oof_ens = (oof_lgb + oof_xgb + oof_cat) / 3.0
t_ens = (t_lgb + t_xgb + t_cat) / 3.0
print(f"\nOOF AUC: {roc_auc_score(y_clean, oof_ens):.4f}")

best_f2, best_t = 0, 0.5
for t in np.arange(0.1, 0.95, 0.01):
    f2 = fbeta_score(y_clean, (oof_ens > t).astype(int), beta=2)
    if f2 > best_f2:
        best_f2, best_t = f2, t

preds = (oof_ens > best_t).astype(int)
print(f"Threshold: {best_t:.2f}  F1={f1_score(y_clean,preds):.4f}")

imp = pd.DataFrame({"feature": features, "importance": m1.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
print(f"\nTop 30 features:")
for _, r in imp.head(30).iterrows():
    new_marker = " ★NEW" if r["feature"] in new_feats else ""
    print(f"  {r['feature']:<35} {r['importance']:>6.0f}{new_marker}")

test["is_mule_prob"] = t_ens
mule_preds = test[test["is_mule_prob"] > best_t]["account_id"].tolist()
print(f"Predicted mules: {len(mule_preds)}")

# %% [markdown]
# ## 6 — Temporal Windows

# %%
print("TEMPORAL WINDOWS")
temporal_threshold = best_t * 0.25
high_prob_ids = set(test[test["is_mule_prob"] > temporal_threshold]["account_id"].tolist())
print(f"Accounts: {len(high_prob_ids):,}")

daily_vol = {}
for i, p in enumerate(parts):
    try:
        ds = pq.read_table(p, columns=["account_id", "transaction_timestamp", "amount"],
                           filters=[("account_id", "in", list(high_prob_ids))])
        df = ds.to_pandas()
    except:
        df = pd.read_parquet(p, columns=["account_id", "transaction_timestamp", "amount"])
        df = df[df["account_id"].isin(high_prob_ids)]
    if df.empty: continue
    df["ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["date"] = df["ts"].dt.date
    df["abs_amount"] = df["amount"].abs()
    grp = df.groupby(["account_id", "date"])["abs_amount"].sum()
    for (aid, dt), vol in grp.items():
        if aid not in daily_vol: daily_vol[aid] = {}
        daily_vol[aid][dt] = daily_vol[aid].get(dt, 0) + vol
    del df
    if (i+1) % 100 == 0: print(f"  [{i+1}/{len(parts)}]")
    gc.collect()

def temporal_window(vol_dict):
    if len(vol_dict) < 3: return "", ""
    dates = sorted(vol_dict.keys())
    vols = np.array([vol_dict[d] for d in dates])
    total = vols.sum()
    if total == 0: return "", ""
    best_start, best_end = 0, len(dates) - 1
    for wd in [14, 30, 60, 90]:
        bv, bs, be = 0, 0, 0
        for j in range(len(vols)):
            k, wv = j, 0
            while k < len(vols) and (dates[k] - dates[j]).days <= wd:
                wv += vols[k]; k += 1
            if wv > bv: bv, bs, be = wv, j, k-1
        if bv / total >= 0.50: best_start, best_end = bs, be; break
    wv = vols[best_start:best_end+1]; wd = dates[best_start:best_end+1]
    if len(wv)>5:
        wc = np.cumsum(wv); wt = wc[-1]
        if wt>0:
            wcdf = wc/wt
            s = min(np.searchsorted(wcdf,0.05),len(wd)-1)
            e = min(np.searchsorted(wcdf,0.95),len(wd)-1)
            wd = wd[s:e+1]
    if len(wd)==0: wd = dates[best_start:best_end+1]
    return f"{wd[0]}T00:00:00", f"{wd[-1]}T23:59:59"

tw = {}
for aid in high_prob_ids:
    tw[aid] = temporal_window(daily_vol.get(aid, {}))

# %% [markdown]
# ## 7 — Submission

# %%
sub = pd.DataFrame({"account_id": test["account_id"], "is_mule": test["is_mule_prob"],
                     "suspicious_start": "", "suspicious_end": ""})
for aid, (s, e) in tw.items():
    mask = sub["account_id"] == aid
    sub.loc[mask, "suspicious_start"] = s
    sub.loc[mask, "suspicious_end"] = e

sub.to_csv(os.path.join(DATA_DIR, "submission_v15.csv"), index=False)
print(f"✅ submission_v15.csv: mules={((sub['is_mule']>0.5).sum()):,}")
print(f"Total: {(time.time()-t0)/60:.1f}min")
print("=" * 60)
print("SAVING MODEL WEIGHTS")
print("=" * 60)

import joblib
import os

model_save_dir = os.path.join(DATA_DIR, "v15_models")
os.makedirs(model_save_dir, exist_ok=True)
for i in range(N_FOLDS):
    joblib.dump(models_lgb[i], os.path.join(model_save_dir, f"lgbm_fold_{i}.pkl"))
    joblib.dump(models_xgb[i], os.path.join(model_save_dir, f"xgb_fold_{i}.pkl"))
    # CatBoost has its own preferred save format
    models_cat[i].save_model(os.path.join(model_save_dir, f"catboost_fold_{i}.cbm"))

print(f"✅ Saved 30 model weight artifacts to: {model_save_dir}")