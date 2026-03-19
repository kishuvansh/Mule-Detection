import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from glob import glob
from datetime import timedelta

print("Loading data...")
train = pd.read_csv("features_train_p2.csv")
test = pd.read_csv("features_test_p2.csv")

# Load branch code for proper OOF Target Encoding
accounts = pd.read_parquet("accounts.parquet")
train = train.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")
test = test.merge(accounts[["account_id", "branch_code"]], on="account_id", how="left")

# Recompute OOF target encoding for branch_mule_rate to fix LEAKAGE
print("Fixing branch_mule_rate leakage via K-Fold Target Encoding...")
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["branch_mule_rate_oof"] = np.nan
global_mean = train["is_mule"].mean()

for tr_idx, val_idx in skf_te.split(train, train["is_mule"]):
    tr_df = train.iloc[tr_idx]
    branch_stats = tr_df.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
    # Smoothing
    branch_stats["rate"] = (branch_stats["sum"] + 10 * global_mean) / (branch_stats["count"] + 10)
    
    val_df = train.iloc[val_idx]
    mapped = val_df["branch_code"].map(branch_stats["rate"]).fillna(global_mean)
    train.loc[val_idx, "branch_mule_rate_oof"] = mapped.values

# Test set target encoding using ALL train data
branch_stats_full = train.groupby("branch_code")["is_mule"].agg(['sum', 'count'])
branch_stats_full["rate"] = (branch_stats_full["sum"] + 10 * global_mean) / (branch_stats_full["count"] + 10)
test["branch_mule_rate_oof"] = test["branch_code"].map(branch_stats_full["rate"]).fillna(global_mean)

# Re-compute composite score with valid branch_mule_rate
print("Recomputing composite score...")
score_cols = [
    "near_threshold_pct", "round_amount_pct", "gap_cv", "degree_centrality", 
    "mule_trigram_count", "branch_mule_rate_oof", "has_prior_freeze"
]
for df in [train, test]:
    c_score = np.zeros(len(df))
    for col in score_cols:
        if col in df.columns:
            m = train[col].mean()
            s = train[col].std()
            c_score += (df[col].fillna(m) - m) / s if s > 0 else 0
    df["composite_score_fixed"] = c_score

# Drop leaked features
drop_cols = ["account_id", "is_mule", "first_large_ts", "open_date", "branch_code", "branch_mule_rate", "composite_score"]
features = [c for c in train.columns if c not in drop_cols and train[c].nunique() > 1]

# Fill missing
train[features] = train[features].fillna(train[features].median())
test[features] = test[features].fillna(train[features].median())

# Exclude red herrings properly
print("Filtering Hard Red Herrings...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train[features].values
y = train["is_mule"].values
oof_screen = np.zeros(len(y))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X[tr], y[tr])
    oof_screen[val] = m.predict_proba(X[val])[:,1]

extreme_fake_mule = (y == 1) & (oof_screen < 0.02)
keep_mask = ~extreme_fake_mule
X_clean = X[keep_mask]
y_clean = y[keep_mask]
print(f"Dropped {extreme_fake_mule.sum()} extreme red herrings.")

print("Training Ensemble Model...")
models_lgb, models_xgb, models_cat = [], [], []
oof_lgb = np.zeros(len(y_clean))
oof_xgb = np.zeros(len(y_clean))
oof_cat = np.zeros(len(y_clean))

t_lgb = np.zeros(len(test))
t_xgb = np.zeros(len(test))
t_cat = np.zeros(len(test))

spw = (y_clean == 0).sum() / max(1, (y_clean == 1).sum())

import sys
for fold, (tr, val) in enumerate(skf.split(X_clean, y_clean)):
    print(f" Fold {fold+1}...")
    Xtr, Xval = X_clean[tr], X_clean[val]
    ytr, yval = y_clean[tr], y_clean[val]
    
    # LGBM
    m1 = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=8, 
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, 
                            random_state=42, verbosity=-1, n_jobs=-1)
    m1.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val] = m1.predict_proba(Xval)[:,1]
    t_lgb += m1.predict_proba(test[features].values)[:,1] / 5.0
    models_lgb.append(m1)
    
    # XGB
    m2 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                           random_state=42, verbosity=0, eval_metric="auc", n_jobs=-1,
                           early_stopping_rounds=50)
    m2.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val] = m2.predict_proba(Xval)[:,1]
    t_xgb += m2.predict_proba(test[features].values)[:,1] / 5.0
    models_xgb.append(m2)
    
    # CatBoost
    m3 = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=7,
                            auto_class_weights='Balanced', random_state=42,
                            verbose=False, early_stopping_rounds=50)
    m3.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cat[val] = m3.predict_proba(Xval)[:,1]
    t_cat += m3.predict_proba(test[features].values)[:,1] / 5.0
    models_cat.append(m3)

oof_ens = (oof_lgb + oof_xgb + oof_cat) / 3.0
t_ens = (t_lgb + t_xgb + t_cat) / 3.0

print(f"OOF AUC LGB: {roc_auc_score(y_clean, oof_lgb):.4f}")
print(f"OOF AUC XGB: {roc_auc_score(y_clean, oof_xgb):.4f}")
print(f"OOF AUC CAT: {roc_auc_score(y_clean, oof_cat):.4f}")
print(f"OOF AUC ENS: {roc_auc_score(y_clean, oof_ens):.4f}")

# F2 tuning for threshold
from sklearn.metrics import precision_score, recall_score, fbeta_score
best_f2, best_t = 0, 0.5
for t in np.arange(0.1, 0.95, 0.01):
    f2 = fbeta_score(y_clean, (oof_ens > t).astype(int), beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_t = t
print(f"Best Threshold for F2: {best_t:.2f} (F2 = {best_f2:.4f}, F1 = {f1_score(y_clean, (oof_ens > best_t).astype(int)):.4f})")

test["is_mule_prob"] = t_ens
mule_preds = test[test["is_mule_prob"] > best_t]["account_id"].tolist()
print(f"Predicted mules: {len(mule_preds)}")


print("\nExtracting temporal windows...")
import pyarrow.parquet as pq

def two_pass_temporal_window(daily_vol_dict):
    if len(daily_vol_dict) < 5:
        return "", ""
    dates = sorted(daily_vol_dict.keys())
    vols = np.array([daily_vol_dict[d] for d in dates])
    
    # Pass 1: find active window via highest volume concentration in 30 days
    best_sum = -1
    best_start = 0
    best_end = 0
    for i in range(len(vols)):
        current_sum = 0
        end_idx = i
        while end_idx < len(vols) and (dates[end_idx] - dates[i]).days <= 30:
            current_sum += vols[end_idx]
            end_idx += 1
        if current_sum > best_sum:
            best_sum = current_sum
            best_start = i
            best_end = end_idx - 1
            
    # Zoom in to densest cluster within that window
    window_vols = vols[best_start:best_end+1]
    window_dates = dates[best_start:best_end+1]
    
    threshold = np.percentile(window_vols, 75)
    flagged = np.where(window_vols >= threshold)[0]
    
    if len(flagged) > 0:
        start_d = window_dates[flagged[0]]
        end_d = window_dates[flagged[-1]]
    else:
        start_d = dates[best_start]
        end_d = dates[best_end]
        
    return f"{start_d}T00:00:00", f"{end_d}T23:59:59"

parts = sorted(glob("transactions/batch-*/part_*.parquet"))
account_timings = {a: {} for a in mule_preds}

for i, p in enumerate(parts):
    ds = pq.read_table(p, columns=["account_id", "transaction_timestamp", "amount"], 
                       filters=[("account_id", "in", mule_preds)])
    df = ds.to_pandas()
    if df.empty: continue
    df["ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["date"] = df["ts"].dt.date
    df["abs_amount"] = df["amount"].abs()
    
    grp = df.groupby(["account_id", "date"])["abs_amount"].sum()
    for (aid, dt), val in grp.items():
        if dt not in account_timings[aid]:
            account_timings[aid][dt] = 0
        account_timings[aid][dt] += val
    if (i+1) % 50 == 0:
        print(f"Scanned {i+1} parts for windows...")

temporal_windows = {}
for aid, d_dict in account_timings.items():
    s, e = two_pass_temporal_window(d_dict)
    temporal_windows[aid] = (s, e)

submission = pd.DataFrame({
    'account_id': test['account_id'],
    'is_mule': test['is_mule_prob']
})
submission['suspicious_start'] = submission['account_id'].apply(lambda x: temporal_windows.get(x, ("", ""))[0])
submission['suspicious_end']   = submission['account_id'].apply(lambda x: temporal_windows.get(x, ("", ""))[1])

submission.to_csv("submission_p3_fixed_leak.csv", index=False)
print("Finished! Wrote submission_p3_fixed_leak.csv")
