#  AML Mule Account Detection (Phase 2)
> **IITD Tryst Hackathon Solution**  
> **Model Version:** V7 (Calibration + Red Herring Avoidance + IoU Push)

Welcome to the solution repository for the AML Mule Account Detection challenge. This document provides a highly structured breakdown of the **Phase 7 Model (v7)**, detailing the installation, core methodologies, and a block-by-block logical walkthrough of our code.

---

## 🛠️ 1. Installation & Environment Setup

###  Prerequisites
Ensure you are running **Python 3.10+**. The pipeline requires roughly 32GB RAM to smoothly process the large transaction parquet files.

### Dependencies
Install the required packages using `pip`:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost "catboost>=1.2" pyarrow
```

### Execution
Assuming your raw data and pre-computed features (`features_train_p2.csv`, `features_test_p2.csv`) are correctly placed inside the `Phase_2/` directory:
```bash
cd Phase_2/

# Option 1: Run headless via nbconvert
jupyter nbconvert --to notebook --execute phase7_model_v7.ipynb --output phase7_model_v7_executed.ipynb

# Option 2: Run interactively
jupyter notebook phase7_model_v7.ipynb
```
*Expected runtime:* **~5 minutes** on a modern multi-core CPU. The final predictions will be saved as `submission_v7.csv`.

---

##  2. Block-by-Block Code Approach & Reasoning

Our approach in V7 was specifically engineered to tackle **Red Herrings (RH)**—benign accounts designed by the hackathon organizers to look identical to mules to trick rudimentary models. 

Here is the step-by-step reasoning for each major block in `phase7_model_v7.ipynb`:

###  Block 0: Setup & Data Loading
* **What it does:** Loads the target labels, account metadata, and pre-computed behavioral features (from earlier pipelines).
* **Reasoning:** Establishing a unified DataFrame is critical. We immediately merge `branch_code` and `alert_reason` from the labels, as these metadata fields hold the key to uncovering geographic anomalies and known behavioral injection patterns.

###  Block 1: OOF Target Encoding
* **What it does:** Calculates the historical mule rate for each bank branch using Out-of-Fold (OOF) encoding, smoothed with a Bayesian prior.
* **Reasoning:** Financial crime is often geographically clustered. Using OOF encoding strictly prevents data leakage while giving the model a powerful prior probability metric (`branch_mule_rate_oof`) based on the account's localized network.

###  Block 2: Red Herring Awareness Features *(Crucial Innovation)*
* **What it does:** Generates 9 distinct binary flags (`sig_dormant`, `sig_structuring`, `sig_rapid`, etc.) based on 13 known money-mule typologies, summing them into a `multi_signal_count`.
* **Reasoning:** **Red Herrings typically mimic exactly ONE suspicious pattern.** Real, organic money mules are messier and trigger *multiple* overlapping alerts (e.g., structuring AND rapid pass-through). By counting the number of corroborating signals, we teach the model to distinguish between a planted anomaly and true mule behavior.

###  Block 3 & 4: Aggressive Red Herring Pruning
* **What it does:** Trains a quick LightGBM screening model. If a labeled mule receives an extremely low OOF probability (< 0.02) or has < 2 signals, it is flagged as a Red Herring and removed/down-weighted from the training pool.
* **Reasoning:** If we train on Red Herrings, the model will learn to flag innocent accounts. By aggressively purging extreme RHs from the training data, we force the downstream models to focus purely on authentic money-laundering characteristics.

###  Block 5: The Gradient Boosting Ensemble
* **What it does:** Trains three advanced gradient boosters (LightGBM, XGBoost, CatBoost) using Stratified 5-Fold Cross-Validation, combining them via **simple averaging**.
* **Reasoning:** Using three diverse tree architectures maximizes predictive robustness. Crucially, we use *simple averaging* (instead of rank averaging) to maintain rigorously **well-calibrated probabilities**. This accurately reflects the true baseline mule rate (~2.8%) without artificially inflating false positives.

###  Block 6 & 7: Post-Filter & Threshold Optimization
* **What it does:** Rule-based dampening for high-probability but low-signal accounts, followed by selecting the F2-optimal classification threshold.
* **Reasoning:** 
  1. **Safety Net:** If a model predicts high risk (>0.3) but the account has < 2 behavioral signals, we multiply the probability by 0.6 to dampen it. We don't trust the model if the behavioral evidence isn't corroborating.
  2. **Recall Focus:** We select the F2-optimal threshold (t ≈ 0.300) to prioritize high recall (catching more mules), directly aligning with the hackathon's scoring incentives.

###  Block 8: Adaptive CDF Temporal Windows
* **What it does:** Streams raw transaction logs to locate the densest 14/30/60/90-day volume windows for flagged accounts, then trims the 5th-95th percentiles using a Cumulative Distribution Function (CDF).
* **Reasoning:** Hackathon scoring heavily relies on Temporal Intersection-over-Union (IoU). Simply taking the first and last transaction dates yields terrible IoU. Our Adaptive CDF isolates the specific burts of high-volume illicit activity, cutting out sparse noise days and maximizing temporal precision.

###  Block 9: Submission Generation
* **What it does:** Maps probabilities and temporal timestamps (`suspicious_start`, `suspicious_end`) to generating the final `submission_v7.csv`.
* **Reasoning:** Ensures strict formatting compliance. Automated sanity checks verify that the predicted mule rate aligns with chronological expectations before concluding execution.

---

##  3. Models Used

Our final predictions are powered by a uniformly weighted ensemble of three state-of-the-art Gradient Boosted Decision Trees:

| Model | Hyperparameters & Configuration | Why we used it |
| :--- | :--- | :--- |
| **LightGBM** | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=8`, `subsample=0.8` | Incredibly fast, handles missing values natively, and excellent at finding complex feature splits for tabular data. |
| **XGBoost** | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=7`, `subsample=0.8` | Highly robust to overfitting via strict regularization; provides a slightly different tree topography than LGBM. |
| **CatBoost** | `iterations=1200`, `learning_rate=0.03`, `depth=7`, `auto_class_weights='Balanced'` | Exceptional out-of-the-box performance; symmetrical trees are highly resistant to structural noise. |

*Note: All three models utilize a scaling weight (`scale_pos_weight` / `auto_class_weights`) to gracefully handle the heavy class imbalance (~97% legitimate, ~3% mules).*

---
*Built for the IITD Tryst Hackathon — Pushing the limits of AI in Anti-Money Laundering.*
