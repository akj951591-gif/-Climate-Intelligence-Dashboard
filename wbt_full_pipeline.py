"""
WBT (Wet Bulb Temperature) — Full Prediction Pipeline
======================================================
Dataset : train.csv (1.4M rows, 125 locations, 1984-2014)
          test.csv  (500K rows, 125 locations, 2015-2025)
Target  : WBT for 10 future days (target_day_1 … target_day_10)
Model   : LightGBM  (MAE objective)
Output  : submission.csv  (matches sample_submission.csv format exactly)

Install : pip install lightgbm scikit-learn pandas numpy

Usage   : python wbt_full_pipeline.py
          (edit TRAIN_PATH / TEST_PATH / OUTPUT_PATH at the top if needed)
"""

# ── paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "train1.csv"
TEST_PATH   = "test.csv"
OUTPUT_PATH = "submission.csv"

SAMPLE_FRAC = 0.7          # fraction of train rows to sample (set 1.0 for full)
RANDOM_SEED = 42

# ── imports ───────────────────────────────────────────────────────────────────
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DATE = datetime.date(1984, 1, 1)   # day 1 = 1984-01-01

def add_date_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Reconstruct calendar features.
    Train has a 'date' column; test has only 'day_index' (0-based) inside row_id."""
    df = df.copy()

    if is_train:
        df["date"]      = pd.to_datetime(df["date"])
        df["day_num"]   = df["row_id"].str.extract(r"D(\d+)")[0].astype(int)
    else:
        df["day_num"]   = df["row_id"].str.extract(r"D(\d+)")[0].astype(int)
        df["date"]      = pd.to_datetime(
            [BASE_DATE + datetime.timedelta(days=int(d) - 1) for d in df["day_num"]]
        )

    df["month"]      = df["date"].dt.month
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["year"]       = df["date"].dt.year
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived / cyclic / interaction features."""
    df = df.copy()

    # cyclic calendar (avoids Dec→Jan discontinuity)
    df["sin_doy"]      = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"]      = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["sin_month"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]    = np.cos(2 * np.pi * df["month"] / 12)

    # thermodynamic / surface energy
    df["temp_diff"]    = df["T2M"]  - df["TSOIL1"]          # air-soil gradient
    df["ts_t2m"]       = df["TS"]   - df["T2M"]             # skin-air gradient
    df["rad_ratio"]    = df["ALLSKY_SFC_SW_DWN"] / (df["CLRSKY_SFC_SW_DWN"] + 1e-6)
    df["rad_net"]      = df["ALLSKY_SFC_SW_DWN"] - df["CLRSKY_SFC_SW_DWN"]
    df["evap_wet"]     = df["EVLAND"]  * df["GWETTOP"]      # moisture flux proxy
    df["evap_root"]    = df["EVLAND"]  * df["GWETROOT"]
    df["wind_stress"]  = df["WS10M"]   * df["PRECTOTCORR"]  # wet-wind interaction

    # wind decomposition (direction is circular)
    df["wdir_sin"]     = np.sin(np.radians(df["WD10M"]))
    df["wdir_cos"]     = np.cos(np.radians(df["WD10M"]))

    # normalised pressure (removes mean offset)
    df["ps_anom"]      = df["PS"] - df["PS"].mean()

    return df


FEATURES = [
    # spatial
    "rel_lat", "rel_lon",
    # atmospheric state
    "T2M", "TSOIL1", "TS",
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
    "CLOUD_AMT", "GWETTOP", "GWETROOT",
    "WS10M", "PS", "PRECTOTCORR", "EVLAND",
    # calendar
    "month", "dayofyear", "year",
    "sin_doy", "cos_doy", "sin_month", "cos_month",
    # engineered
    "temp_diff", "ts_t2m", "rad_ratio", "rad_net",
    "evap_wet", "evap_root", "wind_stress",
    "wdir_sin", "wdir_cos", "ps_anom",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 58)
print("  WBT Prediction Pipeline")
print("=" * 58)

# ── train ─────────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading train data  →  {TRAIN_PATH}")
train = pd.read_csv(TRAIN_PATH)
print(f"      Raw shape : {train.shape}")
print(f"      Nulls     : {train.isnull().sum().sum()}")
print(f"      WBT range : {train['WBT'].min():.2f} – {train['WBT'].max():.2f} °C")

if SAMPLE_FRAC < 1.0:
    train = train.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"      Sampled   : {len(train):,} rows  ({SAMPLE_FRAC*100:.0f}%)")

train = add_date_features(train, is_train=True)
train = engineer_features(train)
train["loc"] = train["row_id"].str.extract(r"(LOC\d+)")[0]

# ── test ──────────────────────────────────────────────────────────────────────
print(f"\n[2/5] Loading test data  →  {TEST_PATH}")
test = pd.read_csv(TEST_PATH)
print(f"      Raw shape  : {test.shape}")
test = add_date_features(test, is_train=False)
test = engineer_features(test)
test["loc"] = test["row_id"].str.extract(r"(LOC\d+)")[0]
print(f"      Date range : {test['date'].min().date()} → {test['date'].max().date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODEL  (GroupKFold — held-out locations for honest evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/5] Training LightGBM  (5-fold GroupKFold by location)")

X      = train[FEATURES]
y      = train["WBT"]
groups = train["loc"]

lgbm_params = {
    "objective":          "regression_l1",   # MAE loss — robust to outliers
    "metric":             "mae",
    "n_estimators":       1000,
    "learning_rate":      0.08,
    "num_leaves":         127,
    "max_depth":          -1,
    "subsample":          0.8,
    "subsample_freq":     1,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "min_child_samples":  30,
    "n_jobs":             -1,
    "verbose":            -1,
    "random_state":       RANDOM_SEED,
}

gkf       = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(train))
fold_models: list[lgb.LGBMRegressor] = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr,  y_tr  = X.iloc[tr_idx],  y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50,  verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_preds[val_idx] = model.predict(X_val)
    fold_mae  = mean_absolute_error(y_val, oof_preds[val_idx])
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    fold_r2   = r2_score(y_val, oof_preds[val_idx])
    held_locs = groups.iloc[val_idx].nunique()
    print(
        f"  Fold {fold+1}/5  "
        f"MAE={fold_mae:.4f}  RMSE={fold_rmse:.4f}  R²={fold_r2:.4f}  "
        f"iter={model.best_iteration_}  locs={held_locs}"
    )
    fold_models.append(model)

oof_mae  = mean_absolute_error(y, oof_preds)
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
oof_r2   = r2_score(y, oof_preds)
print(f"\n  ── OOF Results ──")
print(f"  MAE  = {oof_mae:.4f} °C")
print(f"  RMSE = {oof_rmse:.4f} °C")
print(f"  R²   = {oof_r2:.4f}")

# feature importance (mean across folds)
imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": np.mean([m.feature_importances_ for m in fold_models], axis=0),
}).sort_values("importance", ascending=False)
print(f"\n  Top 10 features:")
for _, row in imp.head(10).iterrows():
    bar = "█" * int(row["importance"] / imp["importance"].max() * 28)
    print(f"    {row['feature']:<24} {bar}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREDICT WBT FOR EVERY TEST ROW
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/5] Predicting WBT for all {len(test):,} test rows")

# Ensemble: average predictions from all 5 fold models
test_X         = test[FEATURES]
test["WBT_pred"] = np.mean(
    [m.predict(test_X) for m in fold_models], axis=0
)

print(f"  WBT pred range : {test['WBT_pred'].min():.2f} – {test['WBT_pred'].max():.2f} °C")
print(f"  WBT pred mean  : {test['WBT_pred'].mean():.2f} °C")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BUILD SUBMISSION  (target_day_k = WBT_pred at day_num + k, same location)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/5] Building submission  →  {OUTPUT_PATH}")

# fast pivot: rows = day_num, cols = location
pivot = test.pivot(index="day_num", columns="loc", values="WBT_pred")

sub = test[["row_id", "loc", "day_num"]].copy()

for k in range(1, 11):
    target_days = sub["day_num"] + k

    # vectorised lookup with fallback to nearest available day
    vals = []
    for loc, tday in zip(sub["loc"], target_days):
        if tday in pivot.index and loc in pivot.columns:
            vals.append(pivot.loc[tday, loc])
        else:
            # edge case: last few rows exceed test range → use closest day
            closest = pivot.index[np.argmin(np.abs(pivot.index - tday))]
            vals.append(pivot.loc[closest, loc] if loc in pivot.columns
                        else test["WBT_pred"].mean())

    sub[f"target_day_{k}"] = vals
    print(f"  target_day_{k:>2}  mean={np.mean(vals):.3f}  "
          f"min={np.min(vals):.2f}  max={np.max(vals):.2f}")

# keep only submission columns (row_id + 10 targets)
submission_cols = ["row_id"] + [f"target_day_{k}" for k in range(1, 11)]
final = sub[submission_cols]

final.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved  {len(final):,} rows × {len(final.columns)} cols  →  {OUTPUT_PATH}")
print(f"  Nulls  : {final.isnull().sum().sum()}")
print(f"\n  Preview:")
print(final.head(3).to_string(index=False))
print("\n" + "=" * 58)
print("  Done.")
print("=" * 58)
import joblib

joblib.dump(fold_models, "wbt_model.pkl")
joblib.dump(FEATURES, "features.pkl")

print("Model saved successfully")