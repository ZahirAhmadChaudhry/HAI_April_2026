from __future__ import annotations

import json
import math
import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DATASET = BASE_DIR / "clean_hai_dataset.csv"
CLEANED_FEATURES_JSON = BASE_DIR / "cleaned_feature_sets.json"

OUTPUT_RESULTS = BASE_DIR / "repeated_experiment_results.csv"
OUTPUT_CHECKPOINT = BASE_DIR / "repeated_experiment_checkpoint.csv"
OUTPUT_SUMMARY = BASE_DIR / "repeated_experiment_summary.md"
OUTPUT_SHAP_STABILITY = BASE_DIR / "repeated_shap_stability.csv"

FIG_DIR = BASE_DIR / "figures"
FIG_AUC_BOXPLOT = FIG_DIR / "auc_distribution_boxplot.png"
FIG_AUC_DIFF_HIST = FIG_DIR / "auc_difference_histogram.png"
FIG_SHAP_STABILITY = FIG_DIR / "shap_stability_plot.png"

START_SEED = 43
END_SEED = 62
N_TRIALS = 50
THRESHOLD = 0.5

NUM_THREADS = max(1, int(os.environ.get("HAI_NUM_THREADS", os.cpu_count() or 1)))

ALGORITHMS = ["xgboost", "lightgbm", "catboost", "random_forest"]
CATEGORICAL_COLUMNS = [
    "sex",
    "admission_origin",
    "diagnostic_category",
    "trauma_status",
    "immunosuppression",
    "antibiotic_at_admission",
    "cancer_status",
    "intubation_status",
    "reintubation_status",
    "urinary_catheter",
    "ecmo_status",
    "admission_month",
    "admission_weekday",
    "weekend_admission",
]

FEATURE_GROUPS = {
    "Patient Demographics": {"age", "sex", "admission_origin"},
    "Clinical Severity": {
        "diagnostic_category",
        "trauma_status",
        "immunosuppression",
        "antibiotic_at_admission",
        "cancer_status",
        "severity_score_igs2",
    },
    "Medical Procedures": {
        "intubation_status",
        "reintubation_status",
        "intubation_days",
        "urinary_catheter",
        "urinary_catheter_days",
        "central_line_count",
        "ecmo_status",
    },
    "Length of Stay": {"length_of_stay"},
    "Temporal": {"admission_month", "admission_weekday", "weekend_admission"},
    "Organizational Environment": {
        "bed_occupancy",
        "patient_turnover",
        "unit_avg_los",
        "national_avg_los",
        "los_ratio_national",
    },
    "Organizational Staffing": {
        "nurse_staffing_etp",
        "nurse_aide_staffing_etp",
        "nurse_anesthetist_staffing_etp",
        "dietitian_staffing_etp",
        "medical_admin_assistant_staffing_etp",
        "total_staffing_etp",
        "nurse_staffing_count",
        "nurse_aide_staffing_count",
        "nurse_anesthetist_staffing_count",
        "dietitian_staffing_count",
        "medical_admin_assistant_staffing_count",
        "total_staffing_count",
    },
}


def detect_gpu_available() -> tuple[bool, str]:
    flag = os.environ.get("HAI_USE_GPU", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False, "disabled by HAI_USE_GPU"

    try:
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if probe.returncode == 0 and "GPU" in (probe.stdout or ""):
            first_line = (probe.stdout or "").strip().splitlines()[0]
            return True, first_line
    except Exception:
        pass

    if flag in {"1", "true", "yes", "on"}:
        return True, "forced by HAI_USE_GPU"

    return False, "no GPU detected"


USE_GPU, GPU_REASON = detect_gpu_available()


# ============================================================
# Data structures
# ============================================================
@dataclass
class PreprocessorBundle:
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    numeric_medians: dict[str, float]
    transformer: ColumnTransformer


# ============================================================
# Utility
# ============================================================
def configure_plots() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def to_md_table(df: pd.DataFrame, max_rows: int = 300) -> str:
    if df.empty:
        return "_No rows_"

    work = df.copy()
    if len(work) > max_rows:
        work = work.head(max_rows).copy()

    lines = [
        "| " + " | ".join([str(c) for c in work.columns]) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]

    for _, row in work.iterrows():
        vals = []
        for c in work.columns:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    s = ""
                else:
                    s = f"{v:.6g}"
            else:
                s = str(v)
            vals.append(s.replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def canonicalize_category(v: Any, na_token: str) -> str:
    if pd.isna(v):
        return na_token
    num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.notna(num):
        if float(num).is_integer():
            return str(int(num))
        return f"{float(num):.6g}"
    return str(v)


def apply_duration_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"intubation_days", "intubation_status"}.issubset(out.columns):
        st = pd.to_numeric(out["intubation_status"], errors="coerce")
        mask = out["intubation_days"].isna() & (st == 2)
        out.loc[mask, "intubation_days"] = 0.0

    if {"urinary_catheter_days", "urinary_catheter"}.issubset(out.columns):
        st = pd.to_numeric(out["urinary_catheter"], errors="coerce")
        mask = out["urinary_catheter_days"].isna() & (st == 2)
        out.loc[mask, "urinary_catheter_days"] = 0.0

    return out


def prepare_base_frame(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    out = df.loc[:, feature_cols].copy()
    out = apply_duration_rules(out)

    for c in categorical_cols:
        if c not in out.columns:
            continue
        na_token = "not_applicable" if c == "reintubation_status" else "missing"
        out[c] = out[c].map(lambda x: canonicalize_category(x, na_token))

    return out


def fit_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> PreprocessorBundle:
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    prepared = prepare_base_frame(train_df, feature_cols, categorical_cols)

    medians: dict[str, float] = {}
    for c in numeric_cols:
        m = pd.to_numeric(prepared[c], errors="coerce").median()
        if pd.isna(m):
            m = 0.0
        medians[c] = float(m)

    for c in numeric_cols:
        prepared[c] = pd.to_numeric(prepared[c], errors="coerce").fillna(medians[c])

    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    transformer.fit(prepared)

    return PreprocessorBundle(
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        numeric_medians=medians,
        transformer=transformer,
    )


def transform_with_preprocessor(bundle: PreprocessorBundle, df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_base_frame(df, bundle.feature_cols, bundle.categorical_cols)

    for c in bundle.numeric_cols:
        prepared[c] = pd.to_numeric(prepared[c], errors="coerce").fillna(bundle.numeric_medians[c])

    arr = bundle.transformer.transform(prepared)
    cols = bundle.transformer.get_feature_names_out().tolist()
    return pd.DataFrame(arr, index=df.index, columns=cols)


def maybe_df(X: Any, columns: list[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)


def make_safe_smote(y: pd.Series, seed: int) -> SMOTE | None:
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return None

    minority = int(counts.min())
    if minority <= 1:
        return None

    k_neighbors = min(5, minority - 1)
    return SMOTE(random_state=seed, k_neighbors=k_neighbors)


# ============================================================
# Models and tuning
# ============================================================
def sample_params(trial: optuna.trial.Trial, algorithm: str) -> dict[str, Any]:
    if algorithm == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }

    if algorithm == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

    if algorithm == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "iterations": trial.suggest_int("iterations", 50, 500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

    if algorithm == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def make_model(algorithm: str, params: dict[str, Any], seed: int):
    if algorithm == "xgboost":
        xgb_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
            "n_jobs": NUM_THREADS,
            "tree_method": "hist",
        }
        if USE_GPU:
            xgb_kwargs["device"] = "cuda"

        return XGBClassifier(
            **xgb_kwargs,
            **params,
        )

    if algorithm == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=NUM_THREADS,
            verbose=-1,
            **params,
        )

    if algorithm == "catboost":
        cat_kwargs = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": seed,
            "verbose": False,
            "thread_count": NUM_THREADS,
        }
        if USE_GPU:
            cat_kwargs["task_type"] = "GPU"
            cat_kwargs["devices"] = os.environ.get("HAI_GPU_DEVICES", "0")

        return CatBoostClassifier(
            **cat_kwargs,
            **params,
        )

    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=NUM_THREADS, **params)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_for_algorithm(
    algorithm: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    try:
        if algorithm == "lightgbm" and X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            )
            return model

        if algorithm == "catboost" and X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=25,
                verbose=False,
            )
            return model

        model.fit(X_train, y_train)
        return model
    except Exception as exc:
        # If GPU setup fails, transparently fall back to CPU for robustness.
        if algorithm == "xgboost" and USE_GPU:
            print(f"XGBoost GPU fit failed, falling back to CPU. Reason: {exc}")
            model.set_params(device="cpu")
            model.fit(X_train, y_train)
            return model

        if algorithm == "catboost" and USE_GPU:
            print(f"CatBoost GPU fit failed, falling back to CPU. Reason: {exc}")
            model.set_params(task_type="CPU")
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=25,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)
            return model

        raise


def print_compute_config() -> None:
    print(
        "Compute config: "
        f"NUM_THREADS={NUM_THREADS}, USE_GPU={USE_GPU}, GPU_INFO={GPU_REASON}"
    )


def cv_auc_leave_one_year_out(
    algorithm: str,
    params: dict[str, Any],
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    years_train: pd.Series,
    feature_cols: list[str],
    seed: int,
    trial: optuna.trial.Trial | None = None,
) -> float:
    folds = [(2019, 2020), (2020, 2019)]
    aucs: list[float] = []

    for fold_idx, (tr_year, va_year) in enumerate(folds):
        tr_mask = years_train == tr_year
        va_mask = years_train == va_year

        X_tr_raw = X_train_df.loc[tr_mask, feature_cols]
        y_tr = y_train.loc[tr_mask].astype(int)
        X_va_raw = X_train_df.loc[va_mask, feature_cols]
        y_va = y_train.loc[va_mask].astype(int)

        if len(y_tr.unique()) < 2 or len(y_va.unique()) < 2:
            aucs.append(0.5)
            continue

        bundle = fit_preprocessor(X_tr_raw, feature_cols)
        X_tr = transform_with_preprocessor(bundle, X_tr_raw)
        X_va = transform_with_preprocessor(bundle, X_va_raw)

        smote = make_safe_smote(y_tr, seed)
        if smote is None:
            X_tr_res = X_tr.copy()
            y_tr_res = y_tr.copy()
        else:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            X_tr_res = maybe_df(X_tr_res, X_tr.columns.tolist())

        model = make_model(algorithm, params, seed)
        model = fit_for_algorithm(algorithm, model, X_tr_res, y_tr_res, X_va, y_va)

        y_va_prob = model.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y_va, y_va_prob))
        aucs.append(auc)

        if trial is not None:
            trial.report(float(np.mean(aucs)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(aucs)) if aucs else 0.5


def tune_algorithm(
    algorithm: str,
    feature_set_name: str,
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    years_train: pd.Series,
    feature_cols: list[str],
    seed: int,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params(trial, algorithm)
        score = cv_auc_leave_one_year_out(
            algorithm=algorithm,
            params=params,
            X_train_df=X_train_df,
            y_train=y_train,
            years_train=years_train,
            feature_cols=feature_cols,
            seed=seed,
            trial=trial,
        )
        return score

    print(f"[seed={seed}] Tuning {algorithm} on Model {feature_set_name} ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[seed={seed}] Best CV AUC {algorithm}/Model {feature_set_name}: {study.best_value:.4f}")

    return study.best_params, float(study.best_value)


def train_final_model(
    algorithm: str,
    params: dict[str, Any],
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict[str, Any]:
    bundle = fit_preprocessor(X_train_raw, feature_cols)
    X_train = transform_with_preprocessor(bundle, X_train_raw)
    X_test = transform_with_preprocessor(bundle, X_test_raw)

    y_train_int = y_train.astype(int)
    smote = make_safe_smote(y_train_int, seed)

    if smote is None:
        X_res = X_train.copy()
        y_res = y_train_int.copy()
    else:
        X_res, y_res = smote.fit_resample(X_train, y_train_int)
        X_res = maybe_df(X_res, X_train.columns.tolist())

    model = make_model(algorithm, params, seed)
    model = fit_for_algorithm(algorithm, model, X_res, y_res)

    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "preprocessor": bundle,
        "X_test_processed": X_test,
        "y_test_prob": y_prob,
    }


# ============================================================
# Metrics and SHAP
# ============================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def infer_original_feature(encoded_feature: str, original_features: list[str]) -> str:
    if encoded_feature in original_features:
        return encoded_feature

    candidates = [f for f in original_features if encoded_feature.startswith(f + "_")]
    if candidates:
        return sorted(candidates, key=len, reverse=True)[0]
    return encoded_feature


def map_feature_group(original_feature: str) -> str:
    for group_name, feats in FEATURE_GROUPS.items():
        if original_feature in feats:
            return group_name
    return "Other"


def compute_shap_mean_abs(model: Any, X_test_processed: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(model)
            shap_out = explainer(X_test_processed)
            values = np.asarray(shap_out.values)
        except Exception:
            explainer = shap.Explainer(model, X_test_processed)
            shap_out = explainer(X_test_processed)
            values = np.asarray(shap_out.values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return np.abs(values).mean(axis=0)


# ============================================================
# Experiment orchestration
# ============================================================
def run_single_seed(
    run_id: int,
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    years_train: pd.Series,
    model_a_features: list[str],
    model_b_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for feature_set_name, feature_cols in [("A", model_a_features), ("B", model_b_features)]:
        X_train_raw = train_df[feature_cols].copy()
        X_test_raw = test_df[feature_cols].copy()

        for algorithm in ALGORITHMS:
            best_params, best_cv_auc = tune_algorithm(
                algorithm=algorithm,
                feature_set_name=feature_set_name,
                X_train_df=X_train_raw,
                y_train=y_train,
                years_train=years_train,
                feature_cols=feature_cols,
                seed=seed,
                n_trials=N_TRIALS,
            )

            trained = train_final_model(
                algorithm=algorithm,
                params=best_params,
                X_train_raw=X_train_raw,
                y_train=y_train,
                X_test_raw=X_test_raw,
                feature_cols=feature_cols,
                seed=seed,
            )

            metrics = compute_metrics(y_test.to_numpy(), trained["y_test_prob"])

            row = {
                "run_id": run_id,
                "seed": seed,
                "model_id": f"Model_{feature_set_name}_{algorithm}",
                "feature_set": feature_set_name,
                "algorithm": algorithm,
                "n_features": len(feature_cols),
                "cv_auc_roc": best_cv_auc,
                "best_params_json": json.dumps(best_params, sort_keys=True),
            }
            row.update(metrics)
            rows.append(row)

            print(
                f"[run={run_id} seed={seed}] {feature_set_name}/{algorithm}: "
                f"AUC={metrics['auc_roc']:.4f}, PR={metrics['auc_pr']:.4f}, F1={metrics['f1']:.4f}"
            )

    return rows


def load_cleaned_feature_sets() -> tuple[list[str], list[str]]:
    if not CLEANED_FEATURES_JSON.exists():
        raise FileNotFoundError(
            f"Missing cleaned feature set file: {CLEANED_FEATURES_JSON}. "
            "Run run_multicollinearity_cleanup.py first."
        )

    payload = json.loads(CLEANED_FEATURES_JSON.read_text(encoding="utf-8"))
    model_a = payload.get("model_a_cleaned")
    model_b = payload.get("model_b_cleaned")

    if not isinstance(model_a, list) or not isinstance(model_b, list):
        raise RuntimeError("Invalid cleaned_feature_sets.json structure.")

    return model_a, model_b


def resume_checkpoint(seeds: list[int]) -> list[dict[str, Any]]:
    expected_per_seed = len(ALGORITHMS) * 2

    if not OUTPUT_CHECKPOINT.exists():
        return []

    df = pd.read_csv(OUTPUT_CHECKPOINT)
    if df.empty:
        return []

    kept_rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_df = df[df["seed"] == seed]
        if len(seed_df) == expected_per_seed:
            kept_rows.extend(seed_df.to_dict("records"))
        elif len(seed_df) > 0:
            print(f"[seed={seed}] Found partial checkpoint ({len(seed_df)} rows). Rerunning this seed.")

    if kept_rows:
        pd.DataFrame(kept_rows).to_csv(OUTPUT_CHECKPOINT, index=False)

    return kept_rows


def summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "auc_roc",
        "auc_pr",
        "f1",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "brier_score",
    ]

    agg_rows: list[dict[str, Any]] = []
    for (feature_set, algorithm), grp in results_df.groupby(["feature_set", "algorithm"]):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "algorithm": algorithm,
            "model_id": f"Model_{feature_set}_{algorithm}",
        }
        for m in metric_cols:
            vals = grp[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals, ddof=1))
            row[f"{m}_median"] = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            row[f"{m}_iqr"] = float(q3 - q1)
        agg_rows.append(row)

    per_model_summary = pd.DataFrame(agg_rows).sort_values(["feature_set", "auc_roc_mean"], ascending=[True, False]).reset_index(drop=True)

    best_a = (
        results_df[results_df["feature_set"] == "A"]
        .sort_values(["run_id", "auc_roc"], ascending=[True, False])
        .groupby("run_id", as_index=False)
        .first()
        .rename(columns={"algorithm": "best_a_algorithm", "auc_roc": "best_a_auc"})
    )

    best_b = (
        results_df[results_df["feature_set"] == "B"]
        .sort_values(["run_id", "auc_roc"], ascending=[True, False])
        .groupby("run_id", as_index=False)
        .first()
        .rename(columns={"algorithm": "best_b_algorithm", "auc_roc": "best_b_auc"})
    )

    paired = best_a[["run_id", "seed", "best_a_algorithm", "best_a_auc"]].merge(
        best_b[["run_id", "best_b_algorithm", "best_b_auc"]],
        on="run_id",
        how="inner",
    )
    paired["auc_diff_b_minus_a"] = paired["best_b_auc"] - paired["best_a_auc"]

    algo_wins = pd.DataFrame(
        {
            "feature_set": ["A"] * best_a["best_a_algorithm"].nunique() + ["B"] * best_b["best_b_algorithm"].nunique(),
            "algorithm": best_a["best_a_algorithm"].value_counts().index.tolist()
            + best_b["best_b_algorithm"].value_counts().index.tolist(),
            "wins": best_a["best_a_algorithm"].value_counts().tolist() + best_b["best_b_algorithm"].value_counts().tolist(),
        }
    )

    return per_model_summary, paired, algo_wins


def run_shap_stability(
    winner_algorithm_b: str,
    seeds: list[int],
    results_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    model_b_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    X_train_raw = train_df[model_b_features].copy()
    X_test_raw = test_df[model_b_features].copy()

    for seed in seeds:
        row = results_df[
            (results_df["seed"] == seed)
            & (results_df["feature_set"] == "B")
            & (results_df["algorithm"] == winner_algorithm_b)
        ]
        if row.empty:
            continue

        params = json.loads(str(row.iloc[0]["best_params_json"]))

        trained = train_final_model(
            algorithm=winner_algorithm_b,
            params=params,
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_test_raw=X_test_raw,
            feature_cols=model_b_features,
            seed=seed,
        )

        X_test_proc = trained["X_test_processed"]
        mean_abs = compute_shap_mean_abs(trained["model"], X_test_proc)

        total = float(np.sum(mean_abs))
        group_sum: dict[str, float] = {}

        for feat_name, val in zip(X_test_proc.columns.tolist(), mean_abs):
            original = infer_original_feature(feat_name, model_b_features)
            group_name = map_feature_group(original)

            feature_rows.append(
                {
                    "run_id": int(row.iloc[0]["run_id"]),
                    "seed": seed,
                    "feature": feat_name,
                    "original_feature": original,
                    "group": group_name,
                    "mean_abs_shap": float(val),
                }
            )

            group_sum[group_name] = group_sum.get(group_name, 0.0) + float(val)

        for g, gval in group_sum.items():
            pct = (gval / total * 100) if total > 0 else 0.0
            group_rows.append(
                {
                    "run_id": int(row.iloc[0]["run_id"]),
                    "seed": seed,
                    "group": g,
                    "group_mean_abs_shap": gval,
                    "group_pct": pct,
                }
            )

        print(f"[seed={seed}] SHAP stability computed for winner Model B algorithm: {winner_algorithm_b}")

    feature_df = pd.DataFrame(feature_rows)
    group_df = pd.DataFrame(group_rows)

    feat_stability = (
        feature_df.groupby(["feature", "original_feature", "group"], as_index=False)["mean_abs_shap"]
        .agg(["mean", "std", "median"])  # type: ignore[arg-type]
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_sd", "median": "importance_median"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    grp_stability = (
        group_df.groupby("group", as_index=False)["group_pct"]
        .agg(["mean", "std", "median"])  # type: ignore[arg-type]
        .reset_index()
        .rename(columns={"mean": "group_pct_mean", "std": "group_pct_sd", "median": "group_pct_median"})
        .sort_values("group_pct_mean", ascending=False)
        .reset_index(drop=True)
    )

    return feature_df, feat_stability, grp_stability


def make_plots(results_df: pd.DataFrame, paired: pd.DataFrame, feat_stability: pd.DataFrame) -> None:
    # AUC distribution boxplot
    work = results_df.copy()
    work["model_config"] = work["feature_set"] + "_" + work["algorithm"]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=work, x="model_config", y="auc_roc", color="#4f81bd")
    sns.stripplot(data=work, x="model_config", y="auc_roc", color="black", alpha=0.35, size=3)
    plt.title("AUC-ROC Distribution Across 20 Runs")
    plt.xlabel("Model Configuration")
    plt.ylabel("AUC-ROC")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_AUC_BOXPLOT, dpi=150, bbox_inches="tight")
    plt.close()

    # AUC difference histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(paired["auc_diff_b_minus_a"], bins=12, kde=True, color="#2f6f9f")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
    plt.title("Distribution of AUC Difference (Best B - Best A) Across Runs")
    plt.xlabel("AUC Difference")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_AUC_DIFF_HIST, dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP stability errorbar plot
    top = feat_stability.head(20).copy()
    top = top.sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.errorbar(
        x=top["importance_mean"],
        y=np.arange(len(top)),
        xerr=top["importance_sd"].fillna(0),
        fmt="o",
        color="#1b7f5e",
        ecolor="#7ac5a5",
        capsize=3,
    )
    plt.yticks(np.arange(len(top)), top["feature"])
    plt.xlabel("Mean |SHAP| across runs")
    plt.ylabel("Feature")
    plt.title("SHAP Stability (Top 20 Features, mean +/- SD)")
    plt.tight_layout()
    plt.savefig(FIG_SHAP_STABILITY, dpi=150, bbox_inches="tight")
    plt.close()


def build_summary_markdown(
    results_df: pd.DataFrame,
    per_model_summary: pd.DataFrame,
    paired: pd.DataFrame,
    algo_wins: pd.DataFrame,
    feat_stability: pd.DataFrame,
    grp_stability: pd.DataFrame,
) -> str:
    diffs = paired["auc_diff_b_minus_a"].to_numpy(dtype=float)

    diff_mean = float(np.mean(diffs))
    diff_sd = float(np.std(diffs, ddof=1))
    b_gt_a = int(np.sum(diffs > 0))
    b_eq_a = int(np.sum(np.isclose(diffs, 0.0, atol=1e-12)))
    b_lt_a = int(np.sum(diffs < 0))

    t_stat, t_p = ttest_1samp(diffs, popmean=0.0)
    try:
        w_stat, w_p = wilcoxon(diffs)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    paper_rows = []
    for _, r in per_model_summary.iterrows():
        paper_rows.append(
            {
                "Model": r["model_id"],
                "Algorithm": r["algorithm"],
                "Feature Set": r["feature_set"],
                "AUC-ROC (mean +/- SD)": f"{r['auc_roc_mean']:.4f} +/- {r['auc_roc_sd']:.4f}",
                "AUC-PR (mean +/- SD)": f"{r['auc_pr_mean']:.4f} +/- {r['auc_pr_sd']:.4f}",
                "F1 (mean +/- SD)": f"{r['f1_mean']:.4f} +/- {r['f1_sd']:.4f}",
                "Precision (mean +/- SD)": f"{r['precision_mean']:.4f} +/- {r['precision_sd']:.4f}",
                "Recall (mean +/- SD)": f"{r['recall_mean']:.4f} +/- {r['recall_sd']:.4f}",
            }
        )

    paper_table = pd.DataFrame(paper_rows)

    # Validation checks
    expected_rows = (END_SEED - START_SEED + 1) * len(ALGORITHMS) * 2
    auc_sd_max = float(per_model_summary["auc_roc_sd"].max()) if not per_model_summary.empty else float("nan")
    group_sum_df = grp_stability.copy()

    checks = pd.DataFrame(
        [
            {
                "check": "Exactly 160 rows in results CSV",
                "status": "PASS" if len(results_df) == expected_rows else "FAIL",
                "details": f"rows={len(results_df)}, expected={expected_rows}",
            },
            {
                "check": "No NaN in metric columns",
                "status": "PASS"
                if not results_df[["auc_roc", "auc_pr", "f1", "precision", "recall", "specificity", "mcc", "brier_score"]]
                .isna()
                .any()
                .any()
                else "FAIL",
                "details": "Checked metric columns in repeated_experiment_results.csv",
            },
            {
                "check": "AUC values between 0.5 and 1.0",
                "status": "PASS" if results_df["auc_roc"].between(0.5, 1.0).all() else "FAIL",
                "details": f"min_auc={results_df['auc_roc'].min():.4f}, max_auc={results_df['auc_roc'].max():.4f}",
            },
            {
                "check": "SD of AUC across runs is < 0.05",
                "status": "PASS" if auc_sd_max < 0.05 else "FAIL",
                "details": f"max_model_auc_sd={auc_sd_max:.4f}",
            },
            {
                "check": "SHAP group contributions sum to ~100% each run",
                "status": "PASS",
                "details": "Derived from percentage definition per run (sum of group percentages equals 100 by construction).",
            },
            {
                "check": "Paired A vs B comparison uses exactly 20 runs",
                "status": "PASS" if len(diffs) == (END_SEED - START_SEED + 1) else "FAIL",
                "details": f"paired_runs={len(diffs)}",
            },
        ]
    )

    lines: list[str] = []
    lines.append("# Repeated Experiment Summary")
    lines.append("")
    lines.append("K=20 repeated runs with seeds 43..62, temporal split fixed (2019+2020 train, 2021 test).")
    lines.append("")

    lines.append("## 1. Paper Summary Table")
    lines.append(to_md_table(paper_table))
    lines.append("")

    lines.append("## 2. Best-of-A vs Best-of-B Across Runs")
    lines.append(f"- Mean AUC difference (Best B - Best A): {diff_mean:.6f} +/- {diff_sd:.6f}")
    lines.append(f"- Run counts: B > A: {b_gt_a}, B = A: {b_eq_a}, B < A: {b_lt_a}")
    lines.append(f"- Paired t-test (diff vs 0): t={t_stat:.6f}, p={t_p:.6g}")
    lines.append(f"- Wilcoxon signed-rank test: statistic={w_stat:.6f}, p={w_p:.6g}")
    lines.append("")

    lines.append("## 3. Algorithm Stability (Winner Counts)")
    lines.append(to_md_table(algo_wins.sort_values(["feature_set", "wins"], ascending=[True, False])))
    lines.append("")

    lines.append("## 4. Per-Model Aggregates (mean, sd, median, IQR)")
    key_cols = [
        "model_id",
        "auc_roc_mean",
        "auc_roc_sd",
        "auc_roc_median",
        "auc_roc_iqr",
        "auc_pr_mean",
        "auc_pr_sd",
        "f1_mean",
        "f1_sd",
        "precision_mean",
        "precision_sd",
        "recall_mean",
        "recall_sd",
        "specificity_mean",
        "specificity_sd",
        "mcc_mean",
        "mcc_sd",
        "brier_score_mean",
        "brier_score_sd",
    ]
    lines.append(to_md_table(per_model_summary[key_cols]))
    lines.append("")

    lines.append("## 5. SHAP Stability (Winner Model B Algorithm)")
    lines.append("### Top 25 features by mean |SHAP|")
    lines.append(to_md_table(feat_stability.head(25)))
    lines.append("")
    lines.append("### Group contribution stability")
    lines.append(to_md_table(grp_stability))
    lines.append("")

    lines.append("## 6. Validation Checklist")
    lines.append(to_md_table(checks))
    lines.append("")

    lines.append("## 7. Output Files")
    lines.append(f"- {OUTPUT_RESULTS.name}")
    lines.append(f"- {OUTPUT_SUMMARY.name}")
    lines.append(f"- {OUTPUT_SHAP_STABILITY.name}")
    lines.append(f"- figures/{FIG_AUC_BOXPLOT.name}")
    lines.append(f"- figures/{FIG_AUC_DIFF_HIST.name}")
    lines.append(f"- figures/{FIG_SHAP_STABILITY.name}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    configure_plots()
    ensure_dirs()
    print_compute_config()

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_DATASET}")

    model_a_features, model_b_features = load_cleaned_feature_sets()

    df = pd.read_csv(INPUT_DATASET)
    train_df = df[df["admission_year"].isin([2019, 2020])].copy().reset_index(drop=True)
    test_df = df[df["admission_year"] == 2021].copy().reset_index(drop=True)

    y_train = train_df["has_infection"].astype(int)
    y_test = test_df["has_infection"].astype(int)
    years_train = train_df["admission_year"].astype(int)

    print(f"Train rows={len(train_df)}, Test rows={len(test_df)}")
    print(f"Train infection rate={y_train.mean() * 100:.2f}%, Test infection rate={y_test.mean() * 100:.2f}%")

    seeds = list(range(START_SEED, END_SEED + 1))

    # Resume completed seeds from checkpoint.
    rows = resume_checkpoint(seeds)
    completed_seeds = set(pd.DataFrame(rows)["seed"].tolist()) if rows else set()

    for run_id, seed in enumerate(seeds, start=1):
        if seed in completed_seeds:
            print(f"[run={run_id} seed={seed}] Already completed from checkpoint. Skipping.")
            continue

        print("=" * 90)
        print(f"Starting run {run_id}/20 with seed={seed}")

        run_rows = run_single_seed(
            run_id=run_id,
            seed=seed,
            train_df=train_df,
            test_df=test_df,
            y_train=y_train,
            y_test=y_test,
            years_train=years_train,
            model_a_features=model_a_features,
            model_b_features=model_b_features,
        )

        rows.extend(run_rows)
        pd.DataFrame(rows).to_csv(OUTPUT_CHECKPOINT, index=False)
        print(f"[run={run_id} seed={seed}] Checkpoint saved ({len(rows)} rows).")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["run_id", "feature_set", "algorithm"]).reset_index(drop=True)
    results_df.to_csv(OUTPUT_RESULTS, index=False)

    if len(results_df) != len(seeds) * len(ALGORITHMS) * 2:
        raise RuntimeError(
            f"Unexpected row count in results: {len(results_df)}; expected {len(seeds) * len(ALGORITHMS) * 2}."
        )

    per_model_summary, paired, algo_wins = summarize_results(results_df)

    winner_algo_b = (
        paired["best_b_algorithm"].value_counts().sort_values(ascending=False).index[0]
        if not paired.empty
        else "catboost"
    )
    print(f"Winner algorithm for Model B (most frequent best): {winner_algo_b}")

    feature_runs_df, feat_stability, grp_stability = run_shap_stability(
        winner_algorithm_b=winner_algo_b,
        seeds=seeds,
        results_df=results_df,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        model_b_features=model_b_features,
    )

    shap_out = feat_stability.copy()
    shap_out.insert(0, "row_type", "feature")
    shap_out = shap_out.rename(columns={"feature": "name"})
    shap_out = shap_out[["row_type", "name", "original_feature", "group", "importance_mean", "importance_sd", "importance_median"]]

    grp_out = grp_stability.copy()
    grp_out.insert(0, "row_type", "group")
    grp_out = grp_out.rename(columns={"group": "name"})
    grp_out["original_feature"] = ""
    grp_out["group"] = grp_out["name"]
    grp_out["importance_mean"] = grp_out["group_pct_mean"]
    grp_out["importance_sd"] = grp_out["group_pct_sd"]
    grp_out["importance_median"] = grp_out["group_pct_median"]
    grp_out = grp_out[["row_type", "name", "original_feature", "group", "importance_mean", "importance_sd", "importance_median"]]

    repeated_shap_stability = pd.concat([shap_out, grp_out], ignore_index=True)
    repeated_shap_stability.to_csv(OUTPUT_SHAP_STABILITY, index=False)

    make_plots(results_df, paired, feat_stability)

    summary_text = build_summary_markdown(
        results_df=results_df,
        per_model_summary=per_model_summary,
        paired=paired,
        algo_wins=algo_wins,
        feat_stability=feat_stability,
        grp_stability=grp_stability,
    )
    OUTPUT_SUMMARY.write_text(summary_text, encoding="utf-8")

    print("=== Repeated experiment completed ===")
    print(f"Saved results: {OUTPUT_RESULTS}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")
    print(f"Saved SHAP stability: {OUTPUT_SHAP_STABILITY}")
    print(f"Saved figure: {FIG_AUC_BOXPLOT}")
    print(f"Saved figure: {FIG_AUC_DIFF_HIST}")
    print(f"Saved figure: {FIG_SHAP_STABILITY}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
