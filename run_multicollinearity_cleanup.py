from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
INPUT_DATASET = BASE_DIR / "clean_hai_dataset.csv"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_HEATMAP = FIG_DIR / "correlation_matrix.png"
OUTPUT_FEATURES_JSON = BASE_DIR / "cleaned_feature_sets.json"
OUTPUT_REPORT = BASE_DIR / "multicollinearity_cleanup_report.md"

MODEL_A_FEATURES = [
    "age",
    "sex",
    "admission_origin",
    "diagnostic_category",
    "trauma_status",
    "immunosuppression",
    "antibiotic_at_admission",
    "cancer_status",
    "severity_score_igs2",
    "intubation_status",
    "reintubation_status",
    "intubation_days",
    "urinary_catheter",
    "urinary_catheter_days",
    "central_line_count",
    "ecmo_status",
    "length_of_stay",
    "admission_month",
    "admission_weekday",
    "weekend_admission",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + [
    "bed_occupancy",
    "patient_turnover",
    "unit_avg_los",
    "national_avg_los",
    "los_ratio_national",
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
]

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

ETP_COUNT_PAIRS = [
    ("nurse_staffing_etp", "nurse_staffing_count"),
    ("nurse_aide_staffing_etp", "nurse_aide_staffing_count"),
    ("nurse_anesthetist_staffing_etp", "nurse_anesthetist_staffing_count"),
    ("dietitian_staffing_etp", "dietitian_staffing_count"),
    ("medical_admin_assistant_staffing_etp", "medical_admin_assistant_staffing_count"),
    ("total_staffing_etp", "total_staffing_count"),
]


def to_md_table(df: pd.DataFrame, max_rows: int = 250) -> str:
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


def prepare_numeric_training(df_train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    work = df_train[MODEL_B_FEATURES].copy()
    work = apply_duration_rules(work)

    numeric_cols = [c for c in MODEL_B_FEATURES if c not in CATEGORICAL_COLUMNS]
    medians: dict[str, float] = {}

    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        med = float(work[c].median()) if not np.isnan(work[c].median()) else 0.0
        medians[c] = med
        work[c] = work[c].fillna(med)

    return work[numeric_cols].copy(), medians, numeric_cols


def correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            r = float(corr.loc[c1, c2])
            if abs(r) > threshold:
                rows.append(
                    {
                        "feature_1": c1,
                        "feature_2": c2,
                        "r": r,
                        "abs_r": abs(r),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return out


def compute_vif(df_numeric: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df_numeric.columns:
        y = df_numeric[col].to_numpy(dtype=float)
        X = df_numeric.drop(columns=[col]).to_numpy(dtype=float)

        if X.shape[1] == 0:
            vif = 1.0
        else:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            y_hat = X_design @ beta

            sse = float(np.sum((y - y_hat) ** 2))
            sst = float(np.sum((y - np.mean(y)) ** 2))

            if sst <= 1e-12:
                vif = float("inf")
            else:
                r2 = 1.0 - (sse / sst)
                if r2 >= 0.999999:
                    vif = float("inf")
                else:
                    vif = 1.0 / (1.0 - r2)

        rows.append({"feature": col, "vif": float(vif)})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    return out


def save_heatmap(corr: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Model B Numeric Features Correlation Matrix (Train Set)")
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET)
    train = df[df["admission_year"].isin([2019, 2020])].copy()

    if train.empty:
        raise RuntimeError("Training split is empty.")

    numeric_train, medians, numeric_cols = prepare_numeric_training(train)

    corr = numeric_train.corr(method="pearson")
    high80 = correlation_pairs(corr, 0.80)
    high90 = correlation_pairs(corr, 0.90)

    save_heatmap(corr)

    vif_df = compute_vif(numeric_train)

    dropped: dict[str, str] = {}
    flagged_keep: list[dict[str, Any]] = []

    # Rule 1: ETP vs Count pairs
    for etp_col, count_col in ETP_COUNT_PAIRS:
        if etp_col in corr.columns and count_col in corr.columns:
            r = float(corr.loc[etp_col, count_col])
            if abs(r) > 0.85:
                dropped[count_col] = (
                    f"Dropped by Rule 1: correlated with {etp_col} at r={r:.4f} (|r|>0.85); keep ETP over count."
                )

    # Rule 2: category-level ETP vs total ETP (flag only)
    for etp_col, _ in ETP_COUNT_PAIRS:
        if etp_col == "total_staffing_etp":
            continue
        if etp_col in corr.columns and "total_staffing_etp" in corr.columns:
            r = float(corr.loc[etp_col, "total_staffing_etp"])
            if abs(r) > 0.90:
                flagged_keep.append(
                    {
                        "feature_1": etp_col,
                        "feature_2": "total_staffing_etp",
                        "r": r,
                        "decision": "Flagged by Rule 2 (kept both for interpretability)",
                    }
                )

    # Rule 3: near-zero variance
    std_series = numeric_train.std(ddof=0)
    for col, std_val in std_series.items():
        if float(std_val) < 0.01 and col not in dropped:
            dropped[col] = f"Dropped by Rule 3: near-zero variance (std={std_val:.6f} < 0.01)."

    # Rule 4: environmental redundancy
    env_cols = {"unit_avg_los", "national_avg_los", "los_ratio_national"}
    if env_cols.issubset(set(numeric_train.columns)):
        calc_ratio = numeric_train["unit_avg_los"] / numeric_train["national_avg_los"]
        observed = numeric_train["los_ratio_national"]
        mask = np.isfinite(calc_ratio) & np.isfinite(observed)

        deterministic = bool(np.allclose(calc_ratio[mask], observed[mask], rtol=1e-4, atol=1e-4)) if mask.any() else False
        if deterministic:
            if "unit_avg_los" not in dropped:
                dropped["unit_avg_los"] = (
                    "Dropped by Rule 4: los_ratio_national is deterministic unit_avg_los/national_avg_los; "
                    "kept los_ratio_national only."
                )
            if "national_avg_los" not in dropped:
                dropped["national_avg_los"] = (
                    "Dropped by Rule 4: los_ratio_national is deterministic unit_avg_los/national_avg_los; "
                    "kept los_ratio_national only."
                )

    model_a_cleaned = MODEL_A_FEATURES.copy()
    model_b_cleaned = [c for c in MODEL_B_FEATURES if c not in dropped]

    decision_rows = []
    for f in MODEL_B_FEATURES:
        if f in dropped:
            decision_rows.append({"feature": f, "decision": "DROP", "reason": dropped[f]})
        else:
            decision_rows.append({"feature": f, "decision": "KEEP", "reason": "Retained after rules."})

    decision_df = pd.DataFrame(decision_rows)

    payload = {
        "metadata": {
            "train_rows": int(len(train)),
            "train_years": [2019, 2020],
            "note": "Computed on training split after numeric imputation and before one-hot encoding.",
        },
        "model_a_cleaned": model_a_cleaned,
        "model_b_cleaned": model_b_cleaned,
        "dropped_features": dropped,
        "flagged_keep_pairs": flagged_keep,
        "numeric_medians_train": medians,
    }

    OUTPUT_FEATURES_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build markdown report.
    lines: list[str] = []
    lines.append("# Multicollinearity Cleanup Report")
    lines.append("")
    lines.append("Computed on training set only (2019+2020), after imputation and before encoding.")
    lines.append("")

    lines.append("## 1. Correlation Pairs with |r| > 0.80")
    lines.append(to_md_table(high80))
    lines.append("")

    lines.append("## 2. Correlation Pairs with |r| > 0.90")
    lines.append(to_md_table(high90))
    lines.append("")

    focus_rows = []
    focus_targets = [
        ("nurse_staffing_etp", "nurse_staffing_count"),
        ("nurse_aide_staffing_etp", "nurse_aide_staffing_count"),
        ("total_staffing_etp", "total_staffing_count"),
        ("nurse_staffing_etp", "total_staffing_etp"),
    ]
    for a, b in focus_targets:
        if a in corr.columns and b in corr.columns:
            focus_rows.append({"feature_1": a, "feature_2": b, "r": float(corr.loc[a, b])})

    lines.append("## 3. Focus Correlations")
    lines.append(to_md_table(pd.DataFrame(focus_rows)))
    lines.append("")

    lines.append("## 4. VIF Table (sorted)")
    lines.append(to_md_table(vif_df))
    lines.append("")

    lines.append("## 5. Rule-Based Decisions")
    lines.append(to_md_table(decision_df))
    lines.append("")

    lines.append("## 6. Flagged-but-Kept Pairs (Rule 2)")
    lines.append(to_md_table(pd.DataFrame(flagged_keep)))
    lines.append("")

    lines.append("## 7. Final Cleaned Feature Sets")
    lines.append(f"- Model A cleaned feature count: {len(model_a_cleaned)}")
    lines.append(f"- Model B cleaned feature count: {len(model_b_cleaned)}")
    lines.append(f"- Cleaned feature JSON: {OUTPUT_FEATURES_JSON.name}")
    lines.append(f"- Correlation heatmap: {OUTPUT_HEATMAP.as_posix()}")
    lines.append("")

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Multicollinearity Cleanup Complete ===")
    print(f"Training rows: {len(train)}")
    print(f"Numeric feature count (Model B): {len(numeric_cols)}")
    print(f"Pairs |r|>0.80: {len(high80)}")
    print(f"Pairs |r|>0.90: {len(high90)}")
    print("\nTop VIF rows:")
    print(vif_df.head(15).to_string(index=False))
    print("\nDropped features:")
    if dropped:
        for k, v in dropped.items():
            print(f"- {k}: {v}")
    else:
        print("- None")

    print("\nCleaned Model B features:")
    print(model_b_cleaned)

    print(f"\nSaved heatmap: {OUTPUT_HEATMAP}")
    print(f"Saved cleaned feature sets: {OUTPUT_FEATURES_JSON}")
    print(f"Saved report: {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
