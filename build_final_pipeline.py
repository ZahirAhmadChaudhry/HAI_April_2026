from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

SPIADI_2019 = BASE_DIR / "Data/Patient_data/Orginal/2019 rea G spiadi.xlsx"
SPIADI_2020 = BASE_DIR / "Data/Patient_data/Orginal/ImportRéa Inf ADI AD_PED-115fiches 2020.xlsx"
SPIADI_2021 = BASE_DIR / "Data/Patient_data/Orginal/rea inf ADI 2021.xlsx"
ETP_FILE = BASE_DIR / "Data/Organizational_data/ETP_1908/Original/ETP Activité journalier de 2018 à 2021 pour le service Réanimation Médicale-ORIGINAL.xlsx"
ORG_FILE = BASE_DIR / "Data/Organizational_data/Original/Organizational_data_original.xlsx"

OUTPUT_DATASET = BASE_DIR / "clean_hai_dataset.csv"
OUTPUT_REPORT = BASE_DIR / "dataset_validation_report.md"
OUTPUT_FEATURE_DICT = BASE_DIR / "feature_dictionary.csv"

ORIGIN_MAPPING_2019 = {
    "DOM": 1,
    "REA": 2,
    "MCO": 3,
    "SSR": 4,
    "PSY": 5,
    "SLD": 6,
    "EHP": 7,
    "AUT": 8,
    "NC": 9,
}

CATEGORICAL_EXPECTED = {
    "sex": {1, 2, 3},
    "diagnosis": {1, 2, 3, 9},
    "immuno": {1, 2, 3, 9},
    "antibio": {1, 2, 9},
    "cancer": {1, 2, 3, 9},
    "trauma": {1, 2, 9},
    "intubated": {1, 2, 9},
    "reintubated": {1, 2},
    "urinary": {1, 2},
    "ecmo": {1, 2, 3, 9},
    "dead": {1, 2},
    "origin": {1, 2, 3, 4, 5, 6, 7, 8, 9},
}

STAFF_CATEGORIES = [
    "general_nurse",
    "nursing_assistant",
    "nurse_anesthetist",
    "dietitian",
    "hospital_services",
    "admin_assistant",
    "medical_admin_assistant",
    "social_worker",
    "psychologist",
    "clinical_research_associate",
    "clinical_study_technician",
]

PROTECTED_STAFF_CATEGORIES = {
    "general_nurse",
    "nursing_assistant",
    "nurse_anesthetist",
    "dietitian",
    "hospital_services",
    "admin_assistant",
    "medical_admin_assistant",
    "all_staff",
}


# ============================================================
# Logging and utilities
# ============================================================
@dataclass
class PipelineLogger:
    lines: list[str]

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)



def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def compact_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text



def find_col(columns: list[str], candidates: list[str]) -> str | None:
    mapping = {c: compact_text(c) for c in columns}
    for cand in candidates:
        cn = compact_text(cand)
        for c, n in mapping.items():
            if n == cn:
                return c
    for cand in candidates:
        cn = compact_text(cand)
        for c, n in mapping.items():
            if cn in n:
                return c
    return None



def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)



def md_escape(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")



def to_md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"
    t = df.copy()
    if max_rows is not None and len(t) > max_rows:
        t = t.head(max_rows)
    headers = [md_escape(c) for c in t.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in t.iterrows():
        lines.append("| " + " | ".join(md_escape(v) for v in row.tolist()) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_Shown first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)



def describe_continuous(series: pd.Series) -> str:
    s = to_numeric(series).dropna()
    if s.empty:
        return "NA"
    mean = s.mean()
    std = s.std()
    median = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"mean={mean:.2f}, sd={std:.2f}; median={median:.2f}, IQR=[{q1:.2f}, {q3:.2f}]"



def describe_categorical(series: pd.Series) -> str:
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return "NA"
    parts = []
    total = len(series)
    for val, cnt in vc.items():
        pct = (cnt / total * 100) if total else 0
        parts.append(f"{val}: {cnt} ({pct:.1f}%)")
    return "; ".join(parts)



def safe_mannwhitney(series_a: pd.Series, series_b: pd.Series) -> float:
    a = to_numeric(series_a).dropna()
    b = to_numeric(series_b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan



def safe_chi_square(series: pd.Series, target: pd.Series) -> float:
    if series.dropna().empty or target.dropna().empty:
        return np.nan
    try:
        table = pd.crosstab(series.fillna("<NA>"), target.fillna("<NA>"))
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        return float(chi2_contingency(table)[1])
    except Exception:
        return np.nan


# ============================================================
# SPIADI processing
# ============================================================

def load_spiadi_workbook(path: Path, year: int, logger: PipelineLogger) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    candidate_sheets = [s for s in xls.sheet_names if "spiadi" in normalize_text(s) and str(year) in normalize_text(s)]
    if candidate_sheets:
        sheet = candidate_sheets[0]
    else:
        # Fallback to largest sheet if name matching is not reliable.
        sheet = max(xls.sheet_names, key=lambda s: xls.parse(s).shape[0] * max(xls.parse(s).shape[1], 1))

    df = xls.parse(sheet)
    logger.log(f"Loaded SPIADI {year}: file={path.name}, sheet={sheet}, shape={df.shape[0]}x{df.shape[1]}")
    return df



def harmonize_spiadi_columns(df: pd.DataFrame, year: int, logger: PipelineLogger) -> pd.DataFrame:
    columns = list(df.columns)

    source_map = {
        "entry": ["entry"],
        "exit": ["exit"],
        "age_years": ["age_years", "age"],
        "sex": ["sex"],
        "origin": ["origin"],
        "diagnosis": ["diagnosis"],
        "trauma": ["trauma"],
        "immuno": ["immuno"],
        "antibio": ["antibio"],
        "cancer": ["cancer"],
        "igsII": ["igsII", "igsii"],
        "intubated": ["intubated"],
        "reintubated": ["reintubated"],
        "intubation_duration": ["intubation_duration"],
        "urinary": ["urinary"],
        "urinary_duration": ["urinary_duration"],
        "kt_count": ["kt_count"],
        "ecmo": ["ecmo"],
        "dead": ["dead"],
        "bact_count": ["bact_count"],
        "pneu_count": ["pneu_count"],
        "uf": ["uf"],
        "birthdate": ["birthdate"],
    }

    out = pd.DataFrame(index=df.index)

    for unified, candidates in source_map.items():
        found = find_col(columns, candidates)
        if found is None:
            if unified == "cancer" and year == 2019:
                out[unified] = 9
                logger.log("2019 cancer column missing -> filled with code 9 (unknown).")
            else:
                out[unified] = np.nan
                logger.log(f"Column missing for {year}: {unified} -> filled with NaN")
        else:
            out[unified] = df[found]

    out["source_year"] = year

    # Type casting
    out["entry"] = to_datetime(out["entry"])
    out["exit"] = to_datetime(out["exit"])
    out["birthdate"] = to_datetime(out["birthdate"])

    numeric_cols = [
        "age_years",
        "sex",
        "diagnosis",
        "trauma",
        "immuno",
        "antibio",
        "cancer",
        "igsII",
        "intubated",
        "reintubated",
        "intubation_duration",
        "urinary",
        "urinary_duration",
        "kt_count",
        "ecmo",
        "dead",
        "bact_count",
        "pneu_count",
        "uf",
    ]

    for col in numeric_cols:
        out[col] = to_numeric(out[col])

    return out


# ============================================================
# ETP processing
# ============================================================

def map_staff_category(label: Any) -> str | None:
    txt = normalize_text(label)
    if not txt or txt == "nan":
        return None

    if "infirmier anesth" in txt or "iade" in txt or "cadre de sante" in txt or "cadre superieur de sante" in txt:
        return "nurse_anesthetist"
    if "aide soignant" in txt:
        return "nursing_assistant"
    if "infirmier" in txt:
        return "general_nurse"
    if "dietet" in txt:
        return "dietitian"
    if "service hospitalier" in txt or re.search(r"\bash\b", txt):
        return "hospital_services"
    if "assistant medico administratif" in txt or "assistant medico administr" in txt:
        return "medical_admin_assistant"
    if "assistant administr" in txt:
        return "admin_assistant"
    if "assistant social" in txt:
        return "social_worker"
    if "psycholog" in txt:
        return "psychologist"
    if "attache de recherche clinique" in txt or "recherche clinique" in txt:
        return "clinical_research_associate"
    if "technicien d'etudes cliniques" in txt or "technicien d etudes cliniques" in txt or "etude clinique" in txt:
        return "clinical_study_technician"

    return None



def is_real_etp_metric(metric_text: Any) -> bool:
    t = normalize_text(metric_text)
    if not t:
        return False
    real_hit = any(k in t for k in ["reel", "reelle", "real", "travaille"])
    theoretical_hit = any(k in t for k in ["theorique", "theory", "remunere"])
    return real_hit and not theoretical_hit



def extract_daily_etp_table(path: Path, logger: PipelineLogger) -> tuple[pd.DataFrame, dict[str, Any]]:
    xls = pd.ExcelFile(path)
    years = ["2019", "2020", "2021"]

    records: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}

    for year in years:
        if year not in xls.sheet_names:
            logger.log(f"ETP sheet missing: {year}")
            continue

        raw = xls.parse(year, header=None)
        top_row = raw.iloc[3]
        metric_row = raw.iloc[4]
        data = raw.iloc[5:].copy().reset_index(drop=True)

        if data.shape[1] < 10:
            logger.log(f"ETP sheet {year} has unexpected structure.")
            continue

        emploi = data.iloc[:, 8]
        valid_rows = emploi.notna()
        data = data.loc[valid_rows].reset_index(drop=True)
        emploi = emploi.loc[valid_rows].astype(str).reset_index(drop=True)

        last_date = pd.NaT
        real_col_count = 0
        date_col_count = 0

        for col_idx in range(raw.shape[1]):
            date_val = pd.to_datetime(top_row.iloc[col_idx], errors="coerce")
            if pd.notna(date_val):
                last_date = date_val.normalize()
                date_col_count += 1

            if pd.isna(last_date):
                continue

            if not is_real_etp_metric(metric_row.iloc[col_idx]):
                continue

            if col_idx >= data.shape[1]:
                continue

            vals = to_numeric(data.iloc[:, col_idx])
            cats = emploi.map(map_staff_category)

            part = pd.DataFrame(
                {
                    "date": [last_date] * len(vals),
                    "category": cats,
                    "etp": vals,
                }
            )
            part = part[part["category"].notna()].copy()
            part["count"] = (part["etp"].fillna(0) > 0).astype(int)

            records.append(part[["date", "category", "etp", "count"]])
            real_col_count += 1

        metadata[year] = {
            "sheet_shape": raw.shape,
            "employee_rows": int(len(data)),
            "date_columns_detected": date_col_count,
            "real_columns_detected": real_col_count,
        }

    if not records:
        raise RuntimeError("No ETP records extracted from original ETP workbook.")

    long_df = pd.concat(records, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce").dt.normalize()

    # Surveillance window: Jan-Jun of 2019-2021.
    long_df = long_df[
        (long_df["date"].dt.year.isin([2019, 2020, 2021]))
        & (long_df["date"].dt.month >= 1)
        & (long_df["date"].dt.month <= 6)
    ].copy()

    agg = (
        long_df.groupby(["date", "category"], dropna=False)
        .agg(etp=("etp", "sum"), count=("count", "sum"))
        .reset_index()
    )

    etp_wide = agg.pivot(index="date", columns="category", values="etp").fillna(0)
    count_wide = agg.pivot(index="date", columns="category", values="count").fillna(0)

    for cat in STAFF_CATEGORIES:
        if cat not in etp_wide.columns:
            etp_wide[cat] = 0.0
        if cat not in count_wide.columns:
            count_wide[cat] = 0.0

    etp_wide = etp_wide[STAFF_CATEGORIES]
    count_wide = count_wide[STAFF_CATEGORIES]

    etp_wide.columns = [f"etp_{c}" for c in etp_wide.columns]
    count_wide.columns = [f"count_{c}" for c in count_wide.columns]

    daily = pd.concat([etp_wide, count_wide], axis=1)

    # Force complete 544-day Jan-Jun date index for 2019-2021.
    full_dates = []
    for y in [2019, 2020, 2021]:
        full_dates.extend(pd.date_range(f"{y}-01-01", f"{y}-06-30", freq="D"))
    full_index = pd.DatetimeIndex(full_dates, name="date")

    daily = daily.reindex(full_index).fillna(0)

    etp_cols = [c for c in daily.columns if c.startswith("etp_")]
    count_cols = [c for c in daily.columns if c.startswith("count_")]
    daily["etp_all_staff"] = daily[etp_cols].sum(axis=1)
    daily["count_all_staff"] = daily[count_cols].sum(axis=1)

    daily = daily.reset_index()

    logger.log(f"ETP daily table built: shape={daily.shape[0]}x{daily.shape[1]} (expected around 544x25)")
    for year, meta in metadata.items():
        logger.log(f"ETP {year} metadata: {meta}")

    return daily, metadata



def aggregate_etp_over_stays(df: pd.DataFrame, daily_etp: pd.DataFrame, logger: PipelineLogger) -> tuple[pd.DataFrame, list[str]]:
    etp_cols = [c for c in daily_etp.columns if c.startswith("etp_")]
    count_cols = [c for c in daily_etp.columns if c.startswith("count_")]

    rows = []
    no_coverage_count = 0

    for _, row in df.iterrows():
        start = row["entry"]
        end = row["exit"]

        if pd.isna(start):
            rows.append({f"mean_{c}": np.nan for c in etp_cols + count_cols})
            no_coverage_count += 1
            continue

        start = pd.Timestamp(start).normalize()
        if pd.isna(end) or pd.Timestamp(end) < pd.Timestamp(start):
            end = start
        else:
            end = pd.Timestamp(end).normalize()

        window = daily_etp[(daily_etp["date"] >= start) & (daily_etp["date"] <= end)]

        if window.empty:
            rows.append({f"mean_{c}": np.nan for c in etp_cols + count_cols})
            no_coverage_count += 1
            continue

        rows.append({f"mean_{c}": float(window[c].mean()) for c in etp_cols + count_cols})

    agg = pd.DataFrame(rows, index=df.index)

    # Drop near-zero variance staffing categories (except protected categories and all_staff).
    dropped = []
    categories = sorted(
        set(c.replace("mean_etp_", "") for c in agg.columns if c.startswith("mean_etp_"))
    )

    for cat in categories:
        if cat in PROTECTED_STAFF_CATEGORIES:
            continue

        etp_col = f"mean_etp_{cat}"
        cnt_col = f"mean_count_{cat}"
        if etp_col not in agg.columns or cnt_col not in agg.columns:
            continue

        etp_series = agg[etp_col].fillna(0)
        cnt_series = agg[cnt_col].fillna(0)
        var_score = max(float(etp_series.var()), float(cnt_series.var()))
        non_zero_ratio = float(((etp_series > 0) | (cnt_series > 0)).mean())

        if var_score < 1e-4 or non_zero_ratio < 0.05:
            agg = agg.drop(columns=[etp_col, cnt_col])
            dropped.append(cat)

    logger.log(f"Aggregated ETP exposure table shape={agg.shape[0]}x{agg.shape[1]}")
    logger.log(f"Patients with no ETP coverage in stay window: {no_coverage_count}")
    logger.log(f"Dropped near-zero staffing categories: {dropped}")

    return agg, dropped


# ============================================================
# Organizational metrics and environmental features
# ============================================================

def parse_mois_annee(series: pd.Series) -> pd.DataFrame:
    txt = series.astype(str).str.strip()
    extracted = txt.str.extract(r"(?P<year>\d{4})\s*[-/]\s*(?P<month>\d{1,2})")
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(extracted["year"], errors="coerce"),
            "month": pd.to_numeric(extracted["month"], errors="coerce"),
        }
    )
    return out



def load_org_monthly_metrics(path: Path, logger: PipelineLogger) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = "Activity_base" if "Activity_base" in xls.sheet_names else xls.sheet_names[0]
    df = xls.parse(sheet)

    period_col = find_col(list(df.columns), ["Mois + Année", "Mois + Annee", "Mois + Ann�e", "mois"])
    gross_col = find_col(list(df.columns), ["Durée de séjour brute", "Duree de sejour brute", "Dur�e de s�jour brute"])
    national_col = find_col(list(df.columns), ["DMS Nationale", "DMS nationale"])
    ratio_col = find_col(
        list(df.columns),
        [
            "Ratio durée de séjour sur DMS nationale",
            "Ratio duree de sejour sur DMS nationale",
            "Ratio dur�e de s�jour sur DMS nationale",
            "ratio",
        ],
    )

    if period_col is None:
        raise RuntimeError("Could not find period column in organizational workbook.")

    period_parts = parse_mois_annee(df[period_col])
    df = df.copy()
    df["period_year"] = period_parts["year"]
    df["period_month"] = period_parts["month"]

    keep_cols = ["period_year", "period_month"]
    rename_map = {}

    if gross_col is not None:
        keep_cols.append(gross_col)
        rename_map[gross_col] = "unit_gross_los"
    if national_col is not None:
        keep_cols.append(national_col)
        rename_map[national_col] = "national_avg_los"
    if ratio_col is not None:
        keep_cols.append(ratio_col)
        rename_map[ratio_col] = "los_ratio_vs_national"

    tmp = df[keep_cols].copy()

    for c in keep_cols:
        if c not in ["period_year", "period_month"]:
            tmp[c] = to_numeric(tmp[c])

    grouped = (
        tmp.dropna(subset=["period_year", "period_month"])
        .groupby(["period_year", "period_month"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns=rename_map)
    )

    grouped["period_year"] = grouped["period_year"].astype(int)
    grouped["period_month"] = grouped["period_month"].astype(int)

    logger.log(f"Organizational monthly metrics table shape={grouped.shape[0]}x{grouped.shape[1]}")
    return grouped



def compute_environment_features(df: pd.DataFrame, logger: PipelineLogger) -> tuple[pd.DataFrame, int, int]:
    out = df.copy()
    entry_dates = out["entry"].dt.normalize()
    exit_dates = out["exit"].dt.normalize()

    valid_los_days = (out["exit"] - out["entry"]).dt.days.dropna()
    median_los_days = int(round(float(valid_los_days.median()))) if not valid_los_days.empty else 0

    missing_exit_mask = exit_dates.isna() & entry_dates.notna()
    missing_exit_count = int(missing_exit_mask.sum())

    exit_for_occupancy = exit_dates.copy()
    if missing_exit_count > 0:
        exit_for_occupancy.loc[missing_exit_mask] = (
            entry_dates.loc[missing_exit_mask] + pd.to_timedelta(median_los_days, unit="D")
        ).dt.normalize()

    bed_occupancy = []
    turnover = []

    for i, d in entry_dates.items():
        if pd.isna(d):
            bed_occupancy.append(np.nan)
            turnover.append(np.nan)
            continue

        overlap = (entry_dates <= d) & (exit_for_occupancy >= d)
        occ = int(overlap.sum() - 1)
        if occ < 0:
            occ = 0

        admissions = int((entry_dates == d).sum())
        discharges = int((exit_for_occupancy == d).sum())

        bed_occupancy.append(occ)
        turnover.append(admissions + discharges)

    out["bed_occupancy_at_admission"] = bed_occupancy
    out["patient_turnover_at_admission"] = turnover

    logger.log(
        "For occupancy/turnover only, estimated exit for "
        f"{missing_exit_count} patients with missing exit using median LOS={median_los_days} days."
    )
    logger.log("Computed bed occupancy and patient turnover at admission.")
    return out, missing_exit_count, median_los_days


# ============================================================
# Final assembly and reporting
# ============================================================

def generate_feature_dictionary(df: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    rows = []

    def classify(col: str) -> str:
        if col == "has_infection":
            return "target"
        if col in {"age", "sex", "admission_origin"}:
            return "patient_demographics"
        if col in {"severity_score_igs2", "immunosuppression", "cancer_status", "diagnostic_category", "trauma_status"}:
            return "clinical_severity"
        if col in {
            "intubation_status",
            "reintubation_status",
            "intubation_days",
            "urinary_catheter",
            "urinary_catheter_days",
            "central_line_count",
            "ecmo_status",
            "icu_mortality",
            "antibiotic_at_admission",
        }:
            return "medical_procedures"
        if col.startswith("mean_etp_") or col.startswith("mean_count_") or col.endswith("staffing_etp") or col.endswith("staffing_count"):
            return "organizational_staffing"
        if col in {"bed_occupancy", "patient_turnover", "unit_avg_los", "national_avg_los", "los_ratio_national"}:
            return "organizational_environment"
        if col in {"admission_year", "admission_month", "admission_weekday", "weekend_admission", "length_of_stay"}:
            return "temporal"
        return "clinical_severity"

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            valid = to_numeric(s).dropna()
            if valid.empty:
                value_range = "NA"
            else:
                value_range = f"{valid.min():.4g} .. {valid.max():.4g}"
        else:
            vals = s.dropna().astype(str).unique().tolist()
            vals = vals[:20]
            value_range = ", ".join(vals)

        rows.append(
            {
                "column_name": col,
                "description": descriptions.get(col, ""),
                "data_type": dtype,
                "value_range": value_range,
                "n_missing": n_missing,
                "n_unique": n_unique,
                "category": classify(col),
            }
        )

    return pd.DataFrame(rows)



def build_validation_report(
    logger: PipelineLogger,
    flow_df: pd.DataFrame,
    final_df: pd.DataFrame,
    pre_drop_df: pd.DataFrame,
    dropped_staff_categories: list[str],
    validation_checks: list[dict[str, Any]],
    feature_dict: pd.DataFrame,
    etp_no_coverage: int,
    occupancy_missing_exit_count: int,
    occupancy_median_los_days: int,
) -> str:
    lines: list[str] = []

    lines.append("# Dataset Validation Report")
    lines.append("")
    lines.append("Generated by build_final_pipeline.py using only copied raw files in HAI_April_2026/Data.")
    lines.append("")

    lines.append("## 1. Patient Flow (CONSORT-style)")
    lines.append(to_md_table(flow_df))
    lines.append("")

    lines.append("## 2. Descriptive Statistics (Infected vs Non-infected)")
    target = "has_infection"

    desc_rows = []
    for col in final_df.columns:
        if col == target:
            continue

        s = final_df[col]
        infected = final_df[final_df[target] == 1][col]
        non_infected = final_df[final_df[target] == 0][col]

        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
            p_val = safe_mannwhitney(infected, non_infected)
            desc_rows.append(
                {
                    "feature": col,
                    "type": "continuous",
                    "missing": int(s.isna().sum()),
                    "infected_summary": describe_continuous(infected),
                    "non_infected_summary": describe_continuous(non_infected),
                    "p_value": p_val,
                }
            )
        else:
            p_val = safe_chi_square(s, final_df[target])
            desc_rows.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "missing": int(s.isna().sum()),
                    "infected_summary": describe_categorical(infected),
                    "non_infected_summary": describe_categorical(non_infected),
                    "p_value": p_val,
                }
            )

    desc_df = pd.DataFrame(desc_rows)
    lines.append(to_md_table(desc_df))
    lines.append("")

    lines.append("### 2.1 By-year infected vs non-infected summaries")
    by_year_rows = []
    for year in sorted(final_df["admission_year"].dropna().unique().astype(int).tolist()):
        year_df = final_df[final_df["admission_year"] == year]
        for col in final_df.columns:
            if col == target:
                continue
            y_inf = year_df[year_df[target] == 1][col]
            y_non = year_df[year_df[target] == 0][col]
            if pd.api.types.is_numeric_dtype(year_df[col]) and year_df[col].nunique(dropna=True) > 10:
                p = safe_mannwhitney(y_inf, y_non)
                summary_inf = describe_continuous(y_inf)
                summary_non = describe_continuous(y_non)
            else:
                p = safe_chi_square(year_df[col], year_df[target])
                summary_inf = describe_categorical(y_inf)
                summary_non = describe_categorical(y_non)
            by_year_rows.append(
                {
                    "year": year,
                    "feature": col,
                    "infected_summary": summary_inf,
                    "non_infected_summary": summary_non,
                    "p_value": p,
                }
            )
    by_year_df = pd.DataFrame(by_year_rows)
    lines.append(to_md_table(by_year_df, max_rows=400))
    lines.append("")

    lines.append("## 3. Target Variable Summary")
    target_counts = final_df[target].value_counts().rename_axis("has_infection").reset_index(name="count")
    target_counts["pct"] = (target_counts["count"] / len(final_df) * 100).round(2)
    lines.append(to_md_table(target_counts))
    lines.append("")

    rate_by_year = final_df.groupby("admission_year")[target].mean().reset_index(name="infection_rate")
    rate_by_year["infection_rate_pct"] = (rate_by_year["infection_rate"] * 100).round(2)
    lines.append("### Infection rate by year")
    lines.append(to_md_table(rate_by_year))
    lines.append("")

    bact_rate = pre_drop_df.groupby("admission_year")["bact_count"].apply(lambda s: (s.fillna(0) > 0).mean()).reset_index(name="bact_rate")
    bact_rate["bact_rate_pct"] = (bact_rate["bact_rate"] * 100).round(2)
    pneu_rate = pre_drop_df.groupby("admission_year")["pneu_count"].apply(lambda s: (s.fillna(0) > 0).mean()).reset_index(name="pneu_rate")
    pneu_rate["pneu_rate_pct"] = (pneu_rate["pneu_rate"] * 100).round(2)

    lines.append("### Bacteremia rate by year")
    lines.append(to_md_table(bact_rate))
    lines.append("")
    lines.append("### Pneumonia rate by year")
    lines.append(to_md_table(pneu_rate))
    lines.append("")

    lines.append("## 4. Missing Data Summary")
    miss = pd.DataFrame(
        {
            "column": final_df.columns,
            "n_missing": final_df.isna().sum().values,
            "pct_missing": (final_df.isna().sum().values / len(final_df) * 100).round(2),
        }
    ).sort_values("pct_missing", ascending=False)
    lines.append(to_md_table(miss))
    lines.append("")

    miss_year_rows = []
    for col in final_df.columns:
        for year, grp in final_df.groupby("admission_year"):
            m = int(grp[col].isna().sum())
            miss_year_rows.append(
                {
                    "column": col,
                    "year": int(year),
                    "n_missing": m,
                    "pct_missing": round(m / len(grp) * 100, 2),
                }
            )
    lines.append("### Missingness pattern by year")
    lines.append(to_md_table(pd.DataFrame(miss_year_rows), max_rows=400))
    lines.append("")

    lines.append("## 5. Data Quality Verification")
    lines.append(to_md_table(pd.DataFrame(validation_checks)))
    lines.append("")

    lines.append("## 6. Temporal Split Preview")
    train = final_df[final_df["admission_year"].isin([2019, 2020])]
    test = final_df[final_df["admission_year"] == 2021]

    split_rows = []
    for split_name, split_df in [("train_2019_2020", train), ("test_2021", test)]:
        split_rows.append(
            {
                "split": split_name,
                "n": len(split_df),
                "infection_rate_pct": round(split_df[target].mean() * 100, 2),
                "age_summary": describe_continuous(split_df["age"]),
                "severity_summary": describe_continuous(split_df["severity_score_igs2"]),
            }
        )

    lines.append(to_md_table(pd.DataFrame(split_rows)))
    lines.append("")

    age_p = safe_mannwhitney(train["age"], test["age"])
    sev_p = safe_mannwhitney(train["severity_score_igs2"], test["severity_score_igs2"])
    inf_p = safe_chi_square(final_df["admission_year"].map(lambda y: "train" if y in [2019, 2020] else "test"), final_df[target])

    lines.append("### Train vs Test distribution comparisons")
    comp_df = pd.DataFrame(
        {
            "comparison": ["age", "severity_score_igs2", "infection_rate"],
            "p_value": [age_p, sev_p, inf_p],
        }
    )
    lines.append(to_md_table(comp_df))
    lines.append("")

    lines.append("## 7. Feature Dictionary")
    lines.append(to_md_table(feature_dict))
    lines.append("")

    lines.append("## Additional Notes")
    lines.append(f"- Dropped near-zero staffing categories: {dropped_staff_categories}")
    lines.append(f"- Patients with no ETP stay-window coverage: {etp_no_coverage}")
    lines.append(
        "- For "
        f"{occupancy_missing_exit_count} patients with missing discharge dates, exit was estimated as "
        f"entry + median LOS ({occupancy_median_los_days} days) for the purpose of computing bed occupancy only."
    )
    lines.append("")

    lines.append("## Runtime Log")
    lines.append("```text")
    lines.extend(logger.lines)
    lines.append("```")

    return "\n".join(lines) + "\n"


# ============================================================
# Main pipeline
# ============================================================

def main() -> int:
    logger = PipelineLogger(lines=[])

    # Validate input files
    required_files = [SPIADI_2019, SPIADI_2020, SPIADI_2021, ETP_FILE, ORG_FILE]
    missing_inputs = [str(p) for p in required_files if not p.exists()]
    if missing_inputs:
        msg = "Missing required input files:\n" + "\n".join(missing_inputs)
        print(msg)
        OUTPUT_REPORT.write_text(msg + "\n", encoding="utf-8")
        return 1

    logger.log("=== STEP 1: Load Raw SPIADI Files ===")
    spiadi_2019 = harmonize_spiadi_columns(load_spiadi_workbook(SPIADI_2019, 2019, logger), 2019, logger)
    spiadi_2020 = harmonize_spiadi_columns(load_spiadi_workbook(SPIADI_2020, 2020, logger), 2020, logger)
    spiadi_2021 = harmonize_spiadi_columns(load_spiadi_workbook(SPIADI_2021, 2021, logger), 2021, logger)

    logger.log("=== STEP 2-3: Select/Harmonize and Combine ===")
    combined = pd.concat([spiadi_2019, spiadi_2020, spiadi_2021], ignore_index=True)
    logger.log(f"Combined shape: {combined.shape[0]}x{combined.shape[1]}")
    logger.log(f"Source year counts:\n{combined['source_year'].value_counts().to_string()}")

    flow_rows: list[dict[str, Any]] = []
    for year in [2019, 2020, 2021]:
        flow_rows.append({"stage": "start", "year": year, "n": int((combined["source_year"] == year).sum())})

    logger.log("=== STEP 4: Apply Exclusions ===")
    clean = combined.copy()

    # 1) Pediatric exclusion
    ped_mask = (clean["uf"] == 6825) | (clean["age_years"] < 18)
    excluded_ped = clean[ped_mask].copy()
    clean = clean[~ped_mask].copy()
    logger.log(f"Excluded pediatric rows (uf==6825 OR age<18): {len(excluded_ped)}; remaining={len(clean)}")
    for year in [2019, 2020, 2021]:
        flow_rows.append(
            {
                "stage": "excluded_pediatric",
                "year": year,
                "n": int((excluded_ped["source_year"] == year).sum()),
            }
        )

    # 2) Erroneous entry dates
    bad_entry_mask = clean["entry"] < pd.Timestamp("2018-01-01")
    excluded_bad_dates = clean[bad_entry_mask].copy()
    logger.log(f"Excluded erroneous entry<2018-01-01 rows: {len(excluded_bad_dates)}")
    if not excluded_bad_dates.empty:
        logger.log(
            "Excluded bad entry dates: "
            + ", ".join(sorted(excluded_bad_dates["entry"].dropna().astype(str).unique().tolist()))
        )
    clean = clean[~bad_entry_mask].copy()

    for year in [2019, 2020, 2021]:
        flow_rows.append(
            {
                "stage": "excluded_bad_entry_date",
                "year": year,
                "n": int((excluded_bad_dates["source_year"] == year).sum()),
            }
        )

    # 3) Duplicate signature exclusion to enforce one row per patient proxy key.
    dup_subset = ["entry", "age_years", "sex", "igsII"]
    dup_mask = clean.duplicated(subset=dup_subset, keep="first")
    excluded_dups = clean[dup_mask].copy()
    if len(excluded_dups) > 0:
        logger.log(f"Excluded duplicate signature rows (entry+age+sex+igsII): {len(excluded_dups)}")
    clean = clean[~dup_mask].copy()

    for year in [2019, 2020, 2021]:
        flow_rows.append(
            {
                "stage": "excluded_duplicate_signature",
                "year": year,
                "n": int((excluded_dups["source_year"] == year).sum()),
            }
        )

    # Enforce Jan-Jun surveillance window
    jan_jun_mask = clean["entry"].dt.month.between(1, 6, inclusive="both")
    outside_window = clean[~jan_jun_mask].copy()
    if len(outside_window) > 0:
        logger.log(f"Excluded outside Jan-Jun window rows: {len(outside_window)}")
    clean = clean[jan_jun_mask].copy()

    for year in [2019, 2020, 2021]:
        flow_rows.append(
            {
                "stage": "excluded_outside_jan_jun",
                "year": year,
                "n": int((outside_window["source_year"] == year).sum()),
            }
        )

    for year in [2019, 2020, 2021]:
        flow_rows.append({"stage": "final_after_exclusions", "year": year, "n": int((clean["source_year"] == year).sum())})

    logger.log(f"Final rows after exclusions: {len(clean)}")
    logger.log(f"Rows per year after exclusions:\n{clean['source_year'].value_counts().sort_index().to_string()}")

    logger.log("=== STEP 5: Clean Data Quality Issues ===")

    # igsII sentinel
    igs_replacements = int((clean["igsII"] == 999).sum())
    clean.loc[clean["igsII"] == 999, "igsII"] = np.nan
    logger.log(f"Replaced igsII=999 with NaN: {igs_replacements}")

    # Origin harmonization for 2019
    before_origin = clean[clean["source_year"] == 2019]["origin"].value_counts(dropna=False)
    logger.log("Origin value counts BEFORE harmonization for 2019:")
    logger.log(before_origin.to_string())

    mask_2019 = clean["source_year"] == 2019

    def harmonize_origin(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        # numeric values pass through
        nv = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if pd.notna(nv):
            return int(nv)
        t = str(val).strip().upper()
        return ORIGIN_MAPPING_2019.get(t, np.nan)

    clean.loc[mask_2019, "origin"] = clean.loc[mask_2019, "origin"].map(harmonize_origin)
    clean["origin"] = to_numeric(clean["origin"])

    after_origin = clean[clean["source_year"] == 2019]["origin"].value_counts(dropna=False)
    logger.log("Origin value counts AFTER harmonization for 2019:")
    logger.log(after_origin.to_string())

    unmapped_origin = int(clean.loc[mask_2019, "origin"].isna().sum())
    if unmapped_origin > 0:
        logger.log(f"WARNING: unmapped 2019 origin values after harmonization: {unmapped_origin}")

    # Cancer in 2019 should be 9
    cancer_2019_ok = bool((clean.loc[mask_2019, "cancer"] == 9).all())
    logger.log(f"Cancer in 2019 all set to 9: {cancer_2019_ok}")

    # Keep code 9 values as unknown categories (as requested).
    # Verify categorical ranges
    unexpected_code_rows = []
    for col, expected_set in CATEGORICAL_EXPECTED.items():
        vals = set(to_numeric(clean[col]).dropna().astype(int).unique().tolist())
        unexpected = sorted(vals - expected_set)
        logger.log(f"Categorical counts for {col}:\n{clean[col].value_counts(dropna=False).to_string()}")
        if unexpected:
            unexpected_code_rows.append({"column": col, "unexpected_values": unexpected})
            logger.log(f"WARNING: unexpected values in {col}: {unexpected}")

    logger.log("=== STEP 6: Compute Derived Features ===")
    clean["los_days"] = (clean["exit"] - clean["entry"]).dt.days
    clean["has_infection"] = ((clean["bact_count"].fillna(0) > 0) | (clean["pneu_count"].fillna(0) > 0)).astype(int)

    clean["admission_year"] = clean["entry"].dt.year
    clean["admission_month"] = clean["entry"].dt.month
    clean["admission_day_of_week"] = clean["entry"].dt.dayofweek
    clean["admitted_on_weekend"] = (clean["admission_day_of_week"] >= 5).astype(int)

    valid_los = clean["los_days"].dropna()
    logger.log(f"LOS valid count={len(valid_los)}, LOS NaN count={int(clean['los_days'].isna().sum())}")
    if not valid_los.empty:
        logger.log(
            f"LOS distribution -> min={valid_los.min()}, max={valid_los.max()}, mean={valid_los.mean():.3f}, median={valid_los.median():.3f}"
        )

    los_lt0 = int((clean["los_days"] < 0).sum())
    los_gt120 = int((clean["los_days"] > 120).sum())
    if los_lt0 or los_gt120:
        logger.log(f"WARNING: LOS anomalies -> <0: {los_lt0}, >120: {los_gt120}")

    logger.log(f"has_infection counts:\n{clean['has_infection'].value_counts().to_string()}")
    infection_rate_by_year = clean.groupby("source_year")["has_infection"].mean() * 100
    logger.log(f"Infection rate by year (%):\n{infection_rate_by_year.to_string()}")

    logger.log("=== STEP 7: Process Daily ETP Staffing Data ===")
    daily_etp, etp_meta = extract_daily_etp_table(ETP_FILE, logger)

    logger.log("=== STEP 8: Aggregate ETP Over Each Patient's Stay ===")
    etp_agg, dropped_staff_categories = aggregate_etp_over_stays(clean, daily_etp, logger)
    etp_no_coverage = int(etp_agg.isna().all(axis=1).sum())

    logger.log("=== STEP 9: Process Organizational Data and Environment Features ===")
    clean, occupancy_missing_exit_count, occupancy_median_los_days = compute_environment_features(clean, logger)
    org_monthly = load_org_monthly_metrics(ORG_FILE, logger)

    clean = clean.merge(
        org_monthly,
        how="left",
        left_on=["admission_year", "admission_month"],
        right_on=["period_year", "period_month"],
    )
    if "period_year" in clean.columns:
        clean = clean.drop(columns=["period_year", "period_month"])

    logger.log("=== STEP 10: Assemble Final Dataset ===")
    model_df = pd.concat([clean.reset_index(drop=True), etp_agg.reset_index(drop=True)], axis=1)
    logger.log(f"Assembled dataset shape before final selection: {model_df.shape[0]}x{model_df.shape[1]}")

    logger.log("=== STEP 11: Final Feature Selection and Renaming ===")
    drop_cols = ["entry", "exit", "uf", "bact_count", "pneu_count", "source_year", "birthdate"]
    final_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns]).copy()

    rename_map = {
        "age_years": "age",
        "origin": "admission_origin",
        "diagnosis": "diagnostic_category",
        "trauma": "trauma_status",
        "immuno": "immunosuppression",
        "antibio": "antibiotic_at_admission",
        "cancer": "cancer_status",
        "igsII": "severity_score_igs2",
        "intubated": "intubation_status",
        "reintubated": "reintubation_status",
        "intubation_duration": "intubation_days",
        "urinary": "urinary_catheter",
        "urinary_duration": "urinary_catheter_days",
        "kt_count": "central_line_count",
        "ecmo": "ecmo_status",
        "dead": "icu_mortality",
        "los_days": "length_of_stay",
        "admission_day_of_week": "admission_weekday",
        "admitted_on_weekend": "weekend_admission",
        "bed_occupancy_at_admission": "bed_occupancy",
        "patient_turnover_at_admission": "patient_turnover",
        "unit_gross_los": "unit_avg_los",
        "los_ratio_vs_national": "los_ratio_national",
        "mean_etp_general_nurse": "nurse_staffing_etp",
        "mean_count_general_nurse": "nurse_staffing_count",
        "mean_etp_nursing_assistant": "nurse_aide_staffing_etp",
        "mean_count_nursing_assistant": "nurse_aide_staffing_count",
        "mean_etp_all_staff": "total_staffing_etp",
        "mean_count_all_staff": "total_staffing_count",
    }

    # Rename extra staffing columns with consistent readable pattern.
    for c in list(final_df.columns):
        if c in rename_map:
            continue
        if c.startswith("mean_etp_"):
            cat = c.replace("mean_etp_", "")
            rename_map[c] = f"{cat}_staffing_etp"
        elif c.startswith("mean_count_"):
            cat = c.replace("mean_count_", "")
            rename_map[c] = f"{cat}_staffing_count"

    final_df = final_df.rename(columns=rename_map)

    # Coerce key categorical columns to integer-like where possible
    categorical_cols = [
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
        "icu_mortality",
        "weekend_admission",
        "has_infection",
    ]
    for c in categorical_cols:
        if c in final_df.columns:
            final_df[c] = to_numeric(final_df[c]).round().astype("Int64")

    # Keep stable sort by admission date semantics
    final_df = final_df.sort_values(["admission_year", "admission_month"], kind="stable").reset_index(drop=True)

    logger.log(f"Final dataset shape: {final_df.shape[0]}x{final_df.shape[1]}")

    # ========================================================
    # Validation checks
    # ========================================================
    logger.log("=== STEP 12: Validation Checks ===")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if status else "FAIL", "details": details})

    # Checklist checks
    row_ok = 405 <= len(final_df) <= 409
    add_check("Row count approx 407 (±2)", row_ok, f"rows={len(final_df)}")

    dup_cols_pre = [c for c in ["entry", "age_years", "sex", "igsII"] if c in clean.columns]
    dup_count = int(clean.duplicated(subset=dup_cols_pre).sum()) if dup_cols_pre else -1
    add_check("No duplicates on entry+age+sex+igsII", dup_count == 0, f"duplicate_count={dup_count}")

    inf_counts = final_df["has_infection"].value_counts(dropna=False)
    no_count = int(inf_counts.get(0, 0))
    yes_count = int(inf_counts.get(1, 0))
    ratio_ok = 0.18 <= (yes_count / len(final_df) if len(final_df) else 0) <= 0.30
    add_check(
        "has_infection approx 77/23 split",
        ratio_ok,
        f"counts_no={no_count}, counts_yes={yes_count}, rate_yes={yes_count / len(final_df):.3f}",
    )

    add_check(
        "No bact_count or pneu_count columns",
        ("bact_count" not in final_df.columns and "pneu_count" not in final_df.columns),
        f"columns_present={[c for c in ['bact_count', 'pneu_count'] if c in final_df.columns]}",
    )

    add_check("No igsII=999 remaining", int((clean["igsII"] == 999).sum()) == 0, f"remaining_999={int((clean['igsII'] == 999).sum())}")

    origin_vals = set(to_numeric(clean["origin"]).dropna().astype(int).unique().tolist())
    add_check("All origin values in 1..9", origin_vals.issubset(set(range(1, 10))), f"origin_values={sorted(origin_vals)}")

    years_ok = set(final_df["admission_year"].dropna().astype(int).unique().tolist()).issubset({2019, 2020, 2021})
    add_check("admission_year only 2019/2020/2021", years_ok, f"years={sorted(final_df['admission_year'].dropna().unique().tolist())}")

    months_ok = set(final_df["admission_month"].dropna().astype(int).unique().tolist()).issubset({1, 2, 3, 4, 5, 6})
    add_check("admission_month only 1..6", months_ok, f"months={sorted(final_df['admission_month'].dropna().unique().tolist())}")

    los_nan_consistency = bool((clean["los_days"].isna() == clean["exit"].isna()).all())
    add_check(
        "LOS NaN only when exit missing",
        los_nan_consistency,
        f"los_nan={int(clean['los_days'].isna().sum())}, exit_nan={int(clean['exit'].isna().sum())}",
    )

    etp_cols_final = [c for c in final_df.columns if "staffing_" in c or c.startswith("mean_etp_") or c.startswith("mean_count_")]
    etp_nan = int(final_df[etp_cols_final].isna().sum().sum()) if etp_cols_final else 0
    etp_check = etp_nan == 0 or etp_no_coverage > 0
    add_check(
        "ETP columns no NaN (or documented no-coverage)",
        etp_check,
        f"total_etp_nan_cells={etp_nan}, patients_no_coverage={etp_no_coverage}",
    )

    # Naming convention check: ensure expected key columns exist
    expected_key_cols = {
        "age",
        "sex",
        "admission_origin",
        "diagnostic_category",
        "immunosuppression",
        "severity_score_igs2",
        "length_of_stay",
        "has_infection",
        "admission_year",
        "admission_month",
    }
    naming_ok = expected_key_cols.issubset(set(final_df.columns))
    add_check("Column names match paper naming convention (core set)", naming_ok, f"missing_core={sorted(expected_key_cols - set(final_df.columns))}")

    # Additional required checks from PRD section 12
    entry_range_ok = bool(
        clean["entry"].dropna().between(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-06-30")).all()
    )
    add_check("All entry dates in 2019-01-01..2021-06-30", entry_range_ok, "checked on cleaned rows")

    target_formula_ok = bool(
        (
            clean["has_infection"]
            == ((clean["bact_count"].fillna(0) > 0) | (clean["pneu_count"].fillna(0) > 0)).astype(int)
        ).all()
    )
    add_check("Target matches bact_count/pneu_count formula", target_formula_ok, "exact row-level comparison")

    los_positive_ok = bool((clean.loc[clean["exit"].notna(), "los_days"] > 0).all())
    add_check("LOS > 0 for all rows with valid exit", los_positive_ok, "checked on cleaned rows")

    leakage_ok = bool("bact_count" not in final_df.columns and "pneu_count" not in final_df.columns)
    add_check("No data leakage columns (bact_count/pneu_count) in final", leakage_ok, "post-selection final columns")

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------
    feature_descriptions = {
        "age": "Patient age (years)",
        "sex": "Patient sex (1=M, 2=F, 3=other)",
        "admission_origin": "Patient origin before ICU (1-9)",
        "diagnostic_category": "Medical/surgical classification",
        "trauma_status": "Trauma patient status",
        "immunosuppression": "Immunosuppression status",
        "antibiotic_at_admission": "Antibiotic use at admission",
        "cancer_status": "Cancer status",
        "severity_score_igs2": "IGS II severity score",
        "intubation_status": "Intubation during stay",
        "reintubation_status": "Re-intubation status",
        "intubation_days": "Duration of intubation (days)",
        "urinary_catheter": "Urinary catheter usage",
        "urinary_catheter_days": "Duration of urinary catheterization (days)",
        "central_line_count": "Number of intravascular catheters",
        "ecmo_status": "ECMO usage",
        "icu_mortality": "Death in ICU",
        "length_of_stay": "ICU length of stay (days)",
        "admission_year": "Year of admission",
        "admission_month": "Month of admission",
        "admission_weekday": "Admission day of week (0=Mon..6=Sun)",
        "weekend_admission": "Weekend admission flag",
        "bed_occupancy": "Number of other patients present in ICU at admission",
        "patient_turnover": "Admissions + discharges on admission day",
        "unit_avg_los": "Unit monthly gross LOS mean",
        "national_avg_los": "National average LOS",
        "los_ratio_national": "Unit LOS to national LOS ratio",
        "nurse_staffing_etp": "Mean daily general nurse ETP during stay",
        "nurse_staffing_count": "Mean daily general nurse headcount during stay",
        "nurse_aide_staffing_etp": "Mean daily nursing assistant ETP during stay",
        "nurse_aide_staffing_count": "Mean daily nursing assistant headcount during stay",
        "total_staffing_etp": "Mean daily total staff ETP during stay",
        "total_staffing_count": "Mean daily total staff headcount during stay",
        "has_infection": "Target: bacteremia or pneumonia during ICU stay",
    }

    feature_dict = generate_feature_dictionary(final_df, feature_descriptions)

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    flow_df = pd.DataFrame(flow_rows)
    report_content = build_validation_report(
        logger=logger,
        flow_df=flow_df,
        final_df=final_df,
        pre_drop_df=model_df,
        dropped_staff_categories=dropped_staff_categories,
        validation_checks=checks,
        feature_dict=feature_dict,
        etp_no_coverage=etp_no_coverage,
        occupancy_missing_exit_count=occupancy_missing_exit_count,
        occupancy_median_los_days=occupancy_median_los_days,
    )

    OUTPUT_REPORT.write_text(report_content, encoding="utf-8")
    logger.log(f"Validation report written: {OUTPUT_REPORT}")

    # Hard stop if any validation failed
    failed_checks = [c for c in checks if c["status"] == "FAIL"]
    if failed_checks:
        logger.log("Validation failed. Output dataset/feature dictionary were NOT saved.")
        logger.log("Failed checks:")
        for chk in failed_checks:
            logger.log(f"- {chk['check']}: {chk['details']}")
        return 2

    # Save outputs only when checks pass
    final_df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    feature_dict.to_csv(OUTPUT_FEATURE_DICT, index=False, encoding="utf-8")

    logger.log(f"Final dataset saved: {OUTPUT_DATASET}")
    logger.log(f"Feature dictionary saved: {OUTPUT_FEATURE_DICT}")
    logger.log("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
