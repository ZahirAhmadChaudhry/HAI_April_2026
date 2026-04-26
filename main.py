from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_repeated_experiments as core


BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = BASE_DIR / "full_experiment_runs"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_algorithms(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one algorithm must be provided.")
    invalid = [p for p in parts if p not in {"xgboost", "lightgbm", "catboost", "random_forest"}]
    if invalid:
        raise ValueError(f"Unsupported algorithms: {invalid}")
    return parts


def configure_core_outputs(run_dir: Path, algorithms: list[str], seeds: list[int], n_trials: int) -> dict[str, Path]:
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    core.RESULTS_DIR = results_dir
    core.FIG_DIR = figures_dir

    core.OUTPUT_RESULTS = results_dir / "repeated_experiment_results.csv"
    core.OUTPUT_CHECKPOINT = results_dir / "repeated_experiment_checkpoint.csv"
    core.OUTPUT_SUMMARY = results_dir / "repeated_experiment_summary.md"
    core.OUTPUT_SHAP_STABILITY = results_dir / "repeated_shap_stability.csv"

    core.PHASE_ANALYSIS_REPORT = results_dir / "phase_1_2_3_analysis_report.md"
    core.PHASE1_SPEARMAN_MATRIX = results_dir / "phase1_clinical_org_spearman_matrix.csv"
    core.PHASE1_TOP20_CORR = results_dir / "phase1_top20_clinical_org_correlations.csv"
    core.PHASE1_RANK_FULL = results_dir / "phase1_rank_stability_full.csv"
    core.PHASE1_RANK_TOP15 = results_dir / "phase1_rank_stability_top15_comparison.csv"
    core.PHASE1_DELTA_SHAP = results_dir / "phase1_delta_shap_ttests.csv"
    core.PHASE2_SEED_BEST = results_dir / "phase2_seed_best_metrics.csv"
    core.PHASE2_WILCOXON = results_dir / "phase2_wilcoxon_comparisons.csv"
    core.PHASE3_CONFIDENCE = results_dir / "phase3_confidence_org_shap_analysis.csv"
    core.PHASE3_THRESHOLD_CURVE = results_dir / "phase3_threshold_cost_curve.csv"
    core.PHASE3_OPT_THRESHOLDS = results_dir / "phase3_optimal_thresholds.csv"

    core.FIG_AUC_BOXPLOT = figures_dir / "auc_distribution_boxplot.png"
    core.FIG_AUC_DIFF_HIST = figures_dir / "auc_difference_histogram.png"
    core.FIG_SHAP_STABILITY = figures_dir / "shap_stability_plot.png"
    core.FIG_PHASE1_SPEARMAN = figures_dir / "phase1_clinical_org_spearman_heatmap.png"
    core.FIG_PHASE3_PR_CURVE = figures_dir / "phase3_model_b_precision_recall_curve.png"
    core.FIG_PHASE3_CONFUSIONS = figures_dir / "phase3_cost_adjusted_confusions.png"

    core.ALGORITHMS = algorithms
    core.N_TRIALS = n_trials
    core.START_SEED = min(seeds)
    core.END_SEED = max(seeds)

    return {
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }


def build_jobs(feature_configs: list[dict[str, Any]], seeds: list[int], algorithms: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    run_id_by_seed = {seed: idx + 1 for idx, seed in enumerate(seeds)}

    for seed in seeds:
        run_id = run_id_by_seed[seed]
        for cfg in feature_configs:
            for algorithm in algorithms:
                jobs.append(
                    {
                        "run_id": run_id,
                        "seed": int(seed),
                        "config_id": str(cfg["config_id"]),
                        "feature_set": str(cfg["feature_set"]),
                        "config_name": str(cfg["config_name"]),
                        "algorithm": algorithm,
                        "features": list(cfg["features"]),
                    }
                )
    return jobs


def job_key(job: dict[str, Any]) -> str:
    return f"{job['seed']}|{job['config_id']}|{job['algorithm']}"


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    if not checkpoint_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(checkpoint_path)
        return df
    except Exception:
        return pd.DataFrame()


def latest_rows_by_job(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "job_key" not in df.columns:
        return pd.DataFrame()

    work = df.copy().reset_index(drop=True)
    work["_order"] = np.arange(len(work))
    latest = work.sort_values("_order").groupby("job_key", as_index=False).tail(1)
    return latest.drop(columns=["_order"], errors="ignore").reset_index(drop=True)


def append_checkpoint_row(checkpoint_path: Path, row: dict[str, Any]) -> None:
    out = pd.DataFrame([row])
    if checkpoint_path.exists():
        out.to_csv(checkpoint_path, mode="a", header=False, index=False)
    else:
        out.to_csv(checkpoint_path, mode="w", header=True, index=False)


def append_event_log(event_log_path: Path, payload: dict[str, Any]) -> None:
    with event_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_single_job(
    job: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    years_train: pd.Series,
    n_trials: int,
) -> dict[str, Any]:
    started_at = now_utc()
    t0 = time.perf_counter()

    row: dict[str, Any] = {
        "job_key": job_key(job),
        "run_id": int(job["run_id"]),
        "seed": int(job["seed"]),
        "config_id": str(job["config_id"]),
        "feature_set": str(job["feature_set"]),
        "config_name": str(job["config_name"]),
        "algorithm": str(job["algorithm"]),
        "n_features": int(len(job["features"])),
        "n_trials": int(n_trials),
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "traceback": "",
        "started_at": started_at,
        "ended_at": "",
        "duration_seconds": float("nan"),
        "cv_auc_roc": float("nan"),
        "best_params_json": "{}",
    }

    try:
        feature_cols = list(job["features"])
        X_train_raw = train_df[feature_cols].copy()
        X_test_raw = test_df[feature_cols].copy()

        best_params, best_cv_auc = core.tune_algorithm(
            algorithm=str(job["algorithm"]),
            feature_set_name=str(job["config_name"]),
            X_train_df=X_train_raw,
            y_train=y_train,
            years_train=years_train,
            feature_cols=feature_cols,
            seed=int(job["seed"]),
            n_trials=n_trials,
        )

        trained = core.train_final_model(
            algorithm=str(job["algorithm"]),
            params=best_params,
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_test_raw=X_test_raw,
            feature_cols=feature_cols,
            seed=int(job["seed"]),
        )

        metrics = core.compute_metrics(y_test.to_numpy(), trained["y_test_prob"])

        row["cv_auc_roc"] = float(best_cv_auc)
        row["best_params_json"] = json.dumps(best_params, sort_keys=True)
        row.update(metrics)

    except Exception as exc:
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        row["traceback"] = traceback.format_exc()

    row["ended_at"] = now_utc()
    row["duration_seconds"] = float(time.perf_counter() - t0)
    return row


def build_run_diagnostics(
    run_dir: Path,
    config: dict[str, Any],
    latest: pd.DataFrame,
    expected_jobs: int,
) -> str:
    ok = latest[latest["status"] == "ok"].copy() if not latest.empty else pd.DataFrame()
    errors = latest[latest["status"] != "ok"].copy() if not latest.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Run Diagnostics")
    lines.append("")
    lines.append(f"- run_dir: {run_dir.as_posix()}")
    lines.append(f"- started_at: {config['started_at']}")
    lines.append(f"- finished_at: {config['finished_at']}")
    lines.append(f"- expected_jobs: {expected_jobs}")
    lines.append(f"- latest_checkpoint_rows: {len(latest)}")
    lines.append(f"- completed_ok_jobs: {len(ok)}")
    lines.append(f"- latest_error_jobs: {len(errors)}")
    lines.append("")

    if not latest.empty:
        by_config = (
            latest.groupby(["feature_set", "algorithm", "status"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["feature_set", "algorithm", "status"]) 
            .reset_index(drop=True)
        )
        lines.append("## Job Status By Configuration")
        lines.append(core.to_md_table(by_config))
        lines.append("")

    if not errors.empty:
        err_cols = [
            "job_key",
            "feature_set",
            "algorithm",
            "error_type",
            "error_message",
            "duration_seconds",
        ]
        err_view = errors[err_cols].copy().head(50)
        lines.append("## Latest Errors")
        lines.append(core.to_md_table(err_view))
        lines.append("")

    return "\n".join(lines) + "\n"


def run_post_analyses(
    results_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_map: dict[str, list[str]],
    phase1_pairs: pd.DataFrame,
    seeds: list[int],
) -> None:
    per_model_summary, paired, algo_wins = core.summarize_results(results_df)

    winner_algo_b = (
        paired["best_b_algorithm"].value_counts().sort_values(ascending=False).index[0]
        if not paired.empty
        else "catboost"
    )
    print(f"Winner algorithm for Model B (most frequent best): {winner_algo_b}")

    _, feat_stability, grp_stability = core.run_shap_stability(
        winner_algorithm_b=winner_algo_b,
        seeds=seeds,
        results_df=results_df,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        model_b_features=feature_map["B"],
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
    repeated_shap_stability.to_csv(core.OUTPUT_SHAP_STABILITY, index=False)

    _, phase1_top15, phase1_delta_shap, outputs_b = core.run_phase1_rank_stability_and_delta_shap(
        results_df=results_df,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        feature_map=feature_map,
    )

    phase2_seed_best, phase2_wilcoxon = core.run_phase2_ablation_wilcoxon(results_df)

    phase3_confidence, _, phase3_opt_thresholds = core.run_phase3_confidence_and_threshold_analyses(
        outputs_b=outputs_b,
        y_test=y_test.to_numpy(),
    )

    core.make_plots(results_df, paired, feat_stability)

    summary_text = core.build_summary_markdown(
        results_df=results_df,
        per_model_summary=per_model_summary,
        paired=paired,
        algo_wins=algo_wins,
        feat_stability=feat_stability,
        grp_stability=grp_stability,
    )
    core.OUTPUT_SUMMARY.write_text(summary_text, encoding="utf-8")

    phase_report_text = core.build_phase_analysis_report(
        correlation_pairs=phase1_pairs,
        rank_top15=phase1_top15,
        delta_shap=phase1_delta_shap,
        phase2_seed_best=phase2_seed_best,
        phase2_wilcoxon=phase2_wilcoxon,
        phase3_confidence=phase3_confidence,
        phase3_thresholds=phase3_opt_thresholds,
    )
    core.PHASE_ANALYSIS_REPORT.write_text(phase_report_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preconfigured full HAI experiments runner with per-experiment checkpoints, "
            "resume support, and detailed diagnostics/statistical reports."
        )
    )
    parser.add_argument("--run-name", default="full_experiments", help="Folder name under full_experiment_runs/.")
    parser.add_argument("--seed-start", type=int, default=43)
    parser.add_argument("--seed-end", type=int, default=62)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--algorithms",
        default="xgboost,lightgbm,catboost,random_forest",
        help="Comma-separated algorithms.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when an experiment fails.")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run for validation (1 seed, RF only, 2 trials).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    algorithms = parse_algorithms(args.algorithms)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    n_trials = int(args.n_trials)

    if args.smoke:
        algorithms = ["random_forest"]
        seeds = [args.seed_start]
        n_trials = 2

    run_dir = RUNS_ROOT / args.run_name
    paths = configure_core_outputs(run_dir=run_dir, algorithms=algorithms, seeds=seeds, n_trials=n_trials)

    core.configure_plots()
    core.ensure_dirs()
    core.print_compute_config()

    if not core.INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {core.INPUT_DATASET}")

    model_a_features, model_b_features = core.load_cleaned_feature_sets()
    feature_configs = core.build_feature_configs(model_a_features, model_b_features)
    feature_map = core.get_feature_map(feature_configs)

    config_payload = {
        "started_at": now_utc(),
        "run_name": args.run_name,
        "run_dir": run_dir.as_posix(),
        "input_dataset": core.INPUT_DATASET.as_posix(),
        "cleaned_features_json": core.CLEANED_FEATURES_JSON.as_posix(),
        "algorithms": algorithms,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "effective_seeds": seeds,
        "n_trials": n_trials,
        "smoke": bool(args.smoke),
        "stop_on_error": bool(args.stop_on_error),
        "gpu_env_hint": "Set HAI_USE_GPU=1 on Linux remote for forced GPU path.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    df = pd.read_csv(core.INPUT_DATASET)
    train_df = df[df["admission_year"].isin([2019, 2020])].copy().reset_index(drop=True)
    test_df = df[df["admission_year"] == 2021].copy().reset_index(drop=True)

    y_train = train_df["has_infection"].astype(int)
    y_test = test_df["has_infection"].astype(int)
    years_train = train_df["admission_year"].astype(int)

    print(f"Train rows={len(train_df)}, Test rows={len(test_df)}")
    print(f"Train infection rate={y_train.mean() * 100:.2f}%, Test infection rate={y_test.mean() * 100:.2f}%")

    clinical_group_names = [
        "Patient Demographics",
        "Clinical Severity",
        "Medical Procedures",
        "Length of Stay",
        "Temporal",
    ]
    org_group_names = ["Organizational Environment", "Organizational Staffing"]
    clinical_features = core.ordered_unique(
        [
            feat
            for group_name in clinical_group_names
            for feat in sorted(core.FEATURE_GROUPS[group_name])
            if feat in df.columns
        ]
    )
    org_features = core.ordered_unique(
        [
            feat
            for group_name in org_group_names
            for feat in sorted(core.FEATURE_GROUPS[group_name])
            if feat in df.columns
        ]
    )
    _, phase1_pairs = core.run_phase1_clinical_org_correlation(df, clinical_features, org_features)

    jobs = build_jobs(feature_configs=feature_configs, seeds=seeds, algorithms=algorithms)
    expected_jobs = len(jobs)
    event_log_path = paths["logs_dir"] / "experiment_events.jsonl"

    checkpoint_df = load_checkpoint(core.OUTPUT_CHECKPOINT)
    latest_before = latest_rows_by_job(checkpoint_df)
    completed_ok = set()
    if not latest_before.empty:
        completed_ok = set(latest_before[latest_before["status"] == "ok"]["job_key"].tolist())

    print(f"Total jobs expected: {expected_jobs}")
    print(f"Resuming from checkpoint with completed jobs: {len(completed_ok)}")

    for idx, job in enumerate(jobs, start=1):
        key = job_key(job)
        if key in completed_ok:
            continue

        print(
            f"[{idx}/{expected_jobs}] seed={job['seed']} config={job['config_id']} "
            f"({job['feature_set']}) algo={job['algorithm']}"
        )

        row = run_single_job(
            job=job,
            train_df=train_df,
            test_df=test_df,
            y_train=y_train,
            y_test=y_test,
            years_train=years_train,
            n_trials=n_trials,
        )
        append_checkpoint_row(core.OUTPUT_CHECKPOINT, row)

        append_event_log(
            event_log_path,
            {
                "ts": now_utc(),
                "job_key": row["job_key"],
                "status": row["status"],
                "seed": row["seed"],
                "config_id": row["config_id"],
                "feature_set": row["feature_set"],
                "algorithm": row["algorithm"],
                "duration_seconds": row["duration_seconds"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            },
        )

        if row["status"] == "ok":
            completed_ok.add(key)
            print(
                f"  -> OK AUC={float(row['auc_roc']):.4f} PR={float(row['auc_pr']):.4f} "
                f"F1={float(row['f1']):.4f} in {float(row['duration_seconds']):.1f}s"
            )
        else:
            print(f"  -> ERROR {row['error_type']}: {row['error_message']}")
            if args.stop_on_error:
                print("Stopping due to --stop-on-error")
                break

    checkpoint_df = load_checkpoint(core.OUTPUT_CHECKPOINT)
    latest = latest_rows_by_job(checkpoint_df)
    latest = latest.sort_values(["run_id", "config_id", "algorithm"]).reset_index(drop=True)

    ok_df = latest[latest["status"] == "ok"].copy()
    ok_df = ok_df.sort_values(["run_id", "config_id", "algorithm"]).reset_index(drop=True)
    ok_df.to_csv(core.OUTPUT_RESULTS, index=False)

    errors_df = latest[latest["status"] != "ok"].copy()
    errors_df.to_csv(paths["results_dir"] / "latest_errors.csv", index=False)

    config_payload["finished_at"] = now_utc()
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    diagnostics_md = build_run_diagnostics(
        run_dir=run_dir,
        config=config_payload,
        latest=latest,
        expected_jobs=expected_jobs,
    )
    (paths["results_dir"] / "run_diagnostics.md").write_text(diagnostics_md, encoding="utf-8")

    if ok_df.empty:
        print("No successful jobs found. See run_diagnostics.md and latest_errors.csv.")
        return 2

    run_post_analyses(
        results_df=ok_df,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        feature_map=feature_map,
        phase1_pairs=phase1_pairs,
        seeds=seeds,
    )

    print("=== Full Experiment Runner Completed ===")
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {core.OUTPUT_CHECKPOINT}")
    print(f"Results: {core.OUTPUT_RESULTS}")
    print(f"Summary: {core.OUTPUT_SUMMARY}")
    print(f"Phase report: {core.PHASE_ANALYSIS_REPORT}")
    print(f"Diagnostics: {paths['results_dir'] / 'run_diagnostics.md'}")

    if len(ok_df) < expected_jobs:
        print(
            f"WARNING: Completed OK jobs {len(ok_df)}/{expected_jobs}. "
            "Resume by rerunning the same command."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
