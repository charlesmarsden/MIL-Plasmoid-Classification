#!/usr/bin/env python3
"""
10-seed comparison covering five areas:

    - seed variance (multi-seed, CV + held-out test)
    - augmentation
    - architectures: gated attention vs TimeMIL vs CRNN
    - choice of channels: JY vs BR-BZ vs BR-BZ-BY-JY with one fixed config
    - label policy: exclude_1 vs include_1

Each arm trains one architecture/config repeatedly per seed with 3-fold CV. Mean and std is reported across seeds.
These enable pairwise (and one unpaired) comparisons and signifcance tests across seeds.

Reuses (imports, does not copy) the existing search engines:
    - BR-BZ/hyperparameter-search-codes/br_bz_gated_attention_hyperparameter_search_261us_80frames_1us.py
      ("multi" engine: any channel combination, no offline augmentation)
    - jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented.py
      ("aug" engine: JY with offline augmented manifest handling)
    - timemil_hyperparameter_search.py  (ImageTimeMIL pooling class only)
    - ../CRNN/hyperparameter-search-codes/jy_crnn_hyperparameter_search_261us_80frames_1us_augmented.py
      (ImageCRNN pooling class only)

The held-out test split is always done with random_state=42.

Paths are controlled by the same env vars as the BR-BZ search (BR_BZ_FIELD_IMAGES_DIR, BR_BZ_OUTPUT_DIR; 
note that the 'BR-BZ search' was a preliminary search done at the start of this project).
Results:
  <output>/suite_results.csv   one row per (experiment, seed)
  <output>/suite_summary.txt   per-trial mean+/-std and paired significance tests
  <output>/runs/<experiment>/seed_<seed>/   full per-run records (fold histories, 
                                            final_test_metrics.json, final_test_predictions.csv, 
                                            checkpoint)

A run whose run_summary.json already exists is skipped.

Examples (these are for Stellar at Princeton):
  python paper_comparison_suite.py
  python paper_comparison_suite.py --experiments aug-on-JY,aug-off-JY
  python paper_comparison_suite.py --smoke # verification before a full run
  python paper_comparison_suite.py --summarize-only
"""

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

#============================================================================
# Engine imports (relative to this file so the same tree works on Flux/Stellar)

SCRIPT_DIR = Path(__file__).resolve().parent # Models/TimeMIL/hyperparameter-search-codes
TIMEMIL_DIR = SCRIPT_DIR.parent # Models/TimeMIL
MODELS_DIR = TIMEMIL_DIR.parent # Models

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TIMEMIL_DIR / "BR-BZ" / "hyperparameter-search-codes"))
sys.path.insert(0, str(MODELS_DIR / "CRNN" / "hyperparameter-search-codes"))
# this is only for timemil so can be ignored
sys.path.insert(0, str(TIMEMIL_DIR / "TimeMIL-github")) # needed to use models.nystrom_attention

import br_bz_gated_attention_hyperparameter_search_261us_80frames_1us as engine_multi
import jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented as engine_aug
import jy_crnn_hyperparameter_search_261us_80frames_1us_augmented as engine_crnn_module

try:
    import timemil_hyperparameter_search as engine_timemil_module
    TIMEMIL_AVAILABLE = True
except Exception as import_error: # nystrom_attention or its deps may be missing
    engine_timemil_module = None
    TIMEMIL_AVAILABLE = False
    TIMEMIL_IMPORT_ERROR = f"{type(import_error).__name__}: {import_error}"

FIXED_SPLIT_SEED = 42 # DON"T CHANGE
DEFAULT_SEEDS = "101,102,103,104,105"
METRIC_NAMES = ["f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall", "threshold", "loss"]


#============================================================================
# Extends the multi-channel engine's build_model so its unchanged train/test 
# machinery can also train TimeMIL and CRNN poolings.

_original_multi_build_model = engine_multi.build_model


def extended_build_model(config, field_shape):
    architecture = config.get("architecture")
    if architecture == "gated_attention":
        return _original_multi_build_model(config, field_shape)

    _, num_channels, image_height, image_width = field_shape
    frame_encoder = engine_multi.FrameEncoder(
        image_height=image_height,
        image_width=image_width,
        channels=tuple(int(value) for value in config["cnn_channels"]),
        embed_dim=int(config["embed_dim"]),
        backbone=config["cnn_backbone"],
        in_channels=num_channels,
    )

    if architecture == "crnn":
        return engine_crnn_module.ImageCRNN(
            frame_encoder=frame_encoder,
            embed_dim=int(config["embed_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            rnn_layers=int(config["rnn_layers"]),
            bidirectional=bool(config["bidirectional"]),
            dropout=float(config["dropout"]),
            classifier_multiplier=float(config["classifier_multiplier"]),
        )

    if architecture == "timemil":
        if not TIMEMIL_AVAILABLE:
            raise RuntimeError(f"TimeMIL is unavailable in this environment ({TIMEMIL_IMPORT_ERROR})")
        return engine_timemil_module.ImageTimeMIL(
            frame_encoder=frame_encoder,
            embed_dim=int(config["embed_dim"]),
            dropout=float(config["dropout"]),
            layers=int(config["timemil_layers"]),
            heads=int(config["heads"]),
            num_landmarks=int(config["num_landmarks"]),
            classifier_multiplier=float(config["classifier_multiplier"]),
        )

    raise ValueError(f"Unknown architecture: {architecture}")


engine_multi.build_model = extended_build_model


#============================================================================
# Experiments

def fixed_base_config(**overrides):
    """Historical JY best-F1 anchor; identical across every trial unless overridden."""
    config = {
        "architecture": "gated_attention",
        "cnn_backbone": "plain",
        "cnn_channels": [16, 32, 64],
        "embed_dim": 128,
        "dropout": 0.3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "loss_type": "bce",
        "scheduler": "cosine",
        "classifier_multiplier": 1.0,
    }
    config.update(overrides)
    return config


def experiment_specs():
    """engine: 'multi' (FieldDataStore, any channels) or 'aug' (JYDataStore, offline aug)"""
    return {
        # 1 + 4: multi-seed baseline
        "baseline-JY": dict(
            engine="multi", channels=["JY"], label_policy="exclude_1",
            config=fixed_base_config(name="baseline-JY"),
        ),
        "channels-BR-BZ": dict(
            engine="multi", channels=["BR", "BZ"], label_policy="exclude_1",
            config=fixed_base_config(name="channels-BR-BZ"),
        ),
        "channels-BR-BZ-BY-JY": dict(
            engine="multi", channels=["BR", "BZ", "BY", "JY"], label_policy="exclude_1",
            config=fixed_base_config(name="channels-BR-BZ-BY-JY"),
        ),
        # 3b: alternative architectures on the best (4-channel) input, so the
        #     architecture conclusion is not only applicable to JY but acorss all spatial profiles
        "arch-timemil-BR-BZ-BY-JY": dict(
            engine="multi", channels=["BR", "BZ", "BY", "JY"], label_policy="exclude_1",
            config=fixed_base_config(name="arch-timemil-BR-BZ-BY-JY", architecture="timemil",
                                     timemil_layers=2, heads=8, num_landmarks=64),
        ),
        "arch-crnn-BR-BZ-BY-JY": dict(
            engine="multi", channels=["BR", "BZ", "BY", "JY"], label_policy="exclude_1",
            config=fixed_base_config(name="arch-crnn-BR-BZ-BY-JY", architecture="crnn",
                                     hidden_dim=256, rnn_layers=1, bidirectional=False),
        ),
        # 3: architecture comparison (JY, same encoder settings)
        "arch-timemil-JY": dict(
            engine="multi", channels=["JY"], label_policy="exclude_1",
            config=fixed_base_config(name="arch-timemil-JY", architecture="timemil",
                                     timemil_layers=2, heads=8, num_landmarks=64),
        ),
        "arch-crnn-JY": dict(
            engine="multi", channels=["JY"], label_policy="exclude_1",
            config=fixed_base_config(name="arch-crnn-JY", architecture="crnn",
                                     hidden_dim=256, rnn_layers=1, bidirectional=False),
        ),
        # 5: label policy (UNPAIRED (different dataset sizes))
        "labels-include1-JY": dict(
            engine="multi", channels=["JY"], label_policy="include_1",
            config=fixed_base_config(name="labels-include1-JY"),
        ),
        # 2: augmentation on/off
        "aug-off-JY": dict(
            engine="aug", augmentation="off",
            config=fixed_base_config(name="aug-off-JY", noise_std=0.0, scale_std=0.0),
        ),
        "aug-on-JY": dict(
            engine="aug", augmentation="on",
            config=fixed_base_config(name="aug-on-JY", noise_std=0.01, scale_std=0.0),
        ),
    }


# (experiment, control, paired?)
COMPARISONS = [
    ("aug-on-JY", "aug-off-JY", True),
    ("arch-timemil-JY", "baseline-JY", True),
    ("arch-crnn-JY", "baseline-JY", True),
    ("arch-timemil-BR-BZ-BY-JY", "channels-BR-BZ-BY-JY", True),
    ("arch-crnn-BR-BZ-BY-JY", "channels-BR-BZ-BY-JY", True),
    ("channels-BR-BZ", "baseline-JY", True),
    ("channels-BR-BZ-BY-JY", "baseline-JY", True),
    ("aug-off-JY", "baseline-JY", True), # engine-consistency check (both = JY, no aug)
    ("labels-include1-JY", "baseline-JY", False), # different test set size -> unpaired
]


#============================================================================
# Data stores and derived manifests

def rebase_manifest_paths(path_series):
    """Rewrites absolute npz paths onto FIELD_IMAGES_DIR."""
    rebased = []
    for raw_path in path_series:
        text = str(raw_path)
        marker = "Field-Images/"
        if marker not in text:
            raise ValueError(f"Cannot rebase manifest path (no '{marker}' segment): {text}")
        tail = text.split(marker, 1)[1]
        rebased.append(str(engine_multi.FIELD_IMAGES_DIR / tail))
    return rebased


def prepare_include1_manifest(manifests_dir):
    """Manifest for include_1.

    cropped_manifest.csv already has confidence-1 has confidence1 shots before any ml work,
    so those shots must be recovered from the raw label CSV at 
    Data-Cleaning/Plasmoids/shot_config_plasmoid_labels.csv). They are then appended with
    is_plasmoid remapped 1->2, so the store's logic keeps them as positives (since >=2 -> 1).
    """
    manifest = pd.read_csv(engine_multi.LABEL_MANIFEST_PATH)
    manifest.loc[manifest["is_plasmoid"] == 1, "is_plasmoid"] = 2

    raw_labels_csv = SCRIPT_DIR.parents[3] / "Data-Cleaning" / "Plasmoids" / "shot_config_plasmoid_labels.csv"
    raw_labels = pd.read_csv(raw_labels_csv, usecols=["Shot", "is_plasmoid"]).drop_duplicates("Shot")
    jy_dir = engine_multi.CHANNEL_DIRS["JY"]
    recovered = raw_labels[
        raw_labels["is_plasmoid"].eq(1)
        & ~raw_labels["Shot"].isin(manifest["Shot"])
        & raw_labels["Shot"].map(lambda shot: (jy_dir / f"shot{int(shot)}.npz").is_file())
    ].copy()
    if recovered.empty:
        raise RuntimeError(
            f"include_1 arm would be identical to the baseline: no confidence-1 shots "
            f"recoverable from {raw_labels_csv} (npz dir: {jy_dir})"
        )
    recovered["is_plasmoid"] = 2
    manifest = pd.concat([manifest[["Shot", "is_plasmoid"]], recovered], ignore_index=True)
    print(f"include_1 manifest: {len(manifest)} shots ({len(recovered)} confidence-1 shots recovered)")

    target = manifests_dir / "label_manifest_include1.csv"
    if not target.exists() or len(pd.read_csv(target)) != len(manifest):  # rebuild stale copies
        manifest.to_csv(target, index=False)
    return target


def prepare_aug_manifest(manifests_dir, augmented_manifest_path, originals_only):
    name = "aug_manifest_originals_only.csv" if originals_only else "aug_manifest_full.csv"
    target = manifests_dir / name
    if not target.exists():
        manifest = pd.read_csv(augmented_manifest_path)
        if "is_augmented" not in manifest.columns:
            if "augmentation" not in manifest.columns:
                raise ValueError(f"{augmented_manifest_path} lacks is_augmented/augmentation columns")
            manifest["is_augmented"] = (manifest["augmentation"].astype(str) != "original").astype(int)
        if originals_only:
            manifest = manifest.loc[manifest["is_augmented"] == 0].reset_index(drop=True)
        manifest["path"] = rebase_manifest_paths(manifest["path"])
        manifest.to_csv(target, index=False)
    return target


def build_store(spec, manifests_dir, augmented_manifest_path, preload):
    if spec["engine"] == "multi":
        if spec["label_policy"] == "include_1":
            manifest_path = prepare_include1_manifest(manifests_dir)
        else:
            manifest_path = engine_multi.LABEL_MANIFEST_PATH
        return engine_multi.FieldDataStore(manifest_path, spec["channels"], preload=preload)

    if spec["engine"] == "aug":
        manifest_path = prepare_aug_manifest(
            manifests_dir, augmented_manifest_path, originals_only=(spec["augmentation"] == "off")
        )
        return engine_aug.JYDataStore(manifest_path, preload=preload)

    raise ValueError(f"Unknown engine: {spec['engine']}")


def store_cache_key(spec):
    if spec["engine"] == "multi":
        return ("multi", tuple(spec["channels"]), spec["label_policy"])
    return ("aug", spec["augmentation"])


#============================================================================
# Seeded splits

def make_splits_seeded(store, engine, folds, cv_seed):
    original_indices = store.original_indices
    development_indices, test_indices = train_test_split(
        original_indices,
        test_size=engine.TEST_FRACTION,
        random_state=FIXED_SPLIT_SEED,
        stratify=store.labels[original_indices],
    )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    development_labels = store.labels[development_indices]
    cv_splits = []
    for train_local, val_local in splitter.split(development_indices, development_labels):
        train_indices = development_indices[train_local]
        val_indices = development_indices[val_local]
        if hasattr(store, "expand_training_indices"):
            train_indices = store.expand_training_indices(train_indices)
        cv_splits.append((train_indices, val_indices))

    return development_indices, test_indices, cv_splits


#============================================================================
# One run = one (experiment, seed)
# 3-fold CV -> final fit -> held-out test

def aggregate_fold_metrics(fold_results):
    aggregated = {}
    for metric in METRIC_NAMES:
        values = [result["best_metrics"][metric] for result in fold_results]
        aggregated[f"cv_mean_{metric}"] = float(np.mean(values))
        aggregated[f"cv_std_{metric}"] = float(np.std(values))
    aggregated["cv_mean_best_epoch"] = float(np.mean([result["best_epoch"] for result in fold_results]))
    aggregated["parameter_count"] = int(fold_results[0]["parameter_count"])
    return aggregated


def execute_run(engine, store, spec, config, run_dir, seed, args, device):
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        print(f"  seed {seed}: run_summary.json exists; skipping (resume).")
        return json.loads(summary_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    development_indices, test_indices, cv_splits = make_splits_seeded(store, engine, args.folds, cv_seed=seed)

    fold_results = []
    for fold_index, (train_indices, val_indices) in enumerate(cv_splits):
        mean, std = store.calculate_mean_std(train_indices)
        print(f"  seed {seed} fold {fold_index + 1}/{len(cv_splits)}: "
              f"train={len(train_indices)} val={len(val_indices)}", flush=True)
        result, _ = engine.train_one_split(
            store=store,
            train_indices=train_indices,
            val_indices=val_indices,
            mean=mean,
            std=std,
            config=config,
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
            min_epochs=args.min_epochs,
            primary_metric=args.primary_metric,
            seed=seed * 1000 + fold_index,
            history_path=run_dir / f"fold_{fold_index}_history.csv",
            keep_best_state=False,
        )
        fold_results.append(result)

    # Final fit + test eval
    original_engine_seed = engine.SEED
    engine.SEED = seed
    try:
        engine.fit_best_and_test(
            store, development_indices, test_indices, config, run_dir, device, args.primary_metric
        )
    finally:
        engine.SEED = original_engine_seed

    final_metrics = json.loads((run_dir / "final_test_metrics.json").read_text())

    summary = {
        "experiment": config["name"],
        "engine": spec["engine"],
        "seed": seed,
        "config": config,
        "development_shots": int(len(development_indices)),
        "development_positives": int(store.labels[development_indices].sum()),
        "test_shots": int(len(test_indices)),
        "test_positives": int(store.labels[test_indices].sum()),
        **aggregate_fold_metrics(fold_results),
        **{f"test_{key}": value for key, value in final_metrics["test_metrics"].items()},
        **{f"finalval_{key}": value for key, value in final_metrics["best_validation_metrics"].items()},
        "duration_minutes": (time.time() - start_time) / 60,
    }
    engine_multi.write_json(summary_path, summary)
    return summary


def summary_to_row(summary, spec):
    row = {key: value for key, value in summary.items() if key != "config"}
    config = summary["config"]
    row["architecture"] = config["architecture"]
    row["channels"] = "-".join(spec.get("channels", ["JY"]))
    row["label_policy"] = spec.get("label_policy", "exclude_1")
    row["augmentation"] = spec.get("augmentation", "none")
    row["config_json"] = json.dumps(engine_multi.json_ready(config), sort_keys=True)
    return row


#============================================================================
# Cross-trial summary and significance tests
# Wilcoxon signed-rank, Student's t tests, and one Welch's t-test

def format_mean_std(values):
    return f"{np.mean(values):.4f} +/- {np.std(values, ddof=1) if len(values) > 1 else 0.0:.4f}"


def write_summary(results_frame, output_root):
    lines = []
    lines.append("PAPER COMPARISON SUITE SUMMARY")
    lines.append(f"generated: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"rows: {len(results_frame)}  (one per experiment x seed)")
    lines.append("")

    key_metrics = ["test_f1", "test_pr_auc", "test_roc_auc", "cv_mean_f1", "cv_mean_pr_auc"]

    lines.append("=" * 90)
    lines.append("PER-ARM AGGREGATES (mean +/- std over seeds)")
    lines.append("=" * 90)
    for experiment, group in results_frame.groupby("experiment", sort=False):
        lines.append(f"\n{experiment}  (n={len(group)} seeds: {sorted(group['seed'].tolist())}, "
                     f"dev={int(group['development_shots'].iloc[0])}/"
                     f"{int(group['development_positives'].iloc[0])}+, "
                     f"test={int(group['test_shots'].iloc[0])}/"
                     f"{int(group['test_positives'].iloc[0])}+, "
                     f"params={int(group['parameter_count'].iloc[0])})")
        for metric in key_metrics:
            lines.append(f"    {metric:<16} {format_mean_std(group[metric].to_numpy())}")

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        scipy_stats = None
        lines.append("\n(scipy unavailable: significance tests skipped)")

    lines.append("")
    lines.append("=" * 90)
    lines.append("COMPARISONS")
    lines.append("=" * 90)
    for experiment, control, paired in COMPARISONS:
        experiment_group = results_frame[results_frame["experiment"] == experiment].sort_values("seed")
        control_group = results_frame[results_frame["experiment"] == control].sort_values("seed")
        if experiment_group.empty or control_group.empty:
            lines.append(f"\n{experiment} vs {control}: SKIPPED (missing runs)")
            continue

        lines.append(f"\n{experiment} vs {control} ({'paired by seed' if paired else 'UNPAIRED - different test sets'}):")
        for metric in ["test_f1", "test_pr_auc", "test_roc_auc", "cv_mean_f1"]:
            experiment_values = experiment_group[metric].to_numpy()
            control_values = control_group[metric].to_numpy()
            delta = float(np.mean(experiment_values) - np.mean(control_values))
            line = (f"    {metric:<16} {format_mean_std(experiment_values)}  vs  "
                    f"{format_mean_std(control_values)}  delta={delta:+.4f}")
            if scipy_stats is not None and len(experiment_values) > 1 and len(control_values) > 1:
                try:
                    if paired and len(experiment_values) == len(control_values) and \
                            experiment_group["seed"].tolist() == control_group["seed"].tolist():
                        t_result = scipy_stats.ttest_rel(experiment_values, control_values)
                        try:
                            w_result = scipy_stats.wilcoxon(experiment_values, control_values)
                            line += f"  paired-t p={t_result.pvalue:.3f} wilcoxon p={w_result.pvalue:.3f}"
                        except ValueError: # all differences zero
                            line += f"  paired-t p={t_result.pvalue:.3f}"
                    else:
                        t_result = scipy_stats.ttest_ind(experiment_values, control_values, equal_var=False)
                        line += f"  welch-t p={t_result.pvalue:.3f}"
                except Exception as stats_error:
                    line += f"  (stats failed: {stats_error})"
            lines.append(line)

    text = "\n".join(lines) + "\n"
    summary_path = output_root / "suite_summary.txt"
    summary_path.write_text(text)
    print("\n" + text)
    print(f"Summary written to {summary_path}")


#============================================================================
# Command line and main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS,
                        help="Comma-separated training seeds; one full CV+test run per seed per trial.")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated subset of experiment names (default: all).")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=engine_multi.DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=engine_multi.DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=engine_multi.DEFAULT_MIN_EPOCHS)
    parser.add_argument("--primary-metric", choices=["f1", "pr_auc"], default="f1")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: <BR_BZ_OUTPUT_DIR>/paper_comparison_suite")
    parser.add_argument("--aug-manifest", type=str, default=None,
                        help="Default: <BR_BZ_FIELD_IMAGES_DIR>/JY_cropped_05us_augmented/augmented_manifest.csv")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="1 seed, 2 folds, 2 epochs, separate _smoke output dir.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_dir) if args.output_dir else engine_multi.BASE_OUTPUT_DIR / "paper_comparison_suite"
    if args.smoke:
        output_root = output_root.parent / (output_root.name + "_smoke")
        args.folds, args.epochs, args.patience, args.min_epochs = 2, 2, 2, 1
        for engine_module in (engine_multi, engine_aug):
            engine_module.FINAL_MAX_EPOCHS = 2
            engine_module.FINAL_EARLY_STOPPING_PATIENCE = 2
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "_manifests"
    manifests_dir.mkdir(exist_ok=True)
    results_csv_path = output_root / "suite_results.csv"

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.smoke:
        seeds = seeds[:1]

    specs = experiment_specs()
    if args.experiments:
        requested = [name.strip() for name in args.experiments.split(",") if name.strip()]
        unknown = sorted(set(requested) - set(specs))
        if unknown:
            raise SystemExit(f"Unknown experiments: {unknown}; available: {sorted(specs)}")
        specs = {name: specs[name] for name in requested}

    if not TIMEMIL_AVAILABLE:
        for name in [key for key, spec in specs.items() if spec["config"]["architecture"] == "timemil"]:
            print(f"WARNING: dropping {name} (TimeMIL import failed: {TIMEMIL_IMPORT_ERROR})")
            specs.pop(name)

    if args.summarize_only:
        if not results_csv_path.exists():
            raise SystemExit(f"No results at {results_csv_path}")
        write_summary(pd.read_csv(results_csv_path), output_root)
        return

    augmented_manifest_path = Path(args.aug_manifest) if args.aug_manifest else (
        engine_multi.FIELD_IMAGES_DIR / "JY_cropped_05us_augmented" / "augmented_manifest.csv"
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Output root: {output_root}")
    print(f"Seeds: {seeds}")
    print(f"Experiments: {list(specs)}")

    rows = []
    current_store = None
    current_store_key = None
    suite_start = time.time()

    for experiment_name, spec in specs.items():
        engine = engine_multi if spec["engine"] == "multi" else engine_aug
        key = store_cache_key(spec)
        if key != current_store_key:
            if current_store is not None:
                del current_store
                gc.collect()
            print(f"\n{'=' * 88}\nLoading data store for {experiment_name}: {key}\n{'=' * 88}", flush=True)
            current_store = build_store(spec, manifests_dir, augmented_manifest_path, preload=not args.no_preload)
            current_store_key = key

        for seed in seeds:
            run_dir = output_root / "runs" / experiment_name / f"seed_{seed}"
            print(f"\n--- {experiment_name} / seed {seed} ---", flush=True)
            try:
                summary = execute_run(engine, current_store, spec, dict(spec["config"]), run_dir, seed, args, device)
            except Exception as error:
                print(f"RUN FAILED ({experiment_name}, seed {seed}): {type(error).__name__}: {error}")
                traceback.print_exc()
                continue
            rows.append(summary_to_row(summary, spec))
            pd.DataFrame(rows).to_csv(results_csv_path, index=False)

    print(f"\nSuite duration: {(time.time() - suite_start) / 3600:.2f} hours")
    if rows:
        write_summary(pd.DataFrame(rows), output_root)
    else:
        print("No completed runs; nothing to summarize.")


if __name__ == "__main__":
    main()
