#!/usr/bin/env python3
"""
The main goal of this script is to obtain data needed to highlight the inherent
interpretability of MIL models, especially the attention scores in the MIL pooling 
stage. There are some nice plots in attention_interpretability_plots.ipynb.

Retrains (or reuses an existing checkpoint for) the exact channels-BR-BZ-BY-JY
configuration from paper_comparison_suite.py (fixed base config, held-out test
split always seed 42, one chosen training seed -- default 101, matching that
suite's first seed for this arm), then runs every held-out test shot through
the trained model and records its per-frame attention weights (the softmax
over the 80 time frames that the model pools over before classifying), along
with the true/false positive/negative outcome at the model's validation-
selected threshold.

Outputs under --output-dir (default <BR_BZ_OUTPUT_DIR>/attention_interpretability/
BR_BZ_BY_JY_seed<seed>/):
  best_hypersearch_model.pth        trained model + config + norm stats + threshold
                                     (from engine.fit_best_and_test; unchanged format)
  final_test_metrics.json / final_test_predictions.csv / final_history.csv /
  misclassified_test_shots.csv      standard engine outputs from the final fit
  test_attention.npz                shots, labels, probabilities, predictions,
                                     categories, attention_weights [N, num_frames],
                                     frame_times_us, threshold
  test_attention_metadata.csv       Shot,label,probability,prediction,category (by Shot)
  attention_{tp,fp,fn,tn}.npz       same fields, filtered to one confusion-matrix category
  plots/<name>.png                  written on demand by --plot-* flags

Usage:
  python attention_interpretability.py                       # train (if needed) + extract
  python attention_interpretability.py --extract-only         # reuse existing checkpoint
  python attention_interpretability.py --plot-category tp
  python attention_interpretability.py --plot-shots 12345,23456
  python attention_interpretability.py --smoke                # 2-epoch correctness check

Can also be imported to call plot_attention_scores(...) directly, e.g. from a
notebook, after pointing npz_path at an existing test_attention.npz (or one of
the per-category files).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

#============================================================================
# BR-BZ multi-channel gated-attention engine (copy kept alongside this script,
# same self-contained pattern as MIL--Plasmoid--Classification/comparison-suite)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import br_bz_gated_attention_hyperparameter_search_261us_80frames_1us as engine

FIXED_SPLIT_SEED = 42 # must match paper_comparison_suite.py
CHANNELS = ["BR", "BZ", "BY", "JY"]
CATEGORIES = ("tp", "fp", "fn", "tn")
CATEGORY_LABELS = {"tp": "True positives", "fp": "False positives", "fn": "False negatives", "tn": "True negatives"}


def fixed_base_config(**overrides):
    """
    Duplicated (not imported) so this script has no dependency on TimeMIL/CRNN being
    importable; keep in sync manually with that file's fixed_base_config() if it changes.
    """
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


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def forward_with_attention(model, frames):
    """
    Replicates GatedAttentionMIL.forward but also returns the per-frame attention
    weights, without modifying the shared engine module (weights are a local variable
    in the original forward, never stored on the model).
    """
    embeddings = model.frame_encoder(frames)
    gated = torch.tanh(model.attention_v(embeddings)) * torch.sigmoid(model.attention_u(embeddings))
    weights = torch.softmax(model.attention_w(gated).squeeze(-1), dim=1)
    pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
    logits = model.classifier(pooled).squeeze(-1)
    return logits, weights


def make_test_split(store):
    return train_test_split(
        store.original_indices, test_size=engine.TEST_FRACTION,
        random_state=FIXED_SPLIT_SEED, stratify=store.labels[store.original_indices],
    )


def train_or_load(store, run_dir, config, seed, device, primary_metric):
    development_indices, test_indices = make_test_split(store)

    original_seed = engine.SEED
    engine.SEED = seed
    try:
        engine.fit_best_and_test(store, development_indices, test_indices, config, run_dir, device, primary_metric)
    finally:
        engine.SEED = original_seed

    return development_indices, test_indices


def extract_test_attention(store, test_indices, checkpoint, device):
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    threshold = float(checkpoint["validation_threshold"])

    model = engine.build_model(checkpoint["config"], store.field_shape).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = engine.IndexedFieldDataset(store, test_indices, mean, std)
    # shuffle=False -> loader seed only affects an unused generator; NUM_WORKERS=0 by
    # default so seed_worker never fires either, hence a fixed constant is fine here.
    loader = engine.make_loader(dataset, batch_size=checkpoint["config"]["batch_size"], shuffle=False,
                                 device=device, seed=0)

    all_shots, all_labels, all_probabilities, all_weights = [], [], [], []
    with torch.no_grad():
        for frames, labels, shots in loader:
            frames = frames.to(device, non_blocking=True)
            logits, weights = forward_with_attention(model, frames)
            all_shots.append(shots.numpy())
            all_labels.append(labels.numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_weights.append(weights.cpu().numpy())

    shots = np.concatenate(all_shots)
    labels = np.concatenate(all_labels).astype(np.int64)
    probabilities = np.concatenate(all_probabilities)
    attention_weights = np.concatenate(all_weights)   # [N, num_frames]
    predictions = (probabilities >= threshold).astype(np.int64)

    categories = np.full(len(shots), "", dtype=object)
    categories[(labels == 1) & (predictions == 1)] = "tp"
    categories[(labels == 0) & (predictions == 1)] = "fp"
    categories[(labels == 1) & (predictions == 0)] = "fn"
    categories[(labels == 0) & (predictions == 0)] = "tn"

    order = np.argsort(shots)
    return {
        "shots": shots[order],
        "labels": labels[order],
        "probabilities": probabilities[order],
        "predictions": predictions[order],
        "categories": categories[order],
        "attention_weights": attention_weights[order],
        "frame_times_us": np.asarray(store.time_values_us, dtype=np.float64),
        "threshold": threshold,
    }


def save_attention_data(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "test_attention.npz",
        shots=data["shots"], labels=data["labels"], probabilities=data["probabilities"],
        predictions=data["predictions"], categories=data["categories"].astype(str),
        attention_weights=data["attention_weights"], frame_times_us=data["frame_times_us"],
        threshold=data["threshold"],
    )

    metadata = pd.DataFrame({
        "Shot": data["shots"], "label": data["labels"], "probability": data["probabilities"],
        "prediction": data["predictions"], "category": data["categories"],
    }).sort_values("Shot")
    metadata.to_csv(output_dir / "test_attention_metadata.csv", index=False)

    for category in CATEGORIES:
        mask = data["categories"] == category
        np.savez(
            output_dir / f"attention_{category}.npz",
            shots=data["shots"][mask], labels=data["labels"][mask],
            probabilities=data["probabilities"][mask], predictions=data["predictions"][mask],
            attention_weights=data["attention_weights"][mask], frame_times_us=data["frame_times_us"],
            threshold=data["threshold"],
        )

    print(
        f"Saved attention data for {len(data['shots'])} test shots to {output_dir} "
        f"(tp={int((data['categories'] == 'tp').sum())}, fp={int((data['categories'] == 'fp').sum())}, "
        f"fn={int((data['categories'] == 'fn').sum())}, tn={int((data['categories'] == 'tn').sum())})"
    )


def load_attention_data(npz_path):
    with np.load(npz_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def plot_attention_scores(data=None, npz_path=None, shot=None, shots=None, category=None,
                           show_mean=True, title=None, show_title=False, output_path=None, ax=None,
                           low_start_frame=None, high_start_frame=None, n_frames=None, frame_stride_us=2.0):
    """
    Plots attention weight vs frame time (us) for one or more held-out test shots.

    Exactly one of `shot` (a single shot number), `shots` (an iterable of shot numbers,
    overlaid as individual lines), or `category` (one of 'tp'/'fp'/'fn'/'tn'/'all', must be given.
    """
    if data is None:
        if npz_path is None:
            raise ValueError("Provide either `data` or `npz_path`.")
        data = load_attention_data(npz_path)

    selectors = [value is not None for value in (shot, shots, category)]
    if sum(selectors) != 1:
        raise ValueError("Pass exactly one of shot=, shots=, or category=.")

    frame_times = np.asarray(data["frame_times_us"], dtype=np.float64)
    shot_to_index = {int(value): index for index, value in enumerate(data["shots"])}

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    else:
        fig = ax.figure

    has_highlights = low_start_frame is not None or high_start_frame is not None
    if has_highlights:
        if n_frames is None:
            raise ValueError("n_frames is required when shading low_start_frame/high_start_frame.")
        stride = int(round(frame_stride_us / engine.TIME_STEP_US))
        for start_frame, color, region_label in (
            (low_start_frame, "tab:orange", "low-attention frames"),
            (high_start_frame, "tab:green", "high-attention frames"),
        ):
            if start_frame is None:
                continue
            end_frame = start_frame + stride * (n_frames - 1)
            # +/- half a frame step centers the shaded span on the discrete frame markers at its edges.
            half_step = engine.TIME_STEP_US / 2.0
            ax.axvspan(frame_times[start_frame] - half_step, frame_times[end_frame] + half_step,
                       color=color, alpha=0.15, label=region_label)

    if category is not None:
        if category == "all":
            mask = np.ones(len(data["shots"]), dtype=bool)
        else:
            mask = data["categories"].astype(str) == category
        shots_to_plot = data["shots"][mask].tolist()
        if not shots_to_plot:
            raise ValueError(f"No test shots found for category={category!r}")

        curves = np.stack([data["attention_weights"][shot_to_index[value]] for value in shots_to_plot])
        for curve in curves:
            # unmarked: with many overlaid shots, per-frame markers here would just clutter the plot.
            ax.plot(frame_times, curve, color="tab:blue", alpha=0.15, linewidth=0.8)
        if show_mean:
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0)
            ax.plot(frame_times, mean_curve, marker="o", markersize=3, color="tab:blue", linewidth=2.2,
                    label=f"mean (n={len(curves)})")
            ax.fill_between(frame_times, mean_curve - std_curve, mean_curve + std_curve,
                             color="tab:blue", alpha=0.2, label="\u00b1 1 std")
        ax.legend(frameon=False)
        default_title = CATEGORY_LABELS.get(category, category)
    else:
        shots_to_plot = [int(shot)] if shot is not None else [int(value) for value in shots]
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(shots_to_plot), 10)))
        for index, value in enumerate(shots_to_plot):
            if value not in shot_to_index:
                raise ValueError(f"Shot {value} is not in the held-out test set attention data.")
            # markers to make clear that each frame is a discrete point
            ax.plot(frame_times, data["attention_weights"][shot_to_index[value]], marker="o", markersize=3,
                    color=colors[index % len(colors)], linewidth=1.6, label=f"shot {value}")
        if len(shots_to_plot) > 1 or has_highlights:
            ax.legend(frameon=False, fontsize=8, ncol=2)
        default_title = f"Shot {shots_to_plot[0]}" if len(shots_to_plot) == 1 else f"{len(shots_to_plot)} shots"

    ax.set_xlabel("Time (\u03bcs)")
    ax.set_ylabel("Attention weight")
    if show_title:
        ax.set_title(title or default_title)
    ax.set_xlim(frame_times[0], frame_times[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, ax


#============================================================================
# Raw field-frame grids (low- vs high-attention frame content)

FIELD_PLOT_CONFIGS = {
    "BY": {"label": "BY", "unit": "G", "coe": 1e4, "cmap": "RdBu_r", "clim": (-500, 500)},
    "JY": {"label": "-JY", "unit": r"$MA/m^2$", "coe": -1e-6, "cmap": "BrBG", "clim": (-1.25, 1.25)},
    "BR": {"label": "BR", "unit": "G", "coe": 1e4, "cmap": "RdBu_r", "clim": (-250, 250)},
    "BZ": {"label": "BZ", "unit": "G", "coe": 1e4, "cmap": "RdBu_r", "clim": (-600, 600)},
}


def load_shot_channel_frames(shot, channel):
    """
    Loads one shot's raw field frames for the model's 80frame window (261-340us, 
    1us step) straight from its npz file

    Returns (frames [80, R, Z], frame_times_us [80], R [R], Z [Z]).
    """
    path = engine.CHANNEL_DIRS[channel] / f"shot{int(shot)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No {channel} data for shot {shot}: {path}")

    with np.load(path, allow_pickle=False) as archive:
        full_field = archive["field"].astype(np.float32)
        raw_time = archive[engine.TIME_KEY]
        R = np.asarray(archive["R"], dtype=np.float64)
        Z = np.asarray(archive["Z"], dtype=np.float64)

    time_us, _ = engine.FieldDataStore._convert_time_to_microseconds(raw_time)
    time_step_us = float(np.median(np.diff(time_us)))
    start_index = int(np.argmin(np.abs(time_us - engine.TIME_START_US)))
    native_stride = int(round(engine.TIME_STEP_US / time_step_us))
    indices = start_index + native_stride * np.arange(engine.NUM_TIME_FRAMES, dtype=np.int64)

    return full_field[indices], time_us[indices], R, Z


def frame_range_indices(start_frame, n_frames, frame_stride_us=2.0):
    stride = int(round(frame_stride_us / engine.TIME_STEP_US))
    if stride < 1:
        raise ValueError(f"frame_stride_us={frame_stride_us} is finer than the model's "
                          f"own {engine.TIME_STEP_US} us frame step.")
    indices = start_frame + stride * np.arange(n_frames, dtype=np.int64)
    if indices[0] < 0 or indices[-1] >= engine.NUM_TIME_FRAMES:
        raise ValueError(f"Requested frames [{indices[0]}, {indices[-1]}] fall outside the "
                          f"0-{engine.NUM_TIME_FRAMES - 1} window (start_frame={start_frame}, "
                          f"n_frames={n_frames}, stride={stride} frames).")
    return indices


def find_attention_window(data, shot, n_frames=10, frame_stride_us=2.0, mode="max"):
    shot_to_index = {int(value): index for index, value in enumerate(data["shots"])}
    if int(shot) not in shot_to_index:
        raise ValueError(f"Shot {shot} is not in the held-out test set attention data.")
    weights = data["attention_weights"][shot_to_index[int(shot)]]

    stride = int(round(frame_stride_us / engine.TIME_STEP_US))
    last_valid_start = engine.NUM_TIME_FRAMES - 1 - stride * (n_frames - 1)
    if last_valid_start < 0:
        raise ValueError(f"A {n_frames}-frame window at {frame_stride_us}us spacing does not "
                          f"fit inside the {engine.NUM_TIME_FRAMES}-frame range.")

    best_start, best_score = None, None
    for start in range(last_valid_start + 1):
        score = float(weights[start + stride * np.arange(n_frames)].mean())
        if best_score is None or (score > best_score if mode == "max" else score < best_score):
            best_start, best_score = start, score
    return best_start, best_score


def _draw_frame_panel(ax, frame, R, Z, channel, use_contour):
    conf = FIELD_PLOT_CONFIGS[channel]
    # image_cache.py already multiplies by conf["coe"] when building these cropped npz
    # caches (field_sel = fields[name][idx] * conf["coe"]). As such, `frame` here is already in
    # its final display units (G / MA/m^2). Do not re-apply coe o/w values get scaled twice.
    if use_contour:
        mesh = ax.contourf(Z, R, frame, cmap=conf["cmap"], levels=21,
                            vmin=conf["clim"][0], vmax=conf["clim"][1])
    else:
        # Z on x / R on y: horizontal panel orientation (Z span 41cm > R span 23.5cm)
        mesh = ax.pcolormesh(Z, R, frame, shading="auto", cmap=conf["cmap"],
                              vmin=conf["clim"][0], vmax=conf["clim"][1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def _add_axis_direction_arrows(ax, origin=(0.06, 0.1), color="black"):
    """Small orientation indicator: arrows along +z (right) and +R (up) in a panel corner."""
    # Panel box aspect ~ Z-span/R-span (~1.74), so the vertical fraction is larger
    # to give both arrows roughly equal on-screen length.
    dx, dy = 0.16, 0.28
    arrow_props = dict(arrowstyle="-|>", color=color, linewidth=1.0, shrinkA=0, shrinkB=0,
                       mutation_scale=7)
    ax.annotate("", xy=(origin[0] + dx, origin[1]), xytext=origin, xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(origin[0], origin[1] + dy), xytext=origin, xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=arrow_props)
    ax.text(origin[0] + dx + 0.03, origin[1], "z", transform=ax.transAxes, fontsize=7,
            ha="left", va="center", color=color)
    ax.text(origin[0], origin[1] + dy + 0.04, "R", transform=ax.transAxes, fontsize=7,
            ha="center", va="bottom", color=color)


def plot_shot_frame_grid(shot, channel, high_start_frame, low_start_frame=0, n_frames=10,
                          frame_stride_us=2.0, use_contour=False, output_path=None):
    """2 rows x n_frames columns of raw field frames for one channel"""
    frames, frame_times_us, R, Z = load_shot_channel_frames(shot, channel)
    row_starts = [low_start_frame, high_start_frame]
    row_labels = ["low-attention region", "high-attention region"]

    fig, axes = plt.subplots(2, n_frames, figsize=(1.5 * n_frames, 3.4), dpi=150)
    mesh = None
    for row, start_frame in enumerate(row_starts):
        indices = frame_range_indices(start_frame, n_frames, frame_stride_us)
        for column, frame_index in enumerate(indices):
            ax = axes[row, column]
            mesh = _draw_frame_panel(ax, frames[frame_index], R, Z, channel, use_contour)
            ax.set_title(f"{frame_times_us[frame_index]:.1f} \u03bcs", fontsize=7, pad=2)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=8)

    conf = FIELD_PLOT_CONFIGS[channel]
    fig.colorbar(mesh, ax=axes.ravel().tolist(), location="right", shrink=0.8, pad=0.02,
                 label=f"{conf['label']} ({conf['unit']})")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, axes


def plot_shot_multichannel_frame_grid(shot, high_start_frame, low_start_frame=0, n_frames=5,
                                       frame_stride_us=2.0, channels=("BR", "BZ", "BY", "JY"),
                                       use_contour=False, gap_width_ratio=0.4, output_path=None):
    """4 rows (one per channel) x (n_frames left + gap + n_frames right) columns"""
    gap_column = n_frames
    width_ratios = [1] * n_frames + [gap_width_ratio] + [1] * n_frames
    fig, axes = plt.subplots(len(channels), 2 * n_frames + 1, figsize=(2.2 * (2 * n_frames + 1), 1.3 * len(channels)),
                              dpi=150, gridspec_kw={"width_ratios": width_ratios})

    for row, channel in enumerate(channels):
        frames, frame_times_us, R, Z = load_shot_channel_frames(shot, channel)
        axes[row, gap_column].axis("off") # the gap itself carries no plot

        mesh = None
        # high-attention frames on the left, low-attention on the right
        for side, start_frame in enumerate([high_start_frame, low_start_frame]):
            indices = frame_range_indices(start_frame, n_frames, frame_stride_us)
            column_offset = 0 if side == 0 else gap_column + 1
            for column, frame_index in enumerate(indices):
                ax = axes[row, column_offset + column]
                mesh = _draw_frame_panel(ax, frames[frame_index], R, Z, channel, use_contour)
                ax.set_title(f"{frame_times_us[frame_index]:.1f} \u03bcs", fontsize=6, pad=2)

        conf = FIELD_PLOT_CONFIGS[channel]
        row_axes = [axes[row, column] for column in range(2 * n_frames + 1) if column != gap_column]
        axes[row, 0].set_ylabel(conf["label"], fontsize=9)
        fig.colorbar(mesh, ax=row_axes, location="right", shrink=0.85, pad=0.02, label=conf["unit"])

    _add_axis_direction_arrows(axes[len(channels) - 1, 0])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, axes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=101,
                        help="Matches one of paper_comparison_suite.py's channels-BR-BZ-BY-JY seeds (101-110).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: <BR_BZ_OUTPUT_DIR>/attention_interpretability/BR_BZ_BY_JY_seed<seed>")
    parser.add_argument("--primary-metric", choices=["f1", "pr_auc"], default="f1")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip training; require an existing best_hypersearch_model.pth in --output-dir.")
    parser.add_argument("--smoke", action="store_true", help="2-epoch quick correctness check.")
    parser.add_argument("--plot-shot", type=int, default=None)
    parser.add_argument("--plot-shots", type=str, default=None, help="Comma-separated shot numbers to overlay.")
    parser.add_argument("--plot-category", choices=["tp", "fp", "fn", "tn", "all"], default=None)
    parser.add_argument("--plot-output", type=str, default=None, help="Default: <output-dir>/plots/<name>.png")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        engine.BASE_OUTPUT_DIR / "attention_interpretability" / f"BR_BZ_BY_JY_seed{args.seed}"
    )

    if args.smoke:
        output_dir = output_dir.parent / (output_dir.name + "_smoke")
        engine.FINAL_MAX_EPOCHS = 2
        engine.FINAL_EARLY_STOPPING_PATIENCE = 2
        engine.DEFAULT_MIN_EPOCHS = 1

    device = select_device()
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    config = fixed_base_config(name="attention-interpretability-BR-BZ-BY-JY")
    store = engine.FieldDataStore(engine.LABEL_MANIFEST_PATH, CHANNELS, preload=not args.no_preload)

    checkpoint_path = output_dir / "best_hypersearch_model.pth"
    if args.extract_only:
        if not checkpoint_path.exists():
            raise SystemExit(f"--extract-only given but {checkpoint_path} does not exist.")
        _, test_indices = make_test_split(store)
    else:
        _, test_indices = train_or_load(store, output_dir, config, args.seed, device, args.primary_metric)

    # Self-produced checkpoint containing numpy arrays/lists alongside the state dict,
    # so weights_only=True (torch>=2.6 default) would fail to unpickle it.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data = extract_test_attention(store, test_indices, checkpoint, device)
    save_attention_data(data, output_dir)

    if args.plot_shot is not None or args.plot_shots is not None or args.plot_category is not None:
        shots = [int(value) for value in args.plot_shots.split(",")] if args.plot_shots else None
        if args.plot_shot is not None:
            name = f"shot_{args.plot_shot}"
        elif shots:
            name = f"shots_{'_'.join(map(str, shots))}"
        else:
            name = f"category_{args.plot_category}"
        plot_output = Path(args.plot_output) if args.plot_output else output_dir / "plots" / f"{name}.png"
        plot_attention_scores(data=data, shot=args.plot_shot, shots=shots, category=args.plot_category,
                               output_path=plot_output)


if __name__ == "__main__":
    main()
