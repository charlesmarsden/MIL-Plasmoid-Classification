#!/usr/bin/env python3
"""
Gated-attention MIL far reaching hyperparameter search for combinations of field images.

Adapted from jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented.py.
Any combination of BR-BZ-BY-JY works.

The test set evaluation writes misclassified_test_shots.csv, indicating FP/FN.

Search considers:

- CNN architecture: plain or residual (residual connections supposedly solve the vanishing gradient and degradation problems)
- CNN channel widths
- embedding dimension
- dropout rate
- learning rate and weight decay
- batch size
- weighted or unweighted BCE
- scheduler (affects how lr changes with epoch
             none: constant
             cosine: lr decays along a cosine curve from its initial value to ~0 at max_epochs
             plateau: lr constant until validation metrix (F1) stops improving for two epochs (patience=2),
                       then decreases by half (factor=0.5). floor is 1e-6)

"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve, 
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


#=============================================================================
# Paths and default search budget

TIME_KEY = "T"
TIME_START_US = 261.0
TIME_STEP_US = 1.0
NUM_TIME_FRAMES = 80
TIME_END_US = TIME_START_US + TIME_STEP_US * (NUM_TIME_FRAMES - 1)

# Overridable via env vars so the exact same .py file can run unmodified on Flux
# (default, /u/cmarsden/docs/PPPL/...) or on Stellar (/u/cmarsden is NOT mounted
# there -- see BR-BZ/job-related-slurm-sh-files/stellar_setup_br_bz.sh), without
# editing this file.
FIELD_IMAGES_DIR = Path(
    os.environ.get(
        "BR_BZ_FIELD_IMAGES_DIR",
        "/u/cmarsden/docs/PPPL/"
        "Advanced-Plasmoid-Classification/Image-Data/"
        "Field-Images",
    )
)

# BR and BZ npz files have no manifest of their own (just shot<N>.npz per shot), so the
# shot list and is_plasmoid labels are taken from the existing JY manifest, which covers
# the same set of shots. The label policy (score 2/3 positive, score 1 excluded) is a
# property of the shot, not of which field is being viewed.
LABEL_MANIFEST_PATH = FIELD_IMAGES_DIR / "JY_cropped_05us" / "cropped_manifest.csv"

CHANNEL_DIRS = {
    "BR": FIELD_IMAGES_DIR / "BR_cropped_05us",
    "BZ": FIELD_IMAGES_DIR / "BZ_cropped_05us",
    "BY": FIELD_IMAGES_DIR / "BY_cropped_05us",
    "JY": FIELD_IMAGES_DIR / "JY_cropped_05us",
}

DEFAULT_CHANNELS = "BR-BZ"

BASE_OUTPUT_DIR = Path(
    os.environ.get(
        "BR_BZ_OUTPUT_DIR",
        "/u/cmarsden/docs/PPPL/"
        "Advanced-Plasmoid-Classification/Image-Data/"
        "TimeMIL-output",
    )
)

SEED = 42
TEST_FRACTION = 0.15
FINAL_VAL_FRACTION_OF_DEVELOPMENT = 0.17647058823529413  # Gives 70/15/15 overall.

DEFAULT_TRIALS = 24
DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_EPOCHS = 6
FINAL_MAX_EPOCHS = 40
FINAL_EARLY_STOPPING_PATIENCE = 7

PRIMARY_METRIC = "f1"  # Change to "pr_auc" for threshold-independent selection.
NUM_WORKERS = 0
PRELOAD_DATA = True
USE_AMP = False
GRADIENT_CLIP_NORM = 2.0
MIN_IMPROVEMENT = 1e-4


def resolve_channels(channels_argument):
    """Turns --channels into an ordered channel-name list.

    Accepts any combination of BR, BZ, BY, JY separated by '-', '_', ',' or '+'
    (case-insensitive), e.g. BR-BZ, BR-BZ-JY, BR-BZ-BY-JY, or a single field like
    BR. BOTH is kept as a legacy alias for BR-BZ. Order is preserved as given and
    determines the channel stacking order and the output directory name.
    """
    normalized = str(channels_argument).strip().upper()
    if normalized == "BOTH":
        return ["BR", "BZ"]

    for separator in ("_", ",", "+"):
        normalized = normalized.replace(separator, "-")
    channels = [part for part in normalized.split("-") if part]

    if not channels:
        raise ValueError(f"--channels value {channels_argument!r} selects no channels")
    unknown = [channel for channel in channels if channel not in CHANNEL_DIRS]
    if unknown:
        raise ValueError(
            f"Unknown channel(s) {unknown} in --channels value {channels_argument!r}; "
            f"expected a combination of {sorted(CHANNEL_DIRS)} (e.g. BR-BZ, BR-BZ-JY, BR-BZ-BY-JY)"
        )
    if len(set(channels)) != len(channels):
        raise ValueError(f"--channels value {channels_argument!r} lists a channel more than once")
    return channels


def channels_label(channels):
    return "_".join(channels)


def resolve_output_dir(channels, primary_metric):
    """Output tree for a given channel combo + selection metric.

    "_new_labels" separates this run's whole output tree (trial/fold caches,
    leaderboard.csv, best_config.json, etc.) from any earlier run under the same
    channels, since trial/fold results are resumed from disk by checking for an
    existing result.json -- reusing the old directory after a label update would
    silently skip folds and keep stale results computed under the old labels.
    A non-default primary_metric similarly gets its own suffix (e.g. "_prauc") so
    it doesn't resume/mix with folds cached under a different selection metric.
    """
    metric_suffix = "" if primary_metric == "f1" else f"_{primary_metric.replace('_', '')}"
    return BASE_OUTPUT_DIR / f"{channels_label(channels)}_261us_80frames_1us_gated_new_labels{metric_suffix}"


#=============================================================================
# Utilities

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True))
    temporary_path.replace(path)


def available_memory_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_dict_to_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def metric_value(metrics, metric_name):
    value = float(metrics[metric_name])
    if not math.isfinite(value):
        return -math.inf
    return value


#=============================================================================
# Data

class FieldDataStore:
    """Owns shot-level labels plus windowed field data for the requested channels (BR/BZ/BY/JY).

    Field npz directories have no manifest of their own (just shot<N>.npz per shot), so the
    shot list and is_plasmoid labels are taken from the JY manifest (label policy is a
    property of the shot, not of which field is being viewed). No augmentation is used:
    every shot contributes exactly one sample.
    """

    def __init__(self, manifest_path, channels, preload=True):
        self.manifest_path = Path(manifest_path)
        self.channels = list(channels)
        if not self.channels:
            raise ValueError("At least one channel (BR and/or BZ) must be selected.")
        unknown_channels = [channel for channel in self.channels if channel not in CHANNEL_DIRS]
        if unknown_channels:
            raise ValueError(f"Unknown channels: {unknown_channels}; expected a subset of {sorted(CHANNEL_DIRS)}")

        self.manifest = pd.read_csv(self.manifest_path)

        required_columns = {"Shot", "is_plasmoid"}
        missing_columns = required_columns.difference(self.manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        self.manifest["Shot"] = pd.to_numeric(self.manifest["Shot"], errors="raise").astype(np.int64)
        self.manifest["is_plasmoid"] = pd.to_numeric(self.manifest["is_plasmoid"], errors="raise").astype(np.int64)
        unexpected_scores = sorted(set(self.manifest["is_plasmoid"].unique().tolist()) - {0, 1, 2, 3})
        if unexpected_scores:
            raise ValueError(f"Unexpected is_plasmoid values: {unexpected_scores}")

        confidence_one_mask = self.manifest["is_plasmoid"].eq(1)
        self.excluded_confidence_one_shots = int(self.manifest.loc[confidence_one_mask, "Shot"].nunique())
        self.manifest = self.manifest.loc[~confidence_one_mask].reset_index(drop=True)
        self.manifest["label"] = self.manifest["is_plasmoid"].isin({2, 3}).astype(np.int64)

        duplicate_shots = self.manifest.loc[self.manifest["Shot"].duplicated(), "Shot"].unique().tolist()
        if duplicate_shots:
            raise ValueError(f"Manifest has duplicate Shot rows: {sorted(duplicate_shots)[:5]}")

        self.shots = self.manifest["Shot"].astype(np.int64).to_numpy()
        self.labels = self.manifest["label"].astype(np.int64).to_numpy()

        unique_labels = set(np.unique(self.labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Expected binary labels 0/1, found {sorted(unique_labels)}")

        # No augmentation: every row is an original, physical shot.
        self.original_indices = np.arange(len(self.manifest), dtype=np.int64)
        self.num_physical_shots = len(self.original_indices)

        self.channel_paths = {}
        for channel in self.channels:
            channel_dir = CHANNEL_DIRS[channel]
            paths = [channel_dir / f"shot{shot}.npz" for shot in self.shots]
            missing_paths = [path for path in paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(
                    f"{len(missing_paths)} {channel} files are missing. First missing file: {missing_paths[0]}"
                )
            self.channel_paths[channel] = paths

        self.fields = None
        self.shot_sums = None            # [num_shots, num_channels]
        self.shot_squared_sums = None    # [num_shots, num_channels]
        self.shot_counts = None          # [num_shots, num_channels] (voxels per channel)
        self.single_channel_shape = None  # (T, H, W) shared by every channel before stacking
        self.field_shape = None          # (T, num_channels, H, W)
        self.time_values_us = None
        self.time_step_us = None
        self.source_time_unit = None

        first_field, first_time_us, source_time_unit = self._load_windowed_multichannel(0, establish_reference=True)
        self.field_shape = tuple(first_field.shape)
        self.time_values_us = first_time_us
        self.time_step_us = float(np.median(np.diff(first_time_us)))
        self.source_time_unit = source_time_unit

        estimated_bytes = first_field.nbytes * len(self.shots)
        available_bytes = available_memory_bytes()

        print(
            f"Dataset: {len(self.shots)} shots, "
            f"positive shots={int(self.labels.sum())}, "
            f"channels={self.channels}, field shape={self.field_shape}"
        )
        print(
            f"Label policy: scores 2/3 are positive; excluded "
            f"{self.excluded_confidence_one_shots} confidence-1 shots."
        )
        print(
            f"Selected time sampling: start={self.time_values_us[0]:.6g} us, "
            f"end={self.time_values_us[-1]:.6g} us, frames={len(self.time_values_us)}, "
            f"step={self.time_step_us:.6g} us, source {TIME_KEY} unit={self.source_time_unit}"
        )
        print(f"Estimated windowed field cache: {estimated_bytes / 2**30:.2f} GiB")

        # crashing before the model starts training would be bad
        if preload and available_bytes is not None and estimated_bytes > 0.60 * available_bytes:
            print(
                "Disabling preload because the estimated cache exceeds 60% of "
                f"available RAM ({available_bytes / 2**30:.2f} GiB)."
            )
            preload = False

        if preload:
            self._preload()
        else:
            print("Using on-demand .npz loading.")
            self._scan_shot_statistics()

    @staticmethod # decorator defining a method of a class without an explicit self argument
    def _convert_time_to_microseconds(raw_time):
        raw_time = np.asarray(raw_time, dtype=np.float64).reshape(-1)
        if raw_time.size < 2:
            raise ValueError(f"Expected at least two values in {TIME_KEY}, found {raw_time.size}")
        if not np.isfinite(raw_time).all():
            raise ValueError(f"{TIME_KEY} contains NaN or infinite values")

        max_abs_time = float(np.max(np.abs(raw_time)))
        if max_abs_time < 1.0:
            return raw_time * 1e6, "seconds"
        return raw_time, "microseconds"

    def _load_windowed_single_channel(self, path, establish_reference=False):
        with np.load(path, allow_pickle=False) as data:
            if "field" not in data:
                raise KeyError(f"{path} does not contain the required 'field' array")
            if TIME_KEY not in data:
                raise KeyError(
                    f"{path} does not contain the required {TIME_KEY!r} time array. "
                    "The requested 261 us start with 1 us spacing cannot be made safely without it."
                )

            full_field = data["field"].astype(np.float32)
            time_us, source_time_unit = self._convert_time_to_microseconds(data[TIME_KEY])

        if full_field.ndim != 3:
            raise ValueError(f"Expected field shape [T,H,W] in {path}, found {full_field.shape}")
        if len(time_us) != full_field.shape[0]:
            raise ValueError(
                f"Time/field length mismatch in {path}: len({TIME_KEY})={len(time_us)} "
                f"but field has {full_field.shape[0]} frames"
            )

        time_differences = np.diff(time_us)
        if np.any(time_differences <= 0):
            raise ValueError(f"{TIME_KEY} must be strictly increasing in {path}")

        time_step_us = float(np.median(time_differences))
        step_tolerance = max(1e-8, abs(time_step_us) * 1e-4)
        if not np.allclose(time_differences, time_step_us, rtol=1e-4, atol=step_tolerance):
            raise ValueError(f"Non-uniform time step detected in {path}")

        endpoint_tolerance = max(1e-6, abs(time_step_us) * 1e-3)
        start_index = int(np.argmin(np.abs(time_us - TIME_START_US)))

        if abs(float(time_us[start_index]) - TIME_START_US) > endpoint_tolerance:
            raise ValueError(
                f"{path} has no frame at {TIME_START_US} us; nearest is "
                f"{time_us[start_index]:.9g} us"
            )

        stride_in_native_steps = TIME_STEP_US / time_step_us
        integer_stride = int(round(stride_in_native_steps))
        if integer_stride < 1:
            raise ValueError(
                f"Requested sampling stride {TIME_STEP_US} us is smaller than the native "
                f"time step {time_step_us:.9g} us in {path}"
            )
        if abs(integer_stride * time_step_us - TIME_STEP_US) > endpoint_tolerance:
            raise ValueError(
                f"Requested sampling stride {TIME_STEP_US} us is not an integer multiple of "
                f"the native time step {time_step_us:.9g} us in {path}"
            )

        selected_indices = start_index + integer_stride * np.arange(NUM_TIME_FRAMES, dtype=np.int64)
        if selected_indices[-1] >= len(time_us):
            raise ValueError(
                f"{path} does not contain {NUM_TIME_FRAMES} frames starting at {TIME_START_US} us "
                f"with {TIME_STEP_US} us spacing"
            )

        expected_time_us = TIME_START_US + TIME_STEP_US * np.arange(NUM_TIME_FRAMES, dtype=np.float64)
        selected_time_us = np.ascontiguousarray(time_us[selected_indices])
        if not np.allclose(selected_time_us, expected_time_us, rtol=1e-7, atol=endpoint_tolerance):
            raise ValueError(
                f"Selected frame times in {path} do not match the requested sequence from "
                f"{TIME_START_US} us to {TIME_END_US} us at {TIME_STEP_US} us spacing"
            )

        selected_field = np.ascontiguousarray(full_field[selected_indices], dtype=np.float32)

        if establish_reference:
            return selected_field, selected_time_us, source_time_unit

        if tuple(selected_field.shape) != self.single_channel_shape:
            raise ValueError(
                f"Inconsistent windowed field shape in {path}: {selected_field.shape}; "
                f"expected {self.single_channel_shape}"
            )
        if not np.allclose(selected_time_us, self.time_values_us, rtol=1e-7, atol=max(1e-7, abs(self.time_step_us) * 1e-5),):
            raise ValueError(f"Selected time grid in {path} differs from the first shot")

        return selected_field, selected_time_us, source_time_unit

    def _load_windowed_multichannel(self, shot_index, establish_reference=False):
        """Loads every requested channel for one shot and stacks them into [T, C, H, W]."""
        channel_fields = []
        for channel_index, channel in enumerate(self.channels):
            path = self.channel_paths[channel][shot_index]
            is_first_reference = establish_reference and channel_index == 0
            field, time_us, source_time_unit = self._load_windowed_single_channel(path, establish_reference=is_first_reference)

            if is_first_reference:
                self.single_channel_shape = tuple(field.shape)
                self.time_values_us = time_us
                self.time_step_us = float(np.median(np.diff(time_us)))
                self.source_time_unit = source_time_unit

            channel_fields.append(field)

        stacked_field = np.ascontiguousarray(np.stack(channel_fields, axis=1), dtype=np.float32)  # [T, C, H, W]
        return stacked_field, self.time_values_us, self.source_time_unit

    def _validate_field(self, field, shot_index):
        if tuple(field.shape) != self.field_shape:
            raise ValueError(
                f"Inconsistent field shape for shot {int(self.shots[shot_index])}: {field.shape}; "
                f"expected {self.field_shape}"
            )
        if not np.isfinite(field).all():
            raise ValueError(f"Shot {int(self.shots[shot_index])} contains NaN or infinite field values.")

    def _preload(self):
        start_time = time.time()
        fields = []
        num_channels = len(self.channels)
        sums = np.empty((len(self.shots), num_channels), dtype=np.float64)
        squared_sums = np.empty((len(self.shots), num_channels), dtype=np.float64)
        counts = np.empty((len(self.shots), num_channels), dtype=np.int64)

        print(f"Preloading windowed {self.channels} sequences into RAM...")
        for index in range(len(self.shots)):
            field, _, _ = self._load_windowed_multichannel(index)

            self._validate_field(field, index)
            sums[index] = field.sum(axis=(0, 2, 3), dtype=np.float64)
            squared_sums[index] = np.square(field, dtype=np.float64).sum(axis=(0, 2, 3))
            counts[index] = field.shape[0] * field.shape[2] * field.shape[3]
            field.setflags(write=False)
            fields.append(field)

            if (index + 1) % 100 == 0 or index + 1 == len(self.shots):
                print(f"  loaded {index + 1}/{len(self.shots)} shots", flush=True)

        self.fields = fields
        self.shot_sums = sums
        self.shot_squared_sums = squared_sums
        self.shot_counts = counts
        print(f"Preload completed in {(time.time() - start_time) / 60:.1f} minutes.")

    def _scan_shot_statistics(self):
        start_time = time.time()
        num_channels = len(self.channels)
        sums = np.empty((len(self.shots), num_channels), dtype=np.float64)
        squared_sums = np.empty((len(self.shots), num_channels), dtype=np.float64)
        counts = np.empty((len(self.shots), num_channels), dtype=np.int64)

        print("Scanning each windowed shot once to cache normalization statistics...")
        for index in range(len(self.shots)):
            field, _, _ = self._load_windowed_multichannel(index)

            self._validate_field(field, index)
            sums[index] = field.sum(axis=(0, 2, 3), dtype=np.float64)
            squared_sums[index] = np.square(field, dtype=np.float64).sum(axis=(0, 2, 3))
            counts[index] = field.shape[0] * field.shape[2] * field.shape[3]

            if (index + 1) % 100 == 0 or index + 1 == len(self.shots):
                print(f"  scanned {index + 1}/{len(self.shots)} shots", flush=True)

        self.shot_sums = sums
        self.shot_squared_sums = squared_sums
        self.shot_counts = counts
        print(f"Statistics scan completed in {(time.time() - start_time) / 60:.1f} minutes.")

    def get_field(self, index):
        if self.fields is not None:
            return self.fields[index]

        field, _, _ = self._load_windowed_multichannel(index)
        self._validate_field(field, index)
        return field

    def calculate_mean_std(self, indices):
        """Per-channel mean/std (each shape [num_channels]) over the selected shots."""
        indices = np.asarray(indices, dtype=np.int64)
        total_sum = self.shot_sums[indices].sum(axis=0, dtype=np.float64)
        total_squared_sum = self.shot_squared_sums[indices].sum(axis=0, dtype=np.float64)
        total_count = self.shot_counts[indices].sum(axis=0, dtype=np.int64)

        if (total_count == 0).any():
            raise ValueError("The selected training split contains no field values.")

        mean = total_sum / total_count
        variance = total_squared_sum / total_count - mean**2
        std = np.sqrt(np.clip(variance, 0.0, None))

        if not np.isfinite(std).all() or (std == 0.0).any():
            raise ValueError(f"Invalid training standard deviation: {std}")

        return mean.astype(np.float32), std.astype(np.float32)


class IndexedFieldDataset(Dataset):
    """View of selected shot indices with fold-specific per-channel normalization."""

    def __init__(self, store, indices, mean, std):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        global_index = int(self.indices[local_index])
        raw_field = self.store.get_field(global_index)  # [T, C, H, W]
        field = (raw_field - self.mean) / self.std
        field = np.ascontiguousarray(field, dtype=np.float32)
        frames = torch.from_numpy(field)  # [T, C, H, W], channels already present
        label = torch.tensor(self.store.labels[global_index], dtype=torch.float32)
        shot = torch.tensor(self.store.shots[global_index], dtype=torch.int64)
        return frames, label, shot

def make_loader(dataset, batch_size, shuffle, device, seed):

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


#=============================================================================
# CNN frame encoders

def group_norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class FrameEncoder(nn.Module):
    """Maps every frame to an embedding using a plain or residual CNN."""

    def __init__(self, image_height, image_width, channels, embed_dim, backbone, in_channels=1):
        super().__init__()
        c1, c2, c3 = channels
        self.backbone_name = backbone
        self.in_channels = int(in_channels)

        if backbone == "plain":
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, c1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        elif backbone == "residual":
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, c1, kernel_size=3, padding=1, bias=False),
                group_norm(c1),
                nn.ReLU(inplace=True),
                ResidualBlock(c1, c1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualBlock(c1, c2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualBlock(c2, c3),
            )
        else:
            raise ValueError(f"Unknown CNN backbone: {backbone}")

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, image_height, image_width)
            flattened_features = int(self.features(dummy).numel())

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_features, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames):
        batch_size, num_frames, channels, height, width = frames.shape
        x = frames.reshape(batch_size * num_frames, channels, height, width)
        x = self.projection(self.features(x))
        return x.reshape(batch_size, num_frames, -1)


# ============================================================================
# Gated-attention MIL

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def make_classifier(input_dim, dropout, multiplier):
    hidden_dim = max(32, int(round(input_dim * multiplier)))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class GatedAttentionMIL(nn.Module):
    def __init__(self, frame_encoder, embed_dim, dropout, classifier_multiplier):
        super().__init__()
        self.frame_encoder = frame_encoder
        attention_dim = max(32, embed_dim // 2)
        self.attention_v = nn.Linear(embed_dim, attention_dim)
        self.attention_u = nn.Linear(embed_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = make_classifier(embed_dim, dropout, classifier_multiplier)
        initialize_weights(self)

    def forward(self, frames):
        embeddings = self.frame_encoder(frames)
        gated = torch.tanh(self.attention_v(embeddings)) * torch.sigmoid(self.attention_u(embeddings))
        weights = torch.softmax(self.attention_w(gated).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(config, field_shape):
    if config.get("architecture") != "gated_attention":
        raise ValueError("This search supports only the gated_attention architecture.")

    _, num_channels, image_height, image_width = field_shape
    channels = tuple(int(value) for value in config["cnn_channels"])
    embed_dim = int(config["embed_dim"])

    frame_encoder = FrameEncoder(
        image_height=image_height,
        image_width=image_width,
        channels=channels,
        embed_dim=embed_dim,
        backbone=config["cnn_backbone"],
        in_channels=num_channels,
    )

    return GatedAttentionMIL(
        frame_encoder=frame_encoder,
        embed_dim=embed_dim,
        dropout=float(config["dropout"]),
        classifier_multiplier=float(config["classifier_multiplier"]),
    )


# ============================================================================
# Training and evaluation

def find_best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)

    if len(thresholds) == 0:
        return 0.0, 0.5 # 0 f1, 0.5 threshold

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    valid = (thresholds >= 0.05) & (thresholds <= 0.95)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        best_index = valid_indices[int(np.argmax(f1_scores[valid]))]
    else:
        best_index = int(np.argmax(f1_scores))

    return float(f1_scores[best_index]), float(thresholds[best_index])


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)

    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except ValueError:
        roc_auc = float("nan")

    return {
        "pr_auc": average_precision_score(labels, probabilities),
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "threshold": float(threshold),
    }


def evaluate(loader, model, criterion, device, threshold=0.5, optimize_threshold=False, return_outputs=False):
    model.eval()
    total_loss = 0.0
    labels_batches = []
    probability_batches = []
    shot_batches = []

    with torch.no_grad():
        for frames, labels, shots in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(frames)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item()
            labels_batches.append(labels.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
            shot_batches.append(shots.numpy())

    labels = np.concatenate(labels_batches)
    probabilities = np.concatenate(probability_batches)
    shots = np.concatenate(shot_batches)

    if optimize_threshold:
        _, threshold = find_best_f1_threshold(labels, probabilities)

    metrics = calculate_metrics(labels, probabilities, threshold)
    metrics["loss"] = total_loss / len(loader)

    if return_outputs:
        return metrics, labels, probabilities, shots
    return metrics


def build_criterion(config, train_labels, device):
    if config["loss_type"] == "bce":
        return nn.BCEWithLogitsLoss()

    if config["loss_type"] == "weighted_bce":
        positive_count = int(np.sum(train_labels == 1))
        negative_count = int(np.sum(train_labels == 0))
        if positive_count == 0:
            raise ValueError("Training split contains no positive labels.")
        pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss type: {config['loss_type']}")


def build_scheduler(config, optimizer, max_epochs):
    scheduler_name = config["scheduler"]
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(loader, model, criterion, optimizer, scaler, device, amp_enabled):
    model.train()
    total_loss = 0.0

    for frames, labels, _ in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with autocast_context:
            logits = model(frames)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Encountered non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_one_split(store, train_indices, val_indices, mean, std, config, device, max_epochs, patience,
                    min_epochs, primary_metric, seed, history_path=None, keep_best_state=False):
    
    set_seed(seed)

    train_dataset = IndexedFieldDataset(store, train_indices, mean, std)
    val_dataset = IndexedFieldDataset(store, val_indices, mean, std)

    train_loader = make_loader(train_dataset, batch_size=config["batch_size"], shuffle=True, device=device, seed=seed)
    val_loader = make_loader(val_dataset, batch_size=config["batch_size"], shuffle=False, device=device, seed=seed+1)

    model = build_model(config, store.field_shape).to(device)
    train_labels = store.labels[np.asarray(train_indices, dtype=np.int64)]
    criterion = build_criterion(config, train_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = build_scheduler(config, optimizer, max_epochs)

    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    best_score = -math.inf
    best_metrics = None
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scaler, device, amp_enabled)
        val_metrics = evaluate(val_loader, model, criterion, device, optimize_threshold=True)
        current_score = metric_value(val_metrics, primary_metric)
        learning_rate = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": learning_rate,
        }
        history.append(history_row)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"PR_AUC={val_metrics['pr_auc']:.4f} "
            f"ROC_AUC={val_metrics['roc_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"threshold={val_metrics['threshold']:.4f}",
            flush=True,
        )

        if current_score > best_score + MIN_IMPROVEMENT:
            best_score = current_score
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            if keep_best_state:
                best_state = state_dict_to_cpu(model)
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if config["scheduler"] == "plateau":
                scheduler.step(current_score)
            else:
                scheduler.step()

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; best {primary_metric} "
                f"was {best_score:.4f} at epoch {best_epoch}.",
                flush=True,
            )
            break

    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)

    result = {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "parameter_count": count_parameters(model),
        "epochs_completed": len(history),
        "duration_minutes": (time.time() - start_time) / 60,
    }

    del train_loader, val_loader, train_dataset, val_dataset, optimizer, scheduler, scaler, criterion
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, best_state


# ============================================================================
# Search plan

# this is a config previosuly found to be quite good after some preliminary experimentation (JY search)
def base_config(**overrides):
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


def anchor_configs():
    return [
        base_config(name="historical_best_f1"),
        base_config(name="historical_residual_small", cnn_backbone="residual", cnn_channels=[8, 16, 32], embed_dim=64, dropout=0.5,
                    learning_rate=3e-4, weight_decay=1e-3, batch_size=4, scheduler="plateau"),
        base_config(name="historical_plain_small", cnn_channels=[8, 16, 32], embed_dim=64, dropout=0.25, 
                    learning_rate=3e-4, weight_decay=1e-5, batch_size=16, scheduler="plateau", classifier_multiplier=0.5),
        base_config(name="historical_best_pr_auc", cnn_channels=[24, 48, 96], embed_dim=256, dropout=0.1, 
                    learning_rate=3e-5, weight_decay=1e-5, batch_size=16, scheduler="plateau", classifier_multiplier=0.5),
    ]


def sample_random_config(rng, random_index):
    return base_config(
        name=f"focused_random_{random_index:03d}",
        cnn_backbone=rng.choices(["plain", "residual"], weights=[0.65, 0.35], k=1)[0],
        cnn_channels=list(
            rng.choices(
                [(8, 16, 32), (16, 32, 64), (24, 48, 96)],
                weights=[0.25, 0.50, 0.25],
                k=1,
            )[0]
        ),
        embed_dim=rng.choices([64, 128, 256], weights=[0.35, 0.45, 0.20], k=1)[0],
        dropout=rng.choices(
            [0.1, 0.25, 0.3, 0.4, 0.5],
            weights=[0.15, 0.2, 0.3, 0.2, 0.15],
            k=1,
        )[0],
        learning_rate=rng.choices([3e-5, 1e-4, 3e-4], weights=[0.2, 0.5, 0.3], k=1)[0],
        weight_decay=rng.choices([1e-5, 1e-4, 1e-3], weights=[0.25, 0.3, 0.45], k=1)[0],
        batch_size=rng.choices([4, 8, 16], weights=[0.2, 0.5, 0.3], k=1)[0],
        loss_type=rng.choices(["bce", "weighted_bce"], weights=[0.8, 0.2], k=1)[0],
        scheduler=rng.choices(
            ["none", "cosine", "plateau"],
            weights=[0.15, 0.35, 0.5],
            k=1,
        )[0],
        classifier_multiplier=rng.choices([0.5, 1.0], weights=[0.35, 0.65], k=1)[0],
    )


def config_signature(config):
    ignored_keys = {"name", "trial_id"}
    filtered = {key: config[key] for key in sorted(config) if key not in ignored_keys}
    return json.dumps(filtered, sort_keys=True)


def create_search_plan(total_trials, seed):
    anchors = anchor_configs()
    configs = anchors[:total_trials]
    signatures = {config_signature(config) for config in configs}
    rng = random.Random(seed)
    random_index = 0

    while len(configs) < total_trials:
        candidate = sample_random_config(rng, random_index)
        random_index += 1
        signature = config_signature(candidate)
        if signature in signatures:
            continue
        signatures.add(signature)
        configs.append(candidate)

    for trial_id, config in enumerate(configs):
        config["trial_id"] = trial_id

    return configs


# ============================================================================
# Cross-validation search

def make_splits(store, folds):
    development_indices, test_indices = train_test_split(store.original_indices, test_size=TEST_FRACTION, random_state=SEED, stratify=store.labels[store.original_indices])

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    cv_splits = []
    development_labels = store.labels[development_indices]

    for train_local, val_local in splitter.split(development_indices, development_labels):
        train_indices = development_indices[train_local]
        val_indices = development_indices[val_local]

        train_shots = set(store.shots[train_indices].tolist())
        val_shots = set(store.shots[val_indices].tolist())
        test_shots = set(store.shots[test_indices].tolist())
        if train_shots.intersection(val_shots) or train_shots.intersection(test_shots):
            raise RuntimeError("Physical Shot leakage detected across data partitions.")

        cv_splits.append((train_indices, val_indices))

    return development_indices, test_indices, cv_splits


def load_or_create_plan(search_dir, total_trials):
    plan_path = search_dir / "search_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f"Loaded existing search plan with {len(plan)} trials.")
        if len(plan) != total_trials:
            print(f"Note: --trials={total_trials} was ignored because the existing plan contains {len(plan)} trials.")
        return plan

    plan = create_search_plan(total_trials, SEED)
    write_json(plan_path, plan)
    return plan


def trial_summary_from_fold_results(config, fold_results, primary_metric):
    metric_names = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "threshold", "loss"]
    summary = {
        "trial_id": config["trial_id"],
        "name": config["name"],
        "status": "ok",
        "folds_completed": len(fold_results),
    }

    for metric_name in metric_names:
        values = [float(result["best_metrics"][metric_name]) for result in fold_results if result["best_metrics"].get(metric_name) is not None]
        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    summary["mean_best_epoch"] = float(np.mean([result["best_epoch"] for result in fold_results]))
    summary["mean_duration_minutes"] = float(np.mean([result["duration_minutes"] for result in fold_results]))
    summary["parameter_count"] = int(fold_results[0]["parameter_count"])
    summary["selection_metric"] = primary_metric
    summary["selection_score"] = summary[f"mean_{primary_metric}"]

    for key, value in config.items():
        if key not in {"trial_id", "name"}:
            summary[key] = value

    return summary


def rebuild_leaderboard(search_dir, plan, fold_count, primary_metric):
    rows = []
    for config in plan:
        trial_dir = search_dir / f"trial_{config['trial_id']:03d}"
        fold_results = []

        for fold_index in range(fold_count):
            result_path = trial_dir / f"fold_{fold_index}" / "result.json"
            if not result_path.exists():
                fold_results = []
                break

            result = json.loads(result_path.read_text())
            if result.get("status") != "ok":
                fold_results = []
                break
            fold_results.append(result)

        if len(fold_results) == fold_count:
            rows.append(trial_summary_from_fold_results(config, fold_results, primary_metric))

    leaderboard = pd.DataFrame(rows)
    if len(leaderboard) == 0:
        return leaderboard

    leaderboard = leaderboard.sort_values(by=[f"mean_{primary_metric}", f"std_{primary_metric}", "mean_pr_auc"], ascending=[False, True, False]).reset_index(drop=True)
    leaderboard.to_csv(search_dir / "leaderboard.csv", index=False)
    return leaderboard


def run_search(store, search_dir, plan, cv_splits, device, max_epochs, patience, min_epochs, primary_metric):
    total_trials = len(plan)
    total_start = time.time()

    for config in plan:
        trial_id = int(config["trial_id"])
        trial_dir = search_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "config.json", config)

        print("\n" + "#"*88)
        print(
            f"Trial {trial_id + 1}/{total_trials}: {config['name']} | "
            f"architecture={config['architecture']} backbone={config['cnn_backbone']} "
            f"channels={config['cnn_channels']} embed_dim={config['embed_dim']} "
            f"dropout={config['dropout']} lr={config['learning_rate']}"
        )
        print("#" * 88, flush=True)

        trial_failed = False
        for fold_index, (train_indices, val_indices) in enumerate(cv_splits):
            fold_dir = trial_dir / f"fold_{fold_index}"
            result_path = fold_dir / "result.json"

            if result_path.exists():
                existing_result = json.loads(result_path.read_text())
                if existing_result.get("status") == "ok":
                    print(f"Fold {fold_index + 1}/{len(cv_splits)} already complete; skipping.")
                    continue

            fold_dir.mkdir(parents=True, exist_ok=True)
            mean, std = store.calculate_mean_std(train_indices)
            fold_seed = SEED + trial_id * 1000 + fold_index

            print(
                f"\nFold {fold_index + 1}/{len(cv_splits)}: "
                f"train={len(train_indices)} shots, "
                f"val={len(val_indices)} shots "
                f"mean={mean.tolist()} std={std.tolist()} (per channel {store.channels})",
                flush=True,
            )

            try:
                result, _ = train_one_split(store=store, train_indices=train_indices, val_indices=val_indices, mean=mean, std=std, 
                                            config=config, device=device, max_epochs=max_epochs, patience=patience, min_epochs=min_epochs, primary_metric=primary_metric, 
                                            seed=fold_seed, history_path=fold_dir / "history.csv", keep_best_state=False, )
                result["status"] = "ok"
                result["mean"] = mean.tolist()
                result["std"] = std.tolist()
                write_json(result_path, result)

                print(
                    f"Fold complete: best_epoch={result['best_epoch']} "
                    f"{primary_metric}={result['best_metrics'][primary_metric]:.4f} "
                    f"PR_AUC={result['best_metrics']['pr_auc']:.4f} "
                    f"duration={result['duration_minutes']:.1f} min",
                    flush=True,
                )
            except Exception as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                error_result = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(result_path, error_result)
                print(f"Fold failed: {type(error).__name__}: {error}", flush=True)
                trial_failed = True
                break

        leaderboard = rebuild_leaderboard(search_dir, plan, len(cv_splits), primary_metric)

        if len(leaderboard) > 0:
            columns = ["trial_id", "name", f"mean_{primary_metric}", f"std_{primary_metric}", "mean_pr_auc", "std_pr_auc", "architecture", "embed_dim", "dropout", "learning_rate"]
            print("\nCurrent top configurations:")
            print(leaderboard[columns].head(5).to_string(index=False))

        if trial_failed:
            print("Continuing to the next configuration.")

    print(f"\nSearch duration: {(time.time() - total_start) / 3600:.2f} hours")
    return rebuild_leaderboard(search_dir, plan, len(cv_splits), primary_metric)


#=============================================================================
# Final fit and untouched test evaluation

def fit_best_and_test(store, development_indices, test_indices, best_config, search_dir, device, primary_metric):
    final_metrics_path = search_dir / "final_test_metrics.json"
    if final_metrics_path.exists():
        print("\nFinal test metrics already exist; final fit is not being repeated.")
        print(final_metrics_path.read_text())
        return

    final_train_indices, final_val_indices = train_test_split(development_indices, test_size=FINAL_VAL_FRACTION_OF_DEVELOPMENT, 
                                                              random_state=SEED + 999, stratify=store.labels[development_indices])
    mean, std = store.calculate_mean_std(final_train_indices)

    print("\n" + "=" * 88)
    print("Final fit of the best cross-validated configuration")
    print("=" * 88)
    print(json.dumps(json_ready(best_config), indent=2, sort_keys=True))
    print(
        f"Final split sizes: train={len(final_train_indices)} shots, "
        f"validation={len(final_val_indices)} shots, test={len(test_indices)} shots"
    )
    print(f"Final training mean={mean.tolist()}, std={std.tolist()} (per channel {store.channels})")

    result, best_state = train_one_split(
        store=store,
        train_indices=final_train_indices,
        val_indices=final_val_indices,
        mean=mean,
        std=std,
        config=best_config,
        device=device,
        max_epochs=FINAL_MAX_EPOCHS,
        patience=FINAL_EARLY_STOPPING_PATIENCE,
        min_epochs=DEFAULT_MIN_EPOCHS,
        primary_metric=primary_metric,
        seed=SEED + 100_000,
        history_path=search_dir / "final_history.csv",
        keep_best_state=True,
    )

    if best_state is None:
        raise RuntimeError("Final training did not produce a checkpoint.")

    model = build_model(best_config, store.field_shape).to(device)
    model.load_state_dict(best_state)

    final_train_labels = store.labels[np.asarray(final_train_indices, dtype=np.int64)]
    criterion = build_criterion(best_config, final_train_labels, device)
    test_dataset = IndexedFieldDataset(store, test_indices, mean, std)
    test_loader = make_loader(test_dataset, batch_size=best_config["batch_size"], shuffle=False, device=device, seed=SEED + 100_001)

    validation_threshold = result["best_metrics"]["threshold"]
    test_metrics, labels, probabilities, shots = evaluate(test_loader, model, criterion, device, threshold=validation_threshold, optimize_threshold=False, return_outputs=True)

    predictions = (probabilities >= validation_threshold).astype(np.int64)
    prediction_frame = pd.DataFrame(
        {
            "Shot": shots,
            "label": labels.astype(np.int64),
            "probability": probabilities,
            "prediction": predictions,
        }
    ).sort_values("Shot")
    prediction_frame.to_csv(search_dir / "final_test_predictions.csv", index=False)

    # Separate file listing only the misclassified test shots and the nature of the
    # misclassification, to help identify inherently flawed/unsuitable shots.
    misclassified_frame = prediction_frame.loc[prediction_frame["label"] != prediction_frame["prediction"]].copy()
    misclassified_frame["misclassification"] = np.where(
        misclassified_frame["label"] == 0, "false_positive", "false_negative"
    )
    misclassified_path = search_dir / "misclassified_test_shots.csv"
    misclassified_frame.to_csv(misclassified_path, index=False)
    print(
        f"\nMisclassified test shots: {len(misclassified_frame)} "
        f"({int((misclassified_frame['misclassification'] == 'false_positive').sum())} false positives, "
        f"{int((misclassified_frame['misclassification'] == 'false_negative').sum())} false negatives) "
        f"written to {misclassified_path}"
    )

    checkpoint = {
        "model_state_dict": best_state,
        "config": best_config,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "validation_threshold": float(validation_threshold),
        "best_validation_epoch": result["best_epoch"],
        "best_validation_metrics": result["best_metrics"],
        "test_metrics": test_metrics,
        "manifest_path": str(store.manifest_path),
        "channels": store.channels,
        "field_shape": store.field_shape,
        "time_key": TIME_KEY,
        "time_window_us": [TIME_START_US, TIME_END_US],
        "time_step_us": store.time_step_us,
        "requested_time_start_us": TIME_START_US,
        "requested_time_end_us": TIME_END_US,
        "requested_sampling_step_us": TIME_STEP_US,
        "requested_num_frames": NUM_TIME_FRAMES,
        "time_values_us": store.time_values_us,
        "primary_metric": primary_metric,
        "final_train_shots": store.shots[final_train_indices],
        "final_train_sample_count": len(final_train_indices),
        "final_validation_shots": store.shots[final_val_indices],
        "test_shots": store.shots[test_indices],
    }
    torch.save(checkpoint, search_dir / "best_hypersearch_model.pth")

    final_summary = {
        "best_config": best_config,
        "best_validation_epoch": result["best_epoch"],
        "best_validation_metrics": result["best_metrics"],
        "test_metrics": test_metrics,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "channels": store.channels,
        "time_window_us": [TIME_START_US, TIME_END_US],
        "time_step_us": store.time_step_us,
        "requested_time_start_us": TIME_START_US,
        "requested_time_end_us": TIME_END_US,
        "requested_sampling_step_us": TIME_STEP_US,
        "requested_num_frames": NUM_TIME_FRAMES,
        "checkpoint": str(search_dir / "best_hypersearch_model.pth"),
    }
    write_json(final_metrics_path, final_summary)

    print("\nUntouched test results:")
    print(
        f"loss={test_metrics['loss']:.4f}, "
        f"PR AUC={test_metrics['pr_auc']:.4f}, "
        f"ROC AUC={test_metrics['roc_auc']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, "
        f"threshold={test_metrics['threshold']:.4f}"
    )

    del test_loader, test_dataset, criterion
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================================
# Command line and main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--primary-metric", choices=["f1", "pr_auc"], default=PRIMARY_METRIC)
    parser.add_argument("--channels", type=str, default=DEFAULT_CHANNELS,
                        help="Which field(s) to feed the CNN as stacked input channels: any combination of "
                             "BR, BZ, BY, JY separated by '-' (e.g. BR-BZ, BR-BZ-JY, BR-BZ-BY-JY), "
                             "a single field (e.g. BR), or BOTH as an alias for BR-BZ.")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    channels = resolve_channels(args.channels)
    output_dir = resolve_output_dir(channels, args.primary_metric)
    search_dir = output_dir / "hyperparameter_search"

    if args.smoke_test:
        args.trials = 4
        args.folds = 2
        args.epochs = 2
        args.patience = 2
        args.min_epochs = 1
        search_dir = output_dir / "hyperparameter_search_smoke"

    if args.reset and search_dir.exists():
        print(f"Removing existing search directory: {search_dir}")
        shutil.rmtree(search_dir)

    search_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Channels: {channels}")
    print(
        f"Budget: trials={args.trials}, folds={args.folds}, max_epochs={args.epochs}, "
        f"patience={args.patience}, primary_metric={args.primary_metric}, "
        f"num_workers={NUM_WORKERS}"
    )
    print(f"Results directory: {search_dir}")

    store = FieldDataStore(LABEL_MANIFEST_PATH, channels=channels, preload=PRELOAD_DATA and not args.no_preload)
    development_indices, test_indices, cv_splits = make_splits(store, args.folds)

    split_summary = {
        "development_size": len(development_indices),
        "test_size": len(test_indices),
        "total_sample_count": len(store.shots),
        "physical_shot_count": store.num_physical_shots,
        "development_positives": int(store.labels[development_indices].sum()),
        "test_positives": int(store.labels[test_indices].sum()),
        "test_shots": store.shots[test_indices],
        "cv_folds": args.folds,
        "seed": SEED,
        "channels": channels,
        "manifest_path": str(LABEL_MANIFEST_PATH),
        "time_key": TIME_KEY,
        "time_window_us": [TIME_START_US, TIME_END_US],
        "time_step_us": store.time_step_us,
        "requested_time_start_us": TIME_START_US,
        "requested_time_end_us": TIME_END_US,
        "requested_sampling_step_us": TIME_STEP_US,
        "requested_num_frames": NUM_TIME_FRAMES,
        "num_frames": store.field_shape[0],
        "num_workers": NUM_WORKERS,
    }
    write_json(search_dir / "split_summary.json", split_summary)

    print(
        f"Development/test split: {len(development_indices)}/{len(test_indices)} shots; "
        f"test positives={int(store.labels[test_indices].sum())}. "
        "The test set will not be used during the search."
    )

    plan = load_or_create_plan(search_dir, args.trials)
    leaderboard = run_search(store=store, search_dir=search_dir, plan=plan, cv_splits=cv_splits, device=device, max_epochs=args.epochs, 
                             patience=args.patience, min_epochs=args.min_epochs, primary_metric=args.primary_metric)

    if len(leaderboard) == 0:
        raise RuntimeError("No configuration completed all cross-validation folds.")

    best_trial_id = int(leaderboard.iloc[0]["trial_id"])
    best_config = next(config for config in plan if int(config["trial_id"]) == best_trial_id)
    write_json(search_dir / "best_config.json", best_config)

    print("\nBest cross-validated configuration:")
    print(json.dumps(json_ready(best_config), indent=2, sort_keys=True))
    print(
        f"Mean CV {args.primary_metric}: "
        f"{leaderboard.iloc[0][f'mean_{args.primary_metric}']:.4f} ± "
        f"{leaderboard.iloc[0][f'std_{args.primary_metric}']:.4f}"
    )
    print(
        f"Mean CV PR AUC: {leaderboard.iloc[0]['mean_pr_auc']:.4f} ± "
        f"{leaderboard.iloc[0]['std_pr_auc']:.4f}"
    )

    if args.smoke_test:
        print(
            "\nSmoke test completed. Run without --smoke-test for the overnight search. "
            "The smoke-test directory is separate from the overnight results."
        )
        return

    fit_best_and_test(store=store, development_indices=development_indices, test_indices=test_indices, best_config=best_config, 
                      search_dir=search_dir, device=device, primary_metric=args.primary_metric)


if __name__ == "__main__":
    main()