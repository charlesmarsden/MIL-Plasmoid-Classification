#!/usr/bin/env python3
"""
CRNN (CNN + LSTM) broad hyperparameter search (JY)

Adapted from ../../TimeMIL/hyperparameter-search-codes/
jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented.py

The search explores:
- CNN architecture (plain vs residual_
- CNN channel widths
- embedding dimension
- LSTM hidden dimension, number of layers, and uni-/bi-directional
- dropout rate
- learning rate and weight decay
- batch size
- weighted or unweighted BCE
- scheduler (none/cosine/plateau)
- level of noise/scale augmentation

Portable path overrides (all default to the original Flux hardcoded paths if
unset, mirroring crnn.py's CRNN_* env vars): CRNN_SEARCH_MANIFEST_PATH,
CRNN_SEARCH_OUTPUT_DIR, CRNN_SEARCH_PATH_REMAP_OLD, CRNN_SEARCH_PATH_REMAP_NEW,
CRNN_SEARCH_NUM_WORKERS.
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


#=========================================================================================
# Paths and search budget (it is set by allowed time, not a prespecified number of trials)

FIELD_NAME = "JY"
TIME_KEY = "T"
TIME_START_US = 261.0
TIME_STEP_US = 1.0
NUM_TIME_FRAMES = 80
TIME_END_US = TIME_START_US + TIME_STEP_US * (NUM_TIME_FRAMES - 1)

# Portable path overrides, all default to the original Flux hardcoded paths if unset
# (mirrors ../crnn.py's CRNN_* env vars, using a CRNN_SEARCH_ prefix so both scripts'
# env vars can be set independently).
MANIFEST_PATH = Path(os.environ.get(
    "CRNN_SEARCH_MANIFEST_PATH",
    "/u/cmarsden/docs/PPPL/"
    "Advanced-Plasmoid-Classification/Image-Data/"
    "Field-Images/JY_cropped_05us_augmented/"
    "augmented_manifest.csv"
))

OUTPUT_DIR = Path(os.environ.get(
    "CRNN_SEARCH_OUTPUT_DIR",
    "/u/cmarsden/docs/PPPL/"
    "Advanced-Plasmoid-Classification/Image-Data/"
    "CRNN-output/JY_261us_80frames_1us_crnn_augmented"
))

SEARCH_DIR = OUTPUT_DIR / "hyperparameter_search"

PATH_REMAP_OLD = os.environ.get("CRNN_SEARCH_PATH_REMAP_OLD")
PATH_REMAP_NEW = os.environ.get("CRNN_SEARCH_PATH_REMAP_NEW")

SEED = 42
TEST_FRACTION = 0.15
FINAL_VAL_FRACTION_OF_DEVELOPMENT = 0.17647058823529413  # Gives 70/15/15 overall.

# Effectively unbounded trial
DEFAULT_TRIALS = 100_000
DEFAULT_TIME_BUDGET_HOURS = None # None = no time limit (run until --trials is exhausted)
DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_EPOCHS = 6
FINAL_MAX_EPOCHS = 40
FINAL_EARLY_STOPPING_PATIENCE = 7

PRIMARY_METRIC = "f1"
NUM_WORKERS = int(os.environ.get("CRNN_SEARCH_NUM_WORKERS", 0))
PRELOAD_DATA = True
USE_AMP = False
GRADIENT_CLIP_NORM = 2.0
MIN_IMPROVEMENT = 1e-4


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

# same as jy_gated_attention_hyperparameter_search_261us_80frames_1us_augmented.py
class JYDataStore:
    """Owns the JY manifest and exposes only the requested physical-time sampling."""

    def __init__(self, manifest_path, preload=True):
        self.manifest_path = Path(manifest_path)
        self.manifest = pd.read_csv(self.manifest_path)

        # Portable path remap (e.g. Flux -> Stellar), same mechanism as ../crnn.py.
        if PATH_REMAP_OLD and PATH_REMAP_NEW:
            self.manifest["path"] = self.manifest["path"].str.replace(PATH_REMAP_OLD, PATH_REMAP_NEW, regex=False)

        required_columns = {"path", "Shot", "is_plasmoid"}
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

        self.paths = [Path(path) for path in self.manifest["path"]]
        missing_paths = [path for path in self.paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"{len(missing_paths)} manifest files are missing. First missing file: {missing_paths[0]}")

        self.labels = self.manifest["label"].astype(np.int64).to_numpy()
        self.shots = self.manifest["Shot"].astype(np.int64).to_numpy()

        if "is_augmented" in self.manifest:
            augmentation_flags = pd.to_numeric(self.manifest["is_augmented"], errors="raise").to_numpy()
            if not set(np.unique(augmentation_flags).tolist()).issubset({0, 1}):
                raise ValueError("Manifest is_augmented values must be 0 or 1")
            self.is_augmented = augmentation_flags.astype(bool)
        else:
            self.is_augmented = np.zeros(len(self.manifest), dtype=bool)

        unique_labels = set(np.unique(self.labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Expected binary labels 0/1, found {sorted(unique_labels)}")

        manifest_groups = self.manifest.assign(is_augmented=self.is_augmented).groupby("Shot")
        inconsistent_labels = manifest_groups["label"].nunique()
        if (inconsistent_labels != 1).any():
            bad_shot = int(inconsistent_labels[inconsistent_labels != 1].index[0])
            raise ValueError(f"Manifest has inconsistent labels for Shot {bad_shot}")

        original_counts = manifest_groups["is_augmented"].apply(lambda values: (~values).sum())
        if (original_counts != 1).any():
            bad_shot = int(original_counts[original_counts != 1].index[0])
            raise ValueError(
                f"Manifest must contain exactly one original row for Shot {bad_shot}; "
                f"found {int(original_counts.loc[bad_shot])}"
            )

        self.original_indices = np.flatnonzero(~self.is_augmented)
        self.num_physical_shots = len(self.original_indices)

        self.fields = None
        self.shot_sums = None
        self.shot_squared_sums = None
        self.shot_counts = None
        self.field_shape = None
        self.time_values_us = None
        self.time_step_us = None
        self.source_time_unit = None

        first_field, first_time_us, source_time_unit = self._load_windowed_field(self.paths[0], establish_reference=True)
        self.field_shape = tuple(first_field.shape)
        self.time_values_us = first_time_us
        self.time_step_us = float(np.median(np.diff(first_time_us)))
        self.source_time_unit = source_time_unit

        estimated_bytes = first_field.nbytes * len(self.paths)
        available_bytes = available_memory_bytes()

        print(
            f"Dataset: {len(self.paths)} samples from {self.num_physical_shots} physical shots, "
            f"augmented={int(self.is_augmented.sum())}, "
            f"positive shots={int(self.labels[self.original_indices].sum())}, "
            f"field={FIELD_NAME}, field shape={self.field_shape}"
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

    def _load_windowed_field(self, path, establish_reference=False):
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

        if tuple(selected_field.shape) != self.field_shape:
            raise ValueError(
                f"Inconsistent windowed field shape in {path}: {selected_field.shape}; "
                f"expected {self.field_shape}"
            )
        if not np.allclose(selected_time_us, self.time_values_us, rtol=1e-7, atol=max(1e-7, abs(self.time_step_us) * 1e-5),):
            raise ValueError(f"Selected time grid in {path} differs from the first shot")

        return selected_field, selected_time_us, source_time_unit

    def _validate_field(self, field, path):
        if tuple(field.shape) != self.field_shape:
            raise ValueError(f"Inconsistent field shape in {path}: {field.shape}; expected {self.field_shape}")
        if not np.isfinite(field).all():
            raise ValueError(f"{path} contains NaN or infinite field values.")

    def _preload(self):
        start_time = time.time()
        fields = []
        sums = np.empty(len(self.paths), dtype=np.float64)
        squared_sums = np.empty(len(self.paths), dtype=np.float64)
        counts = np.empty(len(self.paths), dtype=np.int64)

        print(f"Preloading windowed {FIELD_NAME} sequences into RAM...")
        for index, path in enumerate(self.paths):
            field, _, _ = self._load_windowed_field(path)

            self._validate_field(field, path)
            sums[index] = field.sum(dtype=np.float64)
            squared_sums[index] = np.square(field, dtype=np.float64).sum(dtype=np.float64)
            counts[index] = field.size
            field.setflags(write=False)
            fields.append(field)

            if (index + 1) % 100 == 0 or index + 1 == len(self.paths):
                print(f"  loaded {index + 1}/{len(self.paths)} shots", flush=True)

        self.fields = fields
        self.shot_sums = sums
        self.shot_squared_sums = squared_sums
        self.shot_counts = counts
        print(f"Preload completed in {(time.time() - start_time) / 60:.1f} minutes.")

    def _scan_shot_statistics(self):
        start_time = time.time()
        sums = np.empty(len(self.paths), dtype=np.float64)
        squared_sums = np.empty(len(self.paths), dtype=np.float64)
        counts = np.empty(len(self.paths), dtype=np.int64)

        print("Scanning each windowed shot once to cache normalization statistics...")
        for index, path in enumerate(self.paths):
            field, _, _ = self._load_windowed_field(path)

            self._validate_field(field, path)
            sums[index] = field.sum(dtype=np.float64)
            squared_sums[index] = np.square(field, dtype=np.float64).sum(dtype=np.float64)
            counts[index] = field.size

            if (index + 1) % 100 == 0 or index + 1 == len(self.paths):
                print(f"  scanned {index + 1}/{len(self.paths)} shots", flush=True)

        self.shot_sums = sums
        self.shot_squared_sums = squared_sums
        self.shot_counts = counts
        print(f"Statistics scan completed in {(time.time() - start_time) / 60:.1f} minutes.")

    def get_field(self, index):
        if self.fields is not None:
            return self.fields[index]

        field, _, _ = self._load_windowed_field(self.paths[index])
        self._validate_field(field, self.paths[index])
        return field

    def calculate_mean_std(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        total_sum = self.shot_sums[indices].sum(dtype=np.float64)
        total_squared_sum = self.shot_squared_sums[indices].sum(dtype=np.float64)
        total_count = self.shot_counts[indices].sum(dtype=np.int64)

        if total_count == 0:
            raise ValueError("The selected training split contains no field values.")

        mean = total_sum / total_count
        variance = total_squared_sum / total_count - mean**2
        std = math.sqrt(max(float(variance), 0.0))

        if not math.isfinite(std) or std == 0.0:
            raise ValueError(f"Invalid training standard deviation: {std}")

        return np.float32(mean), np.float32(std)

    def expand_training_indices(self, original_indices):
        original_indices = np.asarray(original_indices, dtype=np.int64)
        if self.is_augmented[original_indices].any():
            raise ValueError("Training expansion must start from original manifest rows.")
        training_shots = self.shots[original_indices]
        return np.flatnonzero(np.isin(self.shots, training_shots))


class IndexedJYDataset(Dataset):
    """
    Per-split indexed viewer of shots from JYDataStore. 
    Applies fold's normalization and augmentation
    """

    def __init__(self, store, indices, mean, std, augment=False, noise_std=0.0, scale_std=0.0):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        self.augment = augment
        self.noise_std = float(noise_std)
        self.scale_std = float(scale_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        global_index = int(self.indices[local_index])
        raw_field = self.store.get_field(global_index)
        field = (raw_field - self.mean) / self.std

        if self.augment:
            if self.scale_std > 0.0:
                scale = np.float32(np.random.normal(loc=1.0, scale=self.scale_std))
                field = field * scale
            if self.noise_std > 0.0:
                noise = np.random.normal(0.0, self.noise_std, size=field.shape).astype(np.float32)
                field = field + noise

        field = np.ascontiguousarray(field, dtype=np.float32)
        frames = torch.from_numpy(field).unsqueeze(1)
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
# CNN frame encoders (same as the gated-attention search)

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
    """Maps every frame to a vector embedding using a plain or residual CNN"""

    def __init__(self, image_height, image_width, channels, embed_dim, backbone):
        super().__init__()
        c1, c2, c3 = channels
        self.backbone_name = backbone

        if backbone == "plain":
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1),
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
                nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
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
            dummy = torch.zeros(1, 1, image_height, image_width)
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


#============================================================================
# CRNN (LSTM sequence pooling) - replaces the gated-attention search's GatedAttentionMIL

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


class ImageCRNN(nn.Module):
    """CNN frame encoder + LSTM sequence pooling (see ../crnn.py for the base version).

    The LSTM reads the time-ordered per-frame CNN embeddings and its final hidden
    state (both directions concatenated if bidirectional) is the bag representation,
    fed into the same classifier-head shape the gated-attention search uses.
    """

    def __init__(self, frame_encoder, embed_dim, hidden_dim, rnn_layers, bidirectional, dropout, classifier_multiplier):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.rnn_directions = 2 if bidirectional else 1
        classifier_input_dim = hidden_dim * self.rnn_directions
        self.classifier = make_classifier(classifier_input_dim, dropout, classifier_multiplier)
        initialize_weights(self)

    def forward(self, frames):
        embeddings = self.frame_encoder(frames) # [B, T, 1, H, W] -> [B, T, embed_dim]
        _, (hidden, _) = self.rnn(embeddings) # hidden: [num_layers * num_directions, B, hidden_dim]

        if self.rnn_directions == 2:
            # concatonate final layer's forward and backward final hidden states
            bag_representation = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            bag_representation = hidden[-1] # final hidden state summarizes the whole sequence

        return self.classifier(bag_representation).squeeze(-1)


def build_model(config, field_shape):
    if config.get("architecture") != "crnn":
        raise ValueError("This search supports only the crnn architecture.")

    _, image_height, image_width = field_shape
    channels = tuple(int(value) for value in config["cnn_channels"])
    embed_dim = int(config["embed_dim"])

    frame_encoder = FrameEncoder(
        image_height=image_height,
        image_width=image_width,
        channels=channels,
        embed_dim=embed_dim,
        backbone=config["cnn_backbone"],
    )

    return ImageCRNN(
        frame_encoder=frame_encoder,
        embed_dim=embed_dim,
        hidden_dim=int(config["hidden_dim"]),
        rnn_layers=int(config["rnn_layers"]),
        bidirectional=bool(config["bidirectional"]),
        dropout=float(config["dropout"]),
        classifier_multiplier=float(config["classifier_multiplier"]),
    )


#============================================================================
# Training and eval (same as the gated-attention search)

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

    train_dataset = IndexedJYDataset(store, train_indices, mean, std, augment=True, noise_std=config["noise_std"], scale_std=config["scale_std"])
    val_dataset = IndexedJYDataset(store, val_indices, mean, std, augment=False)

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


#============================================================================
# Search plan

def base_config(**overrides):
    config = {
        "architecture": "crnn",
        "cnn_backbone": "plain",
        "cnn_channels": [16, 32, 64],
        "embed_dim": 128,
        "hidden_dim": 128,
        "rnn_layers": 1,
        "bidirectional": False,
        "dropout": 0.3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "loss_type": "bce",
        "scheduler": "cosine",
        "noise_std": 0.0,
        "scale_std": 0.0,
        "classifier_multiplier": 1.0,
    }
    config.update(overrides)
    return config


def anchor_configs():
    return [
        base_config(name="default_crnn"),
        base_config(
            name="first_manual_run_replica", embed_dim=256, hidden_dim=256, rnn_layers=1, bidirectional=False,
            dropout=0.2, learning_rate=1e-4, weight_decay=1e-4, batch_size=4, scheduler="none",
        ),
        base_config(
            name="strong_regularization", dropout=0.5, weight_decay=1e-2, batch_size=16, scheduler="plateau",
            noise_std=0.01, scale_std=0.03, classifier_multiplier=0.5,
        ),
        base_config(
            name="bidirectional_small", cnn_backbone="residual", cnn_channels=[8, 16, 32], embed_dim=64,
            hidden_dim=64, bidirectional=True, dropout=0.4, learning_rate=3e-4, weight_decay=1e-3, batch_size=8,
            scheduler="plateau", noise_std=0.01,
        ),
        base_config(
            name="deep_lstm", rnn_layers=2, hidden_dim=128, dropout=0.4, learning_rate=3e-5, weight_decay=1e-3,
            batch_size=16, scheduler="plateau", scale_std=0.03,
        ),
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
        hidden_dim=rng.choices([64, 128, 256], weights=[0.35, 0.45, 0.20], k=1)[0],
        rnn_layers=rng.choices([1, 2], weights=[0.7, 0.3], k=1)[0],
        bidirectional=rng.choices([False, True], weights=[0.65, 0.35], k=1)[0],
        dropout=rng.choices(
            [0.1, 0.25, 0.3, 0.4, 0.5],
            weights=[0.15, 0.2, 0.3, 0.2, 0.15],
            k=1,
        )[0],
        learning_rate=rng.choices([3e-5, 1e-4, 3e-4], weights=[0.2, 0.5, 0.3], k=1)[0],
        weight_decay=rng.choices([1e-5, 1e-4, 1e-3, 1e-2], weights=[0.15, 0.25, 0.35, 0.25], k=1)[0],
        batch_size=rng.choices([4, 8, 16], weights=[0.2, 0.5, 0.3], k=1)[0],
        loss_type=rng.choices(["bce", "weighted_bce"], weights=[0.8, 0.2], k=1)[0],
        scheduler=rng.choices(
            ["none", "cosine", "plateau"],
            weights=[0.15, 0.35, 0.5],
            k=1,
        )[0],
        noise_std=rng.choices([0.0, 0.01], weights=[0.7, 0.3], k=1)[0],
        scale_std=rng.choices([0.0, 0.03], weights=[0.7, 0.3], k=1)[0],
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


#============================================================================
# Cross-validation search

def make_splits(store, folds):
    development_indices, test_indices = train_test_split(store.original_indices, test_size=TEST_FRACTION, random_state=SEED, stratify=store.labels[store.original_indices])

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    cv_splits = []
    development_labels = store.labels[development_indices]

    for train_local, val_local in splitter.split(development_indices, development_labels):
        train_original_indices = development_indices[train_local]
        val_indices = development_indices[val_local]
        train_indices = store.expand_training_indices(train_original_indices)

        train_shots = set(store.shots[train_indices].tolist())
        val_shots = set(store.shots[val_indices].tolist())
        test_shots = set(store.shots[test_indices].tolist())
        if train_shots.intersection(val_shots) or train_shots.intersection(test_shots):
            raise RuntimeError("Physical Shot leakage detected across data partitions.")
        if store.is_augmented[val_indices].any() or store.is_augmented[test_indices].any():
            raise RuntimeError("Validation and test partitions must contain original rows only.")

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


def run_search(store, search_dir, plan, cv_splits, device, max_epochs, patience, min_epochs, primary_metric, time_budget_hours=None):
    total_trials = len(plan)
    total_start = time.time()

    def time_budget_exceeded():
        return time_budget_hours is not None and (time.time() - total_start) / 3600 >= time_budget_hours

    for config in plan:
        if time_budget_exceeded():
            print(
                f"\nTime budget of {time_budget_hours} hours reached; stopping search "
                f"before trial {int(config['trial_id']) + 1}/{total_trials}. Rerun with the same "
                "search directory to resume from here."
            )
            break

        trial_id = int(config["trial_id"])
        trial_dir = search_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "config.json", config)

        print("\n" + "#"*88)
        print(
            f"Trial {trial_id + 1}/{total_trials}: {config['name']} | "
            f"architecture={config['architecture']} backbone={config['cnn_backbone']} "
            f"channels={config['cnn_channels']} embed_dim={config['embed_dim']} "
            f"hidden_dim={config['hidden_dim']} rnn_layers={config['rnn_layers']} "
            f"bidirectional={config['bidirectional']} "
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
                f"train={len(train_indices)} samples from "
                f"{len(np.unique(store.shots[train_indices]))} shots, "
                f"val={len(val_indices)} original shots "
                f"mean={mean:.7g} std={std:.7g}",
                flush=True,
            )

            try:
                result, _ = train_one_split(store=store, train_indices=train_indices, val_indices=val_indices, mean=mean, std=std,
                                            config=config, device=device, max_epochs=max_epochs, patience=patience, min_epochs=min_epochs, primary_metric=primary_metric,
                                            seed=fold_seed, history_path=fold_dir / "history.csv", keep_best_state=False, )
                result["status"] = "ok"
                result["mean"] = float(mean)
                result["std"] = float(std)
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

            if time_budget_exceeded():
                print(
                    f"\nTime budget of {time_budget_hours} hours reached mid-trial "
                    f"(trial {trial_id}, fold {fold_index + 1}/{len(cv_splits)} just completed). "
                    "Stopping search; the remaining folds of this trial will resume next run."
                )
                trial_failed = True  # not a real failure - just stops the fold loop cleanly
                break

        leaderboard = rebuild_leaderboard(search_dir, plan, len(cv_splits), primary_metric)

        if len(leaderboard) > 0:
            columns = ["trial_id", "name", f"mean_{primary_metric}", f"std_{primary_metric}", "mean_pr_auc", "std_pr_auc", "architecture", "embed_dim", "hidden_dim", "dropout", "learning_rate"]
            print("\nCurrent top configurations:")
            print(leaderboard[columns].head(5).to_string(index=False))

        if trial_failed:
            print("Continuing to the next configuration.")

        if time_budget_exceeded():
            break

    print(f"\nSearch duration: {(time.time() - total_start) / 3600:.2f} hours")
    return rebuild_leaderboard(search_dir, plan, len(cv_splits), primary_metric)


#=============================================================================
# Final fit / untouched test evaluation

def fit_best_and_test(store, development_indices, test_indices, best_config, search_dir, device, primary_metric):
    final_metrics_path = search_dir / "final_test_metrics.json"
    if final_metrics_path.exists():
        print("\nFinal test metrics already exist; final fit is not being repeated.")
        print(final_metrics_path.read_text())
        return

    final_train_original_indices, final_val_indices = train_test_split(development_indices, test_size=FINAL_VAL_FRACTION_OF_DEVELOPMENT,
                                                                       random_state=SEED + 999, stratify=store.labels[development_indices])
    final_train_indices = store.expand_training_indices(final_train_original_indices)
    mean, std = store.calculate_mean_std(final_train_indices)

    print("\n" + "=" * 88)
    print("Final fit of the best cross-validated configuration")
    print("=" * 88)
    print(json.dumps(json_ready(best_config), indent=2, sort_keys=True))
    print(
        f"Final split sizes: train={len(final_train_indices)} samples from "
        f"{len(final_train_original_indices)} shots, validation={len(final_val_indices)} "
        f"original shots, test={len(test_indices)} original shots"
    )
    print(f"Final training mean={mean:.8g}, std={std:.8g}")

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
    test_dataset = IndexedJYDataset(store, test_indices, mean, std, augment=False)
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

    checkpoint = {
        "model_state_dict": best_state,
        "config": best_config,
        "normalization_mean": float(mean),
        "normalization_std": float(std),
        "validation_threshold": float(validation_threshold),
        "best_validation_epoch": result["best_epoch"],
        "best_validation_metrics": result["best_metrics"],
        "test_metrics": test_metrics,
        "manifest_path": str(store.manifest_path),
        "field_name": FIELD_NAME,
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
        "final_train_shots": store.shots[final_train_original_indices],
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
        "normalization_mean": float(mean),
        "normalization_std": float(std),
        "field_name": FIELD_NAME,
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


#============================================================================
# Command line and main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--time-budget-hours", type=float, default=DEFAULT_TIME_BUDGET_HOURS)
    parser.add_argument("--folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--primary-metric", choices=["f1", "pr_auc"], default=PRIMARY_METRIC)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.trials = 4
        args.folds = 2
        args.epochs = 2
        args.patience = 2
        args.min_epochs = 1
        search_dir = OUTPUT_DIR / "hyperparameter_search_smoke"
    else:
        search_dir = SEARCH_DIR

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
    print(
        f"Budget: trials={args.trials}, time_budget_hours={args.time_budget_hours}, "
        f"folds={args.folds}, max_epochs={args.epochs}, "
        f"patience={args.patience}, primary_metric={args.primary_metric}, "
        f"num_workers={NUM_WORKERS}"
    )
    print(f"Results directory: {search_dir}")

    store = JYDataStore(MANIFEST_PATH, preload=PRELOAD_DATA and not args.no_preload)
    development_indices, test_indices, cv_splits = make_splits(store, args.folds)

    split_summary = {
        "development_size": len(development_indices),
        "test_size": len(test_indices),
        "total_sample_count": len(store.paths),
        "physical_shot_count": store.num_physical_shots,
        "augmented_sample_count": int(store.is_augmented.sum()),
        "development_positives": int(store.labels[development_indices].sum()),
        "test_positives": int(store.labels[test_indices].sum()),
        "test_shots": store.shots[test_indices],
        "cv_folds": args.folds,
        "seed": SEED,
        "field_name": FIELD_NAME,
        "manifest_path": str(MANIFEST_PATH),
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
                             patience=args.patience, min_epochs=args.min_epochs, primary_metric=args.primary_metric,
                             time_budget_hours=args.time_budget_hours)

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
