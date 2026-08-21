#!/usr/bin/env python3
"""
This was intended and succeeded as a check of the gated attention mil model.
It was struggling on FLARE data, so the positive results here indicated that the data 
(pre)processing might be the best next thing to improve, as opposed to the model itself 
via a far-reaching hyperparameter optimisation.

References to hyperparameter searches are not references to the paper's main search at
MIL-Plasmoid-Classification/hyperparameter-search. They are references to preliminary
searches done much earlier in the project.


The script:
  1. downloads MNIST with torchvision
  2. creates MIL bags
  3. trains the two fixed configurations
  4. selects each checkpoint and decision threshold using validation F1 only
  5. reports untouched-test loss, PR AUC, ROC AUC, accuracy, precision,
     recall, F1, threshold, and confusion-matrix counts
  6. saves histories, predictions, checkpoints, and a comparison JSON/CSV

MNIST-Bags setup:
  - target instance: digit 9
  - a bag is positive iff it contains at least one 9
  - bag lengths are drawn from Gaussian with mean 10 and std 2
  - train and validation use disjoint subsets of the official MNIST training images
  - test bags use only the official MNIST test images
  - 500 train bags, 250 validation bags, 1000 test bags

Originally run on RunPod so no accompanying Stellar Slurm script.
"""

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, 
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import MNIST


#=============================================================================
# Constants

SEED = 42
TARGET_DIGIT = 9
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MEAN_BAG_LENGTH = 10.0
BAG_LENGTH_STD = 2.0
DEFAULT_TRAIN_BAGS = 500
DEFAULT_VAL_BAGS = 250
DEFAULT_TEST_BAGS = 1000
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 7
MIN_EPOCHS = 6
NUM_WORKERS = 4
GRADIENT_CLIP_NORM = 2.0
MIN_IMPROVEMENT = 1e-4
USE_AMP = False

DEFAULT_DATA_ROOT = Path("/workspace/datasets")
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/PPPL/Advanced-Plasmoid-Classification/"
    "Image-Data/TimeMIL-output/MNIST_BAGS_best_models"
)

# Best gated-attention configuration by mean CV F1 in the JY search
GATED_CONFIG = {
    "name": "best_gated_attention",
    "architecture": "gated_attention",
    "cnn_backbone": "plain",
    "cnn_channels": [16, 32, 64],
    "embed_dim": 128,
    "dropout": 0.30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 8,
    "loss_type": "bce",
    "scheduler": "cosine",
    "noise_std": 0.0,
    "scale_std": 0.0,
    "classifier_multiplier": 1.0,
    "timemil_layers": 2,
    "heads": 8,
    "num_landmarks": 64,
}

# Best TimeMIL configuration by mean CV F1 in the JY search (random_028).
TIMEMIL_CONFIG = {
    "name": "best_timemil",
    "architecture": "timemil",
    "cnn_backbone": "residual",
    "cnn_channels": [16, 32, 64],
    "embed_dim": 128,
    "dropout": 0.25,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "batch_size": 8,
    "loss_type": "weighted_bce",
    "scheduler": "none",
    "noise_std": 0.0,
    "scale_std": 0.07,
    "classifier_multiplier": 0.5,
    "timemil_layers": 1,
    "heads": 4,
    "num_landmarks": 64,
}


#=============================================================================
# Nystrom attention import

NYSTROM_IMPLEMENTATION = None

try:
    from models.nystrom_attention import NystromAttention # type: ignore

    NYSTROM_IMPLEMENTATION = "models.nystrom_attention"
except ImportError:
    try:
        from nystrom_attention import NystromAttention # type: ignore

        NYSTROM_IMPLEMENTATION = "nystrom_attention package"
    except ImportError:
        class NystromAttention(nn.Module):
            """Exact-attention fallback with the same constructor/forward interface.

            MNIST bags contain roughly ten tokens, so exact self-attention is cheap.
            On RunPod inside the TimeMIL repository, the project Nyström module is
            expected to be imported instead of this fallback.
            """

            def __init__(
                self,
                dim,
                dim_head,
                heads,
                num_landmarks,
                pinv_iterations=6,
                residual=True,
                dropout=0.0,
            ):
                super().__init__()
                del dim_head, num_landmarks, pinv_iterations
                self.attention = nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.residual = residual
                self.residual_conv = (
                    nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
                    if residual
                    else None
                )

            def forward(self, x, mask=None):
                key_padding_mask = None
                if mask is not None:
                    key_padding_mask = ~mask.bool()
                output, _ = self.attention(
                    x,
                    x,
                    x,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                if self.residual_conv is not None:
                    output = output + self.residual_conv(x.transpose(1, 2)).transpose(1, 2)
                return output

        NYSTROM_IMPLEMENTATION = "built-in exact-attention fallback"


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True))
    temporary_path.replace(path)


def write_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def state_dict_to_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


#=============================================================================
# MNIST-Bags data

class MNISTBags(Dataset):

    def __init__(
        self,
        images,
        digit_labels,
        source_indices,
        num_bags,
        seed,
        target_digit=TARGET_DIGIT,
        mean_bag_length=MEAN_BAG_LENGTH,
        bag_length_std=BAG_LENGTH_STD,
    ):
        if num_bags < 2:
            raise ValueError("num_bags must be at least 2")

        self.images = images
        self.digit_labels = digit_labels
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.num_bags = int(num_bags)
        self.target_digit = int(target_digit)
        self.mean_bag_length = float(mean_bag_length)
        self.bag_length_std = float(bag_length_std)
        self.rng = np.random.default_rng(seed)

        source_targets = self.digit_labels[self.source_indices].cpu().numpy()
        self.non_target_indices = self.source_indices[source_targets != self.target_digit]
        if len(self.non_target_indices) == 0:
            raise ValueError("The source pool contains no non-target MNIST images")
        if not np.any(source_targets == self.target_digit):
            raise ValueError("The source pool contains no target MNIST images")

        self.bag_indices = []
        self.bag_labels = []
        self.instance_labels = []
        self._form_balanced_bags()
        self.lengths = np.asarray([len(indices) for indices in self.bag_indices], dtype=np.int64)

    def _draw_bag_length(self):
        length = int(np.rint(self.rng.normal(self.mean_bag_length, self.bag_length_std)))
        return max(1, length)

    def _sample_positive_bag(self, bag_length):
        # Rejection sampling reproduces random MNIST sampling conditioned on
        # the bag containing at least one target digit
        while True:
            indices = self.rng.choice(self.source_indices, size=bag_length, replace=True)
            labels = self.digit_labels[torch.as_tensor(indices, dtype=torch.long)].cpu().numpy()
            if np.any(labels == self.target_digit):
                return indices.astype(np.int64), labels

    def _sample_negative_bag(self, bag_length):
        indices = self.rng.choice(self.non_target_indices, size=bag_length, replace=True)
        labels = self.digit_labels[torch.as_tensor(indices, dtype=torch.long)].cpu().numpy()
        return indices.astype(np.int64), labels

    def _form_balanced_bags(self):
        positive_count = self.num_bags // 2
        negative_count = self.num_bags - positive_count
        requested_labels = np.asarray([1] * positive_count + [0] * negative_count, dtype=np.int64)
        self.rng.shuffle(requested_labels)

        for bag_label in requested_labels:
            bag_length = self._draw_bag_length()
            if bag_label == 1:
                indices, digit_labels = self._sample_positive_bag(bag_length)
            else:
                indices, digit_labels = self._sample_negative_bag(bag_length)

            instance_labels = (digit_labels == self.target_digit).astype(np.int64)
            observed_bag_label = int(instance_labels.max())
            if observed_bag_label != int(bag_label):
                raise RuntimeError("Constructed bag label does not match its instance labels")

            # Random instance order
            permutation = self.rng.permutation(bag_length)
            self.bag_indices.append(indices[permutation])
            self.instance_labels.append(instance_labels[permutation])
            self.bag_labels.append(int(bag_label))

    def __len__(self):
        return self.num_bags

    def __getitem__(self, index):
        indices = torch.as_tensor(self.bag_indices[index], dtype=torch.long)
        bag = self.images[indices].to(torch.float32).div_(255.0)
        bag = bag.sub_(MNIST_MEAN).div_(MNIST_STD).unsqueeze(1)
        label = torch.tensor(self.bag_labels[index], dtype=torch.float32)
        instance_labels = torch.as_tensor(self.instance_labels[index], dtype=torch.int64)
        return bag, label, instance_labels, torch.tensor(index, dtype=torch.int64)


class LengthBucketBatchSampler(Sampler):
    """Batches bags of equal length"""

    def __init__(self, lengths, batch_size, shuffle, seed):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = defaultdict(list)
        for index, length in enumerate(self.lengths.tolist()):
            self.groups[int(length)].append(index)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        group_items = list(self.groups.items())
        if self.shuffle:
            rng.shuffle(group_items)

        for _, indices in group_items:
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def make_loader(dataset, batch_size, shuffle, seed):
    sampler = LengthBucketBatchSampler(dataset.lengths, batch_size, shuffle, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_datasets(data_root, train_bags, val_bags, test_bags):
    print(f"Downloading/loading MNIST under: {data_root}")
    train_source = MNIST(root=data_root, train=True, download=True)
    test_source = MNIST(root=data_root, train=False, download=True)

    all_train_indices = np.arange(len(train_source.targets), dtype=np.int64)
    train_source_indices, val_source_indices = train_test_split(
        all_train_indices,
        test_size=10_000,
        random_state=SEED,
        stratify=train_source.targets.numpy(),
    )
    test_source_indices = np.arange(len(test_source.targets), dtype=np.int64)

    train_dataset = MNISTBags(
        train_source.data,
        train_source.targets,
        train_source_indices,
        num_bags=train_bags,
        seed=SEED + 10,
    )
    val_dataset = MNISTBags(
        train_source.data,
        train_source.targets,
        val_source_indices,
        num_bags=val_bags,
        seed=SEED + 20,
    )
    test_dataset = MNISTBags(
        test_source.data,
        test_source.targets,
        test_source_indices,
        num_bags=test_bags,
        seed=SEED + 30,
    )
    return train_dataset, val_dataset, test_dataset


def dataset_summary(dataset):
    labels = np.asarray(dataset.bag_labels, dtype=np.int64)
    return {
        "bags": len(dataset),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "mean_bag_length": float(dataset.lengths.mean()),
        "std_bag_length": float(dataset.lengths.std(ddof=1)),
        "min_bag_length": int(dataset.lengths.min()),
        "max_bag_length": int(dataset.lengths.max()),
    }


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
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class FrameEncoder(nn.Module):
    def __init__(self, image_height, image_width, channels, embed_dim, backbone):
        super().__init__()
        c1, c2, c3 = channels

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


#=============================================================================
# TimeMIL and gated-attention MIL models

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


class TransLayer(nn.Module):
    def __init__(self, dim, dropout, heads, num_landmarks):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by heads={heads}")
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


def mexican_hat_wavelet(size, scale, shift):
    coordinates = torch.linspace(
        -(size[1] - 1) // 2,
        (size[1] - 1) // 2,
        size[1],
        device=scale.device,
        dtype=scale.dtype,
    )
    coordinates = coordinates.reshape(1, -1).repeat(size[0], 1)
    coordinates = coordinates - shift

    sign = torch.where(scale >= 0, torch.ones_like(scale), -torch.ones_like(scale))
    safe_scale = sign * scale.abs().clamp_min(1e-3)
    constant = 2 / (3**0.5 * torch.pi**0.25)
    wavelet = constant * (1 - (coordinates / safe_scale) ** 2)
    wavelet = wavelet * torch.exp(-(coordinates / safe_scale) ** 2 / 2)
    return wavelet / safe_scale.abs() ** 0.5


class WaveletEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3):
        cls_token, feature_tokens = x[:, 0], x[:, 1:]
        x = feature_tokens.transpose(1, 2)
        dim = x.shape[1]

        kernels = []
        for wave in (wave1, wave2, wave3):
            scale, shift = wave[0, :], wave[1, :]
            kernels.append(mexican_hat_wavelet((dim, 19), scale, shift))

        positional = sum(
            nn.functional.conv1d(x, kernel.unsqueeze(1), groups=dim, padding="same")
            for kernel in kernels
        )
        x = x.transpose(1, 2)
        x = x + self.projection(positional.transpose(1, 2))
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


def make_wavelet_parameters(dim, initial_values):
    wave = initial_values(2, dim, 1)
    wave[0] = torch.ones(dim, 1) + torch.randn(dim, 1)
    return nn.Parameter(wave)


class TimeMILBlock(nn.Module):
    def __init__(self, dim, dropout, heads, num_landmarks):
        super().__init__()
        self.wave1 = make_wavelet_parameters(dim, torch.randn)
        self.wave2 = make_wavelet_parameters(dim, torch.zeros)
        self.wave3 = make_wavelet_parameters(dim, torch.zeros)
        self.position = WaveletEncoding(dim)
        self.transformer = TransLayer(dim, dropout, heads, num_landmarks)

    def forward(self, x):
        x = self.position(x, self.wave1, self.wave2, self.wave3)
        return self.transformer(x)


def make_classifier(input_dim, dropout, multiplier):
    hidden_dim = max(32, int(round(input_dim * multiplier)))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class ImageTimeMIL(nn.Module):
    def __init__(
        self,
        frame_encoder,
        embed_dim,
        dropout,
        layers,
        heads,
        num_landmarks,
        classifier_multiplier,
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TimeMILBlock(embed_dim, dropout, heads, num_landmarks) for _ in range(layers)]
        )
        self.classifier = make_classifier(embed_dim, dropout, classifier_multiplier)
        initialize_weights(self)

    def forward(self, frames):
        x = self.frame_encoder(frames)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x[:, 0]).squeeze(-1)


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
        gated = torch.tanh(self.attention_v(embeddings))
        gated = gated * torch.sigmoid(self.attention_u(embeddings))
        weights = torch.softmax(self.attention_w(gated).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(config, max_bag_length):
    channels = tuple(int(value) for value in config["cnn_channels"])
    embed_dim = int(config["embed_dim"])
    frame_encoder = FrameEncoder(
        image_height=28,
        image_width=28,
        channels=channels,
        embed_dim=embed_dim,
        backbone=config["cnn_backbone"],
    )

    if config["architecture"] == "gated_attention":
        return GatedAttentionMIL(
            frame_encoder=frame_encoder,
            embed_dim=embed_dim,
            dropout=float(config["dropout"]),
            classifier_multiplier=float(config["classifier_multiplier"]),
        )

    if config["architecture"] == "timemil":
        num_landmarks = min(int(config["num_landmarks"]), int(max_bag_length) + 1)
        return ImageTimeMIL(
            frame_encoder=frame_encoder,
            embed_dim=embed_dim,
            dropout=float(config["dropout"]),
            layers=int(config["timemil_layers"]),
            heads=int(config["heads"]),
            num_landmarks=num_landmarks,
            classifier_multiplier=float(config["classifier_multiplier"]),
        )

    raise ValueError(f"Unknown architecture: {config['architecture']}")


#=============================================================================
# Training and eval

def find_best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.0, 0.5

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
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except ValueError:
        roc_auc = float("nan")

    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(loader, model, criterion, device, threshold=0.5, optimize_threshold=False):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    label_batches = []
    probability_batches = []
    bag_id_batches = []

    with torch.no_grad():
        for frames, labels, _, bag_ids in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(frames)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            label_batches.append(labels.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
            bag_id_batches.append(bag_ids.numpy())

    labels = np.concatenate(label_batches)
    probabilities = np.concatenate(probability_batches)
    bag_ids = np.concatenate(bag_id_batches)
    if optimize_threshold:
        _, threshold = find_best_f1_threshold(labels, probabilities)

    metrics = calculate_metrics(labels, probabilities, threshold)
    metrics["loss"] = float(total_loss / total_examples)
    return metrics, labels, probabilities, bag_ids


def build_criterion(config, train_labels, device):
    if config["loss_type"] == "bce":
        return nn.BCEWithLogitsLoss()
    if config["loss_type"] == "weighted_bce":
        positive_count = int(np.sum(train_labels == 1))
        negative_count = int(np.sum(train_labels == 0))
        if positive_count == 0:
            raise ValueError("Training data contain no positive bags")
        pos_weight = torch.tensor(
            negative_count / positive_count,
            dtype=torch.float32,
            device=device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss type: {config['loss_type']}")


def build_scheduler(config, optimizer, max_epochs):
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if config["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    raise ValueError(f"Unknown scheduler: {config['scheduler']}")


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def augment_frames(frames, noise_std, scale_std):
    if scale_std > 0.0:
        scale = torch.randn(
            frames.shape[0], 1, 1, 1, 1,
            device=frames.device,
            dtype=frames.dtype,
        )
        scale = 1.0 + scale * scale_std
        frames = frames * scale
    if noise_std > 0.0:
        frames = frames + torch.randn_like(frames) * noise_std
    return frames


def train_epoch(loader, model, criterion, optimizer, scaler, device, config, amp_enabled, epoch):
    model.train()
    loader.batch_sampler.set_epoch(epoch)
    total_loss = 0.0
    total_examples = 0

    for frames, labels, _, _ in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frames = augment_frames(
            frames,
            noise_std=float(config["noise_std"]),
            scale_std=float(config["scale_std"]),
        )
        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
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

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return float(total_loss / total_examples)


def train_and_test_model(
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir,
    device,
    max_epochs,
    patience,
):
    model_name = config["name"]
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / "config.json", config)

    set_seed(SEED)
    train_loader = make_loader(train_dataset, config["batch_size"], True, SEED)
    val_loader = make_loader(val_dataset, config["batch_size"], False, SEED + 1)
    test_loader = make_loader(test_dataset, config["batch_size"], False, SEED + 2)

    max_bag_length = max(
        int(train_dataset.lengths.max()),
        int(val_dataset.lengths.max()),
        int(test_dataset.lengths.max()),
    )
    model = build_model(config, max_bag_length).to(device)
    train_labels = np.asarray(train_dataset.bag_labels, dtype=np.int64)
    criterion = build_criterion(config, train_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(config, optimizer, max_epochs)
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    best_score = -math.inf
    best_epoch = None
    best_metrics = None
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    print("\n" + "=" * 92)
    print(f"Training {model_name}: {json.dumps(config, sort_keys=True)}")
    print(f"Parameters: {count_parameters(model):,}")
    print("=" * 92)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            device,
            config,
            amp_enabled,
            epoch,
        )
        val_metrics, _, _, _ = evaluate(
            val_loader,
            model,
            criterion,
            device,
            optimize_threshold=True,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        current_score = float(val_metrics["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": learning_rate,
        }
        history.append(row)
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
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = state_dict_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if config["scheduler"] == "plateau":
                scheduler.step(current_score)
            else:
                scheduler.step()

        if epoch >= MIN_EPOCHS and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; best validation F1 was "
                f"{best_score:.4f} at epoch {best_epoch}."
            )
            break

    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError(f"{model_name} did not produce a valid checkpoint")

    write_rows_csv(model_dir / "history.csv", history)
    model.load_state_dict(best_state)
    validation_threshold = float(best_metrics["threshold"])
    test_metrics, labels, probabilities, bag_ids = evaluate(
        test_loader,
        model,
        criterion,
        device,
        threshold=validation_threshold,
        optimize_threshold=False,
    )
    predictions = (probabilities >= validation_threshold).astype(np.int64)

    prediction_rows = [
        {
            "bag_id": int(bag_id),
            "label": int(label),
            "probability": float(probability),
            "prediction": int(prediction),
            "bag_length": int(test_dataset.lengths[int(bag_id)]),
        }
        for bag_id, label, probability, prediction in zip(
            bag_ids,
            labels,
            probabilities,
            predictions,
        )
    ]
    prediction_rows.sort(key=lambda row: row["bag_id"])
    write_rows_csv(model_dir / "test_predictions.csv", prediction_rows)

    checkpoint = {
        "model_state_dict": best_state,
        "config": config,
        "best_validation_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "target_digit": TARGET_DIGIT,
        "mean_bag_length": MEAN_BAG_LENGTH,
        "bag_length_std": BAG_LENGTH_STD,
    }
    torch.save(checkpoint, model_dir / "best_model.pth")

    result = {
        "model": model_name,
        "architecture": config["architecture"],
        "parameter_count": count_parameters(model),
        "best_validation_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "duration_minutes": (time.time() - start_time) / 60,
        "config": config,
    }
    write_json(model_dir / "result.json", result)

    print(f"\nUntouched test results for {model_name}:")
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
    print(
        f"confusion matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )

    del train_loader, val_loader, test_loader, criterion, optimizer, scheduler, scaler
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


#=============================================================================
# Command line and main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-bags", type=int, default=DEFAULT_TRAIN_BAGS)
    parser.add_argument("--val-bags", type=int, default=DEFAULT_VAL_BAGS)
    parser.add_argument("--test-bags", type=int, default=DEFAULT_TEST_BAGS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("train_bags", "val_bags", "test_bags", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Nyström attention implementation: {NYSTROM_IMPLEMENTATION}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    print(f"Output directory: {args.output_dir}")

    train_dataset, val_dataset, test_dataset = make_datasets(
        args.data_root,
        args.train_bags,
        args.val_bags,
        args.test_bags,
    )
    summaries = {
        "train": dataset_summary(train_dataset),
        "validation": dataset_summary(val_dataset),
        "test": dataset_summary(test_dataset),
    }
    print("\nDataset summary:")
    print(json.dumps(summaries, indent=2))

    experiment_metadata = {
        "seed": SEED,
        "target_digit": TARGET_DIGIT,
        "mnist_normalization_mean": MNIST_MEAN,
        "mnist_normalization_std": MNIST_STD,
        "requested_mean_bag_length": MEAN_BAG_LENGTH,
        "requested_bag_length_std": BAG_LENGTH_STD,
        "num_workers": NUM_WORKERS,
        "nystrom_implementation": NYSTROM_IMPLEMENTATION,
        "dataset_summary": summaries,
        "models": [GATED_CONFIG, TIMEMIL_CONFIG],
    }
    write_json(args.output_dir / "experiment_metadata.json", experiment_metadata)

    results = []
    for config in (GATED_CONFIG, TIMEMIL_CONFIG):
        results.append(
            train_and_test_model(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_dir=args.output_dir,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
        )

    comparison_rows = []
    for result in results:
        metrics = result["test_metrics"]
        comparison_rows.append(
            {
                "model": result["model"],
                "architecture": result["architecture"],
                "parameter_count": result["parameter_count"],
                "best_validation_epoch": result["best_validation_epoch"],
                "test_loss": metrics["loss"],
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "duration_minutes": result["duration_minutes"],
            }
        )

    write_rows_csv(args.output_dir / "model_comparison.csv", comparison_rows)
    write_json(args.output_dir / "model_comparison.json", comparison_rows)

    print("\n" + "=" * 92)
    print("Final untouched-test comparison")
    print("=" * 92)
    header = (
        f"{'model':<24} {'PR_AUC':>8} {'ROC_AUC':>8} {'accuracy':>9} "
        f"{'precision':>9} {'recall':>8} {'F1':>8} {'threshold':>10}"
    )
    print(header)
    for row in comparison_rows:
        print(
            f"{row['model']:<24} {row['test_pr_auc']:>8.4f} "
            f"{row['test_roc_auc']:>8.4f} {row['test_accuracy']:>9.4f} "
            f"{row['test_precision']:>9.4f} {row['test_recall']:>8.4f} "
            f"{row['test_f1']:>8.4f} {row['threshold']:>10.4f}"
        )
    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
