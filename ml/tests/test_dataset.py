"""Unit tests for dataset loading and ImageNet class mapping."""

import torch
from robustlens.eval.dataset import (
    IMAGENETTE_TO_IMAGENET,
    ImagenetteDataset,
    get_imagenette_loader,
)
from robustlens.eval.engine import get_imagenet_classes


def test_imagenette_class_alignment():
    """Verify that Imagenette class IDs map to the correct semantic ImageNet-1K indices."""
    classes = get_imagenet_classes()

    # Critical verified ground-truth mappings
    expected_mappings = {
        0: "tench",
        1: "English springer",
        2: "cassette player",
        3: "chain saw",
        4: "church",
        5: "French horn",
        6: "garbage truck",
        7: "gas pump",
        8: "golf ball",
        9: "parachute",
    }

    for local_id, expected_name in expected_mappings.items():
        imagenet_idx = IMAGENETTE_TO_IMAGENET[local_id]
        class_name = classes[imagenet_idx].lower()
        assert expected_name.lower() in class_name, (
            f"Mapping mismatch for local ID {local_id}: "
            f"expected '{expected_name}', mapped to index {imagenet_idx} ('{classes[imagenet_idx]}')"
        )


def test_dataset_reproducible_loader():
    """Verify DataLoader deterministic subset sampling with seed."""
    loader1 = get_imagenette_loader(batch_size=4, subset_size=8, seed=42)
    loader2 = get_imagenette_loader(batch_size=4, subset_size=8, seed=42)

    labels1 = [int(labels[0]) for _, labels, _ in loader1]
    labels2 = [int(labels[0]) for _, labels, _ in loader2]

    assert labels1 == labels2, "DataLoader with identical seed produced differing subsets"


def test_imagenette_split_integrity():
    """Verify exact source, counts, boundary transition, and split integrity."""
    from collections import Counter
    from robustlens.eval.dataset import (
        TOTAL_MIRROR_ROWS,
        TRAIN_COUNT,
        VAL_COUNT,
        TRAIN_CLASS_COUNTS,
        VAL_CLASS_COUNTS,
        DATASET_PROVENANCE,
    )
    from datasets import load_dataset

    # Verify provenance contract
    assert TOTAL_MIRROR_ROWS == 13394
    assert TRAIN_COUNT == 9469
    assert VAL_COUNT == 3925
    assert TRAIN_COUNT + VAL_COUNT == TOTAL_MIRROR_ROWS
    assert sum(VAL_CLASS_COUNTS.values()) == VAL_COUNT
    assert sum(TRAIN_CLASS_COUNTS.values()) == TRAIN_COUNT
    assert DATASET_PROVENANCE["split_integrity_verified"] is True

    # Load underlying parquet dataset
    full_ds = load_dataset(DATASET_PROVENANCE["source_mirror"], split="train")
    assert len(full_ds) == TOTAL_MIRROR_ROWS

    # Verify class boundary transition from train (0..9468) to val (9469..13393)
    train_end_label = full_ds[TRAIN_COUNT - 1]["label"]
    val_start_label = full_ds[TRAIN_COUNT]["label"]
    assert train_end_label == 9, f"Expected train split to end at class 9, got {train_end_label}"
    assert val_start_label == 0, f"Expected val split to begin at class 0, got {val_start_label}"

    # Verify wrapper slices and exact per-class counts
    val_ds = ImagenetteDataset(split="validation")
    assert len(val_ds) == VAL_COUNT
    val_counts = Counter(val_ds.dataset["label"])
    for class_id, expected_count in VAL_CLASS_COUNTS.items():
        assert val_counts[class_id] == expected_count, (
            f"Validation class {class_id} count mismatch: expected {expected_count}, got {val_counts[class_id]}"
        )

    train_ds = ImagenetteDataset(split="train")
    assert len(train_ds) == TRAIN_COUNT
    train_counts = Counter(train_ds.dataset["label"])
    for class_id, expected_count in TRAIN_CLASS_COUNTS.items():
        assert train_counts[class_id] == expected_count, (
            f"Train class {class_id} count mismatch: expected {expected_count}, got {train_counts[class_id]}"
        )

