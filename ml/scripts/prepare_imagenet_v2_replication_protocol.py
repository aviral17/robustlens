"""Prepare ImageNet-V2 External Replication Protocol.

Mandate 5:
- 10 matching WordNet synsets mapped to ImageNet-1K indices
- Available sample sizes per class (10 images/class) and dataset variant:
  * MatchedFrequency (N=100)
  * Threshold0.7 (N=100)
  * TopImages (N=100)
- Frozen corruption parameters: 18 conditions from calibrated 6-way common support
- Zero recalibration on ImageNet-V2
- Interpretation: Independently collected external replication, not universal out-of-distribution validation
"""

from __future__ import annotations

import json
from pathlib import Path

from robustlens.eval.dataset import IMAGENETTE_TO_IMAGENET

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "imagenet_v2_replication_protocol.json"

SYNSET_NAMES = {
    0: ("n01440764", "tench"),
    1: ("n02102040", "English springer"),
    2: ("n02979186", "cassette player"),
    3: ("n03000684", "chain saw"),
    4: ("n03028079", "church"),
    5: ("n03394916", "French horn"),
    6: ("n03417042", "garbage truck"),
    7: ("n03425413", "gas pump"),
    8: ("n03445777", "golf ball"),
    9: ("n03888257", "parachute"),
}

VARIANTS = {
    "MatchedFrequency": {
        "description": "Matches ImageNet validation set frequency distribution across candidate selection pools",
        "sample_size_per_class": 10,
        "total_classes": 10,
        "total_images": 100,
        "archive_url": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
    },
    "Threshold0.7": {
        "description": "Selects images where MTurk worker consensus >= 0.70",
        "sample_size_per_class": 10,
        "total_classes": 10,
        "total_images": 100,
        "archive_url": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold-0.7.tar.gz",
    },
    "TopImages": {
        "description": "Selects top 10 images with highest worker agreement per class",
        "sample_size_per_class": 10,
        "total_classes": 10,
        "total_images": 100,
        "archive_url": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz",
    },
}


def main():
    print("=" * 80)
    print("MANDATE 5: IMAGENET-V2 EXTERNAL REPLICATION PROTOCOL")
    print("=" * 80)

    classes_table = []
    for local_id, (wnid, name) in SYNSET_NAMES.items():
        in1k_target = IMAGENETTE_TO_IMAGENET[local_id]
        classes_table.append({
            "imagenette_class_id": local_id,
            "wordnet_id": wnid,
            "class_name": name,
            "imagenet_1k_index": in1k_target,
            "matched_frequency_n": 10,
            "threshold_07_n": 10,
            "top_images_n": 10,
            "total_samples_all_variants": 30,
        })
        print(f"Class {local_id}: {name:<18} (WNID: {wnid}, ImageNet-1K: {in1k_target:>3}) | 10 per variant")

    protocol = {
        "replication_study_name": "ImageNet-V2 External Counterfactual Replication",
        "citation": "Recht et al., 'Do ImageNet Classifiers Generalize to ImageNet?' ICML 2019",
        "objective": "Verify transfer of structural fingerprint explanatory power to independently collected natural photographs without parameter retuning.",
        "scientific_interpretation": (
            "Treated as an independently collected external replication of natural images, "
            "evaluating generalization beyond the original Flickr collection pipeline of ILSVRC2012. "
            "It is NOT claimed to be universal out-of-distribution validation, as images still depict "
            "the same 10 ImageNet visual concepts."
        ),
        "calibration_status": "Strictly frozen. Zero recalibration or parameter retuning on ImageNet-V2. All 18 corruption settings applied identically.",
        "sample_sizes_by_variant": VARIANTS,
        "classes": classes_table,
        "total_evaluations_plan": {
            "per_variant_images": 100,
            "conditions_per_image": 19,  # 1 clean + 18 corrupted
            "models": 4,
            "forward_passes_per_variant": 100 * 19 * 4,  # 7,600
            "total_forward_passes_all_3_variants": 300 * 19 * 4,  # 22,800
        },
        "statistical_evaluation": (
            "Fit frozen Model 3 (Spline) and Model 5 (Spline + Fingerprints) using weights derived "
            "from the confirmatory evaluation or run cross-validation on ImageNet-V2 to assess "
            "whether structural fingerprints consistently yield positive Delta R2_OOS."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(protocol, f, indent=2)

    print(f"\nSaved ImageNet-V2 replication protocol to {OUTPUT_JSON}")
    print(f"Total available sample size: 100 images per variant (10 images/class x 10 classes).")
    print(f"Total across all 3 variants: 300 images (22,800 total forward passes).")


if __name__ == "__main__":
    main()
