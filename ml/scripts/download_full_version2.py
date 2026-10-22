"""Targeted download of Version 2 full benchmark artifacts from Kaggle outputs/ subdirectory."""

import os
import sys
import time
import requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

KERNEL_SLUG = "robustlens-order-effects-full-benchmark"
OWNER_SLUG = "aviral199"
VERSION_NUMBER = 2
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "order_effect_study" / "full" / "outputs"

ARTIFACTS = [
    "manifest.json",
    "all_order_predictions.csv.gz",
    "order_effects_summary.csv",
    "order_effects_leaderboard.csv",
    "order_pairwise_significance.csv",
    "order_omnibus_tests.json",
    "distortion_ancova.csv",
    "benchmark_report.json"
]


def check_and_download():
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {OUT_DIR}")
    print(f"Checking {OWNER_SLUG}/{KERNEL_SLUG} (Version {VERSION_NUMBER}) output availability...")

    for fname in ARTIFACTS:
        target_file = OUT_DIR / fname
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"  [ALREADY DOWNLOADED] {fname} ({target_file.stat().st_size:,} bytes)")
            continue

        downloaded = False
        for attempt in range(1, 6):
            try:
                print(f"  Fetching {fname} (attempt {attempt}/5)...")
                req = ApiDownloadKernelOutputRequest()
                req.owner_slug = OWNER_SLUG
                req.kernel_slug = KERNEL_SLUG
                req.file_path = f"outputs/{fname}"
                req.version_number = VERSION_NUMBER

                resp = client.kernels.kernels_api_client.download_kernel_output(req)
                r = requests.get(resp.url, timeout=30)
                if r.status_code == 200:
                    target_file.write_bytes(r.content)
                    print(f"  [SAVED] {fname}: {len(r.content):,} bytes")
                    downloaded = True
                    break
                else:
                    print(f"  [ERROR] {fname}: HTTP {r.status_code}")
            except Exception as e:
                print(f"  [RETRY] {fname} error: {e}")
                time.sleep(2)

        if not downloaded:
            print(f"Failed to download {fname} after 5 attempts.")
            return False

    print("\nAll 8 full benchmark artifacts successfully downloaded!")
    return True


if __name__ == "__main__":
    success = check_and_download()
    sys.exit(0 if success else 1)
