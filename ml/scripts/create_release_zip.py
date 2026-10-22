#!/usr/bin/env python3
"""Create the clean canonical RobustLens_Research_Package.zip containing only reader-facing assets."""

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "reports/final_research_package"
ZIP_PATH = PKG_DIR / "RobustLens_Research_Package.zip"

FILES_TO_INCLUDE = [
    PKG_DIR / "RobustLens_Compound_Corruption_Manuscript.md",
    PKG_DIR / "RobustLens_Compound_Corruption_Manuscript.pdf",
    PKG_DIR / "Technical_Supplement.md",
    PKG_DIR / "REPRODUCIBILITY.md",
    PKG_DIR / "LITERATURE_REVIEW_MATRIX.md",
]

FIGURES = [
    PKG_DIR / "figures/figure1_order_accuracy.png",
    PKG_DIR / "figures/figure2_accuracy_fidelity.png",
    PKG_DIR / "figures/figure3_range_contraction.png",
    PKG_DIR / "figures/figure4_prediction_churn.png",
]

ANALYSIS_FILES = list((PKG_DIR / "analysis").glob("*.csv")) + list((PKG_DIR / "analysis").glob("*.json"))


def build_zip():
    print(f"Creating release archive: {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in FILES_TO_INCLUDE:
            if f.exists():
                arcname = f.name
                zf.write(f, arcname=arcname)
                print(f"  Added: {arcname}")
            else:
                print(f"  Warning: {f} not found yet (will add when created)")

        for fig in FIGURES:
            if fig.exists():
                arcname = f"figures/{fig.name}"
                zf.write(fig, arcname=arcname)
                print(f"  Added: {arcname}")

        for a in ANALYSIS_FILES:
            if a.exists():
                arcname = f"analysis/{a.name}"
                zf.write(a, arcname=arcname)
                print(f"  Added: {arcname}")

    print(f"Successfully created: {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_zip()
