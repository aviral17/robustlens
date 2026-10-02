"""Run robustness evaluation pipeline.

Usage:
    uv run python scripts/run_eval.py --models resnet50 convnext_tiny vit_base_patch16_224 \
        --dataset cifar10 --severities 1 2 3 4 5 --output ../results/
"""


def main():
    """Entry point for robustness evaluation."""
    raise NotImplementedError("Will be implemented in Phase 2-3")


if __name__ == "__main__":
    main()
