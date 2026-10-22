"""Wait for Kaggle kernel to complete and download outputs to results/order_effect_study/full/outputs/"""

import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = "aviral199/robustlens-order-effects-full-benchmark"
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "order_effect_study" / "full" / "outputs"


def main():
    api = KaggleApi()
    api.authenticate()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Monitoring {KERNEL}...", flush=True)
    print(f"Target download directory: {OUT_DIR}", flush=True)

    start_time = time.time()
    max_wait = 1800  # 30 minutes max

    while time.time() - start_time < max_wait:
        res = api.kernels_status(KERNEL)
        status_str = str(getattr(res, "status", ""))
        failure_msg = getattr(res, "failure_message", None) or getattr(res, "failureMessage", None)
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Status: {status_str}", flush=True)

        if "COMPLETE" in status_str:
            print("Kernel execution completed successfully! Downloading outputs...", flush=True)
            api.kernels_output_cli(KERNEL, path=str(OUT_DIR), force=True)
            print("Download complete. Output directory contents:", flush=True)
            for f in OUT_DIR.iterdir():
                print(f"  {f.name}: {f.stat().st_size:,} bytes", flush=True)
            return 0
        elif any(term in status_str for term in ["FAIL", "CANCEL", "ERROR"]):
            print(f"Kernel terminated with status: {status_str}, failure: {failure_msg}", flush=True)
            try:
                logs = api.kernels_logs(KERNEL)
                print("Logs:", flush=True)
                print(logs, flush=True)
            except Exception as e:
                print(f"Failed to fetch logs: {e}", flush=True)
            return 1

        time.sleep(20)

    print("Timeout reached while waiting for kernel.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
