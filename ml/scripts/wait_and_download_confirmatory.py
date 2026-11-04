"""Monitor Kaggle confirmatory execution, stream progress, and download output artifacts."""

import os
import sys
import time
import json
import requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionLogsStreamRequest,
    ApiDownloadKernelOutputRequest,
)

KERNEL_SLUG = "robustlens-distortion-fingerprints-confirmatory"
OWNER_SLUG = "aviral199"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "matched_fidelity_study" / "confirmatory_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = [
    "full_distortion_fingerprints_and_margins.csv.gz",
    "confirmatory_execution_manifest.json",
    "confirmatory_grouped_cv_results.csv",
    "confirmatory_grouped_cv_diagnostics.json",
]


def check_status(api):
    res = api.kernels_status(f"{OWNER_SLUG}/{KERNEL_SLUG}")
    return str(getattr(res, "status", ""))


def fetch_latest_logs(client):
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name = OWNER_SLUG
    req.kernel_slug = KERNEL_SLUG
    try:
        res = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
        sock = res.raw._fp.fp.raw._sock
        sock.settimeout(5.0)
        chunks = []
        for chunk in res.iter_content(chunk_size=8192, decode_unicode=True):
            chunks.append(chunk)
        full_text = "".join(chunks)
        lines = []
        for line in full_text.splitlines():
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                    text = d.get("data", "").rstrip()
                    if text:
                        lines.append(text)
                except Exception:
                    pass
        return lines[-10:] if lines else []
    except Exception:
        return []


def download_outputs(api, client):
    print("\nDownloading kernel output artifacts...")
    # Try downloading via kaggle API output method
    try:
        api.kernels_output(f"{OWNER_SLUG}/{KERNEL_SLUG}", path=str(OUT_DIR))
        print(f"Downloaded outputs to {OUT_DIR}")
    except Exception as e:
        print(f"Standard output download error: {e}")

    # Verify expected files
    found_files = []
    for f in EXPECTED_FILES:
        p = OUT_DIR / f
        if p.exists() and p.stat().st_size > 0:
            found_files.append((f, p.stat().st_size))
            print(f"  [OK] {f}: {p.stat().st_size:,} bytes")
        else:
            print(f"  [MISSING/EMPTY] {f}")
    return len(found_files) == len(EXPECTED_FILES)


def main():
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()

    print("=" * 80)
    print(f"MONITORING KAGGLE KERNEL: {OWNER_SLUG}/{KERNEL_SLUG}")
    print("=" * 80)

    start_t = time.time()
    last_log_tail = []

    while True:
        status = check_status(api)
        elapsed = time.time() - start_t
        print(f"[{elapsed/60:.1f}m] Kernel status: {status}", flush=True)

        if "complete" in status.lower():
            print("\nKernel execution COMPLETED successfully!", flush=True)
            break
        elif "error" in status.lower() or "cancel" in status.lower():
            print(f"\nKernel finished with error status: {status}", flush=True)
            logs = fetch_latest_logs(client)
            for l in logs:
                print(f"  LOG: {l}", flush=True)
            sys.exit(1)

        # Print latest logs
        logs = fetch_latest_logs(client)
        if logs and logs != last_log_tail:
            print("  --- Latest Progress ---", flush=True)
            for l in logs[-5:]:
                clean_l = l.encode("ascii", "replace").decode("ascii")
                print(f"  {clean_l}", flush=True)
            last_log_tail = logs

        time.sleep(30)

    # Download outputs
    success = download_outputs(api, client)
    if success:
        print("\nAll confirmatory artifacts downloaded and verified!")
        sys.exit(0)
    else:
        print("\nSome artifacts were missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
