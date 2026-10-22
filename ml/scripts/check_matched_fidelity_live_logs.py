"""Check live execution logs for the running Kaggle kernel."""

import json
import socket
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

def main():
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()

    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name = "aviral199"
    req.kernel_slug = "robustlens-matched-fidelity-full-benchmark"

    res = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)

    try:
        sock = res.raw._fp.fp.raw._sock
        sock.settimeout(6.0)
    except Exception as e:
        print("Could not set socket timeout:", e)

    chunks = []
    reason = "EOF reached"
    try:
        for chunk in res.iter_content(chunk_size=8192, decode_unicode=True):
            chunks.append(chunk)
    except Exception as e:
        reason = f"Exception: {type(e).__name__}: {e}"

    full_text = "".join(chunks)
    print(f"Stream termination reason: {reason}")
    print(f"Total characters received from stream: {len(full_text):,}")

    lines = full_text.splitlines()
    parsed_lines = []
    for line in lines:
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
                text = d.get("data", "").rstrip()
                if text:
                    t = d.get("time", 0)
                    sname = d.get("stream_name", "")
                    parsed_lines.append(f"[{sname}] (t={t:.1f}s) {text}")
            except Exception:
                parsed_lines.append(line)

    print(f"Total parsed output lines: {len(parsed_lines):,}")
    print("\n=== LAST 30 LINES OF EXECUTION LOG ===")
    for l in parsed_lines[-30:]:
        print(l)

if __name__ == "__main__":
    main()
