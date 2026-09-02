#!/usr/bin/env python
"""Warm up the explainer backend before viewing.

Usage:
    source /scratch/author/icbinb/src/activate_env.sh
    python /scratch/author/icbinb/src/warmup.py [--model esm2_t33_650M_UR50D]

This sends a POST to the server's /api/warmup endpoint. The server must
already be running (start it with server.py). If the model is already
loaded, this is a no-op.

If the server is NOT running, this script will start it in the background,
wait for it to load the model, and then exit — leaving the server running
for the explainer to connect to.

Common model sizes:
    esm2_t6_8M_UR50D       — 8M params, loads in ~10s, fast Jacobian
    esm2_t12_35M_UR50D     — 35M params
    esm2_t30_150M_UR50D    — 150M params, good balance
    esm2_t33_650M_UR50D    — 650M params (paper's headline model)
"""
import argparse
import os
import sys
import time
import urllib.request
import json
import subprocess
import signal

DEFAULT_MODEL = "esm2_t33_650M_UR50D"
BASE_URL = "http://localhost:8765"


def check_status():
    """Check if the server is running and what's loaded."""
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/status", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def warmup(model_name):
    """Send a warmup request to the server."""
    data = json.dumps({"model": model_name}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/warmup",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"ESM-2 model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    global BASE_URL
    BASE_URL = f"http://localhost:{args.port}"

    # Check if server is already running
    st = check_status()
    server_proc = None

    if st is None:
        print(f"Server not running on port {args.port}. Starting it...")
        env = dict(os.environ)
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        server_proc = subprocess.Popen(
            [sys.executable, server_script, "--model", args.model, "--port", str(args.port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait for the server to come up
        print("Waiting for server to start...", flush=True)
        for attempt in range(60):
            time.sleep(2)
            st = check_status()
            if st is not None:
                print(f"Server is up.", flush=True)
                break
        else:
            print("ERROR: server did not start within 120s.", file=sys.stderr)
            if server_proc:
                server_proc.terminate()
            sys.exit(1)

    # Check if model is already loaded
    if st and st.get("loaded") and st.get("model_name") == args.model:
        print(f"Model {args.model} is already loaded "
              f"({st['n_params'] / 1e6:.0f}M params on {st['device']}).")
        print(f"  Server: {BASE_URL}")
        print(f"  Open the HTML explainer — it will auto-connect.")
        return

    # Warm up the model
    if st and not st.get("loaded"):
        print(f"Warming up {args.model}...", flush=True)
        t0 = time.time()
        result = warmup(args.model)
        elapsed = time.time() - t0
        print(f"  Loaded in {result.get('load_time', elapsed):.1f}s "
              f"({result['n_params'] / 1e6:.0f}M params on {result['device']})")
    elif st and st.get("loaded") and st.get("model_name") != args.model:
        print(f"Different model loaded ({st['model_name']}). Unloading and loading {args.model}...")
        # Unload first
        req = urllib.request.Request(f"{BASE_URL}/api/unload", method="POST",
                                     data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
        # Now warmup
        print(f"Warming up {args.model}...", flush=True)
        result = warmup(args.model)
        print(f"  Loaded in {result.get('load_time', 0):.1f}s "
              f"({result['n_params'] / 1e6:.0f}M params on {result['device']})")
    else:
        print(f"Model {args.model} is already loaded.")
        if st:
            print(f"  {st['n_params'] / 1e6:.0f}M params on {st['device']}")

    print(f"\nServer: {BASE_URL}")
    print(f"Open the HTML explainer — it will auto-connect.")
    print(f"Press Ctrl+C to stop the server when done (frees the GPU).")

    # If we started the server, forward its output and wait
    if server_proc:
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server_proc.send_signal(signal.SIGINT)
            server_proc.wait()


if __name__ == "__main__":
    main()
