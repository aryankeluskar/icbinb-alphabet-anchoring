"""C8 completion: the 3B cell only, retried when the GPU has room.

The main C8 sweep (run_c8_scale.py) completed 8M/35M/150M/650M and then hit the registered
budget guard at 3B: `CUDA out of memory. Tried to allocate 100.00 MiB. GPU 0 has ... 49.81 MiB
free`, with three other users' jobs holding ~38 GB. The guard did its job -- the 8M-650M table
was written and the omission reported rather than left as a silent gap.

Nothing was wrong with the run, so nothing needs re-running: this script adds the missing
model and appends to the same CSV. The 5.7 GB checkpoint is already cached, so the only
missing resource is GPU memory.

Rather than fail again or hammer a shared box, it POLLS for headroom and starts only when
enough is free. It is a good citizen on purpose: the box is shared with other users, and a
job that retries aggressively would take memory that another job is about to allocate.

Existing 8M-650M rows are read back and rewritten unchanged, so the CSV stays one file with
one header and the published table remains internally consistent.
"""
import csv
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from run_c8_scale import pick_proteins, sweep_model   # noqa: E402

MODEL = "esm2_t36_3B_UR50D"
BATCH = 4                    # smaller than the 8 that OOMed
NEED_MIB = 26000             # ~5.7 GB fp32 weights + activations + headroom
POLL_S = 180
MAX_WAIT_H = 10


def free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    used, total = [int(x) for x in out.decode().strip().split("\n")[0].split(",")]
    return total - used


def main():
    out_path = os.path.join(HERE, "..", "results", "h2_c8_scale.csv")
    rows = list(csv.DictReader(open(out_path)))
    have = sorted(set(r["model"] for r in rows))
    print("C8 3B retry. Existing CSV has %d rows covering %s" % (len(rows), ", ".join(have)),
          flush=True)
    if MODEL in have:
        print("%s already present -- nothing to do." % MODEL, flush=True)
        return 0

    deadline = time.time() + MAX_WAIT_H * 3600
    while True:
        f = free_mib()
        if f >= NEED_MIB:
            print("[%s] %d MiB free >= %d needed -- starting."
                  % (time.strftime("%H:%M:%S"), f, NEED_MIB), flush=True)
            break
        if time.time() > deadline:
            print("[%s] gave up after %dh; %d MiB free, needed %d. 3B stays omitted and the "
                  "8M-650M table is reported as-is." % (time.strftime("%H:%M:%S"), MAX_WAIT_H,
                                                        f, NEED_MIB), flush=True)
            return 0
        print("[%s] %d MiB free, need %d -- waiting %ds"
              % (time.strftime("%H:%M:%S"), f, NEED_MIB, POLL_S), flush=True)
        time.sleep(POLL_S)

    proteins = pick_proteins()
    try:
        sweep_model(MODEL, BATCH, proteins, "cuda", rows, out_path)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        # Same guard as the parent script: a second failure must not damage what exists.
        print("\n!! %s FAILED AGAIN: %s" % (MODEL, exc), flush=True)
        print("!! the 8M-650M table on disk is untouched and remains the reported result.",
              flush=True)
        return 0

    print("\nappended %s -- CSV now %d rows" % (MODEL, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
