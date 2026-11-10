"""Reading and writing result files.

Embedding  results/.../<method>_seed<k>.npz   Z, runtime, n, kwargs, run_id (+ extras such as stress_curve)
Metrics    results/.../<method>_seed<k>.json  {"run_id": ..., "metrics": {name: value}}

run_id identifies one embedding run. Metrics remember which run they were computed
from, so metrics of an embedding that has since been recomputed count as stale.
All writes are atomic, so a crash never leaves a half-written file behind.
"""
import json
import os
import uuid

import numpy as np

RESERVED_KEYS = {"Z", "runtime", "n", "kwargs", "run_id"}


def canonical(kwargs):
    """JSON round-trip, so kwargs compare equal to what was stored on disk."""
    return json.loads(json.dumps(kwargs, sort_keys=True))


def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # keep the suffix: np.savez appends .npz otherwise
    write(tmp)
    os.replace(tmp, path)


def save_embedding(path, Z, runtime, kwargs, extras=None):
    extras = extras or {}
    clash = RESERVED_KEYS & set(extras)
    if clash:
        raise ValueError(f"extras may not use reserved keys {sorted(clash)}")
    run_id = uuid.uuid4().hex
    _atomic_write(
        path,
        lambda tmp: np.savez(
            tmp,
            Z=np.asarray(Z),
            runtime=runtime,
            n=len(Z),
            kwargs=json.dumps(canonical(kwargs)),
            run_id=run_id,
            **extras,
        ),
    )
    return run_id


def load_embedding(path, key="Z"):
    """The embedding Z, or another array saved with it (e.g. key="stress_curve")."""
    with np.load(path) as f:
        return f[key]


def read_embedding_meta(path):
    with np.load(path) as f:
        return {
            "run_id": str(f["run_id"]),
            "runtime": float(f["runtime"]),
            "n": int(f["n"]),
            "kwargs": json.loads(str(f["kwargs"])),
        }


def save_metrics(path, run_id, metrics):
    payload = {"run_id": run_id, "metrics": metrics}
    _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=1)))


def load_metrics(path, run_id=None):
    """Stored metrics; empty if the file is missing or (when run_id is given) stale."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if run_id is not None and payload.get("run_id") != run_id:
        return {}
    return payload["metrics"]
