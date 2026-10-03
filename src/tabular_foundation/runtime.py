"""Deterministic scheduling, allocation accounting and atomic artifacts."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np


def seed_for(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**63 - 1)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def profile_schedule(seed, update, batch_size, finance_share):
    """One entry per emitted macroepisode; four reuses never change its mixture.

    In the joint profile, finance counts are fixed within each four-update reuse
    group. Five groups emit exactly 20% finance for every integer batch size.
    No finite finance world occurs twice within one logical optimizer batch.
    """
    if finance_share not in (0, .2, 1):
        raise ValueError("Supported controlled profiles have finance_share 0, 0.2 or 1")
    if batch_size < 1 or update < 0:
        raise ValueError("Invalid batch size or update")
    group, reuse = divmod(update, 4)
    if finance_share == .2:
        nfin = batch_size // 5 + int(group % 5 < batch_size % 5)
    else:
        nfin = batch_size if finance_share == 1 else 0
    entries = []
    for j in range(nfin):
        entries.append({"profile": "finance", "world_seed": seed_for(seed, "finance", group, j),
                        "query_seed": seed_for(seed, "finance-query", group, j, reuse),
                        "reuse": reuse})
    for j in range(batch_size - nfin):
        entries.append({"profile": "standard", "world_seed": seed_for(seed, "standard", update, j),
                        "query_seed": seed_for(seed, "standard-query", update, j), "reuse": 0})
    rng = np.random.default_rng(seed_for(seed, "schedule", group))
    return [entries[i] for i in rng.permutation(len(entries))]


def lr_factor(step, horizon):
    if horizon < 1 or step < 0:
        raise ValueError("Invalid schedule")
    warm = max(1, math.ceil(.02 * horizon))
    if step < warm:
        return (step + 1) / warm
    progress = min(1., (step + 1 - warm) / max(1, horizon - warm))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def stage_for(step, stage_end_steps):
    if len(stage_end_steps) != 3 or not 0 < stage_end_steps[0] <= stage_end_steps[1] <= stage_end_steps[2]:
        raise ValueError("Three nondecreasing positive stage endpoints are required")
    return 1 if step < stage_end_steps[0] else 2 if step < stage_end_steps[1] else 3


def charged_gpu_hours(wall_seconds, allocated_gpus):
    if wall_seconds < 0 or allocated_gpus < 0:
        raise ValueError("Negative allocation")
    return wall_seconds * allocated_gpus / 3600


class ProjectLedger:
    """Locked cumulative accounting across local/shared-filesystem run writers.

    The ledger is an accounting tool, not a scheduler reservation system. Use
    scheduler allocations to prevent concurrent jobs overspending between writes.
    A killed process can leave its final unreported interval uncharged; reconcile
    with scheduler sacct/job records before approving another allocation.
    """
    def __init__(self, path, cap=50000.):
        self.path, self.cap = Path(path), float(cap)
        if self.cap <= 0:
            raise ValueError("Project cap must be positive")

    def recorded_hours(self, run_id):
        """Return already charged work, including attempts newer than a checkpoint."""
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"runs": {}}
            return float(data["runs"].get(run_id, 0.))

    def update(self, run_id, cumulative_gpu_hours):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"runs": {}}
            previous = float(data["runs"].get(run_id, 0))
            if cumulative_gpu_hours < previous - 1e-9:
                raise ValueError("Cumulative run accounting cannot move backwards")
            data["runs"][run_id] = float(cumulative_gpu_hours)
            data["total_gpu_hours"] = sum(data["runs"].values())
            data["cap_gpu_hours"] = self.cap
            atomic_json(self.path, data)
            return data["total_gpu_hours"] <= self.cap
