"""Shared helpers for train_basin_riskpg_v1.py and random_baseline_v1.py.

Factored out so both scripts log formulas in the same FORMULAS.jsonl schema and
share one multiprocessing worker pool implementation, instead of drifting apart.
Nothing here touches the sha256-pinned legacy files; this module is new code.
"""
from __future__ import annotations
import os

# Must run before numpy/torch import in THIS process, and inherited by every
# spawned pool worker's own fresh import of this module: 2026-09-29 finding
# (validate_128x256_v3_fixed first launch) -- with N pool workers each
# defaulting to its own multi-thread OpenBLAS/MKL/OMP pool, N x (4-8 threads)
# fought over 16 physical cores and produced ZERO completed batches in 30
# minutes despite every worker's CPU-seconds climbing (busy, not deadlocked).
# One thread per worker process turns that into one thread per core.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib, json, sys
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = "D:/riskpg_stateful_v5s_governor_nextgen_beem"
LEGACY_ROOT = "D:/v4r_groot/Chengbo/workspaces/alphagpt-rl-dl-base"
VOCAB = os.path.join(ROOT, "artifacts/xs_basin_v1/XS_BASIN_VOCAB_V1.json")


def _resolve_panel():
    """Default: the trial 2025janmay panel, unchanged. An explicit override
    must pin the panel's sha256 and is verified at import time in every
    process (main + each pool worker), so a swapped file can never silently
    become the training surface."""
    p = os.environ.get("XS_BASIN_PANEL")
    if not p:
        return os.path.join(ROOT, "runtime/expanded17m_2025janmay",
                            "BROAD893_2025JANMAY_HOURLY44_TARGET_LONG_V1.parquet")
    want = os.environ.get("XS_BASIN_PANEL_SHA256", "").strip().lower()
    if not want:
        raise RuntimeError("XS_BASIN_PANEL is set but XS_BASIN_PANEL_SHA256 is not; refusing to run")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != want:
        raise RuntimeError(f"XS_BASIN_PANEL sha256 mismatch: {p} hashes {got}, expected {want}")
    return p


PANEL = _resolve_panel()

EPISODE_HOURS = 168
N_EPISODES = 8
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0001
MINIMUM_HOLD_HOURS = 24
REWARD_FLOOR = -2.0
MIN_GROSS_EXPOSURE = 0.5
MAX_PROGRAM_LEN = 48


def ensure_sys_path():
    for p in (LEGACY_ROOT, os.path.join(LEGACY_ROOT, "experiments")):
        if p not in sys.path: sys.path.insert(0, p)
    if os.path.join(ROOT, "src") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "src"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_basin_fields(basin_id):
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    basin = next(b for b in vocab["basins"] if b["basin_id"] == basin_id)
    return [f["field"] for f in basin["fields"]]


def load_panel(vocab_fields):
    """(assets, features, times) panel plus what the accounting needs.

    Verbatim copy of train_basin_riskpg_v1.py's load_panel (itself copied from
    smoke_xs_basin_reward_v1.py), centralized here so both scripts and every
    pool worker use one identical implementation.
    """
    import pyarrow.parquet as pq
    need = ["symbol", "timestamp", "forward_trade_return_1h", "funding_rate"]
    cols = need + [f for f in vocab_fields if f not in need]
    d = pq.read_table(PANEL, columns=cols).to_pandas().sort_values(["symbol", "timestamp"])
    syms = np.sort(d["symbol"].unique()); ts = np.sort(d["timestamp"].unique())
    si = d["symbol"].map({s: i for i, s in enumerate(syms)}).to_numpy()
    ti = d["timestamp"].map({t: i for i, t in enumerate(ts)}).to_numpy()
    nS, nT = len(syms), len(ts)

    def grid(col):
        a = np.full((nT, nS), np.nan, dtype=np.float64)
        a[ti, si] = d[col].to_numpy(dtype=np.float64)
        return a

    feats = np.stack([grid(f) for f in vocab_fields])
    fwd = grid("forward_trade_return_1h"); fund = grid("funding_rate")
    eligible = np.isfinite(fwd); funding_event = np.isfinite(fund)
    panel = torch.from_numpy(np.ascontiguousarray(feats.transpose(2, 0, 1)))
    return dict(panel=panel, forward=fwd, funding=np.where(funding_event, fund, 0.0),
                funding_event=funding_event, eligible=eligible,
                n_assets=nS, n_times=nT)


def episode_starts(n_times, rng):
    usable = n_times - EPISODE_HOURS
    edges = np.linspace(0, usable, N_EPISODES + 1).astype(int)
    return [int(rng.integers(edges[i], max(edges[i] + 1, edges[i + 1])))
            for i in range(N_EPISODES)]


def signal_to_weights(signal, eligible):
    w = np.zeros_like(signal)
    for t in range(signal.shape[0]):
        row = signal[t]; live = np.isfinite(row) & eligible[t]; n = int(live.sum())
        if n < 10: continue
        v = row[live]; rank = np.empty(n)
        rank[v.argsort(kind="stable")] = np.arange(n, dtype=float)
        centred = rank - rank.mean(); denom = np.abs(centred).sum()
        if denom <= 0: continue
        w[t, live] = centred / denom
    return w
def evaluate_one(tokens, vm, data, starts, policy_mod, accounting_mod, invalid_cls):
    raw = vm.execute(tokens, data["panel"])
    signal = raw.detach().cpu().numpy().T
    contract = policy_mod.StatefulActionContract(minimum_hold_hours=MINIMUM_HOLD_HOURS)
    acct = accounting_mod.AccountingContract()
    logs, gross = [], []
    for s0 in starts:
        sl = slice(s0, s0 + EPISODE_HOURS); elig = data["eligible"][sl]
        desired = signal_to_weights(signal[sl], elig)
        held = policy_mod.apply_stateful_action_policy(
            desired_weights=desired, eligible=elig, contract=contract)[0]
        res = accounting_mod.evaluate_target_weights_numpy(
            target_weights=held, forward_returns=np.where(elig, data["forward"][sl], 0.0),
            fee_rate=FEE_RATE, slippage_rate=SLIPPAGE_RATE,
            funding_rate=data["funding"][sl], funding_event=data["funding_event"][sl],
            eligible=elig, contract=acct)
        nav = float(np.asarray(res.nav).reshape(-1)[-1])
        logs.append(np.log(nav) if nav > 0 else REWARD_FLOOR)
        gross.append(float(np.abs(np.asarray(held).reshape(EPISODE_HOURS, -1)).sum(1).mean()))
    if float(np.mean(gross)) < MIN_GROSS_EXPOSURE:
        raise invalid_cls("DEGENERATE_NO_TRADE")
    return np.array(logs, dtype=np.float64)


def reward_from_logs(logs):
    if not np.all(np.isfinite(logs)): return REWARD_FLOOR
    return float(np.mean(logs))


def formula_fingerprint(tokens, logs):
    """Cheap behavior fingerprint: hash of the per-episode NAV log-returns.

    Two formulas with different token sequences but the same rounded logs
    behaved identically across all 8 episodes -- that's the collision this is
    meant to catch. Rounding to 4dp so float noise doesn't split near-dupes.
    Re-executing the VM a second time just to fingerprint would double
    evaluation cost, so this reuses the `logs` evaluate_one already produced.
    """
    h = hashlib.sha1(np.round(np.asarray(logs, dtype=np.float64), 4).tobytes())
    return h.hexdigest()[:16]


def append_formula_log(fh, *, update, tokens, vm, fields, reward, invalid_reason, logs):
    row = {
        "update": update,
        "tokens": list(int(t) for t in tokens),
        "expr": vm.describe(tokens, fields),
        "reward": round(float(reward), 6),
        "invalid_reason": invalid_reason,
        "fingerprint": formula_fingerprint(tokens, logs) if logs is not None else None,
        "logs": [round(float(x), 6) for x in logs] if logs is not None else None,
    }
    fh.write(json.dumps(row) + "\n")


class UniformStub(torch.nn.Module):
    """Zero-logit model: softmax over legal-only logits is uniform over legal
    actions. Used for both the training script's null baseline and the
    random-baseline task -- same object, so 'uniform' means the same thing in
    both places."""
    def __init__(self, token_count):
        super().__init__()
        self.token_count = token_count

    def forward(self, inp):
        return torch.zeros(inp.shape[0], self.token_count, device=inp.device), None, None


# --- multiprocessing pool worker: module-level so Windows spawn can pickle it ---
_W = {}


def _acquire_load_lock():
    """Opt-in cross-process serialization of the pool workers' panel load.

    2026-09-30 B2 attempt-1 finding: 12 workers concurrently reading the
    1.72 GB TRAIN parquet, each with an ~8-10 GB pandas transient, crashed
    natively (8x python.exe fault 0x66bc306f in the Application log) on
    93.4 GB RAM while commit sat near its 232 GB ceiling, and the Pool
    repopulated the dead workers in a loop, so update 0 never completed.
    Setting XS_BASIN_PANEL_LOAD_LOCK_DIR to a scratch dir serializes the
    loads; leaving it unset (default) changes nothing for any caller."""
    import shutil
    import time as _time
    d = os.environ.get("XS_BASIN_PANEL_LOAD_LOCK_DIR")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, "panel_load.lockdir")
    deadline = _time.monotonic() + 3600
    while True:
        try:
            os.mkdir(lock)
            return lock
        except FileExistsError:
            try:
                if _time.time() - os.stat(lock).st_mtime > 900:
                    shutil.rmtree(lock, ignore_errors=True)  # crashed holder
                    continue
            except FileNotFoundError:
                pass
            if _time.monotonic() > deadline:
                raise RuntimeError("panel load lock timeout (60 min)")
            _time.sleep(5)


def pool_init(basin_id, seed):
    ensure_sys_path()
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    from okx_alpha_discovery.evaluation import accounting as accounting_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as _vm_mod
    # Opt-in blocked window operators (bit-identical to the whole-panel path:
    # 156 parametrized tests + 2436-formula full-window parity, 2026-09-30).
    # Default unset leaves BLOCK_ELEMS=0, exactly as before.
    if os.environ.get("XS_BASIN_VM_BLOCK_ELEMS"):
        _vm_mod.BLOCK_ELEMS = int(os.environ["XS_BASIN_VM_BLOCK_ELEMS"])
    # Opt-in EMA kernel cache cap (pure memory management, zero numeric
    # change; 2026-10-01 B2 attempt-3 wedge: one float64 T x T kernel is
    # 2.45 GB at T=17520 and the unevicted cache killed workers).
    if os.environ.get("XS_BASIN_EMA_KERNEL_CACHE_MAX"):
        _vm_mod.EMA_KERNEL_CACHE_MAX = int(os.environ["XS_BASIN_EMA_KERNEL_CACHE_MAX"])
    fields = load_basin_fields(basin_id)
    import shutil
    _lock = _acquire_load_lock()
    try:
        data = load_panel(fields)
    finally:
        if _lock is not None:
            shutil.rmtree(_lock, ignore_errors=True)
    vm = XsTsCsStackVM(len(fields))
    _W.update(vm=vm, data=data, policy_mod=policy_mod,
              accounting_mod=accounting_mod, invalid_cls=CandidateLocalInvalid,
              fields=fields)


def root_distribution(tokens_list, lengths, vm):
    """Root = last real token before END (or padding). Counts by kind:name."""
    import collections
    c = collections.Counter()
    for tk, n in zip(tokens_list, lengths):
        body = [int(t) for t in tk[: int(n)] if not vm.is_end(int(t))]
        if not body:
            c[("EMPTY", None)] += 1; continue
        last = body[-1]
        if vm.is_operator(last):
            c[("OP", vm.operator_name(last))] += 1
        elif vm.is_terminal(last):
            c[("TERM", last)] += 1
        else:
            c[("WINDOW", last)] += 1
    return c


def summarize_roots(counter, n):
    top = counter.most_common(5)
    top_share = 100.0 * (top[0][1] / n) if top else 0.0
    return {
        "n_distinct_roots": len(counter),
        "top_root": "{}:{}".format(*top[0][0]) if top else None,
        "top_root_share_pct": round(top_share, 1),
        "top5": [{"root": "{}:{}".format(*k), "share_pct": round(100.0 * v / n, 1)}
                 for k, v in top],
    }


def pool_eval_one(args):
    """args = (tokens, starts) so window rotation is decided by the caller, not
    frozen at pool-init time. Returns (status, payload). status in
    {"ok","invalid","error"}; payload is a list[float] of per-episode logs for
    "ok", else a reason str."""
    tokens, starts = args
    try:
        logs = evaluate_one(tokens, _W["vm"], _W["data"], starts,
                             _W["policy_mod"], _W["accounting_mod"], _W["invalid_cls"])
        return ("ok", logs.tolist())
    except _W["invalid_cls"] as exc:
        return ("invalid", str(exc))
    except Exception as exc:
        return ("error", type(exc).__name__)

