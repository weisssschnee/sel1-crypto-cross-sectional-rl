"""Hard OOS guard for shard building and training.

m1_build_shards.py has no sealed-window check of its own. This module is the
single place that decides whether a shard directory is allowed to be built,
pushed, or trained on, and it refuses anything that reaches 2026-07-01.

Import and call guard_shard_dir() before any build or training step; call
guard_month_list() before handing a month list to the builder.
"""
from __future__ import annotations

import glob
import os
import zipfile

import numpy as np
import pandas as pd

OOS_FIRST = pd.Timestamp('2026-07-01', tz='UTC')
TRAIN_LAST = '2025-12'
DEV_LAST = '2026-06'
ALLOWED_ROLES = {'TRAIN': (f'2024-01', TRAIN_LAST), 'DEV': ('2026-01', DEV_LAST)}


class OOSViolation(RuntimeError):
    pass


def guard_month_list(months, role: str):
    """Reject any month list that touches the sealed window or leaves its role."""
    if role not in ALLOWED_ROLES:
        raise OOSViolation(f'unknown role {role!r}; expected one of {sorted(ALLOWED_ROLES)}')
    lo, hi = ALLOWED_ROLES[role]
    bad = [m for m in months if not (lo <= m <= hi)]
    if bad:
        raise OOSViolation(
            f'{role} build rejected: months outside [{lo}, {hi}] -> {sorted(bad)}. '
            f'Sealed OOS begins {OOS_FIRST}.')
    return sorted(months)


def guard_shard_dir(shard_dir: str, role: str, expect_months=None):
    """Open every npz in shard_dir and verify it ends before the seal.

    Checks the shard's own n_min, not its filename, so a mislabelled file
    cannot slip through.
    """
    if role not in ALLOWED_ROLES:
        raise OOSViolation(f'unknown role {role!r}')
    lo, hi = ALLOWED_ROLES[role]
    paths = sorted(glob.glob(os.path.join(shard_dir, 'm1_shard_*.npz')))
    if not paths:
        raise OOSViolation(f'{shard_dir}: no m1_shard_*.npz found')
    seen = []
    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            month = str(z['month'])
            n_min = int(z['n_min'])
        except (zipfile.BadZipFile, ValueError, KeyError, OSError) as exc:
            raise OOSViolation(
                f'{os.path.basename(p)}: cannot read as a complete shard ({type(exc).__name__}). '
                f'If a build is still running, wait for it to finish before guarding.') from exc
        first = pd.Timestamp(month + '-01', tz='UTC')
        last = first + pd.Timedelta(minutes=n_min - 1)
        if last >= OOS_FIRST:
            raise OOSViolation(
                f'{os.path.basename(p)}: last minute {last} is inside the sealed OOS '
                f'(from {OOS_FIRST}). Refusing.')
        if not (lo <= month <= hi):
            raise OOSViolation(
                f'{os.path.basename(p)}: month {month} is outside the {role} range [{lo}, {hi}].')
        # funding events must also stay inside the month
        fund = z['fund'].reshape(-1, 3)
        if len(fund):
            top = int(fund[:, 1].max())
            if top >= n_min:
                raise OOSViolation(
                    f'{os.path.basename(p)}: funding minute index {top} >= n_min {n_min}.')
        seen.append(month)
    if expect_months is not None:
        missing = sorted(set(expect_months) - set(seen))
        extra = sorted(set(seen) - set(expect_months))
        if missing or extra:
            raise OOSViolation(f'{shard_dir}: missing={missing} unexpected={extra}')
    return seen


def assert_no_oos_source_read(path: str):
    """Parquet footer check: refuse a source file whose timestamps reach the seal."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    names = md.schema.to_arrow_schema().names
    if 'timestamp' not in names:
        return
    j = names.index('timestamp')
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(j).statistics
        if st is None or not st.has_min_max:
            raise OOSViolation(f'{path}: timestamp column has no min/max; cannot verify the seal')
        top = pd.Timestamp(st.max)
        if top.tzinfo is not None:
            top = top.tz_convert('UTC')
        else:
            top = top.tz_localize('UTC')
        if top >= OOS_FIRST:
            raise OOSViolation(f'{path}: timestamp reaches {top}, inside the sealed OOS')


if __name__ == '__main__':
    import sys
    d = sys.argv[1]
    r = sys.argv[2] if len(sys.argv) > 2 else 'TRAIN'
    months = guard_shard_dir(d, r)
    print(f'OK {r} guard passed: {len(months)} shards, all end before {OOS_FIRST}')
    print('  ', ', '.join(months))
