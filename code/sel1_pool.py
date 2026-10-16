"""XS_SEL1 composite books (supplement 10-15 afternoon item 1 sel1_combo, evening item 5 --pool). New file;
sel1_common / sel1_sweep are imported, never modified.

  layout     the signal hours of the requested halves' months (month m: hours m0-1 .. m1-2; the month's row 0 holds
             nothing, as S.select_month) and the mask positions on those hours (TOP100 / not BTC / finite beta: the
             same for every formula, so a formula is stored as its values on those positions only).
  rank z     per signal hour, over the coins where the formula is finite: average rank (ties -> their mean rank,
             pandas' default), minus the hour's mean, / the hour's std (ddof 0); all tied -> 0. The finite set is kept.
  composite  sum over members of sign_j * z_j (sign +1 in the pool, sign of t_A in sel1_combo), members in list order;
             a member not finite at a position adds 0; a position where no member is finite is not finite.
  select     S.select_month's rule through vm_select_gpu's month loop (stable argsort, -0.0 -> +0.0). Rank z is
             strictly increasing in the signal and keeps its ties and its finite set, so a one-member composite
             selects exactly what the member's own signal selects: the empty pool's first reward is V2's reward.
  signal     vm_select_gpu's signal part (VM per year chunk, NULL roll, mask, coverage) on the context's device.
"""
from __future__ import annotations

import numpy as np
import torch

import sel1_common as S

NAN, INF = float("nan"), float("inf")


def masked_signal(view, vm, body):
    """(A, T) float64 on the view's device, NaN off the mask; S.Invalid on low coverage (vm_select_gpu's lines)."""
    dev = view["feats_gpu"].device
    if "mask_gpu" not in view:
        A_, T_ = view["mask"].shape
        view["mask_gpu"] = torch.from_numpy(np.ascontiguousarray(view["mask"])).to(dev)
        view["cover_gpu"] = torch.from_numpy(np.asarray(view["cover_rows"], np.int64)).to(dev)
        if view["offsets"] is not None:
            off = torch.from_numpy(np.asarray(view["offsets"], np.int64)).to(dev)
            view["roll_idx"] = (torch.arange(T_, device=dev)[None, :] - off[:, None]) % T_
    A, T = view["mask"].shape
    s = torch.full((A, T), NAN, dtype=torch.float64, device=dev)
    with torch.no_grad():
        for in0, o0, o1 in view["chunks"]:
            r = vm.execute(list(body), view["feats_gpu"][:, :, in0:o1]).detach()
            s[:, o0:o1] = r[:, o0 - in0:].to(torch.float64)
        if view["offsets"] is not None:
            s = torch.gather(s, 1, view["roll_idx"])
        s = torch.where(view["mask_gpu"] & torch.isfinite(s), s, torch.full_like(s, NAN))
        n = torch.isfinite(s[:, view["cover_gpu"]]).sum(dim=0).cpu().numpy()
    cov = float((n >= S.MIN_COINS).mean())
    if cov < S.COVER_MIN:
        raise S.Invalid("LOW_COVERAGE_%.2f" % cov)
    return s


class Layout:
    def __init__(self, ctx, halves=("A",)):
        self.dev = ctx["feats_gpu"].device
        self.months = [m for m in ctx["months"] if m["half"] in halves]
        hrs = np.concatenate([np.arange(m["m0"] - 1, m["m1"] - 1) for m in self.months])
        self.off = np.r_[0, np.cumsum([m["m1"] - m["m0"] for m in self.months])]
        self.hrs = torch.from_numpy(hrs).to(self.dev)
        self.A = ctx["mask"].shape[0]
        self.mask = torch.from_numpy(np.ascontiguousarray(ctx["mask"][:, hrs].T)).to(self.dev)   # (H, A)
        self.nnz = int(self.mask.sum())

    def rows(self, s):
        return s[:, self.hrs].T                                     # (H, A) float64, NaN off the mask

    def raw(self, s):
        """The signal itself on the mask positions (selections identical to vm_select_gpu)."""
        return self.rows(s)[self.mask]

    def rankz(self, s):
        """Rank z on the mask positions, float64 (NaN where the formula is not finite)."""
        X = self.rows(s)
        fin = torch.isfinite(X)
        key = torch.where(fin, X, torch.full_like(X, INF)).contiguous()
        srt = torch.sort(key, dim=1).values.contiguous()
        lo = torch.searchsorted(srt, key, right=False)
        hi = torch.searchsorted(srt, key, right=True)
        r = torch.where(fin, (lo + hi + 1).to(torch.float64) / 2.0, torch.zeros_like(X))
        n = fin.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float64)
        d = torch.where(fin, r - r.sum(dim=1, keepdim=True) / n, torch.zeros_like(X))
        sd = torch.sqrt((d * d).sum(dim=1, keepdim=True) / n)
        z = torch.where(sd > 0, d / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(X))
        return torch.where(fin, z, torch.full_like(X, NAN))[self.mask]

    def composite(self, vals, signs=None):
        """vals: list of (nnz,) tensors (any float dtype) -> (nnz,) float64; members folded in list order."""
        acc = torch.zeros(self.nnz, dtype=torch.float64, device=self.dev)
        anyf = torch.zeros(self.nnz, dtype=torch.bool, device=self.dev)
        for k, v in enumerate(vals):
            v = v.to(device=self.dev, dtype=torch.float64)
            f = torch.isfinite(v)
            sg = 1.0 if signs is None else float(signs[k])
            acc = acc + torch.where(f, v * sg if sg != 1.0 else v, torch.zeros_like(v))
            anyf = anyf | f
        return torch.where(anyf, acc, torch.full_like(acc, NAN))

    def select(self, vals):
        """(nnz,) values -> per month of the layout (hi, lo, n), S.select_month's rule (vm_select_gpu's loop)."""
        full = torch.full((int(self.off[-1]), self.A), NAN, dtype=torch.float64, device=self.dev)
        full[self.mask] = vals.to(device=self.dev, dtype=torch.float64)
        inf = torch.tensor(INF, dtype=torch.float64, device=self.dev)
        out = []
        for k in range(len(self.months)):
            Sm = full[int(self.off[k]):int(self.off[k + 1])].clone()
            Sm[0] = NAN
            fin = torch.isfinite(Sm)
            lo = torch.argsort(torch.where(fin, Sm, inf) + 0.0, dim=1, stable=True)[:, :S.K]
            hi = torch.argsort(torch.where(fin, -Sm, inf) + 0.0, dim=1, stable=True)[:, :S.K]
            out.append((hi.to(torch.int16).cpu().numpy(), lo.to(torch.int16).cpu().numpy(),
                        fin.sum(dim=1).to(torch.int16).cpu().numpy()))
        return out


def select_signal(view, s, halves=("A",)):
    """vm_select_gpu's month loop on an already masked (A, T) signal (same lines): -> per month (hi, lo, n)."""
    dev = s.device
    inf = torch.tensor(INF, dtype=torch.float64, device=dev)
    out = []
    with torch.no_grad():
        for m in view["months"]:
            if m["half"] not in halves:
                continue
            Sm = s[:, m["m0"] - 1:m["m1"] - 1].T.clone()
            Sm[0] = NAN
            fin = torch.isfinite(Sm)
            lo = torch.argsort(torch.where(fin, Sm, inf) + 0.0, dim=1, stable=True)[:, :S.K]
            hi = torch.argsort(torch.where(fin, -Sm, inf) + 0.0, dim=1, stable=True)[:, :S.K]
            out.append((hi.to(torch.int16).cpu().numpy(), lo.to(torch.int16).cpu().numpy(),
                        fin.sum(dim=1).to(torch.int16).cpu().numpy()))
    return out


class Pool:
    """AlphaGen pool (evening item 5 with the correction). members: dicts body / z (layout tensor) / cost0 (A months,
    the member's own t_dd book) / hash / expr / single. score = t_dd of the members' composite (0.0 empty).
    tdd(list of z lists) -> list of scores: the caller's sweep (composite -> select -> sel1_sweep, rescored)."""

    def __init__(self, n, layout, tdd, rho=0.7):
        self.n, self.layout, self.tdd, self.rho_max = int(n), layout, tdd, float(rho)
        self.members, self.score = [], 0.0

    def sets_for(self, f, rho, j):
        zs = [m["z"] for m in self.members]
        if rho > self.rho_max:
            return "rep", zs[:j] + [f["z"]] + zs[j + 1:]
        return "add", zs + [f["z"]]

    def rho_of(self, f):
        best, j = -INF, None
        for k, m in enumerate(self.members):
            c = month_corr(f["cost0"], m["cost0"])
            if np.isfinite(c) and c > best:
                best, j = c, k
        return best, j

    def evaluate(self, cands):
        """cands: list of dicts (body, z, cost0, hash) -> list of dict(gain, act, j, rho, T) or 'POOL_SAME' /
        'POOL_NONFINITE'."""
        hashes = {m["hash"] for m in self.members}
        out, jobs = [None] * len(cands), []
        for i, f in enumerate(cands):
            if f["hash"] in hashes:
                out[i] = "POOL_SAME"
                continue
            rho, j = self.rho_of(f)
            act, zs = self.sets_for(f, rho, j)
            out[i] = dict(act=act, j=j if act == "rep" else None, rho=None if j is None else rho)
            jobs.append((i, zs))
        ts = self.tdd([zs for _, zs in jobs]) if jobs else []
        for (i, _), t in zip(jobs, ts):
            if not np.isfinite(t):
                out[i] = "POOL_NONFINITE"
            else:
                out[i].update(T=float(t), gain=float(t) - self.score)
        return out

    def execute(self, f, ev):
        """Apply candidate f's action (ev from evaluate). A full pool drops the one of n + 1 whose removal leaves the
        highest t_dd (ties to the first; the newcomer is last). -> dict of what happened."""
        if ev["act"] == "rep":
            old = self.members[ev["j"]]
            self.members[ev["j"]] = f
            self.score = ev["T"]
            return dict(act="rep", out=old["expr"], into=f["expr"], score=self.score)
        self.members.append(f)
        if len(self.members) <= self.n:
            self.score = ev["T"]
            return dict(act="add", into=f["expr"], score=self.score)
        zs = [m["z"] for m in self.members]
        ts = self.tdd([zs[:k] + zs[k + 1:] for k in range(len(zs))])
        k = int(np.nanargmax(np.where(np.isfinite(ts), ts, -INF)))
        gone = self.members.pop(k)
        self.score = float(ts[k])
        return dict(act="add_drop", into=f["expr"], out=gone["expr"], dropped_newcomer=k == len(zs) - 1, score=self.score)

    def state(self):
        return dict(n=self.n, score=self.score, members=[dict((k, m[k]) for k in ("body", "cost0", "hash", "expr", "single", "first_s") if k in m)
                                                         for m in self.members])


def month_corr(a, b):
    """Pearson correlation of two monthly cost0 lists; nan (no evidence) when either is constant."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
