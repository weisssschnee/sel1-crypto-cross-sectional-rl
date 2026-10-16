"""XS_SEL1 searcher fixes, ARMS v2 (order supplement 10-16 noon). New file: the pieces sel1_trainer_x2 and sel1_ga2
use. Nothing here is called on a path whose switches are off, so V2 / ARMS v1 behaviour is untouched.

  sampler     sample_g = m2h_b5_trainer_dbg_memo.sample_batch_g with two options: explore e (each step draws from
              (1 - e) softmax(effective) + e uniform over the legal tokens; the stored effective logits stay the
              policy's own, so TB / PG use log pi) and force (teacher forcing of given bodies through the same
              legality table + grammar mask, for replay / priority-queue rows). explore 0 and no force is the
              original function itself (sample_sweep calls it), so the draw is bit-identical.
  PG risk     risk_rows: one row per body (its first valid row in the batch, plus the hall rows), DSO
              select_risk_elite on those rewards (quantile 1 - eps, method "higher"), elite = r >= q, advantage
              r - q (DSO risk-seeking PG, baseline = the (1 - eps) quantile). dense_update_g = dbg_memo.dense_update
              with the entropy gamma as a parameter: gamma 0.7 is DSO's hierarchical entropy regularizer (gamma^t step
              weights, Landajuela et al. 2021) that dense_update already uses; gamma 1.0 = flat entropy.
  GFN         LogZB: one log Z per basin (T1 / T2 / T3), first-batch init per basin to mean(log R + neglogp) of its
              rows; tb_loss_b: (log Z_b(i) + sum log pi_i - log R_i)^2, mean; replay rows are scored with the policy's
              log pi too (off-policy TB).
  toy         toy_tv: two basins with different reward tables (different partition functions), basin-conditioned
              policy, explore 0.05, replay of the best seen terminals, per-basin log Z -> TV per basin <= 0.05. The
              same run with ONE shared log Z must fail (TV > 0.05): that is the structural defect the toy must see.
    python -I sel1_search2.py selftest      # CPU: risk hand check, explore / force parity, toy TV (both variants)
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import m2h_b5_trainer_dbg_memo as TB
import m2h_common as M

CLIP = 5.0


# --------------------------------------------------------------------------- sampler
def explore_probs(effective, eps):
    """(1 - eps) softmax(effective) + eps uniform over the finite (legal) entries of each row."""
    p = torch.softmax(effective, dim=-1)
    if not eps:
        return p
    fin = torch.isfinite(effective).to(p.dtype)
    return (1.0 - eps) * p + eps * fin / fin.sum(dim=-1, keepdim=True)


def sample_g(model, vm, table, batch_size, *, device, explore=0.0, force=None, generator=None):
    """sample_batch_g's lines; explore > 0 mixes uniform legal draws in; force = list of bodies (END included) replays
    them (batch_size = len(force)). Raises when a forced token is illegal under the mask."""
    if not explore and force is None:
        return TB.sample_batch_g(model, vm, table, batch_size, device=device, generator=generator)
    device = torch.device(device)
    if table.mask.device != device:
        table = table.to(device)
    if force is not None:
        batch_size = len(force)
    max_len = table.max_len
    end = vm.end_token
    mf = table.min_finish.cpu().tolist()
    depth = torch.zeros(batch_size, dtype=torch.long, device=device)
    window = torch.zeros(batch_size, dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
    inp = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    token_steps, logit_steps = [], []
    force_end = torch.full((vm.token_count,), float("-inf"), dtype=table.mask.dtype, device=device)
    force_end[end] = 0.0
    arity_delta = torch.tensor([1 - TB._ARITY[n] for n, _ in TB.OPERATOR_ARITIES], dtype=torch.long, device=device)
    rows = [[] for _ in range(batch_size)]
    done_py = [False] * batch_size
    for step in range(max_len):
        remaining = torch.full_like(depth, max_len - step)
        legal = table.mask[depth, window, remaining]
        legal = torch.where(done[:, None], force_end[None, :], legal)
        fin = torch.isfinite(legal).cpu().numpy()
        gm = np.zeros(fin.shape, dtype=np.float32)
        for i in range(batch_size):
            if not done_py[i]:
                gm[i] = TB._g_mask_row(rows[i], fin[i], max_len - step, vm, mf)
        legal = legal + torch.from_numpy(gm).to(device=device, dtype=legal.dtype)
        if not bool(torch.isfinite(legal).any(dim=1).all()):
            raise RuntimeError("grammar sampler: no legal action at step %d" % step)
        logits, _, _ = model(inp)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        effective = logits.to(table.mask.dtype) + legal
        if force is not None:
            action = torch.tensor([int(f[step]) if step < len(f) else end for f in force], dtype=torch.long, device=device)
            ok = torch.isfinite(effective[torch.arange(batch_size, device=device), action])
            if not bool(ok.all()):
                raise RuntimeError("forced token illegal at step %d rows %s" % (step, torch.nonzero(~ok).flatten()[:8].tolist()))
        else:
            action = torch.multinomial(explore_probs(effective, explore), 1, generator=generator).squeeze(1)
        token_steps.append(action)
        logit_steps.append(effective)
        newly_done = (~done) & (action == end)
        lengths = torch.where(newly_done, torch.full_like(lengths, step + 1), lengths)
        is_term = (action < vm.feature_count) & ~done
        is_win = (action >= vm.window_base) & (action < end) & ~done
        is_op = (action >= vm.operator_base) & (action < vm.window_base) & ~done
        delta = torch.where(is_term, torch.ones_like(depth), torch.zeros_like(depth))
        if bool(is_op.any()):
            slot = (action - vm.operator_base).clamp_(0, arity_delta.shape[0] - 1)
            delta = torch.where(is_op, arity_delta[slot], delta)
        depth = depth + delta
        window = torch.where(is_win, torch.ones_like(window), torch.zeros_like(window))
        for i, tk in enumerate(action.cpu().tolist()):
            if not done_py[i]:
                if tk == end:
                    done_py[i] = True
                else:
                    rows[i], _ = TB._g_step(rows[i], tk, vm)
        done = done | (action == end)
        inp = torch.cat([inp, action[:, None]], dim=1)
        if bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError("grammar sampler exhausted max_len without terminating every program")
    tokens = torch.stack(token_steps, dim=1)
    formulas = [tuple(int(t) for t in row[: int(n)].tolist()) for row, n in zip(tokens, lengths.cpu())]
    return TB._Batch(formulas=formulas, context={"tokens": tokens, "effective_logits": torch.stack(logit_steps, dim=1),
                                                 "lengths": lengths, "max_len": max_len})


def view(model, vm, b):
    return TB.BasinView(model, vm.token_count, TB.basin_token(vm, b))


def sample_sweep(model, vm, tables, counts, device, explore=0.0):
    """sel1_trainer_x.sample(sweep=True) with the explore option (0 -> the same calls, the same draw)."""
    subs = [sample_g(view(model, vm, b), vm, tab, int(n), device=device, explore=explore)
            for b, tab, n in zip(M.BASIN_NAMES, tables, counts)]
    tokens, el, lengths = TB._merge_subs(subs, vm, device)
    formulas = [tuple(int(t) for t in row[:n]) for row, n in zip(tokens.tolist(), lengths.tolist())]
    return TB._Batch(formulas=formulas, context=dict(tokens=tokens, effective_logits=el, lengths=lengths,
                                                     max_len=tokens.shape[1]))


def force_rows(model, vm, tables, bodies, basins, device):
    """Teacher-forced rows for hall bodies (END included), grouped by basin in M.BASIN_NAMES order.
    -> (_Batch, order: list of the input indices in row order, basin index per row)."""
    subs, order, bidx = [], [], []
    for k, (b, tab) in enumerate(zip(M.BASIN_NAMES, tables)):
        ii = [i for i, x in enumerate(basins) if x == b]
        if not ii:
            continue
        subs.append(sample_g(view(model, vm, b), vm, tab, len(ii), device=device, force=[list(bodies[i]) for i in ii]))
        order += ii
        bidx += [k] * len(ii)
    tokens, el, lengths = TB._merge_subs(subs, vm, device)
    return TB._Batch(formulas=None, context=dict(tokens=tokens, effective_logits=el, lengths=lengths)), order, np.asarray(bidx)


def cat_rows(a, b, vm, device):
    """Rows of batch a then batch b (step axis padded as _merge_subs pads) -> (tokens, effective_logits)."""
    t, e, _ = TB._merge_subs([a, b], vm, device)
    return t, e


# --------------------------------------------------------------------------- PG risk-seeking
def risk_rows(rew, valid, keys, eps):
    """-> (elite row indices, advantage r - q, q, number of distinct rows). One row per key (first valid one)."""
    rew = np.asarray(rew, np.float64)
    first = {}
    for i in np.flatnonzero(np.asarray(valid, bool)):
        first.setdefault(keys[i], int(i))
    u = np.array(sorted(first.values()), dtype=np.int64)
    if u.size == 0:
        return u, np.zeros(0), float("nan"), 0
    sel = TB.select_risk_elite(rew[u], float(eps))
    idx = u[sel.elite_indices]
    return idx, rew[idx] - sel.baseline, float(sel.quantile), int(u.size)


def dense_update_g(model, optimizer, tokens, logits, idx_np, adv_np, fam_of, n_groups, device, ent_coef=0.03, gamma=0.7):
    """dbg_memo.dense_update's step on given rows (tokens / logits may include forced rows) with entropy gamma."""
    idx = torch.as_tensor(np.asarray(idx_np), dtype=torch.long, device=device)
    tk = tokens.index_select(0, idx)
    lg = logits.index_select(0, idx)
    advantage = torch.as_tensor(np.asarray(adv_np), dtype=torch.float32, device=device)
    lengths = torch.full((len(idx_np),), tk.shape[1], dtype=torch.long, device=device)
    neglogp, entropy = TB.sequence_neglogp_entropy(lg, tk, lengths, float(gamma))
    pg_loss = torch.mean(advantage * neglogp)
    entropy_loss = -ent_coef * torch.mean(entropy)
    loss = pg_loss + entropy_loss
    optimizer.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP, error_if_nonfinite=False).detach().item())
    skipped = not math.isfinite(gn)
    if skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
    ent = entropy.detach().cpu().numpy()
    fo = np.asarray(fam_of)[np.asarray(idx_np)]
    fam_ent = [float(ent[fo == k].mean()) if (fo == k).any() else None for k in range(n_groups)]
    return dict(elite_count=int(len(idx_np)), pg_loss=float(pg_loss.detach().item()),
                entropy_loss=float(entropy_loss.detach().item()), mean_entropy=float(ent.mean()), gradient_norm=gn,
                gradient_clip_max_norm=CLIP, skipped_nonfinite_update=skipped), fam_ent


# --------------------------------------------------------------------------- GFN, one log Z per basin
class LogZB(torch.nn.Module):
    def __init__(self, n=len(M.BASIN_NAMES)):
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(n))
        self.register_buffer("inited", torch.zeros(n, dtype=torch.bool))


def _neglogp(logits, tokens):
    lengths = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device)
    return TB.sequence_neglogp_entropy(logits, tokens, lengths, 0.7)


def init_logzb(logz, neglogp, logr, bidx):
    """Per basin not yet set and present in the rows: log Z_b = mean(logr + neglogp) of its rows. -> {b: value}."""
    out = {}
    with torch.no_grad():
        x = neglogp.detach().to(torch.float64).cpu().numpy() + np.asarray(logr, np.float64)
        for b in np.unique(bidx):
            if not bool(logz.inited[b]):
                v = float(x[bidx == b].mean())
                logz.v[b] = v
                logz.inited[b] = True
                out[int(b)] = v
    return out


def tb_loss_b(logits, tokens, logz_v, bidx, logr):
    neglogp, entropy = _neglogp(logits, tokens)
    lz = logz_v[torch.as_tensor(np.asarray(bidx), dtype=torch.long, device=logz_v.device)].to(neglogp.dtype)
    resid = lz - neglogp - torch.as_tensor(np.asarray(logr), dtype=neglogp.dtype, device=neglogp.device)
    return torch.mean(resid * resid), resid, neglogp, entropy


def tb_update_b(model, optimizer, logz, z_opt, tokens, logits, bidx, logr, fam_of, n_groups, n_onpolicy):
    """One TB step over all rows (on-policy first, replay after); entropy / residual stats on the on-policy rows."""
    if not bool(logz.inited.all()):
        v = init_logzb(logz, _neglogp(logits, tokens)[0], logr, bidx)
        if v:
            print("GFN2 log Z init per basin %s (first batch)" % {M.BASIN_NAMES[k]: round(x, 6) for k, x in v.items()},
                  flush=True)
    loss, resid, _, entropy = tb_loss_b(logits, tokens, logz.v, bidx, logr)
    optimizer.zero_grad()
    z_opt.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP, error_if_nonfinite=False).detach().item())
    skipped = not (math.isfinite(gn) and logz.v.grad is not None and bool(torch.isfinite(logz.v.grad).all()))
    if skipped:
        optimizer.zero_grad(set_to_none=True)
        z_opt.zero_grad(set_to_none=True)
    else:
        optimizer.step()
        z_opt.step()
    ent = entropy.detach().cpu().numpy()[:n_onpolicy]
    fo = np.asarray(fam_of)[:n_onpolicy]
    fam_ent = [float(ent[fo == k].mean()) if (fo == k).any() else None for k in range(n_groups)]
    r = resid.detach().cpu().numpy()
    upd = dict(pg_loss=float(loss.detach().item()), tb_resid_mean=float(r[:n_onpolicy].mean()),
               tb_resid_abs=float(np.abs(r[:n_onpolicy]).mean()),
               tb_resid_abs_replay=float(np.abs(r[n_onpolicy:]).mean()) if r.size > n_onpolicy else None,
               log_z={M.BASIN_NAMES[k]: float(x) for k, x in enumerate(logz.v.detach().cpu().tolist())},
               mean_entropy=float(ent.mean()), gradient_norm=gn, gradient_clip_max_norm=CLIP, skipped_nonfinite_update=skipped)
    return upd, fam_ent


# --------------------------------------------------------------------------- self-checks
def hand_risk(rew, valid, keys, eps):
    """Independent hand rule: distinct keys' first valid rewards, sorted; q = v[ceil((n - 1)(1 - eps))]."""
    seen, rows = set(), []
    for i in range(len(rew)):
        if valid[i] and keys[i] not in seen:
            seen.add(keys[i])
            rows.append(i)
    v = sorted(rew[i] for i in rows)
    q = v[int(math.ceil((len(v) - 1) * (1.0 - eps)))]
    el = [i for i in rows if rew[i] >= q]
    return el, [rew[i] - q for i in el], q


def check_risk():
    rng = np.random.default_rng(3)
    n = 150
    rew = np.round(rng.normal(0.5, 1.0, n), 3)
    keys = [int(k) for k in rng.integers(0, 110, n)]                      # duplicates on purpose
    for i in range(n):                                                     # a duplicate body has the same reward
        j = keys.index(keys[i])
        rew[i] = rew[j]
    valid = rng.random(n) > 0.15
    idx, adv, q, nu = risk_rows(rew, valid, keys, 0.05)
    el, ah, qh = hand_risk(rew, valid, keys, 0.05)
    ok = list(idx) == el and np.allclose(adv, ah, rtol=0, atol=0) and q == qh and len(set(keys[i] for i in idx)) == len(idx)
    return dict(passed=bool(ok), n_rows=n, n_distinct=nu, q=q, q_hand=qh, elite=list(map(int, idx)), elite_hand=el,
                adv=[float(x) for x in adv], adv_hand=ah)


def toy_tv(shared=False, steps=800, B=64, eps=0.05, n_replay=8, seed=0):
    """Two basins, 2-step binary tree each (4 terminals, one parent each), basin-conditioned policy, explore eps,
    replay of the n_replay best (basin, terminal) seen so far, TB with log Z per basin (shared=True: one log Z)."""
    beta = 4.0
    g = torch.Generator().manual_seed(seed)
    R = np.array([[0.5, 1.2, 0.8, -0.3], [1.6, 1.1, 1.9, 0.4]])
    th1 = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    th2 = torch.nn.Parameter(torch.zeros(2, 2, 2, dtype=torch.float64))
    logz = LogZB(1 if shared else 2).to(torch.float64)
    opt = torch.optim.Adam([th1, th2], lr=5e-2)
    z_opt = torch.optim.Adam(logz.parameters(), lr=0.1)
    best = {}

    def logits_of(b, a1):
        return torch.stack([th1[b], th2[b, a1]], 1)                       # (n, 2, 2)

    def draw(b, e):
        with torch.no_grad():
            a1 = torch.multinomial(explore_probs(th1[b], e), 1, generator=g)[:, 0]
            a2 = torch.multinomial(explore_probs(th2[b, a1], e), 1, generator=g)[:, 0]
        return torch.stack([a1, a2], 1)

    for _ in range(steps):
        b = torch.tensor([0] * B + [1] * B)
        tk = draw(b, eps)
        term = (2 * tk[:, 0] + tk[:, 1]).numpy()
        for bb, t in zip(b.tolist(), term.tolist()):
            best[(bb, t)] = R[bb, t]
        rp = sorted(best, key=lambda k: -best[k])[:n_replay]
        if rp:
            rb = torch.tensor([k[0] for k in rp])
            rt = torch.tensor([[k[1] // 2, k[1] % 2] for k in rp])
            b, tk = torch.cat([b, rb]), torch.cat([tk, rt])
            term = np.concatenate([term, [k[1] for k in rp]])
        logits = logits_of(b, tk[:, 0])
        logr = beta * R[b.numpy(), term]
        bi = np.zeros(len(b), np.int64) if shared else b.numpy()
        if not bool(logz.inited.all()):
            init_logzb(logz, _neglogp(logits, tk)[0], logr, bi)
        loss, _, _, _ = tb_loss_b(logits, tk, logz.v, bi, logr)
        opt.zero_grad()
        z_opt.zero_grad()
        loss.backward()
        opt.step()
        z_opt.step()
    out = dict(shared=shared, tv={}, tv_samples={})
    for bb in (0, 1):
        with torch.no_grad():
            pol = (torch.softmax(th1[bb], 0)[:, None] * torch.softmax(th2[bb], 1)).reshape(-1).numpy()
        tk = draw(torch.full((20000,), bb), 0.0)
        f = np.bincount((2 * tk[:, 0] + tk[:, 1]).numpy(), minlength=4) / 20000.0
        tar = np.exp(beta * R[bb]) / np.exp(beta * R[bb]).sum()
        out["tv"][M.BASIN_NAMES[bb]] = float(0.5 * np.abs(pol - tar).sum())
        out["tv_samples"][M.BASIN_NAMES[bb]] = float(0.5 * np.abs(f - tar).sum())
    out["logz"] = logz.v.detach().tolist()
    out["logz_true"] = [float(np.log(np.exp(beta * R[bb]).sum())) for bb in (0, 1)]
    out["max_tv"] = max(max(out["tv"].values()), max(out["tv_samples"].values()))
    return out


def selftest():
    risk = check_risk()
    per = toy_tv(shared=False)
    one = toy_tv(shared=True)
    ok_toy = per["max_tv"] <= 0.05
    sees = one["max_tv"] > 0.05                                           # the toy must catch a shared log Z
    rep = dict(schema="SEL1_SEARCH2_SELFTEST_V1", risk=risk, toy_per_basin=per, toy_shared_logz=one,
               toy_passed=bool(ok_toy), toy_detects_shared_logz=bool(sees),
               passed=bool(risk["passed"] and ok_toy and sees))
    print(json.dumps(dict(risk_ok=risk["passed"], q=risk["q"], q_hand=risk["q_hand"], elite=len(risk["elite"]),
                          toy_tv_per_basin=per["max_tv"], toy_logz=per["logz"], toy_logz_true=per["logz_true"],
                          toy_tv_shared=one["max_tv"], passed=rep["passed"])), flush=True)
    return rep


if __name__ == "__main__" and sys.argv[1:2] == ["selftest"]:
    r = selftest()
    if len(sys.argv) > 2:
        json.dump(r, open(sys.argv[2], "w", encoding="utf-8"), indent=1, default=float)
    sys.exit(0 if r["passed"] else 7)
