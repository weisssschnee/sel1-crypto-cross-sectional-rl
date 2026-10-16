"""XS_SEL1 GFlowNet step (supplement 10-15 afternoon item 3). New file (it was a plain copy of sel1_trainer.py until
the item was written; the trainer is sel1_trainer_x.py --gfn BETA). Generator, grammar mask and scoring are the
trainer's, unchanged; only the update differs.

  trajectory balance  the formula is generated token by token, so every state has one parent (P_B = 1) and
                      loss_i = (log Z + sum_t log pi(a_t | s_t) - BETA * R_i)^2, mean over the batch.
                      log pi = log_softmax of the stored effective logits (grammar / legality mask applied, padding
                      after END = force-END logits, log pi 0); R_i = the --reward score (t_dd), an invalid formula
                      R = INVALID_R = -3. log Z is a learned scalar with its own Adam (LR_LOGZ = 0.1); on the first
                      batch it is set to mean(BETA R_i - sum log pi_i) (residual mean 0), then learned.
  beta                BETA = --gfn x BETA_MULT (4): t_dd's in-batch std is ~0.48, so at BETA 1 the target is near
                      uniform (supplement 10-16 morning). The effective BETA is printed on the first update.
  step                one backward through the policy and log Z; policy gradient clipped at 5.0 (as dense_update);
                      a non-finite gradient skips the step for both.
    python -I sel1_gfn.py selftest     # 3 hand examples, log Z gradient and first-batch init, BETA x 4, toy TV (CPU)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

INVALID_R = -3.0
LR_LOGZ = 0.1
BETA_MULT = 4.0
LAST_BETA = None                     # the BETA of the last log_rewards call (printed on log Z init)
CLIP = 5.0


def _dso():
    from experiments.dso_reference.dso_math import sequence_neglogp_entropy
    return sequence_neglogp_entropy


def log_rewards(score, valid, beta):
    """BETA * R per row, BETA = beta x BETA_MULT, R = score where valid, INVALID_R otherwise (float64)."""
    r = np.where(np.asarray(valid, bool), np.nan_to_num(np.asarray(score, np.float64), nan=INVALID_R), INVALID_R)
    global LAST_BETA
    LAST_BETA = effective_beta(beta)
    return LAST_BETA * r


def effective_beta(beta):
    return float(beta) * BETA_MULT


def tb_loss(logits, tokens, log_z, logr):
    """logits (B, T, V) effective logits, tokens (B, T), log_z scalar tensor, logr (B,) -> (loss, residuals, neglogp,
    entropy). sequence_neglogp_entropy over the full width, as dense_update calls it."""
    lengths = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device)
    neglogp, entropy = _dso()(logits, tokens, lengths, 0.7)
    resid = log_z - neglogp - torch.as_tensor(logr, dtype=neglogp.dtype, device=neglogp.device)
    return torch.mean(resid * resid), resid, neglogp, entropy


class LogZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool))


def init_logz(logz, neglogp, logr):
    """First batch only: log Z = mean(BETA R_i - sum log pi_i) = mean(logr + neglogp). -> the value set, or None."""
    if bool(logz.inited):
        return None
    with torch.no_grad():
        v = torch.mean(neglogp.detach().to(torch.float64) + torch.as_tensor(logr, dtype=torch.float64,
                                                                          device=neglogp.device))
        logz.v.copy_(v.to(logz.v.dtype))
        logz.inited.fill_(True)
    return float(v)


def tb_update(model, optimizer, logz, z_opt, batch, logr, groups):
    tokens, logits = batch.context["tokens"], batch.context["effective_logits"]
    if not bool(logz.inited):
        lengths = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device)
        v = init_logz(logz, _dso()(logits, tokens, lengths, 0.7)[0], logr)
        print("GFN effective BETA %g (--gfn x BETA_MULT %g), LR_LOGZ %g, log Z init %.6f (first batch)"
              % (float("nan") if LAST_BETA is None else LAST_BETA, BETA_MULT, LR_LOGZ, v), flush=True)
    loss, resid, _, entropy = tb_loss(logits, tokens, logz.v, logr)
    optimizer.zero_grad()
    z_opt.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP, error_if_nonfinite=False).detach().item())
    skipped = not (math.isfinite(gn) and logz.v.grad is not None and bool(torch.isfinite(logz.v.grad)))
    if skipped:
        optimizer.zero_grad(set_to_none=True)
        z_opt.zero_grad(set_to_none=True)
    else:
        optimizer.step()
        z_opt.step()
    ent = entropy.detach().cpu().numpy()
    fam_of = np.concatenate([[k] * int(n) for k, (_, n) in enumerate(groups)])
    fam_ent = [float(ent[fam_of == k].mean()) if (fam_of == k).any() else None for k in range(len(groups))]
    r = resid.detach().cpu().numpy()
    upd = dict(pg_loss=float(loss.detach().item()), tb_resid_mean=float(r.mean()), tb_resid_abs=float(np.abs(r).mean()),
               log_z=float(logz.v.detach().item()), mean_entropy=float(ent.mean()), gradient_norm=gn,
               gradient_clip_max_norm=CLIP, skipped_nonfinite_update=skipped)
    return upd, fam_ent


# --------------------------------------------------------------------------- self-check
def _hand(logits, tokens, log_z, logr):
    """Plain numpy: log_softmax per step (-inf masked), sum of log pi of the taken tokens."""
    out = []
    for lg, tk, r in zip(logits, tokens, logr):
        s = 0.0
        for row, a in zip(lg, tk):
            fin = np.isfinite(row)
            m = row[fin].max()
            lse = m + math.log(sum(math.exp(x - m) for x in row[fin]))
            s += row[a] - lse
        out.append(log_z + s - r)
    out = np.asarray(out)
    return float(np.mean(out ** 2)), out


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import m2h_b5_trainer_dbg_memo  # noqa: F401  (sys.path for experiments.dso_reference)
    ninf = -np.inf
    # 3 samples, V = 4 (token 3 = END), T = 3; row 2 ends at step 1, then force-END padding (log pi 0)
    logits = np.array([
        [[0.5, -1.0, ninf, 0.2], [1.5, 0.0, 0.3, ninf], [ninf, ninf, ninf, 0.0]],
        [[2.0, 0.0, 0.0, ninf], [0.1, 0.2, 0.3, 0.4], [ninf, 1.0, -2.0, 0.5]],
        [[-0.3, 0.7, 1.1, ninf], [0.0, ninf, ninf, 0.9], [ninf, ninf, ninf, 0.0]]], dtype=np.float64)
    tokens = np.array([[0, 2, 3], [1, 3, 1], [2, 3, 3]])
    log_z, logr = 0.4, np.array([1.2, -3.0, 0.25])
    lz = torch.nn.Parameter(torch.tensor(log_z, dtype=torch.float64))
    loss, resid, _, _ = tb_loss(torch.tensor(logits), torch.tensor(tokens), lz, logr)
    loss.backward()
    hand, per = _hand(logits, tokens, log_z, logr)                 # per = hand residuals
    g_hand = float(2.0 * per.mean())                               # d/dlogZ mean(resid^2)
    res = resid.detach().numpy()
    ok_loss = abs(float(loss) - hand) <= 1e-12 * max(1.0, abs(hand))
    ok_each = bool(np.allclose(res, per, rtol=0, atol=1e-12))
    ok_grad = lz.grad is not None and abs(float(lz.grad) - g_hand) <= 1e-12 and abs(float(lz.grad)) > 0
    # first-batch init: log Z = mean(logr + neglogp) = hand mean(logr - sum log pi), so the residual mean is 0
    z = LogZ().to(torch.float64)
    _, _, nlp, _ = tb_loss(torch.tensor(logits), torch.tensor(tokens), z.v, logr)
    v = init_logz(z, nlp, logr)
    v_hand = float(np.mean(logr - (per - log_z + logr)))           # per - log_z + logr = sum log pi (hand)
    _, r0, _, _ = tb_loss(torch.tensor(logits), torch.tensor(tokens), z.v, logr)
    ok_init = (abs(v - v_hand) <= 1e-12 and abs(float(r0.mean())) <= 1e-12 and bool(z.inited)
               and init_logz(z, nlp, logr) is None)
    out = dict(schema="SEL1_GFN_SELFTEST_V1", loss=float(loss), hand=hand, resid_hand=per.tolist(), resid_tb=res.tolist(),
               logz_grad=float(lz.grad), logz_grad_hand=g_hand, ok_loss=bool(ok_loss), ok_each=ok_each, ok_grad=bool(ok_grad),
               logz_init=v, logz_init_hand=v_hand, ok_init=bool(ok_init),
               passed=bool(ok_loss and ok_each and ok_grad and ok_init))
    print(json.dumps(dict((k, out[k]) for k in ("loss", "hand", "logz_grad", "logz_grad_hand", "logz_init",
                                                "logz_init_hand", "passed"))))
    return out




def _toy_tv(steps=500, B=64, seed=0):
    """Toy tree: 2 binary steps -> 4 terminals (one parent each, P_B = 1), R known; BETA = log_rewards(R, 1) =
    BETA_MULT * R. On-policy TB through this module's init_logz / tb_loss, policy Adam 5e-2, log Z Adam LR_LOGZ.
    Pass: TV(policy's terminal distribution, exp(BETA R) / Z) <= 0.05, plus 20k samples' frequencies <= 0.05."""
    g = torch.Generator().manual_seed(seed)
    R = np.array([0.5, 1.2, 0.8, -0.3])
    th1 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    th2 = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    logz = LogZ().to(torch.float64)
    opt = torch.optim.Adam([th1, th2], lr=5e-2)
    z_opt = torch.optim.Adam(logz.parameters(), lr=LR_LOGZ)

    def sample(n):
        with torch.no_grad():
            a1 = torch.multinomial(torch.softmax(th1, 0).expand(n, 2), 1, generator=g)[:, 0]
            a2 = torch.multinomial(torch.softmax(th2[a1], 1), 1, generator=g)[:, 0]
        return torch.stack([a1, a2], 1)

    init = None
    for _ in range(steps):
        tk = sample(B)
        logits = torch.stack([th1.expand(B, 2), th2[tk[:, 0]]], 1)                       # (B, 2, V=2)
        logr = log_rewards(R[(2 * tk[:, 0] + tk[:, 1]).numpy()], np.ones(B, bool), 1.0)
        if init is None:
            lengths = torch.full((B,), 2, dtype=torch.long)
            init = init_logz(logz, _dso()(logits, tk, lengths, 0.7)[0], logr)
        loss, _, _, _ = tb_loss(logits, tk, logz.v, logr)
        opt.zero_grad()
        z_opt.zero_grad()
        loss.backward()
        opt.step()
        z_opt.step()
    with torch.no_grad():
        p1, p2 = torch.softmax(th1, 0), torch.softmax(th2, 1)
        pol = (p1[:, None] * p2).reshape(-1).numpy()
    tk = sample(20000)
    f = np.bincount((2 * tk[:, 0] + tk[:, 1]).numpy(), minlength=4) / 20000.0
    tar = np.exp(BETA_MULT * R) / np.exp(BETA_MULT * R).sum()
    tv, tv_s = float(0.5 * np.abs(pol - tar).sum()), float(0.5 * np.abs(f - tar).sum())
    logz_true = float(np.log(np.exp(BETA_MULT * R).sum()))
    return dict(passed=bool(tv <= 0.05 and tv_s <= 0.05), tv=tv, tv_samples=tv_s, policy=pol.tolist(),
                freq=f.tolist(), target=tar.tolist(), beta=LAST_BETA, logz_init=init,
                logz_final=float(logz.v), logz_true=logz_true)


if __name__ == "__main__" and sys.argv[1:] == ["selftest"]:
    r = selftest()
    ok_beta = effective_beta(1) == 4.0 and LR_LOGZ == 0.1
    toy = _toy_tv()
    print(json.dumps(dict((k, toy[k]) for k in ("passed", "tv", "tv_samples", "beta", "logz_init", "logz_final",
                                                "logz_true", "target", "freq"))), "beta_x4_lr", ok_beta)
    sys.exit(0 if r["passed"] and toy["passed"] and ok_beta else 7)
