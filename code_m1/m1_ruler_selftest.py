"""m1_ruler self-test (no data, no VM): the new ruler must reduce EXACTLY to the old
one when nothing new is switched on, and every new piece must match hand math.
Run from the directory holding m1_common.py / m1_ruler.py / stateful_action_policy.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import m1_common as C
import m1_ruler as R
import stateful_action_policy as P

ok = True
FLAT = {k: v for k, v in R.cost_specs().items() if v[0] == "flat"}
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (("  " + extra) if extra else ""))

rng = np.random.default_rng(11)
T, A = 900, 30

# ---- 1. flat6 settle_multi == C.settle_episode, with and without data gaps ----
for gaps in (0.0, 0.15):
    fwd = rng.normal(0, 3e-4, (T, A)); fwd[rng.random((T, A)) < gaps] = np.nan
    el = np.isfinite(fwd)
    sig = rng.normal(size=(T, A))
    desired = C.rank_weights(sig, el)
    # old world (policy sees real eligibility) -- the pre-existing behaviour
    held_old = np.asarray(P.apply_stateful_action_policy(
        desired_weights=desired, eligible=el,
        contract=P.StatefulActionContract(minimum_hold_hours=15))[0]).reshape(T, A)
    fm = rng.random((T, A)) < 0.002; fr = rng.normal(0, 1e-4, (T, A))
    old = C.settle_episode(held_old, fwd, fm, fr, C.FEE_RATE, C.SLIPPAGE_RATE)
    new, parts = R.settle_multi(held_old, fwd, fm, fr, FLAT)
    check(f"flat6 == C.settle_episode (gaps={gaps})", abs(new["flat6"]["log_nav"] - old[0]) < 1e-12,
          f"{new['flat6']['log_nav']:.10f} vs {old[0]:.10f}")
    check(f"turnover/gross/funding identical (gaps={gaps})",
          abs(parts["turnover"] - old[1]["turnover"]) < 1e-9 and abs(parts["gross"] - old[1]["gross"]) < 1e-12
          and abs(parts["funding"] - old[1]["funding"]) < 1e-12)
    check(f"cost0 >= maker2 >= flat6 (gaps={gaps})",
          new["cost0"]["log_nav"] >= new["maker2"]["log_nav"] >= new["flat6"]["log_nav"])

# ---- 2. rule 0: gaps no longer force retargets; position frozen through the gap ----
fwd = rng.normal(0, 3e-4, (T, A)); fwd[rng.random((T, A)) < 0.15] = np.nan
el = np.isfinite(fwd)
desired = C.rank_weights(rng.normal(size=(T, A)), el)
h_old = np.asarray(P.apply_stateful_action_policy(
    desired_weights=desired, eligible=el,
    contract=P.StatefulActionContract(minimum_hold_hours=240))[0]).reshape(T, A)
h_new = R.held_path(desired, 240, R.BAND, P)
fm0 = np.zeros((T, A), bool); fr0 = np.zeros((T, A))
_, p_old = R.settle_multi(h_old, fwd, fm0, fr0, FLAT)
_, p_new = R.settle_multi(h_new, fwd, fm0, fr0, FLAT)
check("rule 0 removes the gap-driven turnover floor at hold 240",
      p_new["turnover"] < 0.25 * p_old["turnover"], f"turnover {p_old['turnover']:.2f} -> {p_new['turnover']:.2f}")
gross_new = np.abs(h_new).sum(axis=1)
check("rule 0 keeps gross exposure <= 1", gross_new.max() <= 1.0 + 1e-10, f"max gross {gross_new.max():.6f}")
check("rule 0 with no gaps == old policy (identity when nothing is missing)",
      np.array_equal(
          R.held_path(C.rank_weights(rng.normal(size=(T, A)), np.ones((T, A), bool)), 60, R.BAND, P).shape, (T, A)))
full = np.ones((T, A), bool); dsg = C.rank_weights(rng.normal(size=(T, A)), full)
a = np.asarray(P.apply_stateful_action_policy(desired_weights=dsg, eligible=full,
        contract=P.StatefulActionContract(minimum_hold_hours=60))[0]).reshape(T, A)
check("held_path(default band) == contract-default policy when eligibility is all True",
      np.array_equal(a, R.held_path(dsg, 60, 0.5, P)))
# a held asset whose bar vanishes: no forced exit charge, no ghost P&L
f2 = np.zeros((10, 3)); f2[:, :] = 1e-3
f2[5:8, 0] = np.nan                                   # asset 0 goes dark for 3 minutes
w2 = np.zeros((10, 3)); w2[1:, 0] = 0.5; w2[1:, 1] = -0.5
res2, pr2 = R.settle_multi(w2, f2, np.zeros((10, 3), bool), np.zeros((10, 3)), {"flat": ("flat", 0.0006)})
check("frozen-through-gap: only the 2 opening trades are charged", abs(pr2["turnover"] - 1.0) < 1e-12,
      f"turnover {pr2['turnover']}")
check("resume_turn_share reported", "resume_turn_share" in pr2)

# ---- 3. band parameter reaches the policy and is monotone in turnover ----
dsg2 = C.rank_weights(rng.normal(size=(T, A)).cumsum(0) * 0.05 + rng.normal(size=(T, A)), np.ones((T, A), bool))
turn = []
for band in (0.2, 0.5, 1.0, 1.5):
    h = R.held_path(dsg2, 1, band, P)
    turn.append(float(np.abs(np.diff(h, axis=0, prepend=0.0)).sum()))
check("wider hysteresis band never raises turnover", all(x >= y - 1e-9 for x, y in zip(turn, turn[1:])),
      f"turnover by band 0.2/0.5/1.0/1.5 = {[round(x,1) for x in turn]}")

# ---- 4. book walk vs hand integration (linear book, 100k per 1% band) ----
bid = np.broadcast_to(np.array([1e5, 2e5, 3e5, 4e5, 5e5]), (1, 1, 5)).copy()
ask = bid * 2.0
def imp(q, buy):
    return float(R.impact_bps(np.array([[q]]), np.array([[buy]]), bid, ask)[0, 0])
check("sell 50k in band 1 = 25.0bp (bid book)", abs(imp(5e4, False) - 25.0) < 1e-9, f"{imp(5e4, False):.4f}")
check("buy 50k uses the deeper ask book = 12.5bp", abs(imp(5e4, True) - 12.5) < 1e-9, f"{imp(5e4, True):.4f}")
check("sell 150k crosses into band 2 = 75.0bp", abs(imp(1.5e5, False) - 75.0) < 1e-9, f"{imp(1.5e5, False):.4f}")
check("sell 600k runs past the 5% band = 291.667bp", abs(imp(6e5, False) - 1.75e8 / 6e5) < 1e-6, f"{imp(6e5, False):.4f}")
check("q=0 -> 0bp", imp(0.0, True) == 0.0)
nanb = np.full((1, 1, 5), np.nan)
check("NaN depth -> NaN (caller falls back)", np.isnan(R.impact_bps(np.array([[5e4]]), np.array([[True]]), nanb, nanb)[0, 0]))
held1 = np.zeros((6, 1)); held1[2:, 0] = 0.5
f1 = np.zeros((6, 1)); fm1 = np.zeros((6, 1), bool); fr1 = np.zeros((6, 1))
b = np.broadcast_to(bid, (6, 1, 5)).copy()
small, _ = R.settle_multi(held1, f1, fm1, fr1, {"d": ("depth", 0.0005, 1.0, 1e3)}, b, b)
big, _ = R.settle_multi(held1, f1, fm1, fr1, {"d": ("depth", 0.0005, 1.0, 1e6)}, b, b)
check("depth cost grows with NAV", big["d"]["cost"] > small["d"]["cost"] > 0.0005 * 0.5 * 0.99,
      f"1k NAV cost {small['d']['cost']:.3e}  1M NAV cost {big['d']['cost']:.3e}")
try:
    R.settle_multi(held1, f1, fm1, fr1, {"d": ("depth", 0.0005, 1.0, 1e3)})
    check("depth without book arrays raises (no silent flat fallback)", False)
except ValueError:
    check("depth without book arrays raises (no silent flat fallback)", True)

# ---- 5. winsorize ----
x = rng.normal(0, 1e-3, (4, 59)); x[0, 5] = 0.5; x[1, 3] = np.nan; x[3, 9:] = np.nan
w = R.winsorize_cs(x, 1.0)
second = np.sort(x[0])[-2]
check("winsorize keeps NaN positions", np.array_equal(np.isnan(w), np.isnan(x)))
check("winsorize clips the n=59 pump to the 2nd-largest value", abs(w[0, 5] - second) < 1e-15,
      f"pump 0.5 -> {w[0, 5]:.6f} (2nd largest {second:.6f})")
check("winsorize changes only the extremes on a clean row (<=2 cells)", int((w[2] != x[2]).sum()) <= 2)
check("winsorize leaves a 9-live minute alone", np.array_equal(w[3], x[3], equal_nan=True))

# ---- 6. windows: disjoint draws, deterministic tiling, episode covers the longest hold ----
check("episode covers the longest hold", R.EPISODE_MINUTES >= max(C.HOLDS) + 60,
      f"episode {R.EPISODE_MINUTES} min vs max hold {max(C.HOLDS)}")
for n_min in (31 * 1440, 30 * 1440, 28 * 1440):
    E = R.EPISODE_MINUTES
    st = R.disjoint_starts(np.random.default_rng(3), n_min, 12)
    gaps_ok = all(b - a >= E for a, b in zip(st, st[1:]))
    in_ok = st[0] >= C.LOOKBACK_MINUTES and st[-1] + E <= n_min
    tl = R.tile_starts(n_min)
    check(f"disjoint_starts n_min={n_min}: {len(st)} windows, no overlap, inside month", gaps_ok and in_ok,
          f"starts {st}")
    check(f"tile_starts n_min={n_min}: {len(tl)} back-to-back windows inside month",
          all(b - a == E for a, b in zip(tl, tl[1:])) and tl[-1] + E <= n_min)
rr = [tuple(R.disjoint_starts(np.random.default_rng(s), 31 * 1440, 4)) for s in range(40)]
check("disjoint_starts actually varies with the seed", len(set(rr)) > 5, f"{len(set(rr))} distinct of 40")

# ---- 7. aggregation: mean/q25 allowed; second_best / max / median banned ----
arr = [0.3, -0.1, 0.2, 0.9, -0.4, 0.1]
check("mean", abs(R.aggregate(arr, "mean") - np.mean(arr)) < 1e-15)
check("q25", abs(R.aggregate(arr, "q25") - np.quantile(arr, 0.25)) < 1e-15)
for bad in ("second_best", "max", "median"):
    try:
        R.aggregate(arr, bad); check(f"{bad} is banned", False)
    except ValueError:
        check(f"{bad} is banned", True)

# ---- 8. eval_formula end to end with a stub VM (shape/lag/hold/cost plumbing) ----
class _T:
    def __init__(self, a): self.a = a
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
class StubVM:
    def __init__(self, sig): self.sig = sig
    def execute(self, body, sl): return _T(self.sig)
class Inval(Exception): pass
Ael, E, L = 40, R.EPISODE_MINUTES, C.LOOKBACK_MINUTES
n_min = L + 2 * E + 500
shard = dict(month="2099-01", n_min=n_min, symbols=[f"S{i}" for i in range(Ael)],
             fwd=rng.normal(0, 3e-4, (Ael, n_min)), eligible=None,
             fi=np.zeros((Ael, n_min), bool), fr=np.zeros((Ael, n_min)))
shard["fwd"][rng.random((Ael, n_min)) < 0.05] = np.nan
shard["eligible"] = np.isfinite(shard["fwd"])
shard = R.prepare_shard(shard)
check("prepare_shard caches the winsorised grid", shard["fwd_w"].shape == shard["fwd"].shape)
sig_slice = rng.normal(size=(Ael, L + E))
starts = [L, L + E]
sl = [None, None]
res = R.eval_formula(StubVM(sig_slice), (0,), [60, 1440, 10080], sl, shard, starts, P, Inval,
                     costs=("flat6", "maker2"), variants=("win", "raw"))
check("eval_formula returns every hold/variant/cost x window",
      all(len(res["per_hold"][h][v][c]) == 2 for h in (60, 1440, 10080) for v in ("win", "raw") for c in ("flat6", "maker2")))
check("maker2 >= flat6 on every window/hold",
      all(m >= f for h in (60, 1440, 10080) for m, f in zip(res["per_hold"][h]["win"]["maker2"], res["per_hold"][h]["win"]["flat6"])))
check("a hold-10080 formula pays far less turnover than a hold-60 one",
      np.mean([p["turnover"] for p in res["parts"][10080]["win"]]) < 0.1 * np.mean([p["turnover"] for p in res["parts"][60]["win"]]),
      f"{np.mean([p['turnover'] for p in res['parts'][60]['win']]):.1f} vs {np.mean([p['turnover'] for p in res['parts'][10080]['win']]):.1f}")
# lag: the first LAG minutes must carry no weight
sigT = R.window_signal(StubVM(sig_slice), (0,), None, E)
check("window_signal applies the 2-min lag (first rows NaN, shape (T, A))",
      sigT.shape == (E, Ael) and np.isnan(sigT[:C.EXECUTION_LAG_MIN]).all() and np.isfinite(sigT[C.EXECUTION_LAG_MIN:]).all())
check("window_signal aligns: row t+LAG is the VM output at t",
      np.array_equal(sigT[C.EXECUTION_LAG_MIN + 5], sig_slice[:, -E:][:, 5]))
# coverage gate
dead = np.full((Ael, L + E), np.nan)
try:
    R.eval_formula(StubVM(dead), (0,), [60], [None], shard, [L], P, Inval)
    check("zero-coverage window raises invalid_cls", False)
except Inval:
    check("zero-coverage window raises invalid_cls", True)
# reward scalar
rw, arrw = R.reward_of(res, 1440, cost="flat6", variant="win", how="mean")
check("reward_of = mean of the per-window net log NAVs", abs(rw - float(np.mean(res['per_hold'][1440]['win']['flat6']))) < 1e-15)

# ---- 9. tied signals: average rank, so a constant / all-long signal holds NOTHING ----
Tt, At = 50, 30
elt = np.ones((Tt, At), bool)
for nm, s in (("constant 0", np.zeros((Tt, At))), ("all +1", np.ones((Tt, At)))):
    w = C.rank_weights(s, elt)
    check(f"rank_weights {nm}: zero book (gross 0)", np.abs(w).max() == 0.0, f"max|w| {np.abs(w).max():.3g}")
sg = np.sign(rng.normal(size=(Tt, At)))
wg = C.rank_weights(sg, elt)
check("rank_weights +-1 signs: still a real book, gross 1, net 0",
      np.allclose(np.abs(wg).sum(axis=1), 1.0) and np.abs(wg.sum(axis=1)).max() < 1e-12)
half = np.zeros((1, At)); half[0, :15] = 1.0
wh = C.rank_weights(half, np.ones((1, At), bool))[0]
check("rank_weights tied group shares one weight", np.ptp(wh[:15]) < 1e-15 and np.ptp(wh[15:]) < 1e-15)
sn = rng.normal(size=(Tt, At))
order_w = np.argsort(np.argsort(sn, axis=1, kind="stable"), axis=1).astype(float)
cent = order_w - (At - 1) / 2.0
ref = cent / np.abs(cent).sum(axis=1, keepdims=True)
check("rank_weights without ties == plain stable-rank book (bit-identical)",
      np.array_equal(C.rank_weights(sn, elt), ref))
# the real book (rank_weights -> held_path -> settle) on const / all-long earns exactly nothing
fwd9 = rng.normal(5e-4, 3e-4, (Tt, At))                    # a bull tape: every asset drifts up
for nm, s in (("constant", np.zeros((Tt, At))), ("all-long", np.ones((Tt, At)))):
    h9 = R.held_path(C.rank_weights(s, elt), 60, R.BAND, P)
    r9, p9 = R.settle_multi(h9, fwd9, np.zeros((Tt, At), bool), np.zeros((Tt, At)), FLAT)
    check(f"{nm} signal on a bull tape: gross 0, net 0, turnover 0",
          p9["gross"] == 0.0 and r9["flat6"]["log_nav"] == 0.0 and p9["turnover"] == 0.0)

# ---- 10. book_diagnostics ----
fwd10 = rng.normal(0, 1e-3, (400, At))
mkt10 = fwd10.mean(axis=1)
h_long = np.full((400, At), 1.0 / At)                      # fully long, equal weight
d_long = R.book_diagnostics(h_long, fwd10)
check("book_diagnostics: equal-weight long book has net exposure 1, beta 1",
      abs(d_long["net_exposure_mean"] - 1.0) < 1e-12 and abs(d_long["beta_mkt"] - 1.0) < 1e-9,
      f"net {d_long['net_exposure_mean']:.4f} beta {d_long['beta_mkt']:.4f}")
d_rank = R.book_diagnostics(C.rank_weights(rng.normal(size=(400, At)), np.ones((400, At), bool)), fwd10)
check("book_diagnostics: random rank book has net exposure 0, gross 1, small beta",
      abs(d_rank["net_exposure_mean"]) < 1e-12 and abs(d_rank["gross_exposure_mean"] - 1.0) < 1e-9
      and abs(d_rank["beta_mkt"]) < 0.5, f"beta {d_rank['beta_mkt']:.4f}")
fnan = fwd10.copy(); fnan[:, 3] = np.nan
check("book_diagnostics tolerates NaN forward returns", np.isfinite(R.book_diagnostics(h_long, fnan)["beta_mkt"]))
check("eval_formula parts carry net_exposure_mean / gross_exposure_mean / beta_mkt",
      all(k in res["parts"][60]["win"][0] for k in ("net_exposure_mean", "gross_exposure_mean", "beta_mkt")))

print("\nALL PASS" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
