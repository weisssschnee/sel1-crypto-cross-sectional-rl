"""CR1b unit check (no panel read): imports, and resid_rows on synthetic data — the residual is orthogonal to
the regressors and to the intercept on the regression sample, unit std, NaN outside it; a formula that is a
linear mix of the cells gives an all-NaN residual (SD_MIN); the plant-shaped case keeps its own part."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cr1b_common as Q
import cr1b_calib  # noqa: F401
import cr1b_round  # noqa: F401
import cr1b_trainer  # noqa: F401

rng = np.random.default_rng(1)
E, A, P = 40, 60, 8
Z = rng.standard_normal((P, E, A)).astype(np.float32)
K = np.full((E, A), 8.0)
K[:, :5] = 2.0                                     # combo undefined on 5 coins
s = rng.standard_normal((E, A))
zf = 0.7 * Z[0] - 0.4 * Z[3] + 0.5 * s
zf[:, -3:] = np.nan
r = Q.resid_rows(zf, Z, K, 5)
m = np.isfinite(r)
ok = []
ok.append(not m[:, :5].any() and not m[:, -3:].any())
for t in range(E):
    mt = m[t]
    rt = r[t, mt]
    ok.append(abs(rt.mean()) < 1e-9 and abs(rt.std() - 1) < 1e-9)
    ok.append(max(abs(np.dot(rt, Z[p, t, mt].astype(np.float64))) for p in range(P)) < 1e-6)
    ok.append(np.corrcoef(rt, s[t, mt])[0, 1] > 0.8)
lin = (0.3 * Z[1] + 0.2 * Z[2]).astype(np.float64)
r2 = Q.resid_rows(lin, Z, K, 5)
ok.append(not np.isfinite(r2).any())
print("UNIT %s (%d checks)" % ("OK" if all(ok) else "FAIL", len(ok)))
sys.exit(0 if all(ok) else 1)
