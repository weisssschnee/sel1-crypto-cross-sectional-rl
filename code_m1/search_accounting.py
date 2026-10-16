"""Search-size accounting, deflation hurdle, and the secretary-problem stop.

References
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", JPM 40(5):94-105,
doi:10.3905/jpm.2014.40.5.094.

What this module does, and why each piece is here:

1. N accounting. Every formula that receives a reward is a trial. N must be
   written into the run's manifest and BATCH_LOG, because both the deflated
   Sharpe ratio and the secretary-problem stop are functions of N. If N is
   unknown, neither question can be answered after the fact.

2. The hurdle. Bailey & Lopez de Prado give

       SR_0 = sqrt(V) * [ (1-gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - e^-1/N) ]

   with V the variance of the trial returns, gamma = 0.5772 the Euler-Mascheroni
   constant, and Z^-1 the standard-normal quantile. This is the expected
   maximum Sharpe under the null after N trials: anything below it is
   indistinguishable from the best of N random strategies.

   Note the two limits. N=1 gives Z^-1(0) = 0, so a single hand-written
   baseline has hurdle 0 -- it is the floor, not a test. N large gives
   sqrt(2 ln N) scaling in the units of sqrt(V). The hurdle is therefore
   reported in the same units as the measured statistic; comparing a raw
   ICIR against a unit-variance number is meaningless.

3. Secretary problem. The optimal stopping rule for maximizing the maximum of
   N iid draws is to stop at the first observation exceeding
   t* = (1-gamma)/ln N of the way to the running record; the best time budget
   is therefore about 37% of N, not all of N. Running the full budget after a
   clear leader has emerged does not add evidence, it adds selection risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

GAMMA = 0.5772156649015329          # Euler-Mascheroni


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF), Acklam's rational approximation.

    Accurate to ~1e-9 over the whole range, which is far beyond what a
    Sharpe-scale comparison needs, and avoids a scipy dependency.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@dataclass
class SearchAccounting:
    updates: int
    batch: int
    formulas_evaluated: int          # N
    reward_variance_V: float | None = None   # from BATCH_LOG reward
    reward_variance_n: int = 0
    n_columns_screened: int = 0      # terminal columns screened in dry-run
    notes: str = ''

    def hurdle(self, gamma: float = GAMMA, variance: float | None = None):
        """Return (SR_0, breakdown) in the units of sqrt(variance).

        The two Z^-1 terms are both taken on the UPPER tail. Written as
        (1-gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - e^-1/N), the second argument
        approaches 1 from below as N grows, so its quantile must be taken as
        the positive upper-tail value Z^-1(e^-1/N) = -Z^-1(1 - e^-1/N).
        Taking both as literally written makes the combination negative and
        shrinking in magnitude, which cannot be an expected maximum; the
        corrected form grows like sqrt(2 ln N), matching the asymptotic
        expected maximum of N standard normals that gamma = E[Z_max] refers to.
        Verified numerically: N=21 -> 1.675, N=1e6 -> 4.753, vs sqrt(2 ln N)
        of 2.468 and 5.257.
        """
        N = max(int(self.formulas_evaluated), 1)
        V = self.reward_variance_V if variance is None else variance
        if V is None or not (V > 0) or not math.isfinite(V):
            V = 1.0                     # unit-variance floor; label it as such
            unit = True
        else:
            unit = False
        if N <= 1:
            # Z^-1(0) is -inf; the limit of the expected maximum of a single
            # draw is 0, so the hurdle is 0 by definition, not by cancellation.
            return 0.0, dict(N=N, V_used=V, unit_variance_floor=unit,
                             gamma=gamma, Z_inv_1_minus_1_over_N=0.0,
                             Z_inv_exp_minus_1_over_N_upper=0.0,
                             sqrt_2_ln_N=0.0, sqrt_V=math.sqrt(V), inner=0.0)
        z_a = _norm_ppf(1.0 - 1.0 / N)
        z_b = _norm_ppf(math.exp(-1.0 / N))        # upper tail, positive
        inner = (1.0 - gamma) * z_a + gamma * z_b
        sr0 = math.sqrt(V) * inner
        return sr0, dict(N=N, V_used=V, unit_variance_floor=unit,
                         gamma=gamma, Z_inv_1_minus_1_over_N=z_a,
                         Z_inv_exp_minus_1_over_N_upper=z_b,
                         sqrt_2_ln_N=math.sqrt(2.0 * math.log(N)) if N > 1 else 0.0,
                         sqrt_V=math.sqrt(V), inner=inner)

    def stop_budget(self, gamma: float = GAMMA):
        """Secretary-problem budget: the fraction of N worth spending.

        Bailey & Lopez de Prado's optimal stopping for max-of-N draws spends
        about 1/(1-gamma) ~ 2.36 time units per ln N, i.e. the useful budget
        saturates well before the full N. Reported alongside the simple 37%
        rule of thumb so the two can be compared.
        """
        N = max(int(self.formulas_evaluated), 1)
        lnN = math.log(N)
        rule_of_thumb = 0.37
        blp = min(1.0, 1.0 / ((1.0 - gamma) * lnN)) if lnN > 0 else 1.0
        return dict(N=N, ln_N=lnN,
                    budget_fraction_rule_of_thumb=rule_of_thumb,
                    budget_fraction_blp=blp,
                    stop_after_formulas_rule_of_thumb=int(rule_of_thumb * N),
                    stop_after_formulas_blp=int(blp * N))

    def to_dict(self, **kw):
        d = asdict(self)
        d.update(kw)
        sr0, br = self.hurdle(**({'variance': kw['variance']} if 'variance' in kw else {}))
        d['SR_0'] = sr0
        d['hurdle_breakdown'] = br
        d.update(self.stop_budget())
        return d


def format_report(acc: SearchAccounting, measured_test_stat: float | None = None,
                  baseline_n1: float | None = None) -> str:
    """Human-readable comparison. Never prints an absolute number without the
    hurdle it must clear, and never omits the N=1 baseline when supplied."""
    L = []
    sr0, br = acc.hurdle()
    L.append('=' * 74)
    L.append('SEARCH-SIZE ACCOUNTING AND DEFLATION REPORT')
    L.append('=' * 74)
    L.append(f'  updates                     {acc.updates}')
    L.append(f'  batch                       {acc.batch}')
    L.append(f'  N (formulas evaluated)      {acc.formulas_evaluated}')
    if acc.n_columns_screened:
        L.append(f'  N_columns_screened (dry-run) {acc.n_columns_screened}'
                 '   <- separate axis, deflate separately or state why not')
    L.append('')
    L.append('  DEFLATION HURDLE (Bailey & Lopez de Prado 2014, eq. for SR_0)')
    L.append(f'    V (reward variance)        '
             f'{acc.reward_variance_V if acc.reward_variance_V is not None else "NOT SUPPLIED -> unit-variance floor"}')
    L.append(f'    gamma                      {GAMMA:.4f}')
    L.append(f'    Z^-1(1-1/N)                {br["Z_inv_1_minus_1_over_N"]:.4f}')
    L.append(f'    Z^-1(e^-1/N) upper tail     {br["Z_inv_exp_minus_1_over_N_upper"]:.4f}')
    L.append(f'    sqrt(2 ln N) reference       {br["sqrt_2_ln_N"]:.4f}   (asymptotic expected max)')
    L.append(f'    SR_0                       {sr0:.4f}'
             + ('   (unit-variance floor)' if br['unit_variance_floor'] else ''))
    L.append('')
    sb = acc.stop_budget()
    L.append('  SECRETARY-PROBLEM STOP')
    L.append(f'    rule-of-thumb budget       {sb["budget_fraction_rule_of_thumb"]*100:.0f}%'
             f'  -> stop after ~{sb["stop_after_formulas_rule_of_thumb"]} formulas')
    L.append(f'    BLP 1/((1-gamma) ln N)     {sb["budget_fraction_blp"]*100:.0f}%'
             f'  -> stop after ~{sb["stop_after_formulas_blp"]} formulas')
    L.append('    Running past a clear leader raises false-positive probability;')
    L.append('    extra updates are not extra evidence.')
    L.append('')
    L.append('  COMPARISON')
    if measured_test_stat is None:
        L.append('    test statistic             NOT MEASURED (no test window scored)')
    else:
        verdict = 'CLEARS' if measured_test_stat > sr0 else 'FAILS'
        L.append(f'    test statistic             {measured_test_stat:.4f}')
        L.append(f'    vs SR_0                    {verdict}'
                 f'  (margin {measured_test_stat - sr0:+.4f})')
    if baseline_n1 is None:
        L.append('    N=1 baseline               NOT MEASURED')
    else:
        L.append(f'    N=1 baseline               {baseline_n1:.4f}'
                 '   (hurdle 0.000; the floor to beat)')
        if measured_test_stat is not None:
            L.append(f'    vs baseline                '
                     f'{"BEATS" if measured_test_stat > baseline_n1 else "DOES NOT BEAT"}')
    L.append('=' * 74)
    return '\n'.join(L)


if __name__ == '__main__':
    for upd, batch in [(64, 128), (64, 256), (128, 256)]:
        acc = SearchAccounting(updates=upd, batch=batch,
                               formulas_evaluated=upd * batch)
        print(format_report(acc))
        print()
    acc = SearchAccounting(updates=1, batch=1, formulas_evaluated=1)
    print(format_report(acc, measured_test_stat=0.0, baseline_n1=0.0))
