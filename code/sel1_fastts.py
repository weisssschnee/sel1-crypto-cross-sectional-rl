"""Fused trailing-window operators for xs_tscs_vm_v2 (XS_SEL1 补充 A speed-up). New file; the VM is imported, never
modified: install() swaps entries of V._WINDOWED_IMPL in the calling process only.

The original operators build (assets, times, window) work arrays (unfold, isfinite, where, products) and reduce them
with torch's CUDA sum / max kernels. Each replacement here computes those window reductions in one CUDA kernel that
reads the (assets, times) input directly, then runs the original body's elementwise tail in torch on the results.
  counts, TS_RANK's "below" count   integers in float32 (< 2^24): any order gives the same value
  TS_MAX / TS_MIN                    the first maximum / minimum in window order (torch's max with indices)
  float sums                         the order of torch's float sum over a contiguous last dim (ATen Reduce.cuh,
                                     vt0 4): window >= 128 and a multiple of 4: 32 lanes over vectors of 4 (lane l
                                     takes vectors l, l+32, ..; 4 accumulators, combined 0+1+2+3); otherwise
                                     min(pow2 <= window, 32) lanes, lane l takes l, l+lanes, .. into 4 accumulators;
                                     then the warp shuffle tree. Adds and products are __fadd_rn / __fmul_rn (no FMA).
sel1_fastcheck.py compares every replaced operator on every window bit for bit against the original and every
checked body's signal by sha; anything not float32 / CUDA goes to the original operator.
"""
from __future__ import annotations

import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as V

ORIG = dict(V._WINDOWED_IMPL)
REPLACED = ("TS_MEAN", "TS_STD", "TS_ZSCORE", "TS_MAX", "TS_MIN", "TS_RANK", "TS_SLOPE", "TS_CORR")
_KERN = {}
CALLS = {}                                     # kernel launches, so a check can prove the fused path ran

_SRC = r"""
#define F_INF __int_as_float(0x7f800000)

__device__ __forceinline__ bool fin(float v) { return isfinite(v); }

struct Win {                                   // slot k of the trailing window ending at t; NaN before the start
    const float* p; int s0;
    __device__ __forceinline__ float operator()(int k) const {
        return (s0 + k >= 0) ? p[s0 + k] : __int_as_float(0x7fc00000);
    }
};

template <class F>
__device__ __forceinline__ float tsum(const F& f, int W) {   // W is a runtime argument: one compile per kernel
    const bool VEC = W >= 128 && W % 4 == 0;                  // _ok sends W >= 128 with W % 4 != 0 to the original
    const int NV = W / 4;
    int BW = 32;
    if (!VEC) while (BW > W) BW >>= 1;                        // min(pow2 <= W, 32)
    float lane[32];
    for (int l = 0; l < BW; ++l) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (VEC) {
            for (int v = l; v < NV; v += 32) {
                a0 = __fadd_rn(a0, f(4 * v));     a1 = __fadd_rn(a1, f(4 * v + 1));
                a2 = __fadd_rn(a2, f(4 * v + 2)); a3 = __fadd_rn(a3, f(4 * v + 3));
            }
        } else {
            int idx = l;
            while (idx + 3 * BW < W) {
                a0 = __fadd_rn(a0, f(idx));          a1 = __fadd_rn(a1, f(idx + BW));
                a2 = __fadd_rn(a2, f(idx + 2 * BW)); a3 = __fadd_rn(a3, f(idx + 3 * BW));
                idx += 4 * BW;
            }
            if (idx < W) { a0 = __fadd_rn(a0, f(idx)); idx += BW;
              if (idx < W) { a1 = __fadd_rn(a1, f(idx)); idx += BW;
                if (idx < W) { a2 = __fadd_rn(a2, f(idx)); idx += BW;
                  if (idx < W) { a3 = __fadd_rn(a3, f(idx)); } } } }
        }
        lane[l] = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    }
    for (int off = BW / 2; off > 0; off >>= 1) {
        for (int l = 0; l < off; ++l) lane[l] = __fadd_rn(lane[l], lane[l + off]);
    }
    return lane[0];
}

struct Clean  { Win w; __device__ float operator()(int k) const { float v = w(k); return fin(v) ? v : 0.f; } };
struct Clean2 { Win w; __device__ float operator()(int k) const { float v = w(k); v = fin(v) ? v : 0.f; return __fmul_rn(v, v); } };
struct Tc     { Win w; __device__ float operator()(int k) const { return fin(w(k)) ? (float)k : 0.f; } };
struct Tc2    { Win w; __device__ float operator()(int k) const { float t = fin(w(k)) ? (float)k : 0.f; return __fmul_rn(t, t); } };
struct TcX    { Win w; __device__ float operator()(int k) const { float v = w(k); bool f = fin(v);
                                                                  return __fmul_rn(f ? (float)k : 0.f, f ? v : 0.f); } };
struct Cx     { Win x, y; __device__ float operator()(int k) const { float a = x(k), b = y(k); return (fin(a) && fin(b)) ? a : 0.f; } };
struct Cy     { Win x, y; __device__ float operator()(int k) const { float a = x(k), b = y(k); return (fin(a) && fin(b)) ? b : 0.f; } };
struct Cxy    { Win x, y; int m; __device__ float operator()(int k) const {
                    float a = x(k), b = y(k); bool f = fin(a) && fin(b); a = f ? a : 0.f; b = f ? b : 0.f;
                    return m == 0 ? __fmul_rn(a, b) : (m == 1 ? __fmul_rn(a, a) : __fmul_rn(b, b)); } };

__device__ __forceinline__ bool where_(int n, int T, int i, int& a, int& t) {
    if (i >= n) return false;
    a = i / T; t = i - a * T; return true;
}

extern "C" __global__ void k_mom(const float* x, float* cnt, float* s1, float* s2, int n, int T, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, a, t;
    if (!where_(n, T, i, a, t)) return;
    Win w{ x + (long long)a * T, t - W + 1 };
    float c = 0.f;
    for (int k = 0; k < W; ++k) c += fin(w(k)) ? 1.f : 0.f;
    cnt[i] = c; s1[i] = tsum(Clean{w}, W); s2[i] = tsum(Clean2{w}, W);
}

extern "C" __global__ void k_ext(const float* x, float* cnt, float* mx, float* mn, float* below, int n, int T, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, a, t;
    if (!where_(n, T, i, a, t)) return;
    Win w{ x + (long long)a * T, t - W + 1 };
    float xb = x[(long long)a * T + t], c = 0.f, b = 0.f, hi = -F_INF, lo = F_INF;
    for (int k = 0; k < W; ++k) {
        float v = w(k);
        bool f = fin(v);
        c += f ? 1.f : 0.f;
        b += (f && v <= xb) ? 1.f : 0.f;
        float vh = f ? v : -F_INF, vl = f ? v : F_INF;
        if (k == 0 || vh > hi) hi = vh;
        if (k == 0 || vl < lo) lo = vl;
    }
    cnt[i] = c; mx[i] = hi; mn[i] = lo; below[i] = b;
}

extern "C" __global__ void k_slope(const float* x, float* cnt, float* sx, float* st, float* stx, float* stt,
                                   int n, int T, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, a, t;
    if (!where_(n, T, i, a, t)) return;
    Win w{ x + (long long)a * T, t - W + 1 };
    float c = 0.f;
    for (int k = 0; k < W; ++k) c += fin(w(k)) ? 1.f : 0.f;
    cnt[i] = c; sx[i] = tsum(Clean{w}, W); st[i] = tsum(Tc{w}, W); stx[i] = tsum(TcX{w}, W); stt[i] = tsum(Tc2{w}, W);
}

extern "C" __global__ void k_corr(const float* x, const float* y, float* cnt, float* sx, float* sy, float* sxy,
                                  float* sxx, float* syy, int n, int T, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, a, t;
    if (!where_(n, T, i, a, t)) return;
    Win wx{ x + (long long)a * T, t - W + 1 }, wy{ y + (long long)a * T, t - W + 1 };
    float c = 0.f;
    for (int k = 0; k < W; ++k) c += (fin(wx(k)) && fin(wy(k))) ? 1.f : 0.f;
    cnt[i] = c; sx[i] = tsum(Cx{wx, wy}, W); sy[i] = tsum(Cy{wx, wy}, W);
    sxy[i] = tsum(Cxy{wx, wy, 0}, W); sxx[i] = tsum(Cxy{wx, wy, 1}, W); syy[i] = tsum(Cxy{wx, wy, 2}, W);
}
"""


def _torch():
    import torch
    return torch


def _kernel(name):
    if name not in _KERN:                                   # one NVRTC compile per kernel; the window is an argument
        _KERN[name] = _torch().cuda._compile_kernel(_SRC, name)
    return _KERN[name]


def _launch(name, window, ins, n_out):
    """ins: contiguous (A, T) float32 CUDA tensors -> n_out (A, T) float32 window reductions."""
    torch = _torch()
    A, T = ins[0].shape
    outs = [torch.empty((A, T), dtype=torch.float32, device=ins[0].device) for _ in range(n_out)]
    n = A * T
    if n:
        CALLS[name] = CALLS.get(name, 0) + 1
        _kernel(name)(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), args=list(ins) + outs + [n, T, window])
    return outs


def _ok(window, *xs):
    torch = _torch()
    # whole panel or blocks: torch sums every output over the same contiguous window row either way
    return (isinstance(window, int) and window >= 1 and not (window >= 128 and window % 4)
            and all(x.dtype == torch.float32 and x.is_cuda and x.dim() == 2 and x.shape == xs[0].shape for x in xs)
            and xs[0].numel() < 2 ** 31)


def _gate(value, count, window):
    return V._gate_min_periods(value, count, window)


def _moments(x, window):
    count, s1, s2 = _launch("k_mom", window, [x.contiguous()], 3)
    n = count.clamp_min(1.0)
    mean = s1 / n
    sq = s2 / n
    return mean, (sq - mean * mean).clamp_min(0.0), count


def ts_mean(x, window):
    if not _ok(window, x):
        return ORIG["TS_MEAN"](x, window)
    count, s1, _ = _launch("k_mom", window, [x.contiguous()], 3)
    return _gate(s1 / count.clamp_min(1.0), count, window)


def ts_std(x, window):
    if not _ok(window, x):
        return ORIG["TS_STD"](x, window)
    torch = _torch()
    mean, var, count = _moments(x, window)
    n = count.clamp_min(2.0)
    return _gate(torch.sqrt(var * n / (n - 1.0)), count, window)


def ts_zscore(x, window):
    if not _ok(window, x):
        return ORIG["TS_ZSCORE"](x, window)
    torch = _torch()
    mean, var, count = _moments(x, window)
    n = count.clamp_min(2.0)
    std = torch.sqrt((var * n / (n - 1.0)).clamp_min(0.0))
    z = (x - mean) / torch.where(std > 0, std, torch.full_like(std, float("nan")))
    return _gate(z, count, window)


def _ext(x, window):
    return _launch("k_ext", window, [x.contiguous()], 4)


def ts_max(x, window):
    if not _ok(window, x):
        return ORIG["TS_MAX"](x, window)
    count, mx, _, _ = _ext(x, window)
    return _gate(mx, count, window)


def ts_min(x, window):
    if not _ok(window, x):
        return ORIG["TS_MIN"](x, window)
    count, _, mn, _ = _ext(x, window)
    return _gate(mn, count, window)


def ts_rank(x, window):
    if not _ok(window, x):
        return ORIG["TS_RANK"](x, window)
    torch = _torch()
    count, _, _, below = _ext(x, window)
    ranked = 2.0 * (below / count.clamp_min(1.0)) - 1.0
    ranked = torch.where(torch.isfinite(x), ranked, torch.full_like(ranked, float("nan")))
    return _gate(ranked, count, window)


def ts_slope(x, window):
    if not _ok(window, x):
        return ORIG["TS_SLOPE"](x, window)
    torch = _torch()
    count, sx, st, stx, stt = _launch("k_slope", window, [x.contiguous()], 5)
    n = count.clamp_min(1.0)
    mean_t = st / n
    mean_x = sx / n
    cov = stx / n - mean_t * mean_x
    var_t = stt / n - mean_t * mean_t
    slope = cov / torch.where(var_t > 0, var_t, torch.full_like(var_t, float("nan")))
    return _gate(slope, count, window)


def ts_corr(x, y, window):
    if not _ok(window, x, y):
        return ORIG["TS_CORR"](x, y, window)
    torch = _torch()
    count, sx, sy, sxy, sxx, syy = _launch("k_corr", window, [x.contiguous(), y.contiguous()], 6)
    n = count.clamp_min(1.0)
    mx = sx / n
    my = sy / n
    cov = sxy / n - mx * my
    vx = sxx / n - mx * mx
    vy = syy / n - my * my
    denom = torch.sqrt((vx * vy).clamp_min(0.0))
    corr = cov / torch.where(denom > 0, denom, torch.full_like(denom, float("nan")))
    return _gate(corr.clamp(-1.0, 1.0), count, window)


FAST = {"TS_MEAN": ts_mean, "TS_STD": ts_std, "TS_ZSCORE": ts_zscore, "TS_MAX": ts_max, "TS_MIN": ts_min,
        "TS_RANK": ts_rank, "TS_SLOPE": ts_slope, "TS_CORR": ts_corr}


def _cuda_home():
    """torch's NVRTC wrapper wants CUDA_HOME/include to exist; the m1cu128 wheel ships NVRTC without a toolkit."""
    import os
    import tempfile
    if not os.environ.get("CUDA_HOME"):
        h = os.path.join(tempfile.gettempdir(), "sel1_fastts_cuda_home")
        os.makedirs(os.path.join(h, "include"), exist_ok=True)
        os.environ["CUDA_HOME"] = h


def install(names=REPLACED):
    """Swap the named operators in this process; returns the names swapped."""
    _cuda_home()
    for k in names:
        V._WINDOWED_IMPL[k] = FAST[k]
    return tuple(names)


def uninstall():
    V._WINDOWED_IMPL.update(ORIG)
