# -*- coding: utf-8 -*-
"""Streaming (running-integral) QE simulation + per-period caplet pricing.

This is the O(n_paths)-memory simulator the write-up describes: the accrual and
discount integrals are carried along the step loop and only their values at the
caplet fixing/payment indices are kept, so the full (n_paths, n_steps+1) path
matrix is never materialised.

Why it matters at fine dt. dm_sim_qe.simulate_2factor_qe returns key/ois/v as
full path matrices and price_perperiod then cumsums across them; at dt=1/252
(n_steps=7574) that is ~11.6 GiB at 16384 paths, which is what blocks a daily
run. Here the state is (n_eff,) vectors plus an (n_eff, n_save) buffer, and for
the EUR surface n_save = 60: the 1062 caplets sit on a 0.5y fixing grid with
tau=0.5, so each caplet's payment index is the next one's fixing index.

Two further differences from dm_sim_qe, both pure wins:
  * antithetic paths are stacked into the path dimension and stepped in ONE
    loop, instead of running the whole loop twice and concatenating;
  * the normals/uniforms are drawn in step-chunks rather than as six full
    (n_paths, n_steps) matrices.

Numerics are step-for-step identical to dm_sim_qe + dm_calibrate.price_perperiod
(same QE branch logic, same OU/HW recursions, same integral convention). Pass
`rands=` to drive it with externally drawn draws, which is how the equivalence
test pins it against the reference implementation.
"""
import math
import torch


# ----------------------------------------------------------------- QE step
def _qe_step(Vt, th, e, ep2, k, Z, U, psi_c=1.5, om_e=None):
    """One Andersen-QE CIR step. Mirrors dm_sim_qe.cir_qe_paths exactly.
    Returns (V_next, xi) with xi the standardised innovation used for the skew."""
    ome = (1 - e) if om_e is None else om_e   # 1-exp(-k*dt): pass it in at fine dt,
    m = th + (Vt - th) * e                    # where a float32 (1-e) loses ~3 digits
    s2 = Vt * ep2 * e * ome / k + th * ep2 * ome ** 2 / (2 * k)
    psi = s2 / (m * m + 1e-30)
    inv = 2.0 / psi.clamp(min=1e-8)
    b2 = (inv - 1 + torch.sqrt((inv * (inv - 1)).clamp(min=0.0))).clamp(min=0.0)
    Vq = (m / (1 + b2)) * (torch.sqrt(b2) + Z) ** 2
    p = ((psi - 1) / (psi + 1)).clamp(0.0, 1 - 1e-7)
    beta = (1 - p) / (m + 1e-30)
    Ve = torch.where(
        U <= p, torch.zeros_like(Vt),
        torch.log(((1 - p) / (1 - U).clamp(min=1e-30))).clamp(min=0.0) / (beta + 1e-30))
    Vnew = torch.where(psi <= psi_c, Vq, Ve).clamp(min=0.0)
    # eps=0 (the Gaussian/deterministic-vol ladder rungs) gives s2 == 0 exactly: the variance
    # step is deterministic and (V-m)/s is 0/0, so fall back to the driving normal Z, which is
    # what the quadratic branch's innovation tends to as s2 -> 0.
    xi = torch.where(s2 > 0, (Vnew - m) / torch.sqrt(s2.clamp(min=1e-30)), Z)
    return Vnew, xi


# ----------------------------------------------------------------- draws
class _Draws:
    """Step-chunked N(0,1)/U(0,1) source, antithetic-stacked to (2*n, ...).

    Antithetic mirroring matches dm_sim_qe's second `run` call: normals negate,
    uniforms reflect as 1-U. If `full` is given (six (n, n_steps) matrices) the
    draws are served from it instead, so a caller can reproduce the reference
    implementation's exact random stream.
    """

    def __init__(self, n, n_steps, device, seed=None, antithetic=True, chunk=512, full=None):
        self.n, self.n_steps, self.device = n, n_steps, device
        self.anti, self.chunk, self.full = antithetic, chunk, full
        self.lo, self.hi = 0, 0
        self.buf = None
        if full is None and seed is not None:
            torch.manual_seed(seed)

    def _stack(self, z1, u1, z2, u2, zx, zk):
        if not self.anti:
            return z1, u1, z2, u2, zx, zk
        cat = torch.cat
        return (cat([z1, -z1]), cat([u1, 1 - u1]), cat([z2, -z2]),
                cat([u2, 1 - u2]), cat([zx, -zx]), cat([zk, -zk]))

    def step(self, i):
        if self.full is not None:
            z1, u1, z2, u2, zx, zk = self.full
            return self._stack(z1[:, i], u1[:, i], z2[:, i], u2[:, i], zx[:, i], zk[:, i])
        if not (self.lo <= i < self.hi):                  # refill
            self.lo = i
            self.hi = min(i + self.chunk, self.n_steps)
            w = self.hi - self.lo
            n, dev = self.n, self.device
            self.buf = self._stack(
                torch.randn(n, w, device=dev), torch.rand(n, w, device=dev),
                torch.randn(n, w, device=dev), torch.rand(n, w, device=dev),
                torch.randn(n, w, device=dev), torch.randn(n, w, device=dev))
        j = i - self.lo
        return tuple(b[:, j] for b in self.buf)



# ----------------------------------------------------------------- payoff
PAYOFFS = ("arrears_avg", "arrears_comp", "term")


def _window_factor(rate64, n_win, dt):
    """sum_{m=0}^{n-1} exp(-a*m*dt) * dt, in float64 (expm1 keeps the digits at daily dt).
    rate64: (K,1) or scalar tensor; n_win: (n_caplets,) steps per accrual window."""
    num = -torch.expm1(-rate64 * (n_win.to(torch.float64) * dt))
    den = (-torch.expm1(-rate64 * dt)).clamp(min=1e-300)
    return dt * num / den


def _log_growth(payoff, dIkey, x_fix=None, k_fix=None, I_det=None, Bx=None, Bk=None):
    """Exponent g of the caplet rate R = (exp(g) - 1)/tau (not used for arrears_avg).

    arrears_avg  : (1/tau) * integral of r_proj over the window -- the legacy payoff.
    arrears_comp : (exp(integral) - 1)/tau -- daily-compounded in-arrears RFR rate
                   (SOFR/SONIA); its forward is the simple forward off the curve,
                   which is what the stripper uses.
    term         : forward-looking term rate (Euribor), set at t_fix:
                   (exp(E_t_fix[integral]) - 1)/tau. x and k are OU with linear drift,
                   so E_t_fix[integral] = I_det + x(t_fix) Bx + k(t_fix) Bk exactly,
                   whatever the variance path does after t_fix (Henrard 2026, s.3.6/5.2.5:
                   continuously-compounded additive spread over the collateral curve).
    """
    if payoff == "arrears_comp":
        return dIkey
    if payoff == "term":
        return I_det + x_fix * Bx + k_fix * Bk
    raise ValueError(payoff)


def _price(payoff, dIkey, Iois_pay, tau_t, K_t, dim, fit_curve=False,
           f_key_vec=None, f_ois_vec=None, i_fix=None, i_pay=None, dt=None, term_args=None):
    """(pv, F, D) from the per-path window quantities; `dim` is the path axis.

    fit_curve: deterministic shift phi(t) so the model reprices the initial curves
    (Brigo-Mercurio-style extension, computed on the path set because the variance
    is stochastic). The factors are added to the market forwards with no convexity
    term, so E[exp(-int x)] > 1 and the raw model forward sits below the market
    forward (~20-35bp at 10Y). phi never enters the dynamics, only the integrals:
      discount   : int_0^T phi_ois = ln E[exp(-int_0^T x)]  ->  D(T) = P_mkt(T);
      projection : int_fix^pay phi_proj = c, one constant per caplet, solved in closed
                   form so that E[R disc]/D equals the market simple forward.
    The discount shift rescales pv and D alike, so it leaves implied vols unchanged;
    the projection shift moves the forward and hence where each strike sits.
    """
    disc = torch.exp(-Iois_pay)
    if payoff == "arrears_avg":
        assert not fit_curve, "curve fit is defined for the compounded/term payoffs"
        A = dIkey / tau_t
    else:
        g = _log_growth(payoff, dIkey, **(term_args or {}))
        if fit_curve:
            cum = lambda f: torch.cat([torch.zeros(1, dtype=torch.float64, device=f.device),
                                       torch.cumsum(f[:int(i_pay.max())].to(torch.float64) * dt, 0)])
            cf_o, cf_k = cum(f_ois_vec), cum(f_key_vec)
            P_mkt = torch.exp(-cf_o[i_pay]).to(disc.dtype)                     # grid market DF
            disc = disc * (P_mkt / disc.mean(dim, keepdim=True))                 # exp(-Psi(pay))
            lnF1 = (cf_k[i_pay] - cf_k[i_fix]).to(disc.dtype)                   # ln(1 + tau F_mkt)
            D_ = disc.mean(dim, keepdim=True)
            c = lnF1 + torch.log(D_) - torch.log((torch.exp(g) * disc).mean(dim, keepdim=True))
            g = g + c
        A = torch.expm1(g) / tau_t
    D = disc.mean(dim)
    F = (A * disc).mean(dim) / D.clamp(min=1e-12)
    pv = (torch.clamp(A - K_t, min=0.0) * tau_t * disc).mean(dim)
    return pv, F, D

# ----------------------------------------------------------------- driver
def simulate_price_streaming(n_paths, timeline, theta_vec, theta2_vec, eps, v0,
                             kf, ks, wf, lam, gam, xi, f_key_vec, f_ois_vec,
                             rho_vec, i_fix, i_pay, tau_t, K_t, device,
                             seed=None, antithetic=True, chunk=512, rands=None,
                             payoff="arrears_avg", fit_curve=False):
    """Simulate and price the caplets in one pass. Returns (pv, F, D), matching
    dm_calibrate.price_perperiod applied to dm_sim_qe.simulate_2factor_qe output
    for the default payoff; see _underlying for the others."""
    n_steps = len(timeline) - 1
    dt = (timeline[1] - timeline[0]).item()
    n_eff = 2 * n_paths if antithetic else n_paths

    # columns to retain: the union of fixing and payment indices
    keep = torch.unique(torch.cat([i_fix, i_pay])).sort().values
    col_fix = torch.searchsorted(keep, i_fix)
    col_pay = torch.searchsorted(keep, i_pay)
    n_save = int(keep.numel())
    keep_set = {int(v) for v in keep.tolist()}
    col_of = {int(v): j for j, v in enumerate(keep.tolist())}

    kf_f, ks_f, wf_f = float(kf), float(ks), float(wf)
    lam_f, gam_f, xi_f, eps_f = float(lam), float(gam), float(xi), float(eps)
    ep2 = eps_f ** 2
    e_f, e_s = math.exp(-kf_f * dt), math.exp(-ks_f * dt)
    e_lam = math.exp(-lam_f * dt)
    e_gam = math.exp(-gam_f * dt)
    var_k = xi_f * xi_f * (1.0 - math.exp(-2.0 * gam_f * dt)) / max(2.0 * gam_f, 1e-8)
    std_k = math.sqrt(max(var_k, 1e-12))
    sqrt_dt = math.sqrt(dt)

    z = torch.zeros(n_eff, device=device)
    v1 = z + float(v0); v2 = z + float(v0)
    x = z.clone(); kk = z.clone()
    Ikey = z.clone(); Iois = z.clone()
    Bkey = torch.zeros(n_eff, n_save, device=device)
    Bois = torch.zeros(n_eff, n_save, device=device)
    term = payoff == "term"
    if term:                                               # factor state at the fixings
        Bxs = torch.zeros(n_eff, n_save, device=device)
        Bks = torch.zeros(n_eff, n_save, device=device)

    rho_t = rho_vec[:n_steps]
    sqrt_1mr2 = torch.sqrt(torch.clamp(1.0 - rho_t ** 2, min=1e-6))
    draws = _Draws(n_paths, n_steps, device, seed=seed, antithetic=antithetic,
                   chunk=chunk, full=rands)

    for i in range(n_steps):
        if i in keep_set:                                  # integral up to index i
            j = col_of[i]
            Bkey[:, j] = Ikey
            Bois[:, j] = Iois
            if term:
                Bxs[:, j] = x
                Bks[:, j] = kk
        # accrue over [i, i+1) with the state at i (cumsum of key[:, :-1]*dt)
        Ikey = Ikey + (f_key_vec[i] + x + kk) * dt
        Iois = Iois + (f_ois_vec[i] + x) * dt

        Z1, U1, Z2, U2, Zx, Zk = draws.step(i)
        v_i = wf_f * v1 + (1.0 - wf_f) * v2                # variance at i drives the OU step
        v1, xi1 = _qe_step(v1, theta_vec[i], e_f, ep2, kf_f, Z1, U1)
        v2, _ = _qe_step(v2, theta2_vec[i], e_s, ep2, ks_f, Z2, U2)
        randn_x = rho_t[i] * xi1 + sqrt_1mr2[i] * Zx       # leverage/skew
        x = x * e_lam + torch.sqrt(v_i.clamp(min=1e-9)) * sqrt_dt * randn_x
        kk = kk * e_gam + std_k * Zk

    if n_steps in keep_set:
        Bkey[:, col_of[n_steps]] = Ikey
        Bois[:, col_of[n_steps]] = Iois

    dIkey = Bkey[:, col_pay] - Bkey[:, col_fix]
    if term:
        cumf = torch.cat([torch.zeros(1, dtype=torch.float64, device=device),
                          torch.cumsum(f_key_vec[:n_steps].to(torch.float64) * dt, 0)])
        I_det = (cumf[i_pay] - cumf[i_fix]).to(torch.float32)
        n_win = i_pay - i_fix
        f64 = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
        Bx = _window_factor(f64(lam_f), n_win, dt).to(torch.float32)
        Bk = _window_factor(f64(gam_f), n_win, dt).to(torch.float32)
        targs = dict(x_fix=Bxs[:, col_fix], k_fix=Bks[:, col_fix], I_det=I_det, Bx=Bx, Bk=Bk)
    else:
        targs = None
    return _price(payoff, dIkey, Bois[:, col_pay], tau_t, K_t, 0, fit_curve,
                  f_key_vec, f_ois_vec, i_fix, i_pay, dt, targs)


# ----------------------------------------------------------------- batched
class _DrawsBatched:
    """Chunked draws with a leading member dimension, antithetic-stacked.

    Members get independent streams (different slices of one generator), which
    is what the sequential code achieves by passing a different seed per
    evaluation. It cannot reproduce that code's stream bitwise -- chunked draws
    consume the generator in a different order than six full-matrix draws -- so
    the batched path is gated on agreeing with the sequential one under an
    explicitly supplied `full`, and statistically otherwise.
    """

    IS_UNIFORM = (False, True, False, True, False, False)   # z1 u1 z2 u2 zx zk

    def __init__(self, K, n, n_steps, device, seed=None, antithetic=True, chunk=128, full=None):
        self.K, self.n, self.n_steps, self.device = K, n, n_steps, device
        self.anti, self.chunk, self.full = antithetic, chunk, full
        self.lo, self.hi, self.buf = 0, 0, None
        if full is None:
            self.gen = torch.Generator(device=device)
            self.gen.manual_seed(int(seed) if seed is not None else 0)
            n_eff = 2 * n if antithetic else n
            w = min(chunk, n_steps)
            self.buf = [torch.empty(K, n_eff, w, device=device) for _ in range(6)]

    def _stack(self, z1, u1, z2, u2, zx, zk):
        """Only used for the `full` (test) path; the production path mirrors in place."""
        if not self.anti:
            return z1, u1, z2, u2, zx, zk
        c = lambda a, b: torch.cat([a, b], dim=1)          # along the path axis
        return (c(z1, -z1), c(u1, 1 - u1), c(z2, -z2),
                c(u2, 1 - u2), c(zx, -zx), c(zk, -zk))

    def _refill(self, i):
        self.lo, self.hi = i, min(i + self.chunk, self.n_steps)
        w, n, g = self.hi - self.lo, self.n, self.gen
        for b, is_u in zip(self.buf, self.IS_UNIFORM):
            head = b[:, :n, :w]
            head.uniform_(generator=g) if is_u else head.normal_(generator=g)
            if self.anti:                                   # mirror: -Z for normals, 1-U for uniforms
                tail = b[:, n:, :w]
                tail.copy_(head).neg_()
                if is_u:
                    tail.add_(1.0)

    def step(self, i):
        if self.full is not None:
            f = self.full
            return self._stack(*[t[:, :, i] for t in f])
        if not (self.lo <= i < self.hi):
            self._refill(i)
        j = i - self.lo
        return tuple(b[:, :, j] for b in self.buf)


def simulate_price_streaming_batched(n_paths, timeline, theta_mat, theta2_mat, eps_v, v0_v,
                                     kf_v, ks_v, wf_v, lam_v, gam_v, xi_v,
                                     f_key_vec, f_ois_vec, rho_mat,
                                     i_fix, i_pay, tau_t, K_t, device,
                                     seed=None, antithetic=True, chunk=128, rands=None,
                                     payoff="arrears_avg", fit_curve=False):
    """K parameter sets stepped through ONE loop.

    Because the cost is kernel-launch bound, K members cost about the same wall
    clock as one, which is where the speedup comes from: an SPSA iteration's
    2M+1 evaluations become a single pass instead of 2M+1 sequential ones.

    theta_mat/theta2_mat/rho_mat are (K, n_steps+1); the remaining model
    parameters are (K,) vectors. Returns (pv, F, D), each (K, n_caplets).
    """
    K = theta_mat.shape[0]
    n_steps = len(timeline) - 1
    dt = (timeline[1] - timeline[0]).item()
    n_eff = 2 * n_paths if antithetic else n_paths

    keep = torch.unique(torch.cat([i_fix, i_pay])).sort().values
    col_fix = torch.searchsorted(keep, i_fix)
    col_pay = torch.searchsorted(keep, i_pay)
    col_of = {int(v): j for j, v in enumerate(keep.tolist())}
    keep_set = set(col_of)

    col = lambda t: t.reshape(K, 1).to(device)             # per-member scalar -> broadcastable
    # constants in float64 (expm1 for the 1-e terms), then cast: at daily dt
    # 1-exp(-k*dt) ~ 6e-3, and forming it in float32 costs ~3 significant digits.
    d64 = lambda t: t.reshape(K, 1).to(device=device, dtype=torch.float64)
    f32 = lambda t: t.to(torch.float32)
    kf64, ks64, gam64 = d64(kf_v), d64(ks_v), d64(gam_v)
    lam64, xi64, eps64 = d64(lam_v), d64(xi_v), d64(eps_v)
    kf, ks, wf = f32(kf64), f32(ks64), col(wf_v)
    ep2 = f32(eps64 ** 2)
    e_f, e_s = f32(torch.exp(-kf64 * dt)), f32(torch.exp(-ks64 * dt))
    om_f, om_s = f32(-torch.expm1(-kf64 * dt)), f32(-torch.expm1(-ks64 * dt))
    e_lam, e_gam = f32(torch.exp(-lam64 * dt)), f32(torch.exp(-gam64 * dt))
    var_k64 = xi64 * xi64 * (-torch.expm1(-2.0 * gam64 * dt)) / (2.0 * gam64).clamp(min=1e-8)
    std_k = f32(torch.sqrt(var_k64.clamp(min=1e-12)))
    sqrt_dt = math.sqrt(dt)

    zeros = torch.zeros(K, n_eff, device=device)
    v1 = zeros + col(v0_v); v2 = zeros + col(v0_v)
    x = zeros.clone(); kk = zeros.clone()
    Ikey = zeros.clone(); Iois = zeros.clone()
    n_save = int(keep.numel())
    Bkey = torch.zeros(K, n_eff, n_save, device=device)
    Bois = torch.zeros(K, n_eff, n_save, device=device)
    term = payoff == "term"
    if term:                                               # factor state at the fixings
        Bxs = torch.zeros(K, n_eff, n_save, device=device)
        Bks = torch.zeros(K, n_eff, n_save, device=device)

    rho_t = rho_mat[:, :n_steps]
    sqrt_1mr2 = torch.sqrt(torch.clamp(1.0 - rho_t ** 2, min=1e-6))
    draws = _DrawsBatched(K, n_paths, n_steps, device, seed=seed,
                          antithetic=antithetic, chunk=chunk, full=rands)

    for i in range(n_steps):
        if i in keep_set:
            j = col_of[i]
            Bkey[:, :, j] = Ikey
            Bois[:, :, j] = Iois
            if term:
                Bxs[:, :, j] = x
                Bks[:, :, j] = kk
        Ikey = Ikey + (f_key_vec[i] + x + kk) * dt
        Iois = Iois + (f_ois_vec[i] + x) * dt

        Z1, U1, Z2, U2, Zx, Zk = draws.step(i)
        v_i = wf * v1 + (1.0 - wf) * v2
        v1, xi1 = _qe_step(v1, theta_mat[:, i:i + 1], e_f, ep2, kf, Z1, U1, om_e=om_f)
        v2, _ = _qe_step(v2, theta2_mat[:, i:i + 1], e_s, ep2, ks, Z2, U2, om_e=om_s)
        randn_x = rho_t[:, i:i + 1] * xi1 + sqrt_1mr2[:, i:i + 1] * Zx
        x = x * e_lam + torch.sqrt(v_i.clamp(min=1e-9)) * sqrt_dt * randn_x
        kk = kk * e_gam + std_k * Zk

    if n_steps in keep_set:
        Bkey[:, :, col_of[n_steps]] = Ikey
        Bois[:, :, col_of[n_steps]] = Iois

    dIkey = Bkey[:, :, col_pay] - Bkey[:, :, col_fix]
    if term:
        cumf = torch.cat([torch.zeros(1, dtype=torch.float64, device=device),
                          torch.cumsum(f_key_vec[:n_steps].to(torch.float64) * dt, 0)])
        I_det = (cumf[i_pay] - cumf[i_fix]).to(torch.float32)          # (n_caplets,)
        n_win = i_pay - i_fix
        Bx = _window_factor(lam64, n_win, dt).to(torch.float32).unsqueeze(1)   # (K,1,n_cpl)
        Bk = _window_factor(gam64, n_win, dt).to(torch.float32).unsqueeze(1)
        targs = dict(x_fix=Bxs[:, :, col_fix], k_fix=Bks[:, :, col_fix], I_det=I_det, Bx=Bx, Bk=Bk)
    else:
        targs = None
    return _price(payoff, dIkey, Bois[:, :, col_pay], tau_t, K_t, 1, fit_curve,
                  f_key_vec, f_ois_vec, i_fix, i_pay, dt, targs)
