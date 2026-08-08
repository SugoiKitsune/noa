# -*- coding: utf-8 -*-
"""Andersen (2008) QE simulation of the 2-factor lifted CIR Hull-White-Heston model,
with the rate-vol correlation (skew) preserved.

Drop-in alternative to scripts/simulation.fast_simulate_2factor that samples the CIR
variance factors with the Quadratic-Exponential (QE) scheme instead of Euler+reflection.
In the DM low-vol regime the CIR is heavily Feller-violated (2*kappa*theta << eps^2), so
Euler+reflection biases the variance *up* (worst at short maturities), inflating short-end
ATM vols. QE is moment-exact (validated) and, in the exponential branch that dominates
that regime, places an honest point mass at zero instead of reflecting.

Correlation / skew: QE has no single driving Brownian, so we correlate the rate factor x
with the *standardised innovation* of the fast variance factor,
    xi1 = (V1_{t+dt} - m1) / s1   (~N(0,1); exact in the quadratic branch),
via  dW^x = rho(t) * xi1 + sqrt(1 - rho^2) * Z_x.  This recovers the leverage/skew that the
first QE cut dropped, exactly in the quadratic regime and approximately in the exponential.

Interface mirrors fast_simulate_2factor.
"""
import math
import torch

from simulation import fast_ou_paths, fast_hw_paths   # reuse x (OU) and k (HW) generators


def cir_qe_paths(n_paths, n_steps, dt, v0, kappa, theta_vec, eps, device,
                 Z=None, U=None, psi_c=1.5):
    """QE CIR path + standardised innovation. dV = kappa*(theta(t)-V)dt + eps*sqrt(V)dW.
    Returns (v [n,steps+1], xi [n,steps]) where xi ~ N(0,1) is the per-step innovation."""
    v = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    v[:, 0] = float(v0)
    xi = torch.zeros((n_paths, n_steps), dtype=torch.float32, device=device)
    if Z is None:
        Z = torch.randn(n_paths, n_steps, device=device)
    if U is None:
        U = torch.rand(n_paths, n_steps, device=device)
    k = float(kappa); ep2 = float(eps) ** 2
    e = math.exp(-k * dt)
    for i in range(n_steps):
        Vt = v[:, i]
        th = theta_vec[i]
        m = th + (Vt - th) * e
        s2 = Vt * ep2 * e * (1 - e) / k + th * ep2 * (1 - e) ** 2 / (2 * k)
        psi = s2 / (m * m + 1e-30)
        inv = 2.0 / psi.clamp(min=1e-8)
        b2 = (inv - 1 + torch.sqrt((inv * (inv - 1)).clamp(min=0.0))).clamp(min=0.0)
        Vq = (m / (1 + b2)) * (torch.sqrt(b2) + Z[:, i]) ** 2
        p = ((psi - 1) / (psi + 1)).clamp(0.0, 1 - 1e-7)
        beta = (1 - p) / (m + 1e-30)
        Ui = U[:, i]
        Ve = torch.where(
            Ui <= p, torch.zeros_like(Vt),
            torch.log(((1 - p) / (1 - Ui).clamp(min=1e-30))).clamp(min=0.0) / (beta + 1e-30))
        Vnew = torch.where(psi <= psi_c, Vq, Ve).clamp(min=0.0)
        v[:, i + 1] = Vnew
        xi[:, i] = (Vnew - m) / torch.sqrt(s2.clamp(min=1e-30))     # standardised innovation
    return v, xi


def simulate_2factor_qe(n_paths, timeline, theta_vec, epsilon, v0,
                        kf, ks, wf, lam, gam, xi, f_key_vec, f_ois_vec, device,
                        seed=None, rho_vx=None, antithetic=False, theta2_vec=None):
    """QE analogue of fast_simulate_2factor, with rate-vol correlation preserved."""
    n_steps = len(timeline) - 1
    dt = (timeline[1] - timeline[0]).item()
    if seed is not None:
        torch.manual_seed(seed)
    if theta2_vec is None:
        theta2_vec = theta_vec
    wf_f = float(wf)
    rho_t = (rho_vx[:n_steps] if rho_vx is not None else torch.zeros(n_steps, device=device))
    sqrt_1mr2 = torch.sqrt(torch.clamp(1.0 - rho_t ** 2, min=1e-6))

    Z1 = torch.randn(n_paths, n_steps, device=device); U1 = torch.rand(n_paths, n_steps, device=device)
    Z2 = torch.randn(n_paths, n_steps, device=device); U2 = torch.rand(n_paths, n_steps, device=device)
    Zx = torch.randn(n_paths, n_steps, device=device)
    Zk = torch.randn(n_paths, n_steps, device=device)

    def run(z1, u1, z2, u2, zx, zk):
        v1, xi1 = cir_qe_paths(n_paths, n_steps, dt, v0, kf, theta_vec, epsilon, device, z1, u1)
        v2, _ = cir_qe_paths(n_paths, n_steps, dt, v0, ks, theta2_vec, epsilon, device, z2, u2)
        v = wf_f * v1 + (1.0 - wf_f) * v2
        randn_x = rho_t.unsqueeze(0) * xi1 + sqrt_1mr2.unsqueeze(0) * zx   # leverage/skew
        x = fast_ou_paths(n_paths, n_steps, dt, 0.0, lam, v, device, randn_x)
        kk = fast_hw_paths(n_paths, n_steps, dt, 0.0, gam, xi, device, zk)
        key = f_key_vec.unsqueeze(0) + x + kk
        ois = f_ois_vec.unsqueeze(0) + x
        return key, ois, v

    ka, oa, va = run(Z1, U1, Z2, U2, Zx, Zk)
    if antithetic:
        kb, ob, vb = run(-Z1, 1 - U1, -Z2, 1 - U2, -Zx, -Zk)   # mirrored draws
        return torch.cat([ka, kb]), torch.cat([oa, ob]), torch.cat([va, vb])
    return ka, oa, va
