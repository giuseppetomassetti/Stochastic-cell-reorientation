"""
Thermodynamic Analysis of Stochastic Cell Reorientation
========================================================

This script generates Figure E-9 in Section 2 of the Electronic Supplementary Material
for the paper "Using stochastic thermodynamics with internal variables to capture
orientational spreading in cell populations undergoing cyclic stretch".

The script compares three approaches to computing thermodynamic quantities during
the transient regime:
1. Exact solution of the Fokker-Planck equation (black solid lines)
2. Von Mises approximation with extrapolated parameters (blue dashed lines)
3. Von Mises approximation with ODE-predicted parameters (red dotted lines)

Output:
-------
- thermodynamics_comparison_full.jpg: Four-panel figure showing:
  (a) System entropy H(t)
  (b) Non-equilibrium free energy A(t)
  (c) Entropy production rate dH_tot/dt
      * For exact FP: computed using both the flux formula and
        free energy dissipation formula 
      * For VM approximations: computed using free energy dissipation formula
  (d) Cumulative entropy changes (system, bath, and total)

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson, cumulative_trapezoid
from scipy.special import iv
from scipy.optimize import root_scalar

# ==========================================
# Parameters (From Manuscript: Transient Regime)
# ==========================================
epsilon = 1.0
tau = 0.1
k = 1.0
sigma = (tau / epsilon)**2 # = 0.01

N = 200           
L = np.pi         
t_max = 10.0      
theta = np.linspace(0, L, N, endpoint=False) 
dtheta = theta[1] - theta[0]

# Potential U(theta) = 1/2 (cos(2theta) + k)**2
def potential(th):
    return 0.5 * (np.cos(2*th) + k)**2

def potential_prime(th):
    return -2 * (np.cos(2*th) + k) * np.sin(2*th)

U = potential(theta)
U_prime = potential_prime(theta)

# ==========================================
# Fokker-Planck Equation Solver (Scharfetter-Gummel scheme)
# ==========================================
def bernoulli(z):
    """Bernoulli function B(z) = z/(exp(z)-1), with B(0)=1."""
    result = np.empty_like(z, dtype=float)
    small = np.abs(z) < 1e-4
    large_pos = z > 500
    large_neg = z < -500
    mid = ~small & ~large_pos & ~large_neg
    result[small] = 1.0 - z[small] / 2 + z[small]**2 / 12
    result[large_pos] = z[large_pos] * np.exp(-z[large_pos])
    result[large_neg] = -z[large_neg]
    result[mid] = z[mid] / (np.exp(z[mid]) - 1.0)
    return result

# Drift v = -U'(theta) evaluated at cell interfaces theta_{i+1/2}
theta_half = theta + dtheta / 2
v_half = -potential_prime(theta_half)
# Cell Peclet number: Pe = v * dtheta / sigma
Pe_half = v_half * dtheta / sigma
Bp = bernoulli(Pe_half)    # B(Pe)
Bm = bernoulli(-Pe_half)   # B(-Pe)

def fokker_planck(t, p):
    p_ip1 = np.roll(p, -1)  # p_{i+1}, periodic
    # SG flux at interface i+1/2: J_{i+1/2} = (sigma/dtheta)[B(-Pe)*p_i - B(Pe)*p_{i+1}]
    J_half = (sigma / dtheta) * (Bm * p - Bp * p_ip1)
    J_half_im1 = np.roll(J_half, 1)  # J_{i-1/2}
    dpdt = -(J_half - J_half_im1) / dtheta
    return dpdt

def compute_sg_flux(p):
    """Compute SG interface fluxes for a given p (for EPR calculation)."""
    p_ip1 = np.roll(p, -1)
    return (sigma / dtheta) * (Bm * p - Bp * p_ip1)

p0 = np.ones(N) / (N * dtheta)
sol_fp = solve_ivp(fokker_planck, [0, t_max], p0, t_eval=np.linspace(0, t_max, 200), method='Radau')

# ==========================================
# ODE System Solver (Approximate Model)
# ==========================================
def safe_bessel_ratio(v, x):
    """Returns Iv(x)/I0(x) safely for large x."""
    if x < 700:
        return iv(v, x) / iv(0, x)
    else:
        # Asymptotics: Iv(x) ~ e^x / sqrt(2pi x) * (1 - (4v^2-1)/8x + ...)
        # Ratio Iv(x)/I0(x) ~ 1 - v^2 / (2x)
        return 1.0 - (v**2) / (2 * x)

def ode_f(mu, kappa, k_1, k_2):
    if kappa < 1e-4:
        return -4 * k_1 * k_2 * np.sin(2 * mu) / 1e-4 # Avoid singular limit
    elif kappa > 1e4:
        return 2 * k_1 * (np.cos(2 * mu) - k_2) * np.sin(2 * mu)
    else:
        r1 = safe_bessel_ratio(1, kappa/2)
        r2 = safe_bessel_ratio(2, kappa/2)
        term1 = 1 - 2 * r2 / ( (kappa/2) * r1 )
        term2 = 2 / kappa + r2 / r1
        return 2 * k_1 * (term1 * np.cos(2 * mu) - k_2 * term2) * np.sin(2 * mu)

def ode_g(mu, kappa, k_1, k_2):
    if kappa < 1e-4:
        return 8 * k_1 * k_2 * np.cos(2 * mu)
    elif kappa > 1e4:
        return 8 * k_1 * (k_2 * np.cos(2 * mu) - np.cos(4 * mu)) * kappa
    else:
        r1 = safe_bessel_ratio(1, kappa/2)
        r2 = safe_bessel_ratio(2, kappa/2)
        term1 = 1.0 / r1
        term2 = r1
        num = 8 * k_1 * (k_2 * np.cos(2 * mu) - r2 / r1 * np.cos(4 * mu))
        denom = (kappa / 2) * (term1 - term2) - 1
        return num / denom

def ode_h(kappa):
    if kappa < 1e-4:
        return 4 * kappa
    elif kappa > 1e4:
        return 4 * kappa**2
    else:
        r1 = safe_bessel_ratio(1, kappa/2)
        term1 = 1.0 / r1
        term2 = r1
        return 8 / (term1 - term2 - 2 / kappa)

def evolution_ode(t, y, k1, k2, sig):
    mu, kappa = y
    dmu = ode_f(mu, kappa, k1, -k2)
    dkappa = ode_g(mu, kappa, k1, -k2) - sig * ode_h(kappa)
    return [dmu, dkappa]

y0_ode = [np.pi / 2, 1e-5]
sol_ode = solve_ivp(evolution_ode, [0, t_max], y0_ode, args=(1.0, k, sigma), t_eval=sol_fp.t)

# ==========================================
# Thermodynamics Calculation
# ==========================================
# References:
#   Peliti, L., & Pigolotti, S. (2021). Stochastic Thermodynamics: An Introduction.
#   Princeton University Press. See Chapter 3 and 4 for Fokker-Planck thermodynamics.

def estimate_mu_kappa(p_t, theta_grid):
    Z = simpson(p_t, dx=dtheta)
    p_norm = p_t / Z
    c2 = simpson(np.cos(2*theta_grid) * p_norm, dx=dtheta)
    s2 = simpson(np.sin(2*theta_grid) * p_norm, dx=dtheta)
    R = np.hypot(c2, s2)
    mu2 = np.arctan2(s2, c2)
    mu = 0.5 * mu2
    if R < 1e-6:
        kappa = 0.0
    elif R > 0.9999:
        kappa = 2000.0
    else:
        def f_k(kv):
            return safe_bessel_ratio(1, kv/2) - R
        try:
            res = root_scalar(f_k, bracket=[1e-8, 2000], method='bisect')
            kappa = res.root
        except:
            kappa = 0.0
    return mu, kappa

def thermo_from_mu_kappa(mu_val, kappa_val, sigma_val, k_val):
    kp = kappa_val / 2
    if kp < 1e-6:
        s = np.log(np.pi)
        u = 0.5 * (0.5 + k_val**2)
    else:
        # S = ln(pi I0) - kp I1/I0
        # For large kp, I0 ~ e^kp/sqrt(2pi kp), so ln I0 ~ kp - 0.5 ln(2pi kp)
        if kp < 700:
            s = np.log(np.pi * iv(0, kp)) - kp * iv(1, kp) / iv(0, kp)
        else:
            s = np.log(np.pi) + kp - 0.5 * np.log(2 * np.pi * kp) - kp * (1.0 - 1.0/(2*kp))
        
        rat1 = safe_bessel_ratio(1, kp)
        rat2 = safe_bessel_ratio(2, kp)
        u = 0.5 * (0.5 * (1 + rat2 * np.cos(4*mu_val)) + 2*k_val * rat1 * np.cos(2*mu_val) + k_val**2)
    f = u - sigma_val * s
    return s, u, f

times = sol_fp.t
S_ex, U_ex, F_ex = [], [], []
EPR_ex = [] # Entropy Production Rate
EPR_vm = [] # EPR from VM extrapolated
EPR_ode = [] # EPR from VM ODE predicted
S_vm, U_vm, F_vm, K_vm = [], [], [], []
S_ode, U_ode, F_ode, K_ode = [], [], [], []

for i in range(len(times)):
    p_t = np.maximum(sol_fp.y[:, i], 1e-12)
    
    # Exact Thermodynamics
    s_val = -simpson(p_t * np.log(p_t), dx=dtheta)
    u_val = simpson(p_t * U, dx=dtheta)
    S_ex.append(s_val)
    U_ex.append(u_val)
    F_ex.append(u_val - sigma * s_val)
    
    J_half = compute_sg_flux(p_t)
    p_half = 0.5 * (p_t + np.roll(p_t, -1))  # interface density
    integrand = (J_half**2) / (sigma * p_half)
    ep_val = np.sum(integrand) * dtheta
    EPR_ex.append(ep_val)
    
    # VM Extrapolated
    mu_e, kappa_e = estimate_mu_kappa(p_t, theta)
    K_vm.append(kappa_e)
    s, u, f = thermo_from_mu_kappa(mu_e, kappa_e, sigma, k)
    S_vm.append(s); U_vm.append(u); F_vm.append(f)

    # ODE Predicted
    mu_o, kappa_o = sol_ode.y[0, i], sol_ode.y[1, i]
    K_ode.append(kappa_o)
    s, u, f = thermo_from_mu_kappa(mu_o, kappa_o, sigma, k)
    S_ode.append(s); U_ode.append(u); F_ode.append(f)

# EPR from free energy dissipation: S_tot_dot = -dF/dt / sigma
# This is thermodynamically consistent and vanishes at equilibrium.
F_vm_arr = np.array(F_vm)
F_ode_arr = np.array(F_ode)
F_ex_arr = np.array(F_ex)
EPR_vm = np.maximum(-np.gradient(F_vm_arr, times) / sigma, 1e-12)
EPR_ode = np.maximum(-np.gradient(F_ode_arr, times) / sigma, 1e-12)

# For exact FP: also compute EPR from free energy to validate against flux formula
EPR_ex_from_F = np.maximum(-np.gradient(F_ex_arr, times) / sigma, 1e-12)
EPR_ex_arr = np.array(EPR_ex)

# Validation: compare flux-based EPR with free energy-based EPR for exact solution
# Use relative difference only when EPR is significant (> 1% of peak)
EPR_peak = np.max(EPR_ex_arr)
threshold = 0.01 * EPR_peak  # 1% of peak value

# Compute relative error only for significant EPR values
mask = EPR_ex_arr > threshold
if np.any(mask):
    relative_diff_significant = np.abs(EPR_ex_arr[mask] - EPR_ex_from_F[mask]) / EPR_ex_arr[mask]
    max_rel_diff = np.max(relative_diff_significant)
else:
    max_rel_diff = 0.0

print(f"\nValidation: Comparing flux-based EPR with free-energy-based EPR")
print(f"=" * 70)
print(f"EPR peak value: {EPR_peak:.2e}")
print(f"Max relative difference (for EPR > {threshold:.2e}): {max_rel_diff:.2%}")
print(f"  EPR from flux (Peliti-Pigolotti): min={np.min(EPR_ex_arr):.2e}, max={np.max(EPR_ex_arr):.2e}")
print(f"  EPR from free energy dissipation: min={np.min(EPR_ex_from_F):.2e}, max={np.max(EPR_ex_from_F):.2e}")

# Check agreement at different time points
print(f"\nTime point analysis:")
for i in [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]:
    t = times[i]
    epr_flux = EPR_ex_arr[i]
    epr_free = EPR_ex_from_F[i]
    if epr_flux > threshold:
        rel_diff = abs(epr_flux - epr_free) / epr_flux
        print(f"  t={t:.2e}: EPR_flux={epr_flux:.3e}, EPR_free={epr_free:.3e}, rel_diff={rel_diff:.2%}")
    else:
        abs_diff = abs(epr_flux - epr_free)
        print(f"  t={t:.2e}: EPR_flux={epr_flux:.3e}, EPR_free={epr_free:.3e}, abs_diff={abs_diff:.2e} (near equilibrium)")

if max_rel_diff < 0.10:
    print(f"\n✓ Both methods agree excellently for significant EPR values (relative difference < 10%)")
    print(f"  Small absolute differences near equilibrium are expected due to numerical noise.")
else:
    print(f"\n⚠ Relative difference of {max_rel_diff:.2%} exceeds 10% threshold")
print("=" * 70)

# Cumulative Changes (Exact)
S_ex_arr = np.array(S_ex)
U_ex_arr = np.array(U_ex)
Delta_S_sys = S_ex_arr - S_ex_arr[0]
# Delta S_bath = - Delta Q / T = - Delta U / T (since W=0)
Delta_S_bath = -(U_ex_arr - U_ex_arr[0]) / sigma
Delta_S_tot = Delta_S_sys + Delta_S_bath

# ==========================================
# Plotting
# ==========================================
# Set larger font sizes for better readability
plt.rcParams.update({'font.size': 12, 'legend.fontsize': 14})

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. System Entropy H(t)
axs[0, 0].plot(times, S_ex, 'k-', lw=2, label="Exact FP")
axs[0, 0].plot(times, S_vm, 'b--', lw=2, label="VM (Extrapolated)")
axs[0, 0].plot(times, S_ode, 'r:', lw=2, label="VM (ODE Predicted)")
axs[0, 0].set_title(r"(a) System Entropy $H(t)$", fontsize=14)
axs[0, 0].set_xscale('log')
axs[0, 0].set_xlabel("Time", fontsize=12)
axs[0, 0].set_ylabel(r"$H$", fontsize=12)
axs[0, 0].legend(fontsize=12)
axs[0, 0].grid(True, which="both", ls="-", alpha=0.3)

# 2. Free Energy F(t)
axs[0, 1].plot(times, F_ex, 'k-', lw=2, label="Exact FP")
axs[0, 1].plot(times, F_vm, 'b--', lw=2, label="VM (Extrapolated)")
axs[0, 1].plot(times, F_ode, 'r:', lw=2, label="VM (ODE Predicted)")
axs[0, 1].set_title(r"(b) Free Energy ${A}(t)$", fontsize=14)
axs[0, 1].set_xscale('log')
axs[0, 1].set_xlabel("Time", fontsize=12)
axs[0, 1].set_ylabel(r"$\mathcal{A}$", fontsize=12)
axs[0, 1].legend(fontsize=12)
axs[0, 1].grid(True, which="both", ls="-", alpha=0.3)

# 3. Entropy Production Rate
# For exact FP: plot both flux-based (Peliti-Pigolotti) and free-energy-based formulas
axs[1, 0].plot(times, EPR_ex, 'k-', lw=2, label="Exact FP (flux)", zorder=3)
axs[1, 0].plot(times, EPR_ex_from_F, 'k--', lw=1.5, alpha=0.7, label="Exact FP (free energy)", zorder=2)
# For VM approximations: use free energy formula (more robust)
axs[1, 0].plot(times, EPR_vm, 'b--', lw=2, label="VM (Extrapolated)", zorder=1)
axs[1, 0].plot(times, EPR_ode, 'r:', lw=2, label="VM (ODE Predicted)", zorder=1)
axs[1, 0].set_title(r"(c) Entropy Production Rate $\dot{H}_{tot}$", fontsize=14)
axs[1, 0].set_xscale('log')
# axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel("Time", fontsize=12)
axs[1, 0].set_ylabel(r"$\dot{H}_{tot}$", fontsize=12)
axs[1, 0].legend(fontsize=12)
axs[1, 0].grid(True, which="both", ls="-", alpha=0.3)

# 4. Cumulative Entropy Changes (The Balance)
axs[1, 1].plot(times, Delta_S_sys, 'g-', lw=2, label=r"$\Delta H_{sys}$")
axs[1, 1].plot(times, Delta_S_bath, 'b--', lw=2, label=r"$\Delta H_{bath}$")
axs[1, 1].plot(times, Delta_S_tot, 'r-', lw=3, label=r"$\Delta H_{tot}$")
axs[1, 1].set_title(r"(d) Cumulative Entropy Changes $\Delta H(t)$", fontsize=14)
axs[1, 1].set_xscale('log')
axs[1, 1].set_xlabel("Time", fontsize=12)
axs[1, 1].set_ylabel(r"$\Delta H$", fontsize=12)
axs[1, 1].legend(loc='best', fontsize=12)
axs[1, 1].grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
output_path = "thermodynamics_comparison_full.jpg"
plt.savefig(output_path)
print(f"Simulation complete. Full comparison plot saved to {output_path}")
# plt.show()  # Commented out - using Agg backend for non-interactive execution