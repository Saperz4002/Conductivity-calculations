"""
Fermi contours, resonant-transition curves, and JDOS
====================================================

Overview
--------
This script analyzes the interband transition geometry of a two-dimensional
Rashba--altermagnetic continuum model.

The band energies are

    epsilon_lambda(k, phi) = t0 k^2 + lambda d(k, phi),

where lambda = +1 or -1 and

    d(k, phi) =
        sqrt[
            alpha^2 k^2
            + tj^2 k^4 sin^2(2 phi)
        ].

For a fixed chemical potential, the program calculates the two Fermi
contours q_+(phi) and q_-(phi). These contours delimit the momentum-space
region in which vertical interband transitions are Pauli allowed.

The script also determines the resonant curves C_r(omega), defined by

    2 d(k, phi) = hbar omega.

These curves contain all momentum points whose band splitting matches a
selected photon energy.

Main calculations
-----------------
1. Fermi contours
   The radial wavevectors q_+(phi) and q_-(phi) are obtained numerically by
   locating the roots of

       epsilon_+(k, phi) = mu
       epsilon_-(k, phi) = mu.

2. Interband transition boundaries
   The energies evaluated on the Fermi contours define the lower and upper
   angular transition boundaries. Their global extrema give the
   characteristic energies omega_+ and omega_-.

3. Joint density of states
   The zero-temperature Pauli window

       epsilon_-(k, phi) < mu < epsilon_+(k, phi)

   is evaluated on a polar momentum grid. The allowed transition energies
   2 d(k, phi) are accumulated into a histogram to obtain the normalized
   joint density of states (JDOS).

4. Characteristic JDOS peaks
   A simple local-maximum search identifies two representative peak
   energies, omega_p1 and omega_p2. When fewer than two local maxima are
   found, the global JDOS maximum is used as a fallback.

5. Resonant curves
   For omega_+, omega_p1, omega_p2, and omega_-, the radial equation
   2 d(k, phi) = hbar omega is solved analytically for k^2. The resulting
   curves are plotted together with the two Fermi contours.

Outputs
-------
The script creates a two-panel figure containing:

- the Fermi contours and selected resonant curves in momentum space;
- the normalized JDOS with the four characteristic energies marked.

It saves:

- ``fermi_contours_resonant_curves_and_jdos.png``;
- ``selected_resonant_energies.csv``.

Both files are written inside the directory specified by ``OUTDIR``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Fixed parameters
# =========================================================
t0 = 1 / (2 * 0.152)      # eV*A^2
mu = 5.5e-3               # eV = 5.5 meV
alpha = 0.026             # eV*A = 26 meV*A
tj = 1.881                # eV*A^2

omega_max_meV = 30.0
nomega = 700

kmax = 0.15
nk = 900
nphi = 721

k_grid = np.linspace(1e-7, kmax, nk)
phi_grid = np.linspace(0.0, 2*np.pi, nphi, endpoint=False)

OUTDIR = "resonant_curves_rashba_altermagnet"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "legend.fontsize": 11,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
})

# =========================================================
# Model
# =========================================================
def d_of(k, phi, alpha_val=alpha, tj_val=tj):
    return np.sqrt(alpha_val**2 * k**2 + tj_val**2 * k**4 * np.sin(2*phi)**2)

def eps(k, phi, lam, alpha_val=alpha, tj_val=tj):
    return t0 * k**2 + lam * d_of(k, phi, alpha_val, tj_val)

# =========================================================
# Root finder
# =========================================================
def root_from_samples(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    idx = np.where(y[:-1] * y[1:] <= 0)[0]
    if len(idx) == 0:
        return np.nan

    i = idx[0]
    x1, x2 = x[i], x[i+1]
    y1, y2 = y[i], y[i+1]

    if y2 == y1:
        return 0.5 * (x1 + x2)

    return x1 - y1 * (x2 - x1) / (y2 - y1)

# =========================================================
# Fermi contours q_+(phi), q_-(phi)
# =========================================================
def fermi_wavevectors_vs_phi():
    q_plus = np.full_like(phi_grid, np.nan, dtype=float)
    q_minus = np.full_like(phi_grid, np.nan, dtype=float)

    for i, phi in enumerate(phi_grid):
        f_plus = eps(k_grid, phi, +1) - mu
        f_minus = eps(k_grid, phi, -1) - mu

        q_plus[i] = root_from_samples(k_grid, f_plus)
        q_minus[i] = root_from_samples(k_grid, f_minus)

    return q_plus, q_minus

# =========================================================
# Angular boundaries Omega_low/high(phi)
# =========================================================
def transition_boundaries():
    q_plus, q_minus = fermi_wavevectors_vs_phi()

    Omega_plus = 2.0 * d_of(q_plus, phi_grid)   # eV
    Omega_minus = 2.0 * d_of(q_minus, phi_grid) # eV

    Omega_low = np.minimum(Omega_plus, Omega_minus)
    Omega_high = np.maximum(Omega_plus, Omega_minus)

    return q_plus, q_minus, Omega_low, Omega_high

# =========================================================
# JDOS
# =========================================================
def gaussian_smooth(y, sigma_bins=3.0):
    if sigma_bins is None or sigma_bins <= 0:
        return y
    half_width = int(np.ceil(4 * sigma_bins))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins)**2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")

def compute_jdos():
    K, PHI = np.meshgrid(k_grid, phi_grid, indexing="xy")
    D = d_of(K, PHI)

    eps_plus = t0 * K**2 + D
    eps_minus = t0 * K**2 - D

    W = (eps_minus < mu) & (mu < eps_plus)
    transition_energy_meV = 1e3 * (2.0 * D)

    dk = k_grid[1] - k_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    weights = K * dk * dphi / (2*np.pi)**2

    bin_edges = np.linspace(0.0, omega_max_meV, nomega + 1)
    hist, edges = np.histogram(
        transition_energy_meV[W],
        bins=bin_edges,
        weights=weights[W]
    )

    omega_meV = 0.5 * (edges[:-1] + edges[1:])
    bin_width_eV = (edges[1] - edges[0]) / 1e3
    jdos = hist / bin_width_eV
    jdos_smooth = gaussian_smooth(jdos, sigma_bins=3.0)

    if np.nanmax(jdos_smooth) > 0:
        jdos_norm = jdos_smooth / np.nanmax(jdos_smooth)
    else:
        jdos_norm = jdos_smooth

    return omega_meV, jdos_norm

# =========================================================
# Resonant curve k_res(phi; omega)
# Solve 2 d(k,phi) = hbar omega
# =========================================================
def k_resonant(phi_vals, omega_meV, alpha_val=alpha, tj_val=tj):
    """
    Returns k_res(phi) in A^{-1} for a fixed omega_meV.

    Solve:
        2 sqrt(alpha^2 k^2 + tj^2 k^4 sin^2(2phi)) = hbar omega
    Let x = k^2:
        A x^2 + alpha^2 x - (omega/2)^2 = 0
    with A = tj^2 sin^2(2phi)
    """
    omega_eV = omega_meV / 1e3
    target = (omega_eV / 2.0)**2

    s2 = np.sin(2 * phi_vals)**2
    A = tj_val**2 * s2

    x = np.full_like(phi_vals, np.nan, dtype=float)

    small = A < 1e-14
    regular = ~small

    # If A ~ 0, then alpha^2 x = target
    x[small] = target / (alpha_val**2)

    # Otherwise solve quadratic in x = k^2
    disc = alpha_val**4 + 4.0 * A[regular] * target
    x_reg = (-alpha_val**2 + np.sqrt(disc)) / (2.0 * A[regular])

    x[regular] = x_reg
    x[x < 0] = np.nan

    return np.sqrt(x)

# =========================================================
# Simple local maxima finder
# =========================================================
def local_maxima_indices(y):
    idx = []
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            idx.append(i)
    return np.array(idx, dtype=int)

# =========================================================
# Plot
# =========================================================
def plot_resonant_curves():
    q_plus, q_minus, Omega_low, Omega_high = transition_boundaries()
    omega_jdos, jdos_norm = compute_jdos()

    omega_plus = np.nanmin(1e3 * Omega_low)
    omega_minus = np.nanmax(1e3 * Omega_high)

    max_idx = np.argmax(jdos_norm)
    omega_peak_global = omega_jdos[max_idx]

    lm = local_maxima_indices(jdos_norm)
    if len(lm) >= 1:
        # choose first local max
        omega_peak1 = omega_jdos[lm[0]]
    else:
        omega_peak1 = omega_peak_global

    # choose second meaningful energy if available
    if len(lm) >= 2:
        omega_peak2 = omega_jdos[lm[1]]
    else:
        omega_peak2 = omega_peak_global

    energies = [
        (omega_plus, r"$\omega_{+}$"),
        (omega_peak1, r"$\omega_{p1}$"),
        (omega_peak2, r"$\omega_{p2}$"),
        (omega_minus, r"$\omega_{-}$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))

    # ---------------------------------
    # (a) Fermi contours + resonant curves
    # ---------------------------------
    ax = axes[0]

    x_qp = q_plus * np.cos(phi_grid)
    y_qp = q_plus * np.sin(phi_grid)
    x_qm = q_minus * np.cos(phi_grid)
    y_qm = q_minus * np.sin(phi_grid)

    ax.plot(x_qp, y_qp, color="k", lw=1.8, ls="--", label=r"$q_{+}(\phi)$")
    ax.plot(x_qm, y_qm, color="k", lw=1.8, ls="-",  label=r"$q_{-}(\phi)$")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    rows = []

    for (om, lab), c in zip(energies, colors):
        kres = k_resonant(phi_grid, om)
        x = kres * np.cos(phi_grid)
        y = kres * np.sin(phi_grid)

        ax.plot(x, y, color=c, lw=2.0, label=fr"$C_r({lab})$")

        rows.append({"label": lab, "omega_meV": om})

    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x\ (\mathrm{\AA^{-1}})$")
    ax.set_ylabel(r"$k_y\ (\mathrm{\AA^{-1}})$")
    ax.text(0.04, 0.93, r"(a)", transform=ax.transAxes, fontsize=18)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7")

    # ---------------------------------
    # (b) JDOS with markers
    # ---------------------------------
    ax2 = axes[1]
    ax2.plot(omega_jdos, jdos_norm, color="k", lw=2.3)
    ax2.fill_between(omega_jdos, 0, jdos_norm, color="0.90")

    for (om, lab), c in zip(energies, colors):
        y = np.interp(om, omega_jdos, jdos_norm)
        ax2.axvline(om, color=c, ls="--", lw=1.3)
        ax2.plot([om], [y], "o", color=c, ms=6)
        ax2.text(om, min(1.03, y + 0.08), lab, color=c, ha="center", va="bottom", fontsize=12)

    ax2.set_xlim(0, omega_max_meV)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel(r"$\hbar\omega\ (\mathrm{meV})$")
    ax2.set_ylabel(r"$\mathrm{JDOS}/\max$")
    ax2.text(0.04, 0.93, r"(b)", transform=ax2.transAxes, fontsize=18)

    fig.suptitle(
        fr"$\mu={1e3*mu:.1f}\,\mathrm{{meV}},\ "
        fr"\alpha={1e3*alpha:.0f}\,\mathrm{{meV\AA}},\ "
        fr"t_j={tj:.3f}\,\mathrm{{eV\AA^2}}$",
        y=1.01,
        fontsize=14
    )
    fig.tight_layout()

    fig_path = os.path.join(OUTDIR, "fermi_contours_resonant_curves_and_jdos.png")
    fig.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df_path = os.path.join(OUTDIR, "selected_resonant_energies.csv")
    df.to_csv(df_path, index=False)

    print("Saved figure:")
    print(fig_path)
    print("Saved selected energies:")
    print(df_path)

    print(f"omega_+  = {omega_plus:.4f} meV")
    print(f"omega_p1 = {omega_peak1:.4f} meV")
    print(f"omega_p2 = {omega_peak2:.4f} meV")
    print(f"omega_-  = {omega_minus:.4f} meV")

if __name__ == "__main__":
    plot_resonant_curves()
