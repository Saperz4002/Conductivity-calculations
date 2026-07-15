"""
Density of states of Rashba and d-wave altermagnetic models
===========================================================

Overview
--------
This script calculates the band-resolved and total density of states (DOS)
of a two-dimensional continuum model containing a quadratic kinetic term,
Rashba spin-orbit coupling, and a d_xy altermagnetic interaction.

The band energies are

    E_±(k, phi) = t0 k^2
                  ± sqrt[
                      alpha^2 k^2
                      + tj^2 k^4 sin^2(2 phi)
                    ].

For each selected parameter set, the code finds the allowed momentum roots
at a given energy, integrates their contributions over the polar angle, and
plots

    rho_+(E), rho_-(E), and rho(E) = rho_+(E) + rho_-(E).

Available sweep modes
---------------------
1. ``"pure_rashba"``
   Sweeps the Rashba strength ``alpha`` while setting ``tj = 0``.

2. ``"pure_altermagnetic"``
   Sweeps the altermagnetic coupling ``tj`` while setting ``alpha = 0``.

3. ``"mixed_alpha_sweep"``
   Sweeps ``alpha`` while keeping ``tj`` fixed at ``tj_ref``.

4. ``"mixed_tj_sweep"``
   Sweeps ``tj`` while keeping ``alpha`` fixed at ``alpha_ref``.

5. ``"all"``
   Runs all four sweep families.

Available plot modes
--------------------
1. ``"single"``
   Produces one energy window at a time. The selected interval is controlled
   by ``energy_window``, which may be ``"negative"`` or ``"positive"``.

2. ``"combined"``
   Produces a two-panel figure joining the negative- and positive-energy
   sectors. When Rashba coupling is present, the negative-energy panel also
   marks the van Hove energy.

Outputs
-------
Each generated DOS figure is saved as a high-resolution PNG file inside the
directory specified by ``output_dir``.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning


# ============================================================
# User configuration
# ============================================================

sweep_mode = "mixed_tj_sweep"
plot_mode = "combined"          # "single" or "combined"
energy_window = "negative"      # only used if plot_mode = "single"

t0_ref = 3.2895                 # eV Å^2
tj_ref = 1.881                  # eV Å^2
alpha_ref = 0.052               # eV Å = 52 meV Å

Emax = 0.04                     # eV = 40 meV
N_ENERGY = 1000

negative_edge_fraction = 1e-3

YMIN_DOS = 0.0
YMAX_DOS = 0.50

output_dir = "dos_sweep_figures"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Model
# ============================================================

def energy_band(k, phi, band, t0, tj, alpha):
    s2 = np.sin(2.0 * phi) ** 2
    delta = np.sqrt(alpha**2 * k**2 + tj**2 * k**4 * s2)
    return t0 * k**2 + band * delta


def lower_band_minimum(t0, tj, alpha):
    if abs(alpha) < 1e-14:
        return 0.0

    if t0 <= abs(tj):
        raise ValueError("The continuum model is unstable for t0 <= |tj|.")

    if abs(tj) < 1e-14:
        return -alpha**2 / (4.0 * t0)

    return -alpha**2 / (2.0 * (t0 + np.sqrt(t0**2 - tj**2)))

def van_hove_energy(t0, alpha):
    if abs(alpha) < 1e-14:
        return None
    return -alpha**2 / (4.0 * t0)

def energy_grid(t0, tj, alpha, nE=N_ENERGY):
    Emin = lower_band_minimum(t0, tj, alpha)

    if energy_window == "negative" and abs(alpha) > 1e-14:
        left = Emin + negative_edge_fraction * abs(Emin)
        right = -negative_edge_fraction * abs(Emin)

        energies = np.linspace(left, right, nE)
        xlim_meV = (1000.0 * Emin, 0.0)
        suffix = "_negative"

    elif energy_window == "positive":
        energies = np.linspace(0.0, Emax, nE)
        xlim_meV = (0.0, 1000.0 * Emax)
        suffix = "_positive"

    else:
        raise ValueError("Use energy_window = 'negative' or 'positive'.")

    return energies, xlim_meV, suffix, Emin


# ============================================================
# DOS machinery
# ============================================================

def k_roots(phi, energy, band, t0, tj, alpha):
    s2 = np.sin(2.0 * phi) ** 2
    den = 2.0 * (t0**2 - tj**2 * s2)

    if den <= 0:
        return []

    rad = alpha**4 + 4.0 * t0 * alpha**2 * energy + 4.0 * tj**2 * energy**2 * s2

    if rad < 0:
        return []

    roots = []

    for eta in (+1, -1):
        k2 = (alpha**2 + 2.0 * t0 * energy + eta * np.sqrt(rad)) / den

        if k2 <= 0:
            continue

        k = np.sqrt(k2)
        residual = abs(energy_band(k, phi, band, t0, tj, alpha) - energy)

        if residual < 1e-8:
            roots.append(k)

    unique_roots = []

    for k in roots:
        if not any(abs(k - ku) < 1e-8 for ku in unique_roots):
            unique_roots.append(k)

    return unique_roots


def dE_dk(k, phi, band, t0, tj, alpha):
    s2 = np.sin(2.0 * phi) ** 2
    delta = np.sqrt(alpha**2 * k**2 + tj**2 * k**4 * s2)

    if delta == 0:
        return np.nan

    ddelta_dk = (alpha**2 * k + 2.0 * tj**2 * k**3 * s2) / delta
    return 2.0 * t0 * k + band * ddelta_dk


def dos_integrand(phi, energy, band, t0, tj, alpha):
    roots = k_roots(phi, energy, band, t0, tj, alpha)

    value = 0.0

    for k in roots:
        derivative = dE_dk(k, phi, band, t0, tj, alpha)

        if np.isfinite(derivative) and abs(derivative) > 1e-14:
            value += k / abs(derivative)

    return value


def rho_band(energy, band, t0, tj, alpha):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)

            integral, _ = quad(
                lambda phi: dos_integrand(phi, energy, band, t0, tj, alpha),
                0.0,
                np.pi / 2.0,
                limit=400,
            )

        prefactor = 4.0 / (2.0 * np.pi) ** 2
        rho = prefactor * integral

        return rho if np.isfinite(rho) else np.nan

    except Exception:
        return np.nan


def compute_dos(energies, t0, tj, alpha):
    rho_plus = np.array([rho_band(E, +1, t0, tj, alpha) for E in energies])
    rho_minus = np.array([rho_band(E, -1, t0, tj, alpha) for E in energies])
    rho_total = rho_plus + rho_minus

    return rho_plus, rho_minus, rho_total


# ============================================================
# Sweep cases
# ============================================================

def build_cases():
    alpha_values = np.array([7, 13, 26, 52, 100]) * 1e-3
    tj_values = np.array([0.25, 0.50, 0.75, 1.00, 1.25]) * tj_ref

    cases = []

    if sweep_mode in ("pure_rashba", "all"):
        for alpha in alpha_values:
            cases.append({
                "name": f"pure_rashba_alpha_{1000 * alpha:.0f}_meVA",
                "title": "Pure Rashba",
                "t0": t0_ref,
                "tj": 0.0,
                "alpha": alpha,
            })

    if sweep_mode in ("pure_altermagnetic", "all"):
        for tj in tj_values:
            cases.append({
                "name": f"pure_altermagnetic_tj_{tj:.3f}",
                "title": "Pure d-wave altermagnet",
                "t0": t0_ref,
                "tj": tj,
                "alpha": 0.0,
            })

    if sweep_mode in ("mixed_alpha_sweep", "all"):
        for alpha in alpha_values:
            cases.append({
                "name": f"mixed_fixed_tj_alpha_{1000 * alpha:.0f}_meVA",
                "title": "Rashba + d-wave altermagnet",
                "t0": t0_ref,
                "tj": tj_ref,
                "alpha": alpha,
            })

    if sweep_mode in ("mixed_tj_sweep", "all"):
        for tj in tj_values:
            cases.append({
                "name": f"mixed_fixed_alpha_tj_{tj:.3f}",
                "title": "Rashba + d-wave altermagnet",
                "t0": t0_ref,
                "tj": tj,
                "alpha": alpha_ref,
            })

    return cases


# ============================================================
# Plot styling
# ============================================================

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
    })


def style_axes(ax):
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in", length=5, width=1.1, labelsize=11)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", length=3, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_curves(ax, energies_meV, rho_plus, rho_minus, rho_total, show_legend=True):
    ax.plot(energies_meV, rho_plus, color="red", lw=2.0, label=r"$\rho_{+}(E)$")
    ax.plot(energies_meV, rho_minus, color="blue", lw=2.0, label=r"$\rho_{-}(E)$")
    ax.plot(energies_meV, rho_total, color="black", lw=2.0, label=r"$\rho(E)$")

    if show_legend:
        ax.legend(frameon=False, fontsize=10, loc="upper right")


def add_tj_box(ax, tj, x=0.58, y=0.68):
    ax.text(
        x,
        y,
        rf"$t_j = {tj:.4f}\ \mathrm{{eV\,\AA^2}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="square",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
        ),
    )


def add_full_parameter_box(ax, t0, tj, alpha):
    ax.text(
        0.04,
        0.96,
        rf"$t_0 = {t0:.4f}\ \mathrm{{eV\,\AA^2}}$" + "\n"
        rf"$t_j = {tj:.4f}\ \mathrm{{eV\,\AA^2}}$" + "\n"
        rf"$\alpha = {1000 * alpha:.1f}\ \mathrm{{meV\,\AA}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="square",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
        ),
    )


# ============================================================
# Single-window figure
# ============================================================

def plot_single_case(case):
    t0 = case["t0"]
    tj = case["tj"]
    alpha = case["alpha"]

    energies, xlim_meV, suffix, Emin = energy_grid(t0, tj, alpha)
    rho_plus, rho_minus, rho_total = compute_dos(energies, t0, tj, alpha)

    energies_meV = 1000.0 * energies

    figsize = (3.6, 3.4) if energy_window == "negative" else (4.8, 3.4)

    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    plot_curves(ax, energies_meV, rho_plus, rho_minus, rho_total)
    style_axes(ax)

    ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.set_xlim(*xlim_meV)
    ax.set_ylim(YMIN_DOS, YMAX_DOS)
    ax.set_xlabel(r"$E\;(\mathrm{meV})$", fontsize=14)

    if energy_window == "positive":
        ax.set_ylabel(r"$\rho(E)\;(\mathrm{eV}^{-1}\mathrm{\AA}^{-2})$", fontsize=14)
        add_full_parameter_box(ax, t0, tj, alpha)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        add_tj_box(ax, tj, x=0.04, y=0.96)

    plt.tight_layout()

    filename = os.path.join(output_dir, case["name"] + suffix + ".png")
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# Combined negative + positive figure
# ============================================================

def plot_combined_case(case):
    t0 = case["t0"]
    tj = case["tj"]
    alpha = case["alpha"]

    Emin = lower_band_minimum(t0, tj, alpha)

    if Emin >= 0:
        print(f"Skipped {case['name']}: no negative-energy sector.")
        return

    eps_neg = np.linspace(
        Emin + negative_edge_fraction * abs(Emin),
        -negative_edge_fraction * abs(Emin),
        N_ENERGY,
    )

    eps_pos = np.linspace(0.0, Emax, N_ENERGY)

    rho_p_neg, rho_m_neg, rho_t_neg = compute_dos(eps_neg, t0, tj, alpha)
    rho_p_pos, rho_m_pos, rho_t_pos = compute_dos(eps_pos, t0, tj, alpha)

    eps_neg_meV = 1000.0 * eps_neg
    eps_pos_meV = 1000.0 * eps_pos

    set_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.6),
        dpi=300,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.0},
    )

    plot_curves(ax1, eps_neg_meV, rho_p_neg, rho_m_neg, rho_t_neg, show_legend=False)
    plot_curves(ax2, eps_pos_meV, rho_p_pos, rho_m_pos, rho_t_pos, show_legend=True)

    ax1.set_xlim(1000.0 * Emin, 0.0)
    ax2.set_xlim(0.0, 1000.0 * Emax)

    EvH = van_hove_energy(t0, alpha)

    if EvH is not None:
        EvH_meV = 1000.0 * EvH

        vh_line = ax1.axvline(
            EvH_meV,
            color="gray",
            ls="--",
            lw=1.3,
            alpha=0.9,
            label=rf"$E_{{\mathrm{{vH}}}} = {EvH_meV:.4f}\ \mathrm{{meV}}$"
        )

        ax1.legend(
            handles=[vh_line],
            frameon=False,
            fontsize=10,
            loc="upper center"
        )

    ax1.set_ylim(YMIN_DOS, YMAX_DOS)

    fig.supxlabel(r"$E\;(\mathrm{meV})$", fontsize=14, y=-0.02)

    ax1.set_ylabel(r"$\rho(E)\;(\mathrm{eV}^{-1}\mathrm{\AA}^{-2})$", fontsize=14)

    add_tj_box(ax2, tj, x=0.58, y=0.68)

    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2.spines["left"].set_visible(False)

    pos_ticks = np.linspace(0.0, 1000.0 * Emax, 5)
    ax2.set_xticks(pos_ticks)
    ax2.set_xticklabels([""] + [f"{x:.0f}" for x in pos_ticks[1:]])

    for ax in (ax1, ax2):
        style_axes(ax)

    filename = os.path.join(output_dir, case["name"] + "_combined.png")
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# Run
# ============================================================

def main():
    cases = build_cases()

    for case in cases:
        if plot_mode == "single":
            plot_single_case(case)
        elif plot_mode == "combined":
            plot_combined_case(case)
        else:
            raise ValueError("Use plot_mode = 'single' or 'combined'.")


if __name__ == "__main__":
    main()