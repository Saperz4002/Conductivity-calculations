"""
Interband transition windows and JDOS parameter sweeps
======================================================

Overview
--------
This script studies the direct interband transition phase space of a
two-dimensional Rashba--altermagnetic continuum model.

The band energies are

    epsilon_lambda(k, phi) = t0 k^2 + lambda d(k, phi),

where lambda = +1 or -1 and

    d(k, phi) =
        sqrt[
            alpha^2 k^2
            + tj^2 k^4 sin^2(2 phi)
        ].

For a fixed chemical potential, the code calculates the two Fermi
wavevectors q_+(phi) and q_-(phi). Evaluating the band splitting at these
contours gives the angular boundaries of the allowed direct interband
transition window,

    hbar Omega_+(phi) = 2 d(q_+(phi), phi),
    hbar Omega_-(phi) = 2 d(q_-(phi), phi).

The lower and upper boundaries are defined as

    Omega_low(phi)  = min[Omega_+(phi), Omega_-(phi)],
    Omega_high(phi) = max[Omega_+(phi), Omega_-(phi)].

The script also evaluates a JDOS-like quantity for Pauli-allowed vertical
interband transitions,

    JDOS(omega) =
        integral d^2k / (2 pi)^2
        W(k) delta[2 d(k) - hbar omega],

where the zero-temperature occupation window is

    W(k) = 1  when  epsilon_-(k) < mu < epsilon_+(k),
           0  otherwise.

The delta function is approximated numerically by a weighted histogram of
transition energies on a polar momentum grid.

Parameter sweeps
----------------
The program performs two sweeps:

1. Rashba-coupling sweep
   The values in ``alpha_values`` are evaluated while ``tj`` and ``mu``
   remain fixed.

2. Altermagnetic-coupling sweep
   The values in ``tj_values`` are evaluated while ``alpha`` and ``mu``
   remain fixed.

For every parameter set, the code calculates:

- q_+(phi) and q_-(phi);
- Omega_+(phi) and Omega_-(phi);
- the lower and upper transition-window boundaries;
- the raw, smoothed, and normalized JDOS;
- the global transition edges;
- the first two meaningful JDOS peaks, when present;
- the fraction of angular points for which Fermi roots are missing.

Figures
-------
Two separate two-panel figures are produced:

1. Alpha sweep
   Top panel: angular transition windows.
   Bottom panel: normalized JDOS curves.

2. tj sweep
   Top panel: angular transition windows.
   Bottom panel: normalized JDOS curves.

The transition-window panels may optionally be shaded using
``FILL_WINDOWS``. The horizontal plotting range is controlled by
``plot_xmax_meV``, while the JDOS histogram may extend to the larger range
set by ``omega_max_meV``.

Outputs
-------
The script saves:

- one PNG and one PDF figure for the alpha sweep;
- one PNG and one PDF figure for the tj sweep;
- a CSV summary of transition edges and selected JDOS peaks;
- a CSV file containing all angular transition boundaries and Fermi roots;
- a CSV file containing the raw, smoothed, and normalized JDOS data.

All files are written inside the directory specified by ``OUTDIR``. If a
CSV file is locked by Excel or OneDrive, the code automatically saves a new
copy with the suffix ``_new``.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# Separate 2x1 figures for direct interband transition windows
# and JDOS in the Rashba--altermagnetic continuum model
#
# Output figures:
#   1) alpha sweep: top = angular transition window, bottom = JDOS
#   2) tj sweep:    top = angular transition window, bottom = JDOS
#
# Frequency window extended to 50 meV.
# =========================================================

# -----------------------------
# Base model parameters
# -----------------------------
t0 = 1 / (2 * 0.152)      # eV*A^2
mu = 5.5e-3               # eV = 5.5 meV
alpha = 0.026             # eV*A = 26 meV*A
tj = 1.881                # eV*A^2

# Sweep values
alpha_values = np.array([0.007, 0.013, 0.026, 0.052, 0.100])  # eV*A
tj_values = np.array([0.500, 1.000, 1.881, 2.500])     # eV*A^2

# Frequency window
omega_max_meV = 100.0
plot_xmax_meV = 50.0
nomega = 1100
Ew = np.linspace(0.0, omega_max_meV / 1e3, nomega)  # hbar*omega in eV
omega_meV = 1e3 * Ew

# k-space grid
kmax = 0.15               # A^{-1}; increase if Fermi roots are missing
nk = 900
nphi = 721

k_grid = np.linspace(1e-7, kmax, nk)
phi_grid = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
K, PHI = np.meshgrid(k_grid, phi_grid, indexing="xy")

# Output folder: use a new folder to avoid Windows/OneDrive permission conflicts
OUTDIR = "transition_window_JDOS_sweeps_2x1_50meV"
os.makedirs(OUTDIR, exist_ok=True)

# Plot options
FILL_WINDOWS = True       # Set False if the transition-window panels look too crowded
FILL_ALPHA = 0.075
SMOOTH_BINS = 3.0

# JDOS peak-selection parameters
MIN_PEAK_HEIGHT = 0.05
MIN_PEAK_DISTANCE_MEV = 0.60

# =========================================================
# Plot style
# =========================================================
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 17,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
})

# =========================================================
# Model functions
# =========================================================
def d_of(k, phi, alpha_val, tj_val):
    """d(k,phi) for the Rashba--altermagnetic model."""
    return np.sqrt(alpha_val**2 * k**2 + tj_val**2 * k**4 * np.sin(2 * phi)**2)


def eps(k, phi, lam, alpha_val, tj_val):
    """epsilon_lambda(k,phi) = t0*k^2 + lambda*d(k,phi)."""
    return t0 * k**2 + lam * d_of(k, phi, alpha_val, tj_val)

# =========================================================
# Utilities
# =========================================================
def safe_to_csv(df, path, index=False):
    """Write CSV; if file is locked by Excel/OneDrive, save with '_new'."""
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt_path = base + "_new" + ext
        print(f"WARNING: Could not write {path}")
        print(f"Saving instead as {alt_path}")
        df.to_csv(alt_path, index=index)
        return alt_path


def root_from_samples(x, y):
    """Linear-interpolation root finder from sampled data."""
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
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]

    if y2 == y1:
        return 0.5 * (x1 + x2)

    return x1 - y1 * (x2 - x1) / (y2 - y1)


def gaussian_smooth(y, sigma_bins=3.0):
    """Small Gaussian smoothing without scipy."""
    if sigma_bins is None or sigma_bins <= 0:
        return y

    half_width = int(np.ceil(4 * sigma_bins))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins)**2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def select_first_two_jdos_peaks(omega_vals_meV, jdos_norm,
                                min_height=MIN_PEAK_HEIGHT,
                                min_distance_meV=MIN_PEAK_DISTANCE_MEV):
    """Select the first two meaningful local maxima of the normalized JDOS."""
    omega_vals_meV = np.asarray(omega_vals_meV, dtype=float)
    jdos_norm = np.asarray(jdos_norm, dtype=float)

    mask = np.isfinite(omega_vals_meV) & np.isfinite(jdos_norm)
    omega_vals_meV = omega_vals_meV[mask]
    jdos_norm = jdos_norm[mask]

    candidate_indices = []
    for i in range(1, len(jdos_norm) - 1):
        if jdos_norm[i] > jdos_norm[i - 1] and jdos_norm[i] > jdos_norm[i + 1]:
            if jdos_norm[i] >= min_height:
                candidate_indices.append(i)

    selected = []
    for idx in candidate_indices:
        om = omega_vals_meV[idx]
        val = jdos_norm[idx]
        if all(abs(om - om_sel) >= min_distance_meV for om_sel, _ in selected):
            selected.append((om, val))
        if len(selected) == 2:
            break

    return selected

# =========================================================
# Transition boundaries and JDOS
# =========================================================
def fermi_wavevectors_vs_phi(mu_val, alpha_val, tj_val):
    """Compute q_+(phi) and q_-(phi) from eps_+(k,phi)=mu and eps_-(k,phi)=mu."""
    q_plus = np.full_like(phi_grid, np.nan, dtype=float)
    q_minus = np.full_like(phi_grid, np.nan, dtype=float)

    for i, phi in enumerate(phi_grid):
        f_plus = eps(k_grid, phi, +1, alpha_val, tj_val) - mu_val
        f_minus = eps(k_grid, phi, -1, alpha_val, tj_val) - mu_val

        q_plus[i] = root_from_samples(k_grid, f_plus)
        q_minus[i] = root_from_samples(k_grid, f_minus)

    return q_plus, q_minus


def transition_boundaries_vs_phi(mu_val, alpha_val, tj_val):
    """
    Direct interband transition boundaries:
        hbar*Omega_+(phi) = 2 d(q_+(phi), phi)
        hbar*Omega_-(phi) = 2 d(q_-(phi), phi)
    Energies are returned in eV.
    """
    q_plus, q_minus = fermi_wavevectors_vs_phi(mu_val, alpha_val, tj_val)

    Omega_plus = 2.0 * d_of(q_plus, phi_grid, alpha_val, tj_val)
    Omega_minus = 2.0 * d_of(q_minus, phi_grid, alpha_val, tj_val)

    Omega_low = np.minimum(Omega_plus, Omega_minus)
    Omega_high = np.maximum(Omega_plus, Omega_minus)

    return q_plus, q_minus, Omega_plus, Omega_minus, Omega_low, Omega_high


def compute_jdos(mu_val, alpha_val, tj_val, smooth_bins=SMOOTH_BINS):
    """
    JDOS-like quantity for direct interband transitions:
        JDOS(omega) = int d^2k/(2pi)^2 W(k) delta[2d(k)-hbar*omega]
    The delta function is approximated by a weighted histogram of transition energies.
    """
    D = d_of(K, PHI, alpha_val, tj_val)
    eps_plus = t0 * K**2 + D
    eps_minus = t0 * K**2 - D

    W = (eps_minus < mu_val) & (mu_val < eps_plus)
    transition_energy_meV = 1e3 * (2.0 * D)

    dk = k_grid[1] - k_grid[0]
    dphi = phi_grid[1] - phi_grid[0]

    # d^2k/(2pi)^2 = k dk dphi/(2pi)^2
    weights = K * dk * dphi / (2 * np.pi)**2

    bin_edges = np.linspace(0.0, omega_max_meV, nomega + 1)
    hist, edges = np.histogram(
        transition_energy_meV[W],
        bins=bin_edges,
        weights=weights[W]
    )

    omega_centers_meV = 0.5 * (edges[:-1] + edges[1:])

    # Convert histogram to density with respect to energy in eV
    bin_width_eV = (edges[1] - edges[0]) / 1e3
    jdos_raw = hist / bin_width_eV
    jdos_smooth = gaussian_smooth(jdos_raw, sigma_bins=smooth_bins)

    if np.nanmax(jdos_smooth) > 0:
        jdos_norm = jdos_smooth / np.nanmax(jdos_smooth)
    else:
        jdos_norm = jdos_smooth

    return omega_centers_meV, jdos_raw, jdos_smooth, jdos_norm


def compute_case(mu_val, alpha_val, tj_val, label):
    """Compute transition boundaries, JDOS, and characteristic energies for one parameter set."""
    q_plus, q_minus, Omega_plus, Omega_minus, Omega_low, Omega_high = transition_boundaries_vs_phi(
        mu_val, alpha_val, tj_val
    )
    omega_jdos_meV, jdos_raw, jdos_smooth, jdos_norm = compute_jdos(mu_val, alpha_val, tj_val)

    missing_fraction = np.mean(~np.isfinite(Omega_low) | ~np.isfinite(Omega_high))
    omega_low_edge = 1e3 * np.nanmin(Omega_low)
    omega_high_edge = 1e3 * np.nanmax(Omega_high)
    peaks = select_first_two_jdos_peaks(omega_jdos_meV, jdos_norm)

    return {
        "label": label,
        "alpha": alpha_val,
        "tj": tj_val,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "Omega_plus": Omega_plus,
        "Omega_minus": Omega_minus,
        "Omega_low": Omega_low,
        "Omega_high": Omega_high,
        "omega_jdos_meV": omega_jdos_meV,
        "jdos_raw": jdos_raw,
        "jdos_smooth": jdos_smooth,
        "jdos_norm": jdos_norm,
        "missing_fraction": missing_fraction,
        "omega_low_edge": omega_low_edge,
        "omega_high_edge": omega_high_edge,
        "peaks": peaks,
    }

# =========================================================
# Plot helpers
# =========================================================
def setup_phi_axis(ax):
    ax.set_ylim(0, 2)
    ax.set_ylabel(r"$\phi/\pi$")
    ax.set_yticks([0.25, 0.75, 1.25, 1.75])
    ax.set_yticklabels([r"$1/4$", r"$3/4$", r"$5/4$", r"$7/4$"])


def plot_transition_sweep(ax, cases, colors, legend_title, panel_label):
    y = phi_grid / np.pi

    for case, color in zip(cases, colors):
        low = 1e3 * case["Omega_low"]
        high = 1e3 * case["Omega_high"]

        if FILL_WINDOWS:
            finite = np.isfinite(low) & np.isfinite(high)
            ax.fill_betweenx(y[finite], low[finite], high[finite], color=color, alpha=FILL_ALPHA, lw=0)

        ax.plot(low, y, ls="--", lw=1.6, color=color)
        ax.plot(high, y, ls="-", lw=1.9, color=color)

    setup_phi_axis(ax)
    ax.set_xlim(0, plot_xmax_meV)
    ax.text(0.025, 0.88, panel_label, transform=ax.transAxes, fontsize=18)

    handles = [Line2D([0], [0], color=c, lw=2.2, label=case["label"]) for case, c in zip(cases, colors)]
    leg1 = ax.legend(handles=handles, title=legend_title, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7")
    ax.add_artist(leg1)

    boundary_handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.5, label=r"$\Omega_{\rm low}$"),
        Line2D([0], [0], color="0.25", ls="-", lw=1.8, label=r"$\Omega_{\rm high}$"),
    ]
    ax.legend(handles=boundary_handles, loc="lower right", frameon=True, framealpha=1.0, edgecolor="0.7")


def plot_jdos_sweep(ax, cases, colors, panel_label):
    for case, color in zip(cases, colors):
        ax.plot(case["omega_jdos_meV"], case["jdos_norm"], lw=2.3, color=color, label=case["label"])

    ax.axhline(0, color="0.85", lw=0.9, zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, plot_xmax_meV)
    ax.set_ylabel(r"$\mathrm{JDOS}/\max$")
    ax.set_xlabel(r"$\hbar\omega\ \mathrm{(meV)}$")
    ax.text(0.025, 0.84, panel_label, transform=ax.transAxes, fontsize=18)


def apply_axis_style(axes):
    for ax in axes:
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.set_xlim(0, plot_xmax_meV)


def make_two_panel_figure(cases, colors, legend_title, transition_title, figure_title, output_stem):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(8.0, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.0}
    )

    plot_transition_sweep(ax_top, cases, colors, legend_title, r"(a)")
    ax_top.set_title(transition_title, fontsize=15)

    plot_jdos_sweep(ax_bottom, cases, colors, r"(b)")
    ax_bottom.legend(loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7")

    apply_axis_style([ax_top, ax_bottom])

    fig.suptitle(figure_title, fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.09)

    png_path = os.path.join(OUTDIR, f"{output_stem}.png")
    pdf_path = os.path.join(OUTDIR, f"{output_stem}.pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path

# =========================================================
# Save data
# =========================================================
def save_sweep_data(alpha_cases, tj_cases):
    rows_summary = []
    rows_bounds = []
    rows_jdos = []

    for sweep_name, cases in [("alpha_sweep", alpha_cases), ("tj_sweep", tj_cases)]:
        for case in cases:
            peak_dict = {}
            for i, (om, val) in enumerate(case["peaks"], start=1):
                peak_dict[f"omega_p{i}_meV"] = om
                peak_dict[f"jdos_p{i}_over_max"] = val

            rows_summary.append({
                "sweep": sweep_name,
                "label": case["label"],
                "alpha_eVA": case["alpha"],
                "alpha_meVA": 1e3 * case["alpha"],
                "tj_eVA2": case["tj"],
                "mu_meV": 1e3 * mu,
                "missing_fraction": case["missing_fraction"],
                "omega_low_edge_meV": case["omega_low_edge"],
                "omega_high_edge_meV": case["omega_high_edge"],
                **peak_dict,
            })

            for idx in range(len(phi_grid)):
                rows_bounds.append({
                    "sweep": sweep_name,
                    "label": case["label"],
                    "alpha_eVA": case["alpha"],
                    "tj_eVA2": case["tj"],
                    "phi_rad": phi_grid[idx],
                    "phi_over_pi": phi_grid[idx] / np.pi,
                    "Omega_low_meV": 1e3 * case["Omega_low"][idx],
                    "Omega_high_meV": 1e3 * case["Omega_high"][idx],
                    "Omega_plus_meV": 1e3 * case["Omega_plus"][idx],
                    "Omega_minus_meV": 1e3 * case["Omega_minus"][idx],
                    "q_plus_Ainv": case["q_plus"][idx],
                    "q_minus_Ainv": case["q_minus"][idx],
                })

            for idx in range(len(case["omega_jdos_meV"])):
                rows_jdos.append({
                    "sweep": sweep_name,
                    "label": case["label"],
                    "alpha_eVA": case["alpha"],
                    "tj_eVA2": case["tj"],
                    "hbar_omega_meV": case["omega_jdos_meV"][idx],
                    "JDOS_raw": case["jdos_raw"][idx],
                    "JDOS_smooth": case["jdos_smooth"][idx],
                    "JDOS_normalized": case["jdos_norm"][idx],
                })

    summary_csv = os.path.join(OUTDIR, "sweep_characteristic_energies_50meV.csv")
    bounds_csv = os.path.join(OUTDIR, "sweep_transition_boundaries_50meV.csv")
    jdos_csv = os.path.join(OUTDIR, "sweep_jdos_50meV.csv")

    summary_csv = safe_to_csv(pd.DataFrame(rows_summary), summary_csv, index=False)
    bounds_csv = safe_to_csv(pd.DataFrame(rows_bounds), bounds_csv, index=False)
    jdos_csv = safe_to_csv(pd.DataFrame(rows_jdos), jdos_csv, index=False)

    return summary_csv, bounds_csv, jdos_csv

# =========================================================
# Main
# =========================================================
def make_figures():
    print("Computing alpha sweep...")
    alpha_cases = []
    for a in alpha_values:
        label = fr"$\alpha={1e3*a:.0f}\,\mathrm{{meV\AA}}$"
        print(f"  alpha = {1e3*a:.0f} meV A, tj = {tj:.3f} eV A^2")
        alpha_cases.append(compute_case(mu, a, tj, label))

    print("\nComputing tj sweep...")
    tj_cases = []
    for tj_val in tj_values:
        label = fr"$t_j={tj_val:.3f}\,\mathrm{{eV\AA^2}}$"
        print(f"  alpha = {1e3*alpha:.0f} meV A, tj = {tj_val:.3f} eV A^2")
        tj_cases.append(compute_case(mu, alpha, tj_val, label))

    for case in alpha_cases + tj_cases:
        if case["missing_fraction"] > 0.01:
            print(
                f"WARNING: {100*case['missing_fraction']:.1f}% roots missing for {case['label']}. "
                "Try increasing kmax."
            )

    summary_csv, bounds_csv, jdos_csv = save_sweep_data(alpha_cases, tj_cases)

    colors_alpha = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(alpha_cases)]
    colors_tj = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(tj_cases)]

    alpha_png, alpha_pdf = make_two_panel_figure(
        alpha_cases,
        colors_alpha,
        legend_title=r"fixed $t_j=1.881\,\mathrm{eV\AA^2}$",
        transition_title=r"Transition windows: Rashba sweep",
        figure_title=fr"Direct interband transition phase space and JDOS, $\mu={1e3*mu:.1f}\,\mathrm{{meV}}$",
        output_stem="transition_windows_JDOS_alpha_sweep_2x1_50meV",
    )

    tj_png, tj_pdf = make_two_panel_figure(
        tj_cases,
        colors_tj,
        legend_title=r"fixed $\alpha=26\,\mathrm{meV\AA}$",
        transition_title=r"Transition windows: altermagnetic sweep",
        figure_title=fr"Direct interband transition phase space and JDOS, $\mu={1e3*mu:.1f}\,\mathrm{{meV}}$",
        output_stem="transition_windows_JDOS_tj_sweep_2x1_50meV",
    )

    print("\nSaved alpha sweep figure:", alpha_png)
    print("Saved alpha sweep PDF:", alpha_pdf)
    print("Saved tj sweep figure:", tj_png)
    print("Saved tj sweep PDF:", tj_pdf)
    print("Saved summary:", summary_csv)
    print("Saved transition boundaries:", bounds_csv)
    print("Saved JDOS:", jdos_csv)

    return alpha_png, alpha_pdf, tj_png, tj_pdf, summary_csv, bounds_csv, jdos_csv


if __name__ == "__main__":
    make_figures()
