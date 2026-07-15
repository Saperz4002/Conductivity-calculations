"""
Spin-biased Fermi contours and energy-dispersion cuts
=====================================================

Overview
--------
This script combines two visual analyses of a two-dimensional
Rashba--altermagnetic continuum model in the presence of an out-of-plane
spin bias b_z.

The Hamiltonian is represented through the spin-dependent vector

    d(k, b_z) =
        (
            -alpha ky,
             alpha kx,
             2 tj kx ky - b_z
        ),

whose magnitude is

    |d(k, b_z)| =
        sqrt[
            alpha^2 (kx^2 + ky^2)
            + (2 tj kx ky - b_z)^2
        ].

The two energy bands are

    E_lambda(k, b_z) =
        t0 (kx^2 + ky^2)
        + lambda |d(k, b_z)|,

where lambda = +1 and -1 label the upper and lower bands.

Available figure modes
----------------------
Set ``FIGURE_MODE`` to one of the following:

1. ``"fermi_contours"``
   Generate a five-panel figure showing the constant-energy contours

       E_+(k, b_z) = mu,
       E_-(k, b_z) = mu

   for the spin-bias values listed in ``FERMI_BZ_VALUES_MEV``.

   When ``SHOW_EQUILIBRIUM_REFERENCE = True``, the corresponding b_z = 0
   contours are added as gray dashed reference curves.

2. ``"dispersion_cut"``
   Generate a one-dimensional energy-dispersion cut at a fixed spin bias.

   The momentum path is controlled independently by
   ``DISPERSION_PATH_MODE``.

3. ``"both"``
   Generate both the Fermi-contour panel and the biased dispersion cut.

Available dispersion paths
--------------------------
Set ``DISPERSION_PATH_MODE`` to one of the following:

1. ``"110_vs_1m10"``
   Negative side of the horizontal plotting coordinate:
       k parallel to [110]

   Positive side:
       k parallel to [1 -1 0]

2. ``"100_vs_110"``
   Negative side:
       k parallel to [100]

   Positive side:
       k parallel to [110]

Independent configurations
--------------------------
The two original programs used different physical and numerical settings.
To avoid silently changing their results, this merged script retains
separate configuration sections:

- ``FERMI_*`` settings control the Fermi-contour panel;
- ``DISPERSION_*`` settings control the one-dimensional band cut.

In particular, the Rashba coupling and spin-bias values may be different in
the two analyses.

Outputs
-------
The Fermi-contour figure is saved in PNG format inside

    OUTPUT_ROOT / "fermi_contours".

The dispersion-cut figure is saved in both PNG and PDF formats inside

    OUTPUT_ROOT / "dispersion_cuts".

All calculations are organized inside functions and executed through
``main()``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# USER CONFIGURATION
# ============================================================

# Figure options:
#   "fermi_contours"
#   "dispersion_cut"
#   "both"
FIGURE_MODE = "both"

# Momentum-path options for the dispersion cut:
#   "110_vs_1m10"
#   "100_vs_110"
DISPERSION_PATH_MODE = "100_vs_110"

SAVE_FIGURES = True
OUTPUT_ROOT = Path("spin_bias_band_figures")


# ============================================================
# SHARED MODEL PARAMETER
# ============================================================

T0 = 3.2895  # eV Å^2


# ============================================================
# FERMI-CONTOUR CONFIGURATION
# ============================================================

FERMI_TJ = 1.881                 # eV Å^2
FERMI_ALPHA_MEVA = 26.0          # meV Å
FERMI_ALPHA = FERMI_ALPHA_MEVA * 1.0e-3  # eV Å

FERMI_MU = 0.030                 # eV

FERMI_BZ_VALUES_MEV = [
    0.0,
    1.0,
    3.0,
    5.0,
    10.0,
]

FERMI_BZ_VALUES = [
    value * 1.0e-3
    for value in FERMI_BZ_VALUES_MEV
]

FERMI_KMAX = 0.25                # Å^{-1}
FERMI_NGRID_X = 800
FERMI_NGRID_Y = 800

FERMI_XLIM = (
    -0.25,
    0.25,
)

FERMI_YLIM = (
    -0.25,
    0.25,
)

SHOW_EQUILIBRIUM_REFERENCE = True
FERMI_FIGURE_DPI = 600


# ============================================================
# DISPERSION-CUT CONFIGURATION
# ============================================================

DISPERSION_TJ = 1.881            # eV Å^2
DISPERSION_ALPHA = 0.052         # eV Å = 52 meV Å

DISPERSION_MU = 0.030            # eV
DISPERSION_KMAX = 0.10           # Å^{-1}

DISPERSION_BZ_MEV = 10.0
DISPERSION_BZ = DISPERSION_BZ_MEV * 1.0e-3  # eV

DISPERSION_NUM_POINTS = 1200
DISPERSION_FIGURE_DPI = 600


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FERMI_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "fermi_contours"
)

DISPERSION_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "dispersion_cuts"
)

if SAVE_FIGURES:
    FERMI_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISPERSION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# SHARED MODEL FUNCTIONS
# ============================================================

def d_components(
    kx,
    ky,
    alpha,
    tj,
    bz,
):
    """
    Return the three components of d(k, b_z).

    d(k, b_z) =
        (-alpha ky, alpha kx, 2 tj kx ky - b_z).
    """
    dx = -alpha * ky
    dy = alpha * kx
    dz = 2.0 * tj * kx * ky - bz

    return dx, dy, dz


def d_norm_biased(
    kx,
    ky,
    alpha,
    tj,
    bz,
):
    """Return |d(k, b_z)|."""
    dx, dy, dz = d_components(
        kx,
        ky,
        alpha,
        tj,
        bz,
    )

    return np.sqrt(
        dx**2
        + dy**2
        + dz**2
    )


def biased_bands(
    kx,
    ky,
    alpha,
    tj,
    t0,
    bz,
):
    """
    Return the upper and lower biased bands.

    E_lambda(k, b_z) =
        t0 k^2 + lambda |d(k, b_z)|.
    """
    k_squared = (
        kx**2
        + ky**2
    )

    splitting = d_norm_biased(
        kx,
        ky,
        alpha,
        tj,
        bz,
    )

    energy_plus = (
        t0 * k_squared
        + splitting
    )

    energy_minus = (
        t0 * k_squared
        - splitting
    )

    return (
        energy_plus,
        energy_minus,
    )


# ============================================================
# FERMI-CONTOUR UTILITIES
# ============================================================

def build_fermi_grid():
    """Construct the two-dimensional grid used by the Fermi-contour panel."""
    kx = np.linspace(
        -FERMI_KMAX,
        FERMI_KMAX,
        FERMI_NGRID_X,
    )

    ky = np.linspace(
        -FERMI_KMAX,
        FERMI_KMAX,
        FERMI_NGRID_Y,
    )

    return np.meshgrid(
        kx,
        ky,
    )


def safe_contour(
    axis,
    x_grid,
    y_grid,
    values,
    level,
    **kwargs,
):
    """
    Draw a contour only when the requested level is inside the data range.

    This prevents Matplotlib warnings when no Fermi contour exists.
    """
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if minimum <= level <= maximum:
        return axis.contour(
            x_grid,
            y_grid,
            values,
            levels=[level],
            **kwargs,
        )

    return None


def configure_fermi_plot_style():
    """Apply the compact plot style used by the Fermi-contour panel."""
    plt.rcdefaults()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "mathtext.fontset": "stix",
        }
    )


def plot_fermi_contours_on_axis(
    axis,
    kx_grid,
    ky_grid,
    chemical_potential,
    spin_bias,
):
    """
    Draw one Fermi-contour panel for a selected spin bias.

    The current upper and lower contours are colored red and blue.
    Equilibrium contours may be shown as gray dashed references.
    """
    energy_plus, energy_minus = biased_bands(
        kx_grid,
        ky_grid,
        FERMI_ALPHA,
        FERMI_TJ,
        T0,
        spin_bias,
    )

    energy_plus_equilibrium, energy_minus_equilibrium = biased_bands(
        kx_grid,
        ky_grid,
        FERMI_ALPHA,
        FERMI_TJ,
        T0,
        bz=0.0,
    )

    if SHOW_EQUILIBRIUM_REFERENCE:
        safe_contour(
            axis,
            kx_grid,
            ky_grid,
            energy_plus_equilibrium,
            chemical_potential,
            colors="0.60",
            linestyles="--",
            linewidths=1.0,
            zorder=1,
        )

        safe_contour(
            axis,
            kx_grid,
            ky_grid,
            energy_minus_equilibrium,
            chemical_potential,
            colors="0.60",
            linestyles="--",
            linewidths=1.0,
            zorder=1,
        )

    safe_contour(
        axis,
        kx_grid,
        ky_grid,
        energy_plus,
        chemical_potential,
        colors="red",
        linewidths=1.8,
        zorder=5,
    )

    safe_contour(
        axis,
        kx_grid,
        ky_grid,
        energy_minus,
        chemical_potential,
        colors="blue",
        linewidths=1.8,
        zorder=5,
    )

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    axis.axvline(
        0.0,
        color="black",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    axis.grid(
        True,
        color="0.82",
        linewidth=0.8,
        alpha=0.85,
    )

    axis.set_xlim(
        *FERMI_XLIM
    )

    axis.set_ylim(
        *FERMI_YLIM
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel(
        r"$k_x\;(\mathrm{\AA^{-1}})$"
    )

    axis.set_ylabel(
        r"$k_y\;(\mathrm{\AA^{-1}})$"
    )

    axis.tick_params(
        direction="in",
        length=3.5,
        width=0.9,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(
            1.0
        )

    axis.text(
        0.035,
        0.965,
        (
            rf"$\mu={chemical_potential:.3f}\,\mathrm{{eV}}$"
            + "\n"
            + rf"$b_z={spin_bias * 1.0e3:.1f}\,\mathrm{{meV}}$"
            + "\n"
            + rf"$\alpha={FERMI_ALPHA_MEVA:.0f}\,\mathrm{{meV\AA}}$"
            + "\n"
            + rf"$t_j={FERMI_TJ:.3f}\,\mathrm{{eV\AA^2}}$"
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.25",
            "alpha": 0.88,
        },
    )

    legend_handles = [
        plt.Line2D(
            [],
            [],
            color="red",
            linewidth=1.8,
            label=r"$E_+(\mathbf{k},b_z)=\mu$",
        ),
        plt.Line2D(
            [],
            [],
            color="blue",
            linewidth=1.8,
            label=r"$E_-(\mathbf{k},b_z)=\mu$",
        ),
    ]

    if SHOW_EQUILIBRIUM_REFERENCE:
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color="0.60",
                linewidth=1.0,
                linestyle="--",
                label=r"$b_z=0$ reference",
            )
        )

    axis.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fontsize=6.2,
    )


def plot_fermi_contour_panel():
    """
    Create the five-panel Fermi-contour figure.

    The top row contains the first three spin-bias values. The lower row
    contains an equation panel followed by the final two values.
    """
    configure_fermi_plot_style()

    kx_grid, ky_grid = build_fermi_grid()

    figure = plt.figure(
        figsize=(11.5, 6.4),
        dpi=180,
    )

    grid_specification = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        bottom=0.090,
        top=0.970,
        wspace=0.26,
        hspace=0.32,
    )

    panel_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]

    for spin_bias, position in zip(
        FERMI_BZ_VALUES,
        panel_positions,
    ):
        axis = figure.add_subplot(
            grid_specification[position]
        )

        plot_fermi_contours_on_axis(
            axis,
            kx_grid,
            ky_grid,
            FERMI_MU,
            spin_bias,
        )

    formula_axis = figure.add_subplot(
        grid_specification[1, 0]
    )

    formula_axis.axis(
        "off"
    )

    formula_axis.text(
        0.00,
        0.62,
        (
            r"$\varepsilon_{\lambda}(\mathbf{k},b_z)"
            r"=t_0k^2+\lambda d_b(\mathbf{k}),$"
            + "\n"
            + r"$\lambda=\pm.$"
        ),
        fontsize=18,
        horizontalalignment="left",
        verticalalignment="center",
    )

    output_path = (
        FERMI_OUTPUT_DIR
        / "fermi_contours_bz_panel_mu_0p030eV.png"
    )

    if SAVE_FIGURES:
        figure.savefig(
            output_path,
            dpi=FERMI_FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            f"Fermi-contour figure saved to: "
            f"{output_path.resolve()}"
        )

    plt.show()


# ============================================================
# DISPERSION-CUT UTILITIES
# ============================================================

def configure_dispersion_plot_style():
    """Apply the typography used by the one-dimensional dispersion cut."""
    plt.rcdefaults()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
        }
    )


def build_dispersion_path(
    plotting_coordinate,
    path_mode,
):
    """
    Build the piecewise crystallographic momentum path.

    The absolute value of the plotting coordinate is the physical momentum
    magnitude. Its sign selects the left or right crystallographic segment.
    """
    momentum_magnitude = np.abs(
        plotting_coordinate
    )

    left_side = (
        plotting_coordinate < 0.0
    )

    if path_mode == "110_vs_1m10":
        kx_path = (
            momentum_magnitude
            / np.sqrt(2.0)
        )

        ky_path = np.where(
            left_side,
            momentum_magnitude / np.sqrt(2.0),
            -momentum_magnitude / np.sqrt(2.0),
        )

        left_label = (
            r"$\mathbf{k}\parallel [110]$"
        )

        right_label = (
            r"$\mathbf{k}\parallel [1\bar{1}0]$"
        )

        filename_base = (
            "dispersion_cut_bias_110_vs_1m10"
            f"_bz_{DISPERSION_BZ_MEV:.1f}meV"
        )

    elif path_mode == "100_vs_110":
        kx_path = np.where(
            left_side,
            momentum_magnitude,
            momentum_magnitude / np.sqrt(2.0),
        )

        ky_path = np.where(
            left_side,
            0.0,
            momentum_magnitude / np.sqrt(2.0),
        )

        left_label = (
            r"$\mathbf{k}\parallel [100]$"
        )

        right_label = (
            r"$\mathbf{k}\parallel [110]$"
        )

        filename_base = (
            "dispersion_cut_bias_100_vs_110"
            f"_bz_{DISPERSION_BZ_MEV:.1f}meV"
        )

    else:
        raise ValueError(
            "DISPERSION_PATH_MODE must be "
            "'110_vs_1m10' or '100_vs_110'."
        )

    return (
        kx_path,
        ky_path,
        left_label,
        right_label,
        filename_base,
    )


def plot_biased_dispersion_cut():
    """Calculate, plot, and save the selected biased dispersion cut."""
    configure_dispersion_plot_style()

    plotting_coordinate = np.linspace(
        -DISPERSION_KMAX,
        DISPERSION_KMAX,
        DISPERSION_NUM_POINTS,
    )

    (
        kx_path,
        ky_path,
        left_label,
        right_label,
        filename_base,
    ) = build_dispersion_path(
        plotting_coordinate,
        DISPERSION_PATH_MODE,
    )

    energy_plus, energy_minus = biased_bands(
        kx_path,
        ky_path,
        DISPERSION_ALPHA,
        DISPERSION_TJ,
        T0,
        DISPERSION_BZ,
    )

    figure, axis = plt.subplots(
        figsize=(6.4, 4.4),
        dpi=300,
    )

    axis.set_facecolor(
        "white"
    )

    axis.plot(
        plotting_coordinate,
        energy_plus,
        color="red",
        linewidth=2.2,
        label=r"$E_{+}$",
    )

    axis.plot(
        plotting_coordinate,
        energy_minus,
        color="blue",
        linewidth=2.2,
        label=r"$E_{-}$",
    )

    axis.axhline(
        DISPERSION_MU,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    axis.axvline(
        0.0,
        color="gray",
        linewidth=1.0,
        alpha=0.55,
    )

    axis.set_xlabel(
        r"$k\;(\mathrm{\AA}^{-1})$",
        fontsize=15,
    )

    axis.set_ylabel(
        r"$\varepsilon_{\pm}(\mathbf{k},b_z)"
        r"\;(\mathrm{eV})$",
        fontsize=15,
    )

    axis.set_xlim(
        -DISPERSION_KMAX,
        DISPERSION_KMAX,
    )

    axis.set_ylim(
        min(
            np.min(energy_minus),
            0.0,
        ) - 0.004,
        max(
            np.max(energy_plus),
            DISPERSION_MU,
        ) + 0.006,
    )

    x_ticks = np.linspace(
        -DISPERSION_KMAX,
        DISPERSION_KMAX,
        5,
    )

    axis.set_xticks(
        x_ticks
    )

    axis.set_xticklabels(
        [
            rf"${value:.2f}$"
            for value in x_ticks
        ]
    )

    axis.tick_params(
        direction="in",
        length=5,
        width=1.2,
        labelsize=12,
    )

    axis.minorticks_on()

    axis.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(
            1.2
        )

    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
    )

    axis.text(
        0.97,
        0.78,
        rf"$b_z={DISPERSION_BZ_MEV:.0f}\,\mathrm{{meV}}$",
        transform=axis.transAxes,
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
        },
    )

    axis.text(
        -0.5 * DISPERSION_KMAX,
        -0.20,
        left_label,
        transform=axis.get_xaxis_transform(),
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        clip_on=False,
    )

    axis.text(
        +0.5 * DISPERSION_KMAX,
        -0.20,
        right_label,
        transform=axis.get_xaxis_transform(),
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        clip_on=False,
    )

    axis.text(
        -0.95 * DISPERSION_KMAX,
        DISPERSION_MU + 0.002,
        rf"$\mu={1.0e3 * DISPERSION_MU:.0f}\,\mathrm{{meV}}$",
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    figure.tight_layout()

    png_path = (
        DISPERSION_OUTPUT_DIR
        / f"{filename_base}.png"
    )

    pdf_path = (
        DISPERSION_OUTPUT_DIR
        / f"{filename_base}.pdf"
    )

    if SAVE_FIGURES:
        figure.savefig(
            png_path,
            dpi=DISPERSION_FIGURE_DPI,
            bbox_inches="tight",
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        print(
            f"Dispersion PNG saved to: "
            f"{png_path.resolve()}"
        )

        print(
            f"Dispersion PDF saved to: "
            f"{pdf_path.resolve()}"
        )

    plt.show()


# ============================================================
# MODE VALIDATION
# ============================================================

def selected_figures():
    """Validate and return the requested figure modes."""
    valid_modes = {
        "fermi_contours",
        "dispersion_cut",
        "both",
    }

    if FIGURE_MODE not in valid_modes:
        raise ValueError(
            "FIGURE_MODE must be "
            "'fermi_contours', 'dispersion_cut', or 'both'."
        )

    if FIGURE_MODE == "both":
        return [
            "fermi_contours",
            "dispersion_cut",
        ]

    return [
        FIGURE_MODE
    ]


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    """Generate the selected spin-biased band-structure figures."""
    for figure_mode in selected_figures():
        if figure_mode == "fermi_contours":
            plot_fermi_contour_panel()

        elif figure_mode == "dispersion_cut":
            plot_biased_dispersion_cut()

    print(
        "\nAll selected figures are complete."
    )

    print(
        f"Output root: {OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()
