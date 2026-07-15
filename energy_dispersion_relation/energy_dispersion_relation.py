"""
Dispersion cuts of a 2D Rashba–altermagnetic model
===================================================

Overview
--------
This script calculates and plots the energy dispersions of the Hamiltonian

    H(k) = t0 (kx^2 + ky^2) sigma_0
           + 2 tj kx ky sigma_z
           + alpha (kx sigma_y - ky sigma_x).

The two eigenenergies are

    E_lambda(k) = t0 |k|^2 + lambda |d(k)|,

where lambda = +1 or -1 and

    |d(k)| = sqrt[
        alpha^2 (kx^2 + ky^2)
        + 4 tj^2 kx^2 ky^2
    ].

The bands are colored according to the out-of-plane spin expectation value

    <sigma_z>_lambda = lambda (2 tj kx ky) / |d(k)|.

The negative and positive halves of the plotting coordinate correspond to
two different crystallographic momentum directions joined at k = 0.

Available modes
---------------
1. "110_vs_1m10"
   Negative side: k parallel to [110]
   Positive side: k parallel to [1 -1 0]

2. "100_vs_110"
   Negative side: k parallel to [100]
   Positive side: k parallel to [110]

Outputs
-------
The figure is displayed and saved in both PNG and PDF formats inside the
directory specified by OUTPUT_DIR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ============================================================
# USER CONFIGURATION
# ============================================================

# Available options:
#   "110_vs_1m10"
#   "100_vs_110"
MODE = "110_vs_1m10"

# Model parameters in eV and Å units
T0 = 3.2895          # eV Å^2: spin-independent quadratic coefficient
TJ = 1.881           # eV Å^2: d_xy altermagnetic coefficient
ALPHA = 0.052        # eV Å: Rashba SOC strength

# Set TJ = 0.0 to recover the pure Rashba model.
# In that case, <sigma_z> = 0 throughout the momentum path.

MU = 0.030           # eV: chemical potential
K_MAX = 0.10         # Å^{-1}: maximum momentum magnitude
NUM_K_POINTS = 1200

# Figure output
OUTPUT_DIR = Path("dispersion_cuts")
SAVE_DPI = 600


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def d_norm(
    kx: np.ndarray,
    ky: np.ndarray,
    alpha: float = ALPHA,
    tj: float = TJ,
) -> np.ndarray:
    """
    Calculate the magnitude of the spin-dependent d-vector.

    Parameters
    ----------
    kx, ky
        Cartesian momentum components in Å^{-1}.
    alpha
        Rashba spin-orbit coupling in eV Å.
    tj
        Altermagnetic coupling in eV Å^2.

    Returns
    -------
    numpy.ndarray
        Magnitude |d(k)| in eV.
    """
    rashba_term = alpha**2 * (kx**2 + ky**2)
    altermagnetic_term = 4.0 * tj**2 * kx**2 * ky**2

    return np.sqrt(rashba_term + altermagnetic_term)


def energy_band(
    kx: np.ndarray,
    ky: np.ndarray,
    band_index: int,
    t0: float = T0,
    alpha: float = ALPHA,
    tj: float = TJ,
) -> np.ndarray:
    """
    Calculate the energy of one band.

    Parameters
    ----------
    kx, ky
        Cartesian momentum components in Å^{-1}.
    band_index
        Band index lambda. It must be either +1 or -1.
    t0
        Spin-independent quadratic coefficient in eV Å^2.
    alpha
        Rashba spin-orbit coupling in eV Å.
    tj
        Altermagnetic coupling in eV Å^2.

    Returns
    -------
    numpy.ndarray
        Band energy in eV.
    """
    if band_index not in (-1, +1):
        raise ValueError("band_index must be either +1 or -1.")

    kinetic_energy = t0 * (kx**2 + ky**2)
    splitting = d_norm(kx, ky, alpha=alpha, tj=tj)

    return kinetic_energy + band_index * splitting


def sigma_z_expectation(
    kx: np.ndarray,
    ky: np.ndarray,
    band_index: int,
    alpha: float = ALPHA,
    tj: float = TJ,
) -> np.ndarray:
    """
    Calculate the out-of-plane spin expectation value <sigma_z>.

    At a band degeneracy, where |d(k)| = 0, the spin direction is not
    uniquely defined. The returned value is set to zero at those points.

    Parameters
    ----------
    kx, ky
        Cartesian momentum components in Å^{-1}.
    band_index
        Band index lambda. It must be either +1 or -1.
    alpha
        Rashba spin-orbit coupling in eV Å.
    tj
        Altermagnetic coupling in eV Å^2.

    Returns
    -------
    numpy.ndarray
        Dimensionless spin expectation value between -1 and +1.
    """
    if band_index not in (-1, +1):
        raise ValueError("band_index must be either +1 or -1.")

    d = d_norm(kx, ky, alpha=alpha, tj=tj)
    dz = 2.0 * tj * kx * ky

    sigma_z = np.zeros_like(d)

    nondegenerate = d > 1.0e-14
    sigma_z[nondegenerate] = (
        band_index * dz[nondegenerate] / d[nondegenerate]
    )

    return sigma_z


# ============================================================
# MOMENTUM-PATH CONSTRUCTION
# ============================================================

def build_momentum_path(
    plotting_coordinate: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Construct the piecewise crystallographic momentum path.

    The sign of the plotting coordinate distinguishes the two directions,
    while its absolute value gives the physical momentum magnitude.

    Parameters
    ----------
    plotting_coordinate
        Signed coordinate used on the horizontal plot axis.
    mode
        Momentum-path mode.

    Returns
    -------
    kx_path, ky_path
        Cartesian components of the physical momentum path.
    left_label, right_label
        Labels for the two crystallographic directions.
    filename_base
        Base name used when saving the figure.
    """
    momentum_magnitude = np.abs(plotting_coordinate)
    left_side = plotting_coordinate < 0.0

    if mode == "110_vs_1m10":
        # Left:  [110]
        # Right: [1 -1 0]
        kx_path = momentum_magnitude / np.sqrt(2.0)
        ky_path = np.where(
            left_side,
            momentum_magnitude / np.sqrt(2.0),
            -momentum_magnitude / np.sqrt(2.0),
        )

        left_label = r"$\mathbf{k}\parallel[110]$"
        right_label = r"$\mathbf{k}\parallel[1\bar{1}0]$"
        filename_base = "dispersion_cut_110_vs_1m10_sz"

    elif mode == "100_vs_110":
        # Left:  [100]
        # Right: [110]
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

        left_label = r"$\mathbf{k}\parallel[100]$"
        right_label = r"$\mathbf{k}\parallel[110]$"
        filename_base = "dispersion_cut_100_vs_110_sz"

    else:
        valid_modes = ("110_vs_1m10", "100_vs_110")
        raise ValueError(
            f"Unknown mode {mode!r}. Choose one of {valid_modes}."
        )

    return (
        kx_path,
        ky_path,
        left_label,
        right_label,
        filename_base,
    )


# ============================================================
# PLOTTING UTILITIES
# ============================================================

def add_colored_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray,
    cmap,
    norm: Normalize,
    linewidth: float = 2.4,
    linestyle: str = "solid",
) -> LineCollection:
    """
    Add a line whose color varies continuously along its length.

    The color assigned to each line segment is determined from the average
    value of `color_values` at the two endpoints of that segment.
    """
    if not (len(x) == len(y) == len(color_values)):
        raise ValueError(
            "x, y, and color_values must have the same length."
        )

    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate(
        (points[:-1], points[1:]),
        axis=1,
    )

    segment_colors = 0.5 * (
        color_values[:-1] + color_values[1:]
    )

    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        linestyles=linestyle,
    )

    line_collection.set_array(segment_colors)
    ax.add_collection(line_collection)

    return line_collection


def configure_plot_style() -> None:
    """Set the global typography used by the figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
        }
    )


def create_dispersion_figure(
    plotting_coordinate: np.ndarray,
    energy_plus: np.ndarray,
    energy_minus: np.ndarray,
    spin_plus: np.ndarray,
    spin_minus: np.ndarray,
    left_label: str,
    right_label: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create the energy-dispersion figure colored by <sigma_z>.
    """
    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(6.4, 4.4),
        dpi=300,
    )

    ax.set_facecolor("#f7f7f7")

    spin_cmap = plt.colormaps["RdBu_r"]
    spin_norm = Normalize(vmin=-1.0, vmax=1.0)

    upper_line = add_colored_line(
        ax=ax,
        x=plotting_coordinate,
        y=energy_plus,
        color_values=spin_plus,
        cmap=spin_cmap,
        norm=spin_norm,
        linewidth=2.6,
        linestyle="solid",
    )

    add_colored_line(
        ax=ax,
        x=plotting_coordinate,
        y=energy_minus,
        color_values=spin_minus,
        cmap=spin_cmap,
        norm=spin_norm,
        linewidth=2.6,
        linestyle="dashed",
    )

    # Reference lines
    ax.axhline(
        MU,
        color="black",
        linestyle="--",
        linewidth=1.6,
    )

    ax.axvline(
        0.0,
        color="gray",
        linewidth=1.0,
        alpha=0.5,
    )

    # Axis labels and limits
    ax.set_xlabel(
        r"$k\;(\mathrm{\AA}^{-1})$",
        fontsize=15,
    )

    ax.set_ylabel(
        r"$\varepsilon_{\pm}(\mathbf{k})\;(\mathrm{eV})$",
        fontsize=15,
    )

    ax.set_xlim(-K_MAX, K_MAX)

    lower_limit = min(
        energy_minus.min(),
        energy_plus.min(),
        0.0,
    ) - 0.003

    upper_limit = max(
        energy_minus.max(),
        energy_plus.max(),
        MU,
    ) + 0.005

    ax.set_ylim(lower_limit, upper_limit)

    # Momentum ticks
    momentum_ticks = np.linspace(-K_MAX, K_MAX, 5)
    ax.set_xticks(momentum_ticks)
    ax.set_xticklabels(
        [rf"${value:.2f}$" for value in momentum_ticks]
    )

    # Tick and spine formatting
    ax.tick_params(
        direction="in",
        length=5,
        width=1.2,
        labelsize=12,
    )

    ax.minorticks_on()

    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Dummy lines provide uniform black entries in the legend.
    ax.plot(
        [],
        [],
        color="black",
        linewidth=2.4,
        linestyle="solid",
        label=r"$E_{+}$",
    )

    ax.plot(
        [],
        [],
        color="black",
        linewidth=2.4,
        linestyle="dashed",
        label=r"$E_{-}$",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
    )

    # Crystallographic-direction labels
    ax.text(
        -0.5 * K_MAX,
        -0.20,
        left_label,
        transform=ax.get_xaxis_transform(),
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        clip_on=False,
    )

    ax.text(
        +0.5 * K_MAX,
        -0.20,
        right_label,
        transform=ax.get_xaxis_transform(),
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        clip_on=False,
    )

    # Chemical-potential annotation
    ax.text(
        -0.95 * K_MAX,
        MU + 0.002,
        rf"$\mu={1000.0 * MU:.0f}\,\mathrm{{meV}}$",
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # Spin-polarization colorbar
    colorbar = fig.colorbar(
        upper_line,
        ax=ax,
        pad=0.025,
    )

    colorbar.set_label(
        r"$\langle\sigma_z\rangle_\lambda$",
        fontsize=14,
    )

    colorbar.set_ticks(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# OUTPUT
# ============================================================

def save_figure(
    fig: plt.Figure,
    filename_base: str,
) -> None:
    """Save the figure in high-resolution PNG and vector PDF formats."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = OUTPUT_DIR / f"{filename_base}.png"
    pdf_path = OUTPUT_DIR / f"{filename_base}.pdf"

    fig.savefig(
        png_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(f"PNG figure saved to: {png_path.resolve()}")
    print(f"PDF figure saved to: {pdf_path.resolve()}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """Calculate the selected dispersion cut, plot it, and save it."""
    plotting_coordinate = np.linspace(
        -K_MAX,
        K_MAX,
        NUM_K_POINTS,
    )

    (
        kx_path,
        ky_path,
        left_label,
        right_label,
        filename_base,
    ) = build_momentum_path(
        plotting_coordinate,
        MODE,
    )

    energy_plus = energy_band(
        kx_path,
        ky_path,
        band_index=+1,
    )

    energy_minus = energy_band(
        kx_path,
        ky_path,
        band_index=-1,
    )

    spin_plus = sigma_z_expectation(
        kx_path,
        ky_path,
        band_index=+1,
    )

    spin_minus = sigma_z_expectation(
        kx_path,
        ky_path,
        band_index=-1,
    )

    fig, _ = create_dispersion_figure(
        plotting_coordinate=plotting_coordinate,
        energy_plus=energy_plus,
        energy_minus=energy_minus,
        spin_plus=spin_plus,
        spin_minus=spin_minus,
        left_label=left_label,
        right_label=right_label,
    )

    save_figure(
        fig,
        filename_base,
    )

    plt.show()


if __name__ == "__main__":
    main()