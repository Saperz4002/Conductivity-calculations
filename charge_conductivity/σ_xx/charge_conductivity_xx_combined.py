"""
Intraband and interband longitudinal charge conductivity
========================================================

Overview
--------
This script calculates the normalized longitudinal charge conductivity
sigma_xx of a two-dimensional Rashba--altermagnetic continuum model.

The band energies are

    epsilon_lambda(k, phi) = t0 k^2 + lambda d(k, phi),

with

    d(k, phi) =
        sqrt[
            alpha^2 k^2
            + tj^2 k^4 sin^2(2 phi)
        ],

where lambda = +1 and -1 label the upper and lower bands.

The normalized conductivity plotted by the script is

    sigma_tilde_xx = (hbar / e^2) sigma_xx.

Two physically distinct contributions are available:

1. Intraband contribution
   The Fermi-surface or Drude-like response is evaluated through angular
   integrals over the Fermi contours. For every polar angle, the code finds
   the allowed radial momentum roots and sums the corresponding band
   velocities.

2. Interband contribution
   The finite-frequency response caused by transitions between the two bands
   is evaluated on a two-dimensional polar momentum grid. At zero
   temperature, the code applies the Pauli window

       epsilon_-(k) < mu < epsilon_+(k)

   before carrying out the momentum integral.

Contribution modes
------------------
Set ``CONTRIBUTION_MODE`` to one of the following:

- ``"intra"``
  Calculate only the intraband contribution.

- ``"inter"``
  Calculate only the interband contribution.

- ``"both"``
  Calculate the intraband and interband contributions sequentially.

Sweep modes
-----------
Set ``RUN_MODE`` to one of the following:

- ``"mu"``
  Sweep the chemical potential.

- ``"alpha"``
  Sweep the Rashba spin-orbit coupling.

- ``"tj"``
  Sweep the d-wave altermagnetic parameter.

- ``"all"``
  Run all three parameter sweeps.

Numerical settings
------------------
The original intraband and interband scripts used different parameter
arrays, frequency windows, and numerical grids. Those settings are retained
in separate configuration sections below so that merging the programs does
not silently change their original calculations.

Outputs
-------
The script creates separate ``intraband`` and ``interband`` folders inside
``OUTPUT_ROOT``.

For each selected sweep, it saves:

- CSV files containing the computed conductivity data;
- a PNG figure showing the real and imaginary parts of the selected
  contribution to sigma_xx.

The intraband calculation also saves a CSV file containing the individual
upper-band, lower-band, and total Fermi-contour integrals.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning, quad


# ============================================================
# USER CONFIGURATION
# ============================================================

# Contribution options:
#   "intra" -> intraband conductivity only
#   "inter" -> interband conductivity only
#   "both"  -> run both contributions
CONTRIBUTION_MODE = "both"

# Sweep options:
#   "mu", "alpha", "tj", or "all"
RUN_MODE = "all"

# Main output folder
OUTPUT_ROOT = Path("sigma_xx_charge_sweeps")
INTRA_OUTPUT_DIR = OUTPUT_ROOT / "intraband"
INTER_OUTPUT_DIR = OUTPUT_ROOT / "interband"

INTRA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SHARED PHYSICAL PARAMETERS
# ============================================================

HBAR_EVS = 6.582119569e-16  # eV s

# m = 0.152 eV^{-1} Å^{-2}, so t0 = 1/(2m)
T0 = 1.0 / (2.0 * 0.152)    # eV Å^2

BASE_TJ = 1.881              # eV Å^2
BASE_ALPHA = 0.026           # eV Å = 26 meV Å


# ============================================================
# INTRABAND CONFIGURATION
# ============================================================

# Angular resolution. Increase to 8000 for final production plots.
INTRA_NPHI = 4000

INTRA_FIXED_MU = 1.0e-3      # eV = 1 meV

INTRA_ALPHA_SWEEP_FIXED_TJ = BASE_TJ
INTRA_TJ_SWEEP_FIXED_ALPHA = 0.052  # eV Å = 52 meV Å

INTRA_MU_SWEEP = np.array([
    1.0e-4,
    1.0e-3,
    1.9e-3,
    2.8e-3,
    5.5e-3,
])

INTRA_ALPHA_SWEEP = np.array([
    0.013,
    0.026,
    0.052,
    0.100,
])

INTRA_TJ_SWEEP = np.array([
    0.25,
    1.00,
    1.881,
    2.40,
])

INTRA_ETA = 1.0e-3
INTRA_EW = np.linspace(0.0, 0.020, 500)


# ============================================================
# INTERBAND CONFIGURATION
# ============================================================

INTER_MU_REF = 5.5e-3
INTER_ALPHA_REF = BASE_ALPHA
INTER_TJ_REF = BASE_TJ

INTER_MU_SWEEP = np.array([
    1.0e-4,
    1.0e-3,
    1.9e-3,
    2.8e-3,
    5.5e-3,
])

INTER_ALPHA_SWEEP = np.array([
    0.007,
    0.013,
    0.026,
    0.052,
    0.100,
])

INTER_TJ_SWEEP = np.array([
    0.500,
    1.000,
    1.881,
    2.500,
    3.000,
])

INTER_ETA = 1.0e-3
INTER_EW = np.linspace(0.0, 0.050, 500)

INTER_KMAX = 0.15
INTER_NK = 900
INTER_NPHI = 501


# ============================================================
# SHARED MODEL FUNCTIONS
# ============================================================

def d_of(
    k: np.ndarray | float,
    phi: np.ndarray | float,
    alpha: float,
    tj: float,
) -> np.ndarray | float:
    """
    Return the magnitude of the spin-dependent d-vector.

    d(k, phi) =
        sqrt[
            alpha^2 k^2
            + tj^2 k^4 sin^2(2 phi)
        ].
    """
    sin_2phi = np.sin(2.0 * phi)

    return np.sqrt(
        alpha**2 * k**2
        + tj**2 * k**4 * sin_2phi**2
    )


def energy(
    k: np.ndarray | float,
    phi: np.ndarray | float,
    alpha: float,
    tj: float,
    band: int,
) -> np.ndarray | float:
    """
    Return the energy epsilon_band(k, phi).

    ``band`` must be either +1 or -1.
    """
    if band not in (-1, +1):
        raise ValueError("band must be either +1 or -1.")

    return T0 * k**2 + band * d_of(k, phi, alpha, tj)


# ============================================================
# INTRABAND CONDUCTIVITY
# ============================================================

def intraband_roots_k(
    target_energy: float,
    phi: float,
    alpha: float,
    tj: float,
    band: int,
) -> list[float]:
    """
    Solve target_energy = epsilon_band(k, phi) analytically in k^2.

    Only positive roots that reproduce the selected band energy are kept.
    """
    sin_2phi = np.sin(2.0 * phi)

    quadratic_a = T0**2 - tj**2 * sin_2phi**2
    quadratic_b = -(2.0 * T0 * target_energy + alpha**2)
    quadratic_c = target_energy**2

    if abs(quadratic_a) < 1.0e-14:
        return []

    discriminant = (
        quadratic_b**2
        - 4.0 * quadratic_a * quadratic_c
    )

    if discriminant < -1.0e-14:
        return []

    discriminant = max(discriminant, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    k_squared_candidates = [
        (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a),
        (-quadratic_b - sqrt_discriminant) / (2.0 * quadratic_a),
    ]

    roots = []

    for k_squared in k_squared_candidates:
        if k_squared <= 1.0e-14:
            continue

        k = np.sqrt(k_squared)
        residual = abs(
            energy(k, phi, alpha, tj, band)
            - target_energy
        )

        if residual < 1.0e-7:
            roots.append(k)

    unique_roots = []

    for k in sorted(roots):
        if not unique_roots or abs(k - unique_roots[-1]) > 1.0e-9:
            unique_roots.append(k)

    return unique_roots


def intraband_energy_radial_derivative(
    k: float,
    phi: float,
    alpha: float,
    tj: float,
    band: int,
) -> float:
    """Return the radial derivative d epsilon_band / dk."""
    sin_2phi = np.sin(2.0 * phi)
    d_value = d_of(k, phi, alpha, tj)

    if d_value < 1.0e-30:
        return np.nan

    splitting_derivative = (
        alpha**2 * k
        + 2.0 * tj**2 * k**3 * sin_2phi**2
    ) / d_value

    return 2.0 * T0 * k + band * splitting_derivative


def intraband_velocity_x(
    k: float,
    phi: float,
    alpha: float,
    tj: float,
    band: int,
) -> float:
    """Return the band-diagonal x velocity."""
    kx = k * np.cos(phi)
    ky = k * np.sin(phi)
    d_value = d_of(k, phi, alpha, tj)

    if d_value < 1.0e-30:
        return np.nan

    velocity_factor = (
        2.0 * T0
        + band * (
            alpha**2
            + 4.0 * tj**2 * ky**2
        ) / d_value
    )

    return (kx / HBAR_EVS) * velocity_factor


def intraband_charge_weight_xx(
    k: float,
    phi: float,
    alpha: float,
    tj: float,
    band: int,
) -> float:
    """Return the intraband charge weight [v_x^(band,band)]^2."""
    velocity_x = intraband_velocity_x(
        k,
        phi,
        alpha,
        tj,
        band,
    )

    return velocity_x**2


def intraband_contour_integral_xx(
    mu: float,
    alpha: float,
    tj: float,
    band: int,
    nphi: int = INTRA_NPHI,
) -> float:
    """
    Evaluate the Fermi-contour integral for one band.

    I_xx^band(mu) =
        integral dphi
        sum_roots
        k g_xx(k, phi) / |d epsilon_band / dk|.
    """
    phi_values = np.linspace(
        0.0,
        2.0 * np.pi,
        nphi,
        endpoint=False,
    )

    angular_values = np.zeros_like(phi_values)

    for index, phi in enumerate(phi_values):
        roots = intraband_roots_k(
            target_energy=mu,
            phi=phi,
            alpha=alpha,
            tj=tj,
            band=band,
        )

        accumulated_value = 0.0

        for k in roots:
            numerator = (
                k
                * intraband_charge_weight_xx(
                    k,
                    phi,
                    alpha,
                    tj,
                    band,
                )
            )

            denominator = abs(
                intraband_energy_radial_derivative(
                    k,
                    phi,
                    alpha,
                    tj,
                    band,
                )
            )

            if (
                np.isfinite(numerator)
                and np.isfinite(denominator)
                and denominator > 0.0
            ):
                accumulated_value += numerator / denominator

        angular_values[index] = accumulated_value

    return np.trapezoid(
        angular_values,
        phi_values,
    )


def intraband_sigma_xx_tilde(
    hbar_contour_integral: float,
    frequency_grid: np.ndarray = INTRA_EW,
    eta: float = INTRA_ETA,
) -> np.ndarray:
    """
    Return the normalized intraband conductivity.

    This expression preserves the normalization used in the original
    intraband script.
    """
    denominator = (
        frequency_grid + 1j * eta
    ) * (2.0 * np.pi) ** 2

    return (
        1j
        * HBAR_EVS
        * hbar_contour_integral
        / denominator
    )


def make_intraband_cases(
    sweep_mode: str,
) -> list[dict]:
    """Build the intraband parameter cases for one sweep."""
    if sweep_mode == "mu":
        return [
            {
                "label": (
                    rf"$\mu={1.0e3 * mu:.1f}"
                    rf"\,\mathrm{{meV}}$"
                ),
                "mu": mu,
                "alpha": BASE_ALPHA,
                "tj": BASE_TJ,
                "sweep_value": 1.0e3 * mu,
                "sweep_unit": "meV",
            }
            for mu in INTRA_MU_SWEEP
        ]

    if sweep_mode == "alpha":
        return [
            {
                "label": (
                    rf"$\alpha={1.0e3 * alpha:.0f}"
                    rf"\,\mathrm{{meV\AA}}$"
                ),
                "mu": INTRA_FIXED_MU,
                "alpha": alpha,
                "tj": INTRA_ALPHA_SWEEP_FIXED_TJ,
                "sweep_value": 1.0e3 * alpha,
                "sweep_unit": "meV*A",
            }
            for alpha in INTRA_ALPHA_SWEEP
        ]

    if sweep_mode == "tj":
        return [
            {
                "label": (
                    rf"$t_j={tj:.3g}"
                    rf"\,\mathrm{{eV\AA^2}}$"
                ),
                "mu": INTRA_FIXED_MU,
                "alpha": INTRA_TJ_SWEEP_FIXED_ALPHA,
                "tj": tj,
                "sweep_value": tj,
                "sweep_unit": "eV*A^2",
            }
            for tj in INTRA_TJ_SWEEP
        ]

    raise ValueError(
        "sweep_mode must be 'mu', 'alpha', or 'tj'."
    )


def compute_intraband_sweep(
    sweep_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Calculate and save one intraband parameter sweep."""
    cases = make_intraband_cases(sweep_mode)

    point_records = []
    curve_records = []

    for case in cases:
        mu = case["mu"]
        alpha = case["alpha"]
        tj = case["tj"]

        print(
            f"Computing intraband {sweep_mode} sweep: "
            f"{case['label']}, "
            f"mu={1.0e3 * mu:.4g} meV, "
            f"alpha={1.0e3 * alpha:.4g} meV*A, "
            f"tj={tj:.4g} eV*A^2"
        )

        integral_plus = intraband_contour_integral_xx(
            mu,
            alpha,
            tj,
            band=+1,
        )

        integral_minus = intraband_contour_integral_xx(
            mu,
            alpha,
            tj,
            band=-1,
        )

        integral_total = integral_plus + integral_minus
        hbar_integral_total = HBAR_EVS * integral_total

        point_records.append(
            {
                "contribution": "intraband",
                "sweep": sweep_mode,
                "label": case["label"],
                "sweep_value": case["sweep_value"],
                "sweep_unit": case["sweep_unit"],
                "mu_eV": mu,
                "mu_meV": 1.0e3 * mu,
                "alpha_eVA": alpha,
                "alpha_meVA": 1.0e3 * alpha,
                "tj_eVA2": tj,
                "I_xx_plus": integral_plus,
                "I_xx_minus": integral_minus,
                "I_xx_total": integral_total,
                "hbar_I_xx_total": hbar_integral_total,
            }
        )

        conductivity = intraband_sigma_xx_tilde(
            hbar_integral_total
        )

        for frequency, value in zip(
            INTRA_EW,
            conductivity,
        ):
            curve_records.append(
                {
                    "contribution": "intraband",
                    "sweep": sweep_mode,
                    "label": case["label"],
                    "sweep_value": case["sweep_value"],
                    "sweep_unit": case["sweep_unit"],
                    "mu_meV": 1.0e3 * mu,
                    "alpha_meVA": 1.0e3 * alpha,
                    "tj_eVA2": tj,
                    "hbar_omega_meV": 1.0e3 * frequency,
                    "Re_sigma_xx_tilde": np.real(value),
                    "Im_sigma_xx_tilde": np.imag(value),
                }
            )

    points_dataframe = pd.DataFrame(point_records)
    curves_dataframe = pd.DataFrame(curve_records)

    points_path = (
        INTRA_OUTPUT_DIR
        / f"sigma_xx_intra_{sweep_mode}_sweep_points.csv"
    )

    curves_path = (
        INTRA_OUTPUT_DIR
        / f"sigma_xx_intra_{sweep_mode}_sweep_curves.csv"
    )

    points_dataframe.to_csv(
        points_path,
        index=False,
    )

    curves_dataframe.to_csv(
        curves_path,
        index=False,
    )

    return (
        points_dataframe,
        curves_dataframe,
        points_path,
        curves_path,
    )


def intraband_plot_title(
    sweep_mode: str,
) -> str:
    """Return the title for an intraband sweep."""
    if sweep_mode == "mu":
        return (
            r"Intraband $\sigma_{xx}$: "
            rf"$\alpha={1.0e3 * BASE_ALPHA:.0f}"
            r"\,\mathrm{meV\AA}$, "
            rf"$t_j={BASE_TJ:.3g}"
            r"\,\mathrm{eV\AA^2}$"
        )

    if sweep_mode == "alpha":
        return (
            r"Intraband $\sigma_{xx}$ alpha sweep: "
            rf"$\mu={1.0e3 * INTRA_FIXED_MU:.1f}"
            r"\,\mathrm{meV}$, "
            rf"$t_j={INTRA_ALPHA_SWEEP_FIXED_TJ:.3g}"
            r"\,\mathrm{eV\AA^2}$"
        )

    if sweep_mode == "tj":
        return (
            r"Intraband $\sigma_{xx}$ $t_j$ sweep: "
            rf"$\mu={1.0e3 * INTRA_FIXED_MU:.1f}"
            r"\,\mathrm{meV}$, "
            rf"$\alpha={1.0e3 * INTRA_TJ_SWEEP_FIXED_ALPHA:.0f}"
            r"\,\mathrm{meV\AA}$"
        )

    return r"Intraband $\sigma_{xx}$ sweep"


def plot_intraband_sweep(
    points_dataframe: pd.DataFrame,
    sweep_mode: str,
) -> Path:
    """Plot the real and imaginary intraband conductivity."""
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
        sharex=True,
    )

    real_axis, imaginary_axis = axes

    for color, (_, row) in zip(
        colors,
        points_dataframe.iterrows(),
    ):
        conductivity = intraband_sigma_xx_tilde(
            row["hbar_I_xx_total"]
        )

        real_axis.plot(
            1.0e3 * INTRA_EW,
            np.real(conductivity),
            linewidth=2.6,
            color=color,
            label=row["label"],
        )

        imaginary_axis.plot(
            1.0e3 * INTRA_EW,
            np.imag(conductivity),
            linewidth=2.6,
            color=color,
            label=row["label"],
        )

    for axis in axes:
        axis.axhline(
            0.0,
            color="0.85",
            linewidth=1.0,
            zorder=0,
        )

        axis.set_xlim(
            0.0,
            14.0,
        )

        axis.set_xlabel(
            r"$\hbar\omega\,(\mathrm{meV})$"
        )

        axis.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
        )

        axis.minorticks_on()

        axis.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
        )

    real_axis.set_ylabel(
        r"$\mathrm{Re}\!\left["
        r"\frac{\hbar}{e^2}"
        r"\sigma_{xx}^{\mathrm{intra}}"
        r"\right]$"
    )

    imaginary_axis.set_ylabel(
        r"$\mathrm{Im}\!\left["
        r"\frac{\hbar}{e^2}"
        r"\sigma_{xx}^{\mathrm{intra}}"
        r"\right]$"
    )

    real_axis.text(
        0.03,
        0.93,
        r"(a)",
        transform=real_axis.transAxes,
        fontsize=18,
    )

    imaginary_axis.text(
        0.03,
        0.93,
        r"(b)",
        transform=imaginary_axis.transAxes,
        fontsize=18,
    )

    figure.suptitle(
        intraband_plot_title(sweep_mode),
        y=1.03,
        fontsize=16,
    )

    figure.tight_layout()

    figure_path = (
        INTRA_OUTPUT_DIR
        / f"sigma_xx_intra_{sweep_mode}_sweep_re_im.png"
    )

    figure.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)

    return figure_path


def run_intraband_sweep(
    sweep_mode: str,
) -> None:
    """Compute, plot, and report one intraband sweep."""
    (
        points_dataframe,
        _,
        points_path,
        curves_path,
    ) = compute_intraband_sweep(sweep_mode)

    figure_path = plot_intraband_sweep(
        points_dataframe,
        sweep_mode,
    )

    print("\nSaved intraband outputs:")
    print(points_path)
    print(curves_path)
    print(figure_path)


# ============================================================
# INTERBAND CONDUCTIVITY
# ============================================================

def build_interband_grid() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Construct the polar momentum grid used by the interband integral."""
    k_grid = np.linspace(
        1.0e-7,
        INTER_KMAX,
        INTER_NK,
    )

    phi_grid = np.linspace(
        0.0,
        2.0 * np.pi,
        INTER_NPHI,
        endpoint=False,
    )

    radial_grid, angular_grid = np.meshgrid(
        k_grid,
        phi_grid,
        indexing="xy",
    )

    ky_grid = radial_grid * np.sin(angular_grid)

    return (
        k_grid,
        phi_grid,
        radial_grid,
        angular_grid,
        ky_grid,
    )


def interband_sigma_xx_tilde(
    mu: float,
    alpha: float,
    tj: float,
    k_grid: np.ndarray,
    phi_grid: np.ndarray,
    radial_grid: np.ndarray,
    angular_grid: np.ndarray,
    ky_grid: np.ndarray,
    frequency_grid: np.ndarray = INTER_EW,
    eta: float = INTER_ETA,
) -> np.ndarray:
    """
    Return the normalized interband conductivity.

    The zero-temperature Pauli window is one where

        epsilon_-(k) < mu < epsilon_+(k).
    """
    splitting = d_of(
        radial_grid,
        angular_grid,
        alpha,
        tj,
    )

    energy_plus = T0 * radial_grid**2 + splitting
    energy_minus = T0 * radial_grid**2 - splitting

    pauli_window = (
        (energy_minus < mu)
        & (mu < energy_plus)
    ).astype(float)

    numerator = (
        alpha**2
        * ky_grid**2
        * (
            alpha**2
            + 4.0 * tj**2 * ky_grid**2
        )
    )

    conductivity_values = []

    for frequency in frequency_grid:
        complex_frequency = frequency + 1j * eta

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            integrand = (
                pauli_window
                * complex_frequency
                * numerator
                / (
                    splitting**3
                    * (
                        complex_frequency**2
                        - 4.0 * splitting**2
                    )
                )
            )

        integrand = np.nan_to_num(
            integrand,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        polar_integrand = radial_grid * integrand

        radial_integral = np.trapezoid(
            polar_integrand,
            k_grid,
            axis=1,
        )

        momentum_integral = np.trapezoid(
            radial_integral,
            phi_grid,
        )

        conductivity_values.append(
            1j
            * momentum_integral
            / (2.0 * np.pi) ** 2
        )

    return np.asarray(conductivity_values)


def make_interband_cases(
    sweep_mode: str,
) -> tuple[list[dict], str]:
    """Build the interband cases and fixed-parameter title."""
    if sweep_mode == "mu":
        cases = [
            {
                "mu": mu,
                "alpha": INTER_ALPHA_REF,
                "tj": INTER_TJ_REF,
                "label": (
                    rf"$\mu={1.0e3 * mu:.1f}"
                    rf"\,\mathrm{{meV}}$"
                ),
                "varied_label": "mu_meV",
                "varied_value": 1.0e3 * mu,
            }
            for mu in INTER_MU_SWEEP
        ]

        fixed_title = (
            rf"fixed $\alpha={1.0e3 * INTER_ALPHA_REF:.0f}"
            r"\,\mathrm{meV\AA}$, "
            rf"fixed $t_j={INTER_TJ_REF:.3f}"
            r"\,\mathrm{eV\AA^2}$"
        )

        return cases, fixed_title

    if sweep_mode == "alpha":
        cases = [
            {
                "mu": INTER_MU_REF,
                "alpha": alpha,
                "tj": INTER_TJ_REF,
                "label": (
                    rf"$\alpha={1.0e3 * alpha:.0f}"
                    rf"\,\mathrm{{meV\AA}}$"
                ),
                "varied_label": "alpha_meV_A",
                "varied_value": 1.0e3 * alpha,
            }
            for alpha in INTER_ALPHA_SWEEP
        ]

        fixed_title = (
            rf"fixed $\mu={1.0e3 * INTER_MU_REF:.1f}"
            r"\,\mathrm{meV}$, "
            rf"fixed $t_j={INTER_TJ_REF:.3f}"
            r"\,\mathrm{eV\AA^2}$"
        )

        return cases, fixed_title

    if sweep_mode == "tj":
        cases = [
            {
                "mu": INTER_MU_REF,
                "alpha": INTER_ALPHA_REF,
                "tj": tj,
                "label": (
                    rf"$t_j={tj:.3f}"
                    rf"\,\mathrm{{eV\AA^2}}$"
                ),
                "varied_label": "tj_eV_A2",
                "varied_value": tj,
            }
            for tj in INTER_TJ_SWEEP
        ]

        fixed_title = (
            rf"fixed $\mu={1.0e3 * INTER_MU_REF:.1f}"
            r"\,\mathrm{meV}$, "
            rf"fixed $\alpha={1.0e3 * INTER_ALPHA_REF:.0f}"
            r"\,\mathrm{meV\AA}$"
        )

        return cases, fixed_title

    raise ValueError(
        "sweep_mode must be 'mu', 'alpha', or 'tj'."
    )


def compute_interband_sweep(
    sweep_mode: str,
    interband_grid: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[pd.DataFrame, str]:
    """Calculate one interband parameter sweep."""
    (
        k_grid,
        phi_grid,
        radial_grid,
        angular_grid,
        ky_grid,
    ) = interband_grid

    cases, fixed_title = make_interband_cases(
        sweep_mode
    )

    records = []

    for case in cases:
        print(
            f"Computing interband {sweep_mode} sweep: "
            f"{case['label']}, "
            f"mu={1.0e3 * case['mu']:.4g} meV, "
            f"alpha={1.0e3 * case['alpha']:.4g} meV*A, "
            f"tj={case['tj']:.4g} eV*A^2"
        )

        conductivity = interband_sigma_xx_tilde(
            mu=case["mu"],
            alpha=case["alpha"],
            tj=case["tj"],
            k_grid=k_grid,
            phi_grid=phi_grid,
            radial_grid=radial_grid,
            angular_grid=angular_grid,
            ky_grid=ky_grid,
        )

        for frequency, value in zip(
            INTER_EW,
            conductivity,
        ):
            record = {
                "contribution": "interband",
                "sweep": sweep_mode,
                "label": case["label"],
                "mu_eV": case["mu"],
                "mu_meV": 1.0e3 * case["mu"],
                "alpha_eV_A": case["alpha"],
                "alpha_meV_A": 1.0e3 * case["alpha"],
                "tj_eV_A2": case["tj"],
                "eta_eV": INTER_ETA,
                "eta_meV": 1.0e3 * INTER_ETA,
                "hbar_omega_eV": frequency,
                "hbar_omega_meV": 1.0e3 * frequency,
                "Re_sigma_xx_inter_tilde": np.real(value),
                "Im_sigma_xx_inter_tilde": np.imag(value),
                case["varied_label"]: case["varied_value"],
            }

            records.append(record)

    return pd.DataFrame(records), fixed_title


def plot_interband_sweep(
    dataframe: pd.DataFrame,
    sweep_mode: str,
    fixed_title: str,
) -> Path:
    """Plot the real and imaginary interband conductivity."""
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
    )

    real_axis, imaginary_axis = axes

    labels = list(
        dataframe["label"].drop_duplicates()
    )

    for color, label in zip(colors, labels):
        subset = dataframe[
            dataframe["label"] == label
        ]

        real_axis.plot(
            subset["hbar_omega_meV"],
            subset["Re_sigma_xx_inter_tilde"],
            linewidth=2.6,
            color=color,
            label=label,
        )

        imaginary_axis.plot(
            subset["hbar_omega_meV"],
            subset["Im_sigma_xx_inter_tilde"],
            linewidth=2.6,
            color=color,
            label=label,
        )

    for axis in axes:
        axis.axhline(
            0.0,
            color="0.85",
            linewidth=1.0,
            zorder=0,
        )

        axis.set_xlim(
            0.0,
            50.0,
        )

        axis.set_xlabel(
            r"$\hbar\omega\,\mathrm{(meV)}$"
        )

        axis.legend(
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
        )

        axis.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
        )

        axis.minorticks_on()

    real_axis.set_ylabel(
        r"$\mathrm{Re}\!\left["
        r"\frac{\hbar}{e^2}"
        r"\sigma_{xx}^{\mathrm{inter}}"
        r"\right]$"
    )

    imaginary_axis.set_ylabel(
        r"$\mathrm{Im}\!\left["
        r"\frac{\hbar}{e^2}"
        r"\sigma_{xx}^{\mathrm{inter}}"
        r"\right]$"
    )

    real_axis.text(
        0.03,
        0.93,
        r"(a)",
        transform=real_axis.transAxes,
        fontsize=20,
    )

    imaginary_axis.text(
        0.03,
        0.93,
        r"(b)",
        transform=imaginary_axis.transAxes,
        fontsize=20,
    )

    figure.suptitle(
        fixed_title,
        y=1.03,
        fontsize=14,
    )

    figure.tight_layout()

    figure_path = (
        INTER_OUTPUT_DIR
        / f"sigma_xx_inter_{sweep_mode}_sweep_re_im.png"
    )

    figure.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)

    return figure_path


def run_interband_sweep(
    sweep_mode: str,
    interband_grid: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    """Compute, save, plot, and report one interband sweep."""
    dataframe, fixed_title = compute_interband_sweep(
        sweep_mode,
        interband_grid,
    )

    csv_path = (
        INTER_OUTPUT_DIR
        / f"sigma_xx_inter_{sweep_mode}_sweep_curves.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    figure_path = plot_interband_sweep(
        dataframe,
        sweep_mode,
        fixed_title,
    )

    print("\nSaved interband outputs:")
    print(csv_path)
    print(figure_path)


# ============================================================
# PLOT STYLE
# ============================================================

def configure_plot_style() -> None:
    """Set the typography shared by all output figures."""
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 18,
            "legend.fontsize": 11,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "mathtext.fontset": "dejavuserif",
            "font.family": "serif",
        }
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def selected_sweep_modes() -> list[str]:
    """Return and validate the requested sweep modes."""
    valid_modes = {
        "mu",
        "alpha",
        "tj",
        "all",
    }

    if RUN_MODE not in valid_modes:
        raise ValueError(
            "RUN_MODE must be 'mu', 'alpha', 'tj', or 'all'."
        )

    if RUN_MODE == "all":
        return [
            "mu",
            "alpha",
            "tj",
        ]

    return [RUN_MODE]


def selected_contributions() -> list[str]:
    """Return and validate the requested conductivity contributions."""
    valid_modes = {
        "intra",
        "inter",
        "both",
    }

    if CONTRIBUTION_MODE not in valid_modes:
        raise ValueError(
            "CONTRIBUTION_MODE must be "
            "'intra', 'inter', or 'both'."
        )

    if CONTRIBUTION_MODE == "both":
        return [
            "intra",
            "inter",
        ]

    return [CONTRIBUTION_MODE]


def main() -> None:
    """Run the selected conductivity contributions and parameter sweeps."""
    configure_plot_style()

    sweep_modes = selected_sweep_modes()
    contributions = selected_contributions()

    interband_grid = None

    if "inter" in contributions:
        print("Building the interband momentum grid...")
        interband_grid = build_interband_grid()

    for contribution in contributions:
        for sweep_mode in sweep_modes:
            if contribution == "intra":
                run_intraband_sweep(
                    sweep_mode
                )

            elif contribution == "inter":
                if interband_grid is None:
                    raise RuntimeError(
                        "The interband grid was not initialized."
                    )

                run_interband_sweep(
                    sweep_mode,
                    interband_grid,
                )

    print(
        "\nAll selected calculations are complete."
    )

    print(
        f"Output root: {OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()
