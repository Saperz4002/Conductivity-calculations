"""
Intraband and interband spin Hall conductivity
==============================================

Overview
--------
This script calculates the frequency-dependent transverse spin conductivity

    sigma_xy^{s_z}

of a two-dimensional Rashba--altermagnetic continuum model. The band
energies are

    epsilon_lambda(k, phi) = t0 k^2 + lambda d(k, phi),

where lambda = +1 or -1 and

    d(k, phi) =
        sqrt[
            alpha^2 k^2
            + tj^2 k^4 sin^2(2 phi)
        ].

The program combines two separate numerical calculations:

1. Intraband contribution
   The Fermi-surface response is evaluated as an angular integral over the
   radial roots of

       epsilon_lambda(k, phi) = mu.

   For each band and polar angle, the code calculates the spin-Hall weight,
   divides it by the radial energy derivative, and sums the contributions
   from all valid Fermi-contour roots.

2. Interband contribution
   The finite-frequency response from transitions between the two bands is
   evaluated on a two-dimensional polar momentum grid. At zero temperature,
   only momenta satisfying the Pauli window

       epsilon_-(k, phi) < mu < epsilon_+(k, phi)

   contribute to the integral.

Contribution modes
------------------
Set ``CONTRIBUTION_MODE`` to one of the following:

- ``"intra"``
  Calculate only the intraband spin Hall conductivity.

- ``"inter"``
  Calculate only the interband spin Hall conductivity.

- ``"both"``
  Calculate both contributions sequentially and save them separately.

Sweep modes
-----------
Set ``RUN_MODE`` to one of the following:

- ``"mu"``
  Sweep the chemical potential.

- ``"alpha"``
  Sweep the Rashba spin-orbit coupling.

- ``"tj"``
  Sweep the d-wave altermagnetic coupling.

- ``"all"``
  Run all three parameter sweeps.

Normalizations
--------------
To preserve the conventions of the two original programs, the saved
quantities use their original normalizations:

- intraband:
      (hbar / e) sigma_xy,intra^{s_z}

- interband:
      sigma_xy,inter^{s_z} / e

The script does not add the two arrays into a total conductivity because
their original output conventions are not identical. Selecting ``"both"``
runs and saves both calculations without changing either normalization.

Numerical settings
------------------
The intraband and interband calculations retain separate frequency grids
and numerical resolutions:

- the intraband contribution uses a high-resolution Fermi-contour angular
  integral;
- the interband contribution uses a polar two-dimensional momentum grid.

These settings can be adjusted independently in their corresponding
configuration sections.

Outputs
-------
The program creates separate ``intraband`` and ``interband`` directories
inside ``OUTPUT_ROOT``. Each contribution is further organized by sweep
type.

For every selected intraband sweep, it saves:

- a CSV file containing the upper-band, lower-band, and total contour
  integrals;
- a CSV file containing the real and imaginary conductivity curves;
- a PNG figure containing paired real and imaginary panels.

For every selected interband sweep, it saves:

- a CSV file containing the real and imaginary conductivity curves;
- a PNG figure containing paired real and imaginary panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION
# ============================================================

# Contribution options:
#   "intra" -> intraband contribution only
#   "inter" -> interband contribution only
#   "both"  -> run both contributions
CONTRIBUTION_MODE = "both"

# Sweep options:
#   "mu", "alpha", "tj", or "all"
RUN_MODE = "all"

# Root output directory
OUTPUT_ROOT = Path("spin_hall_xy_sweeps")


# ============================================================
# SHARED CONSTANTS AND REFERENCE PARAMETERS
# ============================================================

HBAR_EVS = 6.582119569e-16  # eV s
T0 = 1.0 / (2.0 * 0.152)    # eV Å^2

MU_REFERENCE = 5.5e-3        # eV
ALPHA_REFERENCE = 0.026      # eV Å = 26 meV Å
TJ_REFERENCE = 1.881         # eV Å^2

ETA = 1.0e-3                 # eV

MU_VALUES = np.array([
    1.0e-4,
    1.0e-3,
    1.9e-3,
    2.8e-3,
    5.5e-3,
])

ALPHA_VALUES = np.array([
    0.007,
    0.013,
    0.026,
    0.052,
    0.100,
])

TJ_VALUES = np.array([
    0.50,
    1.00,
    1.881,
    2.50,
    3.00,
])


# ============================================================
# INTRABAND NUMERICAL CONFIGURATION
# ============================================================

INTRA_FREQUENCY_GRID = np.linspace(
    0.0,
    0.020,
    500,
)

INTRA_NPHI = 8000
INTRA_PLOT_XMAX_MEV = 14.0


# ============================================================
# INTERBAND NUMERICAL CONFIGURATION
# ============================================================

INTER_FREQUENCY_GRID = np.linspace(
    0.0,
    0.060,
    500,
)

INTER_KMAX = 0.15
INTER_NK = 900
INTER_NPHI = 501
INTER_PLOT_XMAX_MEV = 50.0


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

INTRA_OUTPUT_ROOT = OUTPUT_ROOT / "intraband"
INTER_OUTPUT_ROOT = OUTPUT_ROOT / "interband"

for contribution_root in (
    INTRA_OUTPUT_ROOT,
    INTER_OUTPUT_ROOT,
):
    for sweep_name in (
        "mu",
        "alpha",
        "tj",
    ):
        (
            contribution_root
            / sweep_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# PLOT STYLE
# ============================================================

PLOT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
]

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
    """Return epsilon_band(k, phi) for band = +1 or -1."""
    if band not in (-1, +1):
        raise ValueError(
            "band must be either +1 or -1."
        )

    return (
        T0 * k**2
        + band * d_of(
            k,
            phi,
            alpha,
            tj,
        )
    )


# ============================================================
# SWEEP CONFIGURATION
# ============================================================

def get_sweep_config(
    sweep_mode: str,
) -> dict:
    """Return the values, parameter mapping, labels, and title for a sweep."""
    if sweep_mode == "mu":
        return {
            "values": MU_VALUES,
            "make_parameters": (
                lambda value: (
                    value,
                    ALPHA_REFERENCE,
                    TJ_REFERENCE,
                )
            ),
            "legend_label": (
                lambda value:
                rf"$\mu={1.0e3 * value:.1f}"
                rf"\,\mathrm{{meV}}$"
            ),
            "title": (
                rf"fixed $\alpha={1.0e3 * ALPHA_REFERENCE:.0f}"
                r"\,\mathrm{meV\AA}$, "
                rf"fixed $t_j={TJ_REFERENCE:.3f}"
                r"\,\mathrm{eV\AA^2}$"
            ),
        }

    if sweep_mode == "alpha":
        return {
            "values": ALPHA_VALUES,
            "make_parameters": (
                lambda value: (
                    MU_REFERENCE,
                    value,
                    TJ_REFERENCE,
                )
            ),
            "legend_label": (
                lambda value:
                rf"$\alpha={1.0e3 * value:.0f}"
                rf"\,\mathrm{{meV\AA}}$"
            ),
            "title": (
                rf"fixed $\mu={1.0e3 * MU_REFERENCE:.1f}"
                r"\,\mathrm{meV}$, "
                rf"fixed $t_j={TJ_REFERENCE:.3f}"
                r"\,\mathrm{eV\AA^2}$"
            ),
        }

    if sweep_mode == "tj":
        return {
            "values": TJ_VALUES,
            "make_parameters": (
                lambda value: (
                    MU_REFERENCE,
                    ALPHA_REFERENCE,
                    value,
                )
            ),
            "legend_label": (
                lambda value:
                rf"$t_j={value:.3f}"
                rf"\,\mathrm{{eV\AA^2}}$"
            ),
            "title": (
                rf"fixed $\mu={1.0e3 * MU_REFERENCE:.1f}"
                r"\,\mathrm{meV}$, "
                rf"fixed $\alpha={1.0e3 * ALPHA_REFERENCE:.0f}"
                r"\,\mathrm{meV\AA}$"
            ),
        }

    raise ValueError(
        "sweep_mode must be 'mu', 'alpha', or 'tj'."
    )


# ============================================================
# INTRABAND ROOTS AND FERMI-CONTOUR INTEGRAL
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

    Only positive roots that reproduce the selected band are retained.
    """
    sin_2phi = np.sin(2.0 * phi)

    coefficient_a = (
        T0**2
        - tj**2 * sin_2phi**2
    )

    coefficient_b = -(
        2.0 * T0 * target_energy
        + alpha**2
    )

    coefficient_c = target_energy**2

    k_squared_candidates = []

    if abs(coefficient_a) < 1.0e-14:
        if abs(coefficient_b) > 1.0e-14:
            k_squared_candidates = [
                -coefficient_c
                / coefficient_b
            ]

    else:
        discriminant = (
            coefficient_b**2
            - 4.0
            * coefficient_a
            * coefficient_c
        )

        if discriminant < -1.0e-14:
            return []

        discriminant = max(
            discriminant,
            0.0,
        )

        sqrt_discriminant = np.sqrt(
            discriminant
        )

        k_squared_candidates = [
            (
                -coefficient_b
                + sqrt_discriminant
            ) / (
                2.0
                * coefficient_a
            ),
            (
                -coefficient_b
                - sqrt_discriminant
            ) / (
                2.0
                * coefficient_a
            ),
        ]

    roots = []

    for k_squared in k_squared_candidates:
        if k_squared <= 1.0e-14:
            continue

        k = np.sqrt(k_squared)

        residual = abs(
            energy(
                k,
                phi,
                alpha,
                tj,
                band,
            )
            - target_energy
        )

        if residual < 1.0e-7:
            roots.append(k)

    unique_roots = []

    for k in sorted(roots):
        if (
            not unique_roots
            or abs(
                k
                - unique_roots[-1]
            ) > 1.0e-9
        ):
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
    splitting = d_of(
        k,
        phi,
        alpha,
        tj,
    )

    if splitting < 1.0e-30:
        return np.nan

    splitting_derivative = (
        alpha**2 * k
        + 2.0
        * tj**2
        * k**3
        * sin_2phi**2
    ) / splitting

    return (
        2.0 * T0 * k
        + band
        * splitting_derivative
    )


def intraband_spin_hall_weight(
    k: float,
    phi: float,
    alpha: float,
    tj: float,
    band: int,
) -> float:
    """
    Return the band-diagonal intraband spin-Hall weight.

    This expression is preserved from the original intraband program.
    """
    kx = k * np.cos(phi)
    ky = k * np.sin(phi)

    splitting = d_of(
        k,
        phi,
        alpha,
        tj,
    )

    if splitting < 1.0e-30:
        return np.nan

    first_factor = (
        tj
        * ky**2
        / HBAR_EVS
    )

    second_factor = (
        1.0
        + band
        * (
            2.0
            * T0
            * kx**2
        )
        / splitting
    )

    third_factor = (
        2.0 * T0
        + band
        * (
            alpha**2
            + 4.0
            * tj**2
            * kx**2
        )
        / splitting
    )

    return (
        first_factor
        * second_factor
        * third_factor
    )


def intraband_contour_integral(
    mu: float,
    alpha: float,
    tj: float,
    band: int,
    nphi: int = INTRA_NPHI,
) -> float:
    """
    Evaluate the Fermi-contour integral for one band.

    The original high angular resolution is retained by default.
    """
    phi_values = np.linspace(
        0.0,
        2.0 * np.pi,
        nphi,
        endpoint=False,
    )

    angular_values = np.zeros_like(
        phi_values
    )

    for index, phi in enumerate(
        phi_values
    ):
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
                * intraband_spin_hall_weight(
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
                accumulated_value += (
                    numerator
                    / denominator
                )

        angular_values[index] = (
            accumulated_value
        )

    return np.trapezoid(
        angular_values,
        phi_values,
    )


def intraband_sigma_xy_tilde(
    hbar_integral: float,
    frequency_grid: np.ndarray = INTRA_FREQUENCY_GRID,
    eta: float = ETA,
) -> np.ndarray:
    """
    Return the normalized intraband response

        (hbar/e) sigma_xy,intra^{s_z}.
    """
    return (
        -hbar_integral
        / (
            1j
            * (
                frequency_grid
                + 1j * eta
            )
            * (2.0 * np.pi) ** 2
        )
    )


# ============================================================
# INTERBAND MOMENTUM GRID AND INTEGRAL
# ============================================================

def build_interband_grid() -> tuple[
    np.ndarray,
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

    kx_grid = (
        radial_grid
        * np.cos(angular_grid)
    )

    return (
        k_grid,
        phi_grid,
        radial_grid,
        angular_grid,
        kx_grid,
    )


def interband_sigma_xy_over_e(
    mu: float,
    alpha: float,
    tj: float,
    interband_grid: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    frequency_grid: np.ndarray = INTER_FREQUENCY_GRID,
    eta: float = ETA,
) -> np.ndarray:
    """
    Return the interband spin Hall conductivity divided by e.

    The matrix-element product and global prefactor are preserved from the
    original interband program.
    """
    (
        k_grid,
        phi_grid,
        radial_grid,
        angular_grid,
        kx_grid,
    ) = interband_grid

    splitting = d_of(
        radial_grid,
        angular_grid,
        alpha,
        tj,
    )

    energy_plus = (
        T0 * radial_grid**2
        + splitting
    )

    energy_minus = (
        T0 * radial_grid**2
        - splitting
    )

    pauli_window = (
        (energy_minus < mu)
        & (mu < energy_plus)
    ).astype(float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        matrix_element_product = (
            T0
            * alpha**2
            * kx_grid**2
            / (
                HBAR_EVS
                * splitting
            )
        ) * (
            2.0
            * tj
            * kx_grid**2
            / splitting
            - 1j
        )

    matrix_element_product = np.nan_to_num(
        matrix_element_product,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    conductivity_values = []

    for frequency in frequency_grid:
        complex_frequency = (
            frequency
            + 1j * eta
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            integrand = (
                pauli_window
                / (2.0 * splitting)
                * (
                    matrix_element_product
                    / (
                        2.0
                        * splitting
                        + complex_frequency
                    )
                    + np.conj(
                        matrix_element_product
                    )
                    / (
                        -2.0
                        * splitting
                        + complex_frequency
                    )
                )
            )

        integrand = np.nan_to_num(
            integrand,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        polar_integrand = (
            radial_grid
            * integrand
        )

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
            * HBAR_EVS
            * momentum_integral
            / (2.0 * np.pi) ** 2
        )

    return np.asarray(
        conductivity_values
    )


# ============================================================
# INTRABAND SWEEP, DATA, AND PLOT
# ============================================================

def compute_intraband_sweep(
    sweep_mode: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute one intraband parameter sweep."""
    configuration = get_sweep_config(
        sweep_mode
    )

    integral_records = []
    curve_records = []

    for value in configuration["values"]:
        mu, alpha, tj = configuration[
            "make_parameters"
        ](value)

        print(
            f"Computing intraband {sweep_mode} sweep: "
            f"mu={1.0e3 * mu:.4g} meV, "
            f"alpha={1.0e3 * alpha:.4g} meV*A, "
            f"tj={tj:.4g} eV*A^2"
        )

        integral_plus = (
            intraband_contour_integral(
                mu,
                alpha,
                tj,
                band=+1,
            )
        )

        integral_minus = (
            intraband_contour_integral(
                mu,
                alpha,
                tj,
                band=-1,
            )
        )

        integral_total = (
            integral_plus
            + integral_minus
        )

        hbar_integral_total = (
            HBAR_EVS
            * integral_total
        )

        integral_records.append(
            {
                "contribution": "intraband",
                "sweep_mode": sweep_mode,
                "sweep_value": value,
                "mu_eV": mu,
                "mu_meV": 1.0e3 * mu,
                "alpha_eV_A": alpha,
                "alpha_meV_A": 1.0e3 * alpha,
                "tj_eV_A2": tj,
                "I_spin_plus": integral_plus,
                "I_spin_minus": integral_minus,
                "I_spin_total": integral_total,
                "hbar_I_spin_total": hbar_integral_total,
            }
        )

        conductivity = intraband_sigma_xy_tilde(
            hbar_integral_total
        )

        for frequency, response in zip(
            INTRA_FREQUENCY_GRID,
            conductivity,
        ):
            curve_records.append(
                {
                    "contribution": "intraband",
                    "sweep_mode": sweep_mode,
                    "sweep_value": value,
                    "mu_eV": mu,
                    "mu_meV": 1.0e3 * mu,
                    "alpha_eV_A": alpha,
                    "alpha_meV_A": 1.0e3 * alpha,
                    "tj_eV_A2": tj,
                    "eta_eV": ETA,
                    "eta_meV": 1.0e3 * ETA,
                    "hbar_omega_eV": frequency,
                    "hbar_omega_meV": 1.0e3 * frequency,
                    "Re_sigma_xy_spin_intra_tilde": np.real(
                        response
                    ),
                    "Im_sigma_xy_spin_intra_tilde": np.imag(
                        response
                    ),
                }
            )

    return (
        pd.DataFrame(integral_records),
        pd.DataFrame(curve_records),
    )


def plot_intraband_sweep(
    curves_dataframe: pd.DataFrame,
    sweep_mode: str,
) -> plt.Figure:
    """Create paired real and imaginary intraband panels."""
    configuration = get_sweep_config(
        sweep_mode
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
    )

    real_axis, imaginary_axis = axes

    for color, value in zip(
        PLOT_COLORS,
        configuration["values"],
    ):
        subset = curves_dataframe[
            curves_dataframe["sweep_value"]
            == value
        ]

        label = configuration[
            "legend_label"
        ](value)

        real_axis.plot(
            subset["hbar_omega_meV"],
            subset[
                "Re_sigma_xy_spin_intra_tilde"
            ],
            linewidth=2.6,
            color=color,
            label=label,
        )

        imaginary_axis.plot(
            subset["hbar_omega_meV"],
            subset[
                "Im_sigma_xy_spin_intra_tilde"
            ],
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
            INTRA_PLOT_XMAX_MEV,
        )

        axis.set_xlabel(
            r"$\hbar\omega\,(\mathrm{meV})$"
        )

        axis.legend(
            loc="best",
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
        r"\frac{\hbar}{e}"
        r"\sigma_{xy,\mathrm{intra}}^{s_z}"
        r"\right]$"
    )

    imaginary_axis.set_ylabel(
        r"$\mathrm{Im}\!\left["
        r"\frac{\hbar}{e}"
        r"\sigma_{xy,\mathrm{intra}}^{s_z}"
        r"\right]$"
    )

    real_axis.text(
        0.02,
        0.96,
        r"(a)",
        transform=real_axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=18,
    )

    imaginary_axis.text(
        0.02,
        0.96,
        r"(b)",
        transform=imaginary_axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=18,
    )

    figure.suptitle(
        configuration["title"],
        y=1.03,
        fontsize=15,
    )

    figure.tight_layout()

    return figure


def run_intraband_sweep(
    sweep_mode: str,
) -> None:
    """Compute, save, and plot one intraband sweep."""
    (
        integrals_dataframe,
        curves_dataframe,
    ) = compute_intraband_sweep(
        sweep_mode
    )

    output_directory = (
        INTRA_OUTPUT_ROOT
        / sweep_mode
    )

    integrals_path = (
        output_directory
        / (
            "spin_hall_intra_sweep_"
            f"{sweep_mode}_integrals.csv"
        )
    )

    curves_path = (
        output_directory
        / (
            "spin_hall_intra_sweep_"
            f"{sweep_mode}_curves.csv"
        )
    )

    figure_path = (
        output_directory
        / (
            "spin_hall_intra_sweep_"
            f"{sweep_mode}_pair.png"
        )
    )

    integrals_dataframe.to_csv(
        integrals_path,
        index=False,
    )

    curves_dataframe.to_csv(
        curves_path,
        index=False,
    )

    figure = plot_intraband_sweep(
        curves_dataframe,
        sweep_mode,
    )

    figure.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nSaved intraband outputs:")
    print(integrals_path)
    print(curves_path)
    print(figure_path)


# ============================================================
# INTERBAND SWEEP, DATA, AND PLOT
# ============================================================

def compute_interband_sweep(
    sweep_mode: str,
    interband_grid: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> pd.DataFrame:
    """Compute one interband parameter sweep."""
    configuration = get_sweep_config(
        sweep_mode
    )

    records = []

    for value in configuration["values"]:
        mu, alpha, tj = configuration[
            "make_parameters"
        ](value)

        print(
            f"Computing interband {sweep_mode} sweep: "
            f"mu={1.0e3 * mu:.4g} meV, "
            f"alpha={1.0e3 * alpha:.4g} meV*A, "
            f"tj={tj:.4g} eV*A^2"
        )

        conductivity = (
            interband_sigma_xy_over_e(
                mu=mu,
                alpha=alpha,
                tj=tj,
                interband_grid=interband_grid,
            )
        )

        for frequency, response in zip(
            INTER_FREQUENCY_GRID,
            conductivity,
        ):
            records.append(
                {
                    "contribution": "interband",
                    "sweep_mode": sweep_mode,
                    "sweep_value": value,
                    "mu_eV": mu,
                    "mu_meV": 1.0e3 * mu,
                    "alpha_eV_A": alpha,
                    "alpha_meV_A": 1.0e3 * alpha,
                    "tj_eV_A2": tj,
                    "eta_eV": ETA,
                    "eta_meV": 1.0e3 * ETA,
                    "hbar_omega_eV": frequency,
                    "hbar_omega_meV": 1.0e3 * frequency,
                    "Re_sigma_xy_spin_inter_over_e": np.real(
                        response
                    ),
                    "Im_sigma_xy_spin_inter_over_e": np.imag(
                        response
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def plot_interband_sweep(
    dataframe: pd.DataFrame,
    sweep_mode: str,
) -> plt.Figure:
    """Create paired real and imaginary interband panels."""
    configuration = get_sweep_config(
        sweep_mode
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
    )

    real_axis, imaginary_axis = axes

    for color, value in zip(
        PLOT_COLORS,
        configuration["values"],
    ):
        subset = dataframe[
            dataframe["sweep_value"]
            == value
        ]

        label = configuration[
            "legend_label"
        ](value)

        real_axis.plot(
            subset["hbar_omega_meV"],
            subset[
                "Re_sigma_xy_spin_inter_over_e"
            ],
            linewidth=2.6,
            color=color,
            label=label,
        )

        imaginary_axis.plot(
            subset["hbar_omega_meV"],
            subset[
                "Im_sigma_xy_spin_inter_over_e"
            ],
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
            INTER_PLOT_XMAX_MEV,
        )

        axis.set_xlabel(
            r"$\hbar\omega\,(\mathrm{meV})$"
        )

        axis.legend(
            loc="best",
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
        r"\sigma_{xy,\mathrm{inter}}^{s_z}/e"
        r"\right]$"
    )

    imaginary_axis.set_ylabel(
        r"$\mathrm{Im}\!\left["
        r"\sigma_{xy,\mathrm{inter}}^{s_z}/e"
        r"\right]$"
    )

    real_axis.text(
        0.02,
        0.96,
        r"(a)",
        transform=real_axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=18,
    )

    imaginary_axis.text(
        0.02,
        0.96,
        r"(b)",
        transform=imaginary_axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=18,
    )

    figure.suptitle(
        configuration["title"],
        y=1.03,
        fontsize=15,
    )

    figure.tight_layout()

    return figure


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
    """Compute, save, and plot one interband sweep."""
    dataframe = compute_interband_sweep(
        sweep_mode,
        interband_grid,
    )

    output_directory = (
        INTER_OUTPUT_ROOT
        / sweep_mode
    )

    csv_path = (
        output_directory
        / (
            "spin_hall_inter_sweep_"
            f"{sweep_mode}.csv"
        )
    )

    figure_path = (
        output_directory
        / (
            "spin_hall_inter_sweep_"
            f"{sweep_mode}_pair.png"
        )
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    figure = plot_interband_sweep(
        dataframe,
        sweep_mode,
    )

    figure.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nSaved interband outputs:")
    print(csv_path)
    print(figure_path)


# ============================================================
# MODE VALIDATION
# ============================================================

def selected_contributions() -> list[str]:
    """Validate and return the requested contribution modes."""
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

    return [
        CONTRIBUTION_MODE
    ]


def selected_sweeps() -> list[str]:
    """Validate and return the requested sweep modes."""
    valid_modes = {
        "mu",
        "alpha",
        "tj",
        "all",
    }

    if RUN_MODE not in valid_modes:
        raise ValueError(
            "RUN_MODE must be "
            "'mu', 'alpha', 'tj', or 'all'."
        )

    if RUN_MODE == "all":
        return [
            "mu",
            "alpha",
            "tj",
        ]

    return [
        RUN_MODE
    ]


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """Run the selected spin Hall contributions and parameter sweeps."""
    contributions = (
        selected_contributions()
    )

    sweep_modes = (
        selected_sweeps()
    )

    interband_grid = None

    if "inter" in contributions:
        print(
            "Building the interband momentum grid..."
        )

        interband_grid = (
            build_interband_grid()
        )

    for contribution in contributions:
        for sweep_mode in sweep_modes:
            if contribution == "intra":
                run_intraband_sweep(
                    sweep_mode
                )

            elif contribution == "inter":
                if interband_grid is None:
                    raise RuntimeError(
                        "The interband grid "
                        "was not initialized."
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
