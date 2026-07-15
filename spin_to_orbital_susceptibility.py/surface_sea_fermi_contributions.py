"""
Fermi-sea and Fermi-surface spin-to-orbital responses
=====================================================

Overview
--------
This script combines the two numerical contributions to the longitudinal
spin-to-orbital susceptibility

    chi_zz^{LS}
        = chi_zz^{LS,sea}
        + chi_zz^{LS,surf}

of a two-dimensional Rashba--altermagnetic continuum model.

The two contributions are intentionally kept as separate numerical modules
because they use different physical expressions and integration methods.

1. Fermi-sea contribution
   The sea term is evaluated at zero spin bias from the occupied states:

       chi_zz^{LS,sea}(mu)
         = sum_lambda integral d^2k/(2 pi)^2
           [partial m_z(k,b_z)/partial b_z]_(b_z=0)
           Theta[mu - epsilon_lambda(k)].

   It is computed directly on a two-dimensional momentum grid with a
   circular continuum cutoff.

2. Fermi-surface contribution
   The surface term is evaluated at finite spin bias from line integrals
   over the Fermi contours:

       chi_zz^{LS,surf}(mu)
         = sum_lambda integral_(epsilon_lambda=mu)
           dl/(2 pi)^2
           m_z(k,b_z) <sigma_z>_(lambda,b_z)
           / |grad_k epsilon_lambda(k,b_z)|.

   The Fermi contours are extracted numerically with Matplotlib and then
   integrated segment by segment.

Top-level response modes
------------------------
Set ``RESPONSE_COMPONENT`` to one of the following:

- ``"sea"``
  Run only the Fermi-sea module.

- ``"surface"``
  Run only the Fermi-surface module.

- ``"both"``
  Run the selected sea mode followed by the selected surface mode.

Independent module modes
------------------------
``SEA_MODE`` controls the Fermi-sea module. Available values are:

- ``"compute_sea_mu"``
- ``"plot_sea_mu_from_excel"``
- ``"compute_sea_eta_sweep"``
- ``"plot_sea_eta_from_excel"``
- ``"plot_parent_summary_from_excel"``
- ``"compute_all"``
- ``"plot_all_from_excel"``

``SURFACE_MODE`` controls the Fermi-surface module. Available values are:

- ``"compute_fixed_bz"``
- ``"compute_multi_bz"``
- ``"compute_eta_sweep"``
- ``"plot_fixed_bz_from_excel"``
- ``"plot_multi_bz_from_excel"``
- ``"plot_eta_sweep_from_excel"``
- ``"plot_parent_summary_from_excel"``
- ``"compute_and_plot_fixed_bz"``
- ``"compute_and_plot_multi_bz"``
- ``"compute_and_plot_eta_sweep"``
- ``"compute_all"``
- ``"plot_all_from_excel"``

Preserved configurations
------------------------
Each original script is encapsulated in its own wrapper function. This keeps
its physical parameters, momentum grid, filenames, plotting style, and data
formats independent and prevents name collisions between the two modules.

The original output folders are retained:

- Fermi sea: ``resultados_chi_sea``
- Fermi surface: ``resultados_chi_surf``

Outputs
-------
Depending on the selected modes, the modules may save:

- numerical data in Excel workbooks;
- CSV copies for post-processing;
- PNG and PDF figures;
- fixed-parameter, spin-bias, and eta-sweep results;
- parent summary figures assembled from saved data.

Heavy calculations and plotting-from-saved-data remain separate so that
figures can be regenerated without repeating the numerical integrations.
"""

# ============================================================
# TOP-LEVEL USER CONFIGURATION
# ============================================================

# Options:
#   "sea"
#   "surface"
#   "both"
RESPONSE_COMPONENT = "both"

# Select one of the documented Fermi-sea modes.
SEA_MODE = "plot_all_from_excel"

# Select one of the documented Fermi-surface modes.
SURFACE_MODE = "plot_all_from_excel"


# ============================================================
# FERMI-SEA MODULE
# ============================================================

def run_fermi_sea(selected_mode: str) -> None:
    """
    Run the selected Fermi-sea calculation or post-processing mode.

    The body below preserves the original fermi_sea.py implementation inside
    a local namespace so that its names do not conflict with the surface
    module.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from matplotlib.lines import Line2D

    # ============================================================
    # MODE
    #   "compute_sea_mu"                 -> calcula chi_sea(mu) y guarda datos
    #   "plot_sea_mu_from_excel"         -> grafica chi_sea(mu) desde Excel
    #   "compute_sea_eta_sweep"          -> calcula chi_sea(mu) para varios eta
    #   "plot_sea_eta_from_excel"        -> grafica sweep eta desde Excel
    #   "plot_parent_summary_from_excel" -> figura padre 1x3 desde Excel
    #   "compute_all"                    -> calcula todo
    #   "plot_all_from_excel"            -> grafica todo desde Excel
    # ============================================================
    mode = selected_mode
    # ============================================================
    # OUTPUT OPTIONS
    # ============================================================
    OUTPUT_DIR = Path("resultados_chi_sea")
    FIG_DIR = OUTPUT_DIR / "figuras"

    EXCEL_SEA_MU = "resultados_chi_sea_mu.xlsx"
    EXCEL_SEA_ETA = "resultados_chi_sea_eta_sweep.xlsx"

    SAVE_FIGURES = True
    SAVE_EXCEL = True
    SAVE_CSV = True
    SHOW_FIGURES = True
    FIG_DPI = 600

    # ============================================================
    # Physical parameters
    # ============================================================
    t0_fixed = 3.2895        # eV Å^2
    tj_fixed = 1.881         # eV Å^2

    alpha_fixed_meVA = 26.0
    alpha_fixed = alpha_fixed_meVA * 1e-3  # eV Å

    # eta = alpha / (t_j k_*)
    k_star_Ainv = 0.10
    eta_base = alpha_fixed / (tj_fixed * k_star_Ainv)

    # Puedes editar estos valores.
    # eta_base se incluye automáticamente para que una curva corresponda a alpha_fixed.
    eta_values = np.array(
        sorted(np.unique(np.round([0.05, 0.10, eta_base, 0.20, 0.35], 12))),
        dtype=float
    )

    mu_values = np.linspace(-0.02, 0.12, 120)  # eV

    # ============================================================
    # k-grid
    # ============================================================
    kmax = 0.35      # Å^{-1}
    N = 701

    kx = np.linspace(-kmax, kmax, N)
    ky = np.linspace(-kmax, kmax, N)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Circular cutoff for continuum model
    CUTOFF_MASK = K2 <= kmax**2

    # ============================================================
    # Unit conversion
    # [e/hbar] * [eV Å^2] -> mu_B
    # ============================================================
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    eV_to_J = 1.602176634e-19
    A2_to_m2 = 1e-20
    mu_B = 9.2740100783e-24

    conv_to_muB = (e / hbar) * eV_to_J * A2_to_m2 / mu_B

    # ============================================================
    # Output helpers
    # ============================================================
    def prepare_output_folders():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)


    def save_figure(fig, filename_stem):
        if not SAVE_FIGURES:
            return

        png_path = FIG_DIR / f"{filename_stem}.png"
        pdf_path = FIG_DIR / f"{filename_stem}.pdf"

        fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=FIG_DPI, bbox_inches="tight")

        print(f"Figura guardada: {png_path}")
        print(f"Figura guardada: {pdf_path}")


    def write_excel(sheets, filename):
        if not SAVE_EXCEL:
            return

        excel_path = OUTPUT_DIR / filename

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

        print(f"Excel guardado: {excel_path}")


    def write_csv_outputs(dfs, subfolder_name):
        if not SAVE_CSV:
            return

        csv_dir = OUTPUT_DIR / subfolder_name
        csv_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dfs.items():
            csv_path = csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"CSV guardado: {csv_path}")


    def parameters_dataframe(extra=None):
        params = {
            "t0_fixed_eV_A2": t0_fixed,
            "tj_fixed_eV_A2": tj_fixed,
            "alpha_fixed_meV_A": alpha_fixed_meVA,
            "alpha_fixed_eV_A": alpha_fixed,
            "k_star_A_inv": k_star_Ainv,
            "eta_base": eta_base,
            "eta_values": ", ".join([f"{v:.8g}" for v in eta_values]),
            "mu_min_eV": float(mu_values.min()),
            "mu_max_eV": float(mu_values.max()),
            "num_mu_values": len(mu_values),
            "kmax_A_inv": kmax,
            "N_grid": N,
            "cutoff": "circular k^2 <= kmax^2",
            "figure_dpi": FIG_DPI,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra is not None:
            params.update(extra)

        return pd.DataFrame(list(params.items()), columns=["parameter", "value"])


    def eta_sweep_dataframe():
        rows = []
        for eta in eta_values:
            alpha_eta = eta * tj_fixed * k_star_Ainv
            rows.append({
                "eta": eta,
                "alpha_eV_A": alpha_eta,
                "alpha_meV_A": 1e3 * alpha_eta,
                "tj_eV_A2": tj_fixed,
                "k_star_A_inv": k_star_Ainv,
            })
        return pd.DataFrame(rows)


    def print_eta_sweep_summary():
        df_eta = eta_sweep_dataframe()
        print("\nBarrido de eta usado:")
        print(df_eta.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        print("")


    # ============================================================
    # Model functions at b_z = 0
    # ============================================================
    def d0_func(KX, KY, alpha, tj):
        return np.sqrt(alpha**2 * (KX**2 + KY**2) + 4.0 * tj**2 * KX**2 * KY**2)


    def energy_lambda_bz0(KX, KY, alpha, tj, t0, lam):
        return t0 * (KX**2 + KY**2) + lam * d0_func(KX, KY, alpha, tj)


    def dmz_dbz_at_bz0_muB_per_eV(KX, KY, alpha, tj):
        """
        ∂m_z(k,bz)/∂b_z |_{bz=0}, in units of μB/eV.

        Formula:
        (e/hbar) * alpha^2 * [1/2 alpha^2 k^2 + 6 tj^2 kx^2 ky^2]
        / [alpha^2 k^2 + 4 tj^2 kx^2 ky^2]^2
        """
        k2 = KX**2 + KY**2
        kx2ky2 = KX**2 * KY**2

        denominator = alpha**2 * k2 + 4.0 * tj**2 * kx2ky2

        numerator = alpha**2 * (
            0.5 * alpha**2 * k2
            + 6.0 * tj**2 * kx2ky2
        )

        dm_eVA2_per_eV = np.divide(
            numerator,
            denominator**2,
            out=np.zeros_like(numerator),
            where=denominator > 1e-28
        )

        return conv_to_muB * dm_eVA2_per_eV


    def chi_sea_vs_mu(alpha, tj, label_info=None):
        """
        Computes chi_sea(mu) at T=0.

        chi_sea(mu) = sum_lambda ∫ d2k/(2π)^2
                      [∂m_z/∂b_z]_{bz=0} Θ(mu - epsilon_lambda(k)).
        """
        print(f"Calculando chi_sea para alpha={1e3*alpha:.3f} meVÅ, tj={tj:.3f} eVÅ²")

        Eplus = energy_lambda_bz0(KX, KY, alpha, tj, t0_fixed, lam=+1)
        Eminus = energy_lambda_bz0(KX, KY, alpha, tj, t0_fixed, lam=-1)

        dmdb = dmz_dbz_at_bz0_muB_per_eV(KX, KY, alpha, tj)

        # Apply cutoff once
        dmdb_cut = np.where(CUTOFF_MASK, dmdb, 0.0)

        prefactor = dkx * dky / (2.0 * np.pi)**2

        rows = []

        for i, mu in enumerate(mu_values, start=1):
            occ_plus = (Eplus < mu) & CUTOFF_MASK
            occ_minus = (Eminus < mu) & CUTOFF_MASK

            chi_plus = prefactor * np.nansum(np.where(occ_plus, dmdb_cut, 0.0))
            chi_minus = prefactor * np.nansum(np.where(occ_minus, dmdb_cut, 0.0))
            chi_total = chi_plus + chi_minus

            row = {
                "mu_eV": mu,
                "mu_meV": 1e3 * mu,
                "alpha_eV_A": alpha,
                "alpha_meV_A": 1e3 * alpha,
                "tj_eV_A2": tj,
                "chi_sea_total_muB_per_eV_A2": chi_total,
                "chi_sea_plus_muB_per_eV_A2": chi_plus,
                "chi_sea_minus_muB_per_eV_A2": chi_minus,
            }

            if label_info is not None:
                row.update(label_info)

            rows.append(row)

            if i % 20 == 0 or i == len(mu_values):
                print(f"  mu {i}/{len(mu_values)} terminado")

        return pd.DataFrame(rows)


    # ============================================================
    # BLOQUE 1: CÁLCULO PESADO
    # ============================================================
    def compute_sea_mu_data():
        df = chi_sea_vs_mu(alpha_fixed, tj_fixed)

        sheets = {
            "sea_mu": df,
            "parameters": parameters_dataframe(),
        }

        write_excel(sheets, EXCEL_SEA_MU)
        write_csv_outputs(
            {"sea_mu": df, "parameters": parameters_dataframe()},
            subfolder_name="csv_sea_mu"
        )

        return df


    def compute_sea_eta_sweep_data():
        print_eta_sweep_summary()

        all_dfs = []

        for eta in eta_values:
            alpha_eta = eta * tj_fixed * k_star_Ainv

            df_eta = chi_sea_vs_mu(
                alpha_eta,
                tj_fixed,
                label_info={
                    "eta": eta,
                    "k_star_A_inv": k_star_Ainv,
                }
            )

            all_dfs.append(df_eta)

        df_long = pd.concat(all_dfs, ignore_index=True)

        df_pivot = df_long.pivot(
            index="mu_eV",
            columns="eta",
            values="chi_sea_total_muB_per_eV_A2"
        ).reset_index()

        df_pivot.columns = ["mu_eV"] + [
            f"chi_sea_eta_{c:.6g}" for c in df_pivot.columns[1:]
        ]

        sheets = {
            "sea_eta_long": df_long,
            "sea_eta_pivot": df_pivot,
            "eta_sweep": eta_sweep_dataframe(),
            "parameters": parameters_dataframe(),
        }

        write_excel(sheets, EXCEL_SEA_ETA)

        write_csv_outputs(
            {
                "sea_eta_long": df_long,
                "sea_eta_pivot": df_pivot,
                "eta_sweep": eta_sweep_dataframe(),
                "parameters": parameters_dataframe(),
            },
            subfolder_name="csv_sea_eta"
        )

        return df_long


    # ============================================================
    # Plot style aprendido
    # ============================================================
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 18,
        "legend.fontsize": 11,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]


    def format_axis(ax):
        ax.axhline(0, color="0.85", lw=1.0, zorder=0)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)


    def boxed_legend(ax, loc="best", ncol=1):
        ax.legend(
            loc=loc,
            ncol=ncol,
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
            fontsize=11,
        )


    # ============================================================
    # BLOQUE 2: POST-PROCESAMIENTO
    # ============================================================
    def plot_sea_mu(df, filename_stem):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        ax1, ax2 = axes

        mu_meV = df["mu_meV"]

        ax1.plot(
            mu_meV,
            df["chi_sea_total_muB_per_eV_A2"],
            lw=2.6,
            color="tab:blue",
            label=r"total",
        )

        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )

        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes,
                 ha="left", va="top", fontsize=18)

        ax2.plot(
            mu_meV,
            df["chi_sea_plus_muB_per_eV_A2"],
            lw=2.6,
            color="tab:red",
            label=r"$\lambda=+$",
        )

        ax2.plot(
            mu_meV,
            df["chi_sea_minus_muB_per_eV_A2"],
            lw=2.6,
            color="tab:blue",
            label=r"$\lambda=-$",
        )

        ax2.plot(
            mu_meV,
            df["chi_sea_total_muB_per_eV_A2"],
            lw=2.0,
            color="k",
            ls="--",
            alpha=0.75,
            label=r"total",
        )

        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz,\lambda}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )

        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes,
                 ha="left", va="top", fontsize=18)

        for ax in axes:
            format_axis(ax)
            boxed_legend(ax)
            ax.set_xlim(mu_meV.min(), mu_meV.max())

        fig.suptitle(
            rf"Fermi-sea spin-to-orbital response"
            + "\n"
            + rf"fixed $\alpha={alpha_fixed_meVA:.0f}\,\mathrm{{meV\AA}}$, "
            + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$, "
            + rf"$k_\mathrm{{max}}={kmax:.2f}\,\mathrm{{\AA^{{-1}}}}$",
            y=1.03,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_sea_eta(df_long, filename_stem):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))

        grouped = list(df_long.groupby("eta"))

        for color, (eta, group) in zip(colors, grouped):
            ax.plot(
                group["mu_meV"],
                group["chi_sea_total_muB_per_eV_A2"],
                lw=2.6,
                color=color,
                label=rf"$\eta={eta:.3f}$",
            )

        format_axis(ax)
        boxed_legend(ax, loc="best")

        ax.set_xlim(df_long["mu_meV"].min(), df_long["mu_meV"].max())

        ax.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )

        fig.suptitle(
            rf"Fermi-sea response for different $\eta=\alpha/(t_j k_*)$"
            + "\n"
            + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$, "
            + rf"$k_*={k_star_Ainv:.2f}\,\mathrm{{\AA^{{-1}}}}$, "
            + rf"$k_\mathrm{{max}}={kmax:.2f}\,\mathrm{{\AA^{{-1}}}}$",
            y=1.03,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_parent_summary(df_mu, df_eta, filename_stem):
        """
        Figura padre 1x3:
        (a) chi_sea total para alpha fijo
        (b) chi_sea resuelta por bandas para alpha fijo
        (c) chi_sea total para sweep de eta
        """
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))
        ax1, ax2, ax3 = axes

        mu_meV = df_mu["mu_meV"]

        # Panel (a): total fixed alpha
        ax1.plot(
            mu_meV,
            df_mu["chi_sea_total_muB_per_eV_A2"],
            lw=2.6,
            color="tab:blue",
            label=r"total",
        )

        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )
        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes,
                 ha="left", va="top", fontsize=18)
        boxed_legend(ax1, loc="best")

        # Panel (b): band-resolved fixed alpha
        ax2.plot(
            mu_meV,
            df_mu["chi_sea_plus_muB_per_eV_A2"],
            lw=2.6,
            color="tab:red",
            label=r"$\lambda=+$",
        )

        ax2.plot(
            mu_meV,
            df_mu["chi_sea_minus_muB_per_eV_A2"],
            lw=2.6,
            color="tab:blue",
            label=r"$\lambda=-$",
        )

        ax2.plot(
            mu_meV,
            df_mu["chi_sea_total_muB_per_eV_A2"],
            lw=2.0,
            color="k",
            ls="--",
            alpha=0.75,
            label=r"total",
        )

        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz,\lambda}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )
        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes,
                 ha="left", va="top", fontsize=18)
        boxed_legend(ax2, loc="best")

        # Panel (c): eta sweep
        grouped = list(df_eta.groupby("eta"))

        for color, (eta, group) in zip(colors, grouped):
            ax3.plot(
                group["mu_meV"],
                group["chi_sea_total_muB_per_eV_A2"],
                lw=2.6,
                color=color,
                label=rf"$\eta={eta:.3f}$",
            )

        ax3.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax3.set_ylabel(
            r"$\chi^{LS,\mathrm{sea}}_{zz}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        )
        ax3.text(0.02, 0.96, r"(c)", transform=ax3.transAxes,
                 ha="left", va="top", fontsize=18)
        boxed_legend(ax3, loc="best")

        # Common format
        for ax in axes:
            format_axis(ax)
            ax.set_xlim(mu_meV.min(), mu_meV.max())

        fig.suptitle(
            rf"Fermi-sea spin-to-orbital response"
            + "\n"
            + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$, "
            + rf"$k_*={k_star_Ainv:.2f}\,\mathrm{{\AA^{{-1}}}}$, "
            + rf"$k_\mathrm{{max}}={kmax:.2f}\,\mathrm{{\AA^{{-1}}}}$",
            y=1.04,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_sea_mu_from_excel():
        excel_path = OUTPUT_DIR / EXCEL_SEA_MU

        if not excel_path.exists():
            raise FileNotFoundError(
                f"No encontré {excel_path}. Primero corre mode='compute_sea_mu'."
            )

        df = pd.read_excel(excel_path, sheet_name="sea_mu")
        plot_sea_mu(df, "chi_sea_mu_from_saved_data")


    def plot_sea_eta_from_excel():
        excel_path = OUTPUT_DIR / EXCEL_SEA_ETA

        if not excel_path.exists():
            raise FileNotFoundError(
                f"No encontré {excel_path}. Primero corre mode='compute_sea_eta_sweep'."
            )

        df = pd.read_excel(excel_path, sheet_name="sea_eta_long")
        plot_sea_eta(df, "chi_sea_eta_sweep_from_saved_data")


    def plot_parent_summary_from_excel():
        excel_mu_path = OUTPUT_DIR / EXCEL_SEA_MU
        excel_eta_path = OUTPUT_DIR / EXCEL_SEA_ETA

        if not excel_mu_path.exists():
            raise FileNotFoundError(
                f"No encontré {excel_mu_path}. Primero corre mode='compute_sea_mu' o mode='compute_all'."
            )

        if not excel_eta_path.exists():
            raise FileNotFoundError(
                f"No encontré {excel_eta_path}. Primero corre mode='compute_sea_eta_sweep' o mode='compute_all'."
            )

        df_mu = pd.read_excel(excel_mu_path, sheet_name="sea_mu")
        df_eta = pd.read_excel(excel_eta_path, sheet_name="sea_eta_long")

        plot_parent_summary(
            df_mu,
            df_eta,
            filename_stem="chi_sea_parent_summary_from_saved_data"
        )


    # ============================================================
    # Composite modes
    # ============================================================
    def compute_all():
        compute_sea_mu_data()
        compute_sea_eta_sweep_data()


    def plot_all_from_excel():
        plot_sea_mu_from_excel()
        plot_sea_eta_from_excel()
        plot_parent_summary_from_excel()


    # ============================================================
    # Run
    # ============================================================
    def main():
        prepare_output_folders()

        if mode == "compute_sea_mu":
            compute_sea_mu_data()

        elif mode == "plot_sea_mu_from_excel":
            plot_sea_mu_from_excel()

        elif mode == "compute_sea_eta_sweep":
            compute_sea_eta_sweep_data()

        elif mode == "plot_sea_eta_from_excel":
            plot_sea_eta_from_excel()

        elif mode == "plot_parent_summary_from_excel":
            plot_parent_summary_from_excel()

        elif mode == "compute_all":
            compute_all()

        elif mode == "plot_all_from_excel":
            plot_all_from_excel()

        else:
            raise ValueError(
                "mode debe ser: "
                "'compute_sea_mu', "
                "'plot_sea_mu_from_excel', "
                "'compute_sea_eta_sweep', "
                "'plot_sea_eta_from_excel', "
                "'plot_parent_summary_from_excel', "
                "'compute_all', "
                "'plot_all_from_excel'."
            )

        print("\nListo. Revisa la carpeta:", OUTPUT_DIR.resolve())

    main()


# ============================================================
# FERMI-SURFACE MODULE
# ============================================================

def run_fermi_surface(selected_mode: str) -> None:
    """
    Run the selected Fermi-surface calculation or post-processing mode.

    The body below preserves the original response.py implementation inside
    a local namespace so that its names do not conflict with the sea module.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    from pathlib import Path
    from datetime import datetime
    from matplotlib.lines import Line2D

    # ============================================================
    # MODE
    #   "compute_fixed_bz"             -> calcula y guarda datos para un bz fijo
    #   "compute_multi_bz"             -> calcula y guarda datos para varios bz
    #   "compute_eta_sweep"            -> calcula y guarda datos para varios eta
    #
    #   "plot_fixed_bz_from_excel"     -> lee Excel y grafica bz fijo
    #   "plot_multi_bz_from_excel"     -> lee Excel y grafica varios bz
    #   "plot_eta_sweep_from_excel"    -> lee Excel y grafica varios eta
    #   "plot_parent_summary_from_excel" -> figura padre con fixed bz, multi bz y eta
    #
    #   "compute_and_plot_fixed_bz"    -> calcula, guarda y grafica bz fijo
    #   "compute_and_plot_multi_bz"    -> calcula, guarda y grafica varios bz
    #   "compute_and_plot_eta_sweep"   -> calcula, guarda y grafica varios eta
    #   "compute_all"                  -> calcula fixed bz, multi bz y eta
    #   "plot_all_from_excel"          -> grafica fixed bz, multi bz, eta y figura padre
    # ============================================================
    mode = selected_mode
    # ============================================================
    # OUTPUT OPTIONS
    # ============================================================
    OUTPUT_DIR = Path("resultados_chi_surf")
    FIG_DIR = OUTPUT_DIR / "figuras"

    EXCEL_NAME_MULTI_BZ = "resultados_numericos_chi_surf_multi_bz.xlsx"
    EXCEL_NAME_FIXED_BZ = "resultados_numericos_chi_surf_fixed_bz.xlsx"
    EXCEL_NAME_ETA = "resultados_numericos_chi_surf_eta_sweep.xlsx"

    SAVE_FIGURES = True
    SAVE_EXCEL = True
    SAVE_CSV = True
    SHOW_FIGURES = True
    FIG_DPI = 600

    # If True, the plotting functions can read old files from resultados_chi_occ
    # when the new surface files are not found.
    ALLOW_LEGACY_OCC_FALLBACK = True
    LEGACY_OUTPUT_DIR = Path("resultados_chi_occ")
    LEGACY_EXCEL_MULTI_BZ = "resultados_numericos_chi_occ.xlsx"
    LEGACY_EXCEL_FIXED_BZ = "fixed_bz_resultados_chi_occ.xlsx"

    # ============================================================
    # Physical parameters
    # ============================================================
    t0_fixed = 3.2895          # eV Å^2
    tj_fixed = 1.881           # eV Å^2

    alpha_fixed_meVA = 26.0    # meV Å
    alpha_fixed = alpha_fixed_meVA * 1e-3  # eV Å

    bz_fixed_meV = 5.0
    bz_fixed = bz_fixed_meV * 1e-3  # eV

    bz_values_meV = np.array([-10, -5, 0, 5, 10, 15, 20, 25], dtype=float)
    bz_values = bz_values_meV * 1e-3  # eV

    mu_values = np.linspace(-0.02, 0.12, 90)  # eV

    # ============================================================
    # Dimensionless eta sweep
    # eta = alpha/(t_j k_star)
    # In this mode, t_j and k_star are fixed and alpha is reconstructed as:
    # alpha = eta * t_j * k_star.
    # ============================================================
    k_star_Ainv = 0.10
    eta_base = alpha_fixed / (tj_fixed * k_star_Ainv)
    eta_values = np.array(sorted([0.05, 0.10, eta_base, 0.20, 0.35]), dtype=float)

    bz_eta_meV = bz_fixed_meV
    bz_eta = bz_eta_meV * 1e-3

    # ============================================================
    # k-grid for contour extraction
    # ============================================================
    kmax = 0.35      # Å^{-1}
    N = 701

    kx = np.linspace(-kmax, kmax, N)
    ky = np.linspace(-kmax, kmax, N)
    KX, KY = np.meshgrid(kx, ky)

    # ============================================================
    # Unit conversion
    # [e/hbar] * [eV Å^2] -> mu_B
    # ============================================================
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    eV_to_J = 1.602176634e-19
    A2_to_m2 = 1e-20
    mu_B = 9.2740100783e-24

    conv_to_muB = (e / hbar) * eV_to_J * A2_to_m2 / mu_B

    # ============================================================
    # Output helpers
    # ============================================================
    def prepare_output_folders():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)


    def save_figure(fig, filename_stem):
        """Guarda la figura en PNG y PDF con dpi = FIG_DPI."""
        if not SAVE_FIGURES:
            return

        png_path = FIG_DIR / f"{filename_stem}.png"
        pdf_path = FIG_DIR / f"{filename_stem}.pdf"

        fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=FIG_DPI, bbox_inches="tight")

        print(f"Figura guardada: {png_path}")
        print(f"Figura guardada: {pdf_path}")


    def parameters_dataframe():
        """Tabla de parámetros para guardar en el Excel."""
        params = {
            "t0_fixed_eV_A2": t0_fixed,
            "tj_fixed_eV_A2": tj_fixed,
            "alpha_fixed_meV_A": alpha_fixed_meVA,
            "alpha_fixed_eV_A": alpha_fixed,
            "bz_fixed_meV": bz_fixed_meV,
            "bz_fixed_eV": bz_fixed,
            "bz_eta_meV": bz_eta_meV,
            "bz_eta_eV": bz_eta,
            "k_star_A_inv": k_star_Ainv,
            "eta_base": eta_base,
            "eta_values": ", ".join([f"{v:.6g}" for v in eta_values]),
            "mu_min_eV": float(mu_values.min()),
            "mu_max_eV": float(mu_values.max()),
            "num_mu_values": len(mu_values),
            "kmax_A_inv": kmax,
            "N_grid": N,
            "figure_dpi": FIG_DPI,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return pd.DataFrame(list(params.items()), columns=["parameter", "value"])


    def write_excel(sheets, filename):
        """Guarda varias hojas en un solo archivo Excel."""
        if not SAVE_EXCEL:
            return

        excel_path = OUTPUT_DIR / filename

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                safe_name = sheet_name[:31]  # Excel limita nombres a 31 caracteres
                df.to_excel(writer, sheet_name=safe_name, index=False)

        print(f"Excel guardado: {excel_path}")


    def write_csv_outputs(dfs, subfolder_name="csv_data"):
        """
        Guarda cada DataFrame como CSV para post-procesamiento rápido.
        """
        if not SAVE_CSV:
            return

        csv_dir = OUTPUT_DIR / subfolder_name
        csv_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dfs.items():
            csv_path = csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"CSV guardado: {csv_path}")


    def resolve_excel_path(primary_path, legacy_path=None, message_if_missing=""):
        """
        Busca primero el archivo nuevo. Si no existe, opcionalmente busca el archivo legacy.
        """
        if primary_path.exists():
            return primary_path

        if ALLOW_LEGACY_OCC_FALLBACK and legacy_path is not None and legacy_path.exists():
            print(f"Usando archivo legacy: {legacy_path}")
            return legacy_path

        raise FileNotFoundError(message_if_missing or f"No encontré {primary_path}.")


    # ============================================================
    # Model functions with spin bias
    # ============================================================
    def dz_func(kx, ky, tj, bz):
        return 2.0 * tj * kx * ky - bz


    def D_func(kx, ky, alpha, tj, bz):
        """
        D_b = d_b^2 = alpha^2 k^2 + (2 tj kx ky - bz)^2
        """
        dz = dz_func(kx, ky, tj, bz)
        return alpha**2 * (kx**2 + ky**2) + dz**2


    def d_func(kx, ky, alpha, tj, bz):
        return np.sqrt(D_func(kx, ky, alpha, tj, bz))


    def energy_lambda(kx, ky, alpha, tj, t0, lam, bz):
        """
        epsilon_lambda(k,bz) = t0 k^2 + lambda d_b(k)
        """
        k2 = kx**2 + ky**2
        d = d_func(kx, ky, alpha, tj, bz)
        return t0 * k2 + lam * d


    def grad_energy_norm(kx, ky, alpha, tj, t0, lam, bz):
        """
        |grad_k epsilon_lambda(k,bz)|
        Units: eV Å
        """
        dz = dz_func(kx, ky, tj, bz)
        d = d_func(kx, ky, alpha, tj, bz)
        d = np.where(d < 1e-18, np.nan, d)

        dEdkx = 2.0 * t0 * kx + lam * (alpha**2 * kx + 2.0 * tj * ky * dz) / d
        dEdky = 2.0 * t0 * ky + lam * (alpha**2 * ky + 2.0 * tj * kx * dz) / d

        return np.sqrt(dEdkx**2 + dEdky**2)


    def orbital_moment_muB(kx, ky, alpha, tj, bz):
        """
        m_z(k,bz) in mu_B.

        m_z(k,bz) =
          (e/hbar) * alpha^2 (tj kx ky + bz/2)
          / [alpha^2 k^2 + (2 tj kx ky - bz)^2]
        """
        D = D_func(kx, ky, alpha, tj, bz)
        D = np.where(D < 1e-24, np.nan, D)

        m_reduced = alpha**2 * (tj * kx * ky + 0.5 * bz) / D  # eV Å^2
        return conv_to_muB * m_reduced


    def spin_z(kx, ky, alpha, tj, lam, bz):
        """
        <sigma_z>_lambda,bz = lambda dz/d
        """
        dz = dz_func(kx, ky, tj, bz)
        d = d_func(kx, ky, alpha, tj, bz)
        d = np.where(d < 1e-18, np.nan, d)

        return lam * dz / d


    def chi_integrand_muB(kx, ky, alpha, tj, lam, bz):
        """
        Surface contribution integrand numerator:

            m_z(k,bz) * <sigma_z>_lambda,bz

        Units: mu_B
        """
        mz = orbital_moment_muB(kx, ky, alpha, tj, bz)
        sz = spin_z(kx, ky, alpha, tj, lam, bz)

        return mz * sz


    # ============================================================
    # Contour extraction and contour integration
    # ============================================================
    def get_contour_segments(Egrid, level):
        """
        Extract contour segments E(kx,ky)=level using matplotlib,
        without displaying the temporary figure.
        """
        if level < np.nanmin(Egrid) or level > np.nanmax(Egrid):
            return []

        fig_tmp, ax_tmp = plt.subplots()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = ax_tmp.contour(KX, KY, Egrid, levels=[level])

        segments = cs.allsegs[0]
        plt.close(fig_tmp)

        return segments


    def integrate_one_band(mu, alpha, tj, t0, lam, bz, Egrid=None):
        """
        Computes the Fermi-surface contribution for one band lambda.
        """
        if Egrid is None:
            Egrid = energy_lambda(KX, KY, alpha, tj, t0, lam, bz)

        segments = get_contour_segments(Egrid, mu)
        total = 0.0

        for verts in segments:
            if len(verts) < 2:
                continue

            diffs = np.diff(verts, axis=0)
            dl = np.sqrt((diffs**2).sum(axis=1))

            mids = 0.5 * (verts[:-1] + verts[1:])
            mx = mids[:, 0]
            my = mids[:, 1]

            F = chi_integrand_muB(mx, my, alpha, tj, lam, bz)
            gradE = grad_energy_norm(mx, my, alpha, tj, t0, lam, bz)

            integrand = F / gradE
            total += np.nansum(dl * integrand)

        return total / (2.0 * np.pi)**2


    def chi_surf(mu, alpha, tj, t0, bz, E_plus=None, E_minus=None):
        """
        Total finite-bz Fermi-surface contribution.
        Units: mu_B / (eV Å^2)
        """
        chi_p = integrate_one_band(mu, alpha, tj, t0, lam=+1, bz=bz, Egrid=E_plus)
        chi_m = integrate_one_band(mu, alpha, tj, t0, lam=-1, bz=bz, Egrid=E_minus)
        return chi_p + chi_m, chi_p, chi_m


    # ============================================================
    # Computation helpers
    # ============================================================
    def compute_chi_vs_mu_for_params(label_dict, alpha, tj, t0, bz):
        """Calcula total, lambda=+ y lambda=- para parámetros dados."""
        label_txt = ", ".join([f"{k}={v}" for k, v in label_dict.items()])
        print(f"Calculando {label_txt} ...")

        E_plus = energy_lambda(KX, KY, alpha, tj, t0, lam=+1, bz=bz)
        E_minus = energy_lambda(KX, KY, alpha, tj, t0, lam=-1, bz=bz)

        rows = []
        for i, mu in enumerate(mu_values, start=1):
            total, plus, minus = chi_surf(
                mu,
                alpha,
                tj,
                t0,
                bz,
                E_plus=E_plus,
                E_minus=E_minus,
            )

            row = {
                "mu_eV": mu,
                "mu_meV": 1e3 * mu,
                "alpha_eV_A": alpha,
                "alpha_meV_A": 1e3 * alpha,
                "tj_eV_A2": tj,
                "bz_eV": bz,
                "bz_meV": 1e3 * bz,
                "chi_total_muB_per_eV_A2": total,
                "chi_plus_muB_per_eV_A2": plus,
                "chi_minus_muB_per_eV_A2": minus,
            }
            row.update(label_dict)
            rows.append(row)

            if i % 10 == 0 or i == len(mu_values):
                print(f"  mu {i}/{len(mu_values)} terminado")

        return pd.DataFrame(rows)


    def compute_chi_vs_mu_for_bz(bz_meV, bz_eV):
        """Calcula total, lambda=+ y lambda=- para un bz fijo."""
        return compute_chi_vs_mu_for_params(
            label_dict={"sweep_mode": "bz", "sweep_value": bz_meV},
            alpha=alpha_fixed,
            tj=tj_fixed,
            t0=t0_fixed,
            bz=bz_eV,
        )


    def compute_chi_vs_mu_for_eta(eta_value):
        """Calcula chi(mu) para un eta fijo, variando alpha = eta*tj*k_star."""
        alpha_eta = eta_value * tj_fixed * k_star_Ainv
        return compute_chi_vs_mu_for_params(
            label_dict={
                "sweep_mode": "eta",
                "sweep_value": eta_value,
                "eta": eta_value,
                "k_star_A_inv": k_star_Ainv,
            },
            alpha=alpha_eta,
            tj=tj_fixed,
            t0=t0_fixed,
            bz=bz_eta,
        )


    def make_pivots(df_long, column_name="bz_meV", prefix="bz"):
        df_total_pivot = df_long.pivot(
            index="mu_eV",
            columns=column_name,
            values="chi_total_muB_per_eV_A2",
        ).reset_index()
        df_total_pivot.columns = ["mu_eV"] + [
            f"chi_total_{prefix}_{c:g}" for c in df_total_pivot.columns[1:]
        ]

        df_plus_pivot = df_long.pivot(
            index="mu_eV",
            columns=column_name,
            values="chi_plus_muB_per_eV_A2",
        ).reset_index()
        df_plus_pivot.columns = ["mu_eV"] + [
            f"chi_plus_{prefix}_{c:g}" for c in df_plus_pivot.columns[1:]
        ]

        df_minus_pivot = df_long.pivot(
            index="mu_eV",
            columns=column_name,
            values="chi_minus_muB_per_eV_A2",
        ).reset_index()
        df_minus_pivot.columns = ["mu_eV"] + [
            f"chi_minus_{prefix}_{c:g}" for c in df_minus_pivot.columns[1:]
        ]

        return df_total_pivot, df_plus_pivot, df_minus_pivot


    # ============================================================
    # Plot style aprendido
    # ============================================================
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 18,
        "legend.fontsize": 11,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]


    def format_learned_axis(ax):
        """
        Aplica el formato visual aprendido:
        ticks inward, minor ticks, cero gris claro, leyenda boxeada.
        """
        ax.axhline(0, color="0.85", lw=1.0, zorder=0)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)


    def learned_legend(ax, loc="best", ncol=1, fontsize=11):
        ax.legend(
            loc=loc,
            ncol=ncol,
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
            fontsize=fontsize,
        )


    def surface_ylabel(total=True):
        if total:
            return r"$\chi^{LS,\mathrm{surf}}_{zz}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"
        return r"$\chi^{LS,\mathrm{surf}}_{zz,\lambda}\;[\mu_B/(\mathrm{eV\,\AA^2})]$"


    # ============================================================
    # Plot functions con formato aprendido
    # ============================================================
    def plot_fixed_bz(df, filename_stem):
        """
        Figura 1x2:
        (a) contribución total
        (b) contribuciones por banda lambda = +/-.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        ax1, ax2 = axes

        mu_meV = 1e3 * df["mu_eV"]

        ax1.plot(mu_meV, df["chi_total_muB_per_eV_A2"], lw=2.6, color="tab:blue", label=r"total")
        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(surface_ylabel(total=True))
        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=18)

        ax2.plot(mu_meV, df["chi_plus_muB_per_eV_A2"], lw=2.6, color="tab:red", label=r"$\lambda=+$")
        ax2.plot(mu_meV, df["chi_minus_muB_per_eV_A2"], lw=2.6, color="tab:blue", label=r"$\lambda=-$")
        ax2.plot(mu_meV, df["chi_total_muB_per_eV_A2"], lw=2.0, color="k", ls="--", alpha=0.75, label=r"total")
        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(surface_ylabel(total=False))
        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=18)

        for ax in axes:
            format_learned_axis(ax)
            learned_legend(ax, loc="best")
            ax.set_xlim(mu_meV.min(), mu_meV.max())

        bz_plot = df["bz_meV"].iloc[0]

        #fig.suptitle(
        #    rf"Surface spin-to-orbital response at fixed $b_z={bz_plot:.1f}\,\mathrm{{meV}}$"
        #    + "\n"
        #    + rf"fixed $\alpha={alpha_fixed_meVA:.0f}\,\mathrm{{meV\AA}}$, "
        #    + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$",
        #    y=1.03,
        #    fontsize=15,
        #)

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_multi_bz(df_long, filename_stem):
        """
        Figura 1x2:
        (a) respuesta total para distintos b_z
        (b) contribuciones lambda=+ y lambda=-, usando el mismo color
            para cada b_z y estilos de línea diferentes.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        ax1, ax2 = axes

        grouped = list(df_long.groupby("bz_meV"))

        for color, (bz_meV, group) in zip(colors, grouped):
            mu_meV = 1e3 * group["mu_eV"]
            ax1.plot(
                mu_meV,
                group["chi_total_muB_per_eV_A2"],
                lw=2.6,
                color=color,
                label=rf"$b_z={bz_meV:.0f}\,\mathrm{{meV}}$",
            )

        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(surface_ylabel(total=True))
        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=18)

        for color, (bz_meV, group) in zip(colors, grouped):
            mu_meV = 1e3 * group["mu_eV"]
            ax2.plot(mu_meV, group["chi_plus_muB_per_eV_A2"], lw=2.2, color=color, ls="-")
            ax2.plot(mu_meV, group["chi_minus_muB_per_eV_A2"], lw=2.2, color=color, ls="--")

        style_legend = [
            Line2D([0], [0], color="k", lw=2.2, ls="-", label=r"$\lambda=+$"),
            Line2D([0], [0], color="k", lw=2.2, ls="--", label=r"$\lambda=-$"),
        ]

        leg1 = ax2.legend(handles=style_legend, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7", fontsize=11)
        ax2.add_artist(leg1)

        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(surface_ylabel(total=False))
        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=18)

        for ax in axes:
            format_learned_axis(ax)
            ax.set_xlim(1e3 * df_long["mu_eV"].min(), 1e3 * df_long["mu_eV"].max())

        learned_legend(ax1, loc="best", ncol=1)

        fig.suptitle(
            rf"Surface spin-to-orbital response for different $b_z$"
            + "\n"
            + rf"fixed $\alpha={alpha_fixed_meVA:.0f}\,\mathrm{{meV\AA}}$, "
            + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$",
            y=1.03,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_eta_sweep(df_eta, filename_stem):
        """
        Figura 1x2:
        (a) respuesta total para distintos eta
        (b) contribuciones lambda=+ y lambda=- para distintos eta.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        ax1, ax2 = axes

        grouped = list(df_eta.groupby("eta"))

        for color, (eta_val, group) in zip(colors, grouped):
            mu_meV = 1e3 * group["mu_eV"]
            ax1.plot(
                mu_meV,
                group["chi_total_muB_per_eV_A2"],
                lw=2.6,
                color=color,
                label=rf"$\eta={eta_val:.3f}$",
            )

        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(surface_ylabel(total=True))
        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=18)

        for color, (eta_val, group) in zip(colors, grouped):
            mu_meV = 1e3 * group["mu_eV"]
            ax2.plot(mu_meV, group["chi_plus_muB_per_eV_A2"], lw=2.2, color=color, ls="-")
            ax2.plot(mu_meV, group["chi_minus_muB_per_eV_A2"], lw=2.2, color=color, ls="--")

        style_legend = [
            Line2D([0], [0], color="k", lw=2.2, ls="-", label=r"$\lambda=+$"),
            Line2D([0], [0], color="k", lw=2.2, ls="--", label=r"$\lambda=-$"),
        ]

        leg1 = ax2.legend(handles=style_legend, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7", fontsize=11)
        ax2.add_artist(leg1)

        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(surface_ylabel(total=False))
        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=18)

        for ax in axes:
            format_learned_axis(ax)
            ax.set_xlim(1e3 * df_eta["mu_eV"].min(), 1e3 * df_eta["mu_eV"].max())

        learned_legend(ax1, loc="best", ncol=1)

        fig.suptitle(
            rf"Surface spin-to-orbital response for different $\eta=\alpha/(t_j k_*)$"
            + "\n"
            + rf"fixed $b_z={bz_eta_meV:.1f}\,\mathrm{{meV}}$, "
            + rf"fixed $t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$, "
            + rf"$k_*={k_star_Ainv:.2f}\,\mathrm{{\AA^{{-1}}}}$",
            y=1.03,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    def plot_parent_summary(df_fixed, df_multi, df_eta, filename_stem):
        """
        Figura padre 1x3:
        (a) fixed bz: contribuciones por banda y total
        (b) multi bz: total para distintos bz
        (c) eta sweep: total para distintos eta
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 20),sharex=True)
        ax1, ax2, ax3 = axes

        # --------------------------------------------------------
        # (a) fixed bz
        # --------------------------------------------------------
        mu_fixed_meV = 1e3 * df_fixed["mu_eV"]
        ax1.plot(mu_fixed_meV, df_fixed["chi_plus_muB_per_eV_A2"], lw=2.4, color="tab:red", label=r"$\lambda=+$")
        ax1.plot(mu_fixed_meV, df_fixed["chi_minus_muB_per_eV_A2"], lw=2.4, color="tab:blue", label=r"$\lambda=-$")
        ax1.plot(mu_fixed_meV, df_fixed["chi_total_muB_per_eV_A2"], lw=2.0, color="k", ls="--", alpha=0.75, label=r"total")

        ax1.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax1.set_ylabel(surface_ylabel(total=True))
        ax1.text(0.02, 0.96, r"(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=18)
        learned_legend(ax1, loc="best")

        # --------------------------------------------------------
        # (b) multi bz
        # --------------------------------------------------------
        for color, (bz_meV, group) in zip(colors, df_multi.groupby("bz_meV")):
            mu_meV = 1e3 * group["mu_eV"]
            ax2.plot(
                mu_meV,
                group["chi_total_muB_per_eV_A2"],
                lw=2.3,
                color=color,
                label=rf"$b_z={bz_meV:.0f}\,\mathrm{{meV}}$",
            )

        ax2.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax2.set_ylabel(surface_ylabel(total=True))
        ax2.text(0.02, 0.96, r"(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=18)
        learned_legend(ax2, loc="best", fontsize=9)

        # --------------------------------------------------------
        # (c) eta sweep
        # --------------------------------------------------------
        for color, (eta_val, group) in zip(colors, df_eta.groupby("eta")):
            mu_meV = 1e3 * group["mu_eV"]
            ax3.plot(
                mu_meV,
                group["chi_total_muB_per_eV_A2"],
                lw=2.3,
                color=color,
                label=rf"$\eta={eta_val:.3f}$",
            )

        ax3.set_xlabel(r"$\mu\,(\mathrm{meV})$")
        ax3.set_ylabel(surface_ylabel(total=True))
        ax3.text(0.02, 0.96, r"(c)", transform=ax3.transAxes, ha="left", va="top", fontsize=18)
        learned_legend(ax3, loc="best", fontsize=9)

        # --------------------------------------------------------
        # Common format
        # --------------------------------------------------------
        for ax in axes:
            format_learned_axis(ax)
            ax.set_xlim(1e3 * mu_values.min(), 1e3 * mu_values.max())

        fig.suptitle(
            rf"Surface spin-to-orbital response $\chi^{{LS,\mathrm{{surf}}}}_{{zz}}$"
            + "\n"
            + rf"$\alpha_0={alpha_fixed_meVA:.0f}\,\mathrm{{meV\AA}}$, "
            + rf"$t_j={tj_fixed:.3f}\,\mathrm{{eV\AA^2}}$, "
            + rf"$k_*={k_star_Ainv:.2f}\,\mathrm{{\AA^{{-1}}}}$",
            y=1.04,
            fontsize=15,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


    # ============================================================
    # BLOQUE 1: CÁLCULO PESADO Y GUARDADO DE DATOS
    # ============================================================
    def compute_fixed_bz_data():
        """
        Calcula chi(mu) para un bz fijo y guarda los datos numéricos.
        No grafica.
        """
        df_fixed = compute_chi_vs_mu_for_bz(bz_fixed_meV, bz_fixed)

        sheets = {
            "fixed_bz_results": df_fixed,
            "parameters": parameters_dataframe(),
        }

        write_excel(sheets, filename=EXCEL_NAME_FIXED_BZ)

        write_csv_outputs(
            {
                "fixed_bz_results": df_fixed,
                "parameters": parameters_dataframe(),
            },
            subfolder_name="csv_fixed_bz",
        )

        return df_fixed


    def compute_multi_bz_data():
        """
        Calcula chi(mu) para varios valores de bz y guarda los datos numéricos.
        No grafica.
        """
        all_dfs = []

        for bz_meV, bz in zip(bz_values_meV, bz_values):
            df_bz = compute_chi_vs_mu_for_bz(bz_meV, bz)
            all_dfs.append(df_bz)

        df_long = pd.concat(all_dfs, ignore_index=True)

        df_total_pivot, df_plus_pivot, df_minus_pivot = make_pivots(df_long, column_name="bz_meV", prefix="bz_meV")

        sheets = {
            "multi_bz_long": df_long,
            "total_pivot": df_total_pivot,
            "plus_pivot": df_plus_pivot,
            "minus_pivot": df_minus_pivot,
            "parameters": parameters_dataframe(),
        }

        write_excel(sheets, filename=EXCEL_NAME_MULTI_BZ)

        write_csv_outputs(
            {
                "multi_bz_long": df_long,
                "total_pivot": df_total_pivot,
                "plus_pivot": df_plus_pivot,
                "minus_pivot": df_minus_pivot,
                "parameters": parameters_dataframe(),
            },
            subfolder_name="csv_multi_bz",
        )

        return df_long


    def compute_eta_sweep_data():
        """
        Calcula chi(mu) para varios valores de eta y guarda los datos numéricos.
        No grafica.
        """
        all_dfs = []

        for eta_val in eta_values:
            df_eta = compute_chi_vs_mu_for_eta(eta_val)
            all_dfs.append(df_eta)

        df_long = pd.concat(all_dfs, ignore_index=True)

        df_total_pivot, df_plus_pivot, df_minus_pivot = make_pivots(df_long, column_name="eta", prefix="eta")

        sheets = {
            "eta_long": df_long,
            "total_pivot": df_total_pivot,
            "plus_pivot": df_plus_pivot,
            "minus_pivot": df_minus_pivot,
            "parameters": parameters_dataframe(),
        }

        write_excel(sheets, filename=EXCEL_NAME_ETA)

        write_csv_outputs(
            {
                "eta_long": df_long,
                "total_pivot": df_total_pivot,
                "plus_pivot": df_plus_pivot,
                "minus_pivot": df_minus_pivot,
                "parameters": parameters_dataframe(),
            },
            subfolder_name="csv_eta_sweep",
        )

        return df_long


    # ============================================================
    # BLOQUE 2: POST-PROCESAMIENTO Y GRÁFICAS DESDE DATOS GUARDADOS
    # ============================================================
    def read_fixed_bz_data():
        excel_path = resolve_excel_path(
            OUTPUT_DIR / EXCEL_NAME_FIXED_BZ,
            LEGACY_OUTPUT_DIR / LEGACY_EXCEL_FIXED_BZ,
            message_if_missing=(
                f"No encontré {OUTPUT_DIR / EXCEL_NAME_FIXED_BZ}. "
                "Primero corre mode='compute_fixed_bz'."
            ),
        )
        return pd.read_excel(excel_path, sheet_name="fixed_bz_results")


    def read_multi_bz_data():
        excel_path = resolve_excel_path(
            OUTPUT_DIR / EXCEL_NAME_MULTI_BZ,
            LEGACY_OUTPUT_DIR / LEGACY_EXCEL_MULTI_BZ,
            message_if_missing=(
                f"No encontré {OUTPUT_DIR / EXCEL_NAME_MULTI_BZ}. "
                "Primero corre mode='compute_multi_bz'."
            ),
        )
        return pd.read_excel(excel_path, sheet_name="multi_bz_long")


    def read_eta_sweep_data():
        excel_path = resolve_excel_path(
            OUTPUT_DIR / EXCEL_NAME_ETA,
            None,
            message_if_missing=(
                f"No encontré {OUTPUT_DIR / EXCEL_NAME_ETA}. "
                "Primero corre mode='compute_eta_sweep'."
            ),
        )
        return pd.read_excel(excel_path, sheet_name="eta_long")


    def plot_fixed_bz_from_excel():
        df_fixed = read_fixed_bz_data()
        plot_fixed_bz(df_fixed, filename_stem="chi_surf_fixed_bz_from_saved_data")


    def plot_multi_bz_from_excel():
        df_long = read_multi_bz_data()
        plot_multi_bz(df_long, filename_stem="chi_surf_multi_bz_from_saved_data")


    def plot_eta_sweep_from_excel():
        df_eta = read_eta_sweep_data()
        plot_eta_sweep(df_eta, filename_stem="chi_surf_eta_sweep_from_saved_data")


    def plot_parent_summary_from_excel():
        df_fixed = read_fixed_bz_data()
        df_multi = read_multi_bz_data()
        df_eta = read_eta_sweep_data()

        plot_parent_summary(
            df_fixed,
            df_multi,
            df_eta,
            filename_stem="chi_surf_parent_summary_fixed_multi_eta",
        )


    # ============================================================
    # MODOS COMPUESTOS
    # ============================================================
    def compute_and_plot_fixed_bz():
        df_fixed = compute_fixed_bz_data()
        plot_fixed_bz(df_fixed, filename_stem="chi_surf_fixed_bz")


    def compute_and_plot_multi_bz():
        df_long = compute_multi_bz_data()
        plot_multi_bz(df_long, filename_stem="chi_surf_multi_bz")


    def compute_and_plot_eta_sweep():
        df_eta = compute_eta_sweep_data()
        plot_eta_sweep(df_eta, filename_stem="chi_surf_eta_sweep")


    def compute_all():
        compute_fixed_bz_data()
        compute_multi_bz_data()
        compute_eta_sweep_data()


    def plot_all_from_excel():
        plot_fixed_bz_from_excel()
        plot_multi_bz_from_excel()
        plot_eta_sweep_from_excel()
        plot_parent_summary_from_excel()


    # ============================================================
    # Run
    # ============================================================
    def main():
        prepare_output_folders()

        if mode == "compute_fixed_bz":
            compute_fixed_bz_data()

        elif mode == "compute_multi_bz":
            compute_multi_bz_data()

        elif mode == "compute_eta_sweep":
            compute_eta_sweep_data()

        elif mode == "plot_fixed_bz_from_excel":
            plot_fixed_bz_from_excel()

        elif mode == "plot_multi_bz_from_excel":
            plot_multi_bz_from_excel()

        elif mode == "plot_eta_sweep_from_excel":
            plot_eta_sweep_from_excel()

        elif mode == "plot_parent_summary_from_excel":
            plot_parent_summary_from_excel()

        elif mode == "compute_and_plot_fixed_bz":
            compute_and_plot_fixed_bz()

        elif mode == "compute_and_plot_multi_bz":
            compute_and_plot_multi_bz()

        elif mode == "compute_and_plot_eta_sweep":
            compute_and_plot_eta_sweep()

        elif mode == "compute_all":
            compute_all()

        elif mode == "plot_all_from_excel":
            plot_all_from_excel()

        else:
            raise ValueError(
                "mode debe ser uno de: "
                "'compute_fixed_bz', 'compute_multi_bz', 'compute_eta_sweep', "
                "'plot_fixed_bz_from_excel', 'plot_multi_bz_from_excel', "
                "'plot_eta_sweep_from_excel', 'plot_parent_summary_from_excel', "
                "'compute_and_plot_fixed_bz', 'compute_and_plot_multi_bz', "
                "'compute_and_plot_eta_sweep', 'compute_all', 'plot_all_from_excel'."
            )

        print("\nListo. Revisa la carpeta:", OUTPUT_DIR.resolve())

    main()


# ============================================================
# TOP-LEVEL MODE VALIDATION
# ============================================================

def selected_response_components() -> list[str]:
    """Validate and return the requested response components."""
    valid_components = {
        "sea",
        "surface",
        "both",
    }

    if RESPONSE_COMPONENT not in valid_components:
        raise ValueError(
            "RESPONSE_COMPONENT must be 'sea', 'surface', or 'both'."
        )

    if RESPONSE_COMPONENT == "both":
        return ["sea", "surface"]

    return [RESPONSE_COMPONENT]


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """Run the selected Fermi-sea and/or Fermi-surface modules."""
    for component in selected_response_components():
        if component == "sea":
            print(
                "\n"
                + "=" * 72
                + f"\nRunning Fermi-sea mode: {SEA_MODE}\n"
                + "=" * 72
            )
            run_fermi_sea(SEA_MODE)

        elif component == "surface":
            print(
                "\n"
                + "=" * 72
                + f"\nRunning Fermi-surface mode: {SURFACE_MODE}\n"
                + "=" * 72
            )
            run_fermi_surface(SURFACE_MODE)

    print("\nAll selected spin-to-orbital response tasks are complete.")


if __name__ == "__main__":
    main()
