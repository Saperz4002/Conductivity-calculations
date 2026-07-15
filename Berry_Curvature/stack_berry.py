"""
Stacked Berry-curvature maps for a Rashba--altermagnetic model
==============================================================

Overview
--------
This script calculates the Berry curvature of a two-dimensional
Rashba--altermagnetic model on a k-space grid and visualizes it as stacked
3D maps for the upper and lower bands.

The Berry curvature used is

    Omega_lambda(k) =
        lambda * alpha^2 * tj * kx * ky
        / [ alpha^2 (kx^2 + ky^2) + 4 tj^2 kx^2 ky^2 ]^(3/2),

where lambda = +1 corresponds to the upper band and lambda = -1 to the
lower band.

For each selected parameter value, the script:
1. evaluates the Berry curvature on the (kx, ky) grid,
2. masks the singular region near k = 0,
3. applies a common color scale to all panels, and
4. stacks the resulting maps vertically in a 3D plot.

Available modes
---------------
1. ``"grid_alpha"``
   Sweeps the Rashba coupling ``alpha`` while keeping the altermagnetic
   parameter ``tj`` fixed at ``tj_fixed``.

2. ``"grid_tj"``
   Sweeps the altermagnetic parameter ``tj`` while keeping the Rashba
   coupling fixed at ``alpha_fixed_meVA``.

Outputs
-------
The script displays a figure with two stacked 3D panels:
- left panel:  upper-band Berry curvature, ``Omega^{xy}_{+}(k)``
- right panel: lower-band Berry curvature, ``Omega^{xy}_{-}(k)``

If ``save_figs = True``, the figure is saved in both PNG and PDF formats
inside the directory specified by ``output_dir``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
# mode:
#   "grid_alpha" -> barre alpha, tj fijo
#   "grid_tj"    -> barre tj, alpha fijo
mode = "grid_alpha"

# =========================================================
# PARÁMETROS DEL MODELO
# =========================================================
t0 = 3.2895              # eV Å^2
tj_fixed = 1.881         # eV Å^2
alpha_fixed_meVA = 26.0  # meV Å

alpha_values_meVA = [13, 26, 52, 100]   # meV Å
tj_values = [0.5, 1.0, 1.881, 3.0]      # eV Å^2

# =========================================================
# PARÁMETROS DE LA MALLA
# =========================================================
kmax = 0.025        # Å^{-1}
N = 501
k0_mask = 0.0004    # Å^{-1}
percentile_clip = 99

# Guardado
save_figs = True
fig_dpi = 600

# Carpeta de salida
output_dir = Path("figures_berry_stacked")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# ESTILO GENERAL
# =========================================================
plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "mathtext.fontset": "stix",
    "font.family": "sans-serif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": fig_dpi
})

# =========================================================
# MALLA EN k
# =========================================================
kx = np.linspace(-kmax, kmax, N)
ky = np.linspace(-kmax, kmax, N)
KX, KY = np.meshgrid(kx, ky)
KR = np.sqrt(KX**2 + KY**2)

# =========================================================
# FUNCIÓN DE CURVATURA DE BERRY
# =========================================================
def berry_curvature(KX, KY, alpha, tj, lam):
    """
    Curvatura de Berry para el modelo Rashba--altermagnético:

    Omega_lambda(k) =
      lam * alpha^2 * tj * kx * ky /
      [ alpha^2(kx^2+ky^2) + 4 tj^2 kx^2 ky^2 ]^(3/2)

    lam = +1 banda superior
    lam = -1 banda inferior

    Unidades: Å^2
    """
    denom = alpha**2 * (KX**2 + KY**2) + 4.0 * tj**2 * KX**2 * KY**2

    out = np.full_like(KX, np.nan, dtype=float)
    mask = denom > 1e-18

    out[mask] = (
        lam * alpha**2 * tj * KX[mask] * KY[mask]
        / (denom[mask] ** 1.5)
    )

    return out

# =========================================================
# PREPARACIÓN DE MAPAS
# =========================================================
def prepare_curvature_for_plot(alpha_meVA, tj):
    alpha = alpha_meVA * 1e-3  # meV Å -> eV Å

    Om_plus = berry_curvature(KX, KY, alpha, tj, lam=+1)
    Om_minus = berry_curvature(KX, KY, alpha, tj, lam=-1)

    Om_plus_plot = Om_plus.copy()
    Om_minus_plot = Om_minus.copy()

    Om_plus_plot[KR < k0_mask] = np.nan
    Om_minus_plot[KR < k0_mask] = np.nan

    return Om_plus_plot, Om_minus_plot


def get_vmax_from_percentile(maps, percentile_clip):
    vals = np.abs(np.concatenate([
        arr[np.isfinite(arr)].ravel() for arr in maps
    ]))

    vmax = np.percentile(vals, percentile_clip)

    if vmax <= 0 or not np.isfinite(vmax):
        vmax = np.nanmax(vals)

    return vmax

# =========================================================
# PREPARAR BARRIDO
# =========================================================
if mode == "grid_alpha":
    parameter_values = alpha_values_meVA

    maps_plus = []
    maps_minus = []

    for alpha_meVA in parameter_values:
        Om_p, Om_m = prepare_curvature_for_plot(alpha_meVA=alpha_meVA, tj=tj_fixed)
        maps_plus.append(Om_p)
        maps_minus.append(Om_m)

elif mode == "grid_tj":
    parameter_values = tj_values

    maps_plus = []
    maps_minus = []

    for tj in parameter_values:
        Om_p, Om_m = prepare_curvature_for_plot(alpha_meVA=alpha_fixed_meVA, tj=tj)
        maps_plus.append(Om_p)
        maps_minus.append(Om_m)

else:
    raise ValueError("mode debe ser 'grid_alpha' o 'grid_tj'.")

# Escala común para ambas bandas
all_maps = maps_plus + maps_minus
vmax = get_vmax_from_percentile(all_maps, percentile_clip)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

print(f"vmax usado para la colorbar: {vmax:.4g} Å²")
print(f"Carpeta de salida: {output_dir.resolve()}")

# =========================================================
# FUNCIÓN: STACK 3D
# =========================================================
def plot_single_stack(ax, maps, parameter_values, mode, title):
    # menor stride = mejor resolución visual
    stride = 2

    X = KX[::stride, ::stride]
    Y = KY[::stride, ::stride]

    # separación vertical
    z_gap = 2.4
    z_positions = z_gap * np.arange(len(maps), dtype=float)

    for i, Om_plot in enumerate(maps):
        C = Om_plot[::stride, ::stride]
        Z = np.full_like(X, z_positions[i], dtype=float)

        facecolors = cm.get_cmap("RdBu_r")(norm(C))
        facecolors[..., -1] = 0.96

        ax.plot_surface(
            X, Y, Z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False
        )

    ax.set_title(title, pad=18, fontsize=15)

    ax.set_xlabel(r"$k_x\ (\mathrm{\AA^{-1}})$", labelpad=14)
    ax.set_ylabel(r"$k_y\ (\mathrm{\AA^{-1}})$", labelpad=10)

    if mode == "grid_alpha":
        ax.set_zlabel(r"$\alpha\ (\mathrm{meV\AA})$", labelpad=10)
    else:
        ax.set_zlabel(r"$t_j\ (\mathrm{eV\AA^2})$", labelpad=10)

    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(-kmax, kmax)
    ax.set_zlim(-0.6 * z_gap, z_positions[-1] + 0.8 * z_gap)

    ax.set_zticks(z_positions)
    ax.set_zticklabels([f"{v:g}" for v in parameter_values])

    # hacer más alto el stack
    ax.set_box_aspect((1, 1, 2.0))

    # vista
    ax.view_init(elev=28, azim=-58)

    # quitar paneles de fondo
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)


def plot_stacked_berry_maps(maps_plus, maps_minus, parameter_values, mode):
    fig = plt.figure(figsize=(14.0, 9.2))
    fig.set_dpi(fig_dpi)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_single_stack(
        ax1,
        maps_plus,
        parameter_values,
        mode,
        title=r"$\Omega^{xy}_{+}(\mathbf{k})$"
    )

    plot_single_stack(
        ax2,
        maps_minus,
        parameter_values,
        mode,
        title=r"$\Omega^{xy}_{-}(\mathbf{k})$"
    )

    # etiquetas de panel
    ax1.text2D(0.02, 0.98, "a)", transform=ax1.transAxes,
               fontsize=18, fontweight="bold", va="top")
    ax2.text2D(0.02, 0.98, "b)", transform=ax2.transAxes,
               fontsize=18, fontweight="bold", va="top")

    # texto superior global
    if mode == "grid_alpha":
        fig.suptitle(
            rf"Berry curvature stack for fixed $t_j = {tj_fixed:g}\ \mathrm{{eV\AA^2}}$",
            fontsize=16,
            y=0.97
        )
        filename_base = "berry_stacked_alpha"
    else:
        fig.suptitle(
            rf"Berry curvature stack for fixed $\alpha = {alpha_fixed_meVA:g}\ \mathrm{{meV\AA}}$",
            fontsize=16,
            y=0.97
        )
        filename_base = "berry_stacked_tj"

    # colorbar común
    mappable = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=[ax1, ax2],
        shrink=0.72,
        pad=0.06
    )
    cbar.set_label(r"$\Omega^{xy}_{\lambda}(\mathbf{k})\ (\mathrm{\AA^2})$", labelpad=12)

    fig.subplots_adjust(
        left=0.03,
        right=0.88,
        bottom=0.06,
        top=0.92,
        wspace=0.12
    )

    if save_figs:
        filename_png = output_dir / f"{filename_base}.png"
        filename_pdf = output_dir / f"{filename_base}.pdf"

        plt.savefig(
            filename_png,
            dpi=fig_dpi,
            bbox_inches="tight",
            pad_inches=0.22,
            facecolor="white"
        )

        plt.savefig(
            filename_pdf,
            bbox_inches="tight",
            pad_inches=0.22,
            facecolor="white"
        )

        print(f"Figura apilada PNG guardada en: {filename_png.resolve()}")
        print(f"Figura apilada PDF guardada en: {filename_pdf.resolve()}")

    plt.show()

# =========================================================
# EJECUCIÓN
# =========================================================
plot_stacked_berry_maps(maps_plus, maps_minus, parameter_values, mode)