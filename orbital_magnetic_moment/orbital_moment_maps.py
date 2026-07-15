import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
# mode:
#   "grid_alpha" -> barre alpha, tj fijo
#   "grid_tj"    -> barre tj, alpha fijo
mode = "grid_tj"

# figure_style:
#   "grid"    -> figura 2x2
#   "stacked" -> figura 3D apilada
figure_style = "stacked"



# =========================================================
# PARÁMETROS DEL MODELO
# =========================================================
t0 = 3.2895              # eV Å^2
tj_fixed = 1.881         # eV Å^2
alpha_fixed_meVA = 26.0  # meV Å

# Usa solo 4 valores
alpha_values_meVA = [13, 26, 52, 100]   # meV Å
tj_values = [0.5, 1.0, 1.881, 3.0]      # eV Å^2

# =========================================================
# PARÁMETROS DE LA MALLA
# =========================================================
kmax = 0.25          # Å^{-1}
N = 501
k0_mask = 0.005      # Å^{-1}
percentile_clip = 99

# Contornos de Fermi (opcionales)
show_fermi = False
mu = 0.03   # eV

# Guardado
save_figs = True
fig_dpi = 600

# Carpeta de salida
output_dir = Path("figures_mz")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONSTANTES FÍSICAS
# =========================================================
e = 1.602176634e-19        # C
hbar = 1.054571817e-34     # J s
eV_to_J = 1.602176634e-19  # J/eV
A2_to_m2 = 1e-20           # Å^2 -> m^2
mu_B = 9.2740100783e-24    # J/T

# Conversión:
# [e/hbar] * [eV Å^2] -> mu_B
conv_to_muB = (e / hbar) * eV_to_J * A2_to_m2 / mu_B

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
# COLORMAP PERSONALIZADO
# Morado -> crema -> dorado
# =========================================================
cmap_mz = LinearSegmentedColormap.from_list(
    "purple_gold_mz",
    ["#5B2A86", "#F7F4EA", "#C98C00"],
    N=512
)

# =========================================================
# MALLA EN k
# =========================================================
kx = np.linspace(-kmax, kmax, N)
ky = np.linspace(-kmax, kmax, N)
KX, KY = np.meshgrid(kx, ky)
KR = np.sqrt(KX**2 + KY**2)

# =========================================================
# FUNCIONES DEL MODELO
# =========================================================
def orbital_moment_reduced(KX, KY, alpha, tj):
    """
    Momento orbital reducido en unidades de eV Å^2:
        m_red(k) = alpha^2 * tj * kx * ky /
                   [alpha^2 k^2 + 4 tj^2 kx^2 ky^2]
    """
    denom = alpha**2 * (KX**2 + KY**2) + 4.0 * tj**2 * KX**2 * KY**2
    num = alpha**2 * tj * KX * KY

    out = np.full_like(KX, np.nan, dtype=float)
    mask = denom > 1e-18
    out[mask] = num[mask] / denom[mask]
    return out


def orbital_moment_muB(KX, KY, alpha, tj):
    """
    Momento orbital en mu_B.
    """
    return conv_to_muB * orbital_moment_reduced(KX, KY, alpha, tj)


def bands(KX, KY, alpha, tj, t0):
    """
    Bandas del modelo Rashba--altermagnético:
        E_lambda(k) = t0 k^2 +- d(k)
    """
    K2 = KX**2 + KY**2
    d = np.sqrt(alpha**2 * K2 + 4.0 * tj**2 * KX**2 * KY**2)

    Eplus = t0 * K2 + d
    Eminus = t0 * K2 - d
    return Eplus, Eminus


def prepare_moment_for_plot(alpha_meVA, tj):
    """
    Calcula m_z(k) y aplica una máscara alrededor de k=0.
    """
    alpha = alpha_meVA * 1e-3   # meV Å -> eV Å
    mz = orbital_moment_muB(KX, KY, alpha, tj)

    mz_plot = mz.copy()
    mz_plot[KR < k0_mask] = np.nan
    return mz_plot


def get_vmax_from_percentile(maps, percentile_clip):
    """
    Obtiene escala simétrica de color usando percentil.
    """
    vals = np.abs(np.concatenate([
        arr[np.isfinite(arr)].ravel() for arr in maps
    ]))
    vmax = np.percentile(vals, percentile_clip)

    if vmax <= 0 or not np.isfinite(vmax):
        vmax = np.nanmax(vals)

    return vmax


# =========================================================
# PREPARAR MAPAS
# =========================================================
if mode == "grid_alpha":
    parameter_values = alpha_values_meVA
    maps = [
        prepare_moment_for_plot(alpha_meVA=alpha_meVA, tj=tj_fixed)
        for alpha_meVA in parameter_values
    ]
elif mode == "grid_tj":
    parameter_values = tj_values
    maps = [
        prepare_moment_for_plot(alpha_meVA=alpha_fixed_meVA, tj=tj)
        for tj in parameter_values
    ]
else:
    raise ValueError("mode debe ser 'grid_alpha' o 'grid_tj'.")

vmax = get_vmax_from_percentile(maps, percentile_clip)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

print(f"vmax usado para la colorbar: {vmax:.4g} mu_B")
print(f"Carpeta de salida: {output_dir.resolve()}")

# =========================================================
# FUNCIÓN: FIGURA 2x2
# =========================================================
def plot_grid_2x2(maps, parameter_values, mode):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.2, 8.0),
        constrained_layout=False
    )
    fig.set_dpi(fig_dpi)

    axes = axes.ravel()
    panel_labels = ["a)", "b)", "c)", "d)"]

    last_im = None

    for i, ax in enumerate(axes):
        mz_plot = maps[i]

        last_im = ax.pcolormesh(
            KX, KY, mz_plot,
            shading="auto",
            cmap=cmap_mz,
            norm=norm
        )

        if show_fermi:
            if mode == "grid_alpha":
                alpha = parameter_values[i] * 1e-3
                tj = tj_fixed
            else:
                alpha = alpha_fixed_meVA * 1e-3
                tj = parameter_values[i]

            Eplus, Eminus = bands(KX, KY, alpha, tj, t0)

            ax.contour(
                KX, KY, Eplus,
                levels=[mu],
                colors="black",
                linewidths=1.0
            )

            ax.contour(
                KX, KY, Eminus,
                levels=[mu],
                colors="black",
                linewidths=1.0,
                linestyles="--"
            )

        # Etiqueta del panel
        ax.text(
            -0.16, 1.08,
            panel_labels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold"
        )

        # Texto interno
        if mode == "grid_alpha":
            text_label = (
                rf"$\alpha = {parameter_values[i]:g}\ \mathrm{{meV\AA}}$" "\n"
                rf"$t_j = {tj_fixed:g}\ \mathrm{{eV\AA^2}}$"
            )
        else:
            text_label = (
                rf"$\alpha = {alpha_fixed_meVA:g}\ \mathrm{{meV\AA}}$" "\n"
                rf"$t_j = {parameter_values[i]:g}\ \mathrm{{eV\AA^2}}$"
            )

        ax.text(
            0.05, 0.95,
            text_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10
        )

        ax.set_xlabel(r"$k_x\ (\mathrm{\AA^{-1}})$", fontsize=12)
        ax.set_ylabel(r"$k_y\ (\mathrm{\AA^{-1}})$", fontsize=12)

        ax.set_aspect("equal")
        ax.set_xlim(-kmax, kmax)
        ax.set_ylim(-kmax, kmax)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.tick_params(
            direction="out",
            length=3.5,
            width=0.9,
            labelsize=10
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.08,
        top=0.95,
        wspace=0.28,
        hspace=0.32
    )

    cbar_ax = fig.add_axes([0.89, 0.22, 0.025, 0.56])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(r"$m_z(\mathbf{k})\ (\mu_B)$", fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    if save_figs:
        if mode == "grid_alpha":
            filename_png = output_dir / "mz_grid_alpha_2x2.png"
            filename_pdf = output_dir / "mz_grid_alpha_2x2.pdf"
        else:
            filename_png = output_dir / "mz_grid_tj_2x2.png"
            filename_pdf = output_dir / "mz_grid_tj_2x2.pdf"

        plt.savefig(filename_png, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
        plt.savefig(filename_pdf, bbox_inches="tight", facecolor="white")

        print(f"Figura 2x2 PNG guardada en: {filename_png.resolve()}")
        print(f"Figura 2x2 PDF guardada en: {filename_pdf.resolve()}")

    plt.show()


# =========================================================
# FUNCIÓN: FIGURA 3D APILADA
# =========================================================
def plot_stacked_mz_maps(maps, parameter_values, mode):
    fig = plt.figure(figsize=(8.2, 10.5))
    fig.set_dpi(fig_dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Menor stride = mayor resolución visual
    stride = 2

    X = KX[::stride, ::stride]
    Y = KY[::stride, ::stride]

    # Separación vertical entre planos
    z_gap = 2.4
    z_positions = z_gap * np.arange(len(maps), dtype=float)

    vmax_local = get_vmax_from_percentile(maps, percentile_clip)
    norm_local = TwoSlopeNorm(vmin=-vmax_local, vcenter=0.0, vmax=vmax_local)

    for i, mz_plot in enumerate(maps):
        C = mz_plot[::stride, ::stride]
        Z = np.full_like(X, z_positions[i], dtype=float)

        facecolors = cmap_mz(norm_local(C))
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

    ax.set_xlabel(r"$k_x\ (\mathrm{\AA^{-1}})$", labelpad=18)
    ax.set_ylabel(r"$k_y\ (\mathrm{\AA^{-1}})$", labelpad=14)

    if mode == "grid_alpha":
        ax.set_zlabel(r"$\alpha\ (\mathrm{meV\AA})$", labelpad=12)
    else:
        ax.set_zlabel(r"$t_j\ (\mathrm{eV\AA^2})$", labelpad=12)

    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(-kmax, kmax)
    ax.set_zlim(-0.6 * z_gap, z_positions[-1] + 0.8 * z_gap)

    # Escala en el eje z
    ax.set_zticks(z_positions)
    ax.set_zticklabels([f"{v:g}" for v in parameter_values])

    # Hace más alto el stack
    ax.set_box_aspect((1, 1, 2.1))

    # Vista
    ax.view_init(elev=30, azim=-58)

    # Quitar paneles de fondo
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)

    # Márgenes
    fig.subplots_adjust(
        left=0.02,
        right=0.82,
        bottom=0.10,
        top=0.98
    )

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm_local, cmap=cmap_mz)
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.62,
        pad=0.08
    )
    cbar.set_label(r"$m_z(\mathbf{k})\ (\mu_B)$", labelpad=14)

    if save_figs:
        if mode == "grid_alpha":
            filename_png = output_dir / "mz_stacked_alpha.png"
            filename_pdf = output_dir / "mz_stacked_alpha.pdf"
        else:
            filename_png = output_dir / "mz_stacked_tj.png"
            filename_pdf = output_dir / "mz_stacked_tj.pdf"

        plt.savefig(
            filename_png,
            dpi=fig_dpi,
            bbox_inches="tight",
            pad_inches=0.25,
            facecolor="white"
        )

        plt.savefig(
            filename_pdf,
            bbox_inches="tight",
            pad_inches=0.25,
            facecolor="white"
        )

        print(f"Figura apilada PNG guardada en: {filename_png.resolve()}")
        print(f"Figura apilada PDF guardada en: {filename_pdf.resolve()}")

    plt.show()
# =========================================================
# EJECUCIÓN
# =========================================================
if figure_style == "grid":
    plot_grid_2x2(maps, parameter_values, mode)

elif figure_style == "stacked":
    plot_stacked_mz_maps(maps, parameter_values, mode)

else:
    raise ValueError("figure_style debe ser 'grid' o 'stacked'.")
