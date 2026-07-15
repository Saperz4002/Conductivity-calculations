"""
Constant-energy spin textures of pure Rashba and altermagnetic models
====================================================================

Overview
--------
This script calculates constant-energy contours in the two-dimensional
momentum plane and overlays the corresponding spin texture for two limiting
cases of a Rashba-altermagnetic Hamiltonian.

For every energy listed in ``E0_list``, the script generates one figure for
the pure Rashba model and one for the pure d_xy altermagnetic model.

Available modes
---------------
1. ``"rashba"``
   Uses the dispersion

       E_±(k) = t0 (kx² + ky²) ± alpha sqrt(kx² + ky²).

   The spin lies in the momentum plane and is represented by black arrows
   tangent to the constant-energy contours. The two bands have opposite
   Rashba helicities.

2. ``"am"``
   Uses the dispersion

       E_±(k) = t0 (kx² + ky²) ± 2 tj kx ky.

   The spin points perpendicular to the momentum plane. A dot represents
   spin along +z, while a cross represents spin along -z.

Outputs
-------
Each figure is displayed and saved in both high-resolution PNG and vector
PDF formats inside the directory specified by ``output_dir``.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Parameters
# ============================================================
t0 = 3.2895      # eV Å^2
tj = 1.881       # eV Å^2
alpha = 0.052    # eV Å

kxmax = 0.10
kymax = 0.10

n_grid_x = 700
n_grid_y = 700
n_samples = 22

E0_list = [7.3e-3]   # eV

output_dir = "spin_textures_pure_cases_styled"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Style
# ============================================================
RED = "red"
BLUE = "blue"
BG = "#f2f2f2"
GRID = "gray"

# ============================================================
# Grid
# ============================================================
kx = np.linspace(-kxmax, kxmax, n_grid_x)
ky = np.linspace(-kymax, kymax, n_grid_y)
KX, KY = np.meshgrid(kx, ky)

# ============================================================
# Model definitions
# ============================================================
def bands_model(KX, KY, mode="rashba"):
    """
    mode = 'rashba' or 'am'
    """
    k2 = KX**2 + KY**2

    if mode == "rashba":
        Delta = alpha * np.sqrt(k2)
        E1 = t0 * k2 + Delta
        E2 = t0 * k2 - Delta

    elif mode == "am":
        # Pure altermagnet: spin-up / spin-down branches.
        # Important: no absolute value.
        E1 = t0 * k2 + 2 * tj * KX * KY   # s_z = +1
        E2 = t0 * k2 - 2 * tj * KX * KY   # s_z = -1

    else:
        raise ValueError("mode must be 'rashba' or 'am'")

    return E1, E2


def spin_xyz(kx, ky, band, mode="rashba"):
    """
    band = +1 or -1
    """
    if mode == "rashba":
        k = np.sqrt(kx**2 + ky**2)

        if np.isscalar(k):
            if k == 0:
                return 0.0, 0.0, 0.0
        else:
            k = np.where(k == 0, np.nan, k)

        sx = -band * ky / k
        sy =  band * kx / k
        sz = 0.0 * k

    elif mode == "am":
        sx = 0.0 * np.array(kx, dtype=float)
        sy = 0.0 * np.array(ky, dtype=float)
        sz = band * np.ones_like(np.array(kx, dtype=float))

    else:
        raise ValueError("mode must be 'rashba' or 'am'")

    return sx, sy, sz


# ============================================================
# Sample equally spaced points along a contour
# ============================================================
def sample_vertices(vertices, n_samples):
    if len(vertices) < 2:
        return vertices

    diffs = np.diff(vertices, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len == 0:
        return vertices[:1]

    targets = np.linspace(0, total_len, n_samples, endpoint=False)
    sampled = []

    for s in targets:
        idx = np.searchsorted(cumlen, s, side="right") - 1
        idx = min(idx, len(seg_lengths) - 1)

        ds = seg_lengths[idx]
        if ds == 0:
            sampled.append(vertices[idx])
        else:
            frac = (s - cumlen[idx]) / ds
            p = vertices[idx] + frac * (vertices[idx + 1] - vertices[idx])
            sampled.append(p)

    return np.array(sampled)


# ============================================================
# Add out-of-plane spin markers: dot/cross
# ============================================================
def add_oop_spin_markers(ax, contour_set, edgecolor, kind="dot", size=36):
    """
    kind = 'dot'   -> spin out of plane, symbol odot
    kind = 'cross' -> spin into plane, symbol otimes
    """
    for level_paths in contour_set.allsegs:
        for verts in level_paths:
            if len(verts) < 5:
                continue

            pts = sample_vertices(verts, n_samples)

            # Outer hollow circle
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=size,
                marker="o",
                facecolors="white",
                edgecolors=edgecolor,
                linewidths=1.15,
                zorder=6
            )

            # Inner symbol
            if kind == "dot":
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=size * 0.17,
                    marker="o",
                    c="black",
                    linewidths=0.8,
                    zorder=7
                )

            elif kind == "cross":
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=size * 0.55,
                    marker="+",
                    c="black",
                    linewidths=1.0,
                    zorder=7
                )

            else:
                raise ValueError("kind must be 'dot' or 'cross'")


# ============================================================
# Add Rashba arrows
# ============================================================
def add_rashba_arrows(ax, contour_set, band, color, arrow_scale=0.010):
    for level_paths in contour_set.allsegs:
        for verts in level_paths:
            if len(verts) < 5:
                continue

            pts = sample_vertices(verts, n_samples)

            for p in pts:
                kxp, kyp = p
                sx, sy, _ = spin_xyz(kxp, kyp, band=band, mode="rashba")
                norm = np.sqrt(sx**2 + sy**2)

                if not np.isfinite(norm) or norm == 0:
                    continue

                ax.arrow(
                    kxp, kyp,
                    arrow_scale * sx,
                    arrow_scale * sy,
                    head_width=0.0038,
                    head_length=0.0050,
                    fc=color,
                    ec=color,
                    linewidth=0.8,
                    length_includes_head=True,
                    zorder=5
                )


# ============================================================
# Add corner spin guide for AM
# ============================================================
def add_am_corner_guide(ax):
    ax.text(
        0.95, 0.95, r'$\uparrow$',
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="gray"
    )

    ax.text(
        0.05, 0.05, r'$\uparrow$',
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="gray"
    )

    ax.text(
        0.95, 0.05, r'$\downarrow$',
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="gray"
    )

    ax.text(
        0.05, 0.95, r'$\downarrow$',
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="gray"
    )


# ============================================================
# Plot one figure
# ============================================================
def plot_texture(E0, mode="rashba", panel_label=None):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 12
    })

    E1, E2 = bands_model(KX, KY, mode=mode)

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=300)
    ax.set_facecolor(BG)

    # Contours
    cs_1 = ax.contour(
        KX, KY, E1,
        levels=[E0],
        colors=[RED],
        linewidths=2.0
    )

    cs_2 = ax.contour(
        KX, KY, E2,
        levels=[E0],
        colors=[BLUE],
        linewidths=2.0
    )

    if mode == "rashba":
        # Pure Rashba: contours + in-plane arrows only
        add_rashba_arrows(ax, cs_1, band=+1, color='black')
        add_rashba_arrows(ax, cs_2, band=-1, color='black')

    elif mode == "am":
        # Pure AM: contours + out-of-plane spin markers
        # Red branch: s_z = +1, represented by dot
        # Blue branch: s_z = -1, represented by cross
        add_oop_spin_markers(ax, cs_1, edgecolor=RED, kind="dot", size=36)
        add_oop_spin_markers(ax, cs_2, edgecolor=BLUE, kind="cross", size=36)
        add_am_corner_guide(ax)

    else:
        raise ValueError("mode must be 'rashba' or 'am'")

    # Axes
    ax.set_xlim(-kxmax, kxmax)
    ax.set_ylim(-kymax, kymax)
    ax.set_aspect("equal")

    xticks = np.linspace(-kxmax, kxmax, 5)
    yticks = np.linspace(-kymax, kymax, 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([rf"${x:.2f}$" for x in xticks])
    ax.set_yticklabels([rf"${y:.2f}$" for y in yticks])

    ax.set_xlabel(r"$k_x\;(\mathrm{\AA}^{-1})$", fontsize=14)
    ax.set_ylabel(r"$k_y\;(\mathrm{\AA}^{-1})$", fontsize=14)

    ax.grid(True, color=GRID, alpha=0.28, linewidth=0.8)
    ax.tick_params(direction="in", length=4, width=1.0, labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Chemical potential / energy box
    mu_meV = 1000 * E0

    if mode == "am":
        textstr = rf"$E_0 = {E0:.1e}\ \mathrm{{eV}}$"
    else:
        textstr = rf"$\mu = {mu_meV:.2f}\ \mathrm{{meV}}$"

    ax.text(
        0.04, 0.96, textstr,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="square",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95
        )
    )

    # Optional panel label
    if panel_label is not None:
        ax.text(
            0.95, 0.93, panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            fontweight="bold"
        )

    plt.tight_layout()

    filename_png = os.path.join(output_dir, f"{mode}_mu_{mu_meV:.2f}_meV.png")
    filename_pdf = os.path.join(output_dir, f"{mode}_mu_{mu_meV:.2f}_meV.pdf")

    fig.savefig(filename_png, dpi=600, bbox_inches="tight")
    fig.savefig(filename_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# Run
# ============================================================
for E0 in E0_list:
    plot_texture(E0, mode="rashba", panel_label="a)")
    plot_texture(E0, mode="am", panel_label="b)")
