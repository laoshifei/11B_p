#!/usr/bin/env python3
"""Plot primary-alpha energy spectra (c.m. vs lab) in one panel.

    branch_id == 0  ->  alpha0  (8Be g.s.)
    branch_id == 1  ->  alpha1  (8Be 2+)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import uproot

FIXED_MARGINS = dict(left=0.18, right=0.86, bottom=0.15, top=0.90)


def plot_162_primary(filename, save=False):
    """画初级 α 能谱，返回 fig。

    filename : ROOT 文件路径（str 或 Path 都行）
    save     : False 不存盘；给字符串/Path 就存到该路径；给 True 存默认文件名
    """
    with uproot.open(filename, handler=uproot.source.file.MemmapSource) as f:
        tree = f["reaction"]
        if "h11b162_labels_are_bookkeeping" in tree:
            if np.any(tree["h11b162_labels_are_bookkeeping"].array(library="np")):
                raise ValueError("Symmetric 162 samples have no unique primary alpha; use all-alpha or permutation-invariant observables.")
        a = tree.arrays(
            ["e_alpha1", "e_alpha1_cm", "branch_id"], library="np"
        )

    bid = a["branch_id"]
    C_CM  = "#B23A48"   # 红   = 质心系
    C_LAB = "#0E7C6B"   # 青绿 = 实验室系

    emax = max(a["e_alpha1"].max(), a["e_alpha1_cm"].max()) * 1.05
    bins = np.linspace(0, emax, 150)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(a["e_alpha1_cm"], bins=bins, histtype="step", lw=1.6, color=C_CM)
    ax.hist(a["e_alpha1"],    bins=bins, histtype="step", lw=1.6, color=C_LAB)

    e_cm = a["e_alpha1_cm"]
    a0_x = np.median(e_cm[bid == 0])
    a1_x = np.median(e_cm[bid == 1])
    ymax = ax.get_ylim()[1]
    ax.annotate(r"$\alpha_1$", xy=(a1_x, ymax * 0.9), ha="center", fontsize=13)
    ax.annotate(r"$\alpha_0$", xy=(a0_x + 0.3, ymax * 0.5), ha="center", fontsize=13)

    handles = [Line2D([0], [0], color=C_CM,  lw=1.6, label="c.m."),
               Line2D([0], [0], color=C_LAB, lw=1.6, label="lab")]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    ax.set_xlabel(r"primary $\alpha$ energy (MeV)")
    ax.set_ylabel("counts")
    fig.subplots_adjust(**FIXED_MARGINS)

    if save:
        path = "162_primary.png" if save is True else str(save)
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    plot_162_primary(args.filename, save=True)   # 终端：安静存 PNG
