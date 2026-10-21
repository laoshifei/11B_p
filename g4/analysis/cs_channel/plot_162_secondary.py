#!/usr/bin/env python3
"""Plot the secondary-alpha energy spectrum and angular distribution (162-keV
resonance), colouring the two sequential channels separately.

The 162-keV resonance proceeds as

    p + 11B  ->  alpha_primary + 8Be  ->  alpha_primary + (alpha + alpha)

The two alphas from the 8Be break-up are the *secondary* alphas (particle
role BeDecayAlpha). They come in two channels, tagged by branch_id:

    branch_id == 0  ->  alpha0  channel : 8Be ground state  break-up
    branch_id == 1  ->  alpha1  channel : 8Be 2+ (3.03 MeV) break-up

This script produces TWO figures, with alpha0- and alpha1-channel secondaries
shown in different colours in both:

    figure 1 : energy spectrum       (counts vs secondary-alpha energy)
    figure 2 : angular distribution  (normalised yield vs cos(chi), where chi
               is the secondary-alpha angle relative to the 8Be recoil axis
               in the c.m. frame -- the *intrinsic* 8Be break-up angle).

Note on the angular distribution: it uses the internal 8Be-frame correlation
angle ``cos_chi_secondary_8be``, NOT the lab-frame polar angle. In the 8Be
frame the alpha0 channel (8Be g.s.) is isotropic (flat at 0.5) and the alpha1
channel (8Be 2+) shows the Legendre A2/A4 anisotropy. The lab-frame polar
angle would instead be smeared forward by the 8Be boost and would not show
this pattern.

Dual use:
    terminal :  python3 plot_162_secondary.py FILE.root
                (saves both PNGs, no window pop-up)
    jupyter  :  from plot_162_secondary import plot_162_secondary
                fig_e, fig_ang = plot_162_secondary("FILE.root")
                (figures display inline automatically under %matplotlib inline)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import uproot

FIXED_MARGINS = dict(left=0.18, right=0.86, bottom=0.15, top=0.90)

# ---- particle-role code for an alpha from the 8Be break-up -----------------
# (H11BParticleRole::BeDecayAlpha in include/H11BParticleLabel.hh)
BE_DECAY_ALPHA = 2

# ---- colours: distinguish the two sequential channels ----------------------
C_A0 = "#1F6FB2"   # blue   = alpha0 channel (8Be g.s.)  secondaries
C_A1 = "#E08B1E"   # orange = alpha1 channel (8Be 2+)    secondaries


def plot_162_secondary(filename, save=False, frame="lab"):
    """Plot secondary-alpha energy spectrum + angular distribution for the
    162-keV resonance.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ROOT file (must contain the ``reaction`` tree).
    save : bool or str, optional
        False (default) -> do not write files.
        True            -> write the two default PNGs.
        str             -> use it as a filename prefix, e.g. save="run162"
                           writes run162_secondary_energy.png and
                           run162_secondary_angular.png.
    frame : {"lab", "cm"}, optional
        Frame for the *energy* spectrum. The angular distribution always uses
        the lab-frame polar angle (the only angle stored in the tree).

    Returns
    -------
    (fig_energy, fig_angular) : tuple of matplotlib.figure.Figure
    """
    if frame not in ("lab", "cm"):
        raise ValueError("frame must be 'lab' or 'cm'")

    # secondary-alpha energy branches for the chosen frame
    e2_key = "e_alpha2" if frame == "lab" else "e_alpha2_cm"
    e3_key = "e_alpha3" if frame == "lab" else "e_alpha3_cm"

    fields = [e2_key, e3_key,
              "cos_chi_secondary_8be",
              "alpha2_role", "alpha3_role",
              "branch_id", "resonance_id"]

    with uproot.open(filename, handler=uproot.source.file.MemmapSource) as f:
        tree = f["reaction"]
        if "h11b162_labels_are_bookkeeping" in tree:
            if np.any(tree["h11b162_labels_are_bookkeeping"].array(library="np")):
                raise ValueError("Symmetric 162 samples have no unique secondary pair; use all three pairs or permutation-invariant observables.")
        a = tree.arrays(fields, library="np")

    is162 = a["resonance_id"] == 162

    # ---- secondary-alpha ENERGIES ------------------------------------------
    # Pool the two secondary slots (alpha2, alpha3), keeping only true
    # 8Be-break-up alphas (role BeDecayAlpha) from the 162 resonance. Each
    # secondary carries the branch_id of its parent event.
    def collect_energy(role_key, e_key):
        m = is162 & (a[role_key] == BE_DECAY_ALPHA)
        return a[e_key][m], a["branch_id"][m]

    e2, b2 = collect_energy("alpha2_role", e2_key)
    e3, b3 = collect_energy("alpha3_role", e3_key)
    energy = np.concatenate([e2, e3])
    e_branch = np.concatenate([b2, b3])
    e_a0 = e_branch == 0
    e_a1 = e_branch == 1

    # ---- secondary-alpha ANGLE (intrinsic 8Be-frame correlation) -----------
    # cos_chi_secondary_8be is one value per event (secondary alpha vs the
    # 8Be recoil axis in the c.m. frame). Select sequential 162 events.
    m_ang = is162 & (a["alpha2_role"] == BE_DECAY_ALPHA)
    cos_chi = a["cos_chi_secondary_8be"][m_ang]
    ang_branch = a["branch_id"][m_ang]
    good = np.isfinite(cos_chi) & (cos_chi >= -1) & (cos_chi <= 1)
    cos_chi = cos_chi[good]
    ang_branch = ang_branch[good]
    ang_a0 = ang_branch == 0
    ang_a1 = ang_branch == 1

    frame_label = "c.m." if frame == "cm" else "lab"

    # ---------------------------------------------------------------- figure 1
    # energy spectrum
    fig_e, ax_e = plt.subplots(figsize=(5.5, 4.5))
    if energy.size:
        emax = energy.max() * 1.05
        ebins = np.linspace(0, emax, 150)
        ax_e.hist(energy[e_a0], bins=ebins, histtype="step", lw=1.6, color=C_A0)
        ax_e.hist(energy[e_a1], bins=ebins, histtype="step", lw=1.6, color=C_A1)
    ax_e.set_xlabel(rf"secondary $\alpha$ energy ({frame_label}) (MeV)")
    ax_e.set_ylabel("counts")
    handles = [Line2D([0], [0], color=C_A0, lw=1.6,
                      label=r"$\alpha_0$ channel (8Be g.s.)"),
               Line2D([0], [0], color=C_A1, lw=1.6,
                      label=r"$\alpha_1$ channel (8Be 2$^+$)")]
    ax_e.legend(handles=handles, loc="upper right", frameon=False)
    fig_e.subplots_adjust(**FIXED_MARGINS)

    # ---------------------------------------------------------------- figure 2
    # angular distribution: intrinsic 8Be-frame correlation angle.
    # density=True -> uniform (isotropic) sits at 0.5 over cos in [-1, 1].
    fig_ang, ax_ang = plt.subplots(figsize=(5.5, 4.5))
    cbins = np.linspace(-1, 1, 50)
    ax_ang.axhline(0.5, ls=":", lw=0.9, color="0.5")  # isotropic reference
    if cos_chi[ang_a0].size:
        ax_ang.hist(cos_chi[ang_a0], bins=cbins, histtype="step",
                    lw=1.6, color=C_A0, density=True)
    if cos_chi[ang_a1].size:
        ax_ang.hist(cos_chi[ang_a1], bins=cbins, histtype="step",
                    lw=1.6, color=C_A1, density=True)
    ax_ang.set_xlim(-1, 1)
    ax_ang.set_ylim(bottom=0)
    ax_ang.set_xlabel(r"$\cos\chi$ (secondary $\alpha$ vs $^8$Be axis)")
    ax_ang.set_ylabel("normalised yield")
    ax_ang.legend(handles=handles, loc="upper right", frameon=False)
    fig_ang.subplots_adjust(**FIXED_MARGINS)

    # ------------------------------------------------------------------- save
    if save:
        prefix = "162" if save is True else str(save)
        e_path   = f"{prefix}_secondary_energy.png"
        ang_path = f"{prefix}_secondary_angular.png"
        fig_e.savefig(e_path, dpi=200)
        fig_ang.savefig(ang_path, dpi=200)
        print(f"wrote {e_path}")
        print(f"wrote {ang_path}")

    return fig_e, fig_ang


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot secondary-alpha energy spectrum and angular "
                    "distribution for the 162-keV resonance.")
    parser.add_argument("filename", help="ROOT file with the 'reaction' tree")
    parser.add_argument("--frame", choices=["lab", "cm"], default="lab",
                        help="frame for the energy spectrum (default: lab)")
    parser.add_argument("--prefix", default="162",
                        help="output filename prefix (default: 162)")
    args = parser.parse_args()

    plot_162_secondary(args.filename, save=args.prefix, frame=args.frame)
