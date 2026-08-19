"""Turn extracted keypoints into gait-signal figures.

Three panels, each answering a question we need answered before any
masking experiment is worth running:
  1. tracking quality  - is every joint actually recovered, frame by frame?
  2. gait signal       - do ankle/wrist trajectories show clean step cycles?
  3. masking groups    - which joints does each pretraining condition hide?
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from pose_topology import (LANDMARK_NAMES, MASK_GROUPS, POSE_CONNECTIONS,
                           region_of)

PALETTE = {"face": "#9aa0a6", "arms": "#4d9de0", "torso": "#e0a04d",
           "legs": "#3fa860"}


def wide(df, col):
    """long -> frames x 33 matrix for one coordinate."""
    return df.pivot_table(index="frame", columns="joint_id", values=col).sort_index()


def fig_quality(df, stem, outdir):
    vis = wide(df, "visibility")
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(vis.T.values, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[df.time_s.min(), df.time_s.max(), 32.5, -0.5])
    ax.set_yticks(range(33))
    ax.set_yticklabels([f"{i:2d} {n}" for i, n in enumerate(LANDMARK_NAMES)],
                       fontsize=7)
    for lab in ax.get_yticklabels():
        idx = int(lab.get_text().split()[0])
        lab.set_color(PALETTE[region_of(idx)])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{stem} — per-joint visibility (tracking quality)")
    fig.colorbar(im, ax=ax, label="visibility", pad=0.01)
    fig.tight_layout()
    path = f"{outdir}/{stem}_quality.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_gait(df, stem, outdir):
    """Gait signal from WORLD landmarks (metres, hip-centred).

    Image-space coordinates are dominated by perspective: the subject walks
    toward and away from the camera, so pixel scale drifts and swamps the
    step cycle. World landmarks are already scale-normalised, so step
    oscillations and arm swing read directly in metres.
    """
    wy = wide(df, "wy")     # vertical, +down
    wz = wide(df, "wz")     # depth, sagittal swing axis
    t = wide(df, "time_s").mean(axis=1).values
    # Derive fps from the data: clips in this set run at 26, 29 and 30 fps,
    # and a hardcoded rate silently rescales every frequency estimate.
    fps = 1.0 / np.median(np.diff(t))

    def clean(sig):
        s = pd.Series(sig).interpolate(limit_direction="both")
        return s.rolling(3, center=True, min_periods=1).median().values

    l_ank, r_ank = clean(-wy[27]), clean(-wy[28])
    l_wr, r_wr = clean(wz[15]), clean(wz[16])

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5))

    ax = axes[0]
    ax.plot(t, l_ank, color="#3fa860", lw=1.1, label="left ankle")
    ax.plot(t, r_ank, color="#14532d", lw=1.1, label="right ankle")
    ax.set_ylabel("ankle height above\nhip centre (m)")
    ax.set_title(f"{stem} — gait signal from world landmarks")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=.25)

    ax = axes[1]
    ax.plot(t, l_wr, color="#4d9de0", lw=1.1, label="left wrist")
    ax.plot(t, r_wr, color="#14456f", lw=1.1, label="right wrist")
    ax.set_ylabel("wrist fore-aft\nposition (m)")
    amp_l, amp_r = np.nanstd(l_wr), np.nanstd(r_wr)
    asym = abs(amp_l - amp_r) / max(amp_l + amp_r, 1e-9) * 100
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              title=f"swing amplitude L={amp_l:.3f} m  R={amp_r:.3f} m  "
                    f"asymmetry {asym:.0f}%", title_fontsize=8)
    ax.grid(alpha=.25)

    # Cadence. Use the L-R ankle difference: the legs move anti-phase, so the
    # difference cancels the common-mode drift from walking toward/away.
    ax = axes[2]
    sig = np.nan_to_num((l_ank - r_ank) - np.nanmean(l_ank - r_ank))
    LO, HI = 0.5, 1.6                       # plausible stride band (Hz)

    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    power = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    band = (freqs > LO) & (freqs < HI)
    fft_f = freqs[band][np.argmax(power[band])]

    # Autocorrelation is the primary estimate: it agrees across two cameras
    # filming the same walk, where the FFT peak does not.
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    ac /= ac[0] + 1e-12
    lo_lag, hi_lag = int(fps / HI), int(fps / LO)
    ac_f = fps / (lo_lag + int(np.argmax(ac[lo_lag:hi_lag])))

    disagree = abs(ac_f - fft_f) / max(ac_f, 1e-9) > 0.15
    ax.plot(freqs[band], power[band], color="#b5651d", lw=1.3, label="ankle L-R spectrum")
    ax.axvline(ac_f, color="#c0392b", ls="--", lw=1.4,
               label=f"autocorr {ac_f*2*60:.0f} steps/min")
    ax.axvline(fft_f, color="#7d3c98", ls=":", lw=1.4,
               label=f"FFT peak {fft_f*2*60:.0f} steps/min")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_title("estimators DISAGREE >15% - cadence unreliable for this clip"
                 if disagree else "estimators agree", fontsize=9,
                 color="#c0392b" if disagree else "#2d7a3e")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.25)

    fig.tight_layout()
    path = f"{outdir}/{stem}_gait.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    flag = "  << DISAGREE, treat as unreliable" if disagree else ""
    print(f"   {fps:.0f} fps | cadence autocorr {ac_f*2*60:.0f} / fft "
          f"{fft_f*2*60:.0f} steps/min | arm-swing asymmetry {asym:.0f}%{flag}")
    return path


def fig_masking(df, stem, outdir):
    """One well-tracked frame, with each condition's masked joints circled.

    A median across frames collapses to a squashed pose because the subject
    turns around mid-clip, so pick the single best-tracked frame instead.
    """
    vis = wide(df, "visibility")
    best = vis.mean(axis=1).idxmax()
    x = wide(df, "x").loc[best]
    y = wide(df, "y").loc[best]
    conds = [("random 15% (baseline)", None)] + [(k, v) for k, v in MASK_GROUPS.items()]

    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 6.2))
    rng = np.random.default_rng(0)
    for ax, (name, masked) in zip(axes, conds):
        if masked is None:
            masked = sorted(rng.choice(33, 5, replace=False).tolist())
        mset = set(masked)
        for a, b in POSE_CONNECTIONS:
            hid = a in mset or b in mset
            ax.plot([x[a], x[b]], [y[a], y[b]],
                    color="#d8d8d8" if hid else PALETTE[region_of(b)],
                    lw=1.0 if hid else 2.0, zorder=1,
                    ls=":" if hid else "-")
        for i in range(33):
            if i in mset:
                ax.scatter(x[i], y[i], s=44, facecolors="none",
                           edgecolors="#c0392b", lw=1.4, zorder=3)
            else:
                ax.scatter(x[i], y[i], s=22, color=PALETTE[region_of(i)], zorder=2)
        ax.set_title(f"{name}\n{len(mset)} of 33 masked", fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor="#c0392b", label="masked during pretraining")]
    fig.legend(handles=handles, loc="lower center", frameon=False)
    fig.suptitle(f"Masking conditions on the tracked skeleton ({stem})", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = f"{outdir}/{stem}_masking_conditions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarize(df, stem):
    vis = wide(df, "visibility")
    n_frames = len(vis)
    print(f"\n[{stem}] {n_frames} tracked frames, {df.time_s.max():.1f}s")
    grp = {}
    for i in range(33):
        grp.setdefault(region_of(i), []).append(vis[i].mean())
    for r, vals in grp.items():
        print(f"   {r:6s} mean visibility {np.mean(vals):.3f}")
    worst = vis.mean().nsmallest(5)
    print("   weakest joints: " +
          ", ".join(f"{LANDMARK_NAMES[i]} {v:.2f}" for i, v in worst.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--outdir", default="outputs/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for pq in args.parquet:
        stem = os.path.splitext(os.path.basename(pq))[0]
        df = pd.read_parquet(pq)
        summarize(df, stem)
        for f in (fig_quality, fig_gait, fig_masking):
            print("   ->", f(df, stem, args.outdir))


if __name__ == "__main__":
    main()
