"""One summary figure: per specimen a cross-section with segmentation overlay, a 3D projection
(max-intensity of the stack + mask outline + skeleton body-length path), and measured vs
published max dimension across the threshold sweep.

Usage: python -I pipeline/make_figure.py ROOT   -> ROOT/outputs/figure_traits.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation

root = Path(sys.argv[1]).resolve()
res = json.loads((root / "outputs" / "results.json").read_text())
names = list(res["specimens"])
fig, axes = plt.subplots(len(names), 3, figsize=(13, 4.1 * len(names)),
                         gridspec_kw={"width_ratios": [1, 1, 0.9]})
axes = np.atleast_2d(axes)
for i, n in enumerate(names):
    s = res["specimens"][n]
    d = np.load(root / "outputs" / "masks" / f"{n}.npz")
    shape = tuple(d["shape"]); vx = float(d["vx"])
    mask = np.unpackbits(d["mask"])[: np.prod(shape)].reshape(shape).astype(bool)
    sm = d["sm"]; path = d["path"]
    # 1) slice through the specimen's centre along the longest image axis
    z = int(np.median(np.nonzero(mask.any(axis=(1, 2)))[0]))
    sl = sm[z]
    ax = axes[i, 0]
    ax.imshow(sl, cmap="gray", vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99.8))
    ax.imshow(np.ma.masked_where(~segmentation.find_boundaries(mask[z]), mask[z]), cmap="autumn", alpha=1)
    ax.set_title(f"{n.replace('_', ' ')}\nslice z={z} with Otsu segmentation outline", fontsize=9)
    ax.axis("off")
    ax.plot([5, 5 + 200 / vx], [sl.shape[0] - 8] * 2, color="w", lw=3)
    ax.text(5, sl.shape[0] - 14, "200 µm", color="w", fontsize=8)
    # 2) 3D projection: MIP along the axis where the specimen is thinnest
    ext = [np.ptp(np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != k)))[0]) for k in range(3)]
    k = int(np.argmin(ext))
    mip = sm.max(axis=k); mproj = mask.any(axis=k)
    ax = axes[i, 1]
    ax.imshow(mip, cmap="gray", vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.9), aspect="equal")
    ax.contour(mproj, levels=[0.5], colors="orange", linewidths=0.7)
    if len(path):
        p2 = np.delete(path, k, axis=1)
        ax.plot(p2[:, 1], p2[:, 0], color="cyan", lw=1.5)
    ax.set_title(f"max-intensity projection (axis {'zyx'[k]}), mask outline,\nbody-core skeleton path (cyan)", fontsize=9)
    ax.axis("off")
    # 3) sweep: max Feret vs published
    ax = axes[i, 2]
    sens = s["sensitivity_otsu_x0.8_to_x1.2"]
    b = s["traits_baseline_otsu"]
    labels = ["max Feret\n3D", "centre-line\nlength", "max Feret\n2D (PC1-2)"]
    vals = [b["max_feret_mm"], b["skeleton_length_mm"], b["max_feret_2d_pc12_mm"]]
    lo = [sens["max_feret_mm"]["min"], sens["skeleton_length_mm"]["min"], vals[2]]
    hi = [sens["max_feret_mm"]["max"], sens["skeleton_length_mm"]["max"], vals[2]]
    x = np.arange(3)
    ax.bar(x, vals, color="#4C72B0", width=0.55, label="measured (Otsu)")
    ax.errorbar(x[:2], vals[:2], yerr=[np.subtract(vals, lo)[:2], np.subtract(hi, vals)[:2]],
                fmt="none", ecolor="k", capsize=4, label="threshold ×0.8–1.2")
    ax.axhline(s["published"]["max_dimension_mm"], color="#C44E52", ls="--",
               label=f"published max dimension {s['published']['max_dimension_mm']} mm")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("mm")
    err = s["comparison"]["max_dimension_vs_max_feret"]["rel_error_pct"]
    ax.set_title(f"max Feret vs published: {err:+.1f} %\nV={b['volume_mm3']:.4f} mm³, A={b['surface_area_mm2']:.2f} mm², "
                 f"sph={b['sphericity']:.2f}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Micro-CT trait extraction feasibility (Faulwetter et al. 2013 polychaete scans, CC0)", fontsize=11)
fig.tight_layout()
fig.savefig(root / "outputs" / "figure_traits.png", dpi=110)
print("written", root / "outputs" / "figure_traits.png")
