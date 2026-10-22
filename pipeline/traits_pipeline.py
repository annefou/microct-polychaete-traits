"""Classical (non-AI) micro-CT trait pipeline for polychaete slice stacks.

Usage: python -I pipeline/traits_pipeline.py ROOT
Reads ROOT/data/<specimen>/stack/*.bmp + SkyScan *.log (from fetch_faulwetter2013.py),
writes ROOT/outputs/results.json, ROOT/outputs/masks/<specimen>_mask.npz (packbits) and
per-specimen arrays used by make_figure.py.

Steps per specimen
  1. Load raw reconstruction values (palette *indices*; some BMPs carry an inverted
     palette, so converting to grey would invert densities). Voxel size from the log:
     xy = "Pixel Size (um)", z = xy * "Section to Section Step".
  2. Resample to isotropic voxels (xy block-mean by the z step).
  3. Denoise: 3D median (r=1) + Gaussian (sigma=1 voxel).
  4. Segment: Otsu threshold -> closing (ball r=2) -> keep largest 3D component ->
     fill holes (3D). Sensitivity: threshold x {0.8, 0.9, 1.1, 1.2}, plus Li and
     Triangle automatic thresholds as alternative methods.
  5. Traits: volume, surface area (marching cubes), max Feret diameter (convex hull),
     PCA axis extents, sphericity, elongation, body length along the skeleton of the
     "body core" (mask opened with a ball to drop chaetae/parapodia), body width
     (2 x distance-transform along that skeleton), and a structure-specific trait:
     volume of the high-density class (jaws/aciculae/chaetae: top class of a
     3-class multi-Otsu inside the specimen).
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import filters, measure, morphology

# Published reference values: Faulwetter et al. 2013, ZooKeys 263, Table 1 ("Size (mm)":
# "maximum dimension (in any direction) of the scanned part of the specimen") and Results
# text for Eunice sp. juveniles: "the small size of the animal (width ca 0.3 mm)".
PUBLISHED = {
    "Eunice_sp_I2E": {"max_dimension_mm": 1.8, "width_mm_approx": 0.3, "voxel_um": 2.162,
                      "what": "juvenile Eunice sp., whole animal, iodine-stained, in ethanol"},
    "Eunice_sp_PTA": {"max_dimension_mm": 2.2, "width_mm_approx": 0.3, "voxel_um": 2.298,
                      "what": "juvenile Eunice sp., whole animal, PTA-stained, in ethanol"},
    "Eunice_roussaei": {"max_dimension_mm": 2.9, "voxel_um": 2.878,
                        "what": "Eunice roussaei, dissected mid-body parapodium, HMDS-dried, in air"},
}
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]


def read_log(stack_dir):
    log = next(Path(stack_dir).glob("*.log")).read_text(errors="ignore")
    px = float(re.findall(r"^Pixel Size \(um\)=\s*([\d.]+)", log, re.M)[-1])
    step = int(re.search(r"Section to Section Step=\s*(\d+)", log).group(1))
    return px, step


def load_stack(stack_dir):
    files = sorted(glob.glob(str(Path(stack_dir) / "*.bmp")))
    first = Image.open(files[0])
    inverted = False
    if first.mode == "P":
        pal = np.array(first.getpalette()[:768]).reshape(-1, 3)[:, 0]
        inverted = pal[0] > pal[-1]
    vol = np.stack([np.asarray(Image.open(f)) for f in files]).astype(np.uint8)  # raw indices
    return vol, inverted, len(files)


def to_isotropic(vol, k):
    if k == 1:
        return vol.astype(np.float32)
    z, y, x = vol.shape
    v = vol[:, : y // k * k, : x // k * k].astype(np.float32)
    return v.reshape(z, y // k, k, x // k, k).mean(axis=(2, 4))


def segment(sm, thr):
    m = sm > thr
    m = ndi.binary_closing(m, morphology.ball(2), border_value=0)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    m = lab == sizes.argmax()
    # Body cavities (gut, coelom) open to the outside via mouth/anus defeat 3D hole filling,
    # so fill holes slice-wise along each axis and take the union, then 3D fill.
    filled = m.copy()
    for ax in range(3):
        mv = np.moveaxis(m, ax, 0)
        filled |= np.moveaxis(np.stack([ndi.binary_fill_holes(s_) for s_ in mv]), 0, ax)
    return ndi.binary_fill_holes(filled)


def longest_skeleton_path(skel, vx):
    pts = np.argwhere(skel)
    if len(pts) < 2:
        return 0.0, pts
    idx = -np.ones(skel.shape, dtype=np.int64); idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols, w = [], [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                q = pts + (dz, dy, dx)
                ok = np.all((q >= 0) & (q < skel.shape), axis=1)
                j = np.full(len(pts), -1); j[ok] = idx[tuple(q[ok].T)]
                sel = j >= 0
                rows += list(np.nonzero(sel)[0]); cols += list(j[sel])
                w += [vx * np.sqrt(dz * dz + dy * dy + dx * dx)] * int(sel.sum())
    g = coo_matrix((w, (rows, cols)), shape=(len(pts), len(pts))).tocsr()
    g = g + g.T
    ncomp, labels = connected_components(g, directed=False)
    big = np.bincount(labels).argmax()
    nodes = np.nonzero(labels == big)[0]
    d0 = dijkstra(g, indices=nodes[0])
    a = nodes[np.argmax(np.where(np.isfinite(d0[nodes]), d0[nodes], -1))]
    da, pred = dijkstra(g, indices=a, return_predecessors=True)
    b = nodes[np.argmax(np.where(np.isfinite(da[nodes]), da[nodes], -1))]
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return float(da[b]), pts[path]


def traits(mask, sm, vx, core_radius_um):
    t = {}
    nvox = int(mask.sum())
    t["volume_mm3"] = nvox * (vx / 1000) ** 3
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), 0.5, spacing=(vx,) * 3)
    t["surface_area_mm2"] = measure.mesh_surface_area(verts, faces) / 1e6
    hull = ConvexHull(verts)
    hv = verts[hull.vertices]
    t["max_feret_mm"] = float(pdist(hv).max()) / 1000
    t["convex_hull_volume_mm3"] = hull.volume / 1e9
    pts = np.argwhere(mask).astype(np.float64) * vx
    c = pts - pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(c[:: max(1, len(c) // 200000)].T))
    order = evals.argsort()[::-1]; evecs = evecs[:, order]; evals = evals[order]
    proj = c @ evecs
    ext = proj.max(0) - proj.min(0)
    t["pca_extent_mm"] = [float(e) / 1000 for e in ext]
    # max dimension in the PC1-PC2 plane (= what a 2D dorsal/lateral view measurement would give)
    hv_c = (hv - hv.mean(0)) @ evecs[:, :2]
    h2 = ConvexHull(hv_c)
    t["max_feret_2d_pc12_mm"] = float(pdist(hv_c[h2.vertices]).max()) / 1000
    t["elongation_pc1_pc2"] = float(np.sqrt(evals[0] / evals[1]))
    V, A = t["volume_mm3"], t["surface_area_mm2"]
    t["sphericity"] = float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / A)
    # body core: remove thin appendages (chaetae, cirri, parapodia) before skeletonising
    r = max(1, int(round(core_radius_um / vx)))
    core = ndi.binary_opening(mask, morphology.ball(r))
    lab, n = ndi.label(core)
    if n:
        s = np.bincount(lab.ravel()); s[0] = 0; core = lab == s.argmax()
    skel = morphology.skeletonize(core)
    L, path = longest_skeleton_path(skel, vx)
    t["skeleton_length_raw_mm"] = L / 1000
    # smooth the centre-line (moving average over ~60 um) to remove voxel zig-zag
    if len(path) > 3:
        w_ = max(3, int(round(60 / vx)) | 1)
        sp = np.stack([ndi.uniform_filter1d(path[:, i].astype(float), w_, mode="nearest") for i in range(3)], 1)
        t["skeleton_length_mm"] = float(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum() * vx / 1000)
    else:
        t["skeleton_length_mm"] = L / 1000
    dt = ndi.distance_transform_edt(core) * vx
    if len(path):
        w = 2 * dt[tuple(path.T)] / 1000
        n_ = len(w); mid = w[n_ // 4: 3 * n_ // 4] if n_ > 8 else w
        t["width_core_median_mm"] = float(np.median(mid))
        t["width_core_max_mm"] = float(w.max())
    t["core_opening_radius_um"] = r * vx
    # structure-specific: high-density class (jaws / aciculae / chaetae)
    vals = sm[mask]
    try:
        th = filters.threshold_multiotsu(vals, classes=3)
        dense = int((vals > th[-1]).sum())
        t["dense_class_volume_mm3"] = dense * (vx / 1000) ** 3
        t["dense_class_fraction"] = dense / nvox
    except ValueError:
        pass
    return t, core, path


def main():
    root = Path(sys.argv[1]).resolve()
    out = root / "outputs"; (out / "masks").mkdir(parents=True, exist_ok=True)
    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": "Faulwetter et al. 2013 ZooKeys 263 doi:10.3897/zookeys.263.4261; data doi:10.5061/dryad.84m54 (Zenodo 4959912, CC0)",
               "specimens": {}}
    for name, pub in PUBLISHED.items():
        sd = root / "data" / name / "stack"
        if not sd.exists():
            continue
        t0 = time.time()
        px, step = read_log(sd)
        raw, inverted, nfiles = load_stack(sd)
        vol = to_isotropic(raw, step)
        vx = px * step  # isotropic voxel edge (um)
        sm = ndi.gaussian_filter(ndi.median_filter(vol, size=3), 1.0)
        # Otsu on a sub-sample of the smoothed volume
        sample = sm.ravel()[:: 7]
        thr = {"otsu": float(filters.threshold_otsu(sample)), "li": float(filters.threshold_li(sample)),
               "triangle": float(filters.threshold_triangle(sample))}
        runs = {}
        core_r = 15.0  # um
        for f in FACTORS:
            m = segment(sm, thr["otsu"] * f)
            tt, core, path = traits(m, sm, vx, core_r)
            runs[f"otsu_x{f}"] = tt
            if f == 1.0:
                ref_mask, ref_core, ref_path = m, core, path
        for meth in ("li", "triangle"):
            m = segment(sm, thr[meth])
            runs[meth], _, _ = traits(m, sm, vx, core_r)
        base = runs["otsu_x1.0"]
        keys = [k for k, v in base.items() if isinstance(v, float)]
        sens = {}
        for k in keys:
            vals = np.array([runs[f"otsu_x{f}"][k] for f in FACTORS if k in runs[f"otsu_x{f}"]])
            sens[k] = {"min": float(vals.min()), "max": float(vals.max()),
                       "range_pct_of_base": float(100 * (vals.max() - vals.min()) / base[k]) if base[k] else None}
        comp = {"max_dimension_vs_max_feret": {
            "published_mm": pub["max_dimension_mm"], "measured_mm": base["max_feret_mm"],
            "rel_error_pct": 100 * (base["max_feret_mm"] - pub["max_dimension_mm"]) / pub["max_dimension_mm"],
            "rel_error_range_pct_over_threshold_sweep": [
                100 * (sens["max_feret_mm"]["min"] - pub["max_dimension_mm"]) / pub["max_dimension_mm"],
                100 * (sens["max_feret_mm"]["max"] - pub["max_dimension_mm"]) / pub["max_dimension_mm"]]}}
        if "width_mm_approx" in pub and "width_core_median_mm" in base:
            comp["width_vs_core_median_width"] = {
                "published_mm": pub["width_mm_approx"], "measured_mm": base["width_core_median_mm"],
                "rel_error_pct": 100 * (base["width_core_median_mm"] - pub["width_mm_approx"]) / pub["width_mm_approx"],
                "note": "published value is approximate ('ca 0.3 mm'); measured = median 2x inscribed radius along mid half of body skeleton"}
        results["specimens"][name] = {
            "description": pub["what"], "n_slices": nfiles, "raw_shape_zyx": list(raw.shape),
            "voxel_xy_um_from_log": px, "section_step": step, "voxel_z_um": px * step,
            "voxel_isotropic_um_used": vx, "palette_inverted_in_bmp": bool(inverted),
            "thresholds": thr, "published": pub, "traits_baseline_otsu": base,
            "sensitivity_otsu_x0.8_to_x1.2": sens, "alt_methods": {k: runs[k] for k in ("li", "triangle")},
            "comparison": comp, "runtime_s": round(time.time() - t0, 1)}
        np.savez_compressed(out / "masks" / f"{name}.npz", mask=np.packbits(ref_mask), core=np.packbits(ref_core),
                            shape=np.array(ref_mask.shape), path=ref_path, vx=vx,
                            sm=np.clip(sm, 0, 255).astype(np.uint8), thr=thr["otsu"])
        print(name, json.dumps(comp), f"{time.time() - t0:.0f}s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
