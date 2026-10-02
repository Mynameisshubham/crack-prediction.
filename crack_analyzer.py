#!/usr/bin/env python3
"""
crack_analyzer.py - image-based crack and deformation analysis for concrete surfaces

Modelled on what commercial inspection tools offer:
  * crack segmentation, then length / width / orientation per crack
  * width in millimetres via scale calibration (known-length object or mm-per-pixel)
  * severity grading and a report (JSON + CSV) with a colour-coded overlay
  * change detection between two inspections (crack growth)
  * deformation / strain mapping between a reference and a loaded image
    (a light-weight 2-D digital image correlation, DIC)

Commands
  analyze  image.jpg                      [--mm-per-px 0.25 | --calib x1,y1,x2,y2,length_mm]
  growth   before.jpg after.jpg           [--days 30] [scale options]
  deform   reference.jpg deformed.jpg     [scale options] [--gauge 15]

Install:  pip install numpy opencv-python scikit-image scipy matplotlib
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_SIDE = 1600  # working resolution cap (speed); measurements are rescaled to the original

# width (mm) -> grade. Indicative bands only; check the limit in your design code
# (e.g. IS 456 / ACI 224 give ~0.2-0.3 mm limits depending on exposure).
SEVERITY = [(0.1, "hairline"), (0.3, "fine"), (1.0, "moderate"), (float("inf"), "severe")]
SEV_COLOR = {"hairline": (0, 200, 0), "fine": (0, 255, 255), "moderate": (0, 140, 255),
             "severe": (0, 0, 255), "n/a": (255, 0, 255)}  # BGR


# ------------------------------------------------------------------ helpers
def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    return img


def downscale(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


def mm_per_px_original(args):
    """Scale of the ORIGINAL image in mm/pixel, or None if uncalibrated."""
    if getattr(args, "mm_per_px", None):
        return float(args.mm_per_px)
    if getattr(args, "calib", None):
        x1, y1, x2, y2, mm = [float(v) for v in args.calib.split(",")]
        d = math.hypot(x2 - x1, y2 - y1)
        if d <= 0:
            sys.exit("--calib points must be different")
        return mm / d
    return None


def severity_of(width_mm):
    if width_mm is None:
        return "n/a"
    for limit, name in SEVERITY:
        if width_mm < limit:
            return name
    return "severe"


def skeleton_length(S):
    """Length of a 1-px skeleton in pixels (diagonal steps count sqrt(2))."""
    S = S.astype(bool)
    h = np.count_nonzero(S[:, :-1] & S[:, 1:])
    v = np.count_nonzero(S[:-1, :] & S[1:, :])
    d1 = np.count_nonzero(S[:-1, :-1] & S[1:, 1:] & ~S[:-1, 1:] & ~S[1:, :-1])
    d2 = np.count_nonzero(S[:-1, 1:] & S[1:, :-1] & ~S[:-1, :-1] & ~S[1:, 1:])
    return float(h + v + math.sqrt(2) * (d1 + d2))


def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------- crack segmentation
def segment_cracks(gray, sens=0.5, sigmas=(1.0, 1.5, 2.0, 3.0), min_len=None):
    """Return (mask, skeleton). Classical pipeline: illumination flattening ->
    multi-scale ridge filter for dark lines -> hysteresis threshold -> length filter."""
    h, w = gray.shape
    if min_len is None:
        min_len = max(20, int(0.03 * max(h, w)))
    g = gray.astype(np.float32)
    flat = g - cv2.GaussianBlur(g, (0, 0), max(h, w) / 15.0)      # remove shading
    resp = sato(flat, sigmas=sigmas, black_ridges=True)           # dark thin structures

    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med))) * 1.4826 + 1e-9    # robust noise level
    k_hi = 14.0 - 8.0 * float(np.clip(sens, 0, 1))                # sens 0..1 -> 14..6 sigma
    k_lo = 0.5 * k_hi
    mask = apply_hysteresis_threshold(resp, med + k_lo * mad, med + k_hi * mad)

    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    mask = drop_small(mask, 20)

    skel = skeletonize(mask)
    lab = label(mask, connectivity=2)
    counts = np.bincount(lab[skel], minlength=lab.max() + 1)      # skeleton px per component
    keep = counts >= min_len
    keep[0] = False
    mask = keep[lab]
    skel = skel & mask
    mask = refine_mask(flat, skel, search=max(8, int(0.012 * max(h, w))))
    return mask, skel


def drop_small(mask, min_area):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], areas >= min_area])
    return keep[lab]


def refine_mask(flat, skel, search=10):
    """Half-maximum width refinement: around each skeleton point keep pixels that are darker than
    50% of the local crack depth (FWHM-style), so the mask hugs the real crack edges instead of
    the (wider) ridge-filter response. Needed for meaningful width measurement."""
    if not skel.any():
        return skel
    fs = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 0.8)
    depth = np.where(skel, -ndi.minimum_filter(fs, size=3), 0).astype(np.float32)
    dist_sk, idx = ndi.distance_transform_edt(~skel, return_indices=True)
    d_near = depth[idx[0], idx[1]]
    d_near = cv2.GaussianBlur(d_near, (0, 0), 4)                  # smooth depth along the crack
    refined = (-fs > 0.5 * d_near) & (dist_sk <= search) & (d_near > 0)
    return refined | skel


# ------------------------------------------------------------ crack analysis
def analyze_image(img, mm_per_px_orig, sens=0.5, min_len=None):
    work, s = downscale(img)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mm_work = (mm_per_px_orig / s) if mm_per_px_orig else None    # mm per working pixel

    mask, skel = segment_cracks(gray, sens=sens, min_len=min_len)
    dist = ndi.distance_transform_edt(mask)

    sk_lab = label(skel, connectivity=2)
    nbr = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junction = skel & (nbr >= 3)
    jl = label(junction, connectivity=2)
    branches = np.zeros(sk_lab.max() + 1, int)
    for jr in regionprops(jl):
        y, x = jr.coords[0]
        branches[sk_lab[y, x]] += 1

    cracks = []
    for r in regionprops(sk_lab):
        ys, xs = r.coords[:, 0], r.coords[:, 1]
        widths = 2.0 * dist[ys, xs]                               # full width at skeleton points
        length_px = skeleton_length(r.image)
        # orientation by PCA (0 deg = horizontal, 90 deg = vertical)
        pts = np.column_stack([xs, ys]).astype(float)
        pts -= pts.mean(0)
        if len(pts) > 2:
            _, vecs = np.linalg.eigh(np.cov(pts.T))
            vx, vy = vecs[:, -1]
            angle = math.degrees(math.atan2(-vy, vx)) % 180.0
        else:
            angle = 0.0
        if angle < 25 or angle > 155:
            direction = "horizontal"
        elif 65 < angle < 115:
            direction = "vertical"
        else:
            direction = "diagonal"
        wmax_px = float(np.percentile(widths, 95))                # robust "max" width
        wmean_px = float(widths.mean())
        c = {
            "id": int(r.label),
            "length_px": round(length_px, 1),
            "max_width_px": round(wmax_px, 2),
            "mean_width_px": round(wmean_px, 2),
            "orientation_deg": round(angle, 1),
            "direction": direction,
            "branch_points": int(branches[r.label]),
            "centroid_xy": [int(r.centroid[1]), int(r.centroid[0])],
        }
        if mm_work:
            c["length_mm"] = round(length_px * mm_work, 1)
            c["max_width_mm"] = round(wmax_px * mm_work, 3)
            c["mean_width_mm"] = round(wmean_px * mm_work, 3)
            c["severity"] = severity_of(c["max_width_mm"])
        else:
            c["severity"] = "n/a"
        cracks.append(c)

    cracks.sort(key=lambda c: -c["length_px"])
    total_len_px = sum(c["length_px"] for c in cracks)
    total_branch = sum(c["branch_points"] for c in cracks)
    if not cracks:
        pattern = "no cracks detected"
    elif len(cracks) >= 3 and total_branch >= 5:
        pattern = "network / map-type cracking (indicative)"
    elif total_branch >= 1:
        pattern = "branched crack(s)"
    else:
        pattern = "isolated crack(s)"

    summary = {
        "crack_count": len(cracks),
        "total_length_px": round(total_len_px, 1),
        "crack_area_percent": round(100.0 * mask.sum() / mask.size, 3),
        "pattern": pattern,
        "calibrated": bool(mm_work),
        "working_scale": round(s, 4),
    }
    if mm_work:
        summary["total_length_mm"] = round(total_len_px * mm_work, 1)
        summary["max_width_mm"] = round(max((c["max_width_mm"] for c in cracks), default=0.0), 3)
        order = ["hairline", "fine", "moderate", "severe"]
        summary["worst_severity"] = max((c["severity"] for c in cracks),
                                        key=order.index, default="none")
    else:
        summary["note"] = ("Not calibrated: widths/lengths are in pixels. Use --mm-per-px or "
                           "--calib x1,y1,x2,y2,length_mm (a scale bar lying in the crack plane).")

    # overlay
    overlay = work.copy()
    thick = cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sk_color = np.zeros_like(work)
    for c in cracks:
        sk_color[thick & (dilate_label(sk_lab, c["id"]))] = SEV_COLOR[c["severity"]]
    painted = sk_color.any(axis=2)
    overlay[painted] = sk_color[painted]
    for c in cracks[:30]:
        x, y = c["centroid_xy"]
        txt = f'#{c["id"]}'
        if "max_width_mm" in c:
            txt += f' {c["max_width_mm"]:.2f}mm'
        cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return summary, cracks, overlay, mask, skel, s


def dilate_label(sk_lab, cid):
    return cv2.dilate((sk_lab == cid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def cmd_analyze(args):
    os.makedirs(args.out, exist_ok=True)
    img = load(args.image)
    summary, cracks, overlay, mask, _, _ = analyze_image(img, mm_per_px_original(args), args.sens)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.png"), overlay)
    cv2.imwrite(os.path.join(args.out, f"{stem}_mask.png"), mask.astype(np.uint8) * 255)
    with open(os.path.join(args.out, f"{stem}_report.json"), "w") as f:
        json.dump({"summary": summary, "cracks": cracks}, f, indent=2, default=to_jsonable)
    if cracks:
        with open(os.path.join(args.out, f"{stem}_cracks.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(cracks[0].keys()))
            wr.writeheader()
            wr.writerows(cracks)
    print(json.dumps(summary, indent=2, default=to_jsonable))
    for c in cracks[:10]:
        print(c)
    print(f"\nSaved results in: {args.out}/")


# ---------------------------------------------------------------- alignment
def align_to(ref_gray, cur_bgr):
    """Warp cur onto ref using ORB + homography (removes camera shake / small viewpoint change)."""
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(cur_gray, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        print("WARNING: too few features to align images; assuming a fixed camera.")
        return cur_bgr, 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:1500]
    if len(matches) < 12:
        print("WARNING: too few matches to align images; assuming a fixed camera.")
        return cur_bgr, 0.0
    src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return cur_bgr, 0.0
    warped = cv2.warpPerspective(cur_bgr, H, (ref_gray.shape[1], ref_gray.shape[0]))
    return warped, float(inl.sum()) / len(inl)


# ------------------------------------------------------- crack growth (change)
def cmd_growth(args):
    os.makedirs(args.out, exist_ok=True)
    before = load(args.before)
    after = load(args.after)
    before, s = downscale(before)
    after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    after, inl = align_to(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), after)

    scale = mm_per_px_original(args)
    mm_work = (scale / s) if scale else None   # mm per pixel of the (already downscaled) working images
    sb, cb, _, mask_b, skel_b, _ = analyze_image(before, mm_work, args.sens)
    sa, ca, _, mask_a, skel_a, _ = analyze_image(after, mm_work, args.sens)

    new = skel_a & ~cv2.dilate(mask_b.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    new_len = skeleton_length(new)
    res = {
        "alignment_inlier_ratio": round(inl, 3),
        "before": {k: sb.get(k) for k in ("crack_count", "total_length_px", "max_width_mm", "total_length_mm")},
        "after": {k: sa.get(k) for k in ("crack_count", "total_length_px", "max_width_mm", "total_length_mm")},
        "new_crack_length_px": round(new_len, 1),
        "length_change_px": round(sa["total_length_px"] - sb["total_length_px"], 1),
    }
    if scale:
        res["new_crack_length_mm"] = round(new_len * mm_work, 1)
        res["width_change_mm"] = round(sa["max_width_mm"] - sb["max_width_mm"], 3)
        if args.days:
            res["width_growth_mm_per_day"] = round(res["width_change_mm"] / args.days, 4)
            res["length_growth_mm_per_day"] = round(
                (sa["total_length_mm"] - sb["total_length_mm"]) / args.days, 3)
    out = after.copy()
    out[skel_b] = (255, 0, 0)       # before = blue
    out[skel_a & ~new] = (0, 200, 0)  # persisting = green
    out[new] = (0, 0, 255)          # new / propagated = red
    cv2.imwrite(os.path.join(args.out, "growth_overlay.png"), out)
    with open(os.path.join(args.out, "growth_report.json"), "w") as f:
        json.dump(res, f, indent=2, default=to_jsonable)
    print(json.dumps(res, indent=2, default=to_jsonable))
    print("Overlay: blue = before, green = persisting, red = new/propagated crack.")


# -------------------------------------------- deformation (light-weight 2-D DIC)
def cmd_deform(args):
    os.makedirs(args.out, exist_ok=True)
    ref = load(args.reference)
    cur = load(args.deformed)
    ref, s = downscale(ref)
    cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_al, inl = align_to(ref_g, cur)                 # remove rigid camera motion
    cur_g = cv2.cvtColor(cur_al, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(ref_g, cur_g, None, 0.5, 5, 41, 5, 7, 1.5, 0)
    g = float(args.gauge)
    u = cv2.GaussianBlur(flow[..., 0], (0, 0), g / 2)
    v = cv2.GaussianBlur(flow[..., 1], (0, 0), g / 2)
    mag = np.hypot(u, v)

    # strain from displacement gradients (small-strain, in-plane)
    exx = np.gradient(u, axis=1)
    eyy = np.gradient(v, axis=0)
    exy = 0.5 * (np.gradient(u, axis=0) + np.gradient(v, axis=1))
    e1 = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)  # max principal

    # mask low-texture areas and borders (displacement is unreliable there)
    gf = ref_g.astype(np.float32)
    mu = cv2.blur(gf, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(gf * gf, (15, 15)) - mu * mu, 0))
    valid = sd > 3.0
    m = 20
    valid[:m, :] = valid[-m:, :] = False
    valid[:, :m] = valid[:, -m:] = False
    for arr in (mag, e1):
        arr[~valid] = np.nan

    scale = mm_per_px_original(args)
    mm_work = (scale / s) if scale else None
    unit = "mm" if mm_work else "px"
    k = mm_work if mm_work else 1.0

    med = np.nanmedian(mag)
    mad = np.nanmedian(np.abs(mag - med)) * 1.4826 + 1e-9
    thr = max(med + 5 * mad, args.disp_thr_px or 0.0)
    hot = np.nan_to_num(mag, nan=0.0) > thr
    iy, ix = np.unravel_index(np.nanargmax(mag), mag.shape)

    res = {
        "alignment_inlier_ratio": round(inl, 3),
        "valid_area_percent": round(100.0 * valid.mean(), 1),
        f"max_displacement_{unit}": round(float(np.nanmax(mag)) * k, 3),
        "max_displacement_location_xy": [int(ix), int(iy)],
        f"median_displacement_{unit}": round(float(med) * k, 3),
        f"noise_floor_{unit}": round(float(mad) * k, 3),
        "hotspot_threshold_px": round(float(thr), 3),
        "hotspot_area_percent": round(100.0 * hot.sum() / hot.size, 2),
        "max_principal_strain_percent": round(float(np.nanmax(e1)) * 100, 4),
        "calibrated": bool(mm_work),
    }
    if float(np.nanmax(mag)) < 3 * mad + med:
        res["verdict"] = "no displacement above the measurement noise floor"
    else:
        res["verdict"] = "localised movement/deformation detected (see hotspots)"

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    bg = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    im0 = ax[0].imshow(bg)
    h0 = ax[0].imshow(mag * k, cmap="jet", alpha=0.55)
    ax[0].set_title(f"Displacement magnitude [{unit}]")
    plt.colorbar(h0, ax=ax[0], fraction=0.046)
    h1 = ax[1].imshow(e1 * 100, cmap="inferno")
    ax[1].set_title("Max principal strain [%]")
    plt.colorbar(h1, ax=ax[1], fraction=0.046)
    ax[2].imshow(bg)
    ax[2].imshow(np.ma.masked_where(~hot, hot), cmap="autumn", alpha=0.6)
    st = max(8, min(ref.shape[:2]) // 25)
    yy, xx = np.mgrid[st // 2:ref.shape[0]:st, st // 2:ref.shape[1]:st]
    sel = valid[yy, xx]
    ax[2].quiver(xx[sel], yy[sel], u[yy, xx][sel], v[yy, xx][sel], color="cyan",
                 angles="xy", scale_units="xy", scale=0.25, width=0.002)
    ax[2].set_title("Hotspots + displacement vectors (x4)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "deformation_map.png"), dpi=150)
    with open(os.path.join(args.out, "deformation_report.json"), "w") as f:
        json.dump(res, f, indent=2, default=to_jsonable)
    print(json.dumps(res, indent=2, default=to_jsonable))
    print(f"Saved results in: {args.out}/")


# --------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--mm-per-px", type=float, help="scale of the ORIGINAL image (mm per pixel)")
        sp.add_argument("--calib", help="x1,y1,x2,y2,length_mm of a known-length object (original pixels)")
        sp.add_argument("--out", default="results", help="output folder")

    a = sub.add_parser("analyze", help="detect + measure cracks in one image")
    a.add_argument("image")
    a.add_argument("--sens", type=float, default=0.5, help="detection sensitivity 0..1 (higher = more cracks)")
    common(a)
    a.set_defaults(fn=cmd_analyze)

    g = sub.add_parser("growth", help="compare cracks between two inspections")
    g.add_argument("before")
    g.add_argument("after")
    g.add_argument("--days", type=float, help="days between the two photos (gives growth rate)")
    g.add_argument("--sens", type=float, default=0.5)
    common(g)
    g.set_defaults(fn=cmd_growth)

    d = sub.add_parser("deform", help="deformation / strain map between two images")
    d.add_argument("reference")
    d.add_argument("deformed")
    d.add_argument("--gauge", type=float, default=15, help="smoothing / gauge length in pixels")
    d.add_argument("--disp-thr-px", type=float, default=None, help="minimum displacement to flag (pixels)")
    common(d)
    d.set_defaults(fn=cmd_deform)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
