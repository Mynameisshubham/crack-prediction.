#!/usr/bin/env python3
"""
pavement_predictor.py - image-based + material-property-based crack PREDICTION for pavements
==========================================================================================

What it does
  1. Reads a pavement photo and measures the cracking that exists NOW (crack_analyzer.py).
  2. Classifies the likely failure mode from the crack geometry (transverse, longitudinal,
     corner/diagonal, alligator/fatigue, block, map/craze, edge, reflective ...).
  3. Takes the fixed pavement properties you assign (concrete grade, steel grade, slab /
     layer thicknesses, subgrade CBR, binder grade, traffic, climate, age ...).
  4. Computes load-fatigue damage (rigid: Westergaard edge stress + IRC:58 fatigue law;
     flexible: AASHTO-93 structural capacity) and scores the other failure cases
     (thermal curling, joint spacing, support loss, steel adequacy, ageing, low-temperature
     cracking, weak subgrade, reflective cracking ...).
  5. Calibrates the model to the cracking seen in the photo and forecasts how the cracking
     will develop year by year, with a sensitivity band and "year to reach" thresholds.

IMPORTANT - this is a SCREENING / RESEARCH tool, not a design check
  * Formulas are simplified; several coefficients are engineering heuristics (marked in the
    report) and must be calibrated with your own field data. For design use IRC:58 / IRC:37
    (IITRIGID / IITPAVE) or your agency's procedure.
  * One photo shows a small patch. Use a representative, scaled (--mm-per-px) photo.

Usage
  python pavement_predictor.py make-spec --type rigid > spec.json      (edit the numbers)
  python pavement_predictor.py predict road.jpg --spec spec.json --mm-per-px 0.5 --out results
  python pavement_predictor.py predict road.jpg --spec spec.json --set age_years=8 --set cvpd_now=2500

Install:  pip install numpy opencv-python scikit-image scipy matplotlib   (needs crack_analyzer.py alongside)
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crack_analyzer as ca

# ============================================================================
# 1.  FIXED-PROPERTY DATABASES  (the "grades" you assign)
# ============================================================================
CONCRETE_GRADES = {"M25": 25, "M30": 30, "M35": 35, "M40": 40, "M45": 45, "M50": 50}   # fck, MPa
STEEL_GRADES = {"Fe250": 250, "Fe415": 415, "Fe500": 500, "Fe500D": 500, "Fe550": 550, "Fe600": 600}  # fy, MPa
# binder grade -> (structural-coefficient factor, stiffness index 0..1 (higher = stiffer/more brittle))
BINDER_GRADES = {"VG-10": (0.95, 0.20), "VG-20": (0.97, 0.35), "VG-30": (1.00, 0.55),
                 "VG-40": (1.03, 0.75), "PMB-40": (1.07, 0.45), "CRMB-60": (1.08, 0.40)}
# base type -> multiplier on subgrade k-value (rigid pavements)  [approximate]
BASE_K_FACTOR = {"GSB": 1.0, "WMM": 1.3, "CTB": 1.8, "DLC": 2.2}

# default axle-load spectrum for rigid fatigue: load as a multiple of the legal axle, share of axles
AXLE_SPECTRUM = [(0.6, 0.30), (0.8, 0.30), (1.0, 0.25), (1.2, 0.12), (1.5, 0.03)]

DEFAULT_RIGID = dict(
    pavement_type="rigid",
    concrete_grade="M40", steel_grade="Fe500",
    slab_thickness_mm=300, joint_spacing_m=4.5,
    steel_dia_mm=0, steel_spacing_mm=200,          # distributed steel; dia 0 = plain (unreinforced) slab
    has_dowels=True, has_tie_bars=True,
    subgrade_cbr=8, base_type="DLC", k_override=None,   # k_override in MPa/m
    legal_axle_kn=100, tyre_pressure_mpa=0.8,
    temp_diff_c=16, thermal_participation=0.2, load_safety_factor=1.2, load_transfer_factor=0.8,
    age_years=5, cvpd_now=1500, growth_pct=5, lane_dist_factor=0.5, axles_per_cv=2.0,
    failure_density=None,                           # crack length density at "failed" state (m/m2); None = default
)
DEFAULT_FLEXIBLE = dict(
    pavement_type="flexible",
    binder_grade="VG-30",
    ac_thickness_mm=150, base_thickness_mm=250, subbase_thickness_mm=200,
    subgrade_cbr=6, drainage_coeff=1.0,
    min_pavement_temp_c=8, overlay_on_cracked_base=False,
    reliability_zr=-1.282, overall_std_dev=0.45, terminal_psi=2.5, initial_psi=4.2,
    age_years=6, cvpd_now=800, growth_pct=5, lane_dist_factor=0.5, vdf=3.5,
    failure_density=None,
)

DEFAULT_FAIL_DENSITY = {"rigid": 1.0, "flexible": 3.0}      # m of crack per m2 at "failed" condition (assumption)
DEFAULT_FAIL_AREA_PCT = 6.0                                  # crack area % at failure when image is not scaled
B_EXP = 1.98                                                 # shape exponent of the damage -> cracking curve
THRESHOLDS = [(0.25, "plan preventive maintenance (sealing / minor repair)"),
              (0.50, "major rehabilitation (overlay / partial-depth repair)"),
              (0.80, "failure - reconstruction / full-depth replacement")]

ASSUMPTIONS = [
    "Rigid: Westergaard edge-load stress, Bradbury warping coefficient, IRC:58 fatigue relation; cracking curve "
    "CRK = 1/(1+FD^-1.98) (MEPDG JPCP form, uncalibrated).",
    "Flexible: AASHTO-93 structural-number capacity (a1=0.44 x binder factor, a2=0.14, a3=0.11); CBR -> Mr by 1500*CBR "
    "(CBR<=10).",
    "Environmental / ageing damage and the 'other failure case' scores are engineering heuristics, not code formulas.",
    "Crack length density at failure (rigid 1.0, flexible 3.0 m/m2; 6% area if unscaled) is an adjustable assumption.",
    "The model is shifted to match the photographed cracking (local calibration); the +/-2x band is a sensitivity range, "
    "not a statistical confidence interval.",
    "A single photo covers a small patch - use a representative, scaled photo.",
]


# ============================================================================
# 2.  RIGID PAVEMENT ENGINE
# ============================================================================
def k_value(spec):
    """Modulus of subgrade reaction, MPa/m (approximate; prefer plate-load test or IRC:58 table)."""
    if spec.get("k_override"):
        return float(spec["k_override"])
    k_sub = 20.0 * math.sqrt(max(float(spec["subgrade_cbr"]), 1.0))
    return k_sub * BASE_K_FACTOR.get(spec["base_type"], 1.0)


def westergaard_edge(P_N, h, E, k_mm, mu=0.15, p=0.8):
    """Edge load stress (MPa) for a circular load P (N) on slab thickness h (mm); also returns l (mm)."""
    a = math.sqrt(P_N / (math.pi * p))
    l = (E * h ** 3 / (12 * (1 - mu ** 2) * k_mm)) ** 0.25
    b = a if a >= 1.724 * h else math.sqrt(1.6 * a * a + h * h) - 0.675 * h
    s = 0.803 * P_N / h ** 2 * (4 * math.log10(l / b) + 0.666 * a / l - 0.034)
    return max(s, 0.0), l


def bradbury_C(L_mm, l_mm):
    lam = L_mm / (l_mm * math.sqrt(8))
    num = 2 * math.cos(lam) * math.cosh(lam)
    den = math.sin(2 * lam) + math.sinh(2 * lam)
    return float(np.clip(1 - (num / den) * (math.tan(lam) + math.tanh(lam)), 0.0, 1.0))


def fatigue_life(sr):
    """IRC:58 fatigue relation: allowable repetitions at stress ratio SR (inf below 0.45)."""
    if sr <= 0.45:
        return math.inf
    if sr < 0.55:
        return (4.2577 / (sr - 0.4325)) ** 3.268
    return 10 ** ((0.9718 - sr) / 0.0828)


def traffic_per_year(spec, years, per_cv):
    """Axle (or ESAL) repetitions in each year 1..years, back/forward-cast from today's CVPD."""
    g = spec["growth_pct"] / 100.0
    age = spec["age_years"]
    j = np.arange(1, years + 1)
    cv_day = spec["cvpd_now"] * (1 + g) ** (j - max(age, 1))
    return cv_day * 365.0 * spec["lane_dist_factor"] * per_cv


def rigid_engine(spec, years):
    fck = CONCRETE_GRADES[spec["concrete_grade"]]
    fy = STEEL_GRADES[spec["steel_grade"]]
    h = float(spec["slab_thickness_mm"])
    L = float(spec["joint_spacing_m"]) * 1000.0
    E = 5000.0 * math.sqrt(fck)                      # IS 456
    MR = 0.7 * math.sqrt(fck)                        # flexural strength, IS 456
    k = k_value(spec)
    k_mm = k / 1000.0
    ltf = spec["load_transfer_factor"] if (spec["has_dowels"] or spec["has_tie_bars"]) else 1.0

    _, l = westergaard_edge(spec["legal_axle_kn"] * 500.0, h, E, k_mm, p=spec["tyre_pressure_mpa"])
    C = bradbury_C(L, l)
    sig_warp = E * 10e-6 * spec["temp_diff_c"] / 2.0 * C       # alpha = 10e-6 /degC

    bins = []
    for mult, share in AXLE_SPECTRUM:
        P = spec["legal_axle_kn"] * mult * 500.0               # half axle, N
        s_load, _ = westergaard_edge(P, h, E, k_mm, p=spec["tyre_pressure_mpa"])
        s_load *= ltf * spec["load_safety_factor"]
        sr = (s_load + spec["thermal_participation"] * sig_warp) / MR
        bins.append(dict(axle_multiple=mult, share=share, load_stress_mpa=round(s_load, 3),
                         stress_ratio=round(sr, 3), allowable_reps=fatigue_life(sr)))

    n = traffic_per_year(spec, years, spec["axles_per_cv"])
    dmg_year = np.zeros(years)
    for b in bins:
        if math.isfinite(b["allowable_reps"]):
            dmg_year += b["share"] * n / b["allowable_reps"]
    fd_load = np.concatenate([[0.0], np.cumsum(dmg_year)])

    # reference (legal axle) stresses used by the failure-case scoring
    s_ref, _ = westergaard_edge(spec["legal_axle_kn"] * 500.0, h, E, k_mm, p=spec["tyre_pressure_mpa"])
    s_ref *= ltf * spec["load_safety_factor"]
    sr_comb = (s_ref + sig_warp) / MR

    # steel (distributed reinforcement) adequacy: As_req = F*W*L/(2*fs)
    W = h / 1000.0 * 25000.0                                   # slab weight, N/m2
    fs = 0.67 * fy
    as_req = 1.5 * W * (L / 1000.0) / (2.0 * fs)               # mm2 per m width
    as_prov = (math.pi * spec["steel_dia_mm"] ** 2 / 4.0) * 1000.0 / spec["steel_spacing_mm"] \
        if spec["steel_dia_mm"] > 0 else 0.0

    props = dict(E_mpa=round(E), flexural_strength_mpa=round(MR, 2), fy_mpa=fy, k_mpa_per_m=round(k, 1),
                 radius_rel_stiffness_mm=round(l), bradbury_C=round(C, 3),
                 warping_stress_mpa=round(sig_warp, 2), legal_axle_edge_stress_mpa=round(s_ref, 2),
                 combined_stress_ratio=round(sr_comb, 3), steel_required_mm2_per_m=round(as_req, 1),
                 steel_provided_mm2_per_m=round(as_prov, 1), fatigue_bins=bins)

    cases = []
    cases.append(dict(case="Load fatigue cracking (transverse / longitudinal / corner)", category="load",
                      score=None, basis="cumulative fatigue damage from the axle spectrum (IRC:58 law)",
                      advice="Control overloading; thicker slab or higher grade concrete; improve joint load transfer."))
    cases.append(dict(case="Thermal curling / warping cracking", category="environment",
                      score=float(np.clip((sr_comb - 0.55) / 0.45, 0, 1)),
                      basis=f"edge load + warping stress / flexural strength = {sr_comb:.2f}",
                      advice="Shorter joint spacing, thicker slab, timely joint sawing, curing."))
    ratio_L = L / h
    cases.append(dict(case="Shrinkage / mid-panel cracking (joint spacing)", category="environment",
                      score=float(np.clip((ratio_L - 15.0) / 9.0, 0, 1)),
                      basis=f"joint spacing / thickness = {ratio_L:.1f} (limit about 24)",
                      advice="Keep joint spacing <= 24 x thickness; saw joints within 8-12 h."))
    cases.append(dict(case="Loss of subgrade support / pumping", category="environment",
                      score=float(np.clip((40.0 - k) / 40.0, 0, 1)),
                      basis=f"effective k = {k:.0f} MPa/m",
                      advice="Improve drainage, stabilised/DLC base, undersealing of voids."))
    if spec["steel_dia_mm"] > 0:
        steel_ratio = as_prov / as_req if as_req > 0 else 9.9
        cases.append(dict(case="Steel inadequacy (crack-width control)", category="environment",
                          score=float(np.clip(1.0 - steel_ratio, 0, 1)),
                          basis=f"{spec['steel_grade']} bars: provided {as_prov:.0f} vs required {as_req:.0f} mm2/m "
                                f"(ratio {steel_ratio:.2f})",
                          advice="Increase bar area or grade to hold cracks tight; check tie-bar/dowel design."))
    else:
        steel_ratio = 1.0
        cases.append(dict(case="Plain (unreinforced) slab", category="info", score=0.0,
                          basis=f"no distributed steel; relies on joint spacing (required if reinforced: {as_req:.0f} mm2/m)",
                          advice="Cracks between joints are not held tight - keep joint spacing short."))
    cases.append(dict(case="Concrete strength shortfall", category="environment",
                      score=float(np.clip((40.0 - fck) / 15.0, 0, 1)),
                      basis=f"{spec['concrete_grade']} (PQC is usually M40, flexural 4.5 MPa)",
                      advice="Use PQC of M40 or higher for heavy traffic."))
    jt = 0.0
    msg = []
    if not spec["has_dowels"]:
        jt += 0.5; msg.append("no dowels at transverse joints -> faulting / corner cracks")
    if not spec["has_tie_bars"]:
        jt += 0.4; msg.append("no tie bars at longitudinal joint -> joint opening / longitudinal cracks")
    cases.append(dict(case="Joint load-transfer deficiency", category="environment", score=min(jt, 1.0),
                      basis="; ".join(msg) if msg else "dowels and tie bars provided",
                      advice="Provide dowels (transverse) and tie bars (longitudinal)."))
    return fd_load, props, cases, float(np.clip(steel_ratio, 0.5, 1.5))


# ============================================================================
# 3.  FLEXIBLE PAVEMENT ENGINE
# ============================================================================
def resilient_modulus_psi(cbr):
    return 1500.0 * cbr if cbr <= 10 else 4326.0 * math.log(cbr) + 241.0


def flexible_engine(spec, years):
    a1f, stiff = BINDER_GRADES[spec["binder_grade"]]
    m = spec["drainage_coeff"]
    sn = (0.44 * a1f * spec["ac_thickness_mm"] + 0.14 * m * spec["base_thickness_mm"]
          + 0.11 * m * spec["subbase_thickness_mm"]) / 25.4
    mr = resilient_modulus_psi(spec["subgrade_cbr"])
    dpsi = spec["initial_psi"] - spec["terminal_psi"]
    logw = (spec["reliability_zr"] * spec["overall_std_dev"] + 9.36 * math.log10(sn + 1) - 0.20
            + math.log10(dpsi / (spec["initial_psi"] - 1.5)) / (0.40 + 1094.0 / (sn + 1) ** 5.19)
            + 2.32 * math.log10(mr) - 8.07)
    w_cap = 10 ** logw
    esal = traffic_per_year(spec, years, spec["vdf"])
    fd_load = np.concatenate([[0.0], np.cumsum(esal / w_cap)])

    age = spec["age_years"]
    cold = float(np.clip((15.0 - spec["min_pavement_temp_c"]) / 25.0, 0, 1))
    props = dict(structural_number=round(sn, 2), subgrade_mr_psi=round(mr), capacity_esal=round(w_cap),
                 binder_stiffness_index=stiff)
    cases = [
        dict(case="Structural fatigue / alligator cracking", category="load", score=None,
             basis=f"cumulative ESAL / capacity (SN={sn:.2f}, capacity {w_cap/1e6:.1f} msa)",
             advice="Strengthen with overlay; fix drainage; control overloading."),
        dict(case="Low-temperature / thermal transverse cracking", category="environment",
             score=float(np.clip(age / 20.0, 0, 1) * stiff * (0.4 + cold)),
             basis=f"binder {spec['binder_grade']}, age {age} y, min pavement temp {spec['min_pavement_temp_c']} C",
             advice="Use softer or polymer-modified binder; seal cracks early."),
        dict(case="Ageing / block cracking", category="environment",
             score=float(np.clip(age / 15.0, 0, 1) * (0.5 + 0.5 * stiff)),
             basis=f"age {age} y with binder stiffness index {stiff:.2f}",
             advice="Surface rejuvenation, fog seal or thin overlay before cracks open."),
        dict(case="Weak subgrade / moisture damage", category="environment",
             score=float(np.clip((8.0 - spec["subgrade_cbr"]) / 6.0, 0, 1)),
             basis=f"subgrade CBR {spec['subgrade_cbr']}%",
             advice="Subgrade improvement, subsurface drainage, thicker granular layers."),
        dict(case="Reflective cracking from underlying layer", category="environment",
             score=0.7 if spec["overlay_on_cracked_base"] else 0.0,
             basis="overlay placed on cracked / cement-treated / rigid base" if spec["overlay_on_cracked_base"]
             else "not applicable",
             advice="Use stress-absorbing interlayer or geosynthetic; crack-and-seat."),
        dict(case="Thin bituminous layer", category="environment",
             score=float(np.clip((100.0 - spec["ac_thickness_mm"]) / 60.0, 0, 1)),
             basis=f"AC thickness {spec['ac_thickness_mm']} mm",
             advice="Increase bituminous thickness for the design traffic."),
    ]
    return fd_load, props, cases, 1.0


# ============================================================================
# 4.  FAILURE-MODE CLASSIFICATION FROM THE IMAGE
# ============================================================================
def classify_failure_modes(cracks, summary, ptype, shape, travel_axis, reflective_possible=False):
    H, W = shape[:2]
    if not cracks:
        return [dict(mode="No cracking detected", score=1.0, evidence="no crack-like features above threshold")]
    along = "vertical" if travel_axis == "vertical" else "horizontal"
    across = "horizontal" if along == "vertical" else "vertical"
    n = len(cracks)
    branches = sum(c["branch_points"] for c in cracks)
    network = summary["pattern"].startswith("network") or branches >= 6     # one connected web also counts
    # direction shares use only simple (<=2 junction) cracks, so a connected web is not mistaken for one long crack
    simple = [c for c in cracks if c["branch_points"] <= 2]
    L = sum(c["length_px"] for c in simple) or 1.0
    share = {d: (sum(c["length_px"] for c in simple if c["direction"] == d) / L if simple else 0.0)
             for d in ("horizontal", "vertical", "diagonal")}
    s_long, s_tran, s_diag = share[along], share[across], share["diagonal"]
    wmax = summary.get("max_width_mm")
    hairline = wmax is not None and wmax < 0.3
    modes = []

    def add(mode, score, evidence):
        modes.append(dict(mode=mode, score=round(float(np.clip(score, 0, 1)), 2), evidence=evidence))

    if network:
        if ptype == "flexible":
            add("Alligator (fatigue) cracking", 0.6 + 0.4 * min(branches / 12.0, 1),
                f"interconnected web: {n} crack object(s), {branches} junctions")
        elif hairline:
            add("Map / craze cracking (shrinkage, ASR, poor curing)", 0.8,
                f"fine network, max width {wmax:.2f} mm")
        else:
            add("Shattered slab / multiple cracking (fatigue + support loss)", 0.7,
                f"interconnected web: {n} crack object(s), {branches} junctions")
    if len(simple) >= 4 and share["horizontal"] >= 0.25 and share["vertical"] >= 0.25 and not network:
        add("Block cracking (ageing)" if ptype == "flexible" else "Panel cracking in two directions", 0.65,
            f"two crack sets at right angles ({share['horizontal']:.0%} / {share['vertical']:.0%} of length)")
    if s_tran >= 0.5:
        sp = ""
        cs = sorted(c["centroid_xy"][1 if along == "vertical" else 0] for c in simple if c["direction"] == across)
        if len(cs) >= 3:
            gaps = np.diff(cs)
            if gaps.mean() > 0 and gaps.std() / gaps.mean() < 0.35:
                sp = f"; regularly spaced (~{gaps.mean():.0f} px)"
                if ptype == "flexible" and reflective_possible:
                    add("Reflective cracking (reflecting joints / cracks below)", 0.85, f"regular transverse cracks{sp}")
        add("Transverse cracking" + (" (thermal / shrinkage / fatigue)" if ptype == "rigid" else " (thermal / low-temperature)"),
            0.4 + 0.6 * s_tran, f"{s_tran:.0%} of crack length is across the traffic direction{sp}")
    if s_long >= 0.5:
        edge = any(min(c["centroid_xy"][0 if along == "vertical" else 1],
                       (W if along == "vertical" else H) - c["centroid_xy"][0 if along == "vertical" else 1])
                   / float(W if along == "vertical" else H) < 0.15 for c in simple if c["direction"] == along)
        if edge:
            add("Edge cracking (weak shoulder / edge support)", 0.75, "longitudinal crack within 15% of the lane edge")
        add("Longitudinal cracking" + (" (wheel-path fatigue / poor joint)" if ptype == "flexible"
                                       else " (tie-bar / joint / support problem)"),
            0.4 + 0.6 * s_long, f"{s_long:.0%} of crack length runs along the traffic direction")
    if s_diag >= 0.4:
        add("Diagonal / corner cracking (loss of support, curling)" if ptype == "rigid"
            else "Diagonal cracking", 0.4 + 0.6 * s_diag, f"{s_diag:.0%} of crack length is diagonal")
    if not modes:
        add("Isolated / irregular cracking", 0.5, f"{n} crack(s), no dominant pattern")
    modes.sort(key=lambda m: -m["score"])
    return modes[:4]


# ============================================================================
# 5.  CALIBRATION + FORECAST
# ============================================================================
def curve(fd, s, b=B_EXP):
    x = np.maximum(fd * s, 1e-12)
    return 1.0 / (1.0 + x ** (-b))


def first_year(P, level, start):
    idx = np.where(P >= level)[0]
    idx = idx[idx >= start]
    return int(idx[0]) if len(idx) else None


def predict(bgr, spec, mm_per_px=None, sens=0.5, travel_axis="vertical", horizon_years=20):
    base = DEFAULT_RIGID if spec.get("pavement_type", "rigid") == "rigid" else DEFAULT_FLEXIBLE
    spec = {**base, **{k: v for k, v in spec.items() if v is not None or k in ("k_override", "failure_density")}}
    ptype = spec["pavement_type"]
    age = int(spec["age_years"])
    years = age + int(horizon_years)

    # ---- image state now
    summary, cracks, overlay, mask, skel, _ = ca.analyze_image(bgr, mm_per_px, sens)
    H, W = bgr.shape[:2]
    if mm_per_px:
        area_m2 = (H * mm_per_px / 1000.0) * (W * mm_per_px / 1000.0)
        dens_now = summary["total_length_mm"] / 1000.0 / area_m2
        dens_fail = spec.get("failure_density") or DEFAULT_FAIL_DENSITY[ptype]
        unit = "m/m2"
    else:
        dens_now = summary["crack_area_percent"]
        dens_fail = DEFAULT_FAIL_AREA_PCT
        unit = "% area"
    p_obs = float(np.clip(dens_now / dens_fail, 0.002, 0.98))

    # ---- engine
    engine = rigid_engine if ptype == "rigid" else flexible_engine
    fd_load, props, cases, steel_factor = engine(spec, years)
    fd_now_load = float(fd_load[age])
    for c in cases:
        if c["score"] is None:
            c["score"] = float(np.clip(fd_now_load, 0, 1))
    env_driver = max([c["score"] for c in cases if c["category"] == "environment"] + [0.0])
    t_env = (60.0 if ptype == "rigid" else 25.0) / (1.0 + 2.0 * env_driver)
    t = np.arange(0, years + 1, dtype=float)
    fd = fd_load + t / t_env

    # ---- calibration to the photographed cracking
    fd_obs = (p_obs / (1 - p_obs)) ** (1.0 / B_EXP)
    if age > 0 and fd[age] > 0:
        shift = fd_obs / fd[age]
        calibrated = True
    else:
        shift, calibrated = 1.0, False
    P = curve(fd, shift)
    P_lo, P_hi = curve(fd, shift * 0.5), curve(fd, shift * 2.0)
    P_now = float(P[age])

    notes = []
    if not calibrated:
        notes.append("New pavement (age 0): forecast is model-only, not calibrated to the photo.")
    elif shift > 3:
        notes.append(f"Observed cracking is about {shift:.1f}x worse than the load + environment model predicts: "
                     "suspect another mechanism (overloading, poor construction/curing, drainage, material defect) "
                     "or under-estimated traffic.")
    elif shift < 0.33:
        notes.append(f"Observed cracking is much lighter than predicted (shift {shift:.2f}): the section may be "
                     "performing better than the inputs suggest - re-check traffic and layer data.")
    if not mm_per_px:
        notes.append("Image not scaled: crack extent is in % area and widths in pixels. Provide mm-per-pixel for mm results.")

    dens = P * dens_fail
    forecast = []
    w_now = summary.get("max_width_mm") if mm_per_px else None
    w_fail = 3.0 / steel_factor
    for yr in range(age, years + 1):
        row = dict(age_years=yr, years_from_now=yr - age, crack_extent_index=round(float(P[yr]), 3),
                   crack_density=round(float(dens[yr]), 3), density_unit=unit,
                   index_low=round(float(P_lo[yr]), 3), index_high=round(float(P_hi[yr]), 3))
        if w_now is not None:
            gain = max(0.0, (P[yr] - P_now) / max(1 - P_now, 1e-3))
            row["max_crack_width_mm"] = round(float(w_now + max(w_fail - w_now, 0) * gain), 2)
        forecast.append(row)

    thr = []
    for lvl, label in THRESHOLDS:
        y_c, y_e, y_l = first_year(P, lvl, 0), first_year(P_hi, lvl, 0), first_year(P_lo, lvl, 0)
        thr.append(dict(level=lvl, action=label, already_reached=bool(y_c is not None and y_c <= age),
                        reached_at_age=y_c, years_from_now=None if y_c is None else max(y_c - age, 0),
                        earliest_age=y_e, latest_age=y_l))

    modes = classify_failure_modes(cracks, summary, ptype, bgr.shape, travel_axis,
                                   spec.get("overlay_on_cracked_base", False))
    cases.sort(key=lambda c: -(c["score"] or 0))
    res = dict(
        pavement_type=ptype, spec=spec,
        image=dict(crack_count=summary["crack_count"], pattern=summary["pattern"],
                   crack_density_now=round(dens_now, 3), density_unit=unit,
                   max_width_mm=w_now, worst_severity=summary.get("worst_severity"),
                   extent_index_now=round(p_obs, 3)),
        observed_failure_modes=modes,
        material_structural_properties={k: v for k, v in props.items() if k != "fatigue_bins"},
        fatigue_bins=props.get("fatigue_bins"),
        failure_case_scores=[dict(case=c["case"], score=round(float(c["score"]), 2), basis=c["basis"],
                                  advice=c["advice"]) for c in cases],
        damage=dict(load_damage_now=round(fd_now_load, 3), environmental_damage_now=round(float(age / t_env), 3),
                    calibration_shift=round(shift, 2), calibrated_to_image=calibrated),
        thresholds=thr, forecast=forecast, notes=notes, assumptions=ASSUMPTIONS,
    )
    fig = make_plot(res, t[age:], P[age:], P_lo[age:], P_hi[age:], age, cases)
    return res, overlay, fig


# ============================================================================
# 6.  PLOT
# ============================================================================
def make_plot(res, t, P, lo, hi, age, cases):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    yrs = t - age
    ax[0].fill_between(yrs, lo, hi, color="#ff5a36", alpha=0.18, label="sensitivity band (x0.5 / x2)")
    ax[0].plot(yrs, P, color="#ff5a36", lw=2.6, label="predicted cracking extent")
    ax[0].scatter([0], [P[0]], color="#16162b", zorder=5, s=60, label="photographed state (today)")
    for lvl, label in THRESHOLDS:
        ax[0].axhline(lvl, ls="--", lw=1, color="#16162b", alpha=0.5)
        ax[0].text(yrs[-1], lvl + 0.01, label.split(" (")[0], ha="right", va="bottom", fontsize=8, color="#16162b")
    ax[0].set_xlabel("years from today"); ax[0].set_ylabel("cracking extent index (0 = none, 1 = failed)")
    ax[0].set_ylim(0, 1.02); ax[0].set_xlim(0, yrs[-1]); ax[0].grid(alpha=0.25)
    ax[0].set_title(f"{res['pavement_type'].capitalize()} pavement - crack development forecast")
    ax[0].legend(loc="lower right", fontsize=9)
    top = [c for c in cases][:7][::-1]
    ax[1].barh([c["case"][:42] for c in top], [c["score"] or 0 for c in top], color="#0f8b8d")
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("risk score (0-1)"); ax[1].set_title("Failure-case risk scores")
    ax[1].tick_params(axis="y", labelsize=8); ax[1].grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


# ============================================================================
# 7.  CLI
# ============================================================================
def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return v


def cmd_make_spec(args):
    spec = DEFAULT_RIGID if args.type == "rigid" else DEFAULT_FLEXIBLE
    print(json.dumps(spec, indent=2))


def cmd_predict(args):
    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"Cannot read image: {args.image}")
    spec = {"pavement_type": args.type}
    if args.spec:
        with open(args.spec) as f:
            spec.update(json.load(f))
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        spec[k] = parse_value(v)
    res, overlay, fig = predict(bgr, spec, args.mm_per_px, args.sens, args.travel_axis, args.horizon)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{base}_cracks_overlay.png"), overlay)
    fig.savefig(os.path.join(args.out, f"{base}_forecast.png"), dpi=150)
    with open(os.path.join(args.out, f"{base}_prediction.json"), "w") as f:
        json.dump(res, f, indent=2, default=to_jsonable)
    with open(os.path.join(args.out, f"{base}_forecast.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(res["forecast"][0].keys()))
        wr.writeheader(); wr.writerows(res["forecast"])

    print(f"\n{res['pavement_type'].upper()} PAVEMENT  |  age {res['spec']['age_years']} y")
    im = res["image"]
    print(f"Photo: {im['crack_count']} crack(s), {im['pattern']}, density {im['crack_density_now']} {im['density_unit']}, "
          f"extent index {im['extent_index_now']}")
    print("Likely failure mode(s) seen:")
    for m in res["observed_failure_modes"]:
        print(f"  - {m['mode']} (score {m['score']}): {m['evidence']}")
    print("Top failure-case risks (from material / structure / traffic):")
    for c in res["failure_case_scores"][:5]:
        print(f"  - {c['case']}: {c['score']}  [{c['basis']}]")
    print("Forecast:")
    for t in res["thresholds"]:
        when = ("beyond horizon" if t["years_from_now"] is None else
                "ALREADY REACHED" if t["already_reached"] else
                f"in ~{t['years_from_now']} y (age {t['reached_at_age']})")
        print(f"  - index {t['level']}: {t['action']} -> {when}")
    for n in res["notes"]:
        print("NOTE:", n)
    print(f"Saved results in: {args.out}/")


def main():
    ap = argparse.ArgumentParser(description="Pavement crack prediction (image + material properties)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-spec", help="print a spec template with default properties")
    m.add_argument("--type", choices=["rigid", "flexible"], default="rigid")
    m.set_defaults(fn=cmd_make_spec)
    p = sub.add_parser("predict", help="analyse a photo and forecast crack development")
    p.add_argument("image")
    p.add_argument("--type", choices=["rigid", "flexible"], default="rigid")
    p.add_argument("--spec", help="JSON file of pavement properties (see make-spec)")
    p.add_argument("--set", action="append", help="override a property, e.g. --set age_years=8")
    p.add_argument("--mm-per-px", type=float, default=None)
    p.add_argument("--sens", type=float, default=0.5)
    p.add_argument("--travel-axis", choices=["vertical", "horizontal"], default="vertical",
                   help="direction of traffic in the image")
    p.add_argument("--horizon", type=int, default=20, help="years to forecast")
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
