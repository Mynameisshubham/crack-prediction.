"""
Concrete Crack & Deformation Analyzer - point-and-click web app (Streamlit)

Needs crack_analyzer.py in the SAME folder.
Run:   streamlit run crack_app.py
"""
import argparse
import contextlib
import io
import json
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import crack_analyzer as ca
import pavement_predictor as pp


# ----------------------------------------------------------------- helpers
def decode(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image file.")
    return img


def scale_args(mode, mm_per_px, calib_text):
    """Build the scale part of the argument namespace used by crack_analyzer."""
    ns = argparse.Namespace(mm_per_px=None, calib=None)
    if mode == "mm per pixel":
        ns.mm_per_px = mm_per_px or None
    elif mode == "known length in photo":
        ns.calib = calib_text or None
    return ns


def run_analyze(file_bytes, sens, scale):
    img = decode(file_bytes)
    summary, cracks, overlay, mask, _, _ = ca.analyze_image(
        img, ca.mm_per_px_original(scale), sens)
    return summary, cracks, overlay, mask


def run_pair(cmd, bytes_a, bytes_b, scale, sens, days=None, gauge=15):
    """Run 'growth' or 'deform' (they read/write files) in a temp folder."""
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = os.path.join(tmp, "a.jpg"), os.path.join(tmp, "b.jpg")
        with open(pa, "wb") as f:
            f.write(bytes_a)
        with open(pb, "wb") as f:
            f.write(bytes_b)
        ns = argparse.Namespace(mm_per_px=scale.mm_per_px, calib=scale.calib,
                                out=os.path.join(tmp, "out"), sens=sens, days=days,
                                gauge=gauge, disp_thr_px=None)
        if cmd == "growth":
            ns.before, ns.after = pa, pb
            fn, rep, img = ca.cmd_growth, "growth_report.json", "growth_overlay.png"
        else:
            ns.reference, ns.deformed = pa, pb
            fn, rep, img = ca.cmd_deform, "deformation_report.json", "deformation_map.png"
        with contextlib.redirect_stdout(io.StringIO()):
            fn(ns)
        with open(os.path.join(ns.out, rep)) as f:
            report = json.load(f)
        image = cv2.cvtColor(cv2.imread(os.path.join(ns.out, img)), cv2.COLOR_BGR2RGB)
        return report, image


def png_bytes(bgr_or_rgb, is_bgr=True):
    img = bgr_or_rgb if is_bgr else cv2.cvtColor(bgr_or_rgb, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img)[1].tobytes()



CONTACT_EMAIL = "2023BCE018@SGGS.AC.IN"

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

:root{
  --ink:#16162b; --paper:#fbf7f0; --card:#ffffff; --coral:#ff5a36; --teal:#0f8b8d; --sand:#efe6d8;
  --ease:cubic-bezier(.22,1,.36,1);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea{
  font-family:'DM Sans',sans-serif !important;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 500px at 105% -10%, rgba(255,90,54,.10), transparent 60%),
    radial-gradient(800px 500px at -10% 110%, rgba(15,139,141,.12), transparent 60%),
    var(--paper);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:var(--sand); border-right:1px solid rgba(22,22,43,.08);}
h1,h2,h3,h4{font-family:'Syne',sans-serif !important; letter-spacing:-.02em; color:var(--ink);}

/* ---------- entrance animation for everything on the page ---------- */
@keyframes fadeUp{from{opacity:0; transform:translateY(22px);} to{opacity:1; transform:none;}}
@keyframes popIn{from{opacity:0; transform:scale(.96);} to{opacity:1; transform:none;}}
@keyframes drift{0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;}}
@keyframes draw{to{stroke-dashoffset:0;}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,90,54,.55);} 70%{box-shadow:0 0 0 10px rgba(255,90,54,0);}}

.block-container > div > div{animation:fadeUp .7s var(--ease) both;}
.block-container > div > div:nth-child(2){animation-delay:.08s;}
.block-container > div > div:nth-child(3){animation-delay:.16s;}
[data-testid="stImage"] img{border-radius:14px; box-shadow:0 10px 30px rgba(22,22,43,.15); animation:popIn .6s var(--ease) both;}

/* ---------- hero ---------- */
.hero{position:relative; overflow:hidden; padding:2.4rem 2.4rem 2.2rem; border-radius:24px; color:#fff;
  background:linear-gradient(120deg,#16162b,#2b1f4a,#0f4c5c,#16162b); background-size:300% 300%;
  animation:drift 14s ease infinite; box-shadow:0 20px 50px rgba(22,22,43,.28);}
.hero .eyebrow{letter-spacing:.28em; font-size:.72rem; font-weight:700; color:#ffb4a2; margin:0 0 .6rem;}
.hero h1{color:#fff !important; font-size:clamp(2rem,4.2vw,3.4rem); line-height:1.05; margin:0 0 .7rem; font-weight:800;}
.hero h1 span{background:linear-gradient(90deg,#ff8a6b,#ffd36b); -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p.sub{max-width:44rem; color:#d9d9ec; font-size:1.02rem; margin:0;}
.hero svg{position:absolute; right:-10px; top:0; height:100%; width:46%; opacity:.9; pointer-events:none;}
.hero .crackline{fill:none; stroke:#ff8a6b; stroke-width:2.6; stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:900; stroke-dashoffset:900; animation:draw 3.2s .5s ease-out forwards;}
.hero .crackline.b{stroke:#ffd36b; stroke-width:1.6; animation-delay:1.3s;}
.chips{margin-top:1.1rem; display:flex; gap:.5rem; flex-wrap:wrap;}
.chip{padding:.3rem .8rem; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
  font-size:.8rem; backdrop-filter:blur(4px); transition:all .3s var(--ease);}
.chip:hover{background:rgba(255,255,255,.24); transform:translateY(-2px);}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"]{gap:.4rem; border-bottom:none;}
[data-baseweb="tab"]{font-family:'Syne',sans-serif !important; font-weight:700; border-radius:12px 12px 0 0;
  padding:.65rem 1.1rem; transition:background .3s var(--ease), transform .3s var(--ease);}
[data-baseweb="tab"]:hover{background:rgba(255,90,54,.10); transform:translateY(-2px);}
[data-baseweb="tab-highlight"]{background:var(--coral) !important; height:3px !important; transition:all .45s var(--ease) !important;}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button{border-radius:14px; border:none; font-weight:700; padding:.65rem 1.4rem;
  transition:transform .25s var(--ease), box-shadow .25s var(--ease), filter .25s;}
.stButton > button[kind="primary"]{background:linear-gradient(90deg,var(--coral),#ff8a3d); color:#fff; animation:pulse 2.6s infinite;}
.stButton > button:hover, .stDownloadButton > button:hover{transform:translateY(-3px) scale(1.02);
  box-shadow:0 12px 26px rgba(255,90,54,.30); filter:brightness(1.05);}
.stButton > button:active{transform:translateY(0) scale(.98);}

/* ---------- metric cards ---------- */
[data-testid="stMetric"]{background:var(--card); border-radius:18px; padding:1rem 1.2rem; border:1px solid rgba(22,22,43,.07);
  border-left:5px solid var(--teal); box-shadow:0 6px 18px rgba(22,22,43,.07); animation:popIn .6s var(--ease) both;
  transition:transform .3s var(--ease), box-shadow .3s var(--ease);}
[data-testid="stMetric"]:hover{transform:translateY(-5px); box-shadow:0 16px 34px rgba(22,22,43,.14);}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif !important; font-weight:800;}

/* ---------- upload box, tables ---------- */
[data-testid="stFileUploader"] section{border:2px dashed rgba(15,139,141,.5); border-radius:18px; background:rgba(255,255,255,.7);
  transition:all .3s var(--ease);}
[data-testid="stFileUploader"] section:hover{border-color:var(--coral); background:#fff; transform:scale(1.01);}
[data-testid="stDataFrame"]{border-radius:14px; overflow:hidden; box-shadow:0 6px 18px rgba(22,22,43,.07);}

/* ---------- footer ---------- */
.footer{margin-top:3rem; padding:1.4rem 1.6rem; border-radius:20px; background:var(--ink); color:#cfcfe6;
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.8rem;}
.footer b{font-family:'Syne',sans-serif; color:#fff; font-size:1.05rem;}
.footer a{color:#ffb4a2 !important; font-weight:700; text-decoration:none; border-bottom:2px solid transparent; transition:border-color .3s;}
.footer a:hover{border-color:#ffb4a2;}
.side-contact{font-size:.85rem; margin-top:1.5rem; padding-top:1rem; border-top:1px dashed rgba(22,22,43,.25);}
.side-contact a{color:var(--coral) !important; font-weight:700; text-decoration:none;}

@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} .hero .crackline{stroke-dashoffset:0;} }
</style>
"""

HERO = """
<div class="hero">
  <svg viewBox="0 0 400 220" preserveAspectRatio="xMaxYMid slice" xmlns="http://www.w3.org/2000/svg">
    <path class="crackline" d="M10 20 L70 62 L58 98 L128 124 L112 160 L196 178 L214 214"/>
    <path class="crackline b" d="M128 124 L190 96 L236 112 L300 70 L372 84"/>
    <path class="crackline b" d="M196 178 L262 168 L318 196"/>
  </svg>
  <p class="eyebrow">STRUCTURAL HEALTH MONITORING</p>
  <h1>Concrete Crack &amp; <span>Deformation</span> Analyzer</h1>
  <p class="sub">Upload a photo. Get crack length, width, severity and growth &mdash; plus a displacement map of how the surface has moved.</p>
  <div class="chips"><span class="chip">Crack mapping</span><span class="chip">Width in mm</span>
  <span class="chip">Severity grading</span><span class="chip">Growth tracking</span><span class="chip">Deformation / strain</span></div>
</div>
"""

FOOTER = f"""
<div class="footer">
  <div><b>Crack &amp; Deformation Analyzer</b><br>
  Screening tool &mdash; not a substitute for an engineer's inspection.</div>
  <div>Questions or feedback?<br><a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></div>
</div>
"""


# --------------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Crack & Deformation Analyzer", page_icon="🧱", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Scale (to get mm)", ["mm per pixel", "known length in photo", "none (pixels only)"])
        mm_per_px, calib_text = 0.0, ""
        if mode == "mm per pixel":
            mm_per_px = st.number_input("mm per pixel", min_value=0.0, value=0.25, step=0.05, format="%.4f")
        elif mode == "known length in photo":
            calib_text = st.text_input("x1,y1,x2,y2,length_mm", placeholder="120,540,620,545,100",
                                       help="Pixel coordinates of two ends of an object of known length "
                                            "(e.g. a ruler) lying in the crack plane.")
        sens = st.slider("Detection sensitivity", 0.0, 1.0, 0.5, 0.05,
                         help="Higher finds fainter cracks but gives more false alarms.")
        scale = scale_args(mode, mm_per_px, calib_text)
        st.markdown(f'<div class="side-contact">Questions? Write to<br>'
                    f'<a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Crack analysis", "Crack growth", "Deformation", "Pavement prediction"])

    # ---- tab 1
    with tab1:
        up = st.file_uploader("Upload a photo of the concrete surface", type=["jpg", "jpeg", "png"], key="one")
        if up and st.button("Analyze cracks", type="primary"):
            try:
                with st.spinner("Analyzing..."):
                    summary, cracks, overlay, mask = run_analyze(up.getvalue(), sens, scale)
            except (SystemExit, ValueError) as e:
                st.error(str(e))
            else:
                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(decode(up.getvalue()), cv2.COLOR_BGR2RGB), caption="Original")
                c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Detected cracks")
                m = st.columns(4)
                m[0].metric("Cracks found", summary["crack_count"])
                if summary.get("calibrated"):
                    m[1].metric("Max width (mm)", summary["max_width_mm"])
                    m[2].metric("Total length (mm)", summary["total_length_mm"])
                    m[3].metric("Worst severity", summary["worst_severity"])
                else:
                    m[1].metric("Total length (px)", summary["total_length_px"])
                    m[2].metric("Crack area %", summary["crack_area_percent"])
                    st.info(summary.get("note", ""))
                st.write("Pattern:", summary["pattern"])
                if cracks:
                    df = pd.DataFrame(cracks).drop(columns=["centroid_xy"])
                    st.dataframe(df, use_container_width=True)
                    d1, d2 = st.columns(2)
                    d1.download_button("Download table (CSV)", df.to_csv(index=False), "cracks.csv")
                    d2.download_button("Download overlay (PNG)", png_bytes(overlay), "overlay.png")

    # ---- tab 2
    with tab2:
        st.write("Upload two photos of the **same view** taken at different times.")
        c1, c2 = st.columns(2)
        before = c1.file_uploader("Earlier photo", type=["jpg", "jpeg", "png"], key="gb")
        after = c2.file_uploader("Later photo", type=["jpg", "jpeg", "png"], key="ga")
        days = st.number_input("Days between photos (optional)", min_value=0.0, value=0.0)
        if before and after and st.button("Compare cracks", type="primary"):
            try:
                with st.spinner("Aligning and comparing..."):
                    report, image = run_pair("growth", before.getvalue(), after.getvalue(),
                                             scale, sens, days or None)
            except (SystemExit, ValueError) as e:
                st.error(str(e))
            else:
                st.image(image, caption="Blue = before, green = persisting, red = new / extended crack")
                st.json(report)

    # ---- tab 3
    with tab3:
        st.write("Upload a **reference** photo and a **later / loaded** photo of the same view.")
        c1, c2 = st.columns(2)
        ref = c1.file_uploader("Reference photo", type=["jpg", "jpeg", "png"], key="dr")
        cur = c2.file_uploader("Later / loaded photo", type=["jpg", "jpeg", "png"], key="dc")
        gauge = st.slider("Smoothing / gauge length (px)", 5, 50, 15)
        if ref and cur and st.button("Measure deformation", type="primary"):
            try:
                with st.spinner("Computing displacement and strain..."):
                    report, image = run_pair("deform", ref.getvalue(), cur.getvalue(),
                                             scale, sens, gauge=gauge)
            except (SystemExit, ValueError) as e:
                st.error(str(e))
            else:
                st.image(image, caption="Displacement, strain and hotspots")
                st.write("**Verdict:**", report.get("verdict"))
                st.json(report)

    # ---- tab 4 : pavement crack prediction (image + material properties)
    with tab4:
        st.write("Predict how cracking will **develop** from a photo plus the fixed pavement properties.")
        up4 = st.file_uploader("Photo of the pavement", type=["jpg", "jpeg", "png"], key="pp_img")
        ptype = st.radio("Pavement type", ["rigid", "flexible"], horizontal=True,
                         format_func=lambda x: "Rigid (concrete)" if x == "rigid" else "Flexible (bituminous)")
        base = pp.DEFAULT_RIGID if ptype == "rigid" else pp.DEFAULT_FLEXIBLE
        spec = {"pavement_type": ptype}

        with st.expander("Materials and structure (fixed properties)", expanded=True):
            c1, c2, c3 = st.columns(3)
            if ptype == "rigid":
                spec["concrete_grade"] = c1.selectbox("Concrete grade", list(pp.CONCRETE_GRADES), index=3)
                spec["steel_grade"] = c2.selectbox("Steel grade", list(pp.STEEL_GRADES), index=2)
                spec["base_type"] = c3.selectbox("Base type", list(pp.BASE_K_FACTOR), index=3)
                spec["slab_thickness_mm"] = c1.number_input("Slab thickness (mm)", 150, 500, base["slab_thickness_mm"], 10)
                spec["joint_spacing_m"] = c2.number_input("Joint spacing (m)", 2.0, 10.0, base["joint_spacing_m"], 0.5)
                spec["subgrade_cbr"] = c3.number_input("Subgrade CBR (%)", 1.0, 30.0, float(base["subgrade_cbr"]), 0.5)
                spec["steel_dia_mm"] = c1.number_input("Distributed steel bar dia (mm, 0 = plain slab)", 0, 25, 0, 2)
                spec["steel_spacing_mm"] = c2.number_input("Steel spacing (mm)", 50, 500, base["steel_spacing_mm"], 25)
                spec["temp_diff_c"] = c3.number_input("Design temperature differential (°C)", 5.0, 30.0,
                                                      float(base["temp_diff_c"]), 0.5)
                spec["has_dowels"] = c1.checkbox("Dowel bars at transverse joints", True)
                spec["has_tie_bars"] = c2.checkbox("Tie bars at longitudinal joint", True)
            else:
                spec["binder_grade"] = c1.selectbox("Binder grade", list(pp.BINDER_GRADES), index=2)
                spec["ac_thickness_mm"] = c2.number_input("Bituminous layer (mm)", 40, 400, base["ac_thickness_mm"], 10)
                spec["base_thickness_mm"] = c3.number_input("Base thickness (mm)", 0, 600, base["base_thickness_mm"], 10)
                spec["subbase_thickness_mm"] = c1.number_input("Sub-base thickness (mm)", 0, 600, base["subbase_thickness_mm"], 10)
                spec["subgrade_cbr"] = c2.number_input("Subgrade CBR (%)", 1.0, 30.0, float(base["subgrade_cbr"]), 0.5)
                spec["min_pavement_temp_c"] = c3.number_input("Minimum pavement temperature (°C)", -30.0, 30.0,
                                                              float(base["min_pavement_temp_c"]), 1.0)
                spec["overlay_on_cracked_base"] = c1.checkbox("Overlay on cracked / rigid base", False)

        with st.expander("Traffic and age", expanded=True):
            t1, t2, t3 = st.columns(3)
            spec["age_years"] = t1.number_input("Pavement age (years)", 0, 60, int(base["age_years"]), 1)
            spec["cvpd_now"] = t2.number_input("Commercial vehicles per day (today)", 0, 50000, int(base["cvpd_now"]), 100)
            spec["growth_pct"] = t3.number_input("Traffic growth (%/yr)", 0.0, 15.0, float(base["growth_pct"]), 0.5)
            spec["lane_dist_factor"] = t1.number_input("Lane distribution factor", 0.1, 1.0, float(base["lane_dist_factor"]), 0.05)
            if ptype == "rigid":
                spec["axles_per_cv"] = t2.number_input("Axles per commercial vehicle", 1.0, 5.0, float(base["axles_per_cv"]), 0.1)
            else:
                spec["vdf"] = t2.number_input("Vehicle damage factor (VDF)", 0.5, 10.0, float(base["vdf"]), 0.5)

        o1, o2 = st.columns(2)
        travel = o1.selectbox("Traffic direction in the photo", ["vertical", "horizontal"])
        horizon = o2.slider("Forecast horizon (years)", 5, 40, 20)

        if up4 and st.button("Predict crack development", type="primary"):
            try:
                img4 = decode(up4.getvalue())
                with st.spinner("Analysing photo and running the prediction..."):
                    res, overlay4, fig = pp.predict(img4, spec, scale.mm_per_px or ca.mm_per_px_original(scale),
                                                    sens, travel, horizon)
            except (SystemExit, ValueError, KeyError) as e:
                st.error(f"Could not run the prediction: {e}")
            else:
                im = res["image"]
                m = st.columns(4)
                m[0].metric("Cracks seen", im["crack_count"])
                m[1].metric(f"Crack density ({im['density_unit']})", im["crack_density_now"])
                m[2].metric("Extent index today", im["extent_index_now"])
                m[3].metric("Calibration shift", res["damage"]["calibration_shift"])
                a, b = st.columns(2)
                a.image(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB), caption="Photo")
                b.image(cv2.cvtColor(overlay4, cv2.COLOR_BGR2RGB), caption="Detected cracks")
                for n in res["notes"]:
                    st.warning(n)
                st.subheader("Failure mode seen in the photo")
                st.dataframe(pd.DataFrame(res["observed_failure_modes"]), use_container_width=True)
                st.subheader("Forecast")
                st.pyplot(fig)
                st.dataframe(pd.DataFrame([{"index": t["level"], "action": t["action"],
                                            "when": ("beyond horizon" if t["years_from_now"] is None else
                                                     "already reached" if t["already_reached"] else
                                                     f"in ~{t['years_from_now']} years")}
                                           for t in res["thresholds"]]), use_container_width=True)
                st.subheader("Failure-case risk scores (from materials, structure, traffic)")
                st.dataframe(pd.DataFrame(res["failure_case_scores"]), use_container_width=True)
                with st.expander("Calculated properties"):
                    st.json(res["material_structural_properties"])
                with st.expander("Assumptions and limits"):
                    for t_ in res["assumptions"]:
                        st.write("•", t_)
                fc = pd.DataFrame(res["forecast"])
                d1, d2, d3 = st.columns(3)
                d1.download_button("Forecast (CSV)", fc.to_csv(index=False), "forecast.csv")
                d2.download_button("Full report (JSON)", json.dumps(res, indent=2, default=pp.to_jsonable), "prediction.json")
                buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150)
                d3.download_button("Forecast chart (PNG)", buf.getvalue(), "forecast.png")

    st.markdown(FOOTER, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
