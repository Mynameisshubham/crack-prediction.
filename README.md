# Pavement Crack Detection & Prediction

Image-based tool to detect and measure cracks in concrete / pavement photos, track crack growth and
deformation, and **predict how cracking will develop** using pavement material properties
(concrete grade, steel grade, layer thicknesses, subgrade CBR, binder grade, traffic, age).

## Features
- **Crack analysis** - length, width (mm), orientation, severity grade, colour-coded overlay, CSV/JSON report
- **Crack growth** - compare two inspections, new/extended cracks, growth rate
- **Deformation** - displacement and strain map between a reference and a later photo
- **Pavement prediction** - failure-mode classification (transverse, longitudinal, corner, alligator, block, map, edge,
  reflective), failure-case risk scores, calibrated year-by-year crack forecast
- **Baseline model** - HOG + ANN classifier after Mirbod & Shoar (2023), Procedia Computer Science 217, on SDNET2018

## Project files
| File | Purpose |
|---|---|
| `crack_app.py` | Streamlit web app (4 tabs) |
| `crack_analyzer.py` | Crack detection, measurement, growth, deformation (also a command-line tool) |
| `pavement_predictor.py` | Material-property + image based crack prediction (also a command-line tool) |
| `crack_hog_ann.py` | HOG + ANN crack / no-crack classifier (SDNET2018) |
| `requirements.txt` | Python dependencies |

## Install
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
streamlit run crack_app.py                                   # web app
python crack_analyzer.py analyze wall.jpg --mm-per-px 0.25   # command line
python pavement_predictor.py make-spec --type rigid > spec.json
python pavement_predictor.py predict road.jpg --spec spec.json --mm-per-px 0.5 --set age_years=8
```

## Limitations
This is a screening / research tool, not a design check. Formulas are simplified and several coefficients are
engineering assumptions that should be calibrated with local field data. Detection uses classical image processing,
so shadows, stains and joints can be flagged as cracks. A single photo covers a small patch. Not a substitute for
an engineer's inspection.

## Dataset
SDNET2018, Utah State University - https://doi.org/10.15142/T3TD19 (download separately; not included).

## Contact
2023BCE018@SGGS.AC.IN
