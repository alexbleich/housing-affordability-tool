# 🏘️ Housing Affordability Visualizer 🏘️

This tool compares policy-driven differences in total development cost (TDC) for a single housing unit type and bedroom count against AMI-based affordability thresholds across Vermont regions. 

It is designed to help visualize how **changes in policy directly impact housing affordability**.

Access the tool here: https://housing-affordability-tool.streamlit.app

---

## User Inputs
- Housing product type: Townhome, Condo, or Apartment
- Number of bedrooms: *(for Townhome and Condo only — Apartments will use a different model coming soon)*
- Number of units to compare: (1–5)
- For each unit:
  - Energy code standard
  - Energy source
  - Infrastructure requirement
  - Finish quality
- Vermont region(s): Chittenden, Addison, and/or Rest of Vermont
- AMI level(s): Choose up to 3

---

## What It Does
- Calculates TDC for each unit based on baseline costs and selected policy options
- Retrieves affordable purchase price thresholds using AMI data for the selected regions
- Generates a side‑by‑side visual comparison of costs vs. affordability

---

## Output
- Bars = TDC for each scenario
- Dashed lines = Affordability thresholds for selected AMI levels and region(s)
- Dual Y‑axis: Left = TDC, Right = Affordability thresholds

---

## Files in This Repository
- `data/assumptions.csv` — Policy cost assumptions (baseline $/sf, energy codes, energy sources, finish quality, infrastructure, etc.)
- `data/chittenden_ami.csv`, `data/addison_ami.csv`, `data/vermont_ami.csv` — VHFA-provided AMI thresholds for Chittenden County, Addison County, and the rest of Vermont
- `housing-affordability-tool.py` — code running the Streamlit app
