# Housing Affordability Visualizer

*A simple web tool showing how policy choices change total development cost (TDC) and who can afford new homes in Vermont.*

**Live app:** https://housing-affordability-tool.streamlit.app

## How to use
1. Choose **home type** (Townhome or Condo) and **bedrooms**.
2. Pick **energy code**, **heating**, **finish quality**, and **new neighborhood** (yes/no).  
3. Select **region** and **household size**, enter **annual income**, then view results.
> *Apartment (rent) model is planned; not enabled yet.*

## What you’ll see
- **Bars** = Total Development Cost (TDC) for your selections.  
- **Green Line** = Affordable purchase price for the income/household you chose.  
- Clear text explaining **required income**, **% of AMI**, and **share of VT households** who could afford it.
 
## Where numbers come from
- **Assumptions & methods:** See **ASSUMPTIONS.md** (CSV schema, formulas, extrapolation, caveats).  
- **Data files:** `vhfa data/*.csv`, `data/assumptions.csv`, `data/vt_inc_dist.csv`.

## Repository
- `housing-affordability-tool.py` — Streamlit app  
- `ASSUMPTIONS.md` — methods & schema  
- `requirements.txt` — this is needed for the app to run; you can ignore it
- `data/` & `vhfa data/` — CSVs

*This tool is intended for policy discussions, not project-level pro formas. Results are illustrative and may change as the underlying data are updated.*
