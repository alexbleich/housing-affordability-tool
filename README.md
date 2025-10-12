# Housing Affordability Visualizer

*See how small policy choices move total development cost (TDC) and who can afford new homes in Vermont.*

**Live app:** https://housing-affordability-tool.streamlit.app

## What it does

- Lets you “build” a simple for-sale home (Townhome or Condo) by choosing:
  - **Energy code**, **heating source**, **finish quality**, and **location**.
- Calculates **Total Development Cost (TDC)** from transparent inputs.
- Converts price ↔ **required household income** using VHFA affordability tables.
- Estimates the **share of VT households** able to afford that price.
- Shows approximate **% of AMI** for the chosen region.

> *Note:* The apartment (rent) model is planned but not enabled yet.

## How to use it

1. **Choose housing type** and **bedrooms**.  
2. **Pick policy options** (energy code, heating, finish, location).  
3. **Select region** and **household size**, then enter an **annual income**.  
4. **View the chart** comparing TDC to affordable purchase price; read the short explainer.  
5. Try additional options to compare what moves affordability most.

## Where the numbers come from

- **Costs:** Combined from `assumptions.csv` into TDC (per-sf, per-unit, fixed).  
- **Price ↔ Income:** Interpolated from VHFA affordability tables by region & household size.  
- **Household share:** Computed from `vt_inc_dist.csv` (est. # and % of VT households at/above required income).  
- **AMI %:** Translates required income to AMI for the selected region.

*Edge cases (e.g., interpolation and carefully limited extrapolation) are documented in* `ASSUMPTIONS.md`

## Intended use & limits

This is a **policy discussion tool**, not a project pro-forma. It is **transparent and directional**. Figures may vary with data sources and versions—always consult `ASSUMPTIONS.md` for context.

## Feedback

Questions or suggestions? Reach out to Summit Properties.
