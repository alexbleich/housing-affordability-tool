# 🏘️ Housing Affordability Visualizer 🏘️

This tool compares policy-driven differences in total development cost (TDC) for a single housing unit type and bedroom count against AMI-based affordability thresholds across Vermont regions.

It is designed to help visualize how **changes in policy directly impact housing affordability**.

Access the tool here: https://housing-affordability-tool.streamlit.app

---

## User Inputs
- Housing product type: **Townhome**, **Condo**, or **Apartment**  
  *(Apartment model is rent-based and coming soon; choose Townhome or Condo to run the for-sale model.)*
- Number of bedrooms *(for Townhome/Condo)*
- Number of units to compare (1–5)
- For each unit:
  - Policy package (**Baseline**, **Top-of-the-Line**, **Below Baseline**)
  - Advanced overrides:
    - Energy code standard
    - Energy source
    - Finish quality
    - Optional custom bar label
- Vermont region(s) for **Chart 1** thresholds: **Chittenden**, **Addison**, and/or **Rest of Vermont**
- AMI level(s): choose up to 3 (30, 50–150% in 5% steps)
- Household context for **Chart 2**:
  - Region (single select)
  - Household size (1–8)
  - Household income (bounded to the AMI table for the selected region & size; **max is the 150% AMI income**)

> **Note:** Infrastructure is currently **ignored** in TDC calculations to avoid overstating per-unit costs for condos. It may be re-enabled later with project-level assumptions.

---

## What It Does
- Calculates TDC per unit using:
  - Baseline $/sf × (multifamily efficiency factor + policy % adders) × unit square footage
  - Plus energy-source $/sf adders  
  - *(Infrastructure adders are intentionally excluded right now.)*
- Retrieves affordable **purchase price** thresholds by AMI and region
- Builds an **invertible mapping between purchase price and household income** from the VHFA table so **both y-axes stay in sync** (with the top duplicate tick pruned)

---

## Output
- **Chart 1 — “Do These Policy Choices Put Homes Within Reach?”**  
  - **Bars:** TDC for each scenario  
  - **Dashed lines:** Affordability thresholds for selected AMI levels & regions  
  - **Dual Y-axis:** Left = TDC; Right = “% AMI + price” tick labels for the selected lines
- **Chart 2 — “What These Costs Mean for Your Constituents”**  
  - **Bars:** TDC for each scenario  
  - **Green line:** Your chosen income mapped to its max affordable purchase price  
  - **Dual Y-axis:** Left = TDC; Right = **required household income** for any TDC value (derived from the same VHFA table)

- **Affordability messaging** beneath Chart 2:
  - Exact equality counts as affordable (small epsilon)  
  - Clear states:
    - **All options affordable** (shows headroom to the most expensive)
    - **None affordable** (shows income needed for the cheapest and the shortfall)
    - **Some affordable**:
      - “Only the cheapest option is affordable” **or**
      - “The lowest _N_ options are affordable” **or**
      - Non-contiguous mix: lists the affordable options and the next shortfall

---

## Files in This Repository
- `data/assumptions.csv` — Policy cost assumptions (baseline $/sf, energy codes, energy sources, finish quality).  
  *(Infrastructure entries are currently ignored in TDC.)*
- `data/chittenden_ami.csv`, `data/addison_ami.csv`, `data/vermont_ami.csv` — VHFA purchase-price & income tables by AMI.  
  *(The app caps at 150% AMI and constrains the income input to the table’s min–max for the selected region & household size.)*
- `housing-affordability-tool.py` — Streamlit app
