# Assumptions, Methods, & Data Schema

This is a technical appendix for the Housing Affordability Visualizer *written for non-coders who want transparency*. This document explains how inputs are read, how costs are combined, how price maps to income, and what safeguards and caveats apply.

## 1) Data Files (and the columns expected)

### 1.1 `data/assumptions.csv`
Each row defines an input the model can use.

- **category** — e.g., `baseline_hard_cost`, `soft_cost`, `bedrooms`, `energy_code`, `finish_quality`, etc.
- **parent_option** — which product it applies to: `townhome`, `condo`, or `default`.
- **option** — the setting within a category (e.g., `rbes`, `passive_house`, `average`, `yes`, `no`, `default`, or a bedroom count like `2`).
- **value_type** — one of:
  - `per_sf`   → dollars **per square foot**
  - `per_unit` → dollars **per unit**
  - `fixed`    → dollars added **once** per comparison
  - `percent`  → **percent added** to TDC
- **value** — numeric amount.

**Normalization & validation**
- Text is lower-cased and trimmed; `psf`, `sf`, `perunit`, `fixedcost` are used in the program and refer to the value types above.
- Missing required columns stop the app with a clear error message.

### 1.2 `vhfa data/*.csv` (Addison, Chittenden, Rest of Vermont)
Affordability tables by region. Expected columns:

- `ami` — % AMI (e.g., 0.30 = 30%, 0.50 = 50%, …, 1.50 = 150%).
- `buy1`, `buy2`, … — affordable purchase price by household size.
- `income1`, `income2`, … — corresponding household incomes.
> The program clamps to the nearest available income by household size to determine AMI level if an exact match is missing.

### 1.3 `data/vt_inc_dist.csv`
Statewide household income distribution. Expected columns:

- `hh_income` — upper bound of income bins (e.g., if a bin is $10k-$15k, '15000' is the entry).
- `num_hhs` — estimated number of households in the bin.
- `percent_hhs` — estimated percent of households in the bin.
> Bins must have **positive width** (`upper > lower`).

## 2) Bedrooms → Square Footage
- Category: `bedrooms`
- `parent_option`: `townhome` or `condo`
- `option`: the bedroom count (`1`, `2`, `3`, `4`)
- `value`: **assumed square footage** for that product/bed count.

## 3) Cost Model — From Line Items to Total Development Cost (TDC)

Let:
- **sf** = selected product’s square footage (from the bedrooms table)
- **Base Hard Cost (per sf)** = `baseline_hard_cost` (category)
- **MF efficiency factor** = `mf_efficiency_factor` (multiplier by product)
- **Energy code %** and **Finish quality %** = percentage adjustments
- **Soft cost %** = percentage applied after hard cost subtotal

**Step-by-step**

1. **Hard cost before soft costs (per sf)**
   - Start with `baseline_hard_cost` (per sf).
   - Add percentage adjustments:
     - Multi-family factor (`mf_efficiency_factor`) is a multiplier.
     - `energy_code` (%) + `finish_quality` (%) are **adders**.
   - Formula:
     - hard_psf_before_soft = baseline_hard_psf * ( mf_factor + (energy_code_pct + finish_quality_pct)/100 )

2. **Apply soft costs (per sf)**
     - hard_psf = hard_psf_before_soft * (1 + soft_cost_pct/100)

4. **Add other components**
- For categories with `default` or the selected options (`energy_source`, `acq_cost`, and any other defaults), their **per_sf**, **per_unit**, and **fixed** values are summed:
  - `per_sf_adders`  = sum of all relevant per-sf items
  - `per_unit_adders` = sum of all relevant per-unit items
  - `fixed_adders`   = sum of all relevant fixed items

4. **Infrastructure toggle**
- Category: `new_neighborhood` with options `yes`/`no` (by product).
- Contributes a **per-unit** amount when `yes`.

**Final TDC:**
- TDC = sf * ( hard_psf + per_sf_adders ) + ( per_unit_adders + infra_per_unit ) + fixed_adders

> **Sign convention:** percent adders may be positive or negative (e.g., rebates). Use care when entering values in `assumptions.csv`.

## 4) Mapping Price ↔ Income (VHFA Tables)

**Household size → buyN/incomeN**
- 1 → `buy1`/`income1`
- 2 → `buy2`/`income2`
- 3–4 → `buy3`/`income3`
- 5–6 → `buy4`/`income4`
- 7–8 → `buy5`/`income5`

**Interpolation (inside the table)**
- The program **linearly interpolates** between tabulated points to convert:
  - **price → income** (*price_to_income*),
  - **income → price** (*income_to_price*).

**Extrapolation (edges)**
- Just outside the table range, the program extends using the **nearest segment’s slope**.
- This keeps results continuous but should be treated as **approximate**.

## 5) Displaying % of AMI

Given the **required income** (from price), the program looks up the region’s AMI table to estimate **% of AMI**:

- If below the lowest tabulated income, show **30% (at least)**.
- If above the highest, show **over 150%**.
- Otherwise, the program selects the nearest AMI level **at or below** the required income and round to a whole %.

## 6) Share of Vermont Households Who Could Afford

Using `vt_inc_dist.csv`:

1. Sum all bins with `hh_income ≥ required_income`.
2. For the bin that **straddles** the threshold, add a **partial** share assuming a **uniform distribution within the bin**.
3. Report:
   - **count** of households at/above, and
   - **percent** of a default denominator (≈ 270,000 Vermont households).
     - The denominator can be changed in code if newer statewide totals are preferred.

## 7) Defaults, Error Handling, & Practical Caveats

- **Default components** at app start: energy code = *regional standard*, heating = *natural gas*, finish = *average*, location = *not a new neighborhood*.
- **Missing files/columns:** the app stops with a clear error message indicating what’s missing.
- Results are **directional** and sensitive to:
  - Cost entries in `assumptions.csv`,
  - Table versions in `vhfa data/*.csv`,
  - The assumed shape of `vt_inc_dist.csv` bins (uniform within bin).
