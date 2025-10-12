# Assumptions

This page intends to provide a clear explanation of how inputs in `assumptions.csv` flow through the model, how prices map to income, and what simplifications are made. This is written for non-coders. If anything is ambiguous, please reach out to Summit Properties.

## What the model does (at a glance)

- You pick a **home type** (Townhome or Condo), **bedrooms**, and a few **build options** (energy code, heating source, finish quality, and whether it’s in a new neighborhood).
- The app calculates a **Total Development Cost (TDC)** from transparent line items in `assumptions.csv`.
- It turns the TDC into the **household income needed to buy** using VHFA affordability tables.
- It estimates **what share of Vermont households** could afford that price and the approximate **% of AMI**.

> *Apartments (rent) are planned but not enabled yet.*

## Data sources used

- `data/assumptions.csv` — all cost inputs used to build up TDC.
- `vhfa data/*_ami.csv` — VHFA affordability tables by region (Addison, Chittenden, “Rest of Vermont”).
- `data/vt_inc_dist.csv` — statewide household income distribution (bins of income and # of households).

## CSV schema (how we read assumptions)

Each row in `assumptions.csv` has:

- **category** — e.g., `baseline_hard_cost`, `energy_code`, `finish_quality`, `energy_source`, `new_neighborhood`, `acq_cost`, `bedrooms`, and others.
- **parent_option** — which product it applies to (`townhome`, `condo`, or `default`).
- **option** — the selectable option name (e.g., `rbes`, `passive_house`, `average`, `yes`, `no`, or `default`).
- **value_type** — one of:
  - `per_sf`  → dollars **per square foot**
  - `per_unit` → dollars **per unit**
  - `fixed`   → a **flat** dollar amount
- **value** — the numeric amount.

**Normalization rules**:
- Case/spacing/underscores don’t matter; the app lower-cases and trims.
- Synonyms like `psf`, `sf`, `perunit`, `fixedcost`, etc., are normalized to the three types above.
- Unknown or missing values are treated as zero for safety.

## Bedrooms and square footage

- `category = bedrooms`, with `parent_option` = product (`townhome` or `condo`) and `option` = the bedroom count (`1`, `2`, `3`, `4`).
- The **value** is the **assumed square footage** for that product/bed count.
- If a bedroom row is missing, the app falls back to a simple default (e.g., 1,000 sf).

## TDC formula (how costs are built)

Let **sf** be the square footage for the chosen product + bedrooms. The **per-sf** and **per-unit** items come from `assumptions.csv`.

1) **Baseline hard cost per sf**
- `baseline_hard_cost` (per-sf) × **multipliers**:

   - **Multi-family efficiency**: `mf_efficiency_factor`(product), default ≈ `1.0`.
   - **Energy code** + **finish quality**: we read the option values as **percent adjustments**.  
     If energy code is `+10%` and finish is `+5%`, together they add **+15%** to the base.

   *Implementation:*  
   `hard_psf = baseline_hard_psf * (mf_factor + (energy_code% + finish_quality%) / 100)`  
   then **soft costs** are applied:

2) **Soft costs**  
- `soft_cost` is a **percent** added *on top* of the hard-cost subtotal:  
  `hard_psf = hard_psf * (1 + soft_cost%/100)`

3) **Energy source, land/acquisition, other defaults**  
- For categories `energy_source`, `acq_cost` (uses option `baseline`), and any other *defaulted* categories, we sum their:
  - `per_sf` → added to the per-sf stack  
  - `per_unit` → added as per-unit items  
  - `fixed` → added once

4) **Infrastructure toggle (new neighborhood?)**  
- `new_neighborhood` has options `yes`/`no` **per product** and contributes a **per-unit** amount when `yes`.

**Final TDC:**
- `TDC = sf * (hard_psf + per_sf_adders) + (per_unit_adders + infra_per_unit) + fixed_adders`

Where:
- *per_sf_adders* = energy source (`per_sf`) + acq (`per_sf`) + other defaults (`per_sf`)  
- *per_unit_adders* = energy source (`per_unit`) + acq (`per_unit`) + other defaults (`per_unit`)  
- *fixed_adders* = energy source (`fixed`) + acq (`fixed`) + other defaults (`fixed`)

## Price ↔ Income mapping (VHFA tables)

- Each `*_ami.csv` has an `ami` column and columns like `buy1`, `buy2`, … and `income1`, `income2`, …
- We select the **buyN**/**incomeN** pair based on **household size**:

  | HH size | Buy column used |
  |---------|------------------|
  | 1       | `buy1`           |
  | 2       | `buy2`           |
  | 3–4     | `buy3`           |
  | 5–6     | `buy4`           |
  | 7–8     | `buy5`           |

  If a `buyN` isn’t present, we **clamp** to the nearest available (e.g., use `buy3`).

- **Interpolation:** inside the range of the table, we **linearly interpolate** between points.
- **Extrapolation (edge behavior):**
  - Below the lowest price or above the highest price, we **extend the line** using the slope of the nearest segment.  
    This keeps results reasonable just outside the table but should be interpreted with care.

## AMI percentage shown

- Given a **required income**, we look up the region’s `ami` vs. `incomeN` table.
- We return:
  - **30%** if below the lowest income point (capped low).
  - **150%** if above the highest income point (capped high).
  - Otherwise the **nearest AMI** for incomes ≤ the required income (rounded to a whole %).

## Share of VT households who could afford it

- `vt_inc_dist.csv` contains income bins for Vermont households (`lower`, `upper`, `num_hhs`).
- We estimate **households at or above** the required income by:
  - Summing all bins with `lower ≥ required_income`, plus
  - A **partial** share of the bin that straddles the threshold, assuming **uniform distribution within the bin**.
- We then report the **count** and the **percent of 270,000 households** (denominator can be changed in code if needed).

## Defaults & guardrails

- **Default components** when you start: energy code = *VT energy code*, heating = *natural gas*, finish = *average*, location = *no (not a new neighborhood)*.
- If a CSV is missing or a column is malformed, the app stops with a clear error message.
- All numeric fields are coerced; any non-numeric values become zero (conservative).

## What to keep in mind (limits)

- This is a **policy discussion tool**, not a project-level pro forma.
- **Per-sf/per-unit/fixed** splits and **percent adders** should be reviewed periodically to reflect current market conditions.
- **Extrapolation** beyond the VHFA table is linear and should be treated as approximate.
- Household share uses a **statewide** income distribution; county-level distributions would change results.

## Updating numbers

1. Edit `assumptions.csv` (costs, toggles, amounts).  
2. Update AMI tables in `vhfa data/` if VHFA releases new versions.  
3. Update `vt_inc_dist.csv` if newer statewide income distribution data are available.  
4. Commit to GitHub — the Streamlit app redeploys automatically.

*If you change column names or add new categories/options, reflect them in `assumptions.csv` and keep the value types to `per_sf`, `per_unit`, or `fixed`.*
