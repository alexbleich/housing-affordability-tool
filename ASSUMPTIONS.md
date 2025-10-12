🧾 ASSUMPTIONS.md

This document explains how the app reads and uses values from the CSV files to compute total development cost (TDC), map prices to incomes, and estimate how many Vermont households could afford a given home. Everything below is implemented in the app and driven entirely by the CSVs—no hard-coded dollar amounts.

Files used by the app

data/assumptions.csv – policy & cost inputs (see schema below)

data/chittenden_ami.csv, data/addison_ami.csv, data/vermont_ami.csv – VHFA purchase-price & income by AMI and household size

data/vt_inc_dist.csv – Vermont household income distribution (bin upper bound, number of households, percent of households)

assumptions.csv schema

Columns (lowercase):

category – cost/setting bucket (e.g., baseline_hard_cost, soft_cost, bedrooms, energy_code, energy_source, finish_quality, new_neighborhood, acq_cost)

parent_option – scope: usually the product type (townhome, condo) or default

option – the specific choice name (e.g., baseline, average, vt_energy_code, rbes, passive_house, yes, no, or bedroom counts like 2, 3, 4)

value_type – one of:

per_sf (applies per square foot)

per_unit (applies once per home)

fixed (applies once per home; does not scale with size or count)

value – numeric cost or percentage (blank/NaN hides an option in the UI)

Notes

All identifiers are read in lowercase and trimmed.

If the same category has both a product-specific row (e.g., parent_option=townhome) and a parent_option=default row, the app adds them together where appropriate.

Any additional categories with option=default act as silent “other defaults” and are included automatically.

How each category is used

baseline_hard_cost

option=baseline, value_type=per_sf → baseline hard cost per square foot.

mf_efficiency_factor

option=default, value_type=per_sf-like multiplier stored as a percentage adder (e.g., 1.00 means 100% of baseline).

May vary by product via parent_option.

soft_cost

option=baseline, percentage adder applied to the hard-cost subtotal only:
hard_psf *= (1 + soft_cost%).

bedrooms

For each product (parent_option=townhome or condo), option is the bedroom count (1,2,3,4) and value is the square footage for that layout.

The app uses these to show only valid bedroom options (e.g., no 1-BR townhomes if absent in the CSV).

energy_code (percent adders to hard cost)

Rows like vt_energy_code, rbes, passive_house with value as a percent adder.

finish_quality (percent adder to hard cost)

Rows like below_average, average, above_average with value as a percent adder.

energy_source (overlays)

May include any combination of per_sf, per_unit, fixed rows.

Both parent_option=default and product-specific rows are summed.

new_neighborhood (toggle overlay)

option=yes/no with typically per_unit values captured when the “In a new neighborhood” toggle is on.

acq_cost (acquisition cost)

Read from option=baseline (usually per_unit=18000).

Added once per home; if present as per_sf or fixed, those are also included.

Other default overlays

Any other category with option=default is added automatically (per_sf + per_unit + fixed), for parent_option in {product, default}.

TDC formula (per home)

The app computes TDC using only values from assumptions.csv:

hard_psf_before_soft = baseline_hard_psf * (mf_efficiency_factor + energy_code% + finish_quality%)
hard_psf             = hard_psf_before_soft * (1 + soft_cost%)

per_sf_overlays   = energy_source.per_sf + other_defaults.per_sf + acq_cost.per_sf
per_unit_overlays = energy_source.per_unit + new_neighborhood.per_unit
                    + other_defaults.per_unit + acq_cost.per_unit
fixed_overlays    = energy_source.fixed + other_defaults.fixed + acq_cost.fixed

TDC = sf * (hard_psf + per_sf_overlays) + per_unit_overlays + fixed_overlays


sf comes from the bedrooms row for the chosen product & bedroom count.

“Other defaults” = any additional category with option=default not listed above.

Price ↔ Income mapping (by household size)

From each region table (chittenden_ami.csv, addison_ami.csv, vermont_ami.csv) the app builds a two-way mapping between purchase price (buyN) and household income (incomeN).

It selects the buy column by household size:

1-person → buy1

2-person → buy2

3–4-person → buy3

5–6-person → buy4

7–8-person → buy5

Within the table range, the mapping is linear interpolation between the two surrounding rows.

Outside the table range, the mapping uses linear extrapolation using the slope of the nearest segment.

The app caps AMI displays at 30% (as “at least”) and 150% (as “over 150%”).

Income distribution method (who can afford it?)

data/vt_inc_dist.csv has upper bounds of income bins (hh_income), household counts (num_hhs), and percent of households (percent_hhs).

To estimate how many households can afford a required income X:

Full bins above X are counted completely.

For the bin that straddles X, the app adds a linear fraction of the bin:
fraction = (upper - X) / (upper - lower).

Sum that total and express it as N households and % of 270,000.

This produces smooth counts/percentages when X falls inside a bin (e.g., at $80k within the $75–$100k bin, roughly 80% of that bin is counted).

Bedroom choices vs. income mapping

Bedroom choice only affects square footage (and thus the TDC bars).

Required income comes from the household-size mapping above (not from bedrooms), which matches how mortgage underwriting and VHFA tables are structured.

Updating assumptions

To add/remove bedroom options, edit the bedrooms rows for each product.

To enable new toggles later, add a category with option=default rows so the app includes it silently now; add UI controls in a future commit.

To change acquisition policy, update acq_cost rows; the app reads them automatically.

Data provenance

VHFA affordability data (purchase price & income by AMI and household size) comes from the linked PDFs/tables referenced in the app.

Vermont household income distribution is adapted to bins with upper bounds + counts + percents to support linear interpolation.
