"""
Housing Affordability Visualizer

Interactive Streamlit app that compares estimated total development cost (TDC) for selected
unit types against income-based affordability thresholds across Vermont regions.
Pick housing type, bedrooms (with assumed SF), and project assumptions (energy code,
energy source, infrastructure, finish quality). Then choose region(s) and AMI levels to
see a side‑by‑side chart of what it costs to build versus what households can afford.

Purpose: illustrate how current development costs often exceed income-based purchase
(or rent, for apartments) thresholds, limiting demand and discouraging new market-rate
housing construction in Vermont.

Author: Alex Bleich
Last updated: August 11, 2025
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# -------------------- PATHS --------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSUMPTIONS_CSV = DATA_DIR / "assumptions.csv"
REGION_FILES = {
    "Chittenden": DATA_DIR / "chittenden_ami.csv",
    "Addison":    DATA_DIR / "addison_ami.csv",
    "Vermont":    DATA_DIR / "vermont_ami.csv",
}

# Public-facing names for the region selector
REGION_PRETTY = {
    "Chittenden": "Chittenden",
    "Addison": "Addison",
    "Vermont": "Rest of Vermont",
}
PRETTY_TO_REGION = {v: k for k, v in REGION_PRETTY.items()}

# -------------------- LOADERS --------------------
@st.cache_data
def load_assumptions(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in ["category", "parent_option", "option", "value_type", "unit"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_regions(files: dict) -> dict:
    out = {}
    for name, p in files.items():
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip().str.lower()
        if "ami" in df.columns:
            df["ami"] = pd.to_numeric(df["ami"], errors="coerce")
        out[name] = df
    return out

assump = load_assumptions(ASSUMPTIONS_CSV)
regions = load_regions(REGION_FILES)

# -------------------- PRETTY LABELS --------------------
PRETTY_MAP = {
    # housing types
    "townhome": "Townhome", "condo": "Condo", "apartment": "Apartment",
    # energy code
    "base_me_nh_code": "Base ME/NH Code",
    "vt_energy_code": "VT Energy Code",
    "evt_high_eff": "EVT High‑Efficient",
    "passive_house": "Passive House",
    # energy source
    "natural_gas": "Natural Gas",
    "all_electric": "All‑Electric",
    "geothermal": "Geothermal",
    # infrastructure
    "yes": "Yes", "no": "No",
    # finish quality
    "above_average": "Above Average", "average": "Average", "below_average": "Below Average",
    # bedroom label
    "studio": "Studio",
}

def pretty(s: str) -> str:
    s = str(s).lower().strip()
    if s in PRETTY_MAP:
        return PRETTY_MAP[s]
    txt = s.replace("_", " ").title()
    txt = (txt.replace(" Ami", " AMI")
              .replace(" Vt ", " VT ")
              .replace(" Nh ", " NH ")
              .replace(" Me ", " ME ")
              .replace(" Evt ", " EVT ")
              .replace(" Mf ", " MF "))
    return txt

def select_with_pretty(label, options, key, index=0):
    labels = [pretty(o) for o in options]
    chosen = st.selectbox(label, labels, index=index, key=key)
    return options[labels.index(chosen)]

def idx_of(options, raw_value, fallback_index=0):
    raw_value = str(raw_value).lower()
    try:
        return options.index(raw_value)
    except ValueError:
        return min(max(fallback_index, 0), max(len(options)-1, 0))

# -------------------- ASSUMPTION HELPERS --------------------
def rows(category, option=None, parent=None):
    q = assump["category"].eq(category)
    if parent is not None:
        q &= assump["parent_option"].eq(str(parent).lower())
    if option is not None:
        q &= assump["option"].eq(str(option).lower())
    return assump[q]

def list_options(category, parent=None):
    r = rows(category, parent=parent)
    return r["option"].tolist()

def baseline_per_sf() -> float:
    r = rows("baseline_cost", option="baseline")
    return float(r.iloc[0]["value"]) if not r.empty else 0.0

def mf_factor(housing_type: str) -> float:
    r = rows("mf_efficiency_factor", option="default", parent=housing_type)
    return float(r.iloc[0]["value"]) if not r.empty else 1.0

def bedroom_sf(housing_type: str, bedroom_label: str) -> float:
    r = rows("bedrooms", option=bedroom_label, parent=housing_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def percent_val(category: str, option: str) -> float:
    r = rows(category, option=option, parent="default")
    if r.empty:
        r = rows(category, option=option)
    return float(r.iloc[0]["value"]) if not r.empty else 0.0

def per_sf_val(category: str, option: str) -> float:
    r = rows(category, option=option, parent="default")
    if r.empty:
        r = rows(category, option=option)
    if r.empty or r.iloc[0]["value_type"] != "per_sf":
        return 0.0
    return float(r.iloc[0]["value"])

def pick_afford_col(bedrooms_int: int, unit_type: str) -> str:
    """
    Townhome/Condo -> purchase thresholds 'buy{1..5}'
    Apartment -> rent thresholds 'rent{0..5}' (placeholder for later)
    """
    b = int(np.clip(bedrooms_int, 0, 5))
    if unit_type in ("townhome", "condo"):
        return f"buy{max(1, b)}"
    return f"rent{b}"  # TODO: wire true rent logic later

def affordability_lines(selected_regions, amis, column_name):
    lines = {}
    for region in selected_regions:
        df = regions[region]
        for ami in amis:
            r = df[df["ami"].eq(ami / 100.0)]
            if not r.empty and column_name in r.columns:
                lines[f"{ami}% AMI - {REGION_PRETTY.get(region, region)}"] = float(r.iloc[0][column_name])
    return lines

def compute_tdc(sf: float, utype: str, energy_code: str, energy_source: str,
                infrastructure: str, finish_quality: str) -> float:
    base = baseline_per_sf()                 # e.g., 400 $/sf
    base_eff = base * mf_factor(utype)       # apply MF factor to baseline $/sf

    # % add-ons (applied to the baseline $/sf, per spec)
    pct_total = (percent_val("energy_code", energy_code)
                 + percent_val("finish_quality", finish_quality))
    add_per_sf_from_pct = base * (pct_total / 100.0)

    # $/sf add-ons (energy source + infrastructure) — infrastructure is $/sf only
    add_per_sf_energy = per_sf_val("energy_source", energy_source)
    infra_per_sf = per_sf_val("infrastructure", infrastructure)

    per_sf_sum = base_eff + add_per_sf_from_pct + add_per_sf_energy + infra_per_sf
    return sf * per_sf_sum

# -------------------- UI --------------------
st.title("🏘️ Housing Affordability Visualizer")
st.write("Compare unit development costs with AMI-based affordability thresholds across Vermont regions.")

num_units = st.slider("How many units would you like to compare?", 1, 5, 2)

units = []
housing_types = ["townhome", "condo", "apartment"]

for i in range(num_units):
    st.subheader(f"Unit {i+1}")
    with st.container(border=True):
        # Unit type (default to Townhome)
        utype = select_with_pretty("Unit type", housing_types, key=f"type_{i}", index=0)

        # Bedrooms (default to "2" if available)
        br_opts = list_options("bedrooms", parent=utype) or ["2"]
        default_idx = br_opts.index("2") if "2" in br_opts else min(1, len(br_opts)-1)
        br = select_with_pretty("Number of bedrooms", br_opts, key=f"br_{i}", index=default_idx)

        # SF mapping
        sf = bedroom_sf(utype, br)
        if np.isnan(sf):
            st.warning("No SF mapping found for this selection; defaulting to 1,000 sf.")
            sf = 1000.0

        st.caption(f"MF Efficiency Factor: {int(round(mf_factor(utype)*100))}% (applied to baseline $/sf)")

        # Defaults requested:
        # Energy code standard = "VT Energy Code"
        code_opts = list_options("energy_code", parent="default") or ["base_me_nh_code", "vt_energy_code"]
        code_choice = select_with_pretty("Energy code standard",
                                         code_opts,
                                         key=f"code_{i}",
                                         index=idx_of(code_opts, "vt_energy_code"))

        # Energy source = "Natural Gas"
        src_opts = list_options("energy_source", parent="default") or ["natural_gas"]
        source_choice = select_with_pretty("Energy source",
                                           src_opts,
                                           key=f"src_{i}",
                                           index=idx_of(src_opts, "natural_gas"))

        # Infrastructure required = "No"
        infra_opts = list_options("infrastructure", parent="default") or ["no", "yes"]
        infra_choice = select_with_pretty("Infrastructure required?",
                                          infra_opts,
                                          key=f"infra_{i}",
                                          index=idx_of(infra_opts, "no"))

        # Finish quality = "Average"
        finish_opts = list_options("finish_quality", parent="default") or ["average", "above_average", "below_average"]
        finish_choice = select_with_pretty("Finish quality",
                                           finish_opts,
                                           key=f"finish_{i}",
                                           index=idx_of(finish_opts, "average"))

        st.caption(
            f"Selected: {pretty(utype)} • {pretty(br)} BR • "
            f"{pretty(code_choice)} • {pretty(source_choice)} • "
            f"Infrastructure: {pretty(infra_choice)} • Finish: {pretty(finish_choice)}"
        )

        units.append({
            "type": utype,
            "bed_label": br,
            "sf": float(sf),
            "energy_code": code_choice,
            "energy_source": source_choice,
            "infrastructure": infra_choice,
            "finish_quality": finish_choice,
        })

# -------------------- Income thresholds --------------------
st.subheader("Income Thresholds")

region_pretty_opts = [REGION_PRETTY[k] for k in REGION_FILES.keys()]
selected_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])
selected_regions = [PRETTY_TO_REGION[p] for p in selected_pretty]

valid_ami_values = [30] + list(range(50, 155, 5))
num_amis = st.slider("How many AMI levels?", 1, 3, 2)
ami_list = []
for j in range(num_amis):
    default = 100 if j == 0 else 150
    ami_list.append(st.selectbox(f"AMI value #{j+1}",
                                 valid_ami_values,
                                 index=valid_ami_values.index(default),
                                 key=f"ami_{j}"))

# -------------------- CALCULATE --------------------
labels, tdcs, br_ints = [], [], []
for i, u in enumerate(units):
    # convert bedroom label -> int for thresholds (studio -> 0)
    if str(u["bed_label"]).lower() == "studio":
        b_int = 0
    else:
        try:
            b_int = int(u["bed_label"])
        except Exception:
            b_int = 2
    br_ints.append(max(1, b_int))  # buy columns are 1..5; studio treated as 1 for buy

    tdcs.append(compute_tdc(u["sf"], u["type"], u["energy_code"], u["energy_source"],
                            u["infrastructure"], u["finish_quality"]))
    # Unique label so bars never collapse, even if type + SF match
    labels.append(f"Unit {i+1}: {int(u['sf'])}sf {pretty(u['type'])}")

# Bedroom count for AMI lines = rounded mean across units (1–5)
line_bedrooms = int(np.clip(round(np.mean(br_ints) if br_ints else 1), 1, 5))
unit_for_lines = "apartment" if any(u["type"] == "apartment" for u in units) else "townhome"
aff_col = pick_afford_col(line_bedrooms, unit_for_lines)
lines = affordability_lines(selected_regions, ami_list, aff_col)

# -------------------- PLOT --------------------
if labels and tdcs:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(labels, tdcs, color="skyblue", edgecolor="black")
    ymax = max(tdcs + (list(lines.values()) or [0])) * 1.12
    ax1.set_ylim(0, ymax)

    for b in bars:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, y + (ymax*0.02), f"${y:,.0f}",
                 ha="center", va="bottom", fontsize=9)

    for i, (lab, val) in enumerate(lines.items()):
        ax1.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)

    ax1.set_ylabel("Development Cost ($)")
    ax1.set_xlabel("Unit & Type")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=20)
    plt.title("Development Cost vs. Affordable Purchase Price Thresholds")
    if lines:
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        vals = list(lines.values())
        labs = [f"{k.split()[0]}\n${lines[k]:,.0f}" for k in lines]
        ax2.set_yticks(vals)
        ylabel = ("Max. Affordable Purchase Price by % AMI"
                  if "buy" in aff_col else
                  "Max. Affordable Gross Rent by % AMI (incl. utilities) — TODO Apartment")
        ax2.set_ylabel(ylabel)
        ax2.set_yticklabels(labs)

    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("No valid unit data provided.")

st.caption("Apartment path currently uses a rent-column placeholder. We’ll wire rent thresholds when you’re ready.")
