# ===== Imports & Paths =====
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ===== Constants & Files =====
ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSUMP = DATA / "assumptions.csv"
REGIONS = {
    "Chittenden": DATA / "chittenden_ami.csv",
    "Addison":    DATA / "addison_ami.csv",
    "Vermont":    DATA / "vermont_ami.csv",
}
REGION_PRETTY = {"Chittenden": "Chittenden", "Addison": "Addison", "Vermont": "Rest of Vermont"}
PRETTY2REG = {v: k for k, v in REGION_PRETTY.items()}

# ===== Data Loading =====
@st.cache_data
def load_assumptions(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in ("category", "parent_option", "option", "value_type", "unit"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_regions(files: dict) -> dict:
    out = {}
    for name, p in files.items():
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        if "ami" in d:
            d["ami"] = pd.to_numeric(d["ami"], errors="coerce")  # fraction (e.g., 1.50 = 150%)
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers =====
PRETTY_OVERRIDES = {
    "townhome": "Townhome", "condo": "Condo", "apartment": "Apartment",
    "base_me_nh_code": "ME/NH Base Code", "vt_energy_code": "VT Energy Code",
    "evt_high_eff": "EVT High-Efficient", "passive_house": "Passive House",
    "natural_gas": "Natural Gas Heating", "all_electric": "All Electric", "geothermal": "Geothermal",
    "yes": "Yes", "no": "No", "above_average": "Above Average", "average": "Average", "below_average": "Below Average",
    "studio": "Studio",
}

def pretty(s: str) -> str:
    s = str(s).lower().strip()
    if s in PRETTY_OVERRIDES:
        return PRETTY_OVERRIDES[s]
    t = s.replace("_", " ").title()
    return (t.replace(" Ami", " AMI").replace(" Vt ", " VT ").replace(" Nh ", " NH ")
             .replace(" Me ", " ME ").replace(" Evt ", " EVT ").replace(" Mf ", " MF "))

def rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None:
        q &= A["parent_option"].eq(str(parent).lower())
    if opt is not None:
        q &= A["option"].eq(str(opt).lower())
    return A[q]

def options(cat, parent=None):
    return rows(cat, parent=parent)["option"].tolist()

def one_val(cat, opt, parent=None, expect_type=None):
    r = rows(cat, opt, parent)
    if r.empty and parent is not None:
        r = rows(cat, opt)
    if r.empty:
        return 0.0
    if expect_type and r.iloc[0]["value_type"] != expect_type:
        return 0.0
    return float(r.iloc[0]["value"])

def select_pretty(label, raw_options, key, default_raw=None, disabled=False):
    raw_options = list(raw_options)
    labels = [pretty(o) for o in raw_options]
    idx = raw_options.index(default_raw) if default_raw in raw_options else 0
    chosen = st.selectbox(label, labels, index=idx, key=key, disabled=disabled)
    return raw_options[labels.index(chosen)]

def bedroom_sf(h_type, br_label):
    r = rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor", "default", h_type)
def baseline_per_sf(): return one_val("baseline_cost", "baseline")

def pick_afford_col(b_int, unit_type):
    b = int(np.clip(b_int, 0, 5))
    return f"buy{max(1, b)}" if unit_type in ("townhome", "condo") else f"rent{b}"

def affordability_lines(region_pretty_list, amis, col):
    lines = {}
    for rp in region_pretty_list:
        region = PRETTY2REG[rp]
        df = R[region]
        for ami in amis:
            row = df[df["ami"].eq(ami / 100.0)]
            if not row.empty and col in row.columns:
                lines[f"{ami}% AMI - {rp}"] = float(row.iloc[0][col])
    return lines

def compute_tdc(sf, htype, energy_code, energy_source, infra, finish):
    base = baseline_per_sf()
    per_sf = base * mf_factor(htype)
    per_sf += base * (one_val("energy_code", energy_code) / 100.0)
    per_sf += base * (one_val("finish_quality", finish) / 100.0)
    per_sf += one_val("energy_source", energy_source, "default", "per_sf")
    per_sf += one_val("infrastructure", infra, "default", "per_sf")
    return sf * per_sf

def is_baseline(code, src, infra, fin) -> bool:
    return (code == "vt_energy_code" and src == "natural_gas" and infra == "no" and fin == "average")

def fmt_money(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return "—"

# ===== Header =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")
st.write("")

# ===== Product & Bedrooms =====
product = select_pretty(
    "What type of housing would you like to analyze?",
    ["townhome", "condo", "apartment"],
    key="global_product",
    default_raw="townhome"
)

if product in ("townhome", "condo"):
    br_opts_global = options("bedrooms", parent=product) or ["2"]
    br_default = "2" if "2" in br_opts_global else br_opts_global[0]
    bedrooms_global = select_pretty("Number of bedrooms", br_opts_global, key="global_bedrooms", default_raw=br_default)
    sf_global = bedroom_sf(product, bedrooms_global)
    if np.isnan(sf_global):
        st.warning("No SF mapping found; defaulting to 1,000 sf.")
        sf_global = 1000.0
else:
    bedrooms_global = None
    sf_global = None
    st.info("Apartment modeling (rent-based) coming soon. For now, choose Townhome or Condo to compare for-sale products.")

# ===== Per-Unit Policy Blocks =====
num_units = st.selectbox("How many units would you like to compare?", [1, 2, 3, 4, 5], index=1, disabled=(product == "apartment"))
units = []
disabled_block = (product == "apartment")
for i in range(num_units):
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
        code = select_pretty(
            "Energy code standard",
            options("energy_code", "default") or ["vt_energy_code"],
            key=f"code_{i}", default_raw="vt_energy_code", disabled=disabled_block
        )
        src = select_pretty(
            "Energy source",
            options("energy_source", "default") or ["natural_gas"],
            key=f"src_{i}", default_raw="natural_gas", disabled=disabled_block
        )
        infra = select_pretty(
            "Infrastructure required?",
            options("infrastructure", "default") or ["no", "yes"],
            key=f"infra_{i}", default_raw="no", disabled=disabled_block
        )
        fin = select_pretty(
            "Finish quality",
            options("finish_quality", "default") or ["average", "above_average", "below_average"],
            key=f"fin_{i}", default_raw="average", disabled=disabled_block
        )
        if disabled_block:
            st.caption("Policy selection disabled for Apartment placeholder.")
        units.append(dict(code=code, src=src, infra=infra, fin=fin))

# ===== Income Thresholds (header outside, controls inside) =====
st.subheader("Income Thresholds")
with st.container(border=True):
    region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
    sel_regions_pretty = st.multiselect(
        "Select region(s)",
        region_pretty_opts,
        default=[REGION_PRETTY["Chittenden"]]
    )
    valid_amis = [30] + list(range(50, 155, 5))
    n_amis = st.selectbox("How many Area Median Income (AMI) levels?", [1, 2, 3], index=0)
    amis = []
    default_cycle = [150, 120, 100]
    for i in range(n_amis):
        default_val = default_cycle[i] if i < len(default_cycle) and default_cycle[i] in valid_amis else 150
        amis.append(
            st.selectbox(
                f"AMI value #{i+1}",
                valid_amis,
                index=valid_amis.index(default_val),
                key=f"ami_{i}"
            )
        )

# ===== Chart =====
labels, tdc_vals, lines = [], [], {}
if product in ("townhome", "condo") and units:
    st.subheader("How did your choices affect affordability?")
    for i, u in enumerate(units, start=1):
        label = f"Baseline {pretty(product)}" if is_baseline(u["code"], u["src"], u["infra"], u["fin"]) else f"{pretty(product)} {i}"
        labels.append(label)
        tdc_vals.append(compute_tdc(sf_global, product, u["code"], u["src"], u["infra"], u["fin"]))

    b_int = int(bedrooms_global)  # 2,3,4
    aff_col = pick_afford_col(b_int, product)
    lines = affordability_lines(sel_regions_pretty, amis, aff_col)

if labels and tdc_vals:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(labels, tdc_vals, color="skyblue", edgecolor="black")
    ymax = max(tdc_vals + (list(lines.values()) or [0])) * 1.12
    ax1.set_ylim(0, ymax)

    for b in bars:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, y + (ymax * 0.02), f"${y:,.0f}",
                 ha="center", va="bottom", fontsize=10)

    for i, (lab, val) in enumerate(lines.items()):
        ax1.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)

    ax1.set_ylabel("Development Cost ($)")
    ax1.set_xlabel("TDC of Your Policy Choices")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=0)
    plt.title("Total Development Cost vs. What Buyers Can Afford")

    if lines:
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        vals = list(lines.values())
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split()[0]}\n${lines[k]:,.0f}" for k in lines])
        ax2.set_ylabel("Max. Affordable Purchase Price by % AMI")

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    st.pyplot(fig)
elif product == "apartment":
    st.info("Select Townhome or Condo to run the for‑sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")

# ===== Who Can Afford This Home? =====
st.subheader("Who Can Afford This Home?")
with st.container(border=True):
    region_single = st.selectbox(
        "Select The Region",
        [REGION_PRETTY[k] for k in REGIONS],
        index=[REGION_PRETTY[k] for k in REGIONS].index(REGION_PRETTY["Chittenden"])
    )
    household_size = st.selectbox("Select Household Size", list(range(1, 9)), index=3)

    # Render label as plain text (no commas; consistent font/size), collapse widget label
    st.write("Input Household Income ($20000 - $300000):")
    user_income = st.number_input(
        label="", label_visibility="collapsed",
        min_value=20000, max_value=300000, step=1000, value=100000, format="%d"
    )

    def affordability_sentence():
        if product not in ("townhome", "condo") or bedrooms_global is None:
            return "Affordability details are available for for-sale products (Townhome or Condo) only."

        reg_key = PRETTY2REG[region_single]
        df = R[reg_key]

        inc_col = f"income{household_size}"
        bed_n = int(bedrooms_global)
        buy_col = f"buy{bed_n}"

        if not {"ami", inc_col, buy_col}.issubset(df.columns):
            return "Required data not found for this region/household size."

        inc_series = pd.to_numeric(df[inc_col], errors="coerce")
        ami_series = pd.to_numeric(df["ami"], errors="coerce")   # fraction
        buy_series = pd.to_numeric(df[buy_col], errors="coerce")

        sub = pd.DataFrame({"income": inc_series, "ami_frac": ami_series, "buy": buy_series}).dropna()
        if sub.empty:
            return "Insufficient data to compute affordability."

        sub = sub.sort_values("income").reset_index(drop=True)
        floor_idx = sub[sub["income"] <= user_income].index.max()
        ceil_idx  = sub[sub["income"] >= user_income].index.min()

        # Choose row and set a clear edge-note when out of range
        if pd.isna(floor_idx) and pd.isna(ceil_idx):
            return "Insufficient data to compute affordability."
        if pd.isna(floor_idx):            # below minimum
            use_idx = 0
            edge_note = f" (closest to {sub.loc[use_idx, 'ami_frac']*100:.0f}% of AMI)"
        elif pd.isna(ceil_idx):           # above maximum
            use_idx = len(sub) - 1
            edge_note = f" (closest to {sub.loc[use_idx, 'ami_frac']*100:.0f}% of AMI)"
        else:
            use_idx = int(floor_idx)
            edge_note = ""

        sel_ami_pct = float(sub.loc[use_idx, "ami_frac"]) * 100.0
        sel_buy = float(sub.loc[use_idx, "buy"])

        return (f"A {household_size} person household making {fmt_money(user_income)} "
                f"({sel_ami_pct:.0f}% of AMI{edge_note}) can afford a "
                f"{fmt_money(sel_buy)} {bed_n} bedroom {pretty(product)}.")

    st.text(affordability_sentence())
