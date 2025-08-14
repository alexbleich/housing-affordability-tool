# housing-affordability-tool.py — Streamlit version (updated)

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------- Paths & region names ----------
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

# ---------- Loaders ----------
@st.cache_data
def load_assumptions(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in ("category","parent_option","option","value_type","unit"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_regions(files: dict) -> dict:
    out = {}
    for name, p in files.items():
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        if "ami" in d: d["ami"] = pd.to_numeric(d["ami"], errors="coerce")
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ---------- Pretty labels & helpers ----------
PRETTY_OVERRIDES = {
    "townhome":"Townhome","condo":"Condo","apartment":"Apartment",
    "base_me_nh_code":"ME/NH Base Code","vt_energy_code":"VT Energy Code",
    "evt_high_eff":"EVT High-Efficient","passive_house":"Passive House",
    "natural_gas":"Natural Gas Heating","all_electric":"All Electric","geothermal":"Geothermal",
    "yes":"Yes","no":"No","above_average":"Above Average","average":"Average","below_average":"Below Average",
    "studio":"Studio",
}
def pretty(s:str)->str:
    s = str(s).lower().strip()
    if s in PRETTY_OVERRIDES: return PRETTY_OVERRIDES[s]
    t = s.replace("_"," ").title()
    return (t.replace(" Ami"," AMI").replace(" Vt "," VT ").replace(" Nh "," NH ")
             .replace(" Me "," ME ").replace(" Evt "," EVT ").replace(" Mf "," MF "))

def rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    if opt    is not None: q &= A["option"].eq(str(opt).lower())
    return A[q]

def options(cat, parent=None):
    return rows(cat, parent=parent)["option"].tolist()

def one_val(cat, opt, parent=None, expect_type=None):
    r = rows(cat, opt, parent)
    if r.empty and parent is not None: r = rows(cat, opt)
    if r.empty: return 0.0
    if expect_type and r.iloc[0]["value_type"] != expect_type: return 0.0
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

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def pick_afford_col(b_int, unit_type):
    b = int(np.clip(b_int, 0, 5))
    return f"buy{max(1,b)}" if unit_type in ("townhome","condo") else f"rent{b}"  # placeholder for rent

def affordability_lines(region_pretty_list, amis, col):
    lines={}
    for rp in region_pretty_list:
        region = PRETTY2REG[rp]
        df = R[region]
        for ami in amis:
            row = df[df["ami"].eq(ami/100.0)]
            if not row.empty and col in row.columns:
                lines[f"{ami}% AMI - {rp}"] = float(row.iloc[0][col])
    return lines

def compute_tdc(sf, htype, energy_code, energy_source, infra, finish):
    base = baseline_per_sf()
    per_sf = base*mf_factor(htype)
    per_sf += base*(one_val("energy_code", energy_code)/100.0)
    per_sf += base*(one_val("finish_quality", finish)/100.0)
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

# ---------- User Interface ----------
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")
st.write("")

product = select_pretty("What type of housing would you like to analyze?",
                        ["townhome","condo","apartment"], key="global_product", default_raw="townhome")

if product in ("townhome","condo"):
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

# Number of units to compare
num_units = st.slider("How many units would you like to compare?", 1, 5, 2, disabled=(product=="apartment"))

# Per-unit policy choices
units=[]
disabled_block = (product=="apartment")
for i in range(num_units):
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
        code = select_pretty("Energy code standard",
                             options("energy_code","default") or ["vt_energy_code"],
                             key=f"code_{i}", default_raw="vt_energy_code", disabled=disabled_block)
        src  = select_pretty("Energy source",
                             options("energy_source","default") or ["natural_gas"],
                             key=f"src_{i}", default_raw="natural_gas", disabled=disabled_block)
        infra= select_pretty("Infrastructure required?",
                             options("infrastructure","default") or ["no","yes"],
                             key=f"infra_{i}", default_raw="no", disabled=disabled_block)
        fin  = select_pretty("Finish quality",
                             options("finish_quality","default") or ["average","above_average","below_average"],
                             key=f"fin_{i}", default_raw="average", disabled=disabled_block)
        if disabled_block: st.caption("Policy selection disabled for Apartment placeholder.")
        units.append(dict(code=code, src=src, infra=infra, fin=fin))

# Income thresholds (wrapped to match new informational boxes)
with st.container(border=True):
    st.subheader("Income Thresholds")
    region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
    sel_regions_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])
    valid_amis = [30] + list(range(50,155,5))
    n_amis = st.slider("How many Area Median Income (AMI) levels?", 1, 3, 1)  # label updated
    amis = [st.selectbox(f"AMI value #1", valid_amis, index=valid_amis.index(150), key="ami_0")]

# ---------- Compute & Plot (for-sale only) ----------
labels, tdc_vals, lines = [], [], {}
if product in ("townhome","condo") and units:
    # Informational box above graph
    with st.container(border=True):
        st.write("**How did your choices affect affordability?**")

    for i,u in enumerate(units, start=1):
        label = f"Baseline {pretty(product)}" if is_baseline(u["code"], u["src"], u["infra"], u["fin"]) else f"{pretty(product)} {i}"
        labels.append(label)
        tdc_vals.append(compute_tdc(sf_global, product, u["code"], u["src"], u["infra"], u["fin"]))

    b_int = 2 if bedrooms_global == "2" else int(bedrooms_global)
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
    ax1.set_xlabel("TDC of Your Policy Choices")  # label updated
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=0)

    plt.title("Total Development Cost vs. What Buyers Can Afford")  # title updated

    if lines:
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        vals = list(lines.values())
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split()[0]}\n${lines[k]:,.0f}" for k in lines])
        ax2.set_ylabel("Max. Affordable Purchase Price by % AMI")

    for (u), b in zip(units, bars):
        x_center = b.get_x() + b.get_width()/2.0
        bar_height = b.get_height()
        txt = (
            "Energy Code:\n"
            f"{pretty(u['code'])}\n\n"
            "Energy Source:\n"
            f"{pretty(u['src'])}\n\n"
            f"Infrastructure: {pretty(u['infra'])}\n\n"
            "Finish Quality:\n"
            f"{pretty(u['fin'])}"
        )
        ax1.text(x_center, bar_height * 0.05, txt, ha="center", va="bottom",
                 fontsize=10, linespacing=1.25, color="black", clip_on=True)

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    st.pyplot(fig)

elif product == "apartment":
    st.info("Select Townhome or Condo to run the for‑sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")

# Informational box below the link (matches "Income Thresholds" size)
with st.container(border=True):
    st.write("**Who Can Afford This Home?**")

# ---------- New controls under the chart ----------
# Single-choice region select (same options as Income Thresholds)
region_single = st.selectbox(
    "Select The Region",
    [REGION_PRETTY[k] for k in REGIONS],
    index=[REGION_PRETTY[k] for k in REGIONS].index(REGION_PRETTY["Chittenden"])
)

household_size = st.selectbox("Select Household Size", list(range(1, 8+1)), index=3)  # 1–8, default 4
user_income = st.number_input(
    "Input Household Income ($20,000-$300,000):",
    min_value=20000, max_value=300000, step=1000, value=100000, format="%d"
)

# ---------- Affordability sentence ----------
def affordability_sentence():
    if product not in ("townhome","condo") or bedrooms_global is None:
        return "Affordability details are available for for-sale products (Townhome or Condo) only."

    reg_key = PRETTY2REG[region_single]
    df = R[reg_key]
    inc_col = f"income{household_size}"
    bed_n = int(bedrooms_global)  # per your note: always 2, 3, or 4
    buy_col = f"buy{bed_n}"

    if inc_col not in df.columns or buy_col not in df.columns or "ami" not in df.columns:
        return "Required data not found for this region/household size. Please check your CSVs."

    series = pd.to_numeric(df[inc_col], errors="coerce")
    ami_series = pd.to_numeric(df["ami"], errors="coerce")  # stored as fraction (e.g., 1.5 for 150%)
    buy_series = pd.to_numeric(df[buy_col], errors="coerce")

    valid = series.notna() & ami_series.notna() & buy_series.notna()
    if not valid.any():
        return "Insufficient data to compute affordability."

    sub = df.loc[valid, [inc_col, "ami", buy_col]].sort_values(inc_col).reset_index(drop=True)

    # Find the floor match (<= user income). If none, handle edge cases.
    floor_idx = sub[sub[inc_col] <= user_income].index.max()
    ceil_idx = sub[sub[inc_col] >= user_income].index.min()

    if pd.isna(floor_idx) and pd.isna(ceil_idx):
        return "Insufficient data to compute affordability."

    # Determine the row to use and whether it's an edge case
    edge_note = ""
    if pd.isna(floor_idx):
        # Below minimum → use minimum row, note "closest to X% of AMI"
        use_idx = int(0)
        edge_note = " (closest tier)"
    elif pd.isna(ceil_idx):
        # Above maximum → use maximum row, note "closest to X% of AMI"
        use_idx = int(len(sub) - 1)
        edge_note = " (closest tier)"
    else:
        use_idx = int(floor_idx)

    sel_ami_pct = float(sub.loc[use_idx, "ami"]) * 100.0
    sel_buy = float(sub.loc[use_idx, buy_col])

    i = household_size
    income_str = fmt_money(user_income)
    buy_str = fmt_money(sel_buy)
    prod_str = pretty(product)

    return (
        f"A {i} person household making {income_str} "
        f"({sel_ami_pct:.0f}% of AMI{edge_note}) can afford a {buy_str} "
        f"{bed_n} bedroom {prod_str}."
    )

st.write("")
st.write(affordability_sentence())
