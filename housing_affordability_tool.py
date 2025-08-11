"""
Housing Affordability Visualizer

Streamlit app comparing total development cost (TDC) per unit against AMI-based
affordability thresholds across VT regions. Choose type, bedrooms (assumed SF),
and assumptions (energy code, source, infrastructure, finish). Then pick region(s)
and AMIs to see TDC vs household affordability.

Author: Alex Bleich
Last updated: August 11, 2025
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----- Paths & Region naming -----
ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSUMP = DATA / "assumptions.csv"
REGIONS = {"Chittenden": DATA/"chittenden_ami.csv",
           "Addison": DATA/"addison_ami.csv",
           "Vermont": DATA/"vermont_ami.csv"}
REGION_PRETTY = {"Chittenden": "Chittenden",
                 "Addison": "Addison",
                 "Vermont": "Rest of Vermont"}
PRETTY2REG = {v:k for k,v in REGION_PRETTY.items()}

# ----- Loaders -----
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

# ----- Small utilities -----
PRETTY_OVERRIDES = {
    "townhome":"Townhome","condo":"Condo","apartment":"Apartment",
    "base_me_nh_code":"Base ME/NH Code","vt_energy_code":"VT Energy Code",
    "evt_high_eff":"EVT High-Efficient","passive_house":"Passive House",
    "natural_gas":"Natural Gas","all_electric":"All-Electric","geothermal":"Geothermal",
    "yes":"Yes","no":"No","above_average":"Above Average","average":"Average",
    "below_average":"Below Average","studio":"Studio",
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
    if r.empty and parent is not None: r = rows(cat, opt)  # fallback
    if r.empty: return 0.0
    if expect_type and r.iloc[0]["value_type"] != expect_type: return 0.0
    return float(r.iloc[0]["value"])

def select_pretty(label, raw_options, key, default_raw=None):
    raw_options = list(raw_options)
    labels = [pretty(o) for o in raw_options]
    idx = raw_options.index(default_raw) if default_raw in raw_options else 0
    chosen = st.selectbox(label, labels, index=idx, key=key)
    return raw_options[labels.index(chosen)]

def bedroom_sf(h_type, br_label):
    r = rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def pick_afford_col(b_int, unit_type):
    b = int(np.clip(b_int, 0, 5))
    return f"buy{max(1,b)}" if unit_type in ("townhome","condo") else f"rent{b}"  # TODO rent mapping

def affordability_lines(regions_pretty, amis, col):
    lines={}
    for rp in regions_pretty:
        region = PRETTY2REG[rp]
        df = R[region]
        for ami in amis:
            row = df[df["ami"].eq(ami/100.0)]
            if not row.empty and col in row.columns:
                lines[f"{ami}% AMI - {rp}"] = float(row.iloc[0][col])
    return lines

def compute_tdc(sf, htype, energy_code, energy_source, infra, finish):
    base = baseline_per_sf()
    per_sf = base*mf_factor(htype)                               # MF on baseline
    per_sf += base*(one_val("energy_code", energy_code)/100.0)   # % of baseline
    per_sf += base*(one_val("finish_quality", finish)/100.0)     # % of baseline
    per_sf += one_val("energy_source", energy_source, expect_type="per_sf")
    per_sf += one_val("infrastructure", infra, expect_type="per_sf")  # $/sf only
    return sf * per_sf

# ----- UI -----
st.title("🏘️ Housing Affordability Visualizer")
st.write("Compare unit development costs with AMI-based affordability thresholds across Vermont regions.")

num_units = st.slider("How many units would you like to compare?", 1, 5, 2)
units=[]
for i in range(num_units):
    st.subheader(f"Unit {i+1}")
    with st.container(border=True):
        h_types = ["townhome","condo","apartment"]
        h = select_pretty("Unit type", h_types, key=f"type_{i}", default_raw="townhome")

        br_opts = options("bedrooms", parent=h) or ["2"]
        br_default = "2" if "2" in br_opts else br_opts[0]
        br = select_pretty("Number of bedrooms", br_opts, key=f"br_{i}", default_raw=br_default)

        sf = bedroom_sf(h, br)
        if np.isnan(sf):
            st.warning("No SF mapping found; defaulting to 1,000 sf.")
            sf = 1000.0

        st.caption(f"MF Efficiency Factor: {int(round(mf_factor(h)*100))}% (applied to baseline $/sf)")

        code = select_pretty("Energy code standard",
                             options("energy_code","default"),
                             key=f"code_{i}", default_raw="vt_energy_code")
        src  = select_pretty("Energy source",
                             options("energy_source","default"),
                             key=f"src_{i}", default_raw="natural_gas")
        infra= select_pretty("Infrastructure required?",
                             options("infrastructure","default"),
                             key=f"infra_{i}", default_raw="no")
        fin  = select_pretty("Finish quality",
                             options("finish_quality","default"),
                             key=f"fin_{i}", default_raw="average")

        st.caption(f"Selected: {pretty(h)} • {pretty(br)} BR • {pretty(code)} • {pretty(src)} • "
                   f"Infrastructure: {pretty(infra)} • Finish: {pretty(fin)}")

        units.append(dict(htype=h, br=br, sf=float(sf), code=code, src=src, infra=infra, fin=fin))

# ----- Income thresholds -----
st.subheader("Income Thresholds")
region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
sel_regions_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])

valid_amis = [30] + list(range(50,155,5))
n_amis = st.slider("How many AMI levels?", 1, 3, 2)
amis = [st.selectbox(f"AMI value #{j+1}", valid_amis,
                     index=valid_amis.index(100 if j==0 else 150), key=f"ami_{j}") for j in range(n_amis)]

# ----- Compute -----
labels, tdc_vals, br_ints = [], [], []
for i,u in enumerate(units):
    br_i = 0 if str(u["br"]).lower()=="studio" else (int(u["br"]) if str(u["br"]).isdigit() else 2)
    br_ints.append(max(1, br_i))
    tdc_vals.append(compute_tdc(u["sf"], u["htype"], u["code"], u["src"], u["infra"], u["fin"]))
    labels.append(f"Unit {i+1}: {int(u['sf'])}sf {pretty(u['htype'])}")

line_beds = int(np.clip(round(np.mean(br_ints) if br_ints else 1), 1, 5))
unit_for_lines = "apartment" if any(u["htype"]=="apartment" for u in units) else "townhome"
aff_col = pick_afford_col(line_beds, unit_for_lines)
lines = affordability_lines(sel_regions_pretty, amis, aff_col)

# ----- Plot -----
if labels and tdc_vals:
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(labels, tdc_vals, color="skyblue", edgecolor="black")
    ymax = max(tdc_vals + (list(lines.values()) or [0])) * 1.12
    ax1.set_ylim(0, ymax)

    for b in bars:
        y = b.get_height()
        ax1.text(b.get_x()+b.get_width()/2, y + (ymax*0.02), f"${y:,.0f}", ha="center", va="bottom", fontsize=9)

    for i,(lab,val) in enumerate(lines.items()):
        ax1.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)

    ax1.set_ylabel("Development Cost ($)")
    ax1.set_xlabel("Unit & Type")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.xticks(rotation=20)
    plt.title("Development Cost vs. Affordable Purchase Price Thresholds")
    if lines:
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx(); ax2.set_ylim(ax1.get_ylim())
        vals = list(lines.values())
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split()[0]}\n${lines[k]:,.0f}" for k in lines])
        ax2.set_ylabel("Max. Affordable Purchase Price by % AMI" if "buy" in aff_col
                       else "Max. Affordable Gross Rent by % AMI (incl. utilities) — TODO Apartment")
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("No valid unit data provided.")

st.caption("Apartment path currently uses a rent-column placeholder. We’ll wire rent thresholds when you’re ready.")
