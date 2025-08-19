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
        if "ami" in d:
            d["ami"] = pd.to_numeric(d["ami"], errors="coerce")  # fraction (e.g., 1.50 = 150%)
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers =====
PRETTY_OVERRIDES = {
    "townhome":"Townhome","condo":"Condo","apartment":"Apartment",
    "base_me_nh_code":"ME/NH Base Code","vt_energy_code":"VT Energy Code",
    "evt_high_eff":"EVT High-Efficient","passive_house":"Passive House",
    "natural_gas":"Natural Gas Heating","all_electric":"All Electric","geothermal":"Geothermal",
    "yes":"Yes","no":"No","above_average":"Above Average","average":"Average","below_average":"Below Average",
    "studio":"Studio",
}
def pretty(s: str) -> str:
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

def options(cat, parent=None): return rows(cat, parent=parent)["option"].tolist()
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
    return f"buy{max(1,b)}" if unit_type in ("townhome","condo") else f"rent{b}"

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
    try: return f"${x:,.0f}"
    except Exception: return "—"

def household_ami_percent(region_key: str, hh_size: int, income_val: float):
    df = R[region_key]; inc_col = f"income{hh_size}"
    if not {"ami", inc_col}.issubset(df.columns): return None
    sub = df[[inc_col, "ami"]].dropna().sort_values(inc_col)
    if sub.empty: return None
    x = sub[inc_col].astype(float).to_numpy()
    y = (sub["ami"]*100.0).astype(float).to_numpy()
    if len(x) == 1: return float(y[0])
    xv = float(np.clip(income_val, x[0], x[-1]))
    return float(np.interp(xv, x, y))

def affordability_at_percent(region_key: str, bed_n: int, ami_percent: float):
    df = R[region_key]; buy_col = f"buy{bed_n}"
    if not {"ami", buy_col}.issubset(df.columns): return None
    sub = df[["ami", buy_col]].dropna().sort_values("ami")
    if sub.empty: return None
    x = (sub["ami"]*100.0).astype(float).to_numpy()
    y = sub[buy_col].astype(float).to_numpy()
    if len(x) == 1: return float(y[0])
    xv = float(np.clip(ami_percent, x[0], x[-1]))
    return float(np.interp(xv, x, y))

# ===== Header =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")
st.write("")

# ===== Product & Bedrooms =====
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

# ===== Per-Unit Policy Blocks =====
num_units = st.selectbox("How many units would you like to compare?", [1,2,3,4,5], index=1, disabled=(product=="apartment"))
units=[]
disabled_block = (product=="apartment")
for i in range(num_units):
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
        code = select_pretty("Energy code standard", options("energy_code","default") or ["vt_energy_code"],
                             key=f"code_{i}", default_raw="vt_energy_code", disabled=disabled_block)
        src  = select_pretty("Energy source", options("energy_source","default") or ["natural_gas"],
                             key=f"src_{i}", default_raw="natural_gas", disabled=disabled_block)
        infra= select_pretty("Infrastructure required?", options("infrastructure","default") or ["no","yes"],
                             key=f"infra_{i}", default_raw="no", disabled=disabled_block)
        fin  = select_pretty("Finish quality", options("finish_quality","default") or ["average","above_average","below_average"],
                             key=f"fin_{i}", default_raw="average", disabled=disabled_block)
        if disabled_block: st.caption("Policy selection disabled for Apartment placeholder.")
        units.append(dict(code=code, src=src, infra=infra, fin=fin))

# ===== Income Thresholds =====
st.subheader("Income Thresholds")
with st.container(border=True):
    region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
    sel_regions_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])
    valid_amis = [30] + list(range(50,155,5))
    n_amis = st.selectbox("How many Area Median Income (AMI) levels?", [1,2,3], index=0)
    amis=[]
    default_cycle=[150,120,100]
    for i in range(n_amis):
        default_val = default_cycle[i] if i < len(default_cycle) and default_cycle[i] in valid_amis else 150
        amis.append(st.selectbox(f"AMI value #{i+1}", valid_amis, index=valid_amis.index(default_val), key=f"ami_{i}"))

# ===== Chart 1: Simple Bars + Selected AMI Lines =====
labels, tdc_vals, lines = [], [], {}
if product in ("townhome","condo") and units:
    st.subheader("How did your choices affect affordability?")
    for i,u in enumerate(units, start=1):
        label = f"Baseline {pretty(product)}" if is_baseline(u["code"], u["src"], u["infra"], u["fin"]) else f"{pretty(product)} {i}"
        labels.append(label)
        tdc_vals.append(compute_tdc(sf_global, product, u["code"], u["src"], u["infra"], u["fin"]))
    b_int = int(bedrooms_global)
    aff_col = pick_afford_col(b_int, product)
    lines = affordability_lines(sel_regions_pretty, amis, aff_col)

if labels and tdc_vals:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(labels, tdc_vals, color="skyblue", edgecolor="black")
    ymax = max(tdc_vals + (list(lines.values()) or [0])) * 1.12
    ax1.set_ylim(0, ymax)
    for b in bars:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, y + (ymax*0.02), f"${y:,.0f}", ha="center", va="bottom", fontsize=10)
    for i,(lab,val) in enumerate(lines.items()):
        ax1.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)
    ax1.set_ylabel("Development Cost ($)")
    ax1.set_xlabel("TDC of Your Policy Choices")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=0)
    plt.title("Total Development Cost vs. What Buyers Can Afford")
    if lines:
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx(); ax2.set_ylim(ax1.get_ylim())
        vals = list(lines.values())
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split()[0]}\n${lines[k]:,.0f}" for k in lines])
        ax2.set_ylabel("Max. Affordable Purchase Price by % AMI")
    fig.subplots_adjust(bottom=0.28); fig.tight_layout()
    st.pyplot(fig)
elif product == "apartment":
    st.info("Select Townhome or Condo to run the for‑sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

# ===== Who Can Afford This Home? (Controls + Sentence) =====
st.subheader("Who Can Afford This Home?")
with st.container(border=True):
    st.markdown("""<div style="font-weight:600; margin:0 0 0.25rem 0;">Select The Region:</div>""", unsafe_allow_html=True)
    region_single = st.selectbox(label="", label_visibility="collapsed",
                                 options=[REGION_PRETTY[k] for k in REGIONS],
                                 index=[REGION_PRETTY[k] for k in REGIONS].index("Chittenden"))
    st.markdown("""<div style="font-weight:600; margin:0.5rem 0 0.25rem 0;">Select Household Size:</div>""", unsafe_allow_html=True)
    household_size = st.selectbox(label="", label_visibility="collapsed", options=list(range(1,9)), index=3)
    st.markdown(
        """
        <div style="font-weight:600; margin:0.5rem 0 0.25rem 0;">
          Input Household Income (<span style="white-space:nowrap; font-variant-numeric: tabular-nums lining-nums; font-feature-settings:'tnum' 1, 'lnum' 1;">$20,000 - $300,000</span>):
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_income = st.number_input(label="", label_visibility="collapsed",
                                  min_value=20000, max_value=300000, step=1000, value=100000, format="%d")

    reg_key = PRETTY2REG[region_single]
    def affordability_sentence():
        if product not in ("townhome","condo") or bedrooms_global is None:
            return "Affordability details are available for for-sale products (Townhome or Condo) only."
        df = R[reg_key]
        inc_col = f"income{household_size}"
        bed_n = int(bedrooms_global)
        buy_col = f"buy{bed_n}"
        if not {"ami", inc_col, buy_col}.issubset(df.columns):
            return "Required data not found for this region/household size."
        sub = (pd.DataFrame({
                "income": pd.to_numeric(df[inc_col], errors="coerce"),
                "ami_frac": pd.to_numeric(df["ami"], errors="coerce"),
                "buy": pd.to_numeric(df[buy_col], errors="coerce"),
            }).dropna().sort_values("income").reset_index(drop=True))
        if sub.empty:
            return "Insufficient data to compute affordability."
        floor_idx = sub[sub["income"] <= user_income].index.max()
        ceil_idx  = sub[sub["income"] >= user_income].index.min()
        if pd.isna(floor_idx) and pd.isna(ceil_idx):
            return "Insufficient data to compute affordability."
        if pd.isna(floor_idx):
            use_idx = 0; status = "closest"
        elif pd.isna(ceil_idx):
            use_idx = len(sub)-1; status = "closest"
        else:
            exact = sub.index[sub["income"].eq(user_income)]
            use_idx = int(exact[0]) if len(exact) else int(floor_idx); status = "exact" if len(exact) else "floor"
        sel_ami_pct = float(sub.loc[use_idx,"ami_frac"])*100.0
        sel_buy = float(sub.loc[use_idx,"buy"])
        ami_phrase = f"closest to {sel_ami_pct:.0f}% of AMI" if status != "exact" else f"{sel_ami_pct:.0f}% of AMI"
        return (f"A {household_size} person household making {fmt_money(user_income)} "
                f"({ami_phrase}) can afford a {fmt_money(sel_buy)} "
                f"{bed_n} bedroom {pretty(product)}.")
    st.markdown(f"""<div style="color:#87CEEB; font-weight:500; margin-top:0.5rem;">{affordability_sentence()}</div>""",
                unsafe_allow_html=True)
    st.write("")

# ===== Chart 2: Your Affordability vs TDC (legend, axis, line label updated) =====
st.write("")
if labels and tdc_vals and product in ("townhome","condo"):
    bed_n = int(bedrooms_global)
    your_ami_pct = household_ami_percent(reg_key, household_size, user_income)
    your_afford_price = affordability_at_percent(reg_key, bed_n, your_ami_pct) if your_ami_pct is not None else None

    fig2, ax = plt.subplots(figsize=(12, 6))
    bars2 = ax.bar(labels, tdc_vals, color="skyblue", edgecolor="black")

    ymax2 = max(tdc_vals + ([your_afford_price] if your_afford_price else [0])) * 1.25
    ax.set_ylim(0, ymax2)

    for b in bars2:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y + (ymax2*0.025), f"${y:,.0f}", ha="center", va="bottom", fontsize=10)

    if your_afford_price is not None and your_ami_pct is not None:
        ax.axhline(y=your_afford_price, linestyle="-", linewidth=2.8, color="#2E7D32", label="Your affordability")

        # inline label next to the green line showing the user's income
        # place near the rightmost bar with a small offset so it doesn't overlap
        x_pos = len(labels) - 0.15
        ax.text(x_pos, your_afford_price, f"Income: {fmt_money(user_income)}",
                va="center", ha="left", fontsize=10, color="#2E7D32", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax.set_ylabel("Development Cost ($)")
    ax.set_xlabel("TDC of Your Policy Choices")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=0)
    plt.title("Your Affordability vs. Policy-Impacted TDC")

    # right axis: no ticks/labels, but keep an explanatory axis title
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks([])
    ax_r.set_ylabel("Income Required to Purchase")

    # legend: only the simple label
    ax.legend(loc="upper right")

    fig2.subplots_adjust(top=0.90, bottom=0.20)
    fig2.tight_layout()
    st.pyplot(fig2)

    if your_afford_price is not None:
        affordable_idxs = [i for i, v in enumerate(tdc_vals) if v <= your_afford_price]
        if affordable_idxs:
            best_i = min(affordable_idxs, key=lambda i: tdc_vals[i])
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem; border-radius:8px; background:#E6F4EA; color:#1E7D34; border:1px solid #C8E6C9;">
                ✅ <b>Success:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>), you can afford <b>{len(affordable_idxs)} of {len(tdc_vals)}</b> option(s). Lowest‑cost affordable: <b>{labels[best_i]}</b>.
                </div>""",
                unsafe_allow_html=True
            )
        else:
            gap = min(tdc_vals) - your_afford_price
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem; border-radius:8px; background:#FDECEA; color:#B71C1C; border:1px solid #F5C6CB;">
                ❌ <b>Not yet:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>), none of the options are affordable. Shortfall vs. lowest‑cost option: <b>{fmt_money(gap)}</b>.
                </div>""",
                unsafe_allow_html=True
            )

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")
