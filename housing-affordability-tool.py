# ===== Imports & Paths =====
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dataclasses import dataclass

# ===== Constants & Files =====
@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    assumptions: Path

ROOT = Path(__file__).parent
PATHS = Paths(root=ROOT, data=ROOT / "data", assumptions=ROOT / "data" / "assumptions.csv")

DATA = PATHS.data
ASSUMP = PATHS.assumptions

REGIONS = {
    "Chittenden": DATA / "chittenden_ami.csv",
    "Addison":    DATA / "addison_ami.csv",
    "Vermont":    DATA / "vermont_ami.csv",
}
REGION_PRETTY = {"Chittenden": "Chittenden", "Addison": "Addison", "Vermont": "Rest of Vermont"}
PRETTY2REG = {v: k for k, v in REGION_PRETTY.items()}

VALID_AMIS = [30] + list(range(50, 155, 5))
AMI_COL = "ami"
DEFAULT_PARENT = "default"

PKG = {
    "baseline": {"label": "Baseline", "code": "vt_energy_code", "src": "natural_gas", "infra": "no", "fin": "average"},
    "top": {"label": "Top-of-the-Line", "code": "passive_house", "src": "geothermal", "infra": "yes", "fin": "above_average"},
    "below": {"label": "Below Baseline", "code": "base_me_nh_code", "src": "natural_gas", "infra": "no", "fin": "below_average"},
}

# ===== Data Loading =====
@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_assumptions(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.lower()
    for c in ("category","parent_option","option","value_type","unit"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_regions(files: dict) -> dict:
    out = {}
    for name, p in files.items():
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        d[AMI_COL] = pd.to_numeric(d[AMI_COL], errors="coerce")
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers =====
def fmt_money(x):
    val = pd.to_numeric(x, errors="coerce")
    return "—" if pd.isna(val) else f"${val:,.0f}"

def _rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    if opt    is not None: q &= A["option"].eq(str(opt).lower())
    return A.loc[q]

def one_val(cat, opt, parent=None, default=0.0):
    r = _rows(cat, opt, parent)
    return float(r.iloc[0]["value"]) if not r.empty else default

def bedroom_sf(h_type, br_label):
    r = _rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def compute_tdc(sf, htype, code, src, infra, fin):
    base = baseline_per_sf()
    pct = (one_val("energy_code", code) + one_val("finish_quality", fin)) / 100.0
    per_sf = base * (mf_factor(htype) + pct)
    per_sf += one_val("energy_source", src, DEFAULT_PARENT)
    per_sf += one_val("infrastructure", infra, DEFAULT_PARENT)
    return sf * per_sf

def pick_afford_col(b_int, unit_type):
    b = int(np.clip(b_int, 0, 5))
    return f"buy{max(1,b)}" if unit_type in ("townhome","condo") else f"rent{b}"

def affordability_lines(region_pretty_list, amis, col):
    lines = {}
    for rp in region_pretty_list:
        df = R[PRETTY2REG[rp]]
        if col not in df.columns: continue
        for ami in sorted(amis):
            ami_capped = min(ami, 150)
            m = df[AMI_COL].eq(ami_capped/100.0)
            if m.any():
                lines[f"{ami_capped}% AMI - {rp}"] = float(df.loc[m, col].iloc[0])
    return lines

def make_price_to_income_mapper(region_key: str, hh_size: int, bed_n: int):
    df = R[region_key]
    inc_col = f"income{hh_size}"
    buy_col = f"buy{bed_n}"
    sub = df[[buy_col, inc_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(buy_col)
    if sub.empty: return None
    x = sub[buy_col].to_numpy(dtype=float)
    y = sub[inc_col].to_numpy(dtype=float)
    def price_to_income(price: float) -> float:
        if price is None or not np.isfinite(price): return np.nan
        return float(np.interp(price, x, y, left=y[0], right=y[-1]))
    return price_to_income

# ===== Plot Utilities =====
def _bar_with_values(ax, labels, values, pad_ratio):
    bars = ax.bar(labels, values, color="skyblue", edgecolor="black")
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y*(1+pad_ratio), fmt_money(y),
                ha="center", va="bottom", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))

def _apply_ylim(ax, ax2, ymax):
    ax.set_ylim(0, ymax)
    if ax2: ax2.set_ylim(0, ymax)

# ===== Chart 1 =====
def draw_chart1(labels, tdc_vals, lines):
    fig, ax = plt.subplots(figsize=(12,6))
    ymax = 1.2 * max(max(tdc_vals, default=1.0), max(lines.values(), default=0.0))
    _bar_with_values(ax, labels, tdc_vals, 0.02)
    for i, (lab, val) in enumerate(sorted(lines.items())):
        ax.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)
    _apply_ylim(ax, None, ymax)
    ax2 = ax.twinx(); _apply_ylim(ax, ax2, ymax)
    if lines:
        vals = [lines[k] for k in sorted(lines)]
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split('%')[0].strip()}%\n{fmt_money(lines[k])}" for k in sorted(lines)])
        ax.legend(loc="upper right")
    else:
        ax2.set_yticks([])
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_title("Total Development Cost vs. What Households Can Afford")
    st.pyplot(fig)

# ===== Chart 2 =====
def draw_chart2(labels, tdc_vals, afford_price, price_to_income):
    fig, ax = plt.subplots(figsize=(12,6))
    ymax = 1.2 * max(max(tdc_vals, default=1.0), float(afford_price or 0.0))
    _bar_with_values(ax, labels, tdc_vals, 0.025)
    if afford_price:
        ax.axhline(y=float(afford_price), linestyle="-", linewidth=2.8, color="#2E7D32",
                   label="Income level mapped to affordable purchase price")
    _apply_ylim(ax, None, ymax)
    rax = ax.twinx(); _apply_ylim(ax, rax, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))
    if price_to_income:
        left_ticks = ax.get_yticks()
        left_min, left_max = ax.get_ylim()
        # Exclude bottom (0) and top (ymax) ticks
        right_ticks = [t for t in left_ticks if (t > left_min) and (t < left_max)]
        rax.set_yticks(right_ticks)
        rax.set_yticklabels([fmt_money(price_to_income(t)) for t in right_ticks])
    else:
        rax.set_yticks([])
    rax.set_ylabel("Income Req. to Purchase")
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_title("Purchase Ability by Income, Household Size, and Region")
    if afford_price: ax.legend(loc="upper right")
    st.pyplot(fig)

# ===== Streamlit App =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")

# Step 1 — Housing Type
st.header("Step 1 — Housing Type")
product = st.radio("What kind of housing?", ["townhome","condo","apartment"], horizontal=True)
apartment_mode = (product=="apartment")
if not apartment_mode:
    br_opts = _rows("bedrooms", parent=product)["option"].tolist() or ["2"]
    bedrooms = st.selectbox("Bedrooms", br_opts, index=(br_opts.index("2") if "2" in br_opts else 0))
    sf = bedroom_sf(product, bedrooms) or 1000.0
else:
    bedrooms, sf = None, None
    st.info("Apartment model (rent) coming soon.")

st.divider()

# Step 2 — Policies
st.header("Step 2 — Pick your Policies")
num_units = st.selectbox("How many units?", [1,2,3,4,5], index=1, disabled=apartment_mode)
units = []
for i in range(num_units):
    st.subheader(f"{product.title()} {i+1}")
    label = st.selectbox("Policy package", list(PKG.keys()), format_func=lambda k: PKG[k]["label"], key=f"pkg_{i}")
    units.append({
        "label": PKG[label]["label"],
        "code": PKG[label]["code"],
        "src": PKG[label]["src"],
        "infra": PKG[label]["infra"],
        "fin": PKG[label]["fin"],
    })
st.divider()

# Step 3 — Compare Costs
st.header("Step 3 — Compare Costs with Affordability")
if not apartment_mode and units:
    labels = [u["label"] for u in units]
    tdc_vals = [compute_tdc(sf, product, u["code"], u["src"], u["infra"], u["fin"]) for u in units]
    amis = [st.selectbox("AMI", VALID_AMIS, index=VALID_AMIS.index(100))]
    sel_regions_pretty = st.multiselect("Select region(s)", list(REGION_PRETTY.values()), default=[REGION_PRETTY["Chittenden"]])
    lines = affordability_lines(sel_regions_pretty, amis, pick_afford_col(int(bedrooms), product))
    st.subheader("Do These Policy Choices Put Homes Within Reach?")
    draw_chart1(labels, tdc_vals, lines)
st.divider()

# Step 4 — Household Context
st.header("Step 4 — Specify Household Context")
region_single = st.selectbox("Region", list(REGION_PRETTY.values()), index=0)
household_size = st.selectbox("Household size", list(range(1,9)), index=3)
user_income = st.number_input("Household income", min_value=20000, max_value=300000, step=1000, value=100000)

if not apartment_mode and units:
    st.subheader("What These Costs Mean for Your Constituents")
    reg_key = PRETTY2REG[region_single]
    def household_ami_percent(region_key, hh_size, income_val):
        df = R[region_key]; inc_col = f"income{hh_size}"
        sub = df[[inc_col, AMI_COL]].dropna().sort_values(inc_col)
        if sub.empty: return None
        return float(np.interp(income_val, sub[inc_col], sub[AMI_COL]*100.0, left=sub[AMI_COL].iloc[0]*100, right=sub[AMI_COL].iloc[-1]*100))
    def affordability_at_percent(region_key, bed_n, ami_percent):
        df = R[region_key]; col = f"buy{bed_n}"
        sub = df[[AMI_COL, col]].dropna().sort_values(AMI_COL)
        if sub.empty: return None
        return float(np.interp(ami_percent, sub[AMI_COL]*100.0, sub[col], left=sub[col].iloc[0], right=sub[col].iloc[-1]))
    ami_pct = household_ami_percent(reg_key, int(household_size), int(user_income))
    afford_price = affordability_at_percent(reg_key, int(bedrooms), ami_pct) if ami_pct else None
    price_to_income = make_price_to_income_mapper(reg_key, int(household_size), int(bedrooms))
    draw_chart2(labels, tdc_vals, afford_price, price_to_income)

    # Affordability logic: success if ANY unit affordable
    required_incomes = [price_to_income(val) for val in tdc_vals if price_to_income]
    affordable = any(user_income >= inc for inc in required_incomes if inc and np.isfinite(inc))
    min_required = min(required_incomes) if required_incomes else None

    if affordable:
        st.success(f"✅ At your income ({fmt_money(user_income)}) and household size ({household_size}), at least one option is affordable.")
    else:
        need_text = fmt_money(min_required) if min_required else "—"
        st.error(f"❌ Keep trying: At your income ({fmt_money(user_income)}) and household size ({household_size}), none of the options are affordable. A household of this size would need at least {need_text} to afford the cheapest option.")

st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")
