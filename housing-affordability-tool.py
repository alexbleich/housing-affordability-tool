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
VALID_AMIS = [30] + list(range(50, 155, 5))
DEFAULT_AMIS = [150, 120, 100]

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
            d["ami"] = pd.to_numeric(d["ami"], errors="coerce")  # fraction
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers: formatting, lookup, UI =====
PRETTY_OVERRIDES = {
    "townhome":"Townhome","condo":"Condo","apartment":"Apartment",
    "base_me_nh_code":"ME/NH Base Code","vt_energy_code":"VT Energy Code",
    "evt_high_eff":"EVT High-Efficient","passive_house":"Passive House",
    "natural_gas":"Natural Gas Heating","all_electric":"All Electric","geothermal":"Geothermal",
    "yes":"Yes","no":"No","above_average":"Above Average","average":"Average","below_average":"Below Average",
    "studio":"Studio",
}
def pretty(x: str) -> str:
    s = str(x).lower().strip()
    if s in PRETTY_OVERRIDES: return PRETTY_OVERRIDES[s]
    t = s.replace("_"," ").title()
    return (t.replace(" Ami"," AMI").replace(" Vt "," VT ").replace(" Nh "," NH ")
             .replace(" Me "," ME ").replace(" Evt "," EVT ").replace(" Mf "," MF "))

def fmt_money(x):
    try: return f"${x:,.0f}"
    except Exception: return "—"

def rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    if opt    is not None: q &= A["option"].eq(str(opt).lower())
    return A[q]

def one_val(cat, opt, parent=None, expect_type=None):
    r = rows(cat, opt, parent)
    if r.empty and parent is not None: r = rows(cat, opt)
    if r.empty: return 0.0
    if expect_type and r.iloc[0]["value_type"] != expect_type: return 0.0
    return float(r.iloc[0]["value"])

def options(cat, parent=None): return rows(cat, parent=parent)["option"].tolist()

def bedroom_sf(h_type, br_label):
    r = rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def compute_tdc(sf, htype, code, src, infra, fin):
    base = baseline_per_sf()
    per_sf = base*mf_factor(htype)
    per_sf += base*(one_val("energy_code", code)/100.0)
    per_sf += base*(one_val("finish_quality", fin)/100.0)
    per_sf += one_val("energy_source", src, "default", "per_sf")
    per_sf += one_val("infrastructure", infra, "default", "per_sf")
    return sf * per_sf

def is_baseline(code, src, infra, fin):
    return (code == "vt_energy_code" and src == "natural_gas" and infra == "no" and fin == "average")

def pick_afford_col(b_int, unit_type):
    b = int(np.clip(b_int, 0, 5))
    return f"buy{max(1,b)}" if unit_type in ("townhome","condo") else f"rent{b}"

def affordability_lines(region_pretty_list, amis, col):
    lines = {}
    for rp in region_pretty_list:
        df = R[PRETTY2REG[rp]]
        for ami in amis:
            m = df["ami"].eq(ami/100.0)
            if m.any() and col in df.columns:
                lines[f"{ami}% AMI - {rp}"] = float(df.loc[m, col].iloc[0])
    return lines

def nearest_ami_row(df: pd.DataFrame, hh_col: str, income: float) -> pd.Series:
    sub = pd.DataFrame({
        "income": pd.to_numeric(df[hh_col], errors="coerce"),
        "ami_frac": pd.to_numeric(df["ami"], errors="coerce"),
    }).dropna().sort_values("income").reset_index(drop=True)
    if sub.empty: return pd.Series(dtype=float)
    diffs = (sub["income"] - income).abs()
    nearest = sub.loc[diffs.eq(diffs.min())]
    return nearest.iloc[-1]  # tie -> higher AMI

def labeled_select(html_label: str, options_list, **kwargs):
    st.markdown(f'<div style="font-weight:600;margin:0.5rem 0 0.25rem 0;">{html_label}</div>', unsafe_allow_html=True)
    return st.selectbox(label="", options=options_list, label_visibility="collapsed", **kwargs)

def labeled_number_input(html_label: str, **kwargs):
    st.markdown(f'<div style="font-weight:600;margin:0.5rem 0 0.25rem 0;">{html_label}</div>', unsafe_allow_html=True)
    return st.number_input(label="", label_visibility="collapsed", **kwargs)

# ===== Plotting =====
def draw_chart1(labels, tdc_vals, lines):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, tdc_vals, color="skyblue", edgecolor="black")
    ymax = max(tdc_vals + (list(lines.values()) or [0])) * 1.2
    ax.set_ylim(0, ymax)
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y + ymax*0.02, fmt_money(y), ha="center", va="bottom", fontsize=10)
    for i,(lab,val) in enumerate(lines.items()):
        ax.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)
    ax.set_ylabel("Development Cost ($)")
    ax.set_xlabel("TDC of Your Policy Choices")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))
    plt.xticks(rotation=0)
    plt.title("Total Development Cost vs. What Buyers Can Afford")
    if lines:
        ax.legend(loc="upper right")
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
        vals = list(lines.values())
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split()[0]}\n{fmt_money(lines[k])}" for k in lines])
        ax2.set_ylabel("Max. Affordable Purchase Price by % AMI")
    fig.subplots_adjust(bottom=0.28); fig.tight_layout()
    st.pyplot(fig)

def draw_chart2(labels, tdc_vals, afford_price, user_income):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, tdc_vals, color="skyblue", edgecolor="black")
    ymax = max(tdc_vals + ([afford_price] if afford_price else [0])) * 1.15
    ax.set_ylim(0, ymax)
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y + ymax*0.025, fmt_money(y), ha="center", va="bottom", fontsize=10)
    if afford_price is not None:
        ax.axhline(y=afford_price, linestyle="-", linewidth=2.8, color="#2E7D32",
                   label="Affordable price at your income")
        ax.annotate(
            f"Your income:\n{fmt_money(user_income)}",
            xy=(1.0, afford_price), xycoords=("axes fraction","data"),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=10, color="#2E7D32", multialignment="center",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none", pad=0)
        )
    ax.set_ylabel("Development Cost ($)")
    ax.set_xlabel("TDC of Your Policy Choices")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))
    plt.xticks(rotation=0)
    plt.title("Your Affordability vs. Policy-Impacted TDC")
    rax = ax.twinx(); rax.set_ylim(ax.get_ylim()); rax.set_yticks([])
    rax.set_ylabel("Income Required to Purchase", labelpad=70)
    fig.subplots_adjust(top=0.90, bottom=0.20, right=0.84)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)

# ===== Header =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")
st.write("")

# ===== Product & Bedrooms =====
product = st.selectbox(
    "What type of housing would you like to analyze?",
    ["townhome","condo","apartment"], index=0, key="global_product"
)
if product in ("townhome","condo"):
    br_opts = options("bedrooms", parent=product) or ["2"]
    bedrooms = st.selectbox("Number of bedrooms", br_opts, index=br_opts.index("2") if "2" in br_opts else 0, key="global_bedrooms")
    sf = bedroom_sf(product, bedrooms) or 1000.0
else:
    bedrooms, sf = None, None
    st.info("Apartment modeling (rent-based) coming soon. For now, choose Townhome or Condo to compare for-sale products.")

# ===== Per-Unit Policy Blocks =====
num_units = st.selectbox("How many units would you like to compare?", [1,2,3,4,5], index=1, disabled=(product=="apartment"))
units = []
for i in range(num_units):
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
        code  = st.selectbox("Energy code standard", options("energy_code","default") or ["vt_energy_code"], index=0, key=f"code_{i}")
        src   = st.selectbox("Energy source",      options("energy_source","default") or ["natural_gas"], index=0, key=f"src_{i}")
        infra = st.selectbox("Infrastructure required?", options("infrastructure","default") or ["no","yes"], index=0, key=f"infra_{i}")
        fin   = st.selectbox("Finish quality", options("finish_quality","default") or ["average","above_average","below_average"],
                             index=(options("finish_quality","default") or ["average"]).index("average") if "average" in (options("finish_quality","default") or []) else 0,
                             key=f"fin_{i}")
        units.append(dict(code=code, src=src, infra=infra, fin=fin))

# ===== Income Thresholds =====
st.subheader("Income Thresholds")
with st.container(border=True):
    region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
    sel_regions_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])
    n_amis = st.selectbox("How many Area Median Income (AMI) levels?", [1,2,3], index=0)
    amis = [st.selectbox(f"AMI value #{i+1}", VALID_AMIS, index=VALID_AMIS.index(DEFAULT_AMIS[i] if i < len(DEFAULT_AMIS) and DEFAULT_AMIS[i] in VALID_AMIS else 150), key=f"ami_{i}")
            for i in range(n_amis)]

# ===== Chart 1 =====
labels, tdc_vals, lines = [], [], {}
if product in ("townhome","condo") and units:
    st.subheader("How did your choices affect affordability?")
    for i,u in enumerate(units, 1):
        labels.append(f"Baseline {pretty(product)}" if is_baseline(u["code"],u["src"],u["infra"],u["fin"]) else f"{pretty(product)} {i}")
        tdc_vals.append(compute_tdc(sf, product, u["code"], u["src"], u["infra"], u["fin"]))
    b_int = int(bedrooms)
    lines = affordability_lines(sel_regions_pretty, amis, pick_afford_col(b_int, product))

if labels and tdc_vals:
    draw_chart1(labels, tdc_vals, lines)
elif product == "apartment":
    st.info("Select Townhome or Condo to run the for‑sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")

# ===== Who Can Afford This Home? =====
st.subheader("Who Can Afford This Home?")
with st.container(border=True):
    region_pretty_list = [REGION_PRETTY[k] for k in REGIONS]
    region_single = labeled_select("Select The Region:", region_pretty_list, index=region_pretty_list.index("Chittenden"))
    household_size = labeled_select("Select Household Size:", list(range(1,9)), index=3)
    user_income = labeled_number_input(
        "Input Household Income (<span style='white-space:nowrap;font-variant-numeric:tabular-nums lining-nums;font-feature-settings:\"tnum\" 1, \"lnum\" 1;'>$20,000 - $300,000</span>):",
        min_value=20000, max_value=300000, step=1000, value=100000, format="%d"
    )

    reg_key = PRETTY2REG[region_single]

    def affordability_sentence():
        if product not in ("townhome","condo") or bedrooms is None: return "Affordability details are available for for-sale products (Townhome or Condo) only."
        df = R[reg_key]; inc_col = f"income{household_size}"; buy_col = f"buy{int(bedrooms)}"
        if not {"ami", inc_col, buy_col}.issubset(df.columns): return "Required data not found for this region/household size."
        nearest = nearest_ami_row(df, inc_col, user_income)
        if nearest.empty: return "Insufficient data to compute affordability."
        sel_ami_pct = float(nearest["ami_frac"]) * 100.0
        sel_buy = float(pd.to_numeric(df[buy_col], errors="coerce")[nearest.name])
        product_sentence = pretty(product).lower()
        return (f"A {household_size} person household making {fmt_money(user_income)} "
                f"(closest to {sel_ami_pct:.0f}% of AMI) can afford a {fmt_money(sel_buy)} "
                f"{int(bedrooms)} bedroom {product_sentence}.")

    st.markdown(f"""<div style="color:#87CEEB;font-weight:500;margin-top:0.5rem;">{affordability_sentence()}</div>""",
                unsafe_allow_html=True)
    st.write("")

# ===== Chart 2 + Outcome =====
st.write("")
if labels and tdc_vals and product in ("townhome","condo"):
    # dynamic affordability line from "Who Can Afford..." inputs
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
        df = R[region_key]; col = f"buy{bed_n}"
        if not {"ami", col}.issubset(df.columns): return None
        sub = df[["ami", col]].dropna().sort_values("ami")
        if sub.empty: return None
        x = (sub["ami"]*100.0).astype(float).to_numpy()
        y = sub[col].astype(float).to_numpy()
        if len(x) == 1: return float(y[0])
        xv = float(np.clip(ami_percent, x[0], x[-1]))
        return float(np.interp(xv, x, y))

    ami_pct = household_ami_percent(reg_key, int(household_size), int(user_income))
    afford_price = affordability_at_percent(reg_key, int(bedrooms), ami_pct) if ami_pct is not None else None

    draw_chart2(labels, tdc_vals, afford_price, int(user_income))

    if afford_price is not None:
        affordable_idxs = [i for i, v in enumerate(tdc_vals) if v <= afford_price]
        if affordable_idxs:
            best_i = min(affordable_idxs, key=lambda i: tdc_vals[i])
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem;border-radius:8px;background:#E6F4EA;color:#1E7D34;border:1px solid #C8E6C9;">
                ✅ <b>Success:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>),
                you can afford <b>{len(affordable_idxs)} of {len(tdc_vals)}</b> option(s). Lowest‑cost affordable: <b>{labels[best_i]}</b>.
                </div>""", unsafe_allow_html=True)
        else:
            gap = min(tdc_vals) - afford_price
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem;border-radius:8px;background:#FDECEA;color:#B71C1C;border:1px solid #F5C6CB;">
                ❌ <b>Not yet:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>),
                none of the options are affordable. Shortfall vs. lowest‑cost option: <b>{fmt_money(gap)}</b>.
                </div>""", unsafe_allow_html=True)
