# ===== Imports & Paths (pre-helpers) =====
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ----- Robust project paths -----
@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    assumptions: Path

ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
PATHS = Paths(
    root=ROOT,
    data=ROOT / "data",
    assumptions=ROOT / "data" / "assumptions.csv",
)

DATA = PATHS.data
ASSUMP = PATHS.assumptions

# ----- Region CSVs -----
REGIONS = {
    "Chittenden": DATA / "chittenden_ami.csv",
    "Addison":    DATA / "addison_ami.csv",
    "Vermont":    DATA / "vermont_ami.csv",  # Rest of Vermont
}
REGION_PRETTY = {
    "Chittenden": "Chittenden",
    "Addison": "Addison",
    "Vermont": "Rest of Vermont",
}
PRETTY2REG = {v: k for k, v in REGION_PRETTY.items()}

# ----- Constants used downstream -----
AMI_COL = "ami"
DEFAULT_PARENT = "default"
AFFORD_EPS = 0.5

ENERGY_CODE_ORDER = ["vt_energy_code", "rbes", "passive_house"]
DEFAULT_COMPONENTS = dict(code="vt_energy_code", src="natural_gas", infra="no", fin="average")

# ===== Data Loading =====
@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_assumptions(p: Path) -> pd.DataFrame:
    """
    Load assumptions.csv (columns: category,parent_option,option,value_type,value).
    - All string ids normalized to lowercase & stripped.
    - value_type normalized to {per_sf, per_unit, fixed}.
    - 'value' kept numeric; blanks remain NaN so UI can hide items with no costs yet.
    """
    if not p.exists():
        st.error(f"Missing assumptions file: {p}"); st.stop()

    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.lower()

    required = {"category", "parent_option", "option", "value_type", "value"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Assumptions CSV missing columns: {missing}"); st.stop()

    for c in ("category", "parent_option", "option", "value_type"):
        df[c] = df[c].astype(str).str.strip().str.lower()

    def _norm_vtype(s: str) -> str:
        t = s.replace("_", "").replace("-", "").replace(" ", "")
        if t in {"persf", "psf", "sf"}:   return "per_sf"
        if t in {"perunit"}:              return "per_unit"
        if t in {"fixed", "flat", "lump", "fixedcost"}: return "fixed"
        return s

    df["value_type"] = df["value_type"].map(_norm_vtype)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_regions(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """
    Load region affordability tables; ensure 'ami' exists and coerce numeric columns.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, p in files.items():
        if not p.exists():
            st.error(f"Missing region file for {name}: {p}"); st.stop()
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        if AMI_COL not in d.columns:
            st.error(f"Region file {name} missing '{AMI_COL}' column."); st.stop()
        d[AMI_COL] = pd.to_numeric(d[AMI_COL], errors="coerce")
        for col in d.columns:
            if re.match(r"(buy|rent|income)\d+$", col):
                d[col] = pd.to_numeric(d[col], errors="coerce")
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers (display + options) =====
PRETTY_OVERRIDES = {
    "townhome":"Townhome  →  (ownership; individual entrance; generally larger than a condo)",
    "condo":"Condo  →  (ownership; entrance from a common corridor; generally smaller than a townhome)",
    "apartment":"Apartment  →  (rental; entrance from a common corridor; generally smaller than condo/townhome)",
    "studio":"Studio",
    "vt_energy_code":"Regionally standard energy code",
    "rbes":"Vermont’s 2024 RBES code (Residential Building Energy Standard)",
    "passive_house":"Passive House Standard",
    "natural_gas":"Natural Gas",
    "all_electric":"All Electric",
    "geothermal":"Geothermal",
    "below_average":"Below Average","average":"Average","above_average":"Above Average",
    "yes":"Yes","no":"No",
}
PRODUCT_SHORT = {"townhome": "Townhome", "condo": "Condo", "apartment": "Apartment"}
TOKEN_UPPER = {" Ami":" AMI"," Vt ":" VT "," Nh ":" NH "," Me ":" ME "," Evt ":" EVT "," Mf ":" MF "}

def pretty(x: str) -> str:
    s = str(x).lower().strip()
    if s in PRETTY_OVERRIDES: return PRETTY_OVERRIDES[s]
    t = s.replace("_"," ").title()
    for k, v in TOKEN_UPPER.items(): t = t.replace(k, v)
    return t

def pretty_short(x: str) -> str:
    return PRODUCT_SHORT.get(str(x).lower().strip(), str(x).title())

def fmt_money(x):
    val = pd.to_numeric(x, errors="coerce")
    return "—" if pd.isna(val) else f"${val:,.0f}"

def _rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    if opt    is not None: q &= A["option"].eq(str(opt).lower())
    return A.loc[q]

def options(cat, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    return A.loc[q, "option"].tolist()

def one_val(cat, opt, parent=None, expect_type=None, default=0.0):
    r = _rows(cat, opt, parent)
    if r.empty and parent is not None: r = _rows(cat, opt)
    if r.empty: return default
    if expect_type and r.iloc[0]["value_type"] != expect_type: return default
    return float(pd.to_numeric(r.iloc[0]["value"], errors="coerce"))

def bedroom_sf(h_type, br_label):
    r = _rows("bedrooms", br_label, h_type)
    return float(pd.to_numeric(r.iloc[0]["value"], errors="coerce")) if not r.empty else np.nan

def bedroom_options(product_key: str) -> list[str]:
    """Use the CSV to decide which bedroom counts to show (e.g., no 1-BR townhomes)."""
    rows = _rows("bedrooms", parent=product_key)
    if rows.empty:
        # sensible fallback
        return ["2","3","4"] if product_key == "townhome" else ["1","2","3"]
    opts = sorted(rows["option"].astype(str).tolist(), key=lambda x: int(x))
    return opts

# ===== Cost model pieces =====
def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type, default=1.0)
def baseline_hard_per_sf(): return one_val("baseline_hard_cost","baseline")
def soft_cost_pct(): return one_val("soft_cost","baseline")

def infra_per_unit(htype: str, opt: str) -> float:
    return one_val("new_neighborhood", opt, parent=htype, default=0.0)

def bool_to_infra_opt(b: bool) -> str:
    return "yes" if bool(b) else "no"

def _sum_values(cat, opt, parents, vtype):
    if opt is None: 
        return 0.0
    r = A[(A["category"].eq(cat)) &
          (A["option"].eq(str(opt).lower())) &
          (A["parent_option"].isin([p.lower() for p in parents])) &
          (A["value_type"].eq(vtype))]
    # robust to NaN in new file
    return float(pd.to_numeric(r["value"], errors="coerce").fillna(0.0).sum()) if not r.empty else 0.0

def _sum_overlay(cat, selected_opt, parents):
    per_sf  = _sum_values(cat, selected_opt, parents, "per_sf")  + _sum_values(cat, "default", parents, "per_sf")
    per_unit= _sum_values(cat, selected_opt, parents, "per_unit")+ _sum_values(cat, "default", parents, "per_unit")
    fixed   = _sum_values(cat, selected_opt, parents, "fixed")   + _sum_values(cat, "default", parents, "fixed")
    return per_sf, per_unit, fixed

def compute_tdc(sf, htype, code, src, infra, fin):
    """Total Development Cost (per home), all values from assumptions.csv."""
    parents = [htype, "default"]

    # --- HARD cost per sf w/ soft-cost multiplier ---
    base_hard_psf = baseline_hard_per_sf()
    mf_mult  = mf_factor(htype)
    pct_mult = (one_val("energy_code", code) + one_val("finish_quality", fin)) / 100.0
    hard_psf_before_soft = base_hard_psf * (mf_mult + pct_mult)
    hard_psf = hard_psf_before_soft * (1.0 + soft_cost_pct()/100.0)

    # --- Policy/overlay components ---
    es_psf, es_pu, es_fx   = _sum_overlay("energy_source", src, parents)
    acq_psf, acq_pu, acq_fx = _sum_overlay("acq_cost", "baseline", parents)
    infra_pu = infra_per_unit(htype, infra) if infra in ("yes", "no") else 0.0

    EXCLUDE = {
        "baseline_hard_cost","soft_cost","mf_efficiency_factor",
        "energy_code","finish_quality","energy_source","new_neighborhood","bedrooms"
    }
    other = A[~A["category"].isin(EXCLUDE)]
    other = other[(other["option"].eq("default")) & (other["parent_option"].isin([htype, "default"]))]

    other_psf = float(other.loc[other["value_type"].eq("per_sf"),  "value"].sum()) if not other.empty else 0.0
    other_pu  = float(other.loc[other["value_type"].eq("per_unit"),"value"].sum()) if not other.empty else 0.0
    other_fx  = float(other.loc[other["value_type"].eq("fixed"),   "value"].sum()) if not other.empty else 0.0

    # --- Total (additive; order doesn’t change the sum) ---
    total  = 0.0
    total += sf * (hard_psf + es_psf + acq_psf + other_psf)   # per-sf pieces (acq_psf usually 0)
    total += es_pu + acq_pu + other_pu + infra_pu             # per-unit pieces (acq_pu carries the $18k)
    total += es_fx + acq_fx + other_fx                        # fixed adders (if any)
    return total

# ===== Price ↔ Income mapping (Chittenden engine) =====
def _best_buy_col_for_bed(df_cols, bed_n: int):
    avail = sorted([int(m.group(1)) for c in df_cols if (m:=re.match(r"buy(\d+)$", c))], key=int)
    if not avail: return None
    le = [n for n in avail if n <= bed_n]
    pick = (le[-1] if le else avail[0])
    return f"buy{pick}"

def build_price_income_transformers(region_key: str, hh_size: int, bed_n: int):
    df = R[region_key]
    inc_col = f"income{hh_size}"
    buy_col = _best_buy_col_for_bed(df.columns.str.lower(), int(bed_n))
    if buy_col is None or not {inc_col, buy_col}.issubset(df.columns.str.lower()):
        return None, None, None, None, None, None
    sub = df[[buy_col, inc_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(buy_col)
    if sub.empty: return None, None, None, None, None, None
    x = sub[buy_col].to_numpy(dtype=float)
    y = sub[inc_col].to_numpy(dtype=float)
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    if len(x) > 1:
        m_lo  = (y[1] - y[0]) / (x[1] - x[0])
        m_hi  = (y[-1] - y[-2]) / (x[-1] - x[-2])
        mi_lo = (x[1] - x[0]) / (y[1] - y[0]) if y[1] != y[0] else 0.0
        mi_hi = (x[-1] - x[-2]) / (y[-1] - y[-2]) if y[-1] != y[-2] else 0.0
    else:
        m_lo = m_hi = mi_lo = mi_hi = 0.0
    def price_to_income(p):
        p_arr = np.asarray(p, dtype=float)
        out = np.empty_like(p_arr)
        if out.size == 0: return out
        low  = p_arr <= xmin
        high = p_arr >= xmax
        mid  = ~(low | high)
        if np.any(low):  out[low]  = ymin + m_lo * (p_arr[low]  - xmin)
        if np.any(high): out[high] = ymax + m_hi * (p_arr[high] - xmax)
        if np.any(mid):  out[mid]  = np.interp(p_arr[mid], x, y)
        return out
    def income_to_price(i):
        i_arr = np.asarray(i, dtype=float)
        out = np.empty_like(i_arr)
        if out.size == 0: return out
        low  = i_arr <= ymin
        high = i_arr >= ymax
        mid  = ~(low | high)
        if np.any(low):  out[low]  = xmin + mi_lo * (i_arr[low]  - ymin)
        if np.any(high): out[high] = xmax + mi_hi * (i_arr[high] - ymax)
        if np.any(mid):  out[mid]  = np.interp(i_arr[mid], y, x)
        return out
    return price_to_income, income_to_price, ymin, ymax, xmin, xmax

def affordable_mask(user_income, required_incomes, eps=AFFORD_EPS):
    r = np.asarray(required_incomes, dtype=float)
    ui = float(user_income)
    return np.isfinite(r) & ((ui + eps) >= r)

def ami_percent_for_income(region_key: str, hh_size: int, required_income: float):
    """Return (percent_ami, capped_bool). Floors to lower bracket; caps to 30/150 with '(at least)' handling upstream."""
    df = R[region_key].copy()
    col = f"income{hh_size}"
    if col not in df.columns: return None, False
    ser_inc = pd.to_numeric(df[col], errors="coerce")
    ser_ami = pd.to_numeric(df[AMI_COL], errors="coerce")  # fractions, 0.30..1.50
    mask = ser_inc.notna() & ser_ami.notna()
    sub = pd.DataFrame({"ami": ser_ami[mask], "inc": ser_inc[mask]}).sort_values("ami")
    if sub.empty: return None, False
    ami_min, ami_max = 0.30, 1.50
    inc_min = float(sub.loc[np.isclose(sub["ami"], ami_min), "inc"].min()) if (sub["ami"]<=ami_min).any() else float(sub["inc"].min())
    inc_max = float(sub.loc[np.isclose(sub["ami"], ami_max), "inc"].max()) if (sub["ami"]>=ami_max).any() else float(sub["inc"].max())
    inc = float(required_income)
    if inc <= inc_min: return 30, True
    if inc >= inc_max: return 150, True
    sub_le = sub[sub["inc"] <= inc]
    if sub_le.empty: return 30, True
    ami_frac = float(sub_le.iloc[-1]["ami"])
    return int(round(ami_frac*100)), False

# ===== Chart Utils =====
def _bar_with_values(ax, labels, values, pad_ratio):
    bars = ax.bar(labels, values, color="#A7D3FF", edgecolor="black")  # lighter blue
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y * (1 + pad_ratio), fmt_money(y), ha="center", va="bottom", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))

def draw_chart(labels, tdc_vals, afford_price, price_to_income, income_to_price):
    fig, ax = plt.subplots(figsize=(12, 6))
    top_bar  = max(tdc_vals) if tdc_vals else 1.0
    top_line = float(afford_price) if (afford_price is not None and np.isfinite(afford_price)) else 0.0
    ymax = 1.2 * max(top_bar, top_line, 1.0)
    _bar_with_values(ax, labels, tdc_vals, pad_ratio=0.025)
    if afford_price is not None and np.isfinite(afford_price):
        ax.axhline(y=float(afford_price), linestyle="-", linewidth=2.8, color="#2E7D32",
                   label="Income level mapped to affordable purchase price")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_xlabel("")
    ax.set_title("Purchase Ability by Income and Household Size")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))
    if price_to_income is not None and income_to_price is not None:
        sec = ax.secondary_yaxis('right', functions=(price_to_income, income_to_price))
        sec.set_ylabel("Income Req. to Purchase")
        sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_money(v)))
        sec.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    else:
        sec = ax.twinx(); sec.set_yticks([]); sec.set_ylabel("Income Req. to Purchase")
    ax.legend(loc="upper right", frameon=True)
    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

def _ami_phrase(pct: int|None, region_label: str, capped_low: bool, capped_high: bool) -> str:
    if pct is None:
        return f"—% of AMI in {region_label}"
    if capped_high:
        return (f"More than 150% of AMI in the rest of Vermont."
                if region_label == "Rest of Vermont"
                else f"More than 150% of Area Median Income in {region_label}.")
    suffix = " (at least)" if capped_low else ""
    return (f"{pct}% of AMI in the rest of Vermont{suffix}."
            if region_label == "Rest of Vermont"
            else f"{pct}% of Area Median Income in {region_label}{suffix}.")

# ===== Session State (units) =====
def _ensure_units(n, product_key="townhome"):
    if "units" not in st.session_state:
        st.session_state.units = []
    while len(st.session_state.units) < n:
        idx = len(st.session_state.units)
        prod_for_label = pretty_short(st.session_state.get("global_product", product_key))
        st.session_state.units.append({
            "components": DEFAULT_COMPONENTS.copy(),
            "custom_label": f"{prod_for_label} {idx+1}",
        })
    if len(st.session_state.units) > n:
        st.session_state.units = st.session_state.units[:n]

def _duplicate_from_previous(i):
    prev = st.session_state.units[i-1]
    st.session_state.units[i] = {
        "components": prev["components"].copy(),
        "custom_label": prev.get("custom_label"),
    }

def _update_component(i, field, value):
    st.session_state.units[i]["components"][field] = value

def _maybe_update_labels_on_product_change(old_prod: str, new_prod: str):
    """If a unit's label still looks like the default for the old product, rewrite to the new product's short default."""
    if "units" not in st.session_state:
        return
    old_short = pretty_short(old_prod)
    new_short = pretty_short(new_prod)
    for idx, u in enumerate(st.session_state.units):
        current = (u.get("custom_label") or "").strip()
        exact_short = f"{old_short} {idx+1}"
        if current == exact_short or re.match(rf"^{re.escape(old_short)}.*\s{idx+1}\s*$", current):
            new_default = f"{new_short} {idx+1}"
            st.session_state.units[idx]["custom_label"] = new_default
            st.session_state[f"label_{idx}"] = new_default

# ===== Unit card (Step 2) =====
def render_unit_card(i: int, disabled: bool = False, product: str = "townhome"):
    u = st.session_state.units[i]
    with st.container(border=True):
        st.subheader(f"Option {i+1}")

        if i > 0 and st.button("Duplicate from previous", key=f"dup_{i}", disabled=disabled):
            _duplicate_from_previous(i); st.rerun()

        st.selectbox("**Energy Efficiency:** What energy code standard would you like to build to?",
                     [c for c in ENERGY_CODE_ORDER if c in options("energy_code", DEFAULT_PARENT)] or ["vt_energy_code","rbes","passive_house"],
                     format_func=pretty, key=f"code_{i}", disabled=disabled,
                     on_change=lambda: _update_component(i, "code", st.session_state[f"code_{i}"]))

        st.selectbox("**Heating:** How would you like to heat the home?",
                     options("energy_source", DEFAULT_PARENT) or ["natural_gas"],
                     format_func=pretty, key=f"src_{i}", disabled=disabled,
                     on_change=lambda: _update_component(i, "src", st.session_state[f"src_{i}"]))

        opt_fin_all = options("finish_quality", DEFAULT_PARENT) or ["below_average","average","above_average"]
        order_map = {"below_average":0, "average":1, "above_average":2}
        opt_fin = sorted(set(opt_fin_all), key=lambda k: order_map.get(k, 99))
        st.selectbox("**Quality:** How “nice” would you like the finish quality (kitchen, bathroom, flooring, etc.) to be?",
                     opt_fin, index=opt_fin.index("average") if "average" in opt_fin else 0,
                     format_func=pretty, key=f"fin_{i}", disabled=disabled,
                     on_change=lambda: _update_component(i, "fin", st.session_state[f"fin_{i}"]))

        current_infra_opt = st.session_state.units[i]["components"]["infra"]
        toggle_val = st.toggle("**Location:** In a new neighborhood",
                               value=(current_infra_opt == "yes"),
                               key=f"infra_toggle_{i}", disabled=disabled)
        new_infra_opt = bool_to_infra_opt(toggle_val)
        if new_infra_opt != current_infra_opt:
            st.session_state.units[i]["components"]["infra"] = new_infra_opt

        with st.expander("Advanced components", expanded=False):
            default_label = f"{pretty_short(product)} {i+1}"
            if not st.session_state.get(f"label_{i}"):
                st.session_state[f"label_{i}"] = st.session_state.units[i].get("custom_label", default_label)
            st.markdown("**Bar label**")
            st.session_state.units[i]["custom_label"] = st.text_input(
                " ", value=st.session_state[f"label_{i}"], key=f"label_{i}",
                label_visibility="collapsed", disabled=disabled
            )
            st.caption("Extras configured in data when available (e.g., Solar, Covered Parking).")

    label = st.session_state.units[i].get("custom_label", f"{pretty_short(product)} {i+1}")
    return {"label": label, "code": st.session_state[f"code_{i}"], "src": st.session_state[f"src_{i}"],
            "infra": st.session_state.units[i]["components"]["infra"], "fin": st.session_state[f"fin_{i}"]}

# ===== Header =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("This tool allows you to see how housing policy directly impacts whether Vermonters at various income levels are able to afford housing.")
st.write("“Build” one type of housing or compare multiple. Can you afford new construction in Vermont?")

# Top link bar (tiny HTML only to keep symmetric spaces around the bar)
st.markdown(
    '[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)'
    ' &nbsp;|&nbsp; '
    '[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)',
    unsafe_allow_html=True
)
st.divider()

# ===== Step 1 – Choose the Housing Type =====
st.header("Step 1 – Choose the Housing Type")

prev_prod = st.session_state.get("global_product_prev", "townhome")

product = st.radio(
    "**What kind of housing are we talking about?**",
    ["townhome","condo","apartment"],
    format_func=pretty,
    horizontal=False,
    key="global_product",)

st.write("")

if product != prev_prod:
    _maybe_update_labels_on_product_change(prev_prod, product)
    st.session_state["global_product_prev"] = product

apartment_mode = (product == "apartment")

if not apartment_mode:
    st.markdown("**Number of bedrooms**")
    br_opts = bedroom_options(product)
    default_idx = br_opts.index("2") if "2" in br_opts else 0
    bedrooms = st.radio(" ", br_opts, index=default_idx, format_func=pretty, horizontal=True,
                        key="global_bedrooms", label_visibility="collapsed")
    sf = bedroom_sf(product, bedrooms) or 1000.0
else:
    bedrooms, sf = None, None
    st.info("Apartment modeling (rent-based) coming soon. For now, choose Townhome or Condo to compare for-sale products.")

st.divider()

# ===== Step 2 – How do you want the housing built? =====
st.header("Step 2 – How do you want the housing built?")
if "num_units" not in st.session_state:
    st.session_state.num_units = 1

def _ensure_and_get_units():
    _ensure_units(st.session_state.num_units, product_key=product)
    return st.session_state.units

_ensure_and_get_units()
units = []
for i in range(st.session_state.num_units):
    units.append(render_unit_card(i, disabled=apartment_mode, product=product))
    st.write("")
st.divider()

# ===== Step 3 – Who can afford this home? =====
st.header("Step 3 – Who can afford this home?")

household_size = st.radio(
    "**Select household size**",
    list(range(1, 9)),
    index=3, horizontal=True, key="household_size",)

# Bounds from Chittenden mapping for selected bedrooms
if not apartment_mode and bedrooms is not None:
    p2i_b, i2p_b, inc_min_b, inc_max_b, *_ = build_price_income_transformers("Chittenden", int(household_size), int(bedrooms))
    if all(v is not None and np.isfinite(v) for v in (inc_min_b, inc_max_b)):
        min_income = int(np.floor(inc_min_b))
        max_income = int(np.ceil(inc_max_b))
    else:
        min_income, max_income = 20000, 300000
else:
    min_income, max_income = 20000, 300000

st.write("How much do you think a Vermont household needs to make to afford this home?")
default_income = int(np.clip(100000, min_income, max_income))
if "user_income" not in st.session_state:
    st.session_state["user_income"] = default_income
else:
    st.session_state["user_income"] = int(np.clip(st.session_state["user_income"], min_income, max_income))

st.number_input(
    " ", min_value=min_income, max_value=max_income, step=1000,
    key="user_income", format="%d", label_visibility="collapsed"
)
user_income = float(st.session_state["user_income"])

# Note 1 (grey) + spacer
st.write(
    f"Minimum/maximum income allowed for this household size: "
    f"{fmt_money(min_income)} – {fmt_money(max_income)}")
st.write("Note - *Statewide Median Household Income: $85,000*")

st.subheader("Let’s see how you did!")
show_results = st.toggle("View the home you built", value=False, key="view_home_toggle")

# ===== Results (Graph + Messaging) =====
def _ami_phrase(pct: int | None, region_label: str, capped_low: bool, capped_high: bool) -> str:
    """Readable AMI line with special phrasing at caps."""
    if pct is None:
        return f"—% of AMI in {region_label}"
    if capped_high:  # >150%
        return (
            f"More than 150% of AMI in the rest of Vermont."
            if region_label == "Rest of Vermont"
            else f"More than 150% of Area Median Income in {region_label}."
        )
    # <30% gets "(at least)"
    suffix = " (at least)" if capped_low else ""
    return (
        f"{pct}% of AMI in the rest of Vermont{suffix}."
        if region_label == "Rest of Vermont"
        else f"{pct}% of Area Median Income in {region_label}{suffix}."
    )

if show_results:
    if not apartment_mode:
        # Build labels and TDCs for current units
        labels, tdc_vals = [], []
        for idx, u in enumerate(units):
            label = u["label"] or f"{pretty_short(product)} {idx+1}"
            labels.append(label)
            tdc_vals.append(compute_tdc(sf, product, u["code"], u["src"], u["infra"], u["fin"]))

        # Chittenden mapping (fixed engine)
        p2i, i2p, *_ = build_price_income_transformers("Chittenden", int(household_size), int(bedrooms))
        if p2i is None or i2p is None:
            st.warning("Not enough data to build the price↔income mapping for this selection.")
        else:
            afford_price = float(i2p(np.array([user_income]))[0])
            draw_chart(labels, tdc_vals, afford_price, p2i, i2p)

            # Success / Failure messages (with labels for multi-home)
            prices = np.asarray(tdc_vals, dtype=float)
            req_incomes = p2i(prices)
            mask = affordable_mask(user_income, req_incomes)
            affordable_labels = [labels[i] for i, ok in enumerate(mask) if ok]

            if len(labels) == 1:
                if len(affordable_labels) == 1:
                    st.success("Congrats! The total development cost of this home is less than the income required to buy it.")
                else:
                    st.error("Not quite! This home is unaffordable for the household income you entered.")
            else:
                k = len(affordable_labels)
                if k == 0:
                    st.error("Not quite! These homes are unaffordable for the household income you entered.")
                elif k == len(labels):
                    st.success("Congrats! The total development cost for all options are less than the income required to buy them.")
                elif k == 1:
                    st.success(f"Congrats! The total development cost for **{affordable_labels[0]}** is less than the income required to buy it.")
                else:  # k == 2
                    st.success(
                        f"Congrats! The total development cost for **{affordable_labels[0]}** & **{affordable_labels[1]}** "
                        "are less than the income required to buy them."
                    )

            # Space between callout and info boxes
            st.write("")

            # "More About ..." boxes per option
            for idx, label in enumerate(labels):
                req_inc = float(p2i(np.array([tdc_vals[idx]]))[0])
                title = "More About This Home" if len(labels) == 1 else f"More About {label}"

                with st.container(border=True):
                    st.subheader(title)

                    # Top bullets
                    st.markdown(f"- You would need to have a household income of **{fmt_money(req_inc)}** to afford this home.")
                    st.markdown("- This is only affordable for **0 of the 270,000 households in Vermont**.")
                    st.markdown("- To afford this home, you would need to make:")

                    # Sub-bullets for the 3 regions (handles caps and phrasing)
                    for rp in ["Chittenden", "Addison", "Rest of Vermont"]:
                        reg_key_line = PRETTY2REG[rp]
                        pct, capped = ami_percent_for_income(reg_key_line, int(household_size), req_inc)
                        capped_low = (pct == 30 and capped)
                        capped_high = (pct == 150 and capped)
                        st.markdown(f"  - {_ami_phrase(pct, rp, capped_low, capped_high)}")

                st.write("")

            # ===== Compare controls =====
            st.subheader("Want to try again? Build another option (or two!) and compare to your first attempt")
            compare_choice = st.radio(
                "**How many homes do you want to build?**",
                [1, 2, 3],
                index={1:0, 2:1, 3:2}[st.session_state.num_units],
                horizontal=True,
                format_func=lambda n: {1:"1 home (current setting)", 2:"2 homes", 3:"3 homes"}[n],
            )
            if compare_choice != st.session_state.num_units:
                st.session_state.num_units = compare_choice
                _ensure_and_get_units()
                st.rerun()
    else:
        st.info("Select Townhome or Condo to run the for-sale model. Apartment model (rent) coming soon.")
