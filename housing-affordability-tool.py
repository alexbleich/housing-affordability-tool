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
DEFAULT_AMIS = [150, 120, 100]
AMI_COL = "ami"
DEFAULT_PARENT = "default"

PKG = {
    "baseline": {"label": "Baseline",          "code": "vt_energy_code",  "src": "natural_gas", "infra": "no", "fin": "average"},
    "top":      {"label": "Top-of-the-Line",   "code": "passive_house",   "src": "geothermal",  "infra": "no", "fin": "above_average"},
    "below":    {"label": "Below Baseline",    "code": "base_me_nh_code", "src": "natural_gas", "infra": "no", "fin": "below_average"},
}

# ===== Data Loading =====
@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_assumptions(p: Path) -> pd.DataFrame:
    if not p.exists():
        st.error(f"Missing assumptions file: {p}"); st.stop()
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.lower()
    must_have = {"category","parent_option","option","value_type","unit","value"}
    missing = must_have - set(df.columns)
    if missing:
        st.error(f"Assumptions CSV missing columns: {missing}"); st.stop()
    for c in ("category","parent_option","option","value_type","unit"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: str(p)})
def load_regions(files: dict) -> dict:
    out = {}
    for name, p in files.items():
        if not p.exists():
            st.error(f"Missing region file for {name}: {p}"); st.stop()
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        if AMI_COL not in d.columns:
            st.error(f"Region file {name} missing '{AMI_COL}' column."); st.stop()
        d[AMI_COL] = pd.to_numeric(d[AMI_COL], errors="coerce")
        out[name] = d
    return out

A = load_assumptions(ASSUMP)
R = load_regions(REGIONS)

# ===== Helpers =====
PRETTY_OVERRIDES = {
    "townhome":"Townhome","condo":"Condo","apartment":"Apartment","studio":"Studio",
    "base_me_nh_code":"ME/NH Base Code","vt_energy_code":"VT Energy Code",
    "evt_high_eff":"EVT High-Efficient","passive_house":"Passive House",
    "natural_gas":"Natural Gas Heating","all_electric":"All Electric","geothermal":"Geothermal",
    "yes":"Yes","no":"No","above_average":"Above Average","average":"Average","below_average":"Below Average",
}
TOKEN_UPPER = {" Ami":" AMI"," Vt ":" VT "," Nh ":" NH "," Me ":" ME "," Evt ":" EVT "," Mf ":" MF "}

def pretty(x: str) -> str:
    s = str(x).lower().strip()
    if s in PRETTY_OVERRIDES: return PRETTY_OVERRIDES[s]
    t = s.replace("_"," ").title()
    for k, v in TOKEN_UPPER.items():
        t = t.replace(k, v)
    return t

def fmt_money(x):
    val = pd.to_numeric(x, errors="coerce")
    return "—" if pd.isna(val) else f"${val:,.0f}"

def _rows(cat, opt=None, parent=None):
    q = A["category"].eq(cat)
    if parent is not None: q &= A["parent_option"].eq(str(parent).lower())
    if opt    is not None: q &= A["option"].eq(str(opt).lower())
    return A.loc[q]

def one_val(cat, opt, parent=None, expect_type=None, default=0.0):
    r = _rows(cat, opt, parent)
    if r.empty and parent is not None:
        r = _rows(cat, opt)
    if r.empty:
        return default
    if expect_type and r.iloc[0]["value_type"] != expect_type:
        return default
    return float(r.iloc[0]["value"])

def _overlay_sums(category: str, option: str, parents: list) -> tuple:
    df = A[(A["category"].eq(category)) & (A["option"].eq(option)) & (A["parent_option"].isin(parents))]
    if df.empty: return 0.0, 0.0, 0.0
    psf = float(pd.to_numeric(df.loc[df["value_type"].eq("per_sf"),   "value"], errors="coerce").sum())
    pu  = float(pd.to_numeric(df.loc[df["value_type"].eq("per_unit"),"value"], errors="coerce").sum())
    fx  = float(pd.to_numeric(df.loc[df["value_type"].eq("fixed"),   "value"], errors="coerce").sum())
    return psf, pu, fx

def options(cat, parent=None): return _rows(cat, parent=parent)["option"].tolist()

def bedroom_sf(h_type, br_label):
    r = _rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def compute_tdc(sf, htype, code, src, infra, fin):
    base_psf = baseline_per_sf()
    mf_mult  = mf_factor(htype)
    pct_mult = (one_val("energy_code", code) + one_val("finish_quality", fin)) / 100.0
    core_psf = base_psf * (mf_mult + pct_mult)

    parents = [htype, DEFAULT_PARENT]

    es_psf, es_pu, es_fx       = _overlay_sums("energy_source", src, parents)
    inf_psf, inf_pu, inf_fx    = _overlay_sums("infrastructure", infra, parents)

    other = A[~A["category"].isin({"baseline_cost","mf_efficiency_factor","energy_code","finish_quality","energy_source","infrastructure","bedrooms"})]
    other = other[(other["option"].eq("default")) & (other["parent_option"].isin(parents))]

    other_psf = float(other.loc[other["value_type"].eq("per_sf"),  "value"].sum()) if not other.empty else 0.0
    other_pu  = float(other.loc[other["value_type"].eq("per_unit"),"value"].sum()) if not other.empty else 0.0
    other_fx  = float(other.loc[other["value_type"].eq("fixed"),  "value"].sum()) if not other.empty else 0.0

    total = 0.0
    total += sf * (core_psf + es_psf + inf_psf + other_psf)
    total += es_pu + inf_pu + other_pu
    total += es_fx + inf_fx + other_fx
    return total

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
                lines[f"{ami_capped}% AMI - {rp}"] = float(pd.to_numeric(df.loc[m, col], errors="coerce").iloc[0])
    return lines

def make_price_to_income_mapper(region_key: str, hh_size: int, bed_n: int):
    df = R[region_key]
    inc_col = f"income{hh_size}"
    buy_col = f"buy{bed_n}"
    if not {AMI_COL, inc_col, buy_col}.issubset(df.columns):
        return None
    sub = df[[buy_col, inc_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return None
    sub = sub.sort_values(buy_col)
    x = sub[buy_col].to_numpy(dtype=float)
    y = sub[inc_col].to_numpy(dtype=float)
    if len(x) < 2:
        return (lambda price: float(y[0]))
    slope_lo = (y[1] - y[0]) / (x[1] - x[0])
    slope_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
    def price_to_income(price: float) -> float:
        if price is None or not np.isfinite(price): return np.nan
        p = float(price)
        if p <= x[0]:  return float(y[0] + slope_lo * (p - x[0]))
        if p >= x[-1]: return float(y[-1] + slope_hi * (p - x[-1]))
        return float(np.interp(p, x, y))
    return price_to_income

# ===== Plot Utilities =====
def _bar_with_values(ax, labels, values, pad_ratio):
    bars = ax.bar(labels, values, color="skyblue", edgecolor="black")
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y * (1 + pad_ratio), fmt_money(y), ha="center", va="bottom", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))

def _apply_ylim(ax, ax2, ymax):
    ax.set_ylim(0, ymax)
    if ax2 is not None:
        ax2.set_ylim(0, ymax)

# ===== Plotting =====
def draw_chart1(labels, tdc_vals, lines):
    fig, ax = plt.subplots(figsize=(12, 6))
    top_bar = max(tdc_vals) if tdc_vals else 1.0
    top_line = max(lines.values()) if lines else 0.0
    ymax = 1.2 * max(top_bar, top_line, 1.0)
    _bar_with_values(ax, labels, tdc_vals, pad_ratio=0.02)
    if lines:
        for i, (lab, val) in enumerate(sorted(lines.items(), key=lambda kv: kv[0])):
            ax.axhline(y=val, linestyle="--", color=f"C{i}", label=lab)
    _apply_ylim(ax, None, ymax)
    ax2 = ax.twinx()
    _apply_ylim(ax, ax2, ymax)
    if lines:
        ordered = [k for k, _ in sorted(lines.items(), key=lambda kv: kv[0])]
        vals = [lines[k] for k in ordered]
        ax2.set_yticks(vals)
        ax2.set_yticklabels([f"{k.split('%')[0].strip()}%\n{fmt_money(lines[k])}" for k in ordered])
        ax.legend(loc="upper right")
    else:
        ax2.set_yticks([])
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_xlabel("")
    ax2.set_ylabel("Max. Affordable Purchase Price (% AMI)")
    ax.set_title("Total Development Cost vs. What Households Can Afford")
    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

def draw_chart2(labels, tdc_vals, afford_price, price_to_income):
    fig, ax = plt.subplots(figsize=(12, 6))
    top_bar = max(tdc_vals) if tdc_vals else 1.0
    top_line = float(afford_price) if afford_price is not None else 0.0
    ymax = 1.2 * max(top_bar, top_line, 1.0)
    _bar_with_values(ax, labels, tdc_vals, pad_ratio=0.025)
    line_handle = None
    if afford_price is not None:
        line_handle = ax.axhline(y=float(afford_price), linestyle="-", linewidth=2.8, color="#2E7D32",
                                 label="Income level mapped to affordable purchase price")
    _apply_ylim(ax, None, ymax)
    rax = ax.twinx()
    _apply_ylim(ax, rax, ymax)
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_xlabel("")
    ax.set_title("Purchase Ability by Income, Household Size, and Region")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_money(x)))
    if price_to_income is not None:
        left_ticks = ax.get_yticks()
        left_min, left_max = ax.get_ylim()
        right_ticks = [t for t in left_ticks if (t > left_min) and (t < left_max)]
        rax.set_yticks(right_ticks[:-1])
        rax.set_yticklabels([fmt_money(price_to_income(t)) for t in right_ticks[:-1]])
    else:
        rax.set_yticks([])
    rax.set_ylabel("Income Req. to Purchase")
    if line_handle is not None:
        ax.legend(handles=[line_handle], loc="upper right", frameon=True)
    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

# ===== Session State (units) =====
def _ensure_units(n):
    if "units" not in st.session_state:
        st.session_state.units = []
    while len(st.session_state.units) < n:
        st.session_state.units.append({
            "package": "baseline",
            "components": dict(code=PKG["baseline"]["code"], src=PKG["baseline"]["src"],
                               infra=PKG["baseline"]["infra"], fin=PKG["baseline"]["fin"]),
            "is_custom": False,
            "custom_label": "Custom",
        })
    if len(st.session_state.units) > n:
        st.session_state.units = st.session_state.units[:n]

def _apply_package(i, pkg_key):
    u = st.session_state.units[i]
    u["package"] = pkg_key
    u["components"] = dict(code=PKG[pkg_key]["code"], src=PKG[pkg_key]["src"],
                           infra=PKG[pkg_key]["infra"], fin=PKG[pkg_key]["fin"])
    u["is_custom"] = False

def _update_component(i, field, value):
    u = st.session_state.units[i]
    u["components"][field] = value
    p = PKG[u["package"]]
    u["is_custom"] = any([
        u["components"]["code"]  != p["code"],
        u["components"]["src"]   != p["src"],
        u["components"]["infra"] != p["infra"],
        u["components"]["fin"]   != p["fin"],
    ])

def _duplicate_from_previous(i):
    prev = st.session_state.units[i-1]
    st.session_state.units[i] = {
        "package": prev["package"],
        "components": prev["components"].copy(),
        "is_custom": prev["is_custom"],
        "custom_label": prev.get("custom_label", "Custom"),
    }

# ===== Header =====
st.title("🏘️ Housing Affordability Visualizer")
st.write("Pick your policies below to see how it affects affordability.")
st.markdown("[View all assumptions and code here](https://github.com/alexbleich/housing-affordability-tool)")
st.write("")

# ===== Step 1 — Housing Type =====
st.header("Step 1 — Housing Type")
product = st.radio("What kind of housing are we talking about?",
                   ["townhome","condo","apartment"],
                   format_func=pretty, horizontal=True, key="global_product")
apartment_mode = (product == "apartment")
if not apartment_mode:
    br_opts = options("bedrooms", parent=product) or ["2"]
    bedrooms = st.selectbox("Number of bedrooms",
                            [*br_opts], index=(br_opts.index("2") if "2" in br_opts else 0),
                            format_func=pretty, key="global_bedrooms")
    sf = bedroom_sf(product, bedrooms) or 1000.0
else:
    bedrooms, sf = None, None
    st.info("Apartment modeling (rent-based) coming soon. For now, choose Townhome or Condo to compare for-sale products.")

st.divider()

# ===== Step 2 — Pick your Policies =====
st.header("Step 2 — Pick your Policies")
num_units = st.selectbox("How many units would you like to compare?", [1,2,3,4,5], index=1, disabled=apartment_mode)

def _ensure_and_get_units():
    _ensure_units(num_units)
    return st.session_state.units
_ = _ensure_and_get_units()

def render_unit_card(i: int, disabled: bool = False):
    u = st.session_state.units[i]
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
        # Package selection + actions
        cols = st.columns([1, 1], vertical_alignment="center")
        with cols[0]:
            pkg_disabled = disabled or u["is_custom"]
            pkg_choice = st.radio(
                "Policy package",
                options=list(PKG.keys()),
                format_func=lambda k: PKG[k]["label"],
                index=list(PKG.keys()).index(u["package"]),
                key=f"pkg_{i}",
                disabled=pkg_disabled
            )
            if u["is_custom"] and not disabled:
                st.caption(f"Modified from “{PKG[u['package']]['label']}”. Click “Reset to package” to change package.")
        with cols[1]:
            c1, c2 = st.columns([1, 1])
            with c1:
                if i > 0 and st.button("Duplicate from previous", key=f"dup_{i}", disabled=disabled):
                    _duplicate_from_previous(i)
                    st.rerun()
            with c2:
                if u["is_custom"] and st.button("Reset to package", key=f"reset_{i}", disabled=disabled):
                    _apply_package(i, u["package"])
                    st.rerun()

        # Apply package change (only when not custom and not disabled)
        if not (u["is_custom"] or disabled) and pkg_choice != u["package"]:
            _apply_package(i, pkg_choice)

        # Advanced component adjustments (code/src/fin only)
        with st.expander("Advanced: adjust components", expanded=False):
            opt_code = options("energy_code", DEFAULT_PARENT) or ["vt_energy_code"]
            opt_src  = options("energy_source", DEFAULT_PARENT) or ["natural_gas"]
            opt_fin  = options("finish_quality", DEFAULT_PARENT) or ["average", "above_average", "below_average"]

            code = st.selectbox(
                "Energy code standard", opt_code,
                index=(opt_code.index(u["components"]["code"]) if u["components"]["code"] in opt_code else 0),
                format_func=pretty, key=f"code_{i}", disabled=disabled
            )
            src = st.selectbox(
                "Energy source", opt_src,
                index=(opt_src.index(u["components"]["src"]) if u["components"]["src"] in opt_src else 0),
                format_func=pretty, key=f"src_{i}", disabled=disabled
            )
            fin = st.selectbox(
                "Finish quality", opt_fin,
                index=(opt_fin.index(u["components"]["fin"]) if u["components"]["fin"] in opt_fin else 0),
                format_func=pretty, key=f"fin_{i}", disabled=disabled
            )

            for field, val in [("code", code), ("src", src), ("fin", fin)]:
                if val != u["components"][field]:
                    _update_component(i, field, val)

            if u["is_custom"]:
                u["custom_label"] = st.text_input(
                    "Bar label", value=u.get("custom_label", "Custom"),
                    key=f"label_{i}", disabled=disabled
                )

        # Infrastructure toggle (placed below Advanced, still inside the card)
        infra_allowed = set(options("infrastructure", product) or ["no", "yes"])
        current_infra_yes = (u["components"]["infra"] == "yes")
        infra_toggle = st.toggle(
            "Infrastructure required?", value=current_infra_yes,
            key=f"infra_{i}", disabled=disabled
        )
        infra = "yes" if (infra_toggle and "yes" in infra_allowed) else "no"
        if infra != u["components"]["infra"]:
            _update_component(i, "infra", infra)

        # Final label for the bar
        label = PKG[u["package"]]["label"] if not u["is_custom"] else (u.get("custom_label") or "Custom")

    return {
        "label": label,
        "code": u["components"]["code"],
        "src": u["components"]["src"],
        "infra": u["components"]["infra"],
        "fin": u["components"]["fin"],
    }

units = []
for i in range(num_units):
    units.append(render_unit_card(i, disabled=apartment_mode))
    st.write("")

st.divider()

# ===== Step 3 — Compare Costs with Affordability =====
st.header("Step 3 — Compare Costs with Affordability")
st.subheader("Affordability Thresholds")
with st.container(border=True):
    n_amis = st.selectbox("How many Area Median Income (AMI) levels?", [1,2,3], index=0)
    def default_ami_for_idx(i):
        defaults3 = [100, 150, 80]
        if n_amis == 3: return defaults3[i]
        return [100, 120][i] if n_amis == 2 else 100
    amis = []
    for i in range(n_amis):
        ami_val = st.selectbox(
            f"AMI value #{i+1}",
            VALID_AMIS,
            index=VALID_AMIS.index(default_ami_for_idx(i)) if default_ami_for_idx(i) in VALID_AMIS else VALID_AMIS.index(100),
            key=f"ami_{i}"
        )
        amis.append(ami_val)
    region_pretty_opts = [REGION_PRETTY[k] for k in REGIONS]
    sel_regions_pretty = st.multiselect("Select region(s)", region_pretty_opts, default=[REGION_PRETTY["Chittenden"]])

# ===== Chart 1 =====
if not apartment_mode and units:
    st.subheader("Do These Policy Choices Put Homes Within Reach?")
    labels, tdc_vals = [], []
    for u in units:
        labels.append(u["label"])
        tdc_vals.append(compute_tdc(sf, product, u["code"], u["src"], u["infra"], u["fin"]))
    lines = affordability_lines(sel_regions_pretty, amis, pick_afford_col(int(bedrooms), product))
    draw_chart1(labels, tdc_vals, lines)
elif apartment_mode:
    st.info("Select Townhome or Condo to run the for-sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

st.divider()

# ===== Step 4 — Specify Household Context =====
st.header("Step 4 — Specify Household Context")
st.subheader("Household Settings")
st.caption("Select region, household size, and income to assess affordability for local households.")
with st.container(border=True):
    region_list_pretty = [REGION_PRETTY[k] for k in REGIONS]
    region_single = st.selectbox("Region", region_list_pretty, index=region_list_pretty.index("Chittenden"))
    household_size = st.selectbox("Select household size", list(range(1,9)), index=3)
    user_income = st.number_input("Household income", min_value=20000, max_value=300000, step=1000, value=100000, format="%d")
    st.caption("Note: AMI capped at 150%. Inputs above 150% use 150%")

# ===== Chart 2 =====
if not apartment_mode and units:
    st.subheader("What These Costs Mean for Your Constituents")
    reg_key = PRETTY2REG[region_single]

    def household_ami_percent(region_key: str, hh_size: int, income_val: float):
        df = R[region_key]; inc_col = f"income{hh_size}"
        if not {AMI_COL, inc_col}.issubset(df.columns): return None
        sub = df[[inc_col, AMI_COL]].dropna().sort_values(inc_col)
        if sub.empty: return None
        x = sub[inc_col].astype(float).to_numpy()
        y = (sub[AMI_COL]*100.0).astype(float).to_numpy()
        if len(x) == 1: return float(y[0])
        xv = float(np.clip(income_val, x[0], x[-1]))
        return float(np.interp(xv, x, y))

    def affordability_at_percent(region_key: str, bed_n: int, ami_percent: float):
        df = R[region_key]; col = f"buy{bed_n}"
        if not {AMI_COL, col}.issubset(df.columns): return None
        sub = df[[AMI_COL, col]].dropna().sort_values(AMI_COL)
        if sub.empty: return None
        x = (sub[AMI_COL]*100.0).astype(float).to_numpy()
        y = sub[col].astype(float).to_numpy()
        if len(x) == 1: return float(y[0])
        xv = float(np.clip(ami_percent, x[0], x[-1]))
        return float(np.interp(xv, x, y))

    ami_pct = household_ami_percent(reg_key, int(household_size), int(user_income))
    afford_price = affordability_at_percent(reg_key, int(bedrooms), ami_pct) if ami_pct is not None else None
    price_to_income = make_price_to_income_mapper(reg_key, int(household_size), int(bedrooms))

    draw_chart2(labels, tdc_vals, afford_price, price_to_income)

    cheapest_price = min(tdc_vals) if tdc_vals else None
    required_income = price_to_income(cheapest_price) if (cheapest_price is not None and price_to_income is not None) else None
    ok = (required_income is not None) and np.isfinite(required_income) and (float(user_income) >= float(required_income))

    if ok:
        st.markdown(
            f"""<div style="padding:0.5rem 0.75rem;border-radius:8px;background:#E6F4EA;color:#1E7D34;border:1px solid #C8E6C9;">
            ✅ <b>Success:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>),
            you can afford <b>the cheapest option</b>.
            </div>""",
            unsafe_allow_html=True
        )
    else:
        need_text = fmt_money(required_income) if (required_income is not None and np.isfinite(required_income)) else "—"
        st.markdown(
            f"""<div style="padding:0.5rem 0.75rem;border-radius:8px;background:#FDECEA;color:#B71C1C;border:1px solid #F5C6CB;">
            ❌ <b>Keep trying:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>),
            none of the options are affordable. A household of this size would need to make at least <b>{need_text}</b> to afford the cheapest option.
            </div>""",
            unsafe_allow_html=True
        )

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")
