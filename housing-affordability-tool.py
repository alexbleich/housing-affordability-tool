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

# Policy packages
PKG = {
    "baseline": {
        "label": "Baseline",
        "code": "vt_energy_code", "src": "natural_gas", "infra": "no", "fin": "average",
    },
    "top": {
        "label": "Top-of-the-Line",
        "code": "passive_house", "src": "geothermal", "infra": "yes", "fin": "above_average",
    },
    "below": {
        "label": "Below Baseline",
        "code": "base_me_nh_code", "src": "natural_gas", "infra": "no", "fin": "below_average",
    },
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

def options(cat, parent=None): return _rows(cat, parent=parent)["option"].tolist()

def bedroom_sf(h_type, br_label):
    r = _rows("bedrooms", br_label, h_type)
    return float(r.iloc[0]["value"]) if not r.empty else np.nan

def mf_factor(h_type): return one_val("mf_efficiency_factor","default",h_type)
def baseline_per_sf(): return one_val("baseline_cost","baseline")

def compute_tdc(sf, htype, code, src, infra, fin):
    base = baseline_per_sf()
    pct = (one_val("energy_code", code) + one_val("finish_quality", fin)) / 100.0
    per_sf = base * (mf_factor(htype) + pct)
    per_sf += one_val("energy_source", src, DEFAULT_PARENT, "per_sf")
    per_sf += one_val("infrastructure", infra, DEFAULT_PARENT, "per_sf")
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
            ami_capped = min(ami, 150)  # enforce cap
            m = df[AMI_COL].eq(ami_capped/100.0)
            if m.any():
                lines[f"{ami_capped}% AMI - {rp}"] = float(pd.to_numeric(df.loc[m, col], errors="coerce").iloc[0])
    return lines

def nearest_ami_row(df: pd.DataFrame, hh_col: str, income: float) -> pd.Series:
    inc = pd.to_numeric(df.get(hh_col), errors="coerce")
    ami = pd.to_numeric(df.get(AMI_COL), errors="coerce")
    mask = inc.notna() & ami.notna()
    if not mask.any():
        return pd.Series(dtype=float)
    inc = inc[mask]; ami = ami[mask]
    order = inc.argsort()
    inc = inc.iloc[order]; ami = ami.iloc[order]
    diffs = (inc - income).abs()
    idx = diffs[diffs.eq(diffs.min())].index[-1]
    return pd.Series({"income": float(inc.loc[idx]), "ami_frac": float(ami.loc[idx])}, name=idx)

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

def draw_chart2(labels, tdc_vals, afford_price, user_income):
    """
    Bars = TDC per unit. Single green line = affordable purchase price at the selected income/household.
    - Legend shows the green line with the exact text required (no auto-named '_child*').
    - Right Y-axis label is pushed far right so it doesn't collide with the green annotation.
    - Top of chart = 1.2 × max(top bar, affordability line).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Y max rule: include bars and the affordability line
    top_bar = max(tdc_vals) if tdc_vals else 1.0
    top_line = float(afford_price) if afford_price is not None else 0.0
    ymax = 1.2 * max(top_bar, top_line, 1.0)

    # Bars + value labels
    _bar_with_values(ax, labels, tdc_vals, pad_ratio=0.025)

    # Affordability line with explicit label for the legend
    line_handle = None
    if afford_price is not None:
        line_handle = ax.axhline(
            y=float(afford_price),
            linestyle="-",
            linewidth=2.8,
            color="#2E7D32",
            label="Income level mapped to affordable purchase price",  # <- legend text
        )
        ax.annotate(
            f"Your income:\n{fmt_money(user_income)}",
            xy=(1.0, float(afford_price)), xycoords=("axes fraction", "data"),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=10, color="#2E7D32", multialignment="center",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none", pad=0),
        )

    # Apply consistent limits; create right axis and push its label to the far right
    _apply_ylim(ax, None, ymax)
    rax = ax.twinx()
    _apply_ylim(ax, rax, ymax)
    rax.set_yticks([])  # ticks hidden by design
    rax.set_ylabel("Income Req. to Purchase")
    rax.yaxis.set_label_coords(1.12, 0.5)  # push right so it sits beyond the green annotation

    # Titles / axis labels (no X label)
    ax.set_ylabel("Total Development Cost (TDC)")
    ax.set_xlabel("")
    ax.set_title("Purchase Ability by Income, Household Size, and Region")

    # Legend: show only the green line with the exact label (no '_child*')
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

# ===== Step 2 — How Different Policies Affect the Same Unit =====
st.header("Step 2 — How Different Policies Affect the Same Unit")
num_units = st.selectbox("How many units would you like to compare?", [1,2,3,4,5], index=1, disabled=apartment_mode)

def _ensure_and_get_units():
    _ensure_units(num_units)
    return st.session_state.units
units_state = _ensure_and_get_units()

def render_unit_card(i: int, disabled: bool = False):
    u = st.session_state.units[i]
    st.subheader(f"{pretty(product)} {i+1}")
    with st.container(border=True):
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
            c1, c2 = st.columns([1,1])
            with c1:
                if i > 0 and st.button("Duplicate from previous", key=f"dup_{i}", disabled=disabled):
                    _duplicate_from_previous(i); st.rerun()
            with c2:
                if u["is_custom"] and st.button("Reset to package", key=f"reset_{i}", disabled=disabled):
                    _apply_package(i, u["package"]); st.rerun()

        if not (u["is_custom"] or disabled) and pkg_choice != u["package"]:
            _apply_package(i, pkg_choice)

        with st.expander("Advanced: adjust components", expanded=False):
            opt_code  = options("energy_code", DEFAULT_PARENT) or ["vt_energy_code"]
            opt_src   = options("energy_source", DEFAULT_PARENT) or ["natural_gas"]
            opt_infra = options("infrastructure", DEFAULT_PARENT) or ["no","yes"]
            opt_fin   = options("finish_quality", DEFAULT_PARENT) or ["average","above_average","below_average"]

            code  = st.selectbox("Energy code standard", opt_code,
                                 index=(opt_code.index(u["components"]["code"]) if u["components"]["code"] in opt_code else 0),
                                 format_func=pretty, key=f"code_{i}", disabled=disabled)
            src   = st.selectbox("Energy source", opt_src,
                                 index=(opt_src.index(u["components"]["src"]) if u["components"]["src"] in opt_src else 0),
                                 format_func=pretty, key=f"src_{i}", disabled=disabled)
            infra = st.selectbox("Infrastructure required?", opt_infra,
                                 index=(opt_infra.index(u["components"]["infra"]) if u["components"]["infra"] in opt_infra else 0),
                                 format_func=pretty, key=f"infra_{i}", disabled=disabled)
            fin   = st.selectbox("Finish quality", opt_fin,
                                 index=(opt_fin.index(u["components"]["fin"]) if u["components"]["fin"] in opt_fin else 0),
                                 format_func=pretty, key=f"fin_{i}", disabled=disabled)

            for field, val in [("code", code), ("src", src), ("infra", infra), ("fin", fin)]:
                if val != u["components"][field]:
                    _update_component(i, field, val)

            if u["is_custom"]:
                u["custom_label"] = st.text_input("Bar label", value=u.get("custom_label", "Custom"), key=f"label_{i}", disabled=disabled)

        label = PKG[u["package"]]["label"] if not u["is_custom"] else (u.get("custom_label") or "Custom")

    return {"label": label, "code": u["components"]["code"], "src": u["components"]["src"],
            "infra": u["components"]["infra"], "fin": u["components"]["fin"]}

units = []
for i in range(num_units):
    units.append(render_unit_card(i, disabled=apartment_mode))
    st.write("")

st.divider()

# ===== Chart 1 Controls (always on, above chart) =====
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
    st.info("Select Townhome or Condo to run the for‑sale model. Apartment model (rent) coming soon.")
else:
    st.info("No valid unit data provided.")

st.divider()

# ===== Step 3 — Specify household context =====
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
            gap = min(tdc_vals) - (afford_price or 0)
            st.markdown(
                f"""<div style="padding:0.5rem 0.75rem;border-radius:8px;background:#FDECEA;color:#B71C1C;border:1px solid #F5C6CB;">
                ❌ <b>Not yet:</b> At your income (<b>{fmt_money(user_income)}</b>) and household size (<b>{household_size}</b>),
                none of the options are affordable. Shortfall vs. lowest‑cost option: <b>{fmt_money(gap)}</b>.
                </div>""", unsafe_allow_html=True)

st.write("")
st.markdown("[VHFA Affordability Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")
