"""
Streamlit Housing Affordability Visualizer

This program compares estimated housing development costs to income-based affordability
thresholds across Vermont. Users select unit types, sizes, regions, and AMI levels to
generate a visual comparison of what it costs to build versus what households can afford.
The goal is to show how high development costs make typical homes unaffordable for most
people, limiting demand and discouraging new market-rate housing construction.

Author: Alex Bleich
Date: June 23rd, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

# -------------------- CONFIGURATION --------------------
data_files = {
    "Chittenden": pd.read_csv("chittenden_ami.csv"),
    "Addison": pd.read_csv("addison_ami.csv"),
    "Vermont": pd.read_csv("vermont_ami.csv")
}
cost_df = pd.read_csv("construction_costs.csv")

# -------------------- INTERFACE --------------------
st.title("🏘️ Housing Affordability Visualizer")
st.markdown("Compare unit development costs with AMI-based affordability thresholds across Vermont regions.")
st.markdown("[View program & method on GitHub](https://github.com/alexbleich/housing-affordability-tool)")


valid_unit_types = ['Apartment', 'Townhome', 'Condo']
unit_labels, development_costs, square_feet_list = [], [], []

num_units = st.slider("How many units would you like to compare?", 1, 5, 1)

for i in range(num_units):
    st.subheader(f"Unit {i + 1}")
    unit_type = st.selectbox("Unit type", valid_unit_types, key=f"type_{i}")
    square_feet = st.number_input("Square footage", min_value=1, max_value=5000, key=f"sf_{i}")
    square_feet_list.append(square_feet)

    est_bedrooms = max(1, min(round((square_feet * 0.28) / 200), 5))
    st.markdown(f"<span style='color: white; font-weight: bold;'>Estimated bedrooms: {est_bedrooms}</span>", unsafe_allow_html=True)

    row = cost_df[cost_df['unit_type'].str.lower() == unit_type.lower()]
    if not row.empty:
        cost_per_sf = row['cost_per_sf'].values[0]
        unit_labels.append(f"{int(square_feet)}sf {unit_type}")
        development_costs.append(cost_per_sf * square_feet)

# Estimate avg bedroom count for affordability lookups
if square_feet_list:
    avg_sf = sum(square_feet_list) / len(square_feet_list)
    bedrooms = max(1, min(round((avg_sf * 0.28) / 200), 5))
else:
    bedrooms = 1

# -------------------- AMI SELECTION --------------------
st.subheader("Income Thresholds")
selected_regions = st.multiselect("Select region(s)", list(data_files.keys()), default=[])
valid_ami_values = [30] + list(range(50, 155, 5))
num_amis = st.slider("How many AMI levels?", 1, 3, 1)

amis = []
for i in range(num_amis):
    ami = st.selectbox(f"AMI value #{i + 1}", valid_ami_values, key=f"ami_{i}")
    if ami not in amis:
        amis.append(ami)

# -------------------- AFFORDABILITY THRESHOLDS --------------------
affordability_lines = {}
col_name = f"buy{bedrooms}"

for region in selected_regions:
    df = data_files[region]
    df.columns = df.columns.str.strip().str.lower()
    df['ami'] = pd.to_numeric(df['ami'], errors='coerce')
    for ami in amis:
        row = df[df['ami'] == ami / 100]
        if not row.empty and col_name in row:
            affordability_lines[f"{ami}% AMI - {region}"] = float(row[col_name].values[0])
        else:
            st.warning(f"{ami}% AMI or '{col_name}' not found in {region} dataset.")

# -------------------- PLOTTING --------------------
if unit_labels and development_costs and any(development_costs):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(unit_labels, development_costs, color='skyblue', edgecolor='black')
    ymax = max(development_costs) * 1.1
    ax1.set_ylim(0, ymax)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + (ymax * 0.02), f"${yval:,.0f}", ha='center', va='bottom', fontsize=9)

    for i, (label, value) in enumerate(affordability_lines.items()):
        ax1.axhline(y=value, linestyle='--', color=f"C{i}", label=label)

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    ax1.set_ylabel("Development Cost ($)")
    plt.xticks(rotation=20)
    plt.title("Development Cost vs. Affordable Purchase Price Thresholds")
    if affordability_lines:
        ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(list(affordability_lines.values()))
    ax2.set_yticklabels([f"{k.split()[0]}\n${affordability_lines[k]:,.0f}" for k in affordability_lines])
    ax2.set_ylabel("Affordability by % AMI")

    fig.tight_layout()
    st.pyplot(fig)

st.markdown("[VHFA Housing Data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf)")
