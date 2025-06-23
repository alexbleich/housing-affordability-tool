import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter

# CONFIGURATION
data_files = {
    "Chittenden": pd.read_csv("chittenden_ami.csv"),
    "Addison": pd.read_csv("addison_ami.csv"),
    "Vermont": pd.read_csv("vermont_ami.csv")
}
cost_df = pd.read_csv("construction_costs.csv")

# Streamlit Interface
st.title("Housing Affordability Tool")

# Unit Comparisons
num_units = st.slider("How many units would you like to compare?", 1, 5, 1)
unit_labels, development_costs, unit_square_feet = [], [], []
valid_unit_types = ['Apartment', 'Townhome', 'Condo']

for i in range(num_units):
    st.subheader(f"Unit {i + 1}")
    unit_type = st.selectbox(f"Unit type for Unit {i + 1}", valid_unit_types, key=f"type_{i}")
    square_feet = st.number_input(f"Square footage for Unit {i + 1}", min_value=1, max_value=5000, key=f"sf_{i}")
    unit_square_feet.append(square_feet)
    est_bedrooms = max(1, min(round((square_feet * 0.28) / 200), 5))
    st.text(f"This unit likely has approximately {est_bedrooms} bedrooms based on square footage.")

    cost_row = cost_df[cost_df['unit_type'] == unit_type]
    if not cost_row.empty:
        cost_per_sf = cost_row['cost_per_sf'].values[0]
        unit_labels.append(f"{int(square_feet)}sf {unit_type}")
        development_costs.append(cost_per_sf * square_feet)

# Estimate bedrooms
# These assumptions can be changed as needed. Currently, they assume 200sf on average and that they take up 28% of total sf
if unit_square_feet:
    avg_sf = sum(unit_square_feet) / len(unit_square_feet)
    bedrooms = max(1, min(round((avg_sf * 0.28) / 200), 5))
    st.text(f"\nAssuming an average of {bedrooms} bedrooms per unit based on total square footage.\n")
else:
    bedrooms = 1  # Default fallback
    st.warning("No valid unit square footage provided. Please enter valid unit data.")

# AMI levels and regions
valid_regions = list(data_files.keys())
selected_regions = st.multiselect("Which locations are you interested in with regard to AMI?", valid_regions, default=[])

valid_ami_values = [30] + list(range(50, 155, 5))
num_amis = st.slider("How many AMI values would you like to show on the chart?", 1, 3, 1)
amis = []
for i in range(num_amis):
    ami = st.selectbox(f"Select AMI value #{i + 1}", valid_ami_values, key=f"ami_{i}")
    amis.append(ami)

# Affordability thresholds
affordability_lines = {}
col_name = f"buy{bedrooms}"
for region in selected_regions:
    df = data_files[region]
    df.columns = df.columns.str.strip().str.lower()
    df['ami'] = pd.to_numeric(df['ami'], errors='coerce')
    for ami in amis:
        row = df[df['ami'] == ami / 100]
        if row.empty:
            st.warning(f"AMI {ami}% not found in {region} dataset.")
            continue
        if col_name not in row:
            st.warning(f"Column '{col_name}' not found in {region} dataset.")
            continue
        affordability_lines[f"{ami}% AMI - {region}"] = float(row[col_name].values[0])

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

if unit_labels and development_costs:
    bars = ax1.bar(unit_labels, development_costs, color='skyblue', edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 5000, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9)

colors = ['red', 'green', 'orange', 'purple', 'brown', 'blue', 'gray', 'darkgreen', 'darkred']
for i, (label, value) in enumerate(affordability_lines.items()):
    ax1.axhline(y=value, linestyle='--', color=colors[i % len(colors)], label=label)

ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
ax1.set_ylabel("Cost ($)", fontsize=12)
plt.xticks(rotation=20)
plt.title("Development Cost vs. Affordable Purchase Price Thresholds")
if affordability_lines:
    ax1.legend()

# Secondary Y-axis for % AMI
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(list(affordability_lines.values()))
ax2.set_yticklabels([k.split()[0] for k in affordability_lines.keys()])
ax2.set_ylabel("% AMI")

st.pyplot(fig)
