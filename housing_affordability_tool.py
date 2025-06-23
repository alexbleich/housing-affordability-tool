"""
This program visualizes housing development costs alongside affordable purchase price thresholds
based on Area Median Income (AMI) levels. Users can compare the cost of developing different unit
types (e.g., apartments, townhomes, condos) across various square footages. The tool estimates
bedroom count from square footage and overlays AMI-based affordability lines from multiple Vermont
regions (Chittenden, Addison, Vermont) to assess how feasible unit costs are for households at
different income levels.

Author: Alex Bleich
Date: June 23rd, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# CONFIGURATION
data_files = {
    "Chittenden": pd.read_csv("chittenden_ami.csv"),
    "Addison": pd.read_csv("addison_ami.csv"),
    "Vermont": pd.read_csv("vermont_ami.csv")
}
cost_df = pd.read_csv("construction_costs.csv")


# Helper functions
def get_valid_int(prompt, min_val, max_val):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a valid integer.")


# Step 1: Unit comparisons
num_units = get_valid_int("How many units would you like to compare? ", 1, 10)
unit_labels, development_costs, valid_unit_types = [], [], ['apartment', 'townhome', 'condo']

for i in range(num_units):
    print(f"\nFor Unit {i + 1}:")
    while (unit_type := input("  Unit type (apartment, townhome, condo): ").strip().lower()) not in valid_unit_types:
        print("  Invalid input. Please enter: apartment, townhome, or condo.")

    square_feet = get_valid_int("  Square footage: ", 1, 10000)
    est_bedrooms = max(1, min(round((square_feet * 0.28) / 200), 5))
    print(f"  Assuming this unit has approximately {est_bedrooms} bedrooms based on square footage.")

    cost_row = cost_df[cost_df['unit_type'] == unit_type]
    if not cost_row.empty:
        cost_per_sf = cost_row['cost_per_sf'].values[0]
        unit_labels.append(f"{int(square_feet)}sf {unit_type}")
        development_costs.append(cost_per_sf * square_feet)

# Step 2: Estimate bedrooms
# Can change these assumptions as needed. Currently, assumes bedrooms are 200 sf and in total take up 28% of a unit
avg_sf = sum([float(label.split('sf')[0]) for label in unit_labels]) / len(unit_labels)
bedrooms = max(1, min(round((avg_sf * 0.28) / 200), 5))
print(f"Assuming an average of {bedrooms} bedrooms per unit based on total square footage.\n")

# Step 3: AMI levels and regions
print("Which locations are you interested in with regard to AMI? (Chittenden, Addison, Vermont)")
print("You can enter multiple locations separated by commas.")
valid_regions = list(data_files.keys())
while True:
    user_input = input("Enter locations: ").strip().lower().replace(" ", "")
    selected_regions = [loc.capitalize() for loc in user_input.split(',') if loc.capitalize() in valid_regions]
    if selected_regions: break
    print("  Please enter at least one valid region.")

valid_ami_values = [30] + list(range(50, 155, 5))
amis, num_amis = [], get_valid_int("\nHow many AMI values would you like to show on the chart? ", 1, 10)
for i in range(num_amis):
    while True:
        try:
            ami = int(input(f"  Enter AMI value #{i + 1} (30, 50–150 by 5s): "))
            if ami in valid_ami_values:
                amis.append(ami)
                break
            print("  Invalid AMI. Choose 30 or a multiple of 5 from 50 to 150.")
        except ValueError:
            print("  Please enter a valid integer.")

# Step 4: Affordability thresholds
affordability_lines = {}
col_name = f"buy{bedrooms}"
for region in selected_regions:
    df = data_files[region]
    df.columns = df.columns.str.strip().str.lower()
    df['ami'] = pd.to_numeric(df['ami'], errors='coerce')
    for ami in amis:
        row = df[df['ami'] == ami / 100]
        if row.empty:
            print(f"Warning: AMI {ami}% not found in {region} dataset.")
            continue
        if col_name not in row:
            print(f"Warning: Column '{col_name}' not found in {region} dataset.")
            continue
        affordability_lines[f"{ami}% AMI - {region}"] = float(row[col_name].values[0])

# Step 5: Plot
plt.figure(figsize=(12, 6))
ax1 = plt.gca()

bars = ax1.bar(unit_labels, development_costs, color='skyblue', edgecolor='black')
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 5000, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9)

colors = ['green', 'green', 'red', 'red', 'orange', 'orange']
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
ax2.set_yticklabels([f"{k.split()[0]}\n${affordability_lines[k]:,.0f}" for k in affordability_lines.keys()])
ax2.set_ylabel("% AMI")

plt.tight_layout()
plt.show()
