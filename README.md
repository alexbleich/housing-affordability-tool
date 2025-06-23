# Housing Affordability Tool

This tool helps visualize and compare the cost of building housing units (e.g., apartments, townhomes, condos) with the price that households can afford based on their income and location.

### 🛠️ User Inputs
- Unit type and square footage (# of bedrooms estimated from sf)
- Target income levels (% of Area Median Income, or AMI)
- Vermont region (Chittenden, Addison, and/or statewide)

### 📈 What It Does
- Estimates development cost based on unit type and square footage
- Retrieves affordable purchase price thresholds using AMI data
- Generates a visual comparison to illustrate financial feasibility

### 📊 Output
The app displays a chart that overlays estimated development costs with income-based affordability thresholds for the selected regions.

## 🔗 Try it Online
The tool is accessible here: https://housing-affordability-tool.streamlit.app/

## 📁 Files
- `chittenden_ami.csv`, `addison_ami.csv`, `vermont_ami.csv`: [VHFA income & affordability data](https://housingdata.org/documents/Purchase-price-and-rent-affordability-expanded.pdf) by region
- `construction_costs.csv`: Estimated development cost per square foot by unit type
- `housing_affordability_tool.py`: Python script that powers the interactive tool
