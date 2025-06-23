# Housing Affordability Tool

This tool helps visualize and compare the cost of building housing units 
(e.g., apartments, townhomes, condos) with the price households can afford 
based on their income and location.

Users input:
- Unit type and square footage
- Household size
- Target income levels (% of Area Median Income (AMI))
- Vermont region (Chittenden, Addison, and/or statewide)

The tool then estimates:
- Development cost based on square footage and unit type
- Affordable purchase price thresholds using AMI data
- A visual comparison to assess affordability

## 📊 Output
The app generates a chart that overlays development costs with AMI-based affordability thresholds for selected regions.

## 🔗 Try it Online
Once deployed, the tool will be accessible here:  
👉 [Streamlit App](https://housing-affordability-tool.streamlit.app/)

## 📁 Files
- `chittenden_ami.csv`, `addison_ami.csv`, `vermont_ami.csv`: VHFA income & affordability data by region
- `construction_costs.csv`: Estimated development cost per square foot by unit type
- `housing_affordability_tool.py`: Python script that powers the interactive tool
