# 🏘️ Housing Affordability Visualizer
This Streamlit app compares policy-driven differences in total development cost (TDC) for a single housing unit type and bedroom count against AMI-based affordability thresholds across Vermont regions.

Live app: https://housing-affordability-tool.streamlit.app/

### 📌 What the Program Does
The tool allows users to see how different policy choices (energy code, energy source, infrastructure requirements, and finish quality) affect the total development cost (TDC) for a unit.
It overlays affordability thresholds based on Area Median Income (AMI) for different Vermont regions. The goal is to help visualize how policy directly impacts affordability.

### 🖥️ User Inputs
Housing product type (Townhome, Condo, Apartment)
Number of bedrooms (for Townhome and Condo only — Apartments use a different model coming soon)
Number of units to compare (1–5)
For each unit:
  Energy code standard
  Energy source
  Infrastructure requirement
  Finish quality
Vermont region(s) (Chittenden, Addison, Rest of Vermont)
AMI level(s) (choose up to 3)

### 📊 Output
The app generates a side-by-side bar chart showing:
Bars: Total development cost (TDC) for each selected unit scenario
Dashed horizontal lines: AMI-based affordability thresholds for the selected region(s) and AMI level(s)
Dual Y-axis:
  Left axis: Development cost ($)
  Right axis: Corresponding affordability value for each AMI line
If a unit selection matches the baseline scenario (2 bedrooms, VT Energy Code, Natural Gas, No Infrastructure, Average Finish), its bar will be labeled Baseline {UnitType}.

### 📂 Files & Data Structure
All data files are stored in the data/ folder in the repo:
File	Purpose
assumptions.csv	Policy cost assumptions (energy codes, sources, finish, infrastructure, etc.)
chittenden_ami.csv	AMI thresholds for Chittenden County
addison_ami.csv	AMI thresholds for Addison County
vermont_ami.csv	AMI thresholds for the rest of Vermont

## Main script:

housing-affordability-tool.py — The Streamlit app. Handles all UI elements, calculations, and chart generation.

📦 How to Run Locally
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run housing-affordability-tool.py
🔗 Useful Links
Live app: Housing Affordability Visualizer

GitHub repo: View all assumptions and code here
