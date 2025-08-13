# 🏠 Housing Affordability Visualizer

This tool compares **policy-driven differences in total development cost (TDC)** for a single housing unit type and bedroom count against **AMI-based affordability thresholds** across Vermont regions.  
It helps policymakers and planners see how **changes in policy impact housing affordability**.

---

## 🖱️ User Inputs
- **Housing product type**: Townhome, Condo, Apartment  
- **Number of bedrooms** *(for Townhome and Condo only — Apartments use a different model coming soon)*  
- **Number of units to compare** (1–5)  
- **For each unit:**
  - Energy code standard  
  - Energy source  
  - Infrastructure requirement  
  - Finish quality  
- **Vermont region(s)**: Chittenden, Addison, Rest of Vermont  
- **AMI level(s)**: Choose up to 3

---

## ⚙️ What It Does
- Calculates **TDC** for each unit based on baseline costs and selected policy options  
- Retrieves **affordable purchase price thresholds** using AMI data for the selected regions  
- Generates a **side‑by‑side visual comparison** of costs vs. affordability  
- **Auto‑labels baseline scenarios** as `Baseline {Unit Type}` when default assumptions are met

---

## 📊 Output
- **Bars** = Total development cost for each scenario  
- **Dashed lines** = Affordability thresholds for selected AMI levels and regions  
- **Dual Y‑axis**: Left = TDC, Right = Affordability thresholds

---

## 🚀 Try it Now
- **Live app:** https://housing-affordability-tool.streamlit.app/  
- **View all assumptions & code:** https://github.com/alexbleich/housing-affordability-tool

---

## 📁 Files in This Repo
- `data/assumptions.csv` — Policy cost assumptions (energy codes, energy sources, finish quality, infrastructure, etc.)  
- `data/chittenden_ami.csv`, `data/addison_ami.csv`, `data/vermont_ami.csv` — AMI thresholds for Chittenden County, Addison County, and the rest of Vermont  
- `housing-affordability-tool.py` — Main Streamlit app

---

## 🛠️ Run Locally
```bash
# Clone
git clone https://github.com/alexbleich/housing-affordability-tool.git
cd housing-affordability-tool

# Install deps
pip install -r requirements.txt

# Launch
streamlit run housing-affordability-tool.py
