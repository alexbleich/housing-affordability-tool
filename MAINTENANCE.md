# 🛠️ Housing Affordability Tool — Maintenance Guide

This guide is for people who are **not programmers** but want to make small changes to the Housing Affordability Visualizer.  
The app runs on [Streamlit Cloud](https://streamlit.io/cloud) and updates automatically when changes are made in GitHub.

---

## How the Tool Works
- **Code:** The main logic is in `housing-affordability-tool.py`.  
- **Assumptions:** Policy costs are stored in `data/assumptions.csv`.  
- **Affordability thresholds:** Income & purchase price tables are in `data/chittenden_ami.csv`, `data/addison_ami.csv`, and `data/vermont_ami.csv`.  
- **Streamlit Cloud:** The app is redeployed automatically when you save changes on GitHub.

---

## The Easiest Way to Make Changes
1. Open the file on GitHub.  
2. Click the ✏️ **pencil icon** to edit.  
3. Change what you need to.
4. Scroll down, add a short commit message (like “updated finish quality costs”), and press **Commit changes**.
5. Refresh the Streamlit app to see your change.

---

## What You Can Safely Change
- **Numbers in CSV files:**  
  - Example: Change the cost per square foot of an energy source in `assumptions.csv`.  
  - Example: Update income levels in the AMI CSVs when new VHFA data is released.  
- **Labels in the Python file:**  
  - Search for the text you want to change (e.g., “Keep trying” message).  
- **Packages:** The top of the Python file has a section called `PKG`. You can rename or add new “packages” (Baseline, Top-of-the-Line, Below Baseline).  

---

## Using ChatGPT to Help
If you need to change the Python file:
1. Open a new chat and use this prompt to get it to understand the context:
   - "I’m working on a Streamlit app written in Python. The app is called Housing Affordability Visualizer. It calculates housing development costs from a CSV of assumptions and compares them to AMI affordability thresholds from VHFA tables. The code is all in one file called housing-affordability-tool.py. I don’t know Python, but I need to make small changes. I’ll paste parts of the code or CSV here, and you should show me the exact edits I need to make. My end goal is to [state your goal clearly — e.g., “add a new finish quality option,” “change the failure message,” “update the AMI values”]. Please give me concrete code or CSV rows I can copy-paste, and explain what you changed."
   - You can change this prompt as you see fit, but it is crucial to do this so it can jump right into helping with exactly what you need.
2. Copy the part of the code you want to change (you can copy the whole script if you'd like).  
3. Paste it into ChatGPT and ask a clear question or tell it specifically what you want.
   - Example: “How do I add a new finish quality option called ‘Luxury’ with a 20% adder?”  
   - Example: “I need the error message to display something different. Change it.”  
4. If you see an error in the Streamlit app, copy the error message and paste it into ChatGPT. It will fix it for you.

---

## Safety Tips
- **Make small edits.** If something breaks, it’s easier to undo.  
- **Use commit messages** like “update AMI data” so you know what was changed.  
- **Don’t delete big sections** unless you are sure what they do.  
- **Remember:** GitHub saves all past versions. You can always roll back if need be, and the commit messages help you find where you want to get back to.

---

## When to Ask for Help
- If you want to add **new inputs** (e.g., a new policy option) or **change how the math works**, copy the relevant code section into ChatGPT and describe what you want.  
- If you only need to **update numbers** (e.g., assumptions, AMI tables), edit the CSVs directly within the `data` folder.
