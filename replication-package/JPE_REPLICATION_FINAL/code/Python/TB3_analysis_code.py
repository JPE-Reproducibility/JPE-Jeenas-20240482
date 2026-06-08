# Script for working on 3-month T-Bill rate data from FRED

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

####### LOADING DATA #########
data = pd.read_csv(PROJECT_ROOT / 'data/raw/TB3MS.csv')
# Setting up date variable
data["observation_date"] = pd.to_datetime(data["observation_date"])
data["year"] = data["observation_date"].dt.year

# Basic analysis
tb3_8589_avg = data.loc[(data["year"] >= 1985) & (data["year"] <= 1989), "TB3MS"].mean()
tb3_9008_avg = data.loc[(data["year"] >= 1990) & (data["year"] <= 2008), "TB3MS"].mean()
pre_to_post_diffd2 = (tb3_8589_avg-tb3_9008_avg)/2
model_base_pi       = 2.0
counterf_pi         = model_base_pi + pre_to_post_diffd2

# Make sure output folder exists, just in case
other_dir = PROJECT_ROOT / "output" / "other"
other_dir.mkdir(parents=True, exist_ok=True)
with open(other_dir / "Text_AppxB10p1_rates.txt", "w") as f:
    f.write(
        f"3-month T-Bill rate 1985--1989: {tb3_8589_avg:.2f}\n"
        f"3-month T-Bill rate 1990--2008: {tb3_9008_avg:.2f}\n"
        f"Counterfactual model inflation rate: {counterf_pi:.2f}\n"
    )



