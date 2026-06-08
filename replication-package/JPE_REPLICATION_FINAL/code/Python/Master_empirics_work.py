# Master Python script for orchestrating the core empirical work in the replication package of "Firm Balance Sheet Liquidity, Monetary Policy Shocks, and Investment Dynamics" by Priit Jeenas
import subprocess, sys
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Define function for easier running of sub-programs
# In Python
def run_python(script_path):
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=PROJECT_ROOT
    )
# In R
def run_r(script_path):
    subprocess.run(
        ["Rscript", str(script_path)],
        check=True,
        cwd=PROJECT_ROOT
    )
# In Julia
def run_julia(script_path):
    subprocess.run(
        ["julia", str(script_path)],
        check=True,
        cwd=PROJECT_ROOT
    )

###
# Empirical analysis
###

# Clean the raw Compustat data, yields 'CS_data_clean.csv'
run_python(PROJECT_ROOT / "code" / "Python" / "cleaning" / "CS_cleaning_code.py")

###
# Work with CRSP to construct distance-to-default measure as control in robustness
###
# Clean raw CRSP and split up into 10 subpieces to manage RAM overload later on
run_python(PROJECT_ROOT / "code" / "Python" / "cleaning" / "CRSP_cleaning_code.py")
# Merge Compustat balance sheet data to CRSP
run_python(PROJECT_ROOT / "code" / "Python" / "cleaning" / "CStoCRSP_merge.py")
# Split the generated data further into 60 subpieces (to make it easier to parallelize, either locally or in High-Performance-Computing cluster)
run_python(PROJECT_ROOT / "code" / "Python" / "cleaning" / "CRSP_par_prep.py")
# Run DD calculations
run_python(PROJECT_ROOT / "code" / "Python" / "DD_work_par_localwrapper.py")  # (Loops locally over DD_work_run.py, with slurmcounter=0--59)

# Compile and organize Worldscope data on firms' incorporation dates
run_python(PROJECT_ROOT / "code" / "Python" / "cleaning" / "Wscope_compile_code.py")

# Python analysis, compiling relevant data sources (DD, incorporation dates, some aggregate series and the Jarocinski-Karadi ffr shocks dataset) and prior cleaning of Compustat panel
run_python(PROJECT_ROOT / "code" / "Python" / "Micro_reg_prep_code.py")

# Core empirical analysis implemented in R on Compustat data
run_r(PROJECT_ROOT / "code" / "R" / "Micro_reg_code.r")

# Work on Federal Reserve Flow of Funds Accounts for firms' aggregate liquidity portfolio (Table A.5)
run_python(PROJECT_ROOT / "code" / "Python" / "FOFA_analysis_code.py")

# Work on BDS for entrants' size calibration target (into Table B.1)
run_julia(PROJECT_ROOT / "code" / "Julia" / "Julia_empirics_code" / "BDS_analysis_code.jl")

# Work on generating counterfactual economy targets for structural model (Appendix B.10.1 text)
run_python(PROJECT_ROOT / "code" / "Python" / "TB3_analysis_code.py")
