# Master Python script for the replication package of "Firm Balance Sheet Liquidity, Monetary Policy Shocks, and Investment Dynamics" by Priit Jeenas
import subprocess, sys, os
from pathlib import Path

# Set working directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

os.chdir(PROJECT_ROOT)

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
# In MATLAB
def run_matlab(script_path):
    matlab_bin = "/Applications/MATLAB_R2020a.app/bin/matlab"
    subprocess.run(
        [matlab_bin, "-batch", f"run('{script_path}')"],
        check=True,
        cwd=PROJECT_ROOT
    )


### 
# Empirical work that is run preceding to model computations
###
run_python(PROJECT_ROOT / "code" / "Python" / "Master_empirics_work.py")

###
# Structural model solution and results
###
run_julia(PROJECT_ROOT / "code" / "Julia" / "Julia_model_code" / "Master_model_work.jl")

###
# Run the remaining analysis that requires model output
###
# Run debt issuance regressions and event studies in model-generated data
run_r(PROJECT_ROOT / "code" / "R" / "Micro_model_reg_code.r")
# Combine empirical and model event-study estimates into Figures A.2-A.4
run_r(PROJECT_ROOT / "code" / "R" / "Micro_eventstudy_plottr.r")
# Combine empirical and model untargeted moments into the body of Table B.2
run_python(PROJECT_ROOT / "code" / "Python" / "Generate_TabB2.py")

# Estimate empirical aggregate IRFs with local projections, and plot alongside model IRFs (Appendix A.6)
run_matlab(str(PROJECT_ROOT / "code" / "MATLAB" / "LPAgg_main.m"))

# End of replication package