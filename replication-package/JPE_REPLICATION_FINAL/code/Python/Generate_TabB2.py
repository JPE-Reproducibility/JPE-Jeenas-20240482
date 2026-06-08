# Script to create the Latex code for Table B.2

from pathlib import Path
import re

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()


data_file = PROJECT_ROOT / "output" / "tables" / "_partof_TabB2_nums_untargmoms.txt"
model_file = PROJECT_ROOT / "output" / "tables" / "_partof_TabB2_nums_untargmoms_model.txt"
out_file = PROJECT_ROOT / "output" / "tables" / "TabB2_body.tex"

row_labels = [
    r"$\sigma(\ell)$",
    r"$\text{cor}(\ell, \log(\text{size}))$",
    r"$\text{cor}(\ell, (i/k)_{+4})$",
    r"$\mathbb{E}[\textrm{lev} | \text{high } \ell]$",
    r"$\mathbb{E}[\textrm{lev} | \text{low } \ell]$",
    r"$\text{freq} (\ell = 0)$",
    r"$\text{freq} (\ell < 0.001)$",
    r"$\text{skew} (\log(k))$",
    r"$\text{freq}(D=1 | \text{small})$",
    r"$\text{freq}(D=1 | \text{large})$",
    r"$\text{freq}(\mathrm{div}_{a}>0)$",
]

# Read data values from the R-style matrix printout
data_vals = []
pattern = re.compile(r"\[\s*\d+\s*,\]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data_vals.append(float(match.group(1)))

# Read model values
model_vals = []
with open(model_file, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if stripped:
            model_vals.append(float(stripped))

if len(data_vals) != len(row_labels):
    raise ValueError(f"Expected {len(row_labels)} data values, found {len(data_vals)}.")

if len(model_vals) != len(row_labels):
    raise ValueError(f"Expected {len(row_labels)} model values, found {len(model_vals)}.")

# Write LaTeX rows
with open(out_file, "w", encoding="utf-8") as f:
    for i, (label, dval, mval) in enumerate(zip(row_labels, data_vals, model_vals)):
        row = f"{label} & {dval:.3f} & {mval:.3f}"
        if i < len(row_labels) - 1:
            row += r" \\"
        f.write(row + "\n")

