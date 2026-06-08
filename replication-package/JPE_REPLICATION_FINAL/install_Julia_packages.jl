# Script that installs the Julia packages required

using Pkg

Pkg.add(name="LaTeXStrings", version="1.3.1")
Pkg.add(name="Interpolations", version="0.15.1")
Pkg.add(name="Roots", version="2.2.1")
Pkg.add(name="QuantEcon", version="0.16.6")
Pkg.add(name="CompEcon", version="0.4.0")
Pkg.add(name="JLD", version="0.13.5")
Pkg.add(name="CSV", version="0.10.14")
Pkg.add(name="Distributions", version="0.25.45")
Pkg.add(name="DataFrames", version="1.5.0")
Pkg.add(name="GLM", version="1.9.0")
Pkg.add(name="Binscatters", version="0.4.0")
Pkg.add(name="Plots", version="1.40.8")
Pkg.add(name="AverageShiftedHistograms", version="0.8.9")