# Code for computing calibration target with BDS age group data
using DataFrames
using CSV

# Define the root directory, robust to interactive or file execution
PROJECT_ROOT = try
    normpath(joinpath(@__DIR__, "..", "..", ".."))
catch
    pwd()
end

cd(PROJECT_ROOT)

# First, working only with economy-wide annual data, compute average firm size
bds_data  = DataFrame(CSV.File(joinpath(PROJECT_ROOT, "data", "raw", "bds2018.csv")));
bds_data.avg_emp = bds_data.emp ./ bds_data.firms;

fage_data = DataFrame(CSV.File(joinpath(PROJECT_ROOT, "data", "raw", "bds2018_fage.csv")));
# Drop missing row
fage_data = fage_data[fage_data.emp.!="(S)",:];

# Convert to numeric
fage_data.emp    = parse.(Float64, fage_data.emp);
fage_data.firms  = parse.(Float64, fage_data.firms);

# Compute cell's average firm employment
fage_data.avg_emp   = fage_data.emp ./ fage_data.firms;

# Back out entrants' data only
ent_data            = fage_data[fage_data.fage .== "a) 0", :];
ent_data.rel_emp    = ent_data.avg_emp ./ bds_data.avg_emp;

# Calibration target:
start_year = 1990;
end_year   = 2008;
# By firm size
rel_ent_size = sum(ent_data[(ent_data.year.>=start_year) .& (ent_data.year.<=end_year), :rel_emp]) / length(ent_data[(ent_data.year.>=start_year) .& (ent_data.year.<=end_year), :rel_emp]);
# Ensure output folder exists, just in case
mkpath(joinpath(PROJECT_ROOT, "output", "tables"))
write(joinpath(PROJECT_ROOT, "output", "tables", "_partof_TabB1_BDS_calibtarget.txt"), "Relative size of entrants in BDS: $(round(rel_ent_size, digits=2))\n");
