## Potential Personal Identifiable Information (PII)

⚠️ We found the following instances of potentially personally identifying information. This may be completely legitimate but might be worth checking. *As a reminder, privacy legislation in many countries (e.g. GDPR in EU) prohibits the dissemination of personal identifiable information without prior (and documented) consent of individuals.* If indeed you want to publish such information with your replication package, you should probably have obtained IRB approval for this - please check!

**Summary:**
- Data files with PII indicators: 5
- Variables flagged in data: 106
- Code files with PII references: 23
- PII references in code: 847

### Summary of Flagged Files

| File Type | File | Variables/References | PII Categories |
|-----------|------|----------------------|----------------|
| Data | `FRB_Z1.csv` | 98 | lon, house, son |
| Data | `LICENSE.txt` | 1 | son |
| Data | `Wscope_data_raw.xlsx` | 1 | name |
| Data | `bds2018.csv` | 3 | birth, loc, location |
| Data | `bds2018_fage.csv` | 3 | birth, loc, location |
| Code | `BDS_analysis_code.jl` | 1 | lat |
| Code | `CRSP_cleaning_code.py` | 3 | name, loc |
| Code | `CRSP_par_prep.py` | 3 | lat, name, loc |
| Code | `CS_cleaning_code.py` | 1 | name |
| Code | `CStoCRSP_merge.py` | 28 | name, lat, loc |
| Code | `DD_work_par_localwrapper.py` | 3 | loc, name |
| Code | `DD_work_run.py` | 35 | loc, name, lat |
| Code | `FOFA_analysis_code.py` | 10 | name, loc, lat |
| Code | `Generate_TabB2.py` | 4 | lat, name, zip |
| Code | `LPAgg_main.m` | 50 | loc, name, lat, son, location |
| Code | `Master_empirics_work.py` | 5 | name, lat, loc |
| Code | `Master_main.py` | 2 | name, loc, lon |
| Code | `Master_model_work.jl` | 135 | block, loc, lat, lon, house, city, name, lname, second, son |
| Code | `Micro_eventstudy_plottr.r` | 7 | name, son, lon, second |
| Code | `Micro_model_reg_code.r` | 10 | name, lat, loc, lname |
| Code | `Micro_reg_code.r` | 175 | name, lat, lon, loc, lname, son, city |
| Code | `Micro_reg_prep_code.py` | 30 | name, lat, loc, zip, lon |
| Code | `TB3_analysis_code.py` | 4 | name, loc, lat |
| Code | `Wscope_compile_code.py` | 16 | name, loc |
| Code | `auxiliary_model_functions.jl` | 288 | name, lat, house, city, lon, second, son, block, loc |
| Code | `calib_functions.jl` | 20 | lat, loc, name, birth, second, lon, block |
| Code | `install_Julia_packages.jl` | 13 | lat, name |
| Code | `printpdf.m` | 4 | name |

*See [Appendix](report-pii-appendix.md) for detailed listing of all flagged instances.*
