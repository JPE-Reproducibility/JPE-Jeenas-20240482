# R script for running the main firm-level empirical analysis on compiled Compustat data

# Define the root directory, robust to interactive or file execution
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path_arg <- args[grep(file_arg, args)]

if (length(script_path_arg) > 0) {
  script_path <- normalizePath(sub(file_arg, "", script_path_arg), winslash = "/")
  PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")
} else {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/")
}
# Set ROOT as working directory
setwd(PROJECT_ROOT)

# Initialize
library(plm)
library(plyr)
library(lmtest)
library(stargazer)
Sys.setenv(LANG = "en")

# Define lag function that behaves correctly and consistently with earlier versions of R
rlag <- function(x, k=1, shift="row", ...) {
  lag(x, k, shift, ...)
}

options(width=120)
# Parameters for figures, overwritten below for LPs
linew_main = 2.2; linew_ci = 1.6; col_change_ind = TRUE
font_ax_val = 1.2; font_leg_val = 1.15; font_lab_val = 1.4

# Parameters
# Age data exists?
age_ind = TRUE
# Distance to default data exists?
DD_ind = TRUE

# Reading data
data <- read.csv(file.path(PROJECT_ROOT, "proprietary-data-not-for-publication", "processed", "Micro_reg_data_compiled.csv"))

# Count observations again
data$obsc <- rep(count(data,'gvkey')$freq,count(data,'gvkey')$freq)

# Specialized sic codes
data$sic_str <- as.character(data$sic)
# Add zero to short SICs
data$sic_str[nchar(data$sic_str) < 4] <- paste("0", data$sic_str[nchar(data$sic_str) < 4], sep="")
# Create broader SICs
data$sic1 <- as.integer(substr(data$sic_str,1,1))
data$sic2 <- as.integer(substr(data$sic_str,1,2))
data$sic3 <- as.integer(substr(data$sic_str,1,3))

data$sic <- as.factor(data$sic)
data$sic1 <- as.factor(data$sic1)
data$sic2 <- as.factor(data$sic2)
data$sic3 <- as.factor(data$sic3)

# Paste together the sic and datacqtr
data$sic_datacqtr <- paste(data$sic_str, as.character(data$datacqtr), sep=",")
data$sic_datacqtr <- as.factor(data$sic_datacqtr)
data$sic1_datacqtr <- paste(as.character(data$sic1), as.character(data$datacqtr), sep=",")
data$sic1_datacqtr <- as.factor(data$sic1_datacqtr)
data$sic2_datacqtr <- paste(as.character(data$sic2), as.character(data$datacqtr), sep=",")
data$sic2_datacqtr <- as.factor(data$sic2_datacqtr)
data$sic3_datacqtr <- paste(as.character(data$sic3), as.character(data$datacqtr), sep=",")
data$sic3_datacqtr <- as.factor(data$sic3_datacqtr)

###
# NEW VARIABLES
###
# Recompute real assets based on GVA deflator
data$atq_real <- 100 * data$atq / data$GVADef

# Take log-size
data$latq <- log(data$atq)
data$latq_real <- log(data$atq_real)
# Net leverage
data$Nlev_rat <- (data$dttq-data$cheq)/data$atq
# Short and long leverage
data$Slev_rat <- (data$dlcq)/data$atq
data$Llev_rat <- (data$dlttq)/data$atq
# Cash-to-assets
data$cheat_rat <- data$cheq/data$atq
# Cash flow (Ippolito et al & Gilchrist, Himmelberg)
data$cshflq <- data$ibq + data$dpq
# Log capital
data$lk_stock <- log(data$k_stock)
# Log sales
data$saleq_real <- 100 * data$saleq / data$GVADef
data$lsaleq      <- log(data$saleq)
data$lsaleq_real <- log(data$saleq_real)

# Create panel dataframe only to compute certain lagged variables, in order to apply the rlag operator correctly
p_data <- pdata.frame(data, index=c("gvkey", "cqtr_num"))
# Quarterly dividend payments
p_data$dvq <- p_data$dvy - rlag(p_data$dvy, 1)
p_data[p_data$fqtr==1, 'dvq'] <- p_data[p_data$fqtr==1, 'dvy']
# Equity issuances
# Compute gross issuance activity
p_data$eqissq <- p_data$sstky - rlag(p_data$sstky,1)
p_data[p_data$fqtr==1, 'eqissq']  <- p_data[p_data$fqtr==1, 'sstky']
# Relative to lagged assets
p_data$eqat_rat <- (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$eqissq / rlag(p_data$atq,1) # Issuance/lagged assets ratio
# Impose in "data"
data$eqissq         <- p_data$eqissq
data$eqat_rat       <- p_data$eqat_rat

data$dvq <- p_data$dvq
# Indicator of previous year positive dividends
p_data$dvq_pos <- as.factor(rowMeans(rlag(p_data$dvq,0:3),na.rm=TRUE)>0)
p_data[p_data$dvy > 0 & !is.na(p_data$dvy),'dvq_pos'] <- TRUE
p_data[p_data$dvy == 0 & !is.na(p_data$dvy) & p_data$fqtr == 4,'dvq_pos'] <- FALSE
data$dvq_pos <- as.integer(p_data$dvq_pos)

# Compute yearly log capital growth (already real "lk_stock") and sales growth
p_data$dlk_stock  <- p_data$lk_stock - rlag(p_data$lk_stock,1)
p_data$dylk_stock <- p_data$lk_stock - rlag(p_data$lk_stock,4)
p_data$dylsaleq_real <- p_data$lsaleq_real - rlag(p_data$lsaleq_real,4)
data$dlk_stock <- p_data$dlk_stock
data$dylk_stock <- p_data$dylk_stock
data$dylsaleq_real <- p_data$dylsaleq_real
# Compute real kinv_rat
p_data$kinv_rat_real    <- p_data$capxq_real/rlag(p_data$k_stock, 1)
data$kinv_rat_real      <- p_data$kinv_rat_real

# Reorder data
data <- data[order(data$gvkey, data$cqtr_num), ]

# Compute age
if (age_ind){ data['age'] = data['cqtr_num'] - data['incdate_num'] }

# SPLIT INTO QUANTILES -- create indicators
c_q = c(0.0,0.33,0.66,1.0)
n_q = length(c_q)-1
q_split_fn <- function(x){
    brks_temp   <- quantile(x, probs=c_q, na.rm=TRUE)
    res_temp    <- cut(x, breaks=brks_temp, labels=1:n_q, include.lowest=TRUE, right=FALSE)
}
data <- ddply(data, c("cqtr_num"), transform, q_lev_rat = q_split_fn(lev_rat))
data <- ddply(data, c("cqtr_num"), transform, q_Nlev_rat = q_split_fn(Nlev_rat))
data <- ddply(data, c("cqtr_num"), transform, q_cheat_rat = q_split_fn(cheat_rat))

data$q_lev_rat <- as.factor(data$q_lev_rat)
data$q_Nlev_rat <- as.factor(data$q_Nlev_rat)
data$q_cheat_rat <- as.factor(data$q_cheat_rat)

# Instead split into groups of equal sums of k_stock
if (TRUE){
    n_q = length(c_q)-1
    aq_fn <- function(x){
        res_temp <- cumsum(x)/sum(x)
    }
    # Order data by year by lev_rat
    data<-data[order(data$cqtr_num, data$lev_rat),]
    # Compute cumulative sums of "k_stock"
    data <- ddply(data, c("cqtr_num"), transform, cs_k_stock = aq_fn(k_stock))
    data$qcs_lev_rat <- cut(data$cs_k_stock, breaks=c_q, labels=1:n_q, include.lowest=TRUE, right=FALSE)
    # Order data by year by cheat_rat
    data<-data[order(data$cqtr_num, data$cheat_rat),]
    # Compute cumulative sums of "k_stock"
    data <- ddply(data, c("cqtr_num"), transform, cs_k_stock = aq_fn(k_stock))
    data$qcs_cheat_rat <- cut(data$cs_k_stock, breaks=c_q, labels=1:n_q, include.lowest=TRUE, right=FALSE)
    # Reorder data
    data <- data[order(data$gvkey, data$cqtr_num), ]
    data$qcs_lev_rat <- as.factor(data$qcs_lev_rat)
    data$qcs_cheat_rat <- as.factor(data$qcs_cheat_rat)
}

# RECREATE panel dataframe
p_data <- pdata.frame(data, index=c("gvkey", "cqtr_num"))
p_data$cqtr_num <- as.double(as.character(p_data$cqtr_num))
p_data$datacqtr <- as.factor(p_data$datacqtr)

# Also create and save the main measure of gross debt issuance variable used throughout
# LT debt issuance
p_data['dltisq']     = p_data$dltisy - rlag(p_data$dltisy,1)
p_data[p_data$fqtr==1, 'dltisq'] = p_data[p_data$fqtr==1, 'dltisy']
# In addition impose that whenever the "annual measure" is zero, the quarterly measure must be as well
p_data[(p_data$dltisy==0) & !(is.na(p_data$dltisy)) & (is.na(p_data$dltisq)), 'dltisq'] = 0
# Ratios
p_data['dltisqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$dltisq/rlag(p_data$atq,1)
p_data['diss_ind'] <- as.double(p_data$dltisqat_rat > 0.01)

# Backup p_data to conduct issuance analysis on separate branch than LP below
saveRDS(p_data, file = file.path(PROJECT_ROOT, "proprietary-data-not-for-publication", "processed", "p_data_saved.rds"))
options(warn=-1)

# Clean lines for calibration moments in the draft, on full sample
if (TRUE){
    ###
    # Calibration targets 
    ###
    # Compute AGGREGATE ratios
    # Introduce flexible beginning and end
    temp_calib_tstart = 1990.0
    temp_calib_tend   = 2008.0
    # Leverage
    aggdebt_t <- aggregate(dttq ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) sum(x))
    aggatq_t <- aggregate(atq ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) sum(x))
    agglev_t  <- aggdebt_t$dttq/aggatq_t$atq
    # Cash-to-assets
    aggcheq_t <- aggregate(cheq ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) sum(x))
    aggcheat_t  <- aggcheq_t$cheq/aggatq_t$atq
    # Debt issuance "by quarter"
    p_data$dltis_ind <- p_data$dltisqat_rat > 0.01
    aggfreq_t <- aggregate(dltis_ind ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) mean(x))

    # Ensure output/tables folder exists
    dir.create(file.path(PROJECT_ROOT, "output"), showWarnings = FALSE)
    dir.create(file.path(PROJECT_ROOT, "output", "tables"), showWarnings = FALSE)
    capture.output(
    print("### Calibration targets ###"),
    print(paste("Aggregate Debt/Assets: ", round(median(agglev_t), 3))),
    print(paste("Aggregate Cash/Assets: ", round(median(aggcheat_t), 3))),
    print(paste("freq(D=1):             ", round(median(aggfreq_t$dltis_ind), 3))),
    file = file.path(PROJECT_ROOT, "output", "tables", "_partof_TabB1_CS_calibtargets.txt")
    )
}

# Clean lines for returning aggregate cash-to-assets ratio time series, on full sample
if (TRUE){
    # Compute AGGREGATE ratios
    # Introduce flexible beginning and end
    temp_calib_tstart = 1985.0
    temp_calib_tend   = 2008.0
    # ATQ
    aggatq_t <- aggregate(atq ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$cheq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) sum(x))
    # Cash-to-assets
    aggcheq_t <- aggregate(cheq ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$cheq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) sum(x))
    aggcheq_t$cheat  <- aggcheq_t$cheq/aggatq_t$atq
    # Quartiles
    cheat_p75_t <- aggregate(cheat_rat ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) quantile(x, 0.75))
    cheat_p50_t <- aggregate(cheat_rat ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) quantile(x, 0.50))
    cheat_p25_t <- aggregate(cheat_rat ~ cqtr_num, p_data[(as.double(as.character(p_data$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data$cqtr_num))<=temp_calib_tend) & !(is.na(p_data$dttq)) & !(is.na(p_data$atq)) & !(is.na(p_data$cheq)),], function(x) quantile(x, 0.25))
    # Put together
    aggcheq_t$cheat_p75 <- cheat_p75_t$cheat_rat
    aggcheq_t$cheat_p50 <- cheat_p50_t$cheat_rat
    aggcheq_t$cheat_p25 <- cheat_p25_t$cheat_rat
    # Export to csv, alongside cqtr_num
    write.csv(aggcheq_t, file.path(PROJECT_ROOT, "interim_output", "CS_agg_cheat_t.csv"))
}


# Statement about aggregate relevance of long-term debt in Compustat overall (Footnote 3)
if (TRUE){
    # Aggregate ratios
    # ST debt
    aggstdebt_t <- aggregate(dlcq ~ cqtr_num, p_data[!(is.na(p_data$dttq)) & !(is.na(p_data$dlcq)),], function(x) sum(x))
    # Total debt
    aggtotdebt_t <- aggregate(dttq ~ cqtr_num, p_data[!(is.na(p_data$dttq)) & !(is.na(p_data$dlcq)),], function(x) sum(x))

    # Basic share
    aggstdebt_t$stshare  <- aggstdebt_t$dlcq / aggtotdebt_t$dttq
    med_stdebt_share    <- median(aggstdebt_t$stshare)
    dir.create(file.path(PROJECT_ROOT, "output", "other"), showWarnings = FALSE, recursive = TRUE)
    writeLines(paste("At least ", round(100 * (1 - med_stdebt_share), 2), "% of total Compustat debt is long-term.", sep = ""), file.path(PROJECT_ROOT, "output", "other", "Footnote3_justification.txt"))
}


# Clean lines for debt issuance lumpiness analysis in the draft
if (TRUE){

    # Select sample period
    p_data <- p_data[p_data$cqtr_num <= 2008.0,]
    p_data <- p_data[p_data$cqtr_num >= 1990.0,]
    # Only consider firms with more than 40 observations
    if (TRUE){
        p_data$obsc <- rep(count(p_data,'gvkey')$freq,count(p_data,'gvkey')$freq)
        p_data <- p_data[p_data$obsc >= 40,]
    }

    # Also create data not yet created for debt issuances
    # LT debt reduction
    p_data['dltrq']     = p_data$dltry - rlag(p_data$dltry,1)
    p_data[p_data$fqtr==1, 'dltrq'] = p_data[p_data$fqtr==1, 'dltry']
    # In addition impose that whenever the "annual measure" is zero, the quarterly measure must be as well
    p_data[(p_data$dltry==0) & !(is.na(p_data$dltry)) & (is.na(p_data$dltrq)), 'dltrq'] = 0
    # NET issuance
    p_data['ndltisq']    = p_data$dltisq - p_data$dltrq
    # NET CHANGE in long-term debt
    p_data['ddlttq']    = p_data$dlttq - rlag(p_data$dlttq,1)
    # Investment
    p_data["kinv"] = p_data$k_stock - rlag(p_data$k_stock,1) # Investment based on k_stock change
    # Change in cash
    p_data["dche"] = p_data$cheq - rlag(p_data$cheq,1) # Investment based on k_stock change
    # Create "other current assets" manually
    p_data["aoqm"] = p_data$actq - p_data$cheq - p_data$invtq
    # Ratios
    p_data['dltisqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$dltisq/rlag(p_data$atq,1)
    p_data['ndltisqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$ndltisq/rlag(p_data$atq,1)
    p_data['ddlttqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$ddlttq/rlag(p_data$atq,1)
    p_data['kinvat_rat'] = p_data$kinv/rlag(p_data$atq_real,1)
    p_data['invat_rat']  = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$capxq/rlag(p_data$atq,1)
    p_data['dcheat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$dche/rlag(p_data$atq,1)
    p_data['dvqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * p_data$dvq/rlag(p_data$atq,1)
    p_data['datqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * (p_data$atq-rlag(p_data$atq,1))/rlag(p_data$atq,1)
    p_data['dinvtqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * (p_data$invtq-rlag(p_data$invtq,1))/rlag(p_data$atq,1)
    p_data['daoqmat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * (p_data$aoqm-rlag(p_data$aoqm,1))/rlag(p_data$atq,1)
    p_data['dintanqat_rat'] = (rlag(p_data$GVADef,1)/p_data$GVADef) * (p_data$intanq-rlag(p_data$intanq,1))/rlag(p_data$atq,1)

    # Check firms which exhibit issuance activity or have long-term debt outstanding
    if (TRUE){
        sum_Llev_WF_dist = aggregate(Llev_rat ~ gvkey, p_data, function(x) sum(x))
        debt_firm_ind = sum_Llev_WF_dist[sum_Llev_WF_dist$Llev_rat > 0,'gvkey']
        LTborrowers_share = length(debt_firm_ind) / dim(sum_Llev_WF_dist)[1]
        writeLines(paste(round(100 * LTborrowers_share, 2), "% of firms have at least some long-term debt outstanding during the sample.", sep = ""), file.path(PROJECT_ROOT, "output", "other", "Footnote15_justification.txt"))
    }

    # Statement about leverage and issuances (dropped Footnote 79)
    if (TRUE){
        writeLines(paste("In Compustat, ", round(100 * mean(p_data$lev_rat >= 0.01), 1), "% firm-quarters have leverage (", round(100 * mean(p_data$Llev_rat >= 0.01), 1), "% for long-term leverage) ratios above 1%, yet exhibit long-term issuances in ", round(100 * mean(p_data$diss_ind, na.rm = TRUE), 1), "% of all observations.", sep = ""), file.path(PROJECT_ROOT, "output", "other", "Footnote3_justification_2.txt"))
    }

    # Statement about aggregate relevance of short-term debt in Compustat overall (Footnote 74)
    if (TRUE){
        # Aggregate ratios
        # ST debt
        aggstdebt_t <- aggregate(dlcq ~ cqtr_num, p_data[!(is.na(p_data$dttq)) & !(is.na(p_data$dlcq)),], function(x) sum(x))
        # Total debt
        aggtotdebt_t <- aggregate(dttq ~ cqtr_num, p_data[!(is.na(p_data$dttq)) & !(is.na(p_data$dlcq)),], function(x) sum(x))

        # Basic share
        aggstdebt_t$stshare  <- aggstdebt_t$dlcq / aggtotdebt_t$dttq
        med_stdebt_share    <- median(aggstdebt_t$stshare)
        writeLines(paste(round(100 * med_stdebt_share, 1), "% of Compustat debt in employed sample matures within a year.", sep = ""), file.path(PROJECT_ROOT, "output", "other", "Footnote74_justification.txt"))
    }

    ###
    # SECTION 2.2 -- LUMPINESS
    ###
    # Create indicators of "action"
    cutval_here = 0.01
    p_data$ndadj_ind <- p_data$ndltisqat_rat > cutval_here  # Net issuance based issuance
    p_data$kadj_ind <- p_data$kinvat_rat > cutval_here      # Net investment based investment
    p_data$inv_ind <- p_data$invat_rat > cutval_here        # Gross investment based investment

    # UNCONDITIONAL ACROSS FIRM-TIME MOMENTS
    # Unconditional mean debt issuance likelihood
    mom_inaction_diss_UC = 1-mean(p_data$diss_ind, na.rm=TRUE) # Gross
    # Conditional on net issuances
    mom_inaction_ndadj_UC = 1- mean(p_data$ndadj_ind, na.rm=TRUE) # Net
    # Conditional on investment
    mom_inaction_inv_UC =1-mean(p_data$inv_ind, na.rm=TRUE) # Gross
    mom_inaction_kadj_UC = 1-mean(p_data$kadj_ind, na.rm=TRUE) # Net
    # WITHIN-FIRM MOMENTS, THEN TAKE MEDIAN
    # Unconditional mean debt issuance likelihood
    inaction_diss_WF_dist = aggregate(diss_ind ~ gvkey, p_data, function(x) 1-mean(x))
    mom_inaction_diss_WF = median(inaction_diss_WF_dist[,"diss_ind"], na.rm=TRUE)
    # Conditional on net issuances
    inaction_ndadj_WF_dist = aggregate(ndadj_ind ~ gvkey, p_data, function(x) 1-mean(x))
    mom_inaction_ndadj_WF = median(inaction_ndadj_WF_dist[,"ndadj_ind"], na.rm=TRUE)
    # Conditional on investment
    inaction_inv_WF_dist = aggregate(inv_ind ~ gvkey, p_data, function(x) 1-mean(x))
    mom_inaction_inv_WF = median(inaction_inv_WF_dist[,"inv_ind"], na.rm=TRUE)
    inaction_kadj_WF_dist = aggregate(kadj_ind ~ gvkey, p_data, function(x) 1-mean(x))
    mom_inaction_kadj_WF = median(inaction_kadj_WF_dist[,"kadj_ind"], na.rm=TRUE)
    # Wilcox signed-rank tests of the comparable moments
    wilcox.test(as.double(p_data$diss_ind), as.double(p_data$inv_ind), paired=TRUE)
    wilcox.test(as.double(p_data$ndadj_ind), as.double(p_data$kadj_ind), paired=TRUE)
    df_gross_adj_WF <- merge(x = inaction_diss_WF_dist, y = inaction_inv_WF_dist, by = "gvkey", all = FALSE)
    wilcox.test(as.double(df_gross_adj_WF$diss_ind), as.double(df_gross_adj_WF$inv_ind), paired=TRUE)
    df_net_adj_WF <- merge(x = inaction_ndadj_WF_dist, y = inaction_kadj_WF_dist, by = "gvkey", all = FALSE)
    wilcox.test(as.double(df_net_adj_WF$ndadj_ind), as.double(df_net_adj_WF$kadj_ind), paired=TRUE)

    # Drop outliers of adjustment rates, for local regression analysis
    # Two-sided
    win_val_q = 0.005
    trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    # Net issuances and investment
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(ndltisqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("ndltisqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(kinvat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("kinvat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dcheat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dcheat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dvqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dvqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(datqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("datqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dinvtqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dinvtqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(daoqmat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("daoqmat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dintanqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dintanqat_rat")] <- NA
    # One-sided
    win_val_q = 0.01
    trim_keep_fn_q <- function(x) x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    # Gross issuances and investment
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dltisqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dltisqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(invat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("invat_rat")] <- NA
    # Reorder data
    p_data <- p_data[order(p_data$gvkey, p_data$cqtr_num), ]
    p_data <- pdata.frame(p_data, index=c("gvkey", "cqtr_num"))

    # Plot histograms
    # Compare net issuance and debt change
    plot_lsh_cut = 0.10
    dir.create(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness"), showWarnings = FALSE, recursive = TRUE)
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness", "FigA1_hist_ndltisqddlttqat_rat.pdf"), width = 8, height = 7)
    par(mar = c(5,5.3,2,2))
    hist(p_data[abs(p_data$ndltisqat_rat)<plot_lsh_cut,"ndltisqat_rat"], col = rgb(1, 0, 0, 0.5), xlab = "Rate value", ylab="Frequency", border = "red", main="", breaks=60, cex.lab = 1.2*font_lab_val, cex.axis = 1.2*font_lab_val, ylim=c(0.0, 130000))
    hist(p_data[abs(p_data$ddlttqat_rat)<plot_lsh_cut,"ddlttqat_rat"], col = rgb(0, 0, 1, 0.0), border = "blue", breaks=60, add=TRUE)
    # legend("topright", legend = c("Issuance", "Change"), col = c("red", "blue"), lty = 1, bty = "n")
    legend("topright", legend = c("Debt issuance", "Change in debt"), fill = c(rgb(1, 0, 0, 0.5), NA), border = c("red", "blue"), cex = 1.2*font_lab_val)
    dev.off()
    # Net issuance and investment together
    font_multr = 1.5;
    options(scipen = 9)
    plot_lsh_cut = 0.10
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness", "Fig1b_hist_ndltisqkinvat_rat.pdf"), width = 8, height = 7)
    par(mar = c(5,5.3,2,2))
    hist(p_data[abs(p_data$ndltisqat_rat)<plot_lsh_cut,"ndltisqat_rat"], col = rgb(1, 0, 0, 0.5), xlab = "Rate value", ylab="Frequency", border = "red", main="", breaks=60, cex.lab = font_multr*font_lab_val, cex.axis = font_multr*font_lab_val, ylim=c(0.0, 130000))
    hist(p_data[abs(p_data$kinvat_rat)<plot_lsh_cut,"kinvat_rat"], col = rgb(0, 0, 1, 0.0), border = "blue", breaks=60, add=TRUE)
    # legend("topright", legend = c("Debt", "Investment"), col = c("red", "blue"), lty = 1, bty = "n")
    legend("topright", legend = c("Debt issuance", "Investment"), fill = c(rgb(1, 0, 0, 0.5), NA), border = c("red", "blue"), cex = font_multr*font_lab_val)
    dev.off()
    # Gross issuance and investment together
    plot_lsh_cut = 0.10
    plot_bsh_cut = 0.0
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness", "Fig1a_hist_dltisqinvat_rat.pdf"), width = 8, height = 7)
    par(mar = c(5,5.3,2,2))
    hist(p_data[(p_data$dltisqat_rat<plot_lsh_cut) & (p_data$dltisqat_rat>=plot_bsh_cut),"dltisqat_rat"], col = rgb(1, 0, 0, 0.5), xlab = "Rate value", ylab="Frequency", border = "red", main="", breaks=60, cex.lab = font_multr*font_lab_val, cex.axis = font_multr*font_lab_val, ylim=c(0.0, 130000))
    hist(p_data[(p_data$invat_rat<plot_lsh_cut) & (p_data$invat_rat>=plot_bsh_cut),"invat_rat"], col = rgb(0, 0, 1, 0.0), border = "blue", breaks=60, add=TRUE)
    # legend("topright", legend = c("Debt", "Investment"), col = c("red", "blue"), lty = 1, bty = "n")
    legend("topright", legend = c("Debt issuance", "Investment"), fill = c(rgb(1, 0, 0, 0.5), NA), border = c("red", "blue"), cex = font_multr*font_lab_val)
    dev.off()

    # Compute kurtosis for net rates
    library(moments)
    # UNCONDITIONAL ACROSS FIRM-TIME MOMENTS
    mom_kurtosis_ndltisqat_UC = kurtosis(p_data$ndltisqat_rat, na.rm=TRUE)
    mom_kurtosis_kinvat_UC = kurtosis(p_data$kinvat_rat, na.rm=TRUE)
    # WITHIN-FIRM MOMENTS, THEN TAKE MEDIAN
    kurtosis_ndltisqat_WF_dist = aggregate(ndltisqat_rat ~ gvkey, p_data, function(x) kurtosis(x))
    mom_kurtosis_ndltisqat_WF = median(kurtosis_ndltisqat_WF_dist[,"ndltisqat_rat"], na.rm=TRUE)
    kurtosis_kinvat_WF_dist = aggregate(kinvat_rat ~ gvkey, p_data, function(x) kurtosis(x))
    mom_kurtosis_kinvat_WF = median(kurtosis_kinvat_WF_dist[,"kinvat_rat"], na.rm=TRUE)
    # Wilcox signed-rank tests of the comparable moments
    df_net_kurtosis_WF <- merge(x = kurtosis_ndltisqat_WF_dist, y = kurtosis_kinvat_WF_dist, by = "gvkey", all = FALSE)
    wilcox.test(as.double(df_net_kurtosis_WF$ndltisqat_rat), as.double(df_net_kurtosis_WF$kinvat_rat), paired=TRUE)

    # Compute autocorrelation
    # UNCONDITIONAL ACROSS FIRM-TIME MOMENTS
    acor_fun = function(x) cor(x, rlag(x,1), use="complete.obs")
    mom_acor_ndltisqat_UC    = acor_fun(p_data$ndltisqat_rat)
    mom_acor_kinvat_UC       = acor_fun(p_data$kinvat_rat)
    mom_acor_dltisqat_UC     = acor_fun(p_data$dltisqat_rat)
    mom_acor_invat_UC        = acor_fun(p_data$invat_rat)
    # WITHIN-FIRM MOMENTS, THEN TAKE MEDIAN
    compute_correlation <- function(x, y) {
        # Check if there are any complete pairs
        complete_pairs <- complete.cases(cbind(x, y))

        if (any(complete_pairs)) {
            # Compute the correlation if there are complete pairs
            return(cor(x, y, use = "complete.obs"))
        } else {
            # Return NA if there are no complete pairs
            return(NA)
        }
    }
    p_data[,"ndltisqat_rat_l1"] <- rlag(p_data$ndltisqat_rat,1)
    p_data[,"kinvat_rat_l1"] <- rlag(p_data$kinvat_rat,1)
    p_data[,"dltisqat_rat_l1"] <- rlag(p_data$dltisqat_rat,1)
    p_data[,"invat_rat_l1"] <- rlag(p_data$invat_rat,1)

    acor_ndltisqat_WF_dist = as.data.frame(ddply(p_data, c("gvkey"), summarize, ndltisqat_rat = compute_correlation(ndltisqat_rat, ndltisqat_rat_l1)))
    mom_acor_ndltisqat_WF = median(acor_ndltisqat_WF_dist[,"ndltisqat_rat"], na.rm=TRUE)
    acor_kinvat_WF_dist = as.data.frame(ddply(p_data, c("gvkey"), summarize, kinvat_rat = compute_correlation(kinvat_rat, kinvat_rat_l1)))
    mom_acor_kinvat_WF = median(acor_kinvat_WF_dist[,"kinvat_rat"], na.rm=TRUE)
    acor_dltisqat_WF_dist = as.data.frame(ddply(p_data, c("gvkey"), summarize, dltisqat_rat = compute_correlation(dltisqat_rat, dltisqat_rat_l1)))
    mom_acor_dltisqat_WF = median(acor_dltisqat_WF_dist[,"dltisqat_rat"], na.rm=TRUE)
    acor_invat_WF_dist = as.data.frame(ddply(p_data, c("gvkey"), summarize, invat_rat = compute_correlation(invat_rat, invat_rat_l1)))
    mom_acor_invat_WF = median(acor_invat_WF_dist[,"invat_rat"], na.rm=TRUE)
    # Wilcox signed-rank tests of the comparable moments
    df_gross_acor_WF <- merge(x = acor_dltisqat_WF_dist, y = acor_invat_WF_dist, by = "gvkey", all = FALSE)
    wilcox.test(as.double(df_gross_acor_WF$dltisqat_rat), as.double(df_gross_acor_WF$invat_rat), paired=TRUE)
    df_net_acor_WF <- merge(x = acor_ndltisqat_WF_dist, y = acor_kinvat_WF_dist, by = "gvkey", all = FALSE)
    wilcox.test(as.double(df_net_acor_WF$ndltisqat_rat), as.double(df_net_acor_WF$kinvat_rat), paired=TRUE)
    p_data[,c("ndltisqat_rat_l1", "kinvat_rat_l1", "dltisqat_rat_l1", "invat_rat_l1")] <- list(NULL)

    # Put everything together in a LaTeX table
    lumpy_mom_data <- data.frame(
        Moment = c("Inaction rate (UC)", "Inaction rate (WF)", "acor (UC)", "acor (WF)"),
        Gross_Debt = c(mom_inaction_diss_UC, mom_inaction_diss_WF, mom_acor_dltisqat_UC, mom_acor_dltisqat_WF),
        Gross_Investment = c(mom_inaction_inv_UC, mom_inaction_inv_WF, mom_acor_invat_UC, mom_acor_invat_WF),
        Net_Debt = c(mom_inaction_ndadj_UC, mom_inaction_ndadj_WF, mom_acor_ndltisqat_UC, mom_acor_ndltisqat_WF),
        Net_Investment = c(mom_inaction_kadj_UC, mom_inaction_kadj_WF, mom_acor_kinvat_UC, mom_acor_kinvat_WF)
    )

    library(xtable)
    # Generate LaTeX code
    lumpy_mom_latex_code <- xtable(lumpy_mom_data, caption = "Moments on the lumpiness of gross and net long-term debt issuance and investment", label = "lumpy_mom_short", align = "ll|cc|cc", table.placement="ht")
    # Add customizations
    nmom <- nrow(lumpy_mom_data)
    lumpy_mom_tab <- capture.output(print(lumpy_mom_latex_code,
        include.rownames = FALSE,
        include.colnames = FALSE,
        add.to.row = list(pos = list(-1, 0, nmom),  # -1 for the header, 0 for the first row
                            command = c("\\hline \\hline \n & \\multicolumn{2}{c|}{Gross} & \\multicolumn{2}{c}{Net} \\\\ \n",
                                        "Moment & Debt & Investment & Debt & Investment \\\\ \n", "\\hline")), caption.placement = "top",
        sanitize.text.function = function(x) x))  # To preserve the LaTeX formatting in the data
    post_tabular_mom <- c("",
        "\\vspace{4pt}",
        "\\begin{minipage}{\\textwidth}",
        "{\\footnotesize \\emph{Notes:} UC -- unconditional pooled moments across all firms and quarters in Compustat sample; WF -- within-firm moments from each firm's individual time series, median across firms. \\emph{Inaction} defined as adjustment rate below 1\\% of lagged total assets. acor denotes autocorrelation across consecutive quarters in firm.",
        "The differences of all WF moments' medians across the corresponding debt and investment moment pairs are statistically significant at the 1\\% level, following the Wilcoxon signed-rank test.",
        "}",
        "\\end{minipage}", "",
        "\\end{table}"
    )
    writeLines(c(lumpy_mom_tab[-length(lumpy_mom_tab)], post_tabular_mom), file.path(PROJECT_ROOT, "output", "tables", "TabA2_lumpy_mom_table_short.tex"))

    # Print text on kurtosis (Appendix A.2)
    writeLines(c( paste("UC kurtosis. Net debt: ", round(mom_kurtosis_ndltisqat_UC, 1), "; Net investment: ", round(mom_kurtosis_kinvat_UC, 1), sep = ""),  paste("WF kurtosis. Net debt: ", round(mom_kurtosis_ndltisqat_WF, 1), "; Net investment: ", round(mom_kurtosis_kinvat_WF, 1), sep = "")  ),  file.path(PROJECT_ROOT, "output", "other", "Text_AppxA2.txt") )


    ###
    # SECTION 2.3 -- DEBT ISSUANCE AND LIQUIDITY
    ###
    library(lfe)

    # First, analysis of predicting debt issuance
    # Introduce new variables used as controls
    p_data$dvqat_rat = p_data$dvq/rlag(p_data$atq,1)
    p_data[(p_data$dvqat_rat < 0) & !is.na(p_data$dvqat_rat),"dvqat_rat"] = 0 # Set negatives to zero
    # Create maturing debt share
    p_data$stdebt_share <- p_data$dlcq/(rlag(p_data$dlcq,1) + rlag(p_data$dlttq,1))
    p_data[p_data$stdebt_share==Inf | is.na(p_data$stdebt_share),"stdebt_share"] <- NA
    # Impose zero where zero debt, by definition
    p_data[((p_data$dlcq + p_data$dlttq)==0),"stdebt_share"] <- 0

    # Drop outliers of variables to be used in debt issuance prediction regressions
    # Two-sided
    win_val_q = 0.005
    trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylsaleq_real))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dylsaleq_real")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(lk_stock))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("lk_stock")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylk_stock))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dylk_stock")] <- NA
    # One-sided
    win_val_q = 0.01
    trim_keep_fn_q <- function(x) x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(cheat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("cheat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(lev_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("lev_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(tobin_q))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("tobin_q")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dvqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dvqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(stdebt_share))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("stdebt_share")] <- NA
    # Reorder data
    p_data <- p_data[order(p_data$gvkey, p_data$cqtr_num), ]
    p_data <- pdata.frame(p_data, index=c("gvkey", "cqtr_num"))

    # Debt issuance prediction regressions ("replicate" Table 1)
    # LT debt issuance lead
    p_data['diss_ind_lead4'] <- rlag(p_data$diss_ind,-4)
    p_data['diss_ind_lead1'] <- rlag(p_data$diss_ind,-1)
    p_data['diss_ind_lead8'] <- rlag(p_data$diss_ind,-8)
    # Run regression
    # Lead4, no firm FEs
    formla_diss_lead4_noFE <- formula("diss_ind_lead4 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr |0| sic3_datacqtr + gvkey")
    reg_diss_lead4_noFE    <- felm(formla_diss_lead4_noFE, data = p_data, na.action="na.exclude")
    # Lead4, with firm FEs
    formla_diss_lead4_FE <- formula("diss_ind_lead4 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr + gvkey |0| sic3_datacqtr + gvkey")
    reg_diss_lead4_FE    <- felm(formla_diss_lead4_FE, data = p_data, na.action="na.exclude")
    # lead1, no firm FEs
    formla_diss_lead1_noFE <- formula("diss_ind_lead1 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr |0| sic3_datacqtr + gvkey")
    reg_diss_lead1_noFE    <- felm(formla_diss_lead1_noFE, data = p_data, na.action="na.exclude")
    # lead1, with firm FEs
    formla_diss_lead1_FE <- formula("diss_ind_lead1 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr + gvkey |0| sic3_datacqtr + gvkey")
    reg_diss_lead1_FE    <- felm(formla_diss_lead1_FE, data = p_data, na.action="na.exclude")
    # lead8, no firm FEs
    formla_diss_lead8_noFE <- formula("diss_ind_lead8 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr |0| sic3_datacqtr + gvkey")
    reg_diss_lead8_noFE    <- felm(formla_diss_lead8_noFE, data = p_data, na.action="na.exclude")
    # lead8, with firm FEs
    formla_diss_lead8_FE <- formula("diss_ind_lead8 ~ 1 + rlag(tobin_q,0) + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq_real,0) + rlag(latq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat,0) + rlag(stdebt_share,0) + factor(fqtr) | sic3_datacqtr + gvkey |0| sic3_datacqtr + gvkey")
    reg_diss_lead8_FE    <- felm(formla_diss_lead8_FE, data = p_data, na.action="na.exclude")

    # Put everything in regression table
    diss_model_list <- list()
    diss_model_list$diss_lead1_noFE <- reg_diss_lead1_noFE
    diss_model_list$diss_lead1_FE <- reg_diss_lead1_FE
    diss_model_list$diss_lead4_noFE <- reg_diss_lead4_noFE
    diss_model_list$diss_lead4_FE <- reg_diss_lead4_FE
    diss_model_list$diss_lead8_noFE <- reg_diss_lead8_noFE
    diss_model_list$diss_lead8_FE <- reg_diss_lead8_FE

    # And write into LaTeX
    cov_labs <- c("Tobin's $q$", "Leverage", "Liquidity", "$D_{i,t}$", "$\\Delta_{3} \\log(\\text{Sales})$", "log(Size)", "log($k_{i,t}$)", "$\\Delta_{3} \\log(k)$", "Div/Assets", "ST share")
    cov_labs_short <- c("Tobin's $q$", "Leverage", "Liquidity", "$D_{i,t}$", "log(Size)", "$\\Delta_{3} \\log(k)$")
    tab_title <- "Debt issuance prediction regression estimates, extended (Data)"
    reg_tab <- capture.output(stargazer(diss_model_list, title=tab_title, covariate.labels=cov_labs, dep.var.labels=c("$D_{i,t+1}$", "$D_{i,t+4}$", "$D_{i,t+8}$", "$D_{i,t+4}$", "$D_{i,t+4}$", "$D_{i,t+4}$"), omit.stat=c("ser", "adj.rsq"), omit=c(11,12,13), model.numbers=FALSE, align=TRUE, no.space=TRUE, label=paste("tab_reg_debtforec_long"), add.lines=list(c("Firm FE", '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}', '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}', '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}')), notes=NULL, omit.table.layout = "n", table.placement = "!htb"))
    post_tabular <- c("","\\vspace{4pt}",
        "\\begin{minipage}{\\textwidth}",
        "{\\footnotesize \\emph{Notes:} Estimates and standard errors (in parentheses) for coefficients in \\eqref{eq_Empirics_debtforec_dynamicssec}. $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01.",
        "Standard errors clustered two-way at firm and SIC 3-digit industry-time levels.",
        "}",
        "\\end{minipage}","",
        "\\end{table}"
        )
    writeLines(c(reg_tab[-length(reg_tab)], post_tabular), file.path(PROJECT_ROOT, "output", "tables", "TabA3_reg_tab_diss_pred_extd_long.tex"))


    # Calculate marginal R^2 for each predictor
    marginal_r2 <- function(full_model, predictor_indic, predictor_lag) {
        rem_predictor_str = paste0("\\+\\s*rlag\\(",predictor_indic,",\\s*",predictor_lag,"\\)")
        full_formula <- formula(full_model)
        full_formula_str    <- as.character(full_formula)
        reduced_formula_str <- as.character(full_formula)
        reduced_formula_str <- gsub(rem_predictor_str, "", reduced_formula_str)
        reduced_formula_strout <- paste(reduced_formula_str[2], reduced_formula_str[1], paste("drop_var + ", reduced_formula_str[3],collapse=""), collapse = " ")
        reduced_formula <- as.formula(reduced_formula_strout)

        # Create a "stand-in variable" that exactly selects the same observations to be included in the "reduced model" as are included in estimating the "full model", conditional on missing values of the dropped variable
        p_data$drop_var <- 1
        p_data[is.na(rlag(p_data[[predictor_indic]],as.numeric(predictor_lag))),"drop_var"] <- NA

        reduced_model    <- felm(reduced_formula, data = p_data, na.action="na.exclude")
        reduced_r2 <- summary(reduced_model)$r.squared

        # Compute marginal R2, setup
        full_r2 <- summary(full_model)$r.squared

        return(full_r2 - reduced_r2)
    }
    # Compute the marginal R2's
    # Create dataframe to automate and collect the marginal R2's
    mr2_controls_list = c("tobin_q", "lev_rat", "cheat_rat", "diss_ind", "dylsaleq_real", "latq", "lk_stock", "dylk_stock", "dvqat_rat", "stdebt_share")
    mr2_lags_list = c("0", "0", "0", "0", "0", "0", "0", "1", "0", "0")
    ncontrs = length(mr2_controls_list)
    # Construct collector for marginal R2's
    mr2_data <- data.frame(cov_labs, matrix(0, nrow=ncontrs, ncol=length(diss_model_list)) )
    colnames(mr2_data) <- c("Controls", names(diss_model_list))
    print("Constructing marginal R^2s.")
    for (iimodel in names(diss_model_list)){
        print(iimodel)
        cctr <- 1
        for (iicontr in mr2_controls_list){
            print(iicontr)
            mr2_temp = marginal_r2(diss_model_list[[iimodel]], iicontr, mr2_lags_list[cctr])
            mr2_data[cctr, iimodel] <- mr2_temp
        cctr <- cctr+1
        }
    }
    # Transform to percentage points
    mr2_data_pp <- mr2_data
    mr2_data_pp[,-1] = 100*mr2_data_pp[,-1]

    # Generate LaTeX code
    mr2_latex_code <- xtable(mr2_data_pp, caption = "Marginal $R^{2}$ in debt issuance prediction regressions, percentage points", label = "tab:mr2_data", align = "lccccccc", table.placement="ht")
    # Add customizations
    nr2 <- nrow(mr2_data_pp)
    r2_tab <- capture.output(print(mr2_latex_code,
        include.rownames = FALSE,
        include.colnames = FALSE,
        add.to.row = list(pos = list(-1, nr2),  # -1 for the header, 0 for the first row
                            command = c("\\hline \\hline \n & \\multicolumn{2}{c}{$D_{i,t+1}$} & \\multicolumn{2}{c}{$D_{i,t+4}$} & \\multicolumn{2}{c}{$D_{i,t+8}$} \\\\ \n", paste0("\\hline Firm FE & \\multicolumn{1}{c}{No} & \\multicolumn{1}{c}{Yes} & ", "\\multicolumn{1}{c}{No} & \\multicolumn{1}{c}{Yes} & ", "\\multicolumn{1}{c}{No} & \\multicolumn{1}{c}{Yes} \\\\ \n", "\\hline \n" ))), caption.placement = "top",
        sanitize.text.function = function(x) x))  # To preserve the LaTeX formatting in the data
    post_tabular_r2 <- c("",
        "\\vspace{4pt}",
        "\\begin{minipage}{\\textwidth}",
        "{\\footnotesize \\emph{Notes:} Marginal $R^{2}$ for explanatory variables in \\eqref{eq_Empirics_debtforec_dynamicssec}.",
        "Computed based on the difference of full model $R^{2}$ and the $R^{2}$ in a restricted model which drops the corresponding control.",
        "}",
        "\\end{minipage}", "",
        "\\end{table}"
    )
    writeLines(c(r2_tab[-length(r2_tab)], post_tabular_r2), file.path(PROJECT_ROOT, "output", "tables", "TabA4_mr2_table_long.tex"))


    # Format regression Table 1 for main text
    if (TRUE){
        # Define the two reported models and the covariates to report
        mod_noFE <- reg_diss_lead1_noFE
        mod_FE   <- reg_diss_lead1_FE
        keep_vars <- c("rlag(tobin_q, 0)", "rlag(lev_rat, 0)", "rlag(cheat_rat, 0)", "rlag(dylk_stock, 1)", "rlag(latq, 0)", "rlag(diss_ind, 0)")
        keep_labs <- c("Tobin's $q$", "Leverage", "Liquidity", "$\\Delta_{3} \\log(k)$", "log(Size)", "$D_{i,t}$")
        # Helper to format numbers
        fmt3 <- function(x) sprintf("%.3f", x)
        fmt2 <- function(x) sprintf("%.2f", x)
        # Extract coefficients and standard errors from the two models
        sum_noFE <- summary(mod_noFE)
        sum_FE   <- summary(mod_FE)
        coef_noFE <- sum_noFE$coefficients
        coef_FE   <- sum_FE$coefficients

        # Pull the marginal R^2 for the same controls
        mr2_noFE <- mr2_data_pp[match(keep_labs, mr2_data_pp$Controls), "diss_lead1_noFE"]
        mr2_FE   <- mr2_data_pp[match(keep_labs, mr2_data_pp$Controls), "diss_lead1_FE"]
        # Build coef and se vectors in correct order
        b_noFE  <- coef_noFE[keep_vars, 1]
        se_noFE <- coef_noFE[keep_vars, 2]
        b_FE    <- coef_FE[keep_vars, 1]
        se_FE   <- coef_FE[keep_vars, 2]

        # Pull observations and R^2
        nobs_noFE <- nobs(mod_noFE)
        nobs_FE   <- nobs(mod_FE)
        r2_noFE <- summary(mod_noFE)$r.squared
        r2_FE   <- summary(mod_FE)$r.squared

        # Generate .tex
        short_tab <- c(
            "\\begin{table}[!htbp]",
            "\\caption{Debt issuance prediction regression estimates, $h=1$}",
            "\\vspace{-14pt}",
            "\\begin{center}",
            "\\begin{tabular}{@{\\extracolsep{1pt}} l *{6}{D{.}{.}{-1}}}",
            "\\hline \\hline",
            paste0("Dep. var.: $D_{i,t+1}$ & ", paste0("\\multicolumn{1}{c}{", keep_labs, "}", collapse = " & "), " \\\\" ),
            "\\hline", paste0("No firm FE & ", paste(fmt3(b_noFE), collapse = " & "),  " \\\\" ),
            paste0( " & ", paste(paste0("(", fmt3(se_noFE), ")"), collapse = " & "),    " \\\\" ),
            paste0( "\\hspace{8pt} Marg.\\ $R^{2}$ (pp) & ", paste(fmt2(mr2_noFE), collapse = " & "), " \\\\" ),
            "\\hline", paste0( "With firm FE & ", paste(fmt3(b_FE), collapse = " & "), " \\\\" ),
            paste0( " & ", paste(paste0("(", fmt3(se_FE), ")"), collapse = " & "),     " \\\\" ),
            paste0( "\\hspace{8pt} Marg.\\ $R^{2}$ (pp) & ", paste(fmt2(mr2_FE), collapse = " & "), " \\\\" ),
            "\\hline",
            paste0( "& \\multicolumn{6}{c}{Observations: ", format(nobs_noFE, big.mark = ","), ", \\quad $R^{2}$: ", fmt3(r2_noFE), " and ", fmt3(r2_FE),         " } \\\\" ),
            "\\hline \\hline",
            "\\end{tabular}",
            "\\label{tab_reg_debtforec_longshort}",
            "",
            "\\vspace{4pt}",
            "\\begin{minipage}{\\textwidth}",
            "{\\footnotesize \\emph{Notes:} Estimates and standard errors (in parentheses) for selected coefficients in \\eqref{eq_Empirics_debtforec_dynamicssec}.",
            "Standard errors clustered two-way at firm and SIC 3-digit industry-time levels.",
            "Marginal $R^{2}$ computed based on the difference of full model $R^{2}$ and the $R^{2}$ in a restricted model which drops the corresponding control.",
            "}",
            "\\end{minipage}",
            "\\end{center}",
            "\\vspace{-10pt}",
            "\\end{table}"
        )

        writeLines(short_tab, file.path(PROJECT_ROOT, "output", "tables", "Tab1_reg_tab_diss_pred_h1_short.tex"))


    }

    # Run event-study regressions
    library(lfe)
    nlags_diss = 8
    nc_diss = 2*nlags_diss+1
    lag_values = c(-nlags_diss:nlags_diss)

    # Start loop
    print("Running event-study regressions")
    for (ind_var in c("diss_ind", "ndltisqat_rat", "kinvat_rat")){
        print(paste0("Event indicator: ", ind_var))

        for (dep_var in c("ndltisqat_rat", "kinvat_rat", "dcheat_rat")){
            print(paste0("Outcome variable: ", dep_var))
            # Run regression
            formla_dissinv <- formula(paste(dep_var, " ~ 1 + rlag(",ind_var,",-8:8) | sic3_datacqtr + gvkey | 0 | sic3_datacqtr +gvkey", sep=""))
            reg_dissinv    <- felm(formla_dissinv, data = p_data, na.action="na.exclude")
            betas_vec <- as.double(coef(reg_dissinv))
            ci_vec <- confint(reg_dissinv)

            # Save coefficients and confidence intervals into csv files
            df_event_emp <- data.frame(
                lag = lag_values,
                beta = betas_vec,
                ci_lo = ci_vec[,1],
                ci_hi = ci_vec[,2]
            )
            write.csv(df_event_emp, file.path(PROJECT_ROOT, "interim_output", paste0("emp_dissreg_vals_ind(", ind_var, ")_dep(", dep_var, ").csv")), row.names = FALSE)
        }
    }

    # Footnote 61:
    all_dep_vars       <- c("ndltisqat_rat", "kinvat_rat", "dcheat_rat", "dinvtqat_rat", "daoqmat_rat", "dintanqat_rat")
    all_dep_vars_coefs <- numeric(6) 
    for (ii in 1:6){
        dep_var <- all_dep_vars[ii]
        formla_dissinv <- formula(paste(dep_var, " ~ 1 + rlag(diss_ind,-8:8) | sic3_datacqtr + gvkey | 0 | sic3_datacqtr +gvkey", sep=""))
        reg_dissinv    <- felm(formla_dissinv, data = p_data, na.action="na.exclude")
        all_dep_vars_coefs[ii] <- coef(reg_dissinv)["rlag(diss_ind, -8:8)0"]
    }
    other_dep_vars_sum <- sum(all_dep_vars_coefs[2:6])
    writeLines(paste0("Avg net issuance is higher by ", round(100 * all_dep_vars_coefs[1], 1), " pp at issuance, with net investment, cash, inventory, other current asset, and intangibles up in total by ", round(100 * other_dep_vars_sum, 1), " pp, accounting for about ", round(100 * other_dep_vars_sum / all_dep_vars_coefs[1], 1), "% of net debt increase."), file.path(PROJECT_ROOT, "output", "other", "Footnote61_justification.txt"))
}


###################
# SECTION 2.4 -- LIQUIDITY AND MONETARY SHOCK RESPONSIVENESS
###################

# Prepare data for local projection analysis 
library(lfe)

# Load the original dataset back into memory
p_data <- readRDS(file.path(PROJECT_ROOT, "proprietary-data-not-for-publication", "processed", "p_data_saved.rds"))

# Set up these lines to do a quick run, while work in progress, skipping long_samp, and only running the LAST base_samp LP
samp_ind_list = c("long_samp", "base_samp")
base_samp_LP_spec_list = c("cheat_only", "cheat_contr_main", "cheat_contr_main_age", "lev_only", "cheat_contr_lev", "cheat_only_noaggs", "cheat_contr_main_noaggs", "cheatdm_only", "cheatdm_contr_main", "cheat_only_JKpmt", "cheat_only_RR")

# Create outer loop to deal with long sample (2016Q4) robustness check in LP analysis
for (samp_ind in samp_ind_list){
    
if (samp_ind == "long_samp"){
    print("Running LP robustness analysis on long (up to 2016Q4) sample.")
} else {
    print("Starting main LP analysis, and reporting untargeted moments, on base sample.")
}

# Load in data
p_data_capx_q <- p_data

# Choose SHOCK MEASURE as JK's FF4 and RESCALE
p_data_capx_q$sffr <- p_data_capx_q$sffr_ff4 / sd(aggregate(sffr_ff4 ~ cqtr_num, data[(data$cqtr_num>=1990) & (data$cqtr_num<=2017),], function(x) mean(x))[,'sffr_ff4'])
# Also, normalize JKpmt by its standard deviation
p_data_capx_q$sffr_JKpmt <- p_data_capx_q$sffr_JKpmt / sd(aggregate(sffr_JKpmt ~ cqtr_num, data[(data$cqtr_num>=1990) & (data$cqtr_num<=2017),], function(x) mean(x))[,'sffr_JKpmt'])

# Also read in Romer-Romer shocks for robustness test
if (samp_ind == "base_samp"){
    RR_sffr <- read.csv(file.path(PROJECT_ROOT, "data", "raw", "sffr_RR.csv"), sep = ";")
    RR_sffr$cqtr_num = as.factor(rep(seq(1983.25,2008, by=0.25), each=3))

    RR_sffr_q       <- aggregate(RR_sffr$RRSHOCK83, by=list(RR_sffr$cqtr_num), sum)
    colnames(RR_sffr_q) <- c("cqtr_num", "RRsh")
    rownames(RR_sffr_q) <- RR_sffr_q$cqtr_num

    # Impose the shock from the RR database
    p_data_capx_q$sffr_RR <- RR_sffr_q[as.character(p_data_capx_q$cqtr_num), "RRsh"]
    # Rescale it
    p_data_capx_q$sffr_RR <- p_data_capx_q$sffr_RR / sd(aggregate(sffr_RR ~ cqtr_num, p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990) & (as.double(as.character(p_data_capx_q$cqtr_num))<=2017),], function(x) mean(x))[,'sffr_RR'])
}

# Recount observations and select sample period
if (samp_ind == "base_samp"){
    LP_samp_end = 2008.0
} else {
    LP_samp_end = 2017.0
}
p_data_capx_q <- p_data_capx_q[p_data_capx_q$cqtr_num <= LP_samp_end,]
p_data_capx_q <- p_data_capx_q[p_data_capx_q$cqtr_num >= 1989.0,]
# Drop based on post-1990 sample (while leaving main sample with lags back to 1989):
p_data_capx_q_count <- p_data_capx_q[p_data_capx_q$cqtr_num >= 1990.0,]
p_data_capx_q_count$obsc <- rep(count(p_data_capx_q_count,'gvkey')$freq,count(p_data_capx_q_count,'gvkey')$freq)
p_data_capx_q_count <- p_data_capx_q_count[p_data_capx_q_count$obsc >= 40,]
p_data_capx_q <- p_data_capx_q[p_data_capx_q$gvkey %in% p_data_capx_q_count$gvkey,]
rm(p_data_capx_q_count)
# Set financial quarter as factor
p_data_capx_q$fqtr <- as.factor(p_data_capx_q$fqtr)

# Construct average interest rate
p_data_capx_q$xintqdt_rat <- p_data_capx_q$xintq/rlag(p_data_capx_q$dttq,1)
p_data_capx_q[p_data_capx_q$xintqdt_rat == -Inf | p_data_capx_q$xintqdt_rat == Inf | is.na(p_data_capx_q$xintqdt_rat), 'xintqdt_rat'] <- NA

# Construct short-term debt share
p_data_capx_q$stdebt_share <- p_data_capx_q$dlcq/(rlag(p_data_capx_q$dlcq,1) + rlag(p_data_capx_q$dlttq,1))
p_data_capx_q[p_data_capx_q$stdebt_share==Inf | is.na(p_data_capx_q$stdebt_share),"stdebt_share"] <- NA
# Impose zero where zero debt, by definition
p_data_capx_q[((p_data_capx_q$dlcq + p_data_capx_q$dlttq)==0),"stdebt_share"] <- 0

if (TRUE){
    print("Trimming outliers in controls...")
    # Two-sided
    win_val_q = 0.005
    trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylsaleq_real))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("dylsaleq_real")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dlk_stock))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("dlk_stock")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylk_stock))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("dylk_stock")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(kinv_rat_real))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("kinv_rat_real")] <- NA
    if (DD_ind){
        p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(DD))
        p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("DD")] <- NA
    }
    # One-sided
    win_val_q = 0.01
    trim_keep_fn_q <- function(x) x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(lev_rat))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("lev_rat")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(cheat_rat))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("cheat_rat")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(tobin_q))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("tobin_q")] <- NA
    p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(stdebt_share))
    p_data_capx_q[!p_data_capx_q$keep_temp_var | is.na(p_data_capx_q$keep_temp_var) ,c("stdebt_share")] <- NA
    # Reorder data
    p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
    p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
}

# Flip around the cheat_rat -- WARNING! ONLY DO THIS LINE ONCE AFTER LOADING IN DATA!
p_data_capx_q["cheat_rat"] <- -p_data_capx_q["cheat_rat"]

# Annualized basis points
p_data_capx_q$xintqdt_rat <- 4*10000*p_data_capx_q$xintqdt_rat
# Capital into percentages
p_data_capx_q$lk_stock <- 100*p_data_capx_q$lk_stock
p_data_capx_q$kinv_rat_real <- 100*p_data_capx_q$kinv_rat_real
p_data_capx_q$lsaleq <- 100*p_data_capx_q$lsaleq
p_data_capx_q$ldttq <- 100*log(p_data_capx_q$dttq)
p_data_capx_q[p_data_capx_q$ldttq==-Inf | is.na(p_data_capx_q$ldttq),"ldttq"] <- NA
p_data_capx_q$ldlttq <- 100*log(p_data_capx_q$dlttq)
p_data_capx_q[p_data_capx_q$ldlttq==-Inf | is.na(p_data_capx_q$ldlttq),"ldlttq"] <- NA
p_data_capx_q$dltisqat_rat <- 100*p_data_capx_q$dltisqat_rat

# Demean liquidity ratio at firm level
p_data_capx_q <- transform(p_data_capx_q, cheat_rat_dm = cheat_rat-ave(cheat_rat, gvkey, FUN=function(x) mean(x, na.rm=TRUE)))
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))

# Drop large acquisitions in k observations for LP analysis
p_data_capx_q['acqq'] = diff(p_data_capx_q$aqcy,1)
p_data_capx_q[p_data_capx_q$fqtr==1, 'acqq'] = p_data_capx_q[p_data_capx_q$fqtr==1, 'aqcy']
p_data_capx_q['acqqat_rat'] = p_data_capx_q$acqq / rlag(p_data_capx_q$atq,1)
p_data_capx_q[p_data_capx_q$acqqat_rat > 0.05 & !is.na(p_data_capx_q$acqqat_rat),"lk_stock"] <- NA

# Relevel cheat_rat -- WARNING! ONLY DO THIS LINE ONCE AFTER LOADING IN DATA!
p_data_capx_q <- within(p_data_capx_q, q_cheat_rat <- relevel(q_cheat_rat, ref=n_q))
p_data_capx_q <- within(p_data_capx_q, qcs_cheat_rat <- relevel(qcs_cheat_rat, ref=3))

# Summary stats for Table A.1 in the draft (August 2023)
if (samp_ind == "base_samp"){
    # Conditioning variables -- time averages of cross-sectional statistics
    sumstat_selvars <- c("cheat_rat", "lev_rat", "atq_real", "dylsaleq_real", "dlk_stock")
    means_conds <- aggregate(p_data_capx_q[,sumstat_selvars], by=list(p_data_capx_q$cqtr_num), FUN=mean, na.rm=TRUE)
    medians_conds <- aggregate(p_data_capx_q[,sumstat_selvars], by=list(p_data_capx_q$cqtr_num), FUN=median, na.rm=TRUE)
    sds_conds <- aggregate(p_data_capx_q[,sumstat_selvars], by=list(p_data_capx_q$cqtr_num), FUN=sd, na.rm=TRUE)
    obsns_conds <- aggregate(p_data_capx_q[,sumstat_selvars], by=list(p_data_capx_q$cqtr_num), FUN=function(x) sum(!is.na(x)))
    # Correlations function
    corfunc <- function(xx,var1,var2) { return(data.frame(cor = cor(xx[,var1], xx[,var2], use = "complete.obs"))) }
    # Create tables for collecting correlations with size
    cors_cheat_size <- ddply(p_data_capx_q, .(cqtr_num), corfunc, var1="cheat_rat", var2="latq_real")
    # Collect
    cors_size_conds <- data.frame(Group1 = cors_cheat_size$cqtr_num, cheat_rat=-cors_cheat_size$cor) # minus, because "che/at flipped" earlier
    for (var_temp in sumstat_selvars[-1]){
        cors_temp_size <- ddply(p_data_capx_q, .(cqtr_num), corfunc, var1=var_temp, var2="latq_real")
        cors_size_conds[,var_temp] = cors_temp_size$cor
    }
    # Create tables for collecting correlations with liquidity
    cors_cheat_cheat <- ddply(p_data_capx_q, .(cqtr_num), corfunc, var1="cheat_rat", var2="cheat_rat")
    # Collect
    cors_cheat_conds <- data.frame(Group1 = cors_cheat_cheat$cqtr_num, cheat_rat=cors_cheat_cheat$cor)
    for (var_temp in sumstat_selvars[-1]){
        cors_temp_cheat <- ddply(p_data_capx_q, .(cqtr_num), corfunc, var1=var_temp, var2="cheat_rat")
        cors_cheat_conds[,var_temp] = -cors_temp_cheat$cor # minus, because "che/at flipped" earlier
    }

    # Create table
    sumstat_df <- data.frame(var_name = sumstat_selvars, Mean=NA, Median=NA, SD=NA, Observations=NA, cor_cheat=NA, cor_size=NA)
    rownames(sumstat_df) <- sumstat_df$var_name
    for (var_temp in sumstat_selvars){
        sumstat_df[var_temp,"Mean"] <- median(means_conds[as.double(as.character(means_conds$`Group.1`))>=1990,var_temp])
        sumstat_df[var_temp,"Median"] <- median(medians_conds[as.double(as.character(medians_conds$`Group.1`))>=1990,var_temp])
        sumstat_df[var_temp,"SD"] <- median(sds_conds[as.double(as.character(sds_conds$`Group.1`))>=1990,var_temp])
        sumstat_df[var_temp,"Observations"] <- sum(obsns_conds[as.double(as.character(obsns_conds$`Group.1`))>=1990,var_temp])
        sumstat_df[var_temp,"cor_cheat"] <- median(cors_cheat_conds[as.double(as.character(cors_cheat_conds$Group1))>=1990,var_temp])
        sumstat_df[var_temp,"cor_size"] <- median(cors_size_conds[as.double(as.character(cors_size_conds$Group1))>=1990,var_temp])
    }
    # Set the cheat_rat moments to positive
    sumstat_df["cheat_rat","Mean"] = abs(sumstat_df["cheat_rat","Mean"]) # abs, because "che/at flipped" earlier
    sumstat_df["cheat_rat","Median"] = abs(sumstat_df["cheat_rat","Median"]) # abs, because "che/at flipped" earlier

    # Automated, formatted output to latex
    library(xtable)
    if (TRUE){
        # Columns required
        sumstat_out <- sumstat_df[, c("Mean", "Median", "SD", "Observations", "cor_cheat", "cor_size")]
        # Rename columns
        colnames(sumstat_out) <- c("Mean", "Median", "St. dev", "Obs", "cor($\\cdot$, Liq)", "cor($\\cdot$, log(Size))" )
        # Rename rows
        rownames(sumstat_out) <- c( "Liquidity", "Leverage", "Size", "$\\Delta_{3} \\log(\\text{Sales})$", "$\\Delta \\log(k)$" )
        # Convert everything to character to control every cell
        sumstat_chr <- data.frame(lapply(sumstat_out, as.character), check.names = FALSE, row.names = rownames(sumstat_out))
        # Format numeric columns to 3 decimals by default
        num_cols <- c("Mean", "Median", "St. dev", "cor($\\cdot$, Liq)", "cor($\\cdot$, log(Size))")
        for (cc in num_cols) {
            sumstat_chr[[cc]] <- formatC(as.numeric(sumstat_out[[cc]]), format = "f", digits = 3)
        }
        # Format observations with commas
        sumstat_chr$Obs <- formatC(sumstat_out$Obs, format = "d", big.mark = ",")
        # Manual fixes for Size row: 2 decimals
        sumstat_chr["Size", c("Mean", "Median", "St. dev")] <- formatC( round(as.numeric(sumstat_chr["Size", c("Mean", "Median", "St. dev")]), 2), format = "f", digits = 2 )
        # Manual fixes for dashes
        sumstat_chr["Liquidity", "cor($\\cdot$, Liq)"] <- "--"
        sumstat_chr["Size", "cor($\\cdot$, log(Size))"] <- "--"
        # In the last row, there's a "-0.000" possible that rounding does not eliminate, ensure it drops the "minus"
        cols_fix <- c("Mean", "Median", "St. dev")
        sumstat_chr["$\\Delta \\log(k)$", cols_fix] <- formatC( ifelse( abs(as.numeric(sumstat_chr["$\\Delta \\log(k)$", cols_fix])) < 5e-4, 0, as.numeric(sumstat_chr["$\\Delta \\log(k)$", cols_fix]) ), format = "f", digits = 3 )
        
        # Build xtable from already-formatted strings
        xt_dtf <- xtable(sumstat_chr, digits = 0)
        align(xt_dtf) <- c("l", "c", "c", "c", "c", "c", "c")
        nrows <- nrow(sumstat_chr)
        print(xt_dtf, file = file.path(PROJECT_ROOT, "output", "tables", "TabA1_sumstat_table.tex"), include.rownames = TRUE, include.colnames = TRUE, sanitize.text.function = identity, sanitize.rownames.function = identity, only.contents = TRUE, add.to.row = list( pos = list(nrows), command = c("\\hline \n")))
    }
}

# Clean lines for untargeted moments and group-specific issuance frequencies in the draft, on local projection sample
if (samp_ind == "base_samp"){
    ###
    # Untargeted moments
    ###

    # Introduce flexible beginning and end
    temp_calib_tstart = 1990.0
    temp_calib_tend   = 2008.0

    # sd(cheat_rat) by SIC3
    sdcheat_t <- aggregate(cheat_rat ~ sic3_datacqtr, p_data_capx_q[as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0 & as.double(as.character(p_data_capx_q$cqtr_num))<=2008.0,], function(x) sd(x))
    # mean(sdcheat_t$cheat_rat, na.rm=TRUE)
    # To compute t-mean WEIGHTED by sample size!!!
    cheat_count_t <- aggregate(cheat_rat ~ sic3_datacqtr, p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0),], length)
    Table_Untarg_row1 <- weighted.mean(sdcheat_t$cheat_rat, cheat_count_t$cheat_rat, na.rm=TRUE)

    # Define function for correlations
    corfunc <- function(xx,var1,var2)
    {
    return(data.frame(cor = cor(xx[,var1], xx[,var2], use = "complete.obs")))
    }
    # Only SIC3-t cells with observations
    chelatq_cors_t <- ddply(p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0) & (p_data_capx_q$sic3_datacqtr %in% cheat_count_t$sic3_datacqtr),], .(sic3_datacqtr), corfunc, var1="latq", var2="cheat_rat")
    Table_Untarg_row2 <- -weighted.mean(chelatq_cors_t$cor, cheat_count_t$cheat_rat, na.rm=TRUE) # minus, because "che/at flipped" earlier

    p_data_capx_q$cheat_rat_l5 <- rlag(p_data_capx_q$cheat_rat,5)
    # Only SIC3-t cells with observations
    p_data_capx_q$chekinvprod <- p_data_capx_q$cheat_rat_l5 * p_data_capx_q$kinv_rat_real
    chekinv_count_t <- aggregate(chekinvprod ~ sic3_datacqtr, p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0),], length)
    chekinv_cors_t <- ddply(p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0) & (p_data_capx_q$sic3_datacqtr %in% chekinv_count_t$sic3_datacqtr),], .(sic3_datacqtr), corfunc, var1="kinv_rat_real", var2="cheat_rat_l5")
    Table_Untarg_row3 <- -weighted.mean(chekinv_cors_t$cor, chekinv_count_t$chekinvprod, na.rm=TRUE) # minus, because "che/at flipped" earlier

    # Skewness of log(capital stock)
    library(moments)
    kskew_t <- aggregate(lk_stock ~ sic3_datacqtr, p_data_capx_q[as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0 & as.double(as.character(p_data_capx_q$cqtr_num))<=2008.0,], function(x) skewness(x))
    # To compute t-mean weighted by sample size
    lk_count_t <- aggregate(lk_stock ~ sic3_datacqtr, p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=1990.0),], length)
    Table_Untarg_row8 <- weighted.mean(kskew_t$lk_stock, lk_count_t$lk_stock, na.rm=TRUE)

    # Positive quarterly dividends (over quarters)
    p_data_capx_q$dvq_pos <- p_data_capx_q$dvq > 0
    aggdivfreq_t <- aggregate(dvq_pos ~ cqtr_num, p_data_capx_q[(as.double(as.character(p_data_capx_q$cqtr_num))>=temp_calib_tstart) & (as.double(as.character(p_data_capx_q$cqtr_num))<=temp_calib_tend) & !(is.na(p_data_capx_q$dttq)) & !(is.na(p_data_capx_q$atq)) & !(is.na(p_data_capx_q$cheq)),], function(x) mean(x))
    median(aggdivfreq_t$dvq_pos)
    # Positive annual dividends (across all times, based on dvy in 4th fqtr)
    p_data_capx_q$dvy_pos <- p_data_capx_q$dvy > 0
    Table_Untarg_row11 <- mean(p_data_capx_q[p_data_capx_q$fqtr==4,"dvy_pos"], na.rm=TRUE)

    # Moments on near-zero liquidity ratio
    Table_Untarg_row6 <- mean(p_data_capx_q$cheat_rat==0, na.rm=TRUE)
    Table_Untarg_row7 <- mean(-p_data_capx_q$cheat_rat<0.001, na.rm=TRUE)  # minus, because "che/at flipped" earlier

    ###
    # Split HIGH/LOW cash and leverage in sample, based on medians
    ###
    p_data_capx_q <- transform(p_data_capx_q, cheat_rat_tdm = cheat_rat-ave(cheat_rat, datacqtr, FUN=function(x) median(x, na.rm=T)))
    p_data_capx_q <- transform(p_data_capx_q, lev_rat_tdm = lev_rat-ave(lev_rat, datacqtr, FUN=function(x) median(x, na.rm=T)))
    # With indicator
    p_data_capx_q["Hlev_ind"]   <- NA
    p_data_capx_q["Hcheat_ind"] <- NA
    p_data_capx_q["Hlatq_ind"] <- NA
    # NOTE: Based on "flipped" cheat_rat
    p_data_capx_q[p_data_capx_q$cheat_rat_tdm >= 0 | is.na(p_data_capx_q$cheat_rat_tdm), "Hcheat_ind"] <- 0
    p_data_capx_q[p_data_capx_q$cheat_rat_tdm < 0 | is.na(p_data_capx_q$cheat_rat_tdm), "Hcheat_ind"] <- 1
    p_data_capx_q[is.na(p_data_capx_q$cheat_rat_tdm), "Hcheat_ind"] <- NA
    # lev_rat
    p_data_capx_q[p_data_capx_q$lev_rat_tdm >= 0 | is.na(p_data_capx_q$lev_rat_tdm), "Hlev_ind"] <- 1
    p_data_capx_q[p_data_capx_q$lev_rat_tdm < 0 | is.na(p_data_capx_q$lev_rat_tdm), "Hlev_ind"] <- 0
    p_data_capx_q[is.na(p_data_capx_q$lev_rat_tdm), "Hlev_ind"] <- NA
    # Also, define large/small firms within SIC3-industry
    p_data_capx_q <- transform(p_data_capx_q, latq_tdm = latq-ave(latq, sic3_datacqtr, FUN=function(x) quantile(x, 0.50, na.rm=T)))
    p_data_capx_q[p_data_capx_q$latq_tdm >= 0 | is.na(p_data_capx_q$latq_tdm), "Hlatq_ind"] <- 1
    p_data_capx_q[p_data_capx_q$latq_tdm < 0 | is.na(p_data_capx_q$latq_tdm), "Hlatq_ind"] <- 0
    p_data_capx_q[is.na(p_data_capx_q$latq_tdm), "Hlatq_ind"] <- NA
    # Put panel back together
    p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
    p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
    p_data_capx_q$cqtr_num <- as.double(as.character(p_data_capx_q$cqtr_num))

    # Compute issuance frequencies by groups
    # Two-way splits
    HH_Dfreq <- mean(p_data_capx_q[(rlag(p_data_capx_q$Hcheat_ind,1)==1) & (rlag(p_data_capx_q$Hlev_ind,1)==1),"diss_ind"], na.rm=TRUE) # HH
    LL_Dfreq <- mean(p_data_capx_q[(rlag(p_data_capx_q$Hcheat_ind,1)==0) & (rlag(p_data_capx_q$Hlev_ind,1)==0),"diss_ind"], na.rm=TRUE) # LL
    LH_Dfreq <- mean(p_data_capx_q[(rlag(p_data_capx_q$Hcheat_ind,1)==0) & (rlag(p_data_capx_q$Hlev_ind,1)==1),"diss_ind"], na.rm=TRUE) # LH
    HL_Dfreq <- mean(p_data_capx_q[(rlag(p_data_capx_q$Hcheat_ind,1)==1) & (rlag(p_data_capx_q$Hlev_ind,1)==0),"diss_ind"], na.rm=TRUE) # HL
    # Print table (for compiling Figure 4 in model code)
    group_spec_Dfreq <- matrix(round(c(HL_Dfreq, HH_Dfreq, LL_Dfreq, LH_Dfreq), 4), nrow = 2, byrow = TRUE, dimnames = list(c("High liq", "Low liq"), c("Low lev", "High lev")))
    writeLines(capture.output(print(group_spec_Dfreq)), file.path(PROJECT_ROOT, "interim_output", "Fig4b_nums_groupspec_Dfreq.txt"))

    Table_Untarg_row9 <- mean(p_data_capx_q[rlag(p_data_capx_q$Hlatq_ind,1)==0,"diss_ind"], na.rm=TRUE)
    Table_Untarg_row10 <- mean(p_data_capx_q[rlag(p_data_capx_q$Hlatq_ind,1)==1,"diss_ind"], na.rm=TRUE)

    # Compute mean leverage by liquidity groups
    Table_Untarg_row4 <- mean(p_data_capx_q[p_data_capx_q$Hcheat_ind==1,"lev_rat"], na.rm=TRUE)
    Table_Untarg_row5 <- mean(p_data_capx_q[p_data_capx_q$Hcheat_ind==0,"lev_rat"], na.rm=TRUE)
    # Print vector of untargeted data moments (for Table B.2)
    writeLines(capture.output(print(as.matrix(c(round(Table_Untarg_row1,3), round(Table_Untarg_row2,3), round(Table_Untarg_row3,3), round(Table_Untarg_row4,3), round(Table_Untarg_row5,3), round(Table_Untarg_row6,3), round(Table_Untarg_row7,3), round(Table_Untarg_row8,3), round(Table_Untarg_row9,3), round(Table_Untarg_row10,3), round(Table_Untarg_row11,3)), ncol=1))), file.path(PROJECT_ROOT, "output", "tables", "_partof_TabB2_nums_untargmoms.txt"))
}

# Locally, set plot features to better format LP figures
if (TRUE){
    linew_main      = 3.0; 
    linew_ci        = 2.0; 
    font_ax_val     = 1.8; 
    font_lab_val    = 1.8; 
    font_leg_val    = 1.6; 
}

#########
# Loop for baseline LP analysis
#########

# Specifiers, same for all specs
fe_ind  = TRUE          # Requesting firm FEs, used in all specifications
dep_var = "lk_stock"
ylim_man_ind = TRUE     # Manual y-bounds on figures
lag_VAR_ct = 1          # Max lag of aggregate controls (Y)
cumsum_resp_ind = FALSE # Instead of cumulative difference, do cumulative sum of LHS variable
win_val_q  = 0.01       # Outcome variable outlier trimming bound

# Select vector of LP specifications to run
if (samp_ind == "base_samp"){
    LP_spec_list = base_samp_LP_spec_list 
} else {
    LP_spec_list = c("cheat_only_2017", "cheat_contr_main_2017")
}

# Start loop over model specifications
for (LP_spec in LP_spec_list){
    print("###")
    print(paste0("Running LP specification: ", LP_spec))

    # Only one sffr cross-term or joint, various
    if (LP_spec %in% c("cheat_only", "lev_only", "cheat_only_noaggs", "cheat_only_2017", "cheatdm_only", "cheat_only_JKpmt", "cheat_only_RR")){
        ALLin_ind = FALSE  # Single cross-term
    } else { 
        ALLin_ind   = TRUE # Joint cross-terms
    }

    # Set "selector_q", i.e., central sffr cross-term control
    if (LP_spec %in% c("cheat_only", "cheat_only_noaggs", "cheat_only_2017", "cheat_only_JKpmt", "cheat_only_RR")){
        selector_q = "cheat_rat"
    } else if (LP_spec %in% c("cheat_contr_main", "cheat_contr_main_age", "cheat_contr_main_noaggs", "cheat_contr_main_2017", "cheatdm_contr_main")) {
        selector_q = "DD"
    } else if (LP_spec %in% c("lev_only", "cheat_contr_lev")){
        selector_q = "lev_rat"
    } else if (LP_spec %in% c("cheatdm_only")){
        selector_q = "cheat_rat_dm"
    }

    # Figure y-bounds
    if (LP_spec %in% c("cheat_only", "cheat_contr_main")){
        ylim_man     = c(-5.2,2.4) # For main text
    } else {
        ylim_man     = c(-6.1,2.1) # For Appendix
    }

    # Which monetary shock to use
    if (LP_spec %in% c("cheat_only_JKpmt")){
        shock_name_ext <- "_JKpmt"
    } else if (LP_spec %in% c("cheat_only_RR")) {
        shock_name_ext <- "_RR"
    } else {
        shock_name_ext <- ""
    }

    # Which confidence bands to use
    if (LP_spec %in% c("cheatdm_only", "cheatdm_contr_main")){
        ci_lev_set = 0.90
    } else{
        ci_lev_set = 0.95
    }

    # Set filename figure labels, for clarity
    if (LP_spec %in% c("cheat_only")){
        fig_main_lab    = "Fig3a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_contr_main")) {
        fig_main_lab    = ""
        fig_extra_lab   = "Fig3b"
    } else if (LP_spec %in% c("cheat_contr_main_age")) {
        fig_main_lab    = ""
        fig_extra_lab   = "FigA12a"
    } else if (LP_spec %in% c("lev_only")) {
        fig_main_lab    = "FigA6a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_contr_lev")) {
        fig_main_lab    = "FigA6b"
        fig_extra_lab   = "FigA5a"
    } else if (LP_spec %in% c("cheat_only_noaggs")) {
        fig_main_lab    = "FigA7a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_contr_main_noaggs")) {
        fig_main_lab    = ""
        fig_extra_lab   = "FigA7b"
    } else if (LP_spec %in% c("cheatdm_only")) {
        fig_main_lab    = "FigA9a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheatdm_contr_main")) {
        fig_main_lab    = ""
        fig_extra_lab   = "FigA9b"
    } else if (LP_spec %in% c("cheat_only_JKpmt")) {
        fig_main_lab    = "FigA11a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_only_RR")) {
        fig_main_lab    = "FigA11b"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_only_2017")) {
        fig_main_lab    = "FigA8a"
        fig_extra_lab   = ""
    } else if (LP_spec %in% c("cheat_contr_main_2017")) {
        fig_main_lab    = ""
        fig_extra_lab   = "FigA8b"
    } else {
        fig_main_lab    = ""
        fig_extra_lab   = ""
    }

# Specification parameters
str_end = ""

# Lag extra back
lplus_capx_q = 0
# Loop
lag_values <- seq(0,20,1)
model_list <- list()
#lag_values <- c(0,1,2)
betas_vec_q <- matrix(0,length(lag_values),1)
ci_vec_q <- array(0,dim=c(length(lag_values),2))
# feols collectors
betas_vec_q_feols <- matrix(0,length(lag_values),1)
ci_vec_q_feols <- array(0,dim=c(length(lag_values),2))

# Add correct string at end of filenames
if (fe_ind) {str_end <- paste(str_end,"_FE", sep="")}
if (ALLin_ind) {str_end <- paste(str_end,"_ALLin", sep=""); allmodel_list <- list()}
if (cumsum_resp_ind) {str_end <- paste(str_end,"_csum", sep=""); allmodel_list <- list()}

### Start loop
ctr <- 0
cat('h=')
for (lplus_capx_q in lag_values){
ctr <- ctr +1
cat(paste0(lplus_capx_q,"..."))
lag0_capx_q=0 + lplus_capx_q # first lag of sffr
lag1_capx_q=0 + lplus_capx_q # last lag of sffr (only use one lag in practice)
lagc1_capx_q=1 + lplus_capx_q # lag of controls

# Aggregate controls (Y)
VAR_vars = c("rlag(ffr, lagc1_capx_q:(lagc1_capx_q+lag_VAR_ct-1))", "rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q+lag_VAR_ct-1))")

### Cross-terms with sffr (DONE)
if (ALLin_ind){
    if (LP_spec %in% c("cheat_contr_lev")){
        # Only one "main" selector_q, alongside cheat_rat
        controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)")
    } else if (LP_spec %in% c("cheat_contr_main_age")) {
        # Core set of controls, with DD as selector_q, adding age
        controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)", "rlag(age, lagc1_capx_q)")
    } else if (LP_spec %in% c("cheatdm_contr_main")) {
        controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat_dm, lagc1_capx_q)", "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)")
    } else {
        # Core set of controls, with DD as selector_q
        controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)")
    }
} else {
    controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""))
}

### Non-cross-terms
if (ALLin_ind){
    # Set nct conditions
    if (LP_spec %in% c("cheat_contr_lev")){
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) + rlag(cheat_rat, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(cheat_rat, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)")
    } else if (LP_spec %in% c("cheat_contr_main_age")){
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) + rlag(cheat_rat, lagc1_capx_q) + rlag(latq, lagc1_capx_q) + rlag(stdebt_share, lagc1_capx_q) + rlag(dylsaleq_real, lagc1_capx_q) + rlag(age, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)", "rlag(age, lagc1_capx_q)")
    } else if (LP_spec %in% c("cheatdm_contr_main")){
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat_dm, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) + rlag(cheat_rat_dm, lagc1_capx_q) + rlag(latq, lagc1_capx_q) + rlag(stdebt_share, lagc1_capx_q) + rlag(dylsaleq_real, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)")
    } else if (LP_spec %in% c("cheat_contr_main_noaggs")){
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)")
    } else {
        # Core set of controls, with DD as selector_q
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(cheat_rat, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) + rlag(cheat_rat, lagc1_capx_q) + rlag(latq, lagc1_capx_q) + rlag(stdebt_share, lagc1_capx_q) + rlag(dylsaleq_real, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)", "rlag(stdebt_share, lagc1_capx_q)")
    }
} else {
    # Set nct conditions
    if (LP_spec %in% c("cheat_only_noaggs")){
        # No Y cross-terms:
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(cheat_rat, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)")
    } else {
        # VAR_ct + main controls
        controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), paste("(rlag(",selector_q,", lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(cheat_rat, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)")
    }
}

# Create differences of outcome variable
diff_series <- p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q)
# Cumulative sum (log) instead of cumulative difference
if (cumsum_resp_ind){
    if (lagc1_capx_q==1){
        diff_series <- p_data_capx_q[,dep_var]
        diff_series <- (p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q))/rlag(p_data_capx_q$k_stock, lagc1_capx_q)
        diff_series[abs(diff_series)==Inf | is.na(diff_series)] <- NA
    } else {
        diff_series <- log(rowSums(rlag(p_data_capx_q[,dep_var], 0:(lagc1_capx_q-1))))
        diff_series <- (p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q))/rlag(p_data_capx_q$k_stock, lagc1_capx_q)
        diff_series[abs(diff_series)==Inf | is.na(diff_series)] <- NA
    }
}
p_data_capx_q$diff_dep_var <- diff_series

# And trim data by quarter based on this diff_series
trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_diff_dep_var = trim_keep_fn_q(diff_dep_var))
p_data_capx_q[is.na(p_data_capx_q$keep_diff_dep_var),'keep_diff_dep_var'] <- TRUE
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
# Set to NA
p_data_capx_q[!p_data_capx_q$keep_diff_dep_var,'diff_dep_var'] <- NA
# Drop outliers
p_data_capx_q_used <- p_data_capx_q

### Control dummies
# Include industry*time fixed effects
dummies_capx_q = c("sic3_datacqtr")
# If requested, add firm fixed effects
if (fe_ind) {
    dummies_capx_q = c(dummies_capx_q, "gvkey")
}

# Paste together regression formula, with clustering at industry*time and firm levels
formla_capx_q <- formula(paste("diff_dep_var ~ rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):(", paste(controls_ct_capx_q, collapse=" + "), ") + ", paste(controls_nct_capx_q, collapse=" + "), "  | ", paste(dummies_capx_q, collapse=" + "), "|0 | sic3_datacqtr + gvkey", sep="" ))

# Check Driscoll-Kraay standard errors with the feols command from the fixest package, for the two baseline specifications
if (LP_spec %in% c("cheat_only", "cheat_contr_main")){
    library(fixest)
    # Locally create the base regression's controls
    p_data_capx_q_used$lsffr <- rlag(p_data_capx_q_used$sffr, lag0_capx_q)
    p_data_capx_q_used$lche  <- rlag(p_data_capx_q_used$cheat_rat, lagc1_capx_q + 0)
    p_data_capx_q_used$llatq  <- rlag(p_data_capx_q_used$latq, lagc1_capx_q + 0)
    p_data_capx_q_used$llev  <- rlag(p_data_capx_q_used$lev_rat, lagc1_capx_q + 0)
    p_data_capx_q_used$ldylsale  <- rlag(p_data_capx_q_used$dylsaleq_real, lagc1_capx_q + 0)
    p_data_capx_q_used$lffr  <- rlag(p_data_capx_q_used$ffr, lagc1_capx_q)
    p_data_capx_q_used$ldlgdp  <- rlag(p_data_capx_q_used$dlgdp_real, lagc1_capx_q)
    # Estimate feols
    if (ALLin_ind==FALSE){
        est_feols <- feols(diff_dep_var ~ lche + llatq + ldylsale + llev + lsffr:lche + lffr:lche + ldlgdp:lche | gvkey + sic3_datacqtr, panel.id=c('gvkey','datacqtr') , data=p_data_capx_q_used, vcov_DK(lag=12), notes = FALSE)
    } else {
        # For covariate cross-terms
        p_data_capx_q_used$lDD  <- rlag(p_data_capx_q_used$DD, lagc1_capx_q + 0)
        p_data_capx_q_used$lstdebt_share  <- rlag(p_data_capx_q_used$stdebt_share, lagc1_capx_q + 0)
        est_feols <- feols(diff_dep_var ~ lche + llatq + ldylsale + llev + lDD + lstdebt_share + lsffr:lche + lffr:lche + ldlgdp:lche + (lsffr + lffr + ldlgdp):(ldylsale + lDD + lstdebt_share + llatq) | gvkey + sic3_datacqtr, panel.id=c('gvkey','datacqtr') , data=p_data_capx_q_used, vcov_DK(lag=12), notes = FALSE)
    }
    # Save coefs and confidence intervals
    betas_vec_q_feols[ctr,1] <- coef(est_feols)["lche:lsffr"]
    ci_vec_q_feols[ctr,] <- as.numeric(confint(est_feols, level=ci_lev_set)["lche:lsffr",])
}

# Estimate
reg_capx_q <- felm(formla_capx_q, data=p_data_capx_q_used)
reg_sum_capx_q <- summary(reg_capx_q)

reg_coefs_capx_q <- coef(reg_sum_capx_q)
# Save coefs
betas_vec_q[ctr,1] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0)", sep=""))]
ci_vec_q[ctr,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0)", sep="")),]

# If ALLin, save regression results anyways
if (ALLin_ind){
   eval(parse(text=paste("allmodel_list$reg_tab_",as.character(lplus_capx_q)," <- reg_capx_q",sep="")))
}

} # End of estimations loop

# Plot results for main variable
dir.create(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP"), showWarnings = FALSE, recursive = TRUE)
if (!(LP_spec %in% c("cheat_contr_main", "cheat_contr_main_age", "cheat_contr_main_noaggs", "cheat_contr_main_2017", "cheatdm_contr_main"))){
pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_main_lab, "_slevs_betas_vec_q(", dep_var, ")(", selector_q, ")", str_end, ".pdf")), width = 8, height = 7)
par(mar = c(5,5.3,2,2))
cl<- rainbow(1, start=0.7)
cl <- c("blue")
markers <- rep(c(4, 19), 1)
if (ylim_man_ind){
    plot(lag_values, betas_vec_q[,1], las=1, type="n", ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
} else {
    plot(lag_values, betas_vec_q[,1], type="n", ylim=c(-0.0002+min(0.00, min(ci_vec_q[,1])) , 0.0002+max(0,max(ci_vec_q[,2]))), xlab = "h", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
}
lines(lag_values,betas_vec_q[,1], type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_vec_q[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_vec_q[,2], lty=2, col=cl[1], lwd=linew_ci);
abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2")
dev.off()
}

# Plot estimates also for feols method (for Driscoll-Kraay)
if (LP_spec %in% c("cheat_only", "cheat_contr_main")){
    if (ALLin_ind) {fig_DK_lab <- "FigA10b"} else {fig_DK_lab <- "FigA10a"}
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_DK_lab, "_slevs_betas_vec_q(", dep_var, ")(cheat_rat)", str_end, "_feols.pdf")), width = 8, height = 7)
    par(mar = c(5,5.3,2,2))
    cl<- rainbow(1, start=0.7)
    cl <- c("blue")
    markers <- rep(c(4, 19), 1)
    if (ylim_man_ind){
        plot(lag_values, betas_vec_q_feols[,1], las=1, type="n", ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
    } else {
        plot(lag_values, betas_vec_q_feols[,1], type="n", ylim=c(-0.0002+min(0.00, min(ci_vec_q[,1])) , 0.0002+max(0,max(ci_vec_q[,2]))), xlab = "h", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
    }
    lines(lag_values,betas_vec_q_feols[,1], type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_vec_q_feols[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_vec_q_feols[,2], lty=2, col=cl[1], lwd=linew_ci);
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2")
    dev.off()
}

# If there are also other variables of interest in the regression, do the same for those
if (ALLin_ind) {
    c_name <- 1
    if (LP_spec %in% "cheatdm_contr_main") {
        varname="cheat_rat_dm"
    } else {
        varname="cheat_rat"
    }

    betas_extra <- matrix(0,length(lag_values),1)
    ci_extra <- array(0,dim=c(length(lag_values),2))
    # print(varname)
    ctr <- 0
    for (lplus_capx_q in lag_values){
        ctr <- ctr +1
        eval(parse(text=paste("cur_mod<-allmodel_list$reg_tab_",as.character(lplus_capx_q),sep="")))
        eval(parse(text=paste("cur_mod_sum<-summary(allmodel_list$reg_tab_",as.character(lplus_capx_q),")",sep="")))
        reg_coefs_extra <- coef(cur_mod_sum)
        # Save coefs
        betas_extra[ctr,1] <- reg_coefs_extra[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", sep=""))]
        ci_extra[ctr,] <- confint(cur_mod, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", sep="")),]
    }
    # Plot
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_extra_lab, "_slevs_betas_vec_q(", dep_var, ")(", varname, ")", str_end, ".pdf")), width = 8, height = 7)
    par(mar = c(5,5.3,2,2))
    cl<- c("blue")
    markers <- rep(c(4, 19), 1)
    if (ylim_man_ind){
        plot(lag_values, betas_extra[,1], las=1, type="n",, ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
    } else {
        plot(lag_values, betas_extra[,1], type="n",, ylim=c(-0.0002+min(0.00, min(ci_extra[,1])) , 0.0002+max(0,max(ci_extra[,2]))), xlab = "h", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
    }
    lines(lag_values,betas_extra[,1], type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_extra[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_extra[,2], lty=2, col=cl[1], lwd=linew_ci);
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2")
    dev.off()

}


}

print("###")

} # End samp_ind loop


########
# Estimate LP-IV specifications, for comparison with model
########

# FFR units into 25bp
p_data_capx_q$dffr      <- p_data_capx_q$dffr / 0.25
# Specifiers, same for all specs
fe_ind  = TRUE          # Requesting firm FEs, used in all specifications
win_val_q  = 0.01       # Outcome variable outlier trimming bound

# Select vector of LP-IV specifications to run
LPIV_spec_list = c("cheat_IV", "cheatdm_IV", "xintqdt_IV", "dltisqat_IV")

# Start loop over model specifications
for (LPIV_spec in LPIV_spec_list){
    print("###")
    print(paste0("Running LP-IV specification: ", LPIV_spec))

    # Set "selector_q", i.e., central sffr cross-term control
    if (LPIV_spec %in% c("cheat_IV", "xintqdt_IV", "dltisqat_IV")){
        selector_q = "cheat_rat"
    } else if (LPIV_spec %in% c("cheatdm_IV")){
        selector_q = "cheat_rat_dm"
    }

    # Set "dep_var"
    if (LPIV_spec %in% c("cheat_IV", "cheatdm_IV")){
        dep_var = "lk_stock"
    } else if (LPIV_spec %in% c("xintqdt_IV")){
        dep_var = "xintqdt_rat"
    } else if (LPIV_spec %in% c("dltisqat_IV")){
        dep_var = "dltisqat_rat"
    }

    # Which confidence bands to use
    if (LPIV_spec %in% c("cheatdm_IV")){
        ci_lev_set = 0.90
    } else{
        ci_lev_set = 0.95
    }

    # Set filename figure labels, for clarity
    if (LPIV_spec %in% c("cheat_IV")){
        fig_IV_lab    = "Fig6"
    } else if (LPIV_spec %in% c("cheatdm_IV")) {
        fig_IV_lab    = "FigB2"
    } else if (LPIV_spec %in% c("xintqdt_IV")) {
        fig_IV_lab    = "FigB3b"
    } else if (LPIV_spec %in% c("dltisqat_IV")) {
        fig_IV_lab    = "FigB3a"
    }

    # Maximum lag H and figure y-bounds
    if (dep_var == "lk_stock"){
        lag_values <- c(0:20)
        ylim_man= c(-6.1, 2.1)
    } else if (dep_var == "xintqdt_rat"){
        lag_values<-c(0:8)
        ylim_man= c(-40,120)
    } else if (dep_var == "dltisqat_rat"){
        lag_values<-c(0:8)
        ylim_man= c(-0.8,0.3)
    }

# Specification parameters
str_end = "_IV"

# Lag extra back
lplus_capx_q = 0
# Loop

betas_vec_q <- matrix(0,length(lag_values),1)
ci_vec_q <- array(0,dim=c(length(lag_values),2))

# Add correct string at end of filenames
if (fe_ind) {str_end <- paste(str_end,"_FE", sep="")}

### Start loop
ctr <- 0
cat('h=')
for (lplus_capx_q in lag_values){
ctr <- ctr +1
cat(paste0(lplus_capx_q,"..."))
lagc0_capx_q=0 + lplus_capx_q # lag of earlier controls
lagc1_capx_q=1 + lplus_capx_q # lag of later controls

# Aggregate controls (Y)
VAR_vars = c("rlag(ffr, lagc1_capx_q:(lagc1_capx_q))", "rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q))")

### Cross-terms variables list
ct_varname_list =c(selector_q)

# Create explicit cross-terms in all possible cross-term variables
inst_controls_ct_capx_q_all =c(selector_q)
for (varname in inst_controls_ct_capx_q_all){
    eval(parse(text=paste("p_data_capx_q$",varname,"Xsffr<-rlag(p_data_capx_q$",varname,",",as.character(lagc1_capx_q),")*rlag(p_data_capx_q$sffr,",as.character(lagc0_capx_q),")",sep="")))
    eval(parse(text=paste("p_data_capx_q$",varname,"Xdffr<-rlag(p_data_capx_q$",varname,",",as.character(lagc1_capx_q),")*rlag(p_data_capx_q$dffr,",as.character(lagc0_capx_q),")",sep="")))
}

### Non-cross-terms
controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), paste("(rlag(",selector_q,", lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "rlag(latq, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", "rlag(cheat_rat, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)")

# Create differences of outcome variable
diff_series <- p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q)
p_data_capx_q$diff_dep_var <- diff_series

# And trim data by quarter based on this diff_series
trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_diff_dep_var = trim_keep_fn_q(diff_dep_var))
p_data_capx_q[is.na(p_data_capx_q$keep_diff_dep_var),'keep_diff_dep_var'] <- TRUE
p_data_capx_q[!p_data_capx_q$keep_diff_dep_var,'diff_dep_var'] <- NA
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
# Drop outliers
p_data_capx_q_used <- p_data_capx_q

### Control dummies
# Include industry*time fixed effects
dummies_capx_q = c("sic3_datacqtr")

# If requested, add firm fixed effects
if (fe_ind) {
    dummies_capx_q = c(dummies_capx_q, "gvkey")
}

# Paste together regression formula, with clustering at industry*time and firm levels
formla_capx_q <- formula(paste("diff_dep_var ~ ", paste(controls_nct_capx_q, collapse=" + "), "  | ", paste(dummies_capx_q, collapse=" + "), "|(", paste(paste(ct_varname_list, "Xdffr",sep=""), collapse="|"), "~", paste(paste(ct_varname_list, "Xsffr",sep=""), collapse="+"),") | sic3_datacqtr + gvkey" ))

# Estimate
reg_capx_q <- felm(formla_capx_q, data=p_data_capx_q_used)
reg_sum_capx_q <- summary(reg_capx_q)

reg_coefs_capx_q <- coef(reg_sum_capx_q)
# Save coefs
betas_vec_q[ctr,1] <- reg_coefs_capx_q[, "Estimate"][paste("`",selector_q,"Xdffr(fit)`", sep="")]
ci_vec_q[ctr,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("`",selector_q,"Xdffr(fit)`", sep="")),]

} # End of estimations loop

# Save regression coefficients and confidence intervals into file, to be used by model solution code in creating figures
write.csv(betas_vec_q, file.path(PROJECT_ROOT, "interim_output", paste0(fig_IV_lab, "_slevsbetas_vec_q_vals(", dep_var, ")(", selector_q, ")", str_end, ".csv")))
write.csv(ci_vec_q, file.path(PROJECT_ROOT, "interim_output", paste0(fig_IV_lab, "_slevsbetas_vec_q_cis(", dep_var, ")(", selector_q, ")", str_end, ".csv")))

} # End of "LPIV_spec" loop

########
# Estimate triple-interaction specifications (age and equity-reliance)
########

# NOTE: 'young_ind' is the code stand-in for a generic group-identifier

# Specifiers, same for all specs
fe_ind  = TRUE          # Requesting firm FEs, used in all specifications
win_val_q  = 0.01       # Outcome variable outlier trimming bound
selector_q  = "cheat_rat"
dep_var     = "lk_stock"
ci_lev_set = 0.95
ci_true_ind = TRUE  # Compute correct confidence intervals for sums of parameters, based on delta method
shock_name_ext = "" # Baseline shocks

# Select vector of LPT specifications to run
LPT_spec_list = c("age_LPT", "edep_perm_LPT", "eiss_dyn_LPT")

for (LPT_spec in LPT_spec_list){
    print("###")
    print(paste0("Running LPT specification: ", LPT_spec))

    # Define 'young_ind'
    if (LPT_spec%in% c("age_LPT")){
        # Young firms
        p_data_capx_q['young_ind'] <- as.integer(p_data_capx_q$age < 15.0)
        # Filename and legend labels
        fig_LPT_lab = "FigA12b"
        young_label   = "Younger"
        old_label  = "Older"
    } else if (LPT_spec%in% c("edep_perm_LPT")){
        # "Equity-reliant" firms ("permanent types")
        sum_ediss_WF_dist = aggregate(eqat_rat ~ gvkey, p_data_capx_q, function(x) mean(x))
        eissuer_firm_ind = sum_ediss_WF_dist[sum_ediss_WF_dist$eqat_rat > median(sum_ediss_WF_dist$eqat_rat),'gvkey']
        # Identify "high issuers"
        p_data_capx_q$young_ind <- as.integer(p_data_capx_q$gvkey %in% eissuer_firm_ind)
        # Filename and legend labels
        fig_LPT_lab = "FigA13a"
        young_label   = "Equity-dependent"
        old_label  = "Others"
    } else if (LPT_spec%in% c("eiss_dyn_LPT")){
        p_data_capx_q$eiss_ind <- as.integer(p_data_capx_q$eqat_rat > 0.01)
        p_data_capx_q$ysumeiss <- rowSums(rlag(p_data_capx_q$eiss_ind,0:3))
        # Identify "lead issuers"
        p_data_capx_q$young_ind <- as.integer(rlag(p_data_capx_q$ysumeiss,-4) > 0)
        # Filename and legend labels
        fig_LPT_lab = "FigA13b"
        young_label   = "Equity issuers"
        old_label  = "Non-issuers"
    }


# Specification parameters
str_end = ""

# Lag extra back
lplus_capx_q = 0
# Loop
lag_values <- seq(0,20,1)
betas_vec_q <- matrix(0,length(lag_values),1)
ci_vec_q <- array(0,dim=c(length(lag_values),2))
yb_vec_q <- matrix(0,length(lag_values),1)
yb_ci_vec_q <- array(0,dim=c(length(lag_values),2))
ylevb_vec_q <- matrix(0,length(lag_values),1)
ylevb_ci_vec_q <- array(0,dim=c(length(lag_values),2))

# Add correct string at end of filenames
if (fe_ind) {str_end <- paste(str_end,"_FE", sep="")}

### Start loop
ctr <- 0
cat('h=')
for (lplus_capx_q in lag_values){
ctr <- ctr +1
cat(paste0(lplus_capx_q,"..."))
lag0_capx_q=0 + lplus_capx_q # first lag of sffr
lag1_capx_q=0 + lplus_capx_q # last lag of sffr
lagc1_capx_q=1 + lplus_capx_q # lag of controls

# Aggregate controls (Y)
VAR_vars = c("rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q))", "rlag(ffr, lagc1_capx_q:(lagc1_capx_q))")

### Cross-terms
controls_ct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(young_ind, lagc1_capx_q)", paste("rlag(",selector_q,", lagc1_capx_q + 0):rlag(young_ind, lagc1_capx_q)", sep=""))
### Non-cross-terms
controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(young_ind, lagc1_capx_q)", paste("rlag(",selector_q,", lagc1_capx_q + 0):rlag(young_ind, lagc1_capx_q)", sep=""), "rlag(latq, lagc1_capx_q)", "rlag(lev_rat, lagc1_capx_q)", "rlag(cheat_rat, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) +", "rlag(",selector_q,", lagc1_capx_q):rlag(young_ind, lagc1_capx_q)" ," +  rlag(young_ind, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""), "(rlag(latq, lagc1_capx_q) + rlag(lev_rat, lagc1_capx_q) + rlag(cheat_rat, lagc1_capx_q) + rlag(dylsaleq_real, lagc1_capx_q)):rlag(young_ind, lagc1_capx_q)")

# Create differences of outcome variable
diff_series <- p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q)
p_data_capx_q$diff_dep_var <- diff_series
# And trim data by quarter based on this diff_series
trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_diff_dep_var = trim_keep_fn_q(diff_dep_var))
p_data_capx_q[is.na(p_data_capx_q$keep_diff_dep_var),'keep_diff_dep_var'] <- TRUE
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
# Set to NA
p_data_capx_q[!p_data_capx_q$keep_diff_dep_var,'diff_dep_var'] <- NA
# Drop outliers
p_data_capx_q_used <- p_data_capx_q

### Control dummies
# Include industry*time fixed effects
dummies_capx_q = c("sic3_datacqtr")
# If requested, add firm fixed effects
if (fe_ind) {
    dummies_capx_q = c(dummies_capx_q, "gvkey")
}

# Paste together regression formula, with clustering at industry*time and firm levels
formla_capx_q <- formula(paste("diff_dep_var ~ rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):(", paste(controls_ct_capx_q, collapse=" + "), ") + ", paste(controls_nct_capx_q, collapse=" + "), "  | ", paste(dummies_capx_q, collapse=" + "), "|0 | sic3_datacqtr + gvkey", sep="" ))

# Estimate
reg_capx_q <- felm(formla_capx_q, data=p_data_capx_q_used)
reg_sum_capx_q <- summary(reg_capx_q)

reg_coefs_capx_q <- coef(reg_sum_capx_q)
# Save coefs
betas_vec_q[ctr,1] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0)", sep=""))]
ci_vec_q[ctr,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0)", sep="")),]
# Save coefs for young_ind
yb_vec_q[ctr,1] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0):rlag(young_ind, lagc1_capx_q)", sep=""))]
yb_ci_vec_q[ctr,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0):rlag(young_ind, lagc1_capx_q)", sep="")),]

# Compute and collect true confidence intervals of sums of parameters, based on delta method
if (ci_true_ind){
    crit_val = 1.96
    if (ci_lev_set==0.90){crit_val = 1.645}
    VCV_cur = reg_capx_q$clustervcv
    nK = dim(VCV_cur)[1]
    sel_vec <- numeric(nK)
    lev_ind <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), row.names(VCV_cur))
    yb_ind  <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q,", lagc1_capx_q + 0):rlag(young_ind, lagc1_capx_q)", sep=""), row.names(VCV_cur))
    sel_vec[c(lev_ind,yb_ind)] <- c(1,1)
    SE_sum <- sqrt(sel_vec %*% VCV_cur %*% sel_vec)
    # Save
    ylevb_vec_q[ctr,1] <- betas_vec_q[ctr] + yb_vec_q[ctr]
    ylevb_ci_vec_q[ctr,] <- ylevb_vec_q[ctr,1] + c(-1,1) * crit_val * SE_sum
}

} # End of estimations loop

# Plot results for main coefs and coefs + young_ind -- with correct confidence intervals
pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_LPT_lab, "_slevs_betas-yb_vec_q_citrue(", dep_var, ")(", selector_q, ")", str_end, ".pdf")), width = 8, height = 7)
par(mar = c(5,5.3,2,2))
cl<- rainbow(1, start=0.7)
cl <- c("blue")
cl2 <- c("red")
markers <- rep(c(4, 19), 1)
plot(lag_values, betas_vec_q[,1], las=1, type="n", ylim=c(-0.0002+min(0.00, min(ci_vec_q[,1], ylevb_ci_vec_q[,1])) , 0.0002+max(0,max(ci_vec_q[,2], ylevb_ci_vec_q[,2]))), xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
lines(lag_values,betas_vec_q[,1], type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_vec_q[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_vec_q[,2], lty=2, col=cl[1], lwd=linew_ci);
lines(lag_values,ylevb_vec_q[,1], type="o", col=cl2[1], pch=markers[2], lwd=linew_main); lines(lag_values, ylevb_ci_vec_q[,1], lty=2, col=cl2[1], lwd=linew_ci); lines(lag_values, ylevb_ci_vec_q[,2], lty=2, col=cl2[1], lwd=linew_ci);
abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2"); legend("topleft", lwd=rep(linew_main,2), lty=rep(2,1), col=c("blue","red"), pch=markers, cex=font_leg_val, legend=c(old_label,young_label))
dev.off()

} # End of "LPT_spec" loop



########
# Estimate specifications based on quantile-groupings
########

# Specifiers, same for all specs
fe_ind  = TRUE          # Requesting firm FEs, even used in all specifications
win_val_q  = 0.01       # Outcome variable outlier trimming bound
selector_q = "q_Nlev_rat" # Run only specifications with q_Nlev_rat inside
dep_var     = "lk_stock"
ci_lev_set = 0.95
shock_name_ext = ""     # Baseline shocks
lag_VAR_ct = 1

# Parameters for figures
paper_2line_ind = TRUE
paper_2line_linew_main_set = linew_main; paper_2line_linew_ci_set = linew_ci;
ylim_man_ind = TRUE # Manual limits on y-axis
ylim_man     = c(-2.3,1.4) 

# Select vector of LPT specifications to run
LPq_spec_list = c("q_Nlev_only", "q_cheat_Nlev")

for (LPq_spec in LPq_spec_list){
    print("###")
    print(paste0("Running LPq specification: ", LPq_spec))

    # Only one sffr cross-term or joint, various
    if (LPq_spec %in% c("q_Nlev_only")){
        ALLin_ind = FALSE  # Single cross-term
    } else if (LPq_spec %in% c("q_cheat_Nlev")){ 
        ALLin_ind   = TRUE # Joint cross-terms
    }

    if (LPq_spec %in% c("q_Nlev_only")){
        fig_main_lab    = "FigA6c"
        fig_extra_lab   = ""
    } else if (LPq_spec %in% c("q_cheat_Nlev")){ 
        fig_main_lab    = "FigA6d"
        fig_extra_lab   = "FigA5b"
    }

# Specification parameters
str_end = ""

# Lag extra back
lplus_capx_q = 0
# Loop
lag_values <- seq(0,20,1)
betas_vec_q <- matrix(0,length(lag_values),n_q-1)
ci_vec_q <- array(0,dim=c(length(lag_values),n_q-1,2))

# Create explicit dummies with appropriate "levels" for the quantiles, using arbitrary outcome variable, that is never NA
dummy_mat <- model.matrix(formula(paste("latq ~ atq + as.numeric(as.character(gvkey)) +", selector_q, sep="")), data=p_data_capx_q)
if (selector_q == 'q_cheat_rat') {
    p_data_capx_q[,c(paste(selector_q, as.character(1:(n_q-1)), sep=""))] = dummy_mat[,c(paste(selector_q, as.character(1:(n_q-1)), sep=""))]
} else {
    if (n_q>=2) {
        p_data_capx_q[,c(paste(selector_q, as.character(2:n_q), sep=""))] = dummy_mat[,c(paste(selector_q, as.character(2:n_q), sep=""))]
    } else {
        # For binary splits:
        p_data_capx_q[,c(paste(selector_q, as.character(2:n_q), sep=""))] = dummy_mat[,c(paste(selector_q, sep=""))]
    }
}

# Add correct string at end of filenames
if (fe_ind) {str_end <- paste(str_end,"_FE", sep="")}
if (ALLin_ind) {str_end <- paste(str_end,"_ALLin", sep=""); allmodel_list <- list()}

### Start loop
ctr <- 0
cat('h=')
for (lplus_capx_q in lag_values){
ctr <- ctr +1
cat(paste0(lplus_capx_q,"..."))
lag0_capx_q=0 + lplus_capx_q # first lag of dffr
lag1_capx_q=0 + lplus_capx_q # last lag of dffr
lagc0_capx_q=0 + lplus_capx_q # lag of earlier controls
lagc1_capx_q=1 + lplus_capx_q # lag of later controls

# Aggregate controls (Y)
VAR_vars = c("rlag(ffr, lagc1_capx_q:(lagc1_capx_q+lag_VAR_ct-1))", "rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q+lag_VAR_ct-1))")

### Cross-terms
if (ALLin_ind){
    controls_ct_capx_q =c(paste("rlag(",selector_q, as.character(2:n_q),", lagc1_capx_q + 0)", sep=""), "rlag(q_cheat_rat, lagc1_capx_q)")
} else {
    if (selector_q == 'q_cheat_rat'){
        controls_ct_capx_q =c(paste("rlag(",selector_q, as.character(1:(n_q-1)),", lagc1_capx_q + 0)", sep=""))
    } else {
        controls_ct_capx_q =c(paste("rlag(",selector_q, as.character(2:n_q),", lagc1_capx_q + 0)", sep=""))
    }
}

### Non-cross-terms
if (ALLin_ind){
    controls_nct_capx_q = c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(latq, lagc1_capx_q)", "rlag(q_cheat_rat, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q) + rlag(q_cheat_rat, lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""))
} else {
    controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(latq, lagc1_capx_q)", "rlag(q_cheat_rat, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q)):(", paste(VAR_vars, collapse=" + "),")",sep=""))
}

# Create differences of outcome variable
diff_series <- p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q)
p_data_capx_q$diff_dep_var <- diff_series
# And trim data by quarter based on this diff_series
trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_diff_dep_var = trim_keep_fn_q(diff_dep_var))
p_data_capx_q[is.na(p_data_capx_q$keep_diff_dep_var),'keep_diff_dep_var'] <- TRUE
p_data_capx_q[!p_data_capx_q$keep_diff_dep_var,'diff_dep_var'] <- NA
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
# Drop outliers
p_data_capx_q_used <- p_data_capx_q

### Control dummies
# Include industry*time fixed effects
dummies_capx_q = c("sic3_datacqtr")
# If requested, add firm fixed effects
if (fe_ind) {
    dummies_capx_q = c(dummies_capx_q, "gvkey")
}

# Paste together regression formula, with clustering at industry*time and firm levels
formla_capx_q <- formula(paste("diff_dep_var ~ rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):(", paste(controls_ct_capx_q, collapse=" + "), ") + ", paste(controls_nct_capx_q, collapse=" + "), "  | ", paste(dummies_capx_q, collapse=" + "), "|0 | sic3_datacqtr + gvkey", sep="" ))

# Estimate
reg_capx_q <- felm(formla_capx_q, data=p_data_capx_q_used)
reg_sum_capx_q <- summary(reg_capx_q)

reg_coefs_capx_q <- coef(reg_sum_capx_q)
# Save coefs
for (ii in 1:(n_q-1)){
    if (selector_q == 'q_cheat_rat'){
        betas_vec_q[ctr,ii] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep=""))]
        ci_vec_q[ctr,ii,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep="")),]
    } else {
        betas_vec_q[ctr,ii] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep=""))]
        ci_vec_q[ctr,ii,] <- confint(reg_capx_q, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep="")),]
    }
}

# If ALLin, save regression results anyways
if (ALLin_ind){
   eval(parse(text=paste("allmodel_list$reg_tab_",as.character(lplus_capx_q)," <- reg_capx_q",sep="")))
}

} # End of estimations loop


# Plot results for main variable
pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_main_lab, "_betas_vec_q(", dep_var, ")(", selector_q, ")", str_end, ".pdf")), width = 8, height = 7)
par(mar = c(5,5,2,2))
cl<- rainbow(n_q, start=0.1)
cl<-rev(cl[-1])
if (col_change_ind & n_q == 2){cl[1] <- "blue"}
if (col_change_ind){cl[3] <- "sienna1"}
if (paper_2line_ind){linew_main = paper_2line_linew_main_set; linew_ci = paper_2line_linew_ci_set; confint_ind = TRUE; cl[2] <- "sienna1"}
markers <- rep(c(4, 19), length=n_q-1)
if(n_q>2){yexpn=expression(gamma["j,h"]^x)} else {yexpn=expression(gamma["h"]^x)}
if (ylim_man_ind){
    plot(lag_values, betas_vec_q[,1], las=1, type="n", ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
} else {
    plot(lag_values, betas_vec_q[,1], las=1, type="n", ylim=c(-0.0002+min(0.00, min(ci_vec_q[,,1])) , 0.0002+max(0,max(ci_vec_q[,,2]))), xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
}
for (ii in 1:(n_q-1)){
    lines(lag_values,betas_vec_q[,ii], type="o", col=cl[ii], pch=markers[ii], lwd=linew_main); lines(lag_values, ci_vec_q[,ii,1], lty=2, col=cl[ii], lwd=linew_ci); lines(lag_values,ci_vec_q[,ii,2], lty=2, col=cl[ii], lwd=linew_ci);
}
if (selector_q == "q_cheat_rat"){
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2");
    if(n_q>2){ legend("topleft", lty=rep(1,n_q-1), lwd=linew_main, col=cl, pch=markers, legend=paste("j=(", as.character(c_q[1:(length(c_q)-2)]),",", as.character(c_q[2:(length(c_q)-1)]), ")",sep=""), cex=font_leg_val)}
} else {
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2");
    if(n_q>2){ legend("topleft", lty=rep(1,n_q-1), lwd=linew_main, col=cl, pch=markers, legend=paste("j=(", as.character(c_q[2:(length(c_q)-1)]),",", as.character(c_q[3:(length(c_q))]), ")",sep=""), cex=font_leg_val)}
}
dev.off()


# If there are also other variables of interest in the regression, do the same for those
if (ALLin_ind){
n_q_extra_vec = c(n_q,n_q)
c_name <- 1
for (varname in c("q_cheat_rat")){
    nqc <- n_q_extra_vec[c_name]
    betas_extra <- matrix(0,length(lag_values),nqc-1)
    ci_extra <- array(0,dim=c(length(lag_values),nqc-1,2))
    ctr <- 0
    for (lplus_capx_q in lag_values){
        ctr <- ctr +1
        eval(parse(text=paste("cur_mod<-allmodel_list$reg_tab_",as.character(lplus_capx_q),sep="")))
        eval(parse(text=paste("cur_mod_sum<-summary(allmodel_list$reg_tab_",as.character(lplus_capx_q),")",sep="")))
        reg_coefs_extra <- coef(cur_mod_sum)
        # Save coefs
        if (nqc < 2){
            betas_extra[ctr,1] <- reg_coefs_extra[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", sep=""))]
            ci_extra[ctr,1,] <- confint(cur_mod, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", sep="")),]
        } else {
        for (ii in 1:(nqc-1)){
            if (varname == "q_cheat_rat") {
                betas_extra[ctr,ii] <- reg_coefs_extra[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", as.character(ii), sep=""))]
                ci_extra[ctr,ii,] <- confint(cur_mod, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", as.character(ii), sep="")),]
            } else {
                betas_extra[ctr,ii] <- reg_coefs_extra[, "Estimate"][c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", as.character(ii+1), sep=""))]
                ci_extra[ctr,ii,] <- confint(cur_mod, level=ci_lev_set)[c(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",varname,", lagc1_capx_q)", as.character(ii+1), sep="")),]
            }
    } } }
    # Plot
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_extra_lab, "_betas_vec_q(", dep_var, ")(", varname, ")", str_end, ".pdf")), width = 8, height = 7)
    par(mar = c(5,5,2,2))
    cl<- rainbow(nqc, start=0.1)
    cl<-rev(cl[-1])
    if (col_change_ind & nqc == 2){cl[1] <- "blue"}
    if (col_change_ind & nqc > 2){cl[3] <- "sienna1"}
    if (paper_2line_ind & nqc > 2){linew_main = paper_2line_linew_main_set; linew_ci = paper_2line_linew_ci_set; confint_ind = TRUE; cl[2] <- "sienna1"}
    markers <- rep(c(4, 19), length=nqc-1)
    if(nqc>2){yexpn=expression(gamma["j,h"]^x)} else {yexpn=expression(gamma["h"]^x)}
    if (nqc<2){
        if (ylim_man_ind){
            plot(lag_values, betas_extra[,1], las=1, type="n",, ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
        } else {
            plot(lag_values, betas_extra[,1], las=0, type="n",, ylim=c(-0.0002+min(0.00, min(ci_extra[,,1])) , 0.0002+max(0,max(ci_extra[,,2]))), xlab = "h", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
        }
        lines(lag_values,betas_extra[,1], type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_extra[,1,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_extra[,1,2], lty=2, col=cl[1], lwd=linew_ci);
        abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2")
    } else {
        if (ylim_man_ind){
            plot(lag_values, betas_extra[,1], las=1, type="n", ylim=ylim_man, xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
        } else{
            plot(lag_values, betas_extra[,1], las=1, type="n", ylim=c(-0.0002+min(0.00, min(ci_extra[,,1])) , 0.0002+max(0,max(ci_extra[,,2]))), xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
        }
        for (ii in 1:(nqc-1)){
            lines(lag_values,betas_extra[,ii], type="o", col=cl[ii], pch=markers[ii], lwd=linew_main); lines(lag_values, ci_extra[,ii,1], lty=2, col=cl[ii], lwd=linew_ci); lines(lag_values,ci_extra[,ii,2], lty=2, col=cl[ii], lwd=linew_ci);
        }
        if (varname == "q_cheat_rat"){
            abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2");
            if (n_q>2){ legend("topleft", lty=rep(1,n_q-1), lwd=linew_main, col=cl, pch=markers, legend=paste("j=(", as.character(c_q[1:(length(c_q)-2)]),",", as.character(c_q[2:(length(c_q)-1)]), ")",sep=""), cex=font_leg_val) }
        } else {
            abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2");
            if (n_q>2){ legend("topleft", lty=rep(1,n_q-1), lwd=linew_main, col=cl, pch=markers, legend=paste("j=(", as.character(c_q[2:(length(c_q)-1)]),",", as.character(c_q[3:(length(c_q)-0)]), ")",sep=""), cex=font_leg_val) }
        }
    }
    dev.off()

    c_name <- c_name+1
}
}

} # End of LPq loop





#########
# LP estimation of "level effects"
#########

# Specifiers, same for all specs
fe_ind  = TRUE          # Requesting firm FEs, used in all specifications
dep_var = "lk_stock"
ylim_man_ind = TRUE     # Manual y-bounds on figures
lag_VAR_nct = 1          # Max lag of aggregate controls (Y)
cumsum_resp_ind = FALSE # Instead of cumulative difference, do cumulative sum of LHS variable
win_val_q  = 0.01       # Outcome variable outlier trimming bound
ci_lev_set = 0.95
selector_q = "qcs_cheat_rat"
fig_main_lab    = "FigA14a"
fig_lev_lab     = "FigA14b"
shock_name_ext  = ""        # Use baseline shock
ci_true_ind     = TRUE      # Compute correct confidence intervals for sums of parameters, based on delta method
plot_agg_ci_ind = TRUE      # Plot confidence interval for implied aggregate K response
ylim_man_main   = c(-2,1.5)   # y-axis bounds on main group-level difference plot (A.14a)
ylim_man_agg    = c(-1.6,0.9)   # y-axis bounds on level effects plot (A.14a)

print("###")
print(paste0("Running LP specification for level effects."))

# Lag extra back
lplus_capx_q = 0
# Loop
lag_values <- c(0:20)
model_list <- list()
# Set up collector matrices for point estimates and confidence bounds
betas_vec_q <- matrix(0,length(lag_values),n_q-1)
ci_vec_q <- array(0,dim=c(length(lag_values),n_q-1,2))
# Also collect level effect (delta) on base group
deltas_vec_q <- matrix(0,length(lag_values))
deltas_ci_vec_q <- array(0,dim=c(length(lag_values),2))
# And compute and collect the true confidence intervals on the sums of the parameters
levbetas_vec_q <- matrix(0,length(lag_values),n_q-1)
levci_vec_q <- array(0,dim=c(length(lag_values),n_q-1,2))
# And compute and collect the true confidence intervals on the mean of the sums of the parameters (the aggregate K response)
agg_deltas_comp <- matrix(0,length(lag_values))
agg_deltas_ci_comp <- array(0,dim=c(length(lag_values),2))

# Specification parameters
str_end = ""

# Create explicit dummies with appropriate "levels" for the quantiles, using arbitrary outcome variable, that is never NA
dummy_mat <- model.matrix(formula(paste("k_stock ~ atq + ", selector_q, sep="")), data=p_data_capx_q)
if (selector_q == 'q_cheat_rat' | selector_q == 'qcs_cheat_rat') {
    p_data_capx_q[,c(paste(selector_q, as.character(1:(n_q-1)), sep=""))] = dummy_mat[,c(paste(selector_q, as.character(1:(n_q-1)), sep=""))]
} else {
    p_data_capx_q[,c(paste(selector_q, as.character(2:n_q), sep=""))] = NA
    p_data_capx_q[!is.na(p_data_capx_q[,selector_q]),c(paste(selector_q, as.character(2:n_q), sep=""))] = dummy_mat[,c(paste(selector_q, as.character(2:n_q), sep=""))]
}

# Add correct string at end of filenames
if (fe_ind) {str_end <- paste(str_end,"_FE", sep="")}

### Start loop
ctr <- 0
cat('h=')
for (lplus_capx_q in lag_values){
ctr <- ctr +1
cat(paste0(lplus_capx_q,"..."))
lag0_capx_q=0 + lplus_capx_q # first lag of dffr
lag1_capx_q=0 + lplus_capx_q # last lag of dffr
lagc0_capx_q=0 + lplus_capx_q # lag of earlier controls
lagc1_capx_q=1 + lplus_capx_q # lag of later controls

# Compute the dependent variable's average yearly growth rate
agg_diff_series <- p_data_capx_q["dylk_stock"]
colnames(agg_diff_series) <- "agg_diff_dep_var"
p_data_capx_q$agg_diff_dep_var <- agg_diff_series$agg_diff_dep_var
agg_mdiffs <- ddply(p_data_capx_q, c("cqtr_num"), summarize, mdiff = mean(agg_diff_dep_var,na.rm=TRUE))
rownames(agg_mdiffs) <- agg_mdiffs$cqtr_num
# Add to main dataframe
p_data_capx_q$agg_mdiff_dep_var <- agg_mdiffs[as.character(p_data_capx_q$cqtr_num),"mdiff"]

# Aggregate controls (Y), non-cross-terms
VAR_vars = c("rlag(ffr, lagc1_capx_q:(lagc1_capx_q+lag_VAR_nct-1))", "rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q+lag_VAR_nct-1))", "rlag(agg_mdiff_dep_var, lagc1_capx_q:(lagc1_capx_q+lag_VAR_nct-1))")
# Aggregate controls (Y), cross-terms
VAR_vars_ct = c("rlag(ffr, lagc1_capx_q:(lagc1_capx_q+lag_VAR_nct-1))", "rlag(dlgdp_real, lagc1_capx_q:(lagc1_capx_q+lag_VAR_nct-1))")

### Cross-terms
controls_ct_capx_q =c(paste("rlag(",selector_q, as.character(1:(n_q-1)),", lagc1_capx_q + 0)", sep=""))

### Non-cross-terms
# FCpp (no DD) + VAR_ct
controls_nct_capx_q =c(paste("rlag(",selector_q,", lagc1_capx_q + 0)", sep=""), "rlag(latq_tdm, lagc1_capx_q)", "rlag(sffr, lag0_capx_q:lag1_capx_q)", VAR_vars, "rlag(lev_rat, lagc1_capx_q)", "rlag(dylsaleq_real, lagc1_capx_q)", paste("(rlag(",selector_q,", lagc1_capx_q)):(", paste(VAR_vars_ct, collapse=" + "),")",sep=""))

# Create differences of outcome variable
diff_series <- p_data_capx_q[,dep_var] - rlag(p_data_capx_q[,dep_var], lagc1_capx_q)
p_data_capx_q$diff_dep_var <- diff_series

# And trim data by quarter based on this diff_series
trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
p_data_capx_q <- ddply(p_data_capx_q, c("cqtr_num"), transform, keep_diff_dep_var = trim_keep_fn_q(diff_dep_var))
p_data_capx_q[is.na(p_data_capx_q$keep_diff_dep_var),'keep_diff_dep_var'] <- TRUE
p_data_capx_q[!p_data_capx_q$keep_diff_dep_var,'diff_dep_var'] <- NA
# Reorder data
p_data_capx_q <- p_data_capx_q[order(p_data_capx_q$gvkey, p_data_capx_q$cqtr_num), ]
p_data_capx_q <- pdata.frame(p_data_capx_q, index=c("gvkey", "cqtr_num"))
# Drop outliers
p_data_capx_q_used <- p_data_capx_q

### Control dummies
# No industry FE (because add firm FE below)
dummies_capx_q = c("")
# If requested, add firm fixed effects
if (fe_ind) {
    dummies_capx_q = c(dummies_capx_q, "gvkey")
} else {dummies_capx_q = c(dummies_capx_q, "0")}

# Paste together regression formula, with clustering at industry*time and firm levels
formla_capx_q <- formula(paste("diff_dep_var ~ rlag(sffr, lag0_capx_q:lag1_capx_q):(", paste(controls_ct_capx_q, collapse=" + "), ") + ", paste(controls_nct_capx_q, collapse=" + "), "  | ", paste(dummies_capx_q, collapse=" + "), "|0 | sic3_datacqtr + gvkey" ))

# Estimate
reg_capx_q <- felm(formla_capx_q, data=p_data_capx_q_used)
reg_sum_capx_q <- summary(reg_capx_q)

reg_coefs_capx_q <- coef(reg_sum_capx_q)
# Save coefs
for (ii in 1:(n_q-1)){
    if (selector_q == 'q_cheat_rat'| selector_q == 'qcs_cheat_rat'){
        betas_vec_q[ctr,ii] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep=""))]
        ci_vec_q[ctr,ii,] <- confint(reg_capx_q)[c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep="")),]
    } else {
        betas_vec_q[ctr,ii] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep=""))]
        ci_vec_q[ctr,ii,] <- confint(reg_capx_q)[c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep="")),]
    }
}

# Coefs for level effect
deltas_vec_q[ctr] <- reg_coefs_capx_q[, "Estimate"][c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q)"))]
deltas_ci_vec_q[ctr,] <- confint(reg_capx_q)[c(paste("rlag(sffr, lag0_capx_q:lag1_capx_q)")),]

# Compute and collect correct confidence intervals of sums of parameters
# Level effects on qs_cheat_rat groups
if (ci_true_ind){
    crit_val = 1.96
    if (ci_lev_set==0.90){crit_val = 1.645}
    VCV_cur = reg_capx_q$clustervcv
    nK = dim(VCV_cur)[1]
    lev_ind <- match(paste("rlag(sffr, lag0_capx_q:lag1_capx_q)", sep=""), row.names(VCV_cur))
    for (ii in 1:(n_q-1)){
        sel_vec <- numeric(nK)
        if (selector_q == 'qcs_cheat_rat'){
            yb_ind  <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep=""), row.names(VCV_cur))
        } else {
            yb_ind  <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep=""), row.names(VCV_cur))
        }
        sel_vec[c(lev_ind,yb_ind)] <- c(1,1)
        SE_sum <- sqrt(sel_vec %*% VCV_cur %*% sel_vec)
        # Save
        levbetas_vec_q[ctr,ii] <- betas_vec_q[ctr,ii] + deltas_vec_q[ctr]
        levci_vec_q[ctr,ii,] <- levbetas_vec_q[ctr,ii] + c(-1,1) * crit_val * SE_sum
    }
}
# Do the same for level effect of mean of qs_cheat_rat groups' level responses (proxy for aggregate K response)
if (ci_true_ind){
    crit_val = 1.96
    if (ci_lev_set==0.90){crit_val = 1.645}
    VCV_cur = reg_capx_q$clustervcv
    nK = dim(VCV_cur)[1]
    # Set up selector vector
    sel_vec <- numeric(nK)
    # Weight "1" on baseline level delta (as it appears in all groups' level responses)
    lev_ind <- match(paste("rlag(sffr, lag0_capx_q:lag1_capx_q)", sep=""), row.names(VCV_cur))
    sel_vec[lev_ind] <- 1
    # Weight "1/n_q" on groups' responses
    for (ii in 1:(n_q-1)){
        if (selector_q == 'qcs_cheat_rat'){
            yb_ind  <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii),", lagc1_capx_q + 0)", sep=""), row.names(VCV_cur))
        } else {
            yb_ind  <- match(paste("rlag(sffr",shock_name_ext,", lag0_capx_q:lag1_capx_q):rlag(",selector_q, as.character(ii+1),", lagc1_capx_q + 0)", sep=""), row.names(VCV_cur))
        }
        sel_vec[yb_ind] <- 1/n_q
    }
    # Standard error of sum
    SE_sum <- sqrt(sel_vec %*% VCV_cur %*% sel_vec)
    # Save sum (level effect) and its confidence interval
    agg_deltas_comp[ctr] <- as.double(sel_vec %*% coef(reg_sum_capx_q)[,1])
    agg_deltas_ci_comp[ctr,] <- agg_deltas_comp[ctr] + c(-1,1) * crit_val * SE_sum
}

} # End of estimations loop


# Plot results for main variable group differences
pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_main_lab, "_tot_betas_vec_q(", dep_var, ")(", selector_q, ")", str_end, ".pdf")), width = 8, height = 7)
par(mar = c(5,5.5,2,2))
cl<- rainbow(n_q, start=0.1)
cl<-rev(cl[-1])
if (col_change_ind){cl[3] <- "sienna1"}
if (paper_2line_ind){linew_main = paper_2line_linew_main_set; linew_ci = paper_2line_linew_ci_set; confint_ind = TRUE; cl[2] <- "sienna1"}
markers <- rep(c(4, 19), length=n_q-1)
if (ylim_man_ind){
    plot(lag_values, betas_vec_q[,1], type="n", las=1, ylim=ylim_man_main, xlab = "Quarters (h)", ylab="", cex.lab=font_lab_val, cex.axis=font_ax_val)
    mtext("Percent", side = 2, line = 4, cex=font_lab_val)
} else {
    plot(lag_values, betas_vec_q[,1], type="n", las=1, ylim=c(-0.0002+min(0.00, min(ci_vec_q[,,1])) , 0.0002+max(0,max(ci_vec_q[,,2]))), xlab = "Quarters (h)", ylab="Percent", cex.lab=font_lab_val, cex.axis=font_ax_val)
}
for (ii in 1:(n_q-1)){
    lines(lag_values,betas_vec_q[,ii], type="o", col=cl[ii], pch=markers[ii], lwd=linew_main); lines(lag_values, ci_vec_q[,ii,1], lty=2, col=cl[ii], lwd=linew_ci); lines(lag_values,ci_vec_q[,ii,2], lty=2, col=cl[ii], lwd=linew_ci);
}
if (selector_q == "q_cheat_rat"| selector_q == 'qcs_cheat_rat'){
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2"); legend("bottomleft", lty=rep(1,n_q-1), lwd=linew_main, col=cl, pch=markers, legend=paste("j=(", as.character(c_q[1:(length(c_q)-2)]),",", as.character(c_q[2:(length(c_q)-1)]), ")",sep=""), cex=font_leg_val)
} else {
    abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2"); legend("bottomleft", lty=rep(1,n_q-1), col=cl, pch=markers, legend=paste("j=(", as.character(c_q[2:(length(c_q)-1)]),",", as.character(c_q[3:(length(c_q))]), ")",sep=""), cex=font_leg_val)
}
dev.off()

if (TRUE){
    # Plot the level and betas together
    pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_LP", paste0(fig_lev_lab, "_tot_levbetas_vec_q_wagg(", dep_var, ")(", selector_q, ")", str_end, "_2l.pdf")), width = 8, height = 7)
    par(mar = c(5,5.5,2,2))
    cl<- rainbow(n_q, start=0.1)
    cl<-rev(cl[-1])
    linew_main = paper_2line_linew_main_set; linew_ci = paper_2line_linew_ci_set; confint_ind = TRUE; cl[2] <- "sienna1"
    markers <- rep(c(4, 19), length=n_q-1)
    alpha_set = 0.2 # Opacity level of see-through lines
    if (selector_q == "qcs_cheat_rat"){
        plot(lag_values, levbetas_vec_q[,1], type="n", las=1, ylim=ylim_man_agg, xlab = "Quarters (h)", ylab="", cex.lab=font_lab_val, cex.axis=font_ax_val)
        mtext("Percent", side = 2, line = 4, cex=font_lab_val)
        lines(lag_values,levbetas_vec_q[,1], type="o", col=rgb(t(col2rgb("blue")/255), alpha=alpha_set), pch=markers[1], lwd=linew_main); lines(lag_values, levci_vec_q[,1,1], lty=2, col=rgb(t(col2rgb("blue")/255), alpha=alpha_set), lwd=linew_ci); lines(lag_values,levci_vec_q[,1,2], lty=2, col=rgb(t(col2rgb("blue")/255), alpha=alpha_set), lwd=linew_ci);
        lines(lag_values,deltas_vec_q, col=rgb(t(col2rgb("darkgreen")/255), alpha=alpha_set), type="o", pch=6, lwd=linew_main); lines(lag_values,deltas_ci_vec_q[,1], lty=2, col =rgb(t(col2rgb("darkgreen")/255), alpha=alpha_set), lwd=linew_ci); lines(lag_values,deltas_ci_vec_q[,2], lty=2, col =rgb(t(col2rgb("darkgreen")/255), alpha=alpha_set), lwd=linew_ci)
        lines(lag_values,agg_deltas_comp, col = "black", type="o", pch=15, lwd=linew_main+1.0);
        if (plot_agg_ci_ind){
            lines(lag_values,agg_deltas_ci_comp[,1], lty=2, col ="black", lwd=linew_ci)
            lines(lag_values,agg_deltas_ci_comp[,2], lty=2, col ="black", lwd=linew_ci)
        }
        # No VAR IRF line
        abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2"); legend("bottomleft", lty=rep(1,1), lwd=c(linew_main,linew_main,linew_main+1.0), col=c("blue","darkgreen","black"), pch=c(markers[1],6,15), legend=c(paste("j=(", as.character(c_q[1]),",", as.character(c_q[2]), ")",sep=""), paste("j=(",as.character(c_q[n_q]),",1.0)",sep=""), "Aggregate K"), cex=font_leg_val)
    } else {
        plot(lag_values, levbetas_vec_q[,(n_q-1)], type="n", ylim=c(-0.0002+min(0.00, min(levci_vec_q[,n_q-1,1])) , 0.0002+max(0,max(levci_vec_q[,n_q-1,2]))), xlab = expression(h), ylab="Percent")
        lines(lag_values, levbetas_vec_q[,(n_q-1)], type="o", col=cl[(n_q-1)], pch=markers[(n_q-1)], lwd=linew_main); lines(lag_values, levci_vec_q[,(n_q-1),1], lty=2, col=cl[(n_q-1)], lwd=linew_ci); lines(lag_values,levci_vec_q[,(n_q-1),2], lty=2, col=cl[(n_q-1)], lwd=linew_ci);
        lines(lag_values,deltas_vec_q, col = "black", type="o", pch=6, lwd=linew_main);  lines(lag_values,deltas_ci_vec_q[,1], lty=2, col = "black", lwd=linew_ci); lines(lag_values,deltas_ci_vec_q[,2], lty=2, col = "black", lwd=linew_ci)
        abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2"); legend("bottomleft", lty=rep(1,1), col=c(cl[n_q-1],"black"), pch=c(markers[n_q-1],6), legend=c(paste("j=(", as.character(c_q[(length(c_q)-1)]),",", as.character(c_q[(length(c_q))]), ")",sep=""), paste("j=(0.0,",as.character(c_q[2]),")",sep="")), cex=font_leg_val)
}
dev.off()
}

