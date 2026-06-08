# R script for working with model-generated data
# MICRO-REGRESSIONS

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

# Define lag function that behaves consistently with earlier versions of R
rlag <- function(x, k=1, shift="row", ...) {
  lag(x, k, shift, ...)
}

# Fix some model parameters
modl_Pi     = 1.02^(1.0/4)
modl_gamma  = 1-1/16
gammatil_ss = modl_gamma/modl_Pi
modl_delta  = 0.025
options(warn=-1)

# Reading data
data <- read.csv(file.path(PROJECT_ROOT, "interim_output", "model_interim_output", "df_model_panel.csv"))

# Construct new variables
data$atq <- data$k_stock + data$cheq
data$latq <- log(data$atq)
data$lev_rat <- data$dttq/data$atq
data$Nlev_rat <- (data$dttq-data$cheq)/data$atq
# Cash-to-assets
data$cheat_rat <- data$cheq/data$atq
# Log capital
data$lk_stock <- log(data$k_stock)

# RECREATE panel dataframe
p_data <- pdata.frame(data, index=c("gvkey", "cqtr_num"))
p_data$cqtr_num <- as.double(as.character(p_data$cqtr_num))

# Also create debt issuance variables used below
# Net LT debt issuance
p_data['ndltisq']     = p_data$dttq - rlag(p_data$dttq,1)
# Gross LT debt issuance
p_data['dltisq']      = p_data$dttq - gammatil_ss*rlag(p_data$dttq,1)

# IMPORTANT: For any "flow" variables, shift them "forward" by one, so we can interpret the "stock" variables as measured at the "end" of the period, like in Compustat, without shifting all the stock variables
p_data$saleq = rlag(p_data$saleq,1)
p_data$dvq = rlag(p_data$dvq,1)
p_data$Ba_mod = rlag(p_data$Ba_mod,1)
p_data$Ba_prob_mod = rlag(p_data$Ba_prob_mod,1)
# Further adjustments
p_data$lsaleq = log(p_data$saleq)
p_data$dlsaleq = p_data$lsaleq - rlag(p_data$lsaleq,1)
p_data$dlk_stock = p_data$lk_stock - rlag(p_data$lk_stock,1)
p_data$dylsaleq = p_data$lsaleq - rlag(p_data$lsaleq,4)
p_data$dylk_stock = p_data$lk_stock - rlag(p_data$lk_stock,4)

# "Clean lines" for debt issuance lumpiness analysis in the draft (April 2025)
if (TRUE){
    # Investment
    p_data["kinv"] = p_data$k_stock - rlag(p_data$k_stock,1) # Investment based on k_stock change
    p_data["inv"] = p_data$k_stock - (1-modl_delta)*rlag(p_data$k_stock,1) # Gross investment
    # Change in cash
    p_data["dche"] = p_data$cheq - rlag(p_data$cheq,1) # Investment based on k_stock change
    # Ratios
    p_data['dltisqat_rat'] = p_data$dltisq/rlag(p_data$atq,1)
    p_data['ndltisqat_rat'] = p_data$ndltisq/rlag(p_data$atq,1)
    p_data['kinvat_rat'] = p_data$kinv/rlag(p_data$atq,1)
    p_data['invat_rat'] = p_data$inv/rlag(p_data$atq,1)
    p_data['dcheat_rat'] = p_data$dche/rlag(p_data$atq,1)
    p_data['dvqat_rat'] = p_data$dvq/rlag(p_data$atq,1)
    p_data['datqat_rat'] = (p_data$atq-rlag(p_data$atq,1))/rlag(p_data$atq,1)

    ###
    # SECTION 2.2 -- LUMPINESS
    ###
    # Create indicators of "action"
    cutval_here = 0.01
    p_data$diss_ind <- p_data$dltisqat_rat > cutval_here    # Gross issuance based issuance
    p_data$ndadj_ind <- p_data$ndltisqat_rat > cutval_here  # Net issuance based issuance
    p_data$kadj_ind <- p_data$kinvat_rat > cutval_here      # Net investment based investment
    p_data$inv_ind <- p_data$invat_rat > cutval_here        # Gross investment based investment

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
    # One-sided
    win_val_q = 0.01
    trim_keep_fn_q <- function(x) x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    # Gross issuances
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dltisqat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dltisqat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(invat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("invat_rat")] <- NA
    # Reorder data
    p_data <- p_data[order(p_data$gvkey, p_data$cqtr_num), ]
    p_data <- pdata.frame(p_data, index=c("gvkey", "cqtr_num"))
    
    ###
    # SECTION 2.3 -- DEBT ISSUANCE AND LIQUIDITY
    ###

    library(lfe)
    library(xtable)

    # First, analysis of predicting debt issuance
    # Drop outliers of variables to be used in debt issuance prediction regressions
    # Two-sided
    win_val_q = 0.005
    trim_keep_fn_q <- function(x) x >= quantile(x,win_val_q,na.rm=TRUE) & x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylsaleq))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dylsaleq")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(lk_stock))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("lk_stock")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(dylk_stock))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("dylk_stock")] <- NA
    # One-sided
    win_val_q = 0.01
    trim_keep_fn_q <- function(x) x <= quantile(x,1.0-win_val_q,na.rm=TRUE)
    # Gross issuances
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(cheat_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("cheat_rat")] <- NA
    p_data <- ddply(p_data, c("cqtr_num"), transform, keep_temp_var = trim_keep_fn_q(lev_rat))
    p_data[!p_data$keep_temp_var | is.na(p_data$keep_temp_var), c("lev_rat")] <- NA
    # Reorder data
    p_data <- p_data[order(p_data$gvkey, p_data$cqtr_num), ]
    p_data <- pdata.frame(p_data, index=c("gvkey", "cqtr_num"))

    # Debt issuance prediction regressions ("replicate" Table 1)
    # LT debt issuance lead
    p_data$diss_ind <- as.numeric(p_data$diss_ind)
    p_data['diss_ind_lead4'] <- rlag(p_data$diss_ind,-4)
    p_data['diss_ind_lead1'] <- rlag(p_data$diss_ind,-1)
    p_data['diss_ind_lead8'] <- rlag(p_data$diss_ind,-8)
    # Run regression
    pred_contrs_str = "1 + rlag(lev_rat,0) + rlag(cheat_rat,0) + rlag(diss_ind, 0) + rlag(dylsaleq,0) + rlag(lk_stock,0) + rlag(dylk_stock,1) + rlag(dvqat_rat, 0)"
    # Lead 4, no firm FEs
    formla_diss_lead4_noFE <- formula(paste("diss_ind_lead4 ~ ", pred_contrs_str, " | cqtr_num |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead4_noFE    <- felm(formla_diss_lead4_noFE, data = p_data, na.action="na.exclude")
    # Lead 4, with firm FEs
    formla_diss_lead4_FE <- formula(paste("diss_ind_lead4 ~ ", pred_contrs_str, " | cqtr_num + gvkey |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead4_FE    <- felm(formla_diss_lead4_FE, data = p_data, na.action="na.exclude")
    # Lead 1, no firm FEs
    formla_diss_lead1_noFE <- formula(paste("diss_ind_lead1 ~ ", pred_contrs_str, " | cqtr_num |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead1_noFE    <- felm(formla_diss_lead1_noFE, data = p_data, na.action="na.exclude")
    # Lead 1, with firm FEs
    formla_diss_lead1_FE <- formula(paste("diss_ind_lead1 ~ ", pred_contrs_str, " | cqtr_num + gvkey |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead1_FE    <- felm(formla_diss_lead1_FE, data = p_data, na.action="na.exclude")
    # Lead 8, no firm FEs
    formla_diss_lead8_noFE <- formula(paste("diss_ind_lead8 ~ ", pred_contrs_str, " | cqtr_num |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead8_noFE    <- felm(formla_diss_lead8_noFE, data = p_data, na.action="na.exclude")
    # Lead 1, with firm FEs
    formla_diss_lead8_FE <- formula(paste("diss_ind_lead8 ~ ", pred_contrs_str, " | cqtr_num + gvkey |0| cqtr_num + gvkey"), sep="")
    reg_diss_lead8_FE    <- felm(formla_diss_lead8_FE, data = p_data, na.action="na.exclude")

    # Put everything in regression table
    diss_model_list <- list()
    diss_model_list$diss_lead1_noFE <- reg_diss_lead1_noFE
    diss_model_list$diss_lead1_FE <- reg_diss_lead1_FE
    diss_model_list$diss_lead4_noFE <- reg_diss_lead4_noFE
    diss_model_list$diss_lead4_FE <- reg_diss_lead4_FE
    diss_model_list$diss_lead8_noFE <- reg_diss_lead8_noFE
    diss_model_list$diss_lead8_FE <- reg_diss_lead8_FE

    cov_labs <- c("Leverage", "Liquidity", "$D_{i,t}$", "$\\Delta_{3} \\log(\\text{Sales})$", "log($k_{i,t}$)", "$\\Delta_{3} \\log(k)$", "Div/Assets")
    tab_title <- "Debt issuance prediction regression estimates, extended (Model)"
    reg_tab <- capture.output(stargazer(diss_model_list, title=tab_title, covariate.labels=cov_labs, dep.var.labels=c("$D_{i,t+1}$", "$D_{i,t+4}$", "$D_{i,t+8}$", "$D_{i,t+4}$", "$D_{i,t+4}$", "$D_{i,t+4}$"), omit.stat=c("ser", "adj.rsq"), omit=c(9,10,11), model.numbers=FALSE, align=TRUE, no.space=TRUE, label=paste("mod_tab_reg_debtforec"), add.lines=list(c("Firm FE", '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}', '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}', '\\multicolumn{1}{c}{No}', '\\multicolumn{1}{c}{Yes}')), notes=NULL, omit.table.layout = "n", table.placement = "!htb"))
        post_tabular <- c("","\\vspace{4pt}",
        "\\begin{minipage}{\\textwidth}",
        "{\\footnotesize \\emph{Notes:} Estimates and standard errors (in parentheses) for coefficients in \\eqref{eq_Empirics_debtforec_dynamicssec}. $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01.",
        "Estimated on sample of 20,000 ``public'' firms, observed for 40 quarters, in model stationary distribution.",
        "Standard errors clustered two-way at firm and time levels.",
        "}",
        "\\end{minipage}","",
        "\\end{table}"
        )
    dir.create(file.path(PROJECT_ROOT, "output"), showWarnings = FALSE)
    dir.create(file.path(PROJECT_ROOT, "output", "tables"), showWarnings = FALSE)
    writeLines(c(reg_tab[-length(reg_tab)], post_tabular), file.path(PROJECT_ROOT, "output", "tables", "TabB3_mod_reg_tab_diss_pred_extd.tex"))


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
    mr2_controls_list = c("lev_rat", "cheat_rat", "diss_ind", "dylsaleq", "lk_stock", "dylk_stock", "dvqat_rat")
    mr2_lags_list = c("0", "0", "0", "0", "0", "1", "0")
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
    mr2_latex_code <- xtable(mr2_data_pp, caption = "Marginal $R^{2}$ in debt issuance regressions, percentage points (Model)", label = "mod_tab:mr2_data", align = "lccccccc", table.placement="ht")
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
        "{\\footnotesize \\emph{Notes:} Marginal $R^{2}$ for explanatory variables in \\eqref{eq_Empirics_debtforec_dynamicssec}, estimated on sample of 20,000 ``public'' firms, observed for 40 quarters, in model stationary distribution.",
        "Computed based on the difference of full model $R^{2}$ and the $R^{2}$ in a restricted model which drops the corresponding control.",
        "}",
        "\\end{minipage}", "",
        "\\end{table}"
    )
    writeLines(c(r2_tab[-length(r2_tab)], post_tabular_r2), file.path(PROJECT_ROOT, "output", "tables", "TabB4_mod_mr2_table.tex"))


    # Run event-study regressions and save results into .csv-files
    library(lfe)
    nlags_diss = 8
    nc_diss = 2*nlags_diss+1
    lag_values = c(-nlags_diss:nlags_diss)
    ylim_man_diss_ind = TRUE
    ylim_man_diss = 100*c(-0.030, 0.130)
    ylim_man_ndlt = 100*c(-0.030, 0.130)
    # Simultaneous controlling
    p_data$inv_ind <- as.numeric(p_data$inv_ind)

    # Start loop
    print("Running event-study regressions")
    for (dep_var in c("ndltisqat_rat", "kinvat_rat", "dcheat_rat")){
        print(paste0("Outcome variable: ", dep_var))

        # Collect all into a dataframe to save and share
        df_event = data.frame(lags = lag_values)

        for (ind_var in c("diss_ind", "ndltisqat_rat", "kinvat_rat")){
            print(paste0("Event indicator: ", ind_var))

            formla_dissinv <- formula(paste(dep_var, " ~ 1 + rlag(",ind_var,",-8:8) | cqtr_num + gvkey | 0 | cqtr_num +gvkey", sep=""))
            reg_dissinv    <- felm(formla_dissinv, data = p_data, na.action="na.exclude")
            betas_vec <- as.double(coef(reg_dissinv))
            ci_vec <- confint(reg_dissinv)

            # Save into dataframe
            if (ylim_man_diss_ind & (ind_var=="diss_ind" | ind_var=="inv_ind")){ 
                df_event[ind_var] = 100*betas_vec
            } else {
                df_event[ind_var] = betas_vec
            }
        }
        # Save the dataframe
        dir.create(file.path(PROJECT_ROOT, "interim_output"), showWarnings = FALSE)
        write.csv(df_event, file.path(PROJECT_ROOT, "interim_output", paste0("mod_dissreg_vals_dep(", dep_var, ").csv")))
    }

}

