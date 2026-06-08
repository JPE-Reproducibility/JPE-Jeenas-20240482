# Separate script to jointly plot event-study regression coefficients from data and model

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

# Parameters for figures
linew_main = 2.2; linew_ci = 1.6
font_ax_val = 1.2; font_leg_val = 1.15; font_lab_val = 1.4

# Plot event-study regressions
nlags_diss = 8
lag_values = c(-nlags_diss:nlags_diss)
ylim_man_diss_ind = TRUE
ylim_man_diss = 100*c(-0.023, 0.0605)
ylim_man_diss_appx = 100*c(-0.032, 0.0605)
ylim_man_inv  = 100*c(-0.010, 0.022)
ylim_man_diss_mod = 100*c(-0.030, 0.130)
font_multr = 1.5

# Start loop
print("Plotting event-study coefficients")
for (ind_var in c("diss_ind", "ndltisqat_rat", "kinvat_rat")){
    print(paste0("Event indicator: ", ind_var))

    # Structure filenames and plotting
    if (ind_var=="diss_ind"){
        plot_dataonly_ind = TRUE
        figlab_data = "Fig2"
        figlab_mod = "FigA2"
    } else if (ind_var=="ndltisqat_rat"){
        plot_dataonly_ind = FALSE
        figlab_mod = "FigA3"
    } else {
        plot_dataonly_ind = FALSE
        figlab_mod = "FigA4"
    }

    for (dep_var in c("ndltisqat_rat", "kinvat_rat", "dcheat_rat")){
        print(paste0("Outcome variable: ", dep_var))
        # Load empirical saved csv
        emp_df_event <- read.csv(file.path(PROJECT_ROOT, "interim_output", paste0("emp_dissreg_vals_ind(", ind_var, ")_dep(", dep_var, ").csv")))
        lag_values <- emp_df_event$lag
        betas_vec  <- emp_df_event$beta
        ci_vec     <- as.matrix(emp_df_event[, c("ci_lo", "ci_hi")])

        # Structure filenames
        if (dep_var=="ndltisqat_rat"){
            panlab = "a"
        } else if (dep_var=="kinvat_rat"){
            panlab = "b"
        } else {
            panlab = "c"
        }

        # Plot results for main variable (data only)
        if (plot_dataonly_ind){
            pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness", paste0(figlab_data, panlab, "_dissreg(", dep_var, ")(", ind_var, ").pdf")), width = 8, height = 7)
            par(mar = c(5,5.3,2,2+1), mgp=c(3.8,1,0), las=1)
            cl<- rainbow(1, start=0.7)
            cl <- c("blue")
            markers <- rep(c(4, 19), 1)
            if (ylim_man_diss_ind & (ind_var=="diss_ind")){
                    if (ind_var=="diss_ind"){
                        if (dep_var=="ndltisqat_rat"){
                            ylim_main = ylim_man_diss;
                        } else {
                            ylim_main = 0.35*ylim_man_diss;
                        }
                    } else {
                        ylim_main = ylim_man_inv;
                    }
                plot(lag_values, 100*betas_vec, las=1, type="n", ylim=ylim_main, xlab = "Quarters since issuance", ylab="Percentage points", cex.lab=font_multr*font_lab_val, cex.axis=font_multr*font_lab_val)
                lines(lag_values, 100*betas_vec, type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, 100*ci_vec[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,100*ci_vec[,2], lty=2, col=cl[1], lwd=linew_ci);
            } else {
                plot(lag_values, betas_vec, las=1, type="n", ylim=c(-0.0002+min(0.00, min(ci_vec[,1])) , 0.0002+max(0,max(ci_vec[,2]))), xlab = "Quarters since issuance", ylab="Percent", cex.lab=font_multr*font_lab_val, cex.axis=font_multr*font_lab_val)
                lines(lag_values, betas_vec, type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_vec[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_vec[,2], lty=2, col=cl[1], lwd=linew_ci);
            }
            abline(h=0.0); grid (NULL,NULL, lty = 6, col = "cornsilk2")
            dev.off()
        }

        # Also, plot comparison with model -- on SEPARATE y-axes, "aligning zeros"
        mod_df_event = read.csv(file.path(PROJECT_ROOT, "interim_output", paste0("mod_dissreg_vals_dep(", dep_var, ").csv")))
        mod_betas    = mod_df_event[,ind_var]

        # Plot results for main variable ALONGSIDE MODEL
        pdf(file.path(PROJECT_ROOT, "output", "figures", "empirics_lumpiness", paste0(figlab_mod, panlab, "_dissreg(", dep_var, ")(", ind_var, ")_wmod_2yax.pdf")), width = 8, height = 7)
        par(mar = c(5,5.3,2,2+3), mgp=c(3.8,1,0), las=1)
        cl<- rainbow(1, start=0.7)
        cl <- c("blue","red")
        markers <- rep(c(4, 19), 1)
        if (ylim_man_diss_ind & (ind_var=="diss_ind")){
            if (ind_var=="diss_ind"){
                if (dep_var=="ndltisqat_rat"){
                    ylim_main = ylim_man_diss_appx;
                } else {
                    ylim_main = 0.35*ylim_man_diss_appx;
                }
            } else {
                ylim_main = ylim_man_inv;
            }
            plot(lag_values, 100*betas_vec, las=1, type="n", ylim=ylim_main, xlab = "Quarters since issuance", ylab="Percentage points", cex.lab=font_multr*font_lab_val, cex.axis=font_multr*font_ax_val)
            lines(lag_values, 100*betas_vec, type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, 100*ci_vec[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,100*ci_vec[,2], lty=2, col=cl[1], lwd=linew_ci);
            # Model:
            ylim_alt = (diff(range(mod_betas)) / diff(range(100*betas_vec, 100*ci_vec))) * ylim_main
            par(new = TRUE)  # Overlay the plot
            plot(lag_values, mod_betas, type="n", axes = FALSE, xlab = "", ylab = "", ylim = ylim_alt);
            lines(lag_values, mod_betas, type="o", col=cl[2], pch=markers[2], lwd=linew_main);
            axis(side=4, at=pretty(ylim_alt), cex.axis = font_multr*font_ax_val)  # Add the secondary y-axis on the right
            mtext("Percentage points (Model)", side = 4, line = 4, cex = font_multr*font_lab_val, las=0)
            legend("topright", lty=rep(1,2), col=cl, pch=markers, legend=c("Data", "Model"), cex = font_multr*font_lab_val)
        } else {
            ylim_main = c(-0.065+min(0.00, min(ci_vec[,1])) , 0.065+max(0,max(ci_vec[,2])))
            plot(lag_values, betas_vec, las=1, type="n", ylim=ylim_main, xlab = "Quarters since issuance", ylab="Percent", cex.lab=font_multr*font_lab_val, cex.axis=font_multr*font_ax_val)
            lines(lag_values, betas_vec, type="o", col=cl[1], pch=markers[1], lwd=linew_main); lines(lag_values, ci_vec[,1], lty=2, col=cl[1], lwd=linew_ci); lines(lag_values,ci_vec[,2], lty=2, col=cl[1], lwd=linew_ci);
            # Model:
            ylim_alt = (diff(range(mod_betas)) / diff(range(betas_vec, ci_vec))) * ylim_main
            par(new = TRUE)  # Overlay the plot
            plot(lag_values, mod_betas, type="n", axes = FALSE, xlab = "", ylab = "", ylim = ylim_alt);
            lines(lag_values, mod_betas, type="o", col=cl[2], pch=markers[2], lwd=linew_main);
            axis(side=4, at=pretty(ylim_alt), cex.axis=font_multr*font_ax_val)  # Add the secondary y-axis on the right
            mtext("Percent (Model)", side = 4, line = 4, cex = font_multr*font_lab_val, las=0)
            legend("topright", lty=rep(1,2), col=cl, pch=markers, legend=c("Data", "Model"), cex = font_multr*font_lab_val)
        }
        abline(h=0.0, lty=2)
        dev.off()
    }
}
