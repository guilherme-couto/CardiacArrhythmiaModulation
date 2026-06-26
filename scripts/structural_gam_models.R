#!/usr/bin/env Rscript

# ==========================================================
# ENVIRONMENT CONFIGURATION
# ==========================================================
local_lib <- Sys.getenv("R_LIBS_USER")
dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(local_lib)

required_packages <- c("mgcv", "ggplot2", "dplyr", "scales", "patchwork")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# ==========================================================
# ARGS & SETUP
# ==========================================================
args = commandArgs(trailingOnly=TRUE)
if (length(args) < 1) stop("Error: Provide the path to the directory containing master_dataset.csv.")
batch_dir <- args[1]
dim_arg   <- ifelse(length(args) >= 2, args[2], "Unknown_Dim")
shape_arg <- ifelse(length(args) >= 3, args[3], "Unknown_Shape")

scenario_dir <- file.path(batch_dir, dim_arg, shape_arg)
if (!dir.exists(scenario_dir)) stop(paste("Error: Scenario directory not found at", scenario_dir))

scenario_name <- paste("Scenario", paste0("(", dim_arg, "-", shape_arg, ")"))
if(dim_arg == "2D" && shape_arg == "full") scenario_name <- "Scenario I"
if(dim_arg == "2D" && shape_arg == "ellipse") scenario_name <- "Scenario II"
if(dim_arg == "3D" && shape_arg == "full") scenario_name <- "Scenario III"
if(dim_arg == "3D" && shape_arg == "ellipse") scenario_name <- "Scenario IV"

file_path <- file.path(scenario_dir, "data", "master_dataset.csv")
if (!file.exists(file_path)) stop(paste("Error: master_dataset.csv not found at", file_path))

out_dir <- file.path(scenario_dir, "analysis", "structural_gam_curves")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("==========================================================\n")
cat(paste("STARTING CONTINUOUS GAM ANALYSIS FOR", toupper(scenario_name), "\n"))
cat("==========================================================\n")

# ==========================================================
# DATA CLEANING
# ==========================================================
data <- read.csv(file_path)

# Convert invalid numeric flags (-9999.0) to NA
data[data == -9999.0] <- NA

# Ensure target variable is numeric 0/1
data$sustained_reentry <- as.numeric(data$sustained_reentry)

# Define continuous metrics to evaluate
metrics <- c("density", "norm_mean_cluster_size", "mean_aspect_ratio", "fractal_dimension", 
             "morans_i", "betti_0", "betti_1", "betti_2", 
             "topological_conduction_eff", "min_path_x")

metric_labels <- list(
  "density" = "Fibrosis Density",
  "norm_mean_cluster_size" = "Norm. Mean Cluster Size",
  "mean_aspect_ratio" = "Mean Aspect Ratio",
  "fractal_dimension" = "Fractal Dimension",
  "morans_i" = "Moran's I",
  "betti_0" = "Betti 0",
  "betti_1" = "Betti 1",
  "betti_2" = "Betti 2",
  "topological_conduction_eff" = "TCE",
  "min_path_x" = "Shortest Perc. Path"
)

# ==========================================================
# GAM MODELING & PLOTTING
# ==========================================================
report_file <- file.path(out_dir, "structural_gam_report.txt")
sink(report_file)
cat("==========================================================\n")
cat(" CONTINUOUS STRUCTURAL GAM ANALYSIS REPORT\n")
cat(paste(" SCENARIO:", scenario_name, "\n"))
cat("==========================================================\n")
cat("Models fitted using binomial GAM (logit link) to evaluate the\n")
cat("non-linear influence of individual structural metrics on reentry.\n\n")

plot_list <- list()

for (m in metrics) {
  if (!(m %in% colnames(data))) next
  
  # Clean data for this specific metric
  df_m <- data[!is.na(data[[m]]) & !is.na(data$sustained_reentry), ]
  
  # Skip if too few valid samples or if the variable is a constant
  if (nrow(df_m) < 50 || var(df_m[[m]]) == 0) {
    cat(sprintf("[!] Skipping %s: Insufficient variance or valid samples.\n", m))
    next
  }
  
  cat(paste("\n>> Evaluating Metric:", metric_labels[[m]], "(", m, ")\n"))
  
  # Fit Univariate GAM: sustained_reentry ~ s(metric, k=5)
  # k=5 restricts the splines so it doesn't overfit to noise
  form <- as.formula(paste("sustained_reentry ~ s(", m, ", bs='cs', k=5)"))
  
  model <- NULL
  tryCatch({
    model <- gam(form, family = binomial(link = "logit"), data = df_m, method = "REML", optimizer = c("outer", "bfgs"))
  }, error = function(e) {
    cat(sprintf("   Failed to fit model for %s: %s\n", m, e$message))
  })
  
  if (is.null(model)) next
  
  # Extract statistics for the report
  s_sum <- summary(model)
  edf <- s_sum$edf[1] # Effective degrees of freedom (Complexity of the curve)
  p_val <- s_sum$s.table[1, "p-value"]
  dev_exp <- s_sum$dev.expl * 100 # Deviance explained (Pseudo R-squared)
  
  sig_star <- ifelse(p_val < 0.001, "***", ifelse(p_val < 0.01, "**", ifelse(p_val < 0.05, "*", "ns")))
  cat(sprintf("   P-value            : %.2e [%s]\n", p_val, sig_star))
  cat(sprintf("   Deviance Explained : %.2f%%\n", dev_exp))
  cat(sprintf("   Non-linearity (EDF): %.2f (1 = Linear, >1 = Non-linear)\n", edf))
  
  # Predict values for the smooth curve
  grid <- data.frame(val = seq(min(df_m[[m]]), max(df_m[[m]]), length.out = 300))
  colnames(grid) <- m
  
  pred <- predict(model, newdata = grid, type = "link", se.fit = TRUE)
  grid$prob <- model$family$linkinv(pred$fit)
  grid$ci_lower <- model$family$linkinv(pred$fit - 1.96 * pred$se.fit)
  grid$ci_upper <- model$family$linkinv(pred$fit + 1.96 * pred$se.fit)
  
  # Plot the continuous GAM curve
  plot_title <- paste(scenario_name, "- Reentry Risk vs", metric_labels[[m]])
  
  p <- ggplot(grid, aes(x = .data[[m]], y = prob)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), fill = "#d62728", alpha = 0.2) +
    geom_line(color = "#d62728", linewidth = 1.5) +
    # The 'rug' adds little ticks at the bottom showing where the real data points are
    geom_rug(data = df_m, aes(x = .data[[m]]), inherit.aes = FALSE, alpha = 0.05, color = "black") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, min(max(grid$ci_upper)*1.1, 1.0))) +
    labs(title = plot_title, 
         subtitle = sprintf("Deviance Explained: %.1f%% | p-value: %.2e", dev_exp, p_val),
         x = metric_labels[[m]], 
         y = "Probability of Sustained Reentry") +
    theme_minimal(base_family="sans", base_size=16) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      axis.text = element_text(size=14, color="black", face="bold"),
      axis.title = element_text(size=16, face="bold"),
      plot.title = element_text(size=18, face="bold"),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.8)
    )
  
  # Save plot
  safe_name <- paste0("gam_continuous_", m, ".png")
  ggsave(file.path(out_dir, safe_name), plot = p, width = 10, height = 7, dpi = 300, bg = "white")

  plot_list[[m]] <- p
}

sink()

# ==========================================================
# COMBINED PLOT GENERATION
# ==========================================================
if (length(plot_list) > 0) {
  cat("\n>> Generating Combined Summary Figure...\n")
  
  combined_plot <- wrap_plots(plot_list, ncol = 3) +
    plot_annotation(
      title = paste(scenario_name, "- Comprehensive Structural Risk Overview"),
      theme = theme(plot.title = element_text(size = 28, face = "bold", hjust = 0.5))
    )
    
  safe_name_combined <- "gam_continuous_combined.png"
  ggsave(file.path(out_dir, safe_name_combined), plot = combined_plot, width = 24, height = 18, dpi = 300, bg = "white")
}

cat(paste("\nSUCCESS! Analysis complete. Reports and plots saved to:", out_dir, "\n"))