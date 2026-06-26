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

out_dir <- file.path(scenario_dir, "analysis", "aggregated_gam_curves")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("==========================================================\n")
cat(paste(" STARTING AGGREGATED GAM ANALYSIS FOR", toupper(scenario_name), "\n"))
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
metrics <- c("norm_mean_cluster_size", "mean_aspect_ratio", "fractal_dimension", 
             "morans_i", "betti_0", "betti_1", "betti_2", 
             "topological_conduction_eff", "min_path_x")

metric_labels <- list(
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
# AGGREGATION, GAM MODELING & PLOTTING
# ==========================================================
report_file <- file.path(out_dir, "aggregated_gam_report.txt")
sink(report_file)
cat("==========================================================\n")
cat(" AGGREGATED STRUCTURAL GAM ANALYSIS REPORT\n")
cat(paste(" SCENARIO:", scenario_name, "\n"))
cat("==========================================================\n")
cat("Models fitted using quasibinomial GAM on data aggregated purely\n")
cat("by Density, creating a universal, phenotype-agnostic risk map.\n")
cat("The predictor is the MEAN of the metric across all geometries,\n")
cat("and the outcome is the EMPIRICAL PROBABILITY of sustained reentry.\n\n")

plot_list <- list()

for (m in metrics) {
  if (!(m %in% colnames(data))) next
  
  df_m <- data[!is.na(data[[m]]) & !is.na(data$sustained_reentry), ]
  if (nrow(df_m) < 50 || var(df_m[[m]]) == 0) next
  
  df_agg <- df_m %>%
    group_by(density) %>%
    summarise(
      mean_metric = mean(.data[[m]], na.rm = TRUE),
      prob_reentry = mean(sustained_reentry, na.rm = TRUE),
      successes = sum(sustained_reentry == 1, na.rm = TRUE),
      trials = n(),
      .groups = 'drop'
    ) %>%
    mutate(failures = trials - successes)
  
  df_agg <- df_agg %>% filter(trials >= 5)
  
  cat(paste("\n>> Evaluating Metric:", metric_labels[[m]], "(", m, ")\n"))
  
  form <- as.formula("cbind(successes, failures) ~ s(mean_metric, bs='cs', k=5)")
  
  model <- NULL
  tryCatch({
    model <- gam(form, family = quasibinomial(link = "logit"), data = df_agg, method = "REML", optimizer = c("outer", "bfgs"))
  }, error = function(e) {
    cat(sprintf("   Failed to fit model for %s: %s\n", m, e$message))
  })
  
  if (is.null(model)) next
  
  s_sum <- summary(model)
  edf <- s_sum$edf[1]
  p_val <- s_sum$s.table[1, "p-value"]
  dev_exp <- s_sum$dev.expl * 100
  
  sig_star <- ifelse(p_val < 0.001, "***", ifelse(p_val < 0.01, "**", ifelse(p_val < 0.05, "*", "ns")))
  cat(sprintf("   P-value            : %.2e [%s]\n", p_val, sig_star))
  cat(sprintf("   Deviance Explained : %.2f%%\n", dev_exp))
  cat(sprintf("   Non-linearity (EDF): %.2f (1 = Linear, >1 = Non-linear)\n", edf))
  
  # Geração da Curva Suavizada (Predição)
  grid <- data.frame(mean_metric = seq(min(df_agg$mean_metric), max(df_agg$mean_metric), length.out = 300))
  pred <- predict(model, newdata = grid, type = "link", se.fit = TRUE)
  grid$prob <- model$family$linkinv(pred$fit)
  grid$ci_lower <- model$family$linkinv(pred$fit - 1.96 * pred$se.fit)
  grid$ci_upper <- model$family$linkinv(pred$fit + 1.96 * pred$se.fit)
  
  plot_title <- metric_labels[[m]]
  
  p <- ggplot() +
    geom_ribbon(data = grid, aes(x = mean_metric, ymin = ci_lower, ymax = ci_upper), fill = "#d62728", alpha = 0.2) +
    geom_line(data = grid, aes(x = mean_metric, y = prob), color = "#d62728", linewidth = 1.5) +
    geom_point(data = df_agg, aes(x = mean_metric, y = prob_reentry), color = "#222222", size = 4, alpha = 0.8) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, min(max(grid$ci_upper)*1.1, 1.0))) +
    labs(title = plot_title, 
         subtitle = sprintf("Deviance Explained: %.1f%% | p-value: %.2e", dev_exp, p_val),
         x = paste("Mean", metric_labels[[m]]), 
         y = "Aggregated Prob. of Reentry") +
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
  
  safe_name <- paste0("gam_aggregated_", m, ".png")
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
      title = paste(scenario_name, "- Universal Aggregated Structural Risk Overview"),
      subtitle = "Points represent the universal mean metric vs empirical probability of reentry per density step.",
      theme = theme(
        plot.title = element_text(size = 28, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 18, hjust = 0.5, color="grey30")
      )
    )
    
  safe_name_combined <- "gam_aggregated_combined.png"
  ggsave(file.path(out_dir, safe_name_combined), plot = combined_plot, width = 24, height = 18, dpi = 300, bg = "white")
}

cat(paste("\nSUCCESS! Analysis complete. Reports and plots saved to:", out_dir, "\n"))