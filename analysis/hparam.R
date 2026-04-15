library(ggplot2)
library(dplyr)
library(arrow)
library(tidyr)

data <- read_parquet("results/hparam_ablation.parquet")

# ===== MAIN PLOT: ODDS by Config and Direction =====
summary <- data %>%
  group_by(config, samples, direction) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Order configs by samples dynamically
config_order <- summary %>%
  select(config, samples) %>%
  distinct() %>%
  arrange(samples, config) %>%
  pull(config)

summary$config <- factor(summary$config, levels = config_order)

# Create config labels with sample counts
summary$config_label <- paste0(summary$config, "\n(", summary$samples, ")")
summary$config_label <- factor(summary$config_label, 
  levels = unique(summary$config_label[order(summary$samples, as.character(summary$config))]))

main_plot <- ggplot(summary, aes(x = config_label, y = mean_odds, fill = direction)) +
  geom_col(position = position_dodge(width = 0.8), alpha = 0.8) +
  geom_errorbar(
    aes(ymin = mean_odds - se, ymax = mean_odds + se),
    position = position_dodge(width = 0.8),
    width = 0.2
  ) +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
  labs(
    title = "Effect of Batch Size x Steps on DAS Localization",
    subtitle = "BabyLM-100M checkpoint, 3 seeds per config",
    x = "Configuration (samples)",
    y = "Max ODDS",
    fill = "Transfer\nDirection"
  )

ggsave("analysis/figures/hparam_main.pdf", main_plot, width = 14, height = 6)

# ===== SAMPLES AGGREGATED PLOT =====
samples_summary <- data %>%
  group_by(samples, direction) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

samples_plot <- ggplot(samples_summary, aes(x = factor(samples), y = mean_odds, fill = direction)) +
  geom_col(position = position_dodge(width = 0.8), alpha = 0.8) +
  geom_errorbar(
    aes(ymin = mean_odds - se, ymax = mean_odds + se),
    position = position_dodge(width = 0.8),
    width = 0.2
  ) +
  theme_bw(base_size = 12) +
  labs(
    title = "Effect of Training Samples on DAS Localization",
    subtitle = "Aggregated across all batch/step configurations",
    x = "Training Samples",
    y = "Max ODDS",
    fill = "Transfer\nDirection"
  )

ggsave("analysis/figures/hparam_samples.pdf", samples_plot, width = 10, height = 6)

# ===== ASYMMETRY PLOT =====
asymmetry <- data %>%
  filter(train != eval) %>%
  group_by(config, samples, direction, seed) %>%
  summarize(mean_odds = mean(max_odds), .groups = "drop") %>%
  pivot_wider(names_from = direction, values_from = mean_odds)

# Handle column names with arrow
colnames(asymmetry) <- gsub("→", "_to_", colnames(asymmetry))

if ("wh_to_topicalization" %in% colnames(asymmetry) && "topicalization_to_wh" %in% colnames(asymmetry)) {
  asymmetry$ratio <- asymmetry$wh_to_topicalization / (asymmetry$topicalization_to_wh + 0.01)
  
  asym_summary <- asymmetry %>%
    group_by(samples) %>%
    summarize(
      mean_ratio = mean(ratio, na.rm = TRUE),
      se = sd(ratio, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  asym_plot <- ggplot(asym_summary, aes(x = factor(samples), y = mean_ratio)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = mean_ratio - se, ymax = mean_ratio + se), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    theme_bw(base_size = 12) +
    labs(
      title = "Transfer Asymmetry by Training Samples",
      subtitle = "Ratio > 1 means Wh->Topic stronger than Topic->Wh",
      x = "Training Samples",
      y = "Asymmetry Ratio (Wh->Topic / Topic->Wh)"
    )
  
  ggsave("analysis/figures/hparam_asymmetry.pdf", asym_plot, width = 8, height = 5)
}

print("Hyperparameter plots saved to analysis/plots/")
