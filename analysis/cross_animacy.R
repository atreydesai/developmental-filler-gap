# Cross-Animacy Analysis: Visualizations and Statistical Tests
# Tests whether animacy matching provides a boost (Boguraev et al. 2025)

library(ggplot2)
library(dplyr)
library(arrow)
library(tidyr)
library(emmeans)

# Create output directory
# Create output directory (run this script from the project root directory)
output_dir <- "analysis/figures/cross_animacy"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Load data
data <- read_parquet("results/cross_animacy.parquet")

cat("Cross-Animacy Analysis\n")
cat("Total observations:", nrow(data), "\n")
cat("Checkpoints:", paste(sort(unique(data$tokens_M)), collapse=", "), "\n\n")

# =============================================================
# 1. ANIMACY BOOST VISUALIZATION
# =============================================================

boost_summary <- data %>%
  group_by(tokens_M, animacy_match) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

boost_plot <- ggplot(boost_summary, aes(x = tokens_M, y = mean_odds, color = animacy_match)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = mean_odds - se, ymax = mean_odds + se, fill = animacy_match), 
              alpha = 0.2, color = NA) +
  scale_x_log10(breaks = c(1, 2, 5, 10, 20, 50, 100),
                labels = c("1M", "2M", "5M", "10M", "20M", "50M", "100M")) +
  scale_color_manual(values = c("match" = "#2E7D32", "no_match" = "#C62828"),
                     labels = c("match" = "Animacy Match", "no_match" = "Animacy Mismatch")) +
  scale_fill_manual(values = c("match" = "#2E7D32", "no_match" = "#C62828"),
                    labels = c("match" = "Animacy Match", "no_match" = "Animacy Mismatch")) +
  theme_bw(base_size = 12) +
  theme(legend.position = c(0.2, 0.8), 
        legend.background = element_rect(fill = alpha("white", 0.8))) +
  labs(
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Condition",
    fill = "Condition"
  )

ggsave(file.path(output_dir, "animacy_boost.pdf"), boost_plot, width = 8, height = 5)
cat("Saved: animacy_boost.pdf\n")

# =============================================================
# 2. BOOST BY CONSTRUCTION TYPE
# =============================================================

boost_by_construction <- data %>%
  filter(train_construction == eval_construction) %>%  # Within-construction only
  group_by(tokens_M, train_construction, animacy_match) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

construction_plot <- ggplot(boost_by_construction, 
                            aes(x = tokens_M, y = mean_odds, color = animacy_match, linetype = train_construction)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_log10(breaks = c(1, 2, 5, 10, 20, 50, 100),
                labels = c("1M", "2M", "5M", "10M", "20M", "50M", "100M")) +
  scale_color_manual(values = c("match" = "#2E7D32", "no_match" = "#C62828")) +
  theme_bw(base_size = 12) +
  theme(legend.position = "right") +
  labs(
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Animacy",
    linetype = "Construction"
  )

ggsave(file.path(output_dir, "animacy_boost_by_construction.pdf"), construction_plot, width = 10, height = 5)
cat("Saved: animacy_boost_by_construction.pdf\n")

# =============================================================
# 3. BOOST MAGNITUDE OVER TIME
# =============================================================

boost_magnitude <- boost_summary %>%
  pivot_wider(names_from = animacy_match, values_from = c(mean_odds, se)) %>%
  mutate(boost = mean_odds_match - mean_odds_no_match)

magnitude_plot <- ggplot(boost_magnitude, aes(x = tokens_M, y = boost)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(size = 3, color = "darkblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_log10(breaks = c(1, 2, 5, 10, 20, 50, 100),
                labels = c("1M", "2M", "5M", "10M", "20M", "50M", "100M")) +
  theme_bw(base_size = 12) +
  labs(
    x = "Training Tokens (log scale)",
    y = "Animacy Boost (Match - Mismatch)"
  )

ggsave(file.path(output_dir, "animacy_boost_magnitude.pdf"), magnitude_plot, width = 8, height = 4)
cat("Saved: animacy_boost_magnitude.pdf\n")

# =============================================================
# 4. STATISTICAL TESTS
# =============================================================

cat("\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
cat("STATISTICAL ANALYSIS: ANIMACY MATCHING EFFECT\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

# Linear model: max_odds ~ animacy_match + tokens_M + construction
model <- lm(max_odds ~ animacy_match * factor(tokens_M) + train_construction, 
            data = data %>% filter(train_construction == eval_construction))

cat("Model: max_odds ~ animacy_match * factor(tokens_M) + train_construction\n")
cat("(Within-construction experiments only)\n\n")

cat("Model Summary:\n")
print(summary(model))

# Overall animacy match effect
cat("\n\nAnimacy Match vs Mismatch (Overall):\n")
emm_animacy <- emmeans(model, ~ animacy_match)
print(pairs(emm_animacy))

# Effect size
residual_sd <- sigma(model)
animacy_contrast <- summary(pairs(emm_animacy))
cat(sprintf("\nCohen's d = %.2f\n", abs(animacy_contrast$estimate) / residual_sd))

# Animacy effect at key checkpoints
cat("\n\nAnimacy Effect by Checkpoint:\n")
emm_by_ckpt <- emmeans(model, ~ animacy_match | factor(tokens_M))
print(pairs(emm_by_ckpt, adjust = "none"))

# Save stats output
sink(file.path(output_dir, "stats_output.txt"))
cat("CROSS-ANIMACY STATISTICAL ANALYSIS\n")
cat("===================================\n\n")
cat("Model: max_odds ~ animacy_match * factor(tokens_M) + train_construction\n\n")
print(summary(model))
cat("\n\nAnimacy Match vs Mismatch:\n")
print(pairs(emm_animacy))
cat(sprintf("\nCohen's d = %.2f\n", abs(animacy_contrast$estimate) / residual_sd))
cat("\n\nAnimacy Effect by Checkpoint:\n")
print(pairs(emm_by_ckpt, adjust = "none"))
sink()

cat("\nSaved: stats_output.txt\n")

cat("\n=== All cross-animacy plots saved to:", output_dir, "===\n")
