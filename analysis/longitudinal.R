library(ggplot2)
library(dplyr)
library(arrow)
library(tidyr)

data <- read_parquet("results/longitudinal.parquet")

# ===== MAIN PLOT: Development Curves =====
summary <- data %>%
  group_by(tokens_M, direction) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

emergence_plot <- ggplot(summary, aes(x = tokens_M, y = mean_odds, color = direction)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_odds - se, ymax = mean_odds + se, fill = direction), 
              alpha = 0.2, color = NA) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Emergence of Filler-Gap Mechanisms Across Training",
    subtitle = "BabyLM checkpoints, batch=25 steps=80, 6 seeds",
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Transfer\nDirection",
    fill = "Transfer\nDirection"
  )

ggsave("analysis/figures/longitudinal_emergence.pdf", 
       emergence_plot, width = 12, height = 6)

# ===== ASYMMETRY PLOT (Original Style) =====
asymmetry <- data %>%
  filter(train != eval) %>%
  group_by(tokens_M, direction, seed) %>%
  summarize(mean_odds = mean(max_odds), .groups = "drop") %>%
  pivot_wider(names_from = direction, values_from = mean_odds)

colnames(asymmetry) <- gsub("→", "_to_", colnames(asymmetry))

if ("wh_to_topicalization" %in% colnames(asymmetry) && "topicalization_to_wh" %in% colnames(asymmetry)) {
  asymmetry$ratio <- asymmetry$wh_to_topicalization / (asymmetry$topicalization_to_wh + 0.01)
  
  asym_plot <- ggplot(asymmetry, aes(x = tokens_M, y = ratio)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "loess", se = TRUE, color = "steelblue") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                  labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
    theme_bw(base_size = 12) +
    labs(
      title = "Transfer Asymmetry Across Training",
      subtitle = "Ratio > 1 means Wh->Topic stronger than Topic->Wh",
      x = "Training Tokens (log scale)",
      y = "Asymmetry Ratio (Wh->Topic / Topic->Wh)"
    )
  
  ggsave("analysis/figures/longitudinal_asymmetry.pdf", 
         asym_plot, width = 10, height = 5)
}

# ===== WITHIN CONSTRUCTION PLOT =====
within <- data %>%
  filter(train == eval) %>%
  group_by(tokens_M, direction) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

within_plot <- ggplot(within, aes(x = tokens_M, y = mean_odds, color = direction)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_odds - se, ymax = mean_odds + se, fill = direction), 
              alpha = 0.2, color = NA) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Within-Construction DAS Localization",
    subtitle = "How well each construction's filler-gap is localized",
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Construction",
    fill = "Construction"
  )

ggsave("analysis/figures/longitudinal_within.pdf", 
       within_plot, width = 10, height = 5)

# ===== TRANSFER vs WITHIN RATIO =====
all_summary <- data %>%
  mutate(type = if_else(train == eval, "within", "transfer")) %>%
  group_by(tokens_M, type) %>%
  summarize(mean_odds = mean(max_odds), .groups = "drop") %>%
  pivot_wider(names_from = type, values_from = mean_odds)

all_summary$transfer_ratio <- all_summary$transfer / all_summary$within

ratio_plot <- ggplot(all_summary, aes(x = tokens_M, y = transfer_ratio)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(size = 3, color = "darkgreen") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Cross-Construction Generalization Ratio",
    subtitle = "Transfer ODDS / Within ODDS (1.0 = perfect generalization)",
    x = "Training Tokens (log scale)",
    y = "Transfer / Within Ratio"
  )

ggsave("analysis/figures/longitudinal_ratio.pdf", 
       ratio_plot, width = 10, height = 5)

# ===== ANIMATE vs INANIMATE COMPARISON =====
# Emergence plot faceted by animacy
summary_animacy <- data %>%
  group_by(tokens_M, direction, animacy) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

animacy_emergence <- ggplot(summary_animacy, aes(x = tokens_M, y = mean_odds, color = direction)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_odds - se, ymax = mean_odds + se, fill = direction), 
              alpha = 0.2, color = NA) +
  facet_wrap(~animacy, ncol = 2) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Emergence by Animacy Condition",
    subtitle = "BabyLM checkpoints, 6 seeds",
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Direction",
    fill = "Direction"
  )

ggsave("analysis/figures/longitudinal_animacy_facet.pdf", 
       animacy_emergence, width = 14, height = 6)

# Asymmetry by animacy
asymmetry_animacy <- data %>%
  filter(train != eval) %>%
  group_by(tokens_M, direction, animacy, seed) %>%
  summarize(mean_odds = mean(max_odds), .groups = "drop") %>%
  pivot_wider(names_from = direction, values_from = mean_odds)

colnames(asymmetry_animacy) <- gsub("→", "_to_", colnames(asymmetry_animacy))

if ("wh_to_topicalization" %in% colnames(asymmetry_animacy) && "topicalization_to_wh" %in% colnames(asymmetry_animacy)) {
  asymmetry_animacy$ratio <- asymmetry_animacy$wh_to_topicalization / (asymmetry_animacy$topicalization_to_wh + 0.01)
  
  asym_animacy_plot <- ggplot(asymmetry_animacy, aes(x = tokens_M, y = ratio, color = animacy)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "loess", se = TRUE) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                  labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
    scale_color_manual(values = c("animate" = "#E69F00", "inanimate" = "#56B4E9")) +
    theme_bw(base_size = 12) +
    labs(
      title = "Transfer Asymmetry by Animacy",
      subtitle = "Ratio > 1 means Wh->Topic stronger than Topic->Wh",
      x = "Training Tokens (log scale)",
      y = "Asymmetry Ratio",
      color = "Animacy"
    )
  
  ggsave("analysis/figures/longitudinal_asymmetry_animacy.pdf", 
         asym_animacy_plot, width = 10, height = 5)
}

# Within-construction by animacy
within_animacy <- data %>%
  filter(train == eval) %>%
  group_by(tokens_M, direction, animacy) %>%
  summarize(
    mean_odds = mean(max_odds, na.rm = TRUE),
    se = sd(max_odds, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

within_animacy_plot <- ggplot(within_animacy, aes(x = tokens_M, y = mean_odds, color = direction, linetype = animacy)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
                labels = c("1M", "2M", "3M", "5M", "10M", "20M", "50M", "100M", "200M", "500M", "1000M")) +
  theme_bw(base_size = 12) +
  labs(
    title = "Within-Construction by Animacy",
    subtitle = "Solid = animate, Dashed = inanimate",
    x = "Training Tokens (log scale)",
    y = "Max ODDS",
    color = "Construction",
    linetype = "Animacy"
  )

ggsave("analysis/figures/longitudinal_within_animacy.pdf", 
       within_animacy_plot, width = 10, height = 5)

print("All longitudinal plots (including animacy) saved to analysis/plots/")
