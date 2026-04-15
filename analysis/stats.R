library(dplyr)
library(arrow)
library(emmeans)

data <- read_parquet("results/longitudinal.parquet")
dev_data <- data %>% filter(tokens_M <= 100)

cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
cat("STATISTICAL ANALYSIS: FILLER-GAP LONGITUDINAL STUDY\n")
cat("Model: max_odds ~ factor(tokens_M) + direction + animacy\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

# ============================================================
# 1. LINEAR MODEL (No random effects)
# ============================================================
cat("\n1. LINEAR MODEL\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

model <- lm(
  max_odds ~ factor(tokens_M) + direction + animacy, 
  data = dev_data
)

cat("\nModel Summary:\n")
print(summary(model))

# ANOVA table for main effects
cat("\n\nANOVA (Type III):\n")
print(anova(model))

# ============================================================
# 2. POST-HOC TESTS WITH EMMEANS
# ============================================================
cat("\n\n2. POST-HOC TESTS (Estimated Marginal Means)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

# 2a. Within vs Across (custom contrast on direction)
cat("\n2a. Within vs Across Construction:\n")
emm_dir <- emmeans(model, ~ direction)
# Order: topic->topic, topic->wh, wh->topic, wh->wh
contrast_wa <- list("Within - Across" = c(0.5, -0.5, -0.5, 0.5))
print(contrast(emm_dir, contrast_wa))

# 2b. Animate vs Inanimate
cat("\n2b. Animate vs Inanimate:\n")
emm_anim <- emmeans(model, ~ animacy)
print(pairs(emm_anim))

# 2c. Transfer Asymmetry (wh->topic vs topic->wh)
cat("\n2c. Transfer Asymmetry (Wh->Topic vs Topic->Wh):\n")
# Order: topic->topic, topic->wh, wh->topic, wh->wh
contrast_asym <- list("Wh->Topic - Topic->Wh" = c(0, -1, 1, 0))
print(contrast(emm_dir, contrast_asym))

# 2d. Effect of training (tokens_M)
cat("\n2d. Effect of Training (select checkpoints):\n")
emm_tokens <- emmeans(model, ~ factor(tokens_M))
# Compare 10M vs 50M vs 100M
cat("Pairwise comparisons at 10M, 50M, 100M:\n")
emm_subset <- emmeans(model, ~ factor(tokens_M), at = list(`factor(tokens_M)` = c("10", "50", "100")))
print(pairs(emm_subset, adjust = "holm"))

# ============================================================
# 3. EFFECT SIZES
# ============================================================
cat("\n\n3. EFFECT SIZES (Cohen's d approximations)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

residual_sd <- sigma(model)

# Within vs Across
wa_contrast <- summary(contrast(emm_dir, contrast_wa))
cat(sprintf("Within vs Across: d = %.2f\n", wa_contrast$estimate / residual_sd))

# Animate vs Inanimate  
anim_pairs <- summary(pairs(emm_anim))
cat(sprintf("Animate vs Inanimate: d = %.2f\n", anim_pairs$estimate / residual_sd))

# Asymmetry
asym_contrast <- summary(contrast(emm_dir, contrast_asym))
cat(sprintf("Asymmetry (Wh->Topic - Topic->Wh): d = %.2f\n", asym_contrast$estimate / residual_sd))

cat("\n\nAnalysis complete.\n")
