# Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints?
This repository contains all the code necessary to run the experiments and analyses for the paper _Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints?_

In particular, it includes:

- Training and evaluating DAS interventions for single-source and leave-one-out construction variants (both in the single-clause and embedded-clause case), following [Boguraev et al. (2025)](https://aclanthology.org/2025.emnlp-main.1271/).
- Longitudinal experiments tracking how DAS-localized filler-gap representations emerge across BabyLM training checkpoints (1M–1000M tokens).
- Cross-animacy experiments testing whether animacy matching between train and eval datasets produces a localization boost.
- Hyperparameter ablation experiments examining the effect of batch size and training steps on DAS localization.
- Performing statistical analyses and generating the plots in the paper.

> [!IMPORTANT]
> This project is built upon the work of [Boguraev et al. (2025)](https://aclanthology.org/2025.emnlp-main.1271/) ([`causal-filler-gap`](https://github.com/SashaBoguraev/causal-filler-gap)), which is itself forked from [CausalGym](https://github.com/aryamanarora/causalgym) (Arora et al. 2024). Much of the code in `causalgym/` and `data/templates/` is taken directly from those repositories. The following files have been modified or added in this work:
>
> **Modified from `causal-filler-gap`:**
> - `causalgym/data.py` — Extended tokenization alignment logic
> - `causalgym/utils.py` — Cache directory made configurable via environment variable
>
> **New in this work:**
> - `causalgym/run_longitudinal.py` — Longitudinal runner
> - `commands/run_longitudinal.sh` — Longitudinal experiment pipeline
> - `commands/run_hparam_ablation.sh` — Hyperparameter ablation pipeline
> - `commands/cross_animacy/run_*.sbatch` — SLURM batch scripts for cross-animacy experiments
> - `data/templates/wh_topicalization.json`, `data/templates/wh_topicalization_inanimate.json` — Animate and inanimate wh/topicalization minimal-pair templates
> - `analysis/longitudinal.R`, `analysis/longitudinal_developmental.R` — Developmental emergence plots
> - `analysis/hparam.R` — Hyperparameter ablation plots
> - `analysis/cross_animacy.R` — Cross-animacy visualization and statistical tests
> - `analysis/stats.R` — Linear model with post-hoc tests and effect sizes for longitudinal data
> - `results/programs/` — Scripts to aggregate raw experiment logs into parquet/CSV outputs
>
> Please cite [Boguraev et al. (2025)](https://aclanthology.org/2025.emnlp-main.1271/) and [CausalGym](https://github.com/aryamanarora/causalgym) (Arora et al. 2024) in addition to our work if you use this repository.

## Instructions

### Set-Up

First, install the requirements:

`pip install -r requirements.txt`

All scripts should be run from the **project root directory**.

### Longitudinal Experiments

These experiments train DAS interventions across BabyLM training checkpoints (1M–1000M tokens) and examine how filler-gap representations emerge developmentally.

```bash
bash commands/run_longitudinal.sh [steps] [batch_size]
```

Then process results into a parquet file:

```bash
python results/programs/process_results.py
```

### Hyperparameter Ablation

Tests the effect of different batch size × training step configurations on DAS localization.

```bash
bash commands/run_hparam_ablation.sh
```

Then process results:

```bash
python results/programs/process_hparam.py
```

### Cross-Animacy Experiments

Tests whether animacy matching between train and eval datasets produces a localization boost. Uses SLURM batch scripts divided across checkpoint ranges. **Before submitting**, update `--partition` and `--account` in each `.sbatch` file to match your cluster, then submit:

```bash
sbatch commands/cross_animacy/run_1.sbatch  # checkpoints 1M–4M
sbatch commands/cross_animacy/run_2.sbatch  # checkpoints 5M–7M
sbatch commands/cross_animacy/run_3.sbatch  # checkpoints 8M–10M
sbatch commands/cross_animacy/run_4.sbatch  # checkpoints 20M–40M
sbatch commands/cross_animacy/run_5.sbatch  # checkpoint 50M
sbatch commands/cross_animacy/run_6.sbatch  # checkpoint 100M
```

Then process results:

```bash
python results/programs/process_cross_animacy.py
```

### Analysis

Once you have generated your data (or using the pre-computed parquet files in `results/`), you can run the provided R analysis scripts in the `analysis/` folder. Run all scripts from the **project root directory**:

```bash
Rscript analysis/<script_name>.R
```

- Development curves, asymmetry, and animacy plots (full range): `analysis/longitudinal.R`
- Developmental range only (1M–100M): `analysis/longitudinal_developmental.R`
- Hyperparameter ablation plots: `analysis/hparam.R`
- Cross-animacy visualization and statistics: `analysis/cross_animacy.R`
- Linear model with post-hoc tests and effect sizes: `analysis/stats.R`

We also provide pre-computed parquet files for all longitudinal analyses in `results/` so the analysis can be reproduced without re-running experiments.

#

If you are having any issues running any code, please do not hesitate to reach out or file an issue!

## Citation

If you use any of our code in your work, please cite our paper:

```bibtex
Will be updated soon!
```

Please also cite [Boguraev et al. (2025)](https://aclanthology.org/2025.emnlp-main.1271/), whose codebase this work builds upon:

```bibtex
@inproceedings{boguraev-etal-2025-causal,
    title = "Causal Interventions Reveal Shared Structure Across {E}nglish Filler{--}Gap Constructions",
    author = "Boguraev, Sasha  and Potts, Christopher  and Mahowald, Kyle",
    editor = "Christodoulopoulos, Christos  and Chakraborty, Tanmoy  and Rose, Carolyn  and Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1271/",
    doi = "10.18653/v1/2025.emnlp-main.1271",
    pages = "25032--25053"
}
```

Also please cite the original CausalGym paper:

```bibtex
@inproceedings{arora-etal-2024-causalgym,
    title = "{C}ausal{G}ym: Benchmarking causal interpretability methods on linguistic tasks",
    author = "Arora, Aryaman and Jurafsky, Dan and Potts, Christopher",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.785",
    doi = "10.18653/v1/2024.acl-long.785",
    pages = "14638--14663"
}
```
