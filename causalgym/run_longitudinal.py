#!/usr/bin/env python3
"""
Longitudinal runner for BabyLM filler-gap experiments.
Usage: python causalgym/run_longitudinal.py [--steps 16] [--batch-size 25]
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS, CACHE_DIR
from das import experiment


def run_longitudinal(
    model: str = "BabyLM-community/babylm-baseline-100m-gpt2",
    checkpoints: list[str] = None,
    seeds: list[int] = None,
    train_datasets: list[str] = None,
    eval_datasets: list[str] = None,
    steps: int = 16,
    batch_size: int = 25,
):  
    # Defaults
    if checkpoints is None:
        checkpoints = ["1M", "10M", "100M", "1000M"]
    if seeds is None:
        seeds = [42, 43, 44]
    if train_datasets is None:
        train_datasets = ["wh_topicalization/wh_question_animate",
                          "wh_topicalization/topicalization_animate"]
    if eval_datasets is None:
        eval_datasets = train_datasets.copy()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 50)
    print("Optimized Longitudinal Filler-Gap Experiment")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {steps}, Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 50)
    
    total_runs = len(checkpoints) * len(seeds) * len(train_datasets) * len(eval_datasets)
    run_count = 0
    
    for ckpt in checkpoints:
        revision = f"chck_{ckpt}"
        print(f"\n{'=' * 50}")
        print(f"Loading checkpoint: {ckpt} (revision: {revision})")
        print("=" * 50)
        
        # Load model ONCE per checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        
        weight_type = WEIGHTS.get(model, torch.float16) if device == "cuda:0" else torch.float32
        attn_impl = "sdpa" if device == "cuda:0" else "eager"
        
        gpt = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            torch_dtype=weight_type,
            cache_dir=CACHE_DIR,
            attn_implementation=attn_impl,
        ).to(device)
        
        print(f"✓ Model loaded: {gpt.config.architectures[0]}")
        print(f"  Attention: {attn_impl}, dtype: {weight_type}")
        
        # Loop through all seeds and dataset combinations with loaded model
        for seed in seeds:
            print(f"\n  Seed: {seed}")
            
            for train_ds in train_datasets:
                for eval_ds in eval_datasets:
                    run_count += 1
                    train_name = train_ds.split("/")[-1]
                    eval_name = eval_ds.split("/")[-1]
                    das_label = f"{train_name}_to_{eval_name}"
                    log_folder = f"logs/longitudinal/{ckpt}/seed_{seed}"
                    
                    print(f"    [{run_count}/{total_runs}] {train_name} → {eval_name}")
                    
                    # Run experiment with pre-loaded model
                    experiment(
                        model=model,
                        dataset=train_ds,
                        steps=steps,
                        eval_steps=25,
                        grad_steps=1,
                        batch_size=batch_size,
                        intervention_site="block_output",
                        strategy="last",
                        lr=5e-3,
                        only_das=True,
                        das_label=das_label,
                        revision=revision,
                        log_folder=log_folder,
                        eval_dataset=eval_ds,
                        seed=seed,
                        tokenizer=tokenizer,  # Pass pre-loaded tokenizer
                        gpt=gpt,              # Pass pre-loaded model
                    )
        
        # Clear GPU memory before loading next checkpoint
        del gpt
        del tokenizer
        torch.cuda.empty_cache()
        print(f"\n✓ Checkpoint {ckpt} complete, GPU memory cleared")
    
    print("\n" + "=" * 50)
    print("All experiments complete!")
    print(f"Total runs: {run_count}")
    print("Run: python results/programs/process_results.py")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal runner for BabyLM filler-gap experiments"
    )
    parser.add_argument("--model", type=str, 
                        default="BabyLM-community/babylm-baseline-100m-gpt2")
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=["1M", "10M", "100M", "1000M"],
                        help="Checkpoint revisions (e.g., 1M 10M 100M)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44],
                        help="Random seeds for reproducibility")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=25)
    
    args = parser.parse_args()
    
    run_longitudinal(
        model=args.model,
        checkpoints=args.checkpoints,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
