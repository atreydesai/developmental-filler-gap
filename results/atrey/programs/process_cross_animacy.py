"""
Process cross-animacy experiment results.
Tests animacy matching: does train_animacy == eval_animacy help?
"""
import json
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

def process_single_file(args):
    """Process a single JSON file - worker function for multiprocessing."""
    json_file, checkpoint, seed = args
    
    try:
        with open(json_file) as f:
            result = json.load(f)
        
        metadata = result["metadata"]
        train = metadata["dataset"].split("/")[-1]
        eval_ds = metadata.get("eval_dataset", metadata["dataset"]).split("/")[-1]
        file_seed = metadata.get("seed", seed)
        
        # Calculate max ODDS per position directly
        pos_data = {}
        for item in result["data"]:
            pos = item["pos"]
            odds = (item["base_p_base"] - item["base_p_src"] + 
                    item["p_src"] - item["p_base"])
            iia = 1 if item["p_src"] > item["p_base"] else 0
            
            if pos not in pos_data:
                pos_data[pos] = {"max_odds": odds, "iia_sum": iia, "count": 1}
            else:
                pos_data[pos]["max_odds"] = max(pos_data[pos]["max_odds"], odds)
                pos_data[pos]["iia_sum"] += iia
                pos_data[pos]["count"] += 1
        
        # Extract animacy from train and eval dataset names
        train_animacy = "animate" if "animate" in train and "inanimate" not in train else "inanimate"
        eval_animacy = "animate" if "animate" in eval_ds and "inanimate" not in eval_ds else "inanimate"
        
        # Extract construction type
        train_construction = "wh" if "wh_question" in train else "topic"
        eval_construction = "wh" if "wh_question" in eval_ds else "topic"
        
        # Animacy match?
        animacy_match = "match" if train_animacy == eval_animacy else "no_match"
        
        rows = []
        for pos, vals in pos_data.items():
            rows.append({
                "checkpoint": checkpoint,
                "seed": file_seed,
                "train": train,
                "eval": eval_ds,
                "train_animacy": train_animacy,
                "eval_animacy": eval_animacy,
                "train_construction": train_construction,
                "eval_construction": eval_construction,
                "animacy_match": animacy_match,
                "pos": pos,
                "max_odds": vals["max_odds"],
                "mean_iia": vals["iia_sum"] / vals["count"]
            })
        
        return rows
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return []


def process_cross_animacy(log_folder="logs/cross_animacy"):
    """Process cross-animacy results using multiprocessing."""
    
    log_path = Path(log_folder)
    if not log_path.exists():
        print(f"Error: {log_folder} does not exist")
        return None
    
    # Collect all file tasks
    # Skip incomplete checkpoints
    skip_checkpoints = set()  # Still running
    
    tasks = []
    for ckpt_dir in sorted(log_path.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        checkpoint = ckpt_dir.name
        
        if checkpoint in skip_checkpoints:
            print(f"Skipping {checkpoint} (incomplete)")
            continue
        
        for seed_dir in ckpt_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name.replace("seed_", "")
            
            for json_file in seed_dir.glob("*.json"):
                tasks.append((json_file, checkpoint, seed))
    
    num_workers = max(1, cpu_count() - 2)
    print(f"Processing {len(tasks)} files using {num_workers} CPU cores...")
    
    # Process in parallel
    all_rows = []
    with Pool(processes=num_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_single_file, tasks)):
            all_rows.extend(rows)
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{len(tasks)} files...")
    
    if not all_rows:
        print("No data found!")
        return None
    
    print(f"Creating DataFrame from {len(all_rows)} rows...")
    df = pd.DataFrame(all_rows)
    
    # Add direction label (construction-based)
    df["direction"] = df["train_construction"] + "→" + df["eval_construction"]
    
    # Convert checkpoint to numeric for sorting
    df["tokens_M"] = df["checkpoint"].str.replace("M", "").astype(int)
    df = df.sort_values(["tokens_M", "seed", "train", "eval", "pos"])
    
    # Save full results
    output = Path("results/cross_animacy.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    print(f"Saved {len(df)} rows to {output}")
    
    # Summary by animacy_match
    summary = df.groupby(["checkpoint", "tokens_M", "direction", "animacy_match"]).agg({
        "max_odds": ["mean", "std", "count"],
        "mean_iia": ["mean", "std"]
    }).reset_index()
    summary = summary.sort_values("tokens_M")
    summary.to_csv("results/cross_animacy_summary.csv", index=False)
    print(f"Saved summary to results/cross_animacy_summary.csv")
    
    # Key comparison: match vs no_match
    print("\n=== ANIMACY MATCHING EFFECT ===")
    match_summary = df.groupby(["tokens_M", "animacy_match"]).agg({
        "max_odds": "mean"
    }).reset_index()
    
    pivot = match_summary.pivot(index="tokens_M", columns="animacy_match", values="max_odds")
    if "match" in pivot.columns and "no_match" in pivot.columns:
        pivot["boost"] = pivot["match"] - pivot["no_match"]
        print(pivot[["match", "no_match", "boost"]].to_string())
        pivot.to_csv("results/cross_animacy_boost.csv")
        print("\nSaved animacy boost comparison to results/cross_animacy_boost.csv")
    
    return df


if __name__ == "__main__":
    import time
    start = time.time()
    process_cross_animacy()
    print(f"\nTotal time: {time.time() - start:.1f} seconds")
