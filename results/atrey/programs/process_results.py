"""
Process longitudinal experiment results - FAST version.
Uses multiprocessing and streaming JSON for speed.
"""
import json
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os

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
        
        # Calculate max ODDS per position directly (memory efficient)
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
        
        # Extract animacy from train dataset name
        animacy = "animate" if "animate" in train and "inanimate" not in train else "inanimate"
        
        rows = []
        for pos, vals in pos_data.items():
            rows.append({
                "checkpoint": checkpoint,
                "seed": file_seed,
                "train": train,
                "eval": eval_ds,
                "animacy": animacy,
                "pos": pos,
                "max_odds": vals["max_odds"],
                "mean_iia": vals["iia_sum"] / vals["count"]
            })
        
        return rows
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return []


def process_longitudinal(log_folder="logs/longitudinal"):
    """Process results using multiprocessing for speed."""
    
    log_path = Path(log_folder)
    if not log_path.exists():
        print(f"Error: {log_folder} does not exist")
        return None
    
    # Collect all file tasks
    tasks = []
    for ckpt_dir in sorted(log_path.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        checkpoint = ckpt_dir.name
        
        for seed_dir in ckpt_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name.replace("seed_", "")
            
            for json_file in seed_dir.glob("*.json"):
                tasks.append((json_file, checkpoint, seed))
    
    num_workers = max(1, cpu_count() - 2)  # Leave 2 cores for other processes
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
    
    # Add direction label
    df["direction"] = df.apply(
        lambda r: f"{r['train'].split('_')[0]}→{r['eval'].split('_')[0]}", axis=1
    )
    
    # Convert checkpoint to numeric for sorting
    df["tokens_M"] = df["checkpoint"].str.replace("M", "").astype(int)
    df = df.sort_values(["tokens_M", "seed", "train", "eval", "pos"])
    
    output = Path("results/longitudinal.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    print(f"Saved {len(df)} rows to {output}")
    
    # Summary - overall
    summary = df.groupby(["checkpoint", "tokens_M", "direction"]).agg({
        "max_odds": ["mean", "std"],
        "mean_iia": ["mean", "std"]
    }).reset_index()
    summary = summary.sort_values("tokens_M")
    summary.to_csv("results/summary.csv", index=False)
    print(f"Saved summary to results/summary.csv")
    
    # Summary by animacy
    for animacy in ["animate", "inanimate"]:
        anim_df = df[df["animacy"] == animacy]
        anim_summary = anim_df.groupby(["checkpoint", "tokens_M", "direction"]).agg({
            "max_odds": ["mean", "std"],
            "mean_iia": ["mean", "std"]
        }).reset_index()
        anim_summary = anim_summary.sort_values("tokens_M")
        anim_summary.to_csv(f"results/summary_{animacy}.csv", index=False)
        print(f"Saved {animacy} summary to results/summary_{animacy}.csv")
    
    return df


if __name__ == "__main__":
    import time
    start = time.time()
    process_longitudinal()
    print(f"\nTotal time: {time.time() - start:.1f} seconds")
