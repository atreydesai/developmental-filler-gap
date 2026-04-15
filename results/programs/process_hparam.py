"""
Process hyperparameter ablation results - memory optimized version.
Processes files one at a time to avoid OOM.
"""
import json
import pandas as pd
from pathlib import Path
import re

def process_hparam(log_folder="logs/hparam"):
    """Process hyperparameter ablation results with low memory usage."""
    
    log_path = Path(log_folder)
    if not log_path.exists():
        print(f"Error: {log_folder} does not exist")
        return None
    
    all_summaries = []
    
    for config_dir in log_path.iterdir():
        if not config_dir.is_dir():
            continue
        
        # Parse config from folder name: b4_s100 -> batch=4, steps=100
        match = re.match(r'b(\d+)_s(\d+)', config_dir.name)
        if not match:
            continue
        batch_size = int(match.group(1))
        steps = int(match.group(2))
        samples = batch_size * steps
        
        print(f"Processing {config_dir.name} ({samples} samples)...")
        
        for seed_dir in config_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name.replace("seed_", "")
            
            for json_file in seed_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        result = json.load(f)
                    
                    metadata = result["metadata"]
                    train = metadata["dataset"].split("/")[-1]
                    eval_ds = metadata.get("eval_dataset", metadata["dataset"]).split("/")[-1]
                    file_seed = metadata.get("seed", seed)
                    
                    # Calculate max ODDS and mean IIA per position directly
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
                    
                    # Add summary for this file
                    for pos, vals in pos_data.items():
                        all_summaries.append({
                            "batch_size": batch_size,
                            "steps": steps,
                            "samples": samples,
                            "config": f"b{batch_size}_s{steps}",
                            "seed": file_seed,
                            "train": train,
                            "eval": eval_ds,
                            "pos": pos,
                            "max_odds": vals["max_odds"],
                            "mean_iia": vals["iia_sum"] / vals["count"]
                        })
                    
                    # Clear memory
                    del result
                    
                except Exception as e:
                    print(f"  Error processing {json_file.name}: {e}")
    
    if not all_summaries:
        print("No data found!")
        return None
    
    df = pd.DataFrame(all_summaries)
    
    # Add direction label
    df["direction"] = df.apply(
        lambda r: f"{r['train'].split('_')[0]}→{r['eval'].split('_')[0]}", axis=1
    )
    
    output = Path("results/hparam_ablation.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    print(f"Saved {len(df)} rows to {output}")
    
    # Summary by config and direction
    summary = df.groupby(["config", "samples", "direction"]).agg({
        "max_odds": ["mean", "std"],
        "mean_iia": ["mean", "std"]
    }).reset_index()
    summary.to_csv("results/hparam_summary.csv", index=False)
    print(f"Saved summary to results/hparam_summary.csv")
    
    return df

if __name__ == "__main__":
    process_hparam()
