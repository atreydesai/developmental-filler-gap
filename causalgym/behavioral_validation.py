"""
Behavioral validation: Test Wilcox licensing effects via surprisal.
Validates that the model correctly licenses filler-gap dependencies.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_surprisal(model, tokenizer, sentence, device="cpu"):
    """Compute surprisal at final token position."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, :-1, :]
    log_probs = logits.log_softmax(dim=-1)
    target_ids = inputs.input_ids[0, 1:]
    
    # Return surprisal at last token
    return -log_probs[-1, target_ids[-1]].item()

def validate_licensing(model_path, revision=None):
    """
    Validate filler-gap licensing behavior using surprisal.
    
    Tests two effects from Wilcox et al.:
    1. Gap needs filler: Surprisal(-F+G) > Surprisal(+F+G)
    2. Filler needs gap: Surprisal(+F-G) > Surprisal(-F-G)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision).to(device)
    model.eval()
    
    # Test conditions
    tests = {
        "+F+G": "Who did the teacher like?",      # Grammatical: filler + gap
        "-F+G": "Did the teacher like?",          # Ungrammatical: no filler, gap
        "+F-G": "Who did the teacher like him",   # Ungrammatical: filler, no gap
        "-F-G": "Did the teacher like him"        # Grammatical: no filler, no gap
    }
    
    results = {}
    print(f"\nModel: {model_path}" + (f" (revision: {revision})" if revision else ""))
    print("-" * 50)
    
    for cond, sentence in tests.items():
        surp = compute_surprisal(model, tokenizer, sentence, device)
        results[cond] = {"sentence": sentence, "surprisal": surp}
        print(f"{cond}: {surp:.3f} | {sentence}")
    
    # Validate licensing effects
    effect1 = results["-F+G"]["surprisal"] > results["+F+G"]["surprisal"]
    effect2 = results["+F-G"]["surprisal"] > results["-F-G"]["surprisal"]
    
    print("-" * 50)
    print(f"Effect 1 (gap needs filler): {effect1}")
    print(f"  -F+G ({results['-F+G']['surprisal']:.3f}) > +F+G ({results['+F+G']['surprisal']:.3f})")
    print(f"Effect 2 (filler needs gap): {effect2}")
    print(f"  +F-G ({results['+F-G']['surprisal']:.3f}) > -F-G ({results['-F-G']['surprisal']:.3f})")
    
    return results, effect1, effect2

def validate_across_checkpoints(checkpoints=None):
    """Validate licensing across multiple checkpoints."""
    if checkpoints is None:
        checkpoints = ["chck_1M", "chck_10M", "chck_100M", "chck_1000M"]
    
    model_id = "BabyLM-community/babylm-baseline-100m-gpt2"
    
    all_results = {}
    for ckpt in checkpoints:
        print(f"\n{'='*50}")
        results, e1, e2 = validate_licensing(model_id, revision=ckpt)
        all_results[ckpt] = {"results": results, "effect1": e1, "effect2": e2}
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BabyLM-community/babylm-baseline-100m-gpt2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--all-checkpoints", action="store_true")
    args = parser.parse_args()
    
    if args.all_checkpoints:
        validate_across_checkpoints()
    else:
        validate_licensing(args.model, args.revision)
