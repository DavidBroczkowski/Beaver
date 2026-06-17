"""Sorting Experiment - model must produce a sorted list of integers"""
import csv
import torch
from pathlib import Path

DATASET_NAME = "sort"

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET_PATH = str(_DATA_DIR / "sort.csv")

def load_input_rows():
    """
    Returns the data in the csv file for the dataset

    Output:
        - a 2d list containing each input as a list of Strings
    """
    with open(_DEFAULT_DATASET_PATH, 'r', newline='') as csvfile:
        data = csv.reader(csvfile)
        return list(data)

def load_prompts(**kwargs) -> list[dict]:
    data = load_input_rows()

    instances = []
    for row in data:
        input_dict = {
            "input": (int(num) for num in row),
            "prompt": " ".join(row)
        }
        instances.append(input_dict)
    return instances

def constraint_fn(instance: dict, sequence: str) -> bool:
    """True = acceptable, False = violation."""
    seq_list = sorted(int(x) for x in sequence.split())
    in_list = sorted(int(x) for x in instance["input"].split())
    return seq_list == in_list

def check_call_fn(instance, decoded_sequences, token_lists):
    """Optional fast pre-filter — skip expensive checks on short prefixes."""
    return True

def instance_context_fn(instance: dict) -> str:
    """Cache key for this instance's constraint context."""
    return ""

if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run Sort experiment.")
    parser.add_argument("--model", required=True) # must be a path to the model 
    parser.add_argument("--log_dir", default="beaver_logs")
    args, _ = parser.parse_known_args()

    beaver.run(
        prompts=load_prompts(),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name=DATASET_NAME,
        instance_context_fn=instance_context_fn,
        model=torch.load(args.model, weights_only=False).eval(),
        log_dir=args.log_dir,
    )
