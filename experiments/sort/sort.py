"""Sorting Experiment - model must produce a sorted list of integers"""
import json
import torch
import numpy as np
from pathlib import Path

DATASET_NAME = "sort"

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET_PATH = str(_DATA_DIR / "sort.json")

def load_input_rows():
    """
    Returns the data in the json file for the dataset

    Output:
        - a dictionary containing the input and appropriate tags
    """
    with open(_DEFAULT_DATASET_PATH, 'r') as file:
        data = json.load(file)
        return data

def load_prompts(**kwargs) -> list[dict]:
    data = load_input_rows()
    inputs = data["inputs"]
    tags = data["tags"]

    instances = []
    for i in range(len(inputs)):
        instances.append(
            {
                "prompt": inputs[i],
                "inputs": inputs[i],
                "tags": tags[i]
            }
        )

    return instances

def constraint_fn(instance: dict, sequence: str) -> bool:
    """True = acceptable, False = violation."""
    seq_list = sorted(int(x) for x in sequence.split())
    in_list = sorted(int(x) for x in instance["inputs"])
    return seq_list == in_list

def check_call_fn(instance, decoded_sequences, token_lists):
    """Don't check incomplete prefixes — let complete_flag trigger the check at EOS."""
    return np.zeros(len(decoded_sequences), dtype=bool)

def instance_context_fn(instance: dict) -> str:
    """Cache key includes the sorted input so results are per-instance, not per-sequence."""
    return ",".join(sorted(instance["inputs"]))

if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run Sort experiment.")
    parser.add_argument("--model", required=True) # must be a path to the model 
    parser.add_argument("--log_dir", default="beaver_logs")
    args, _ = parser.parse_known_args()
    loaded_model = torch.load(args.model, weights_only=False).eval()


    beaver.run(
        prompts=load_prompts(),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name=DATASET_NAME,
        instance_context_fn=instance_context_fn,
        model=loaded_model,
        log_dir=args.log_dir,
    )
