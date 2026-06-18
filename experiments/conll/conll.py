"""CoNLL Experiment - model must correctly classify proper nouns via named entity recoginition"""
import json
import torch
from pathlib import Path

DATASET_NAME = "conll"

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET_PATH = str(_DATA_DIR / "test.json")

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
    return sequence == instance["tags"]

def check_call_fn(instance, decoded_sequences, token_lists):
    """Optional fast pre-filter — skip expensive checks on short prefixes."""
    return True

def instance_context_fn(instance: dict) -> str:
    """Cache key for this instance's constraint context."""
    return ""

if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run CoNLL-2003 experiment.")
    parser.add_argument("--model", required=True) # must be a path to the model 
    parser.add_argument("--log_dir", default="beaver_logs")
    parser.add_argument("--glove_embed", default=1)
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
        glove_embed=args.glove_embed
    )
