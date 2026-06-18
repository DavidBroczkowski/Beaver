"""Shared worker state and utilities for verification workers.

All multiprocessing worker state lives here as a single SimpleNamespace (`_w`),
replacing the 17+ individual `_worker_*` globals that were duplicated across
frontier_verifier.py and sampling_verifier.py.
"""

import time
import functools
import traceback
import types

from openai import OpenAI
import httpx
import numpy as np
import torch

from beaver.utils import log_json
from beaver.utils.tokenizer_utils import initialize_llguidance
from beaver.utils.glove_emb_utils import embed


# ---------------------------------------------------------------------------
# Single global worker state — populated by init_worker_state() in each
# spawned process.  Every field that was previously a _worker_* global
# becomes an attribute of _w.
# ---------------------------------------------------------------------------
_w = types.SimpleNamespace()


def init_worker_state(config_dict):
    """Initialize worker process state from a config dictionary.

    Called as the multiprocessing Pool initializer.  *config_dict* must be
    pickleable (strings, numbers, booleans, top-level function references).

    NOTE: We clear and repopulate _w in-place (rather than reassigning) so
    that modules which imported _w via ``from worker_common import _w``
    keep a reference to the same object.
    """
    # Clear any previous state without reassigning the object
    _w.__dict__.clear()

    _w.model_name = config_dict["model_name"]  # Store model (parent nn.Module)
    _w.tokenizer = config_dict["tokenizer"]
    _w.lltokenizer = initialize_llguidance(_w.tokenizer.w_idx)
    _w.vocab_size = config_dict["vocab_size"]  # Store total vocab size including special tokens
    _w.ebnf = config_dict["ebnf"]
    _w.dataset_name = config_dict["dataset_name"]
    _w.use_cache = config_dict.get("use_cache", True)
    _w.temperature = config_dict.get("temperature", 1.0)
    _w.top_p = config_dict.get("top_p", 1.0)
    _w.top_k = config_dict.get("top_k", -1)
    _w.eos_tokens = config_dict.get("eos_tokens", [])
    _w.idx_emb = config_dict.get("idx_emb")
    _w.gen_length = config_dict.get("gen_length", 128)
    _w.epsilon = config_dict.get("epsilon", 0.01)
    _w.verbose = config_dict.get("verbose", False)
    _w.max_iterations = config_dict.get("max_iterations", 100)
    _w.semantic_symbol = config_dict.get("semantic_symbol", None)
    _w.num_logprobs = config_dict.get("num_logprobs", 100)
    _w.use_grammar = config_dict.get("use_grammar", True)
    _w.chat_mode = config_dict.get("chat_mode", False)
    _w.system_message = config_dict.get("system_message", None)
    _w.fewshot_messages = config_dict.get("fewshot_messages", [])

    # Re-register the constraint so _REGISTRY is populated in this worker process.
    # With spawn start method, workers start fresh and don't inherit the main
    # process's _REGISTRY, so we pass the functions through config_dict.
    if "check_fn" in config_dict:
        from beaver.constraints.base_constraints import register_constraint

        register_constraint(
            config_dict["dataset_name"],
            check_call_fn=config_dict["check_call_fn"],
            instance_context_fn=config_dict["instance_context_fn"],
            check_fn=config_dict["check_fn"],
        )

    # Frontier-specific — only set when present in config
    if "frontier_topp" in config_dict:
        _w.frontier_topp = config_dict["frontier_topp"]
        _w.frontier_topk = config_dict["frontier_topk"]
        _w.frontier_scoring_strategy = config_dict["frontier_scoring_strategy"]


# ---------------------------------------------------------------------------
# Helper function to build prompt with optional chat template
# ---------------------------------------------------------------------------


def build_prompt(instance, continuation):
    """Build the final prompt string sent to the model as input

    Args:
        instance: Dict with at least ``prompt`` (list of String). May also contain
            ``system_prompt`` and ``fewshot_messages`` (per-instance overrides).
        continuation: List of token IDs generated so far (the partial sequence
            being extended).

    Returns:
        A torch.Tensor containing the word embeddings of the prompt
    """
    if _w.verbose:
        print("[DEBUG] Tokenizing prompt into ids...")

    prompt_token_ids = _w.tokenizer.tokenize(instance["prompt"]) # after this will be in the form of a list of indices
    
    if _w.verbose:
        print("[DEBUG] Embedding prompt...")
    
    if _w.glove_embed:
        if continuation:
            return embed(prompt_token_ids + continuation)

        return embed(prompt_token_ids)
    else:
        if continuation:
            return prompt_token_ids + continuation
        return prompt_token_ids


# ---------------------------------------------------------------------------
# model_generate — get logprobs from model
# ---------------------------------------------------------------------------

def model_generate_next_token_logprobs(instance, continuation):
    """Get next-token logprobs from vLLM server using OpenAI-compatible API.

    Args:
        instance: Instance dict (must have ``prompt``; may have
            ``system_prompt``, ``fewshot_messages``).
        continuation: List of token IDs generated so far.

    Returns:
        (np.ndarray, str): ``(logprobs_array, prompt)`` where logprobs_array
        has shape [N, 2] with [token_id, logprob] pairs, and prompt is the
        final string sent to the server.
    """
    try:
        # --- encode prompt into tokens ---------------------------------------------
        prompt = build_prompt(instance, continuation)

        # FIXME: right now this is a list of vectors, aka word embeddings
        if _w.verbose:
            print("[DEBUG] Prompt built successfully")
            print(f"prompt: {prompt}")

        # forward pass
        model = _w.model_name

        if _w.verbose:
            print("[DEBUG] Running the forward pass...")
        logits = model.forward(prompt)

        logprobs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float)
        if _w.verbose:
            print(f"[DEBUG] Received logprobs: {logprobs}")

        logprobs_w_ids = []
        for logprob in logprobs.items():
            token_id = i
            logprobs_w_ids.append([token_id, logprob])
            i += 1

        # prompt used to be an array of ids of words, changed now, so have to figure that out
        return np.array(logprobs), prompt

    except Exception as e:
        # FIXME: Likely would want some more robust error handling here
        print(f"[ERROR] An error occurred when attempting to compute the next token log probabilities, {e}")

# ---------------------------------------------------------------------------
# worker_setup — shared boilerplate at the start of _worker_process_instance
# ---------------------------------------------------------------------------

def worker_setup(args):
    """Common setup for worker process instances.

    Returns (instance, log_file, profile_log_file).
    """
    if _w.verbose:
        print("[DEBUG] Setting up worker...")

    instance, run_log_dir = args

    assert _w.dataset_name is not None, "Worker dataset_name not initialized"

    log_file = run_log_dir / f"{instance['idx']}.jsonl"
    profile_log_file = run_log_dir / f"{instance['idx']}.profile.json"

    setup_info = {
        "idx": instance["idx"],
        "prompt": instance["prompt"],
        "max_iterations": _w.max_iterations,
        "use_grammar": _w.use_grammar,
    }
    log_json(setup_info, log_file)

    if _w.verbose:
        print("[DEBUG] Worker setup completed")
    return instance, log_file, profile_log_file


# ---------------------------------------------------------------------------
# apply_top_p_top_k — parameterized pruning shared by both verifiers
# ---------------------------------------------------------------------------


def apply_top_p_top_k(log_probs):
    """Apply top-p and top-k pruning, removing filtered token entries.

    Args:
        log_probs: np.array of shape [N, 2] with columns [token_id, logprob],
                   assumed sorted by logprob descending.
    Returns:
        (filtered_log_probs, culled_prob_sum) where filtered_log_probs has the
        same [N', 2] shape with pruned rows removed.
    """
    if _w.verbose:
        print("[DEBUG] Applying top_p and top_k pruning...")

    if _w.top_p >= 1.0 and _w.top_k < 0:
        return log_probs, 0.0

    # Sort by logprob descending (in case input isn't sorted)
    order = np.argsort(log_probs[:, 1])[::-1]
    sorted_lp = log_probs[order]

    keep = len(sorted_lp)
    culled_prob_sum = 0.0

    # Top-k: keep only the k highest-logprob entries
    if _w.top_k > 0 and keep > _w.top_k:
        culled_prob_sum += np.exp(sorted_lp[_w.top_k :, 1]).sum()
        sorted_lp = sorted_lp[: _w.top_k]
        keep = _w.top_k

    # Top-p: keep smallest set whose cumulative prob >= top_p
    if _w.top_p < 1.0:
        probs = np.exp(sorted_lp[:, 1])
        cumsum = np.cumsum(probs)
        cutoff = np.searchsorted(cumsum, _w.top_p, side="right") + 1
        if cutoff < keep:
            culled_prob_sum += probs[cutoff:].sum()
            sorted_lp = sorted_lp[:cutoff]

    if _w.verbose:
        print("[DEBUG] Pruning complete")
    return sorted_lp, culled_prob_sum


def logprobs_dict_to_tensor(logprobs_dict, vocab_size=None):
    """Convert logprobs dict to tensor for existing code compatibility.

    Args:
        logprobs_dict: Dict mapping {token_id: logprob}
        vocab_size: Optional vocab size. If None, uses _w.vocab_size.

    Returns:
        torch.Tensor: Log probabilities tensor of shape (vocab_size,)
                      with -inf for missing tokens
    """
    if vocab_size is None:
        vocab_size = _w.vocab_size

    log_probs = torch.full((vocab_size,), float("-inf"))
    for token_id, logprob in logprobs_dict.items():
        log_probs[token_id] = logprob

    return log_probs


# ---------------------------------------------------------------------------
# get_grammar_mask — grammar constraint mask used by both verifiers
# ---------------------------------------------------------------------------


def get_grammar_mask(tokens):
    """Get grammar validity mask for next tokens.

    Args:
        tokens: List of token IDs
        logprobs_dict: Dict {token_id: logprob} (not used - kept for compatibility)

    Returns:
        boolean numpy array of shape (vocab_size,).
    """
    if _w.verbose:
        print("[DEBUG] Retrieving grammar mask via LLGuidance...")

    if _w.use_grammar:
        from beaver.verifiers.llguidance_grammar import get_next_token_bool_mask

        return get_next_token_bool_mask(tokens, _w.lltokenizer, _w.ebnf)

    # Return all True for vocab_size
    return np.ones(_w.vocab_size, dtype=bool)


# ---------------------------------------------------------------------------
# log_profiling — write profiling data, optionally print in verbose mode
# ---------------------------------------------------------------------------


def log_profiling(profiling_data, profile_log_file):
    """Log profiling data to file; print if verbose."""
    log_json(profiling_data, profile_log_file)
    if _w.verbose:
        print(profiling_data)



# ---------------------------------------------------------------------------
# safe_worker — decorator that catches exceptions so a worker crash returns
# an error dict instead of killing the process silently.
# ---------------------------------------------------------------------------


def safe_worker(fn):
    """Wrap a worker function so exceptions produce an error result dict."""

    @functools.wraps(fn)
    def wrapper(args):
        try:
            return fn(args)
        except Exception as e:
            try:
                idx = args[0]["idx"]
            except Exception:
                idx = "unknown"
            tb = traceback.format_exc()
            print(f"[Worker ERROR] Instance {idx} failed: {e}\n{tb}")
            return {
                "idx": idx,
                "transitions": 0,
                "time_s": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

    return wrapper
