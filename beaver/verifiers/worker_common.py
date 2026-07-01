"""Shared worker state and utilities for verification workers.

All multiprocessing worker state lives here as a single SimpleNamespace (`_w`),
replacing the 17+ individual `_worker_*` globals that were duplicated across
frontier_verifier.py.
"""

import time
import functools
import traceback
import types

from openai import OpenAI
import httpx
import numpy as np
import torch

from beaver.utils.utils import log_json
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

    print(f"[DEBUG] config_dict: {config_dict}")
    # Clear any previous state without reassigning the object
    _w.__dict__.clear()

    _w.model_name = config_dict["model_name"]  # Store model (parent nn.Module)
    _w.tokenizer = config_dict["tokenizer"]
    _w.lltokenizer = initialize_llguidance(_w.tokenizer.w_idx, _w.tokenizer.idx_w)
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
    _w.glove_embed = config_dict["glove_embed"]

    # Map each tag-vocab index to its corresponding word-vocab index so that
    # model outputs (over d_vocab_out tag classes) can be expressed in the
    # word-vocab token-ID space that the rest of BEAVER uses.
    _w.tag_to_word = np.array(
        [_w.tokenizer.w_idx.get(tag, 0) for tag in _w.tokenizer.idx_t],
        dtype=np.int64,
    )
    # Cache full-sequence logits keyed by prompt token-ID tuple so that the
    # model is only run once per instance regardless of how many frontier
    # expansions that instance triggers.
    _w._logit_cache = {}

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
    """Build the token-ID list sent to the model.

    Args:
        instance: Dict with at least ``prompt`` (list of str tokens).
        continuation: List of token IDs generated so far.

    Returns:
        List[int] of token IDs (prompt + continuation).
    """
    if _w.verbose:
        print("[DEBUG] Tokenizing prompt into ids...")

    # tokenize() expects a batch (list of sentences); wrap in outer list,
    # then take row 0 to get a flat list of token IDs.
    prompt_token_ids = _w.tokenizer.tokenize([instance["prompt"]])[0].tolist()

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
def model_generate_logprobs_transformer(instance, continuation):
    """Get next-token logprobs from the local Transformer model.

    The model outputs over d_vocab_out tag-vocab classes.  We remap those
    indices to the word-vocab indices BEAVER uses throughout via _w.tag_to_word.

    Args:
        instance: Instance dict with ``prompt`` as a list of str tokens
                  (already normalised with BOS/EOS by _prepare_dataset).
        continuation: List of word-vocab token IDs generated so far.

    Returns:
        (np.ndarray, list): logprobs_array of shape [word_vocab_size, 2] with
        columns [word_token_id, logprob], and the prompt token-ID list.
    """
    seq_len = x.shape[1]

    # Convert instance into a prompt made of token ids
    prompt_token_ids = build_prompt(instance = instance, continuation = continuation)

    if _w.verbose:
            print(f"[DEBUG] Prompt token IDs: {prompt_token_ids}")
    
    prompt_key = tuple(prompt_token_ids)

    if _w.verbose:
        print("[DEBUG] Cache miss — running forward pass...")

    # get model
    model = _w.model_name

    # convert to tensor
    x = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0)

    # run on model
    mask = torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits = model(x.to(model.device), mask=mask.to(model.device))

    tag_log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    # create vector for the indexes and then combine with the logits
    word_vocab_size = len(_w.tokenizer.idx_w)
    log_probs_remapped = np.full(word_vocab_size, -1e9, dtype=np.float64)

    # map tag indices to word indices 
    for tag_idx, word_idx in enumerate(_w.tag_to_word):
        log_probs_remapped[word_idx] = tag_log_probs[tag_idx]
    
    # combine word indices and log probs
    logprobs_w_ids = np.column_stack([
        np.arange(word_vocab_size, dtype=np.float64),
        log_probs_remapped,
    ])

    return logprobs_w_ids, prompt_token_ids


def model_generate_next_token_logprobs(instance, continuation):
    """Get next-token logprobs from the local TransformerProgram model.

    The TPM is non-autoregressive: one forward pass on the prompt produces
    logits at every output position simultaneously.  We cache that result and
    index position k = len(continuation) for the k-th generation step, so the
    model is only ever run once per instance.

    The model outputs over d_vocab_out tag-vocab classes.  We remap those
    indices to the word-vocab indices BEAVER uses throughout via _w.tag_to_word.

    Args:
        instance: Instance dict with ``prompt`` as a list of str tokens
                  (already normalised with BOS/EOS by _prepare_dataset).
        continuation: List of word-vocab token IDs generated so far.

    Returns:
        (np.ndarray, list): logprobs_array of shape [word_vocab_size, 2] with
        columns [word_token_id, logprob], and the prompt token-ID list.
    """
    try:
        # Convert instance into a prompt made of token ids
        prompt_token_ids = build_prompt(instance = instance, continuation = None)

        if _w.verbose:
            print(f"[DEBUG] Prompt token IDs: {prompt_token_ids}")

        # ── Run model once per prompt; cache all position logits ──────────
        prompt_key = tuple(prompt_token_ids)
        if prompt_key not in _w._logit_cache:
            if _w.verbose:
                print("[DEBUG] Cache miss — running forward pass...")
            model = _w.model_name
            x = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0)
            seq_len = x.shape[1]
            # Full (non-causal) attention: each output position must see all
            # input positions simultaneously for the RASP sort algorithm to
            # compute element ranks correctly.
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                logits = model(x.to(model.device), mask=mask.to(model.device))
            # Store [seq_len, d_vocab_out] on CPU; drop the batch dim.
            _w._logit_cache[prompt_key] = logits[0].cpu()

        all_logits = _w._logit_cache[prompt_key]  # [seq_len, d_vocab_out]

        # Find the correct logits to return, range of 1 to seq_len-2 because of BOS and EOS tokens
        k = len(continuation)
        output_pos = k + 1
        max_output_pos = all_logits.shape[0] - 1  # exclude trailing </s> position

        word_vocab_size = len(_w.tokenizer.idx_w)
        log_probs_remapped = np.full(word_vocab_size, -1e9, dtype=np.float64)

        if output_pos < max_output_pos:
            logits_k = all_logits[output_pos]  # [d_vocab_out]
            tag_log_probs = (
                torch.nn.functional.log_softmax(logits_k.float(), dim=-1)
                .numpy()
            )
            # Scatter tag-vocab log-probs into the word-vocab slots.
            for tag_idx, word_idx in enumerate(_w.tag_to_word):
                log_probs_remapped[word_idx] = tag_log_probs[tag_idx]
        else:
            # Beyond the model's output length: signal end-of-sequence via </s>.
            # Has to happen because a TPM does not have EOS in its vocab, and is non-autoregressive
            eos_id = _w.tokenizer.w_idx.get("</s>", 1)
            log_probs_remapped[eos_id] = 0.0

        if _w.verbose:
            print(f"[DEBUG] Generation step k={k}, output_pos={output_pos}")

        logprobs_w_ids = np.column_stack([
            np.arange(word_vocab_size, dtype=np.float64),
            log_probs_remapped,
        ])

        return logprobs_w_ids, prompt_token_ids

    except Exception as e:
        print(f"[ERROR] An error occurred when attempting to compute the next token log probabilities, {e}")

def model_generate_logprobs(instance):
    """Get next-token logprobs from the local TransformerProgram model.

    The TPM is non-autoregressive: one forward pass on the prompt produces
    logits at every output position simultaneously. We return all logits here
    as log probabilities.

    The model outputs over d_vocab_out tag-vocab classes.  We remap those
    indices to the word-vocab indices BEAVER uses throughout via _w.tag_to_word.

    Args:
        instance: Instance dict with ``prompt`` as a list of str tokens
                  (already normalised with BOS/EOS by _prepare_dataset).

    Returns:
        (np.ndarray, list): logprobs_array of shape [word_vocab_size, 2] with
        columns [word_token_id, logprob], and the prompt token-ID list.
    """
    try:
        # Convert instance into a prompt made of token ids
        prompt_token_ids = build_prompt(instance = instance, continuation = None)

        if _w.verbose:
            print(f"[DEBUG] Prompt token IDs: {prompt_token_ids}")

        # ── Run model once per prompt; cache all position logits ──────────
        prompt_key = tuple(prompt_token_ids)
        
        if _w.verbose:
            print("[DEBUG] Cache miss — running forward pass...")
        if prompt_key not in _w._logit_cache:
            model = _w.model_name
            x = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0)
            seq_len = x.shape[1]
            # Full (non-causal) attention: each output position must see all
            # input positions simultaneously for the RASP sort algorithm to
            # compute element ranks correctly.
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                logits = model(x.to(model.device), mask=mask.to(model.device))
            # Store [seq_len, d_vocab_out] on CPU; drop the batch dim.
            _w._logit_cache[prompt_key] = logits[0].cpu()

        output_logits = _w._logit_cache[prompt_key][1:-1] # [N-2, d_vocab_out]

        # create a mask?
        N_out, V = output_logits.shape

        log_probs_remapped = np.full((N_out, V), -1e9, dtype=np.float64) # holds our probs
        tag_log_probs = torch.nn.functional.log_softmax(output_logits, dim=-1).numpy() # softmax logits

        # create a NumPy array for the token ids
        token_ids = np.broadcast_to(np.arange(V, dtype=np.float64), (N_out, V))  # [N-2, V]

        # convert tag index to word index
        log_probs_remapped = np.full(
            (N_out, V), -1e9, dtype=np.float64
        )  # [N-2, V]
        log_probs_remapped[:, _w.tag_to_word] = tag_log_probs

        model_logprobs = np.stack([token_ids, log_probs_remapped], axis=-1)       # [N-2, V, 2]

        return model_logprobs, prompt_token_ids

    except Exception as e:
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

def apply_top_p_top_k_tpm(log_probs):
    """Apply top-p and top-k pruning, removing filtered token entries.
       For use with Transformer Program Model logits

    Args:
        log_probs: np.array of shape [N, V, 2] with columns [token_id, logprob],
                   assumed sorted by logprob descending, where N is the sequence length 
                   of the prompt
    Returns:
        (filtered_log_probs, culled_prob_sum) where filtered_log_probs has the
        same [N, V, 2] shape with pruned tokens set to -1e9.
    """
    if _w.verbose:
        print("[DEBUG] Applying top_p and top_k pruning for a TPM...")

    if _w.top_p >= 1.0 and _w.top_k < 0:
        return log_probs, 0.0

    # Sort by logprob descending (in case input isn't sorted)
    order = np.argsort(log_probs[:, :, 1], axis=1)[:, ::-1]          # [N, V]
    sorted_lp = np.take_along_axis(log_probs, order[:, :, np.newaxis], axis=1)  # [N, V, 2]

    keep = sorted_lp.shape[1]  # V — number of vocab entries per position
    culled_prob_sum = np.zeros(sorted_lp.shape[0], dtype=np.float64)  # [N]

    # Top-k: keep only the k highest-logprob entries per position
    if _w.top_k > 0 and keep > _w.top_k:
        culled_prob_sum += np.exp(sorted_lp[:, _w.top_k:, 1]).sum(axis=1)

        indices = np.arange(sorted_lp.shape[1])[np.newaxis, :]  # [1, V]
        cull_mask = indices >= _w.top_k                           # [1, V] broadcasts to [N, V]

        sorted_lp[:, :, 1] = np.where(cull_mask, -1e9, sorted_lp[:, :, 1])
        keep = _w.top_k

    # Top-p: per position, keep smallest set whose cumulative prob >= top_p
    if _w.top_p < 1.0:
        probs = np.exp(sorted_lp[:, :, 1])           # [N, V]
        cumsum = np.cumsum(probs, axis=1)             # [N, V]
        # First index where cumsum reaches top_p, per position — gives [N]
        cutoff = np.where(
            np.any(cumsum >= _w.top_p, axis=1),
            np.argmax(cumsum >= _w.top_p, axis=1) + 1,
            keep,  # never reached threshold — keep everything
        )

        indices = np.arange(sorted_lp.shape[1])[np.newaxis, :]  # [1, V]
        cull_mask = indices >= cutoff[:, np.newaxis]              # [N, V]

        if np.any(cutoff < keep):
            culled_prob_sum += (probs * cull_mask).sum(axis=1)
            sorted_lp[:, :, 1] = np.where(cull_mask, -1e9, sorted_lp[:, :, 1])

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
