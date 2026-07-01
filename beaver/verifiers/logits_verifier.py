import time
import torch
import json
import numpy as np

from beaver.constraints.base_constraints import (
    enforce_semantic_constraint,
)
from beaver.utils.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.utils.glove_emb_utils import get_glove_embeddings
from beaver.verifiers.worker_common import (
    _w,
    init_worker_state,
    log_profiling,
    model_generate_logprobs,
    safe_worker,
    worker_setup,
)
from beaver.utils.programs import (
    TransformerProgramModel,
    softmax, softmax_no_temp, gumbel_hard, gumbel_soft, argmax,
)

_SAMPLE_FN_MAP = {
    "softmax": softmax,
    "softmax_no_temp": softmax_no_temp,
    "gumbel_hard": gumbel_hard,
    "gumbel_soft": gumbel_soft,
    "argmax": argmax,
}

def verify_tpm_via_logits(model_logprobs, prompt_w_ids, instance):
    """
    Compute bounds on a Transformer Program Model (TPM) through multiplication
    of logits from model_logprobs.

    Inputs:
        - model_logprobs: a [N, V, 2] NumPy array containing the log probabilities of the entire output and their associated token ids
        - prompt_w_ids: a List[int] contaning the prompt token ids
    Outputs:
        - result: a float containing the probability that a satisfying output is given
        - culled_prob_sum: the resulting probability whose state is unknown and was pruned via Top P or Top K
    """
    import functools

    N, V = model_logprobs[:, :, 1].shape # note, log probs here are the second one, ids come first

    # prune the probs
    sorted_lp, culled_prob_sum = apply_top_p_top_k_tpm(model_logprobs)

    #resize back to original [N, V]
    log_probs_ordered = np.full((N, V), -1e9, dtype=np.float64)

    token_ids = sorted_lp[:, :, 0].astype(int)  # [N, V]
    log_probs  = sorted_lp[:, :, 1]              # [N, V]

    n_idx = np.arange(N)[:, np.newaxis]          # [N, 1] for broadcasting
    log_probs_ordered[n_idx, token_ids] = log_probs

    log_joint = functools.reduce(
        lambda acc, pos_lp: acc[..., np.newaxis] + pos_lp,
        log_probs_ordered   # iterates over N positions, each slice is [V]
    )
    # shape [V, V, ..., V] — N dimensions
    # log_joint[v0, v1, v2, ...] = log P(y0=v0) + log P(y1=v1) + ...

    probs = np.exp(log_joint).ravel()                      # [V^N]
    idx = np.indices([V] * (N)).reshape(N, -1).T             # [V^N, N] — token IDs per sequence

    # filter near-zero sequences 
    valid = probs > 1e-30
    probs, idx = probs[valid], idx[valid]

    # decode all sequences to strings 
    decoded = np.array([
        " ".join(_w.tokenizer.idx_w[v] for v in seq)
        for seq in idx
    ])

    # check constraint for each decoded sequence
    # no need for pre-check here as we know all of correct length
    # may change later as the pre-checks differ between tasks
    satisfies = enforce_semantic_constraint(dataset_name=_w.dataset_name, instance=instance, decoded_sequences=decoded)

    # sum probability mass of satisfying sequences
    result = probs[satisfies].sum()

    return result, culled_prob_sum

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

@safe_worker
def _worker_process_instance(args):
    # setup worker
    instance, log_file, profile_log_file = worker_setup(args)

    instance_start_time = time.time()

    model_generate_time = time.time()

    # generate the probabilities of the output based on the model
    model_logprobs, final_prompt = model_generate_logprobs(instance)

    # compute the exact probability via the logits
    verify_start_time = time.time()
    satisfy_prob, culled_prob_sum = verify_tpm_via_logits(model_logprobs, final_prompt, instance)
    verify_end_time = time.time()

    # Have to sum as culled_prob_sum gives probs by position, so sum over the positions
    culled_prob_sum = np.sum(culled_prob_sum)

    step_end_time = time.time()

    # log results
    running_results = {
        "exact_prompt": final_prompt,
        "violation prob sum": 1.0 - (satisfy_prob + culled_prob_sum),
        "pruned prob sum": culled_prob_sum,
        "upper_bound": satisfy_prob + culled_prob_sum,
        "lower_bound": satisfy_prob,
    }
    log_json(running_results, log_file)
    profiling_data = {
        "model_generate": verify_start_time - model_generate_time,
        "check_validity": verify_end_time - verify_start_time,
        "total_time": step_end_time - instance_start_time,
    }
    log_profiling(
        profiling_data,
        profile_log_file,
    )
    
    instance_end_time = time.time()
    return {
        "idx": instance["idx"],
        **running_results,
        "instance_run_time": instance_end_time - instance_start_time,
    }

class LogitsVerifier(BaseVerifier):
    """
    A verifier that uses the logits produced by the non-autoregressive Transformer Program Model
    to create bounds on the satisfiability of its output on given input. 
    """
    def __init__(self, model, dataset, prompts, **kwargs):
        super().__init__(model, dataset, prompts, **kwargs)

        model = self.model_name
        # load model
        with open(kwargs["model_args"], 'r') as args_file:
            model_args_dict = json.load(args_file)

        state_dict = torch.load(model, weights_only=False)

        if kwargs["model_type"] == "program":
            # 'max_length' is the training-time name for the sequence-length param;
            # TransformerProgramModel calls it 'n_ctx'.
            if 'n_ctx' not in model_args_dict:
                if 'max_length' in model_args_dict:
                    model_args_dict['n_ctx'] = model_args_dict['max_length']
                elif 'pos_embed.W' in state_dict:
                    model_args_dict['n_ctx'] = state_dict['pos_embed.W'].shape[0]
            # Infer d_vocab_out from the checkpoint's unembed weight; the output
            # vocabulary (tags) is often smaller than the input vocabulary.
            if 'd_vocab_out' not in model_args_dict and 'unembed.W_U' in state_dict:
                model_args_dict['d_vocab_out'] = state_dict['unembed.W_U'].shape[1]
            # args.json stores sample_fn as a string name; resolve to the actual
            # callable.  Use argmax at inference for deterministic discrete behaviour.
            if isinstance(model_args_dict.get('sample_fn'), str):
                model_args_dict['sample_fn'] = _SAMPLE_FN_MAP.get(
                    model_args_dict['sample_fn'], argmax
                )
            loaded_model = TransformerProgramModel(
                d_vocab=len(self.tokenizer.idx_w),
                idx_t=self.tokenizer.idx_t,
                **model_args_dict,
            )
        #FIXME: move this transformer code to the frontier verifier!
        # elif kwargs["model_type"] == "transformer":
        #     if 'n_ctx' not in model_args_dict and 'max_length' in model_args_dict:
        #         model_args_dict['n_ctx'] = model_args_dict['max_length']
        #     loaded_model = Transformer(d_vocab=len(self.tokenizer.idx_w), **model_args_dict)
        else:
            raise ValueError(
                f"model_type must be \"program\" or \"transformer\", got {kwargs['model_type']!r}"
            )

        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        self.model_name = loaded_model

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
        if self.verbose:
            print("[DEBUG] Retrieving glove embeddings...")
        config["idx_emb"] = get_glove_embeddings(self.tokenizer.idx_w, "data/glove.840B.300d.txt")
        if self.verbose:
            print("[DEBUG] Retrieved glove embeddings")
        config["vocab_size"] = len(self.tokenizer.idx_w)
    
        return self._run_pool(
            dataset,
            run_log_dir,
            worker_fn=_worker_process_instance,
            init_fn=init_worker_state,
            config=config,
        )