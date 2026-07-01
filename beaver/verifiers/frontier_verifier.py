"""Frontier (branch-and-bound) verifier for LLM verification."""

import time
import torch
import json
import numpy as np

from beaver.constraints.base_constraints import (
    check_semantic_call,
    enforce_semantic_constraint,
)
from beaver.utils.utils import log_json
from beaver.utils.glove_emb_utils import (
    get_glove_embeddings
)

from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier import Frontier, FrontierElement
from beaver.utils.transformers import Transformer
from beaver.verifiers.worker_common import (
    _w,
    apply_top_p_top_k,
    apply_top_p_top_k_tpm,
    get_grammar_mask,
    init_worker_state,
    log_profiling,
    model_generate_next_token_logprobs,
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


@safe_worker
def _worker_process_instance(args):
    """Top-level function for multiprocessing — processes a single instance."""
    instance, log_file, profile_log_file = worker_setup(args)

    def update_frontier(frontier, previous_element, log_probs, bit_mask):
        """Expand a frontier element, filtering by grammar + semantics.

        Args:
            frontier: The Frontier object
            previous_element: FrontierElement being expanded
            log_probs: numpy array of shape [N, 2] with [token_id, logprob] pairs
            bit_mask: Grammar validity mask (torch.Tensor or np.ndarray)

        Returns:
            (new_elements, delta_incomplete_prob_sum, delta_complete_prob_sum,
             presemantic_check_time, semantic_check_time)
        """

        # Convert bit_mask to numpy if it's a torch tensor
        if isinstance(bit_mask, torch.Tensor):
            bit_mask = bit_mask.cpu().numpy()

        # Extract token IDs from logprobs array and filter by grammar mask
        all_token_ids = log_probs[:, 0].astype(int)
        in_vocab_mask = all_token_ids < len(bit_mask)
        valid_indices = all_token_ids[in_vocab_mask]

        valid_mask = bit_mask[valid_indices]
        valid_indices = valid_indices[valid_mask]

        if _w.verbose:
            print(f"in_vocab_mask sum: {np.sum(in_vocab_mask)} / {len(in_vocab_mask)}")
            print(f"Valid mask sum: {np.sum(valid_mask)} / {len(valid_mask)}")
            print(f"Valid indices {len(valid_indices)}")

        if len(valid_indices) == 0:
            return [], 0, 0, 0, 0

        # Create a dict for fast logprob lookup: {token_id: logprob}
        logprobs_dict = {
            int(log_probs[i, 0]): log_probs[i, 1] for i in range(len(log_probs))
        }

        # Decode tokens individually — idx_w[i] gives a single string directly;
        # decode() returns a list and would produce a 2-D numpy array.
        decoded_tokens = np.array(
            [
                "" if i in _w.eos_tokens else _w.tokenizer.idx_w[i]
                for i in valid_indices
            ]
        )

        # Construct full decoded sequences as space-joined strings.
        current_decoded = " ".join(_w.tokenizer.decode(previous_element.tokens, skip_special_tokens=True)).strip()
        decoded_sequences = np.array(
            [(current_decoded + " " + tok).strip() for tok in decoded_tokens]
        )

        # Construct full token lists
        token_lists = np.array(
            [
                previous_element.tokens + [int(valid_indices[idx])]
                for idx in range(len(valid_indices))
            ],
            dtype=object,
        )

        # Determine completion flags
        if len(previous_element.tokens) >= _w.gen_length - 1:
            complete_flag = np.ones(len(decoded_sequences), dtype=bool)
        else:
            complete_flag = np.array(
                [i in _w.eos_tokens for i in valid_indices], dtype=bool
            )

        if _w.verbose:
            print(
                f"Complete flag: {sum(complete_flag)}: "
                f"{[i for i in valid_indices[complete_flag]]}"
            )
            print(f"Worker eos tokens: {_w.eos_tokens}")

        presemantic_check_time = time.time()

        # Semantic checking
        semantic_check_mask = np.logical_or(
            check_semantic_call(
                _w.dataset_name, instance, decoded_sequences, token_lists
            ),
            complete_flag,
        )
        semantic_check_indices = np.where(semantic_check_mask)[0]

        semantic_correct_indices = np.array([], dtype=np.intp)
        if len(semantic_check_indices) > 0:
            sequences_to_check = decoded_sequences[semantic_check_indices]
            semantic_correctness_mask = enforce_semantic_constraint(
                _w.dataset_name, instance, sequences_to_check, use_cache=_w.use_cache
            )
            semantic_correct_indices = semantic_check_indices[semantic_correctness_mask]

        semantic_check_time = time.time()

        if _w.verbose:
            print(f"Check mask: {sum(semantic_check_mask)}")
            print(f"Check indices: {valid_indices[semantic_check_indices]}")
            print(f"Correct indices: {valid_indices[semantic_correct_indices]}")

        violations = set(valid_indices[semantic_check_indices]) - set(
            valid_indices[semantic_correct_indices]
        )
        non_violations = set(valid_indices) - set(violations)

        total_violation_prob = np.sum(
            np.exp(
                np.array(
                    [previous_element.logprob + logprobs_dict[v] for v in violations]
                )
            )
        ).item()

        # Build new frontier elements
        new_elements = []

        for idx in range(len(valid_indices)):
            token_id = int(valid_indices[idx])
            if token_id in violations:
                continue
            new_tokens = previous_element.tokens + [token_id]
            new_elem = FrontierElement(
                element_id=frontier.total_elements,
                token=token_id,
                tokens=new_tokens,
                logprob=previous_element.logprob + logprobs_dict[token_id],
                is_completed=complete_flag[idx].item(),
            )

            frontier.total_elements += 1
            new_elements.append(new_elem)

        return (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            len(violations),
            total_violation_prob,
        )

    # ── Main processing logic ────────────────────────────────────────

    instance_start_time = time.time()
    transitions = 0
    frontier = Frontier(
        max_size=_w.gen_length,
        scoring_strategy=_w.frontier_scoring_strategy,
    )
    incomplete_prob_sum = 1.0
    complete_prob_sum = 0.0
    pruned_prob_sum = 0.0
    violation_prob_sum = 0.0

    running_results = {}

    while transitions < _w.max_iterations:

        # pick the top incomplete element from the frontier
        start_step_time = time.time()
        element = frontier.pick_top_incomplete()
        if element is None:
            if _w.verbose:
                print(f"Frontier is empty at transition {transitions}")
            break

        model_generate_time = time.time()

        # --- generate the probabilities of the model --------------------------------
        if _w.verbose:
            print("[DEBUG] Beginning logprob generation...")

        model_logprobs, final_prompt = model_generate_next_token_logprobs(
            instance, element.tokens
        )

        # apply top-p and top-k, no need to change this here
        # Note: for RASP tasks, our model is very good, with probs around 0.99 for a certain token
        # therefore, this step will usually limit the Frontier to the correct answer
        # and will have a low number of leaves in it as a result
        logprobs, reduced_logprobs = apply_top_p_top_k(model_logprobs)

        # Calculate pruned prob (from tokens that are not counted)
        culled_prob_sum = np.exp(element.logprob) * max(
            1 - np.sum(np.exp(logprobs[:, 1])), 0.0
        )
        check_validity_time = time.time()

        vocab_mask = get_grammar_mask(element.tokens)

        if _w.verbose:
            print("[DEBUG] Retrieved grammar mask")

        if _w.verbose:
            print("[DEBUG] Begin frontier updating...")
        (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            num_violations,
            total_violation_prob,
        ) = update_frontier(frontier, element, logprobs, vocab_mask)

        if _w.verbose:
            print("[DEBUG] Frontier updated")

        frontier.add_to_element(element, new_elements)

        if _w.verbose:
            print("[DEBUG] Begin pruning incomplete leaves...")

        frontier_pruned_prob = frontier.prune_incomplete_leaves(
            topp=_w.frontier_topp, topk=_w.frontier_topk
        )

        if _w.verbose:
            print("[DEBUG] Incomplete leaves pruned")

        update_results_time = time.time()

        incomplete_prob_sum -= np.exp(element.logprob)
        incomplete_prob_sum -= frontier_pruned_prob
        for elem in new_elements:
            if elem.is_completed:
                complete_prob_sum += np.exp(elem.logprob)
            else:
                incomplete_prob_sum += np.exp(elem.logprob)

        incomplete_prob_sum = max(
            incomplete_prob_sum, 0.0
        )  # Guard against negative probabilities
        complete_prob_sum = min(
            complete_prob_sum, 1.0
        )  # Guard against probabilities > 1.0

        pruned_prob_sum += culled_prob_sum + frontier_pruned_prob

        violation_prob_sum += total_violation_prob

        upper_bound = min(
            incomplete_prob_sum + complete_prob_sum + pruned_prob_sum, 1.0
        )

        lower_bound = complete_prob_sum

        if _w.verbose:
            print("[DEBUG] Bounds updated")

        end_step_time = time.time()
        running_results = {
            "transition": transitions,
            "expanded element": element.tokens,
            "decoded element": _w.tokenizer.decode(element.tokens),
            "exact_prompt": final_prompt,
            "num_violations": num_violations,
            "total_violation_prob": total_violation_prob,
            "num_new_elements": len(new_elements),
            "incomplete_size": len(frontier._incomplete_leaves),
            "complete_size": len(frontier._complete_leaves),
            "incomplete prob sum": incomplete_prob_sum,
            "complete prob sum": complete_prob_sum,
            "violation prob sum": violation_prob_sum,
            "pruned prob sum": pruned_prob_sum,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
        }
        log_json(running_results, log_file)
        profiling_data = {
            "element_selection_time": model_generate_time - start_step_time,
            "model_generate": check_validity_time - model_generate_time,
            "grammar_mask": presemantic_check_time - check_validity_time,
            "semantic_check": semantic_check_time - presemantic_check_time,
            "frontier_add": update_results_time - semantic_check_time,
            "check_validity": update_results_time - check_validity_time,
            "update_results": end_step_time - update_results_time,
            "total_time": end_step_time - start_step_time,
        }
        log_profiling(
            profiling_data,
            profile_log_file,
        )

        if _w.verbose:
            print(json.dumps(running_results, indent=2))
            frontier.debug_frontier(_w.tokenizer)

        transitions += 1

        # this is a little interesting. In the paper they said if it got above epsilon it flagged
        # but here it is 10 times epsilon? Maybe I am getting something confused?
        if pruned_prob_sum > 10 * _w.epsilon:
            raise RuntimeError(
                f"Error: Pruned probability sum {pruned_prob_sum} exceeds reasonable threshold {10 * _w.epsilon} at transition {transitions}"
            )

        if upper_bound - lower_bound < _w.epsilon:
            if _w.verbose:
                print(
                    f"Ending frontier analysis {instance['idx']} "
                    f"since incomplete probability is below epsilon"
                )
            break

    instance_end_time = time.time()
    return {
        "idx": instance["idx"],
        **running_results,
        "instance_run_time": instance_end_time - instance_start_time,
    }

class FrontierVerifier(BaseVerifier):
    def __init__(self, model, dataset, prompts, **kwargs):
        super().__init__(model, dataset, prompts, **kwargs)
        self.frontier_topp = kwargs.get("max_frontier_prob", 1.0)
        self.frontier_topk = kwargs.get("max_frontier_size", -1)
        self.frontier_scoring_strategy = kwargs.get(
            "frontier_scoring_strategy", "highest-prob"
        )

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
        elif kwargs["model_type"] == "transformer":
            if 'n_ctx' not in model_args_dict and 'max_length' in model_args_dict:
                model_args_dict['n_ctx'] = model_args_dict['max_length']
            if 'd_vocab_out' not in model_args_dict and 'unembed.W_U' in state_dict:
                model_args_dict['d_vocab_out'] = state_dict['unembed.W_U'].shape[1]
            loaded_model = Transformer(
                d_vocab=model_args_dict['vocab_size'],
                idx_t=self.tokenizer.idx_t,
                **model_args_dict
            )
        else:
            raise ValueError(
                f"model_type must be \"program\" or \"transformer\", got {kwargs['model_type']!r}"
            )

        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        self.model_name = loaded_model

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
        config["frontier_topp"] = self.frontier_topp
        config["frontier_topk"] = self.frontier_topk
        config["frontier_scoring_strategy"] = self.frontier_scoring_strategy

        if self.verbose:
            print("[DEBUG] Retrieving glove embeddings...")
        config["idx_emb"] = get_glove_embeddings(self.tokenizer.idx_w, "data/glove.840B.300d.txt")
        if self.verbose:
            print("[DEBUG] Retrieved glove embeddings")

        config["vocab_size"] = len(self.tokenizer.idx_w)

        # dataset = self._tokenize_dataset(dataset)

        return self._run_pool(
            dataset,
            run_log_dir,
            worker_fn=_worker_process_instance,
            init_fn=init_worker_state,
            config=config,
        )
