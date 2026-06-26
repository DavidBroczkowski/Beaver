import multiprocessing as mp
import os
import json
from abc import ABC, abstractmethod
from typing import Optional
import torch
import sys

from tqdm import tqdm
from beaver.utils.tokenizer_utils import NLTK_Tokenizer
from beaver.utils.programs import (
    TransformerProgramModel,
    softmax, softmax_no_temp, gumbel_hard, gumbel_soft, argmax,
)
from beaver.utils.transformers import Transformer

_SAMPLE_FN_MAP = {
    "softmax": softmax,
    "softmax_no_temp": softmax_no_temp,
    "gumbel_hard": gumbel_hard,
    "gumbel_soft": gumbel_soft,
    "argmax": argmax,
}

import beaver.constraints  # ensures all dataset modules are registered


class BaseVerifier(ABC):
    def __init__(
        self,
        model,
        dataset,
        prompts,
        grammar,
        semantic_symbol=None,
        **kwargs,
    ) -> None:
        self.grammar = grammar

        # Grammar loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grammars_dir = os.path.join(current_dir, "..", "grammars")

        grammar_file = None
        if grammar is not None:
            for suffix in ["_grammar_sglang.lark", "_grammar.lark"]:
                potential_file = os.path.join(grammars_dir, f"{grammar}{suffix}")
                if os.path.exists(potential_file):
                    grammar_file = potential_file
                    break
            if grammar_file is None:
                raise FileNotFoundError(
                    f"Could not find grammar file for '{grammar}' in {grammars_dir}"
                )
            with open(grammar_file, "r") as f:
                self.ebnf = f.read()
        else:
            self.ebnf = None

        from beaver.constraints.base_constraints import _REGISTRY
        if dataset not in _REGISTRY:
            raise ValueError(
                f"Unknown dataset: {dataset}. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        self.dataset_name = dataset
        self.use_cache: bool = kwargs.get("use_cache", True)

        self.tokenizer = NLTK_Tokenizer(prompts)
        self.semantic_symbol = semantic_symbol

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
            loaded_model = Transformer(d_vocab=len(self.tokenizer.idx_w), **model_args_dict)
        else:
            raise ValueError(
                f"model_type must be \"program\" or \"transformer\", got {kwargs['model_type']!r}"
            )

        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        self.model_name = loaded_model

        # Common generation parameters (previously duplicated in subclasses)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", -1)
        self.max_iterations = kwargs.get("max_iterations", 1000)
        self.epsilon = kwargs.get("epsilon", 0.01)
        self.eos_tokens: list[int] = [
            tok_id
            for attr in ["eos_token_id", "pad_token_id"]
            if (tok_id := kwargs.get(attr, getattr(self.tokenizer, attr, None))) is not None
        ]
        self.gen_length: int = kwargs.get("gen_length", 128)
        self.verbose: bool = kwargs.get("verbose", False)
        self.max_workers = kwargs.get("max_workers", 1)
        self.num_logprobs = kwargs.get("num_logprobs", 100)
        self.use_grammar = kwargs.get("use_grammar", True)
        self.chat_mode = kwargs.get("chat_mode", False)
        self.system_message = kwargs.get("system_message", None)
        self.fewshot_messages = kwargs.get("fewshot_messages", [])
        self.glove_embed = kwargs["glove_embed"]

    def _build_worker_config(self):
        """Build a pickleable config dict for init_worker_state()."""
        from beaver.constraints.base_constraints import _REGISTRY

        check_call_fn, instance_context_fn, check_fn = _REGISTRY[self.dataset_name]
        return {
            "model_name": self.model_name,
            "ebnf": self.ebnf,
            "dataset_name": self.dataset_name,
            "use_cache": self.use_cache,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "eos_tokens": self.eos_tokens,
            "gen_length": self.gen_length,
            "epsilon": self.epsilon,
            "num_logprobs": self.num_logprobs,
            "verbose": self.verbose,
            "max_iterations": self.max_iterations,
            "semantic_symbol": self.semantic_symbol,
            "use_grammar": self.use_grammar,
            "chat_mode": self.chat_mode,
            "system_message": self.system_message,
            "fewshot_messages": self.fewshot_messages,
            "check_call_fn": check_call_fn,
            "instance_context_fn": instance_context_fn,
            "check_fn": check_fn,
            "tokenizer": self.tokenizer,
            "glove_embed": self.glove_embed,
        }

    def _run_pool(self, dataset, run_log_dir, worker_fn, init_fn, config):
        """Run worker_fn over dataset using multiprocessing or single-thread.

        Workers should return a dict with at least:
            idx, transitions, time_s, and verifier-specific fields.
        The tqdm bar updates as each instance finishes (unordered) and
        shows running averages in the postfix.
        """
        results = []
        worker_args = [(instance, run_log_dir) for instance in dataset]
        total_time = 0.0
        total_transitions = 0

        def _update_bar(bar, result):
            nonlocal total_time, total_transitions
            if result is not None:
                total_time += result.get("time_s", 0)
                total_transitions += result.get("transitions", 0)
                n_done = bar.n + 1
                bar.set_postfix_str(
                    f"avg {total_time / n_done:.1f}s/inst, "
                    f"avg {total_transitions / n_done:.0f} trans/inst, "
                    f"last: {result.get('transitions', '?')} trans "
                    f"{result.get('time_s', 0):.1f}s"
                )
            bar.update(1)

        if self.max_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=self.max_workers,
                initializer=init_fn,
                initargs=(config,),
            ) as pool:
                bar = tqdm(total=len(dataset), desc="Processing instances")
                for result in pool.imap_unordered(worker_fn, worker_args, chunksize=1):
                    results.append(result)
                    _update_bar(bar, result)
                bar.close()
        else:
            init_fn(config)
            bar = tqdm(total=len(dataset), desc="Processing instances")
            for instance in dataset:
                result = worker_fn((instance, run_log_dir))
                results.append(result)
                _update_bar(bar, result)
            bar.close()

        return results

    @abstractmethod
    def __call__(self, dataset, run_log_dir):
        raise NotImplementedError
