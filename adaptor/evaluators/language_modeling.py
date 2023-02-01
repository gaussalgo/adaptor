import abc
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head, AdaptationDataset


class Perplexity(EvaluatorBase):

    compatible_heads: List[Head] = [Head.MLM, Head.CLM]

    @staticmethod
    def _per_batch_perplexity(model: torch.nn.Module,
                              tokenizer: PreTrainedTokenizer,
                              dataset: AdaptationDataset) -> Tuple[List[int], List[int]]:
        pass

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:

        per_batch_perplexity = 0
        num_samples = 0

        for batch in dataset:
            masks_pos = batch["input_ids"] == tokenizer.mask_token_id
            expected_tokens = batch["labels"][masks_pos]
            actual_tokens = model(**batch).logits.argmax(-1).flatten().tolist()

            expected.extend(expected_tokens)
            actual.extend(actual_tokens)

        return expected, actual

