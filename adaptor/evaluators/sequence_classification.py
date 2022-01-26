from typing import List

import torch
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head, AdaptationDataset


class Accuracy(EvaluatorBase):
    """
    Sequence classification accuracy, where each input sample of dataset falls into a single category.
    """

    compatible_heads: List[Head] = [Head.SEQ_CLASSIFICATION]
    smaller_is_better: bool = False

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        """
        Refer to the superclass documentation.
        """
        expected = []
        actual = []

        for batch in dataset:
            expected.extend(batch["labels"])
            actual.extend(model(**batch).argmax(-1))

        assert len(expected) == len(actual)

        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct / len(expected)
