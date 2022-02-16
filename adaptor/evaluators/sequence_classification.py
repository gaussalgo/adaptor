import abc
from typing import List

import torch
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head, AdaptationDataset


class SeqClassificationEvaluator(EvaluatorBase, abc.ABC):
    """
    Base class of sequence classification evaluators. Inputs format is constraint by `compatible_heads`,
    checked when the evaluator is passed into the evaluated objective.
    """
    compatible_heads: List[Head] = [Head.SEQ_CLASSIFICATION]


class SequenceAccuracy(SeqClassificationEvaluator):
    """
    Sequence classification accuracy, where each input sample of dataset falls into a single category.
    """

    smaller_is_better: bool = False

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        """
        Refer to the superclass documentation.
        """
        expected = []
        actual = []

        for batch in dataset:
            expected.extend(batch["labels"])
            actual.extend(model(**batch).logits.argmax(-1))

        assert len(expected) == len(actual)

        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct.item() / len(expected)
