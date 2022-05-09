import abc
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head, AdaptationDataset


class TokenClassificationEvaluator(EvaluatorBase, abc.ABC):
    """
    Base class of token classification evaluators. Inputs format is constraint by `compatible_heads`,
    checked when the evaluator is passed into the evaluated objective.
    """
    compatible_heads: List[Head] = [Head.TOKEN_CLASSIFICATION]

    def _collect_token_predictions(self,
                                   model: torch.nn.Module,
                                   dataset: AdaptationDataset) -> Tuple[List[int], List[int]]:
        expected = []
        actual = []

        for batch in dataset:
            expected.extend(batch["labels"].flatten().tolist())
            actual.extend(model(**batch).logits.argmax(-1).flatten().tolist())

        assert len(expected) == len(actual), "A number of model predictions does not match inputs."

        return expected, actual


class MeanFScore(TokenClassificationEvaluator):
    """
    Per-category F1-score of token classification,
    each input sample of dataset gets a sequence of `labels` of `num_input_ids` length.
    """

    smaller_is_better: bool = False

    @staticmethod
    def _per_category_fscore(category_i: int, expected: List[int], actual: List[int]) -> float:
        """
        F-Score for a single category, given a flattened list of expected and actual labels.
        """
        true_pos = sum(exp == act == category_i for exp, act in zip(expected, actual))
        false_pos = sum((exp != act) and (act == category_i) for exp, act in zip(expected, actual))
        false_neg = sum((exp != act) and (act != category_i) for exp, act in zip(expected, actual))

        return (true_pos /
                (true_pos + (1/2 * (false_pos + false_neg))))

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset,
                 ignored_index: int = -100) -> float:
        """
        Refer to the superclass documentation.
        """
        expected, actual = self._collect_token_predictions(model, dataset)

        all_categories = set(cat_i for cat_i in expected if cat_i != ignored_index)
        per_category_fscores = [self._per_category_fscore(cat_i, expected, actual) for cat_i in all_categories]

        return sum(per_category_fscores) / len(all_categories)


class AverageAccuracy(TokenClassificationEvaluator):

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset,
                 ignored_index: int = -100) -> float:

        expected, actual = self._collect_token_predictions(model, dataset)

        num_matching = sum(e_i == a_i for e_i, a_i in zip(expected, actual) if e_i != ignored_index)
        num_all = sum(e_i != ignored_index for e_i in expected)

        return num_matching / num_all
