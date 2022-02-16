import abc
from typing import List

import torch
from transformers import PreTrainedTokenizer

from adaptor.utils import Head, AdaptationDataset


class EvaluatorBase(abc.ABC):
    """
    Base class of all evaluators, usable by the instances of Objectives.
    Subclass and implement this interface to create new evaluators:
    1. pick compatible heads: determines which objectives can be evaluated using your evaluator instance.
    2. fill in `smaller_is_better`: if smaller values of your evaluator are better (like 'loss'),
    or worse (like 'accuracy').
    3. Either override `__call__` or only `evaluate_str` (in case of generative evaluators).
       Note that __call__ performs its own iteration over the eval Dataset.
       See some implemented evaluators for examples.
    """

    compatible_heads: List[Head]
    smaller_is_better: bool

    def __init__(self, decides_convergence: bool = False):
        """
        Shared init of all Evaluator instances.
        :param decides_convergence: whether this evaluator is used for an early-stopping of its objective
                                    (when given one of StoppingStrategy.*ObjectiveConverged)
        """
        self.determines_convergence = decides_convergence

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        """
        Performs an evaluation of this `model` on a given `dataset`.
        :param model: A model to be evaluated. `dataset` batches are passed as inputs to this `model`.
        :param tokenizer: `model`'s corresponding tokenizer: useful for GenerativeEvaluators, but might not be necessary
        :param dataset: A dataset to evaluate the current model on.

        :return: A single float value evaluating the model quality.
        """
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__)
