import abc
from typing import List

import torch
from transformers import PreTrainedTokenizer

from adaptor.utils import Head, AdaptationDataset


class EvaluatorBase(abc.ABC):
    """
    Base class of all evaluators, usable by the instances of Objectives.
    Subclass and implement this interface to create new evaluators.
    """

    compatible_heads: List[Head]
    smaller_is_better: bool

    def __init__(self, decides_convergence: bool = False):
        self.determines_convergence = decides_convergence

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        """
        Evaluation of this instance
        :param tokenizer:
        :param model:
        :param dataset:
        """
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__)
