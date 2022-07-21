from typing import List

import torch
from transformers import PreTrainedTokenizer

from .evaluator_base import EvaluatorBase
from ..utils import Head, AdaptationDataset


class Perplexity(EvaluatorBase):

    compatible_heads: List[Head] = [Head.MLM, Head.CLM, Head.SEQ2SEQ]

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        raise NotImplementedError()
