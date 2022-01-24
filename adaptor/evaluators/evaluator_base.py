import abc
from typing import List, Optional, Dict, Union

import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from adaptor.utils import Head


class EvaluatorBase(abc.ABC):

    compatible_head: Head
    smaller_is_better: bool

    def __init__(self, decides_convergence: bool = False):
        self.determines_convergence = decides_convergence

    @abc.abstractmethod
    def __call__(self,
                 inputs: Optional[List[Union[Dict[str, torch.LongTensor], BatchEncoding]]] = None,
                 model: Optional[torch.nn.Module] = None,
                 logit_outputs: Optional[List[torch.FloatTensor]] = None,
                 labels: Optional[List[torch.LongTensor]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None):
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__)
