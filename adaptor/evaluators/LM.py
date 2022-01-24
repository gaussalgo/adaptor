from typing import List, Optional, Dict, Union

import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head


class Perplexity(EvaluatorBase):

    compatible_head: Head = Head.LANGUAGE_MODEL

    def __call__(self,
                 inputs: Optional[List[Union[Dict[str, torch.LongTensor], BatchEncoding]]] = None,
                 model: Optional[torch.nn.Module] = None,
                 logit_outputs: Optional[List[torch.FloatTensor]] = None,
                 labels: Optional[List[torch.LongTensor]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None):
        raise NotImplementedError()
