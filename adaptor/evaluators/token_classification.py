from typing import List, Optional, Dict, Union

import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head


class MacroAccuracy(EvaluatorBase):

    compatible_head: Head = Head.TOKEN_CLASSIFICATION
    smaller_is_better: bool = False

    def __call__(self,
                 inputs: Optional[List[Union[Dict[str, torch.LongTensor], BatchEncoding]]] = None,
                 model: Optional[torch.nn.Module] = None,
                 logit_outputs: Optional[List[torch.FloatTensor]] = None,
                 labels: Optional[List[torch.LongTensor]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None):

        return sum([logits.argmax(-1) == label
                    for logits, label in zip(logit_outputs, labels)]) / torch.stack(labels).flatten().shape[0]

