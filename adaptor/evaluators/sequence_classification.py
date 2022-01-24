from typing import List, Optional, Dict, Union

import torch
from transformers import PreTrainedTokenizer, BatchEncoding

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head


class Accuracy(EvaluatorBase):

    compatible_head: Head = Head.SEQ_CLASSIFICATION
    smaller_is_better: bool = False

    def __call__(self,
                 inputs: Optional[List[Union[Dict[str, torch.LongTensor], BatchEncoding]]] = None,
                 model: Optional[torch.nn.Module] = None,
                 logit_outputs: Optional[List[torch.FloatTensor]] = None,
                 labels: Optional[List[torch.LongTensor]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None):
        labels_all = torch.hstack(labels)
        assert labels_all.dim() == 1, "%s evaluator does not support evaluation of multinomial classification." % self

        num_correct = sum(torch.vstack(logit_outputs).argmax(-1) == labels_all)
        return (num_correct / len(labels_all)).item()
