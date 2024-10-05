import abc
from typing import List, Union, Dict, Iterable, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq, BatchEncoding

from .seq2seq import SequentialMixin
from ..objectives.objective_base import UnsupervisedObjective
from ..utils import Head


class CausalLanguageModeling(SequentialMixin, UnsupervisedObjective, abc.ABC):
    compatible_head = Head.CLM

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model, pad_to_multiple_of=8)

    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Causal language modeling, as implemented by GPT-2.

        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.

        :return: a single-item torch tensor with registered grad_fn.
        """
        # from transformers.GPT2LMHeadModel.forward()
        shift_logits = logit_outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
