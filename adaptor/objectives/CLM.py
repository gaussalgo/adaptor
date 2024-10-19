import abc
from typing import Union, Dict, Optional, List, Iterable, Iterator

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

    def _get_seq2seq_collated_iterator(self,
                                       source_texts: Iterable[str],
                                       target_texts: Iterable[str]) -> Iterator[BatchEncoding]:
        """
        Creates an iterator over batches of encoded `source_texts` as inputs and `target_texts` as labels.
        Override this to implement custom mapping, or unsupervised seq2seq objective. See e.g. BackTranslation.
        :param source_texts: Input texts.
        :param target_texts: Output (expected) texts to translate input texts into.
        :return: Iterator of encoded batches.
        """
        features_batch = []
        asserted_equal = False  # speedup: avoid repeated assertions of string equality
        self.tokenizer.src_lang = self.source_lang_id
        self.tokenizer.tgt_lang = self.target_lang_id

        for source_text, target_text in zip(source_texts, target_texts):
            if not asserted_equal:
                assert source_text == target_text, ("CLM objective expects both texts to be the same. "
                                                    "If you override this objective, it's possible that you should "
                                                    "rather override SequentialMixin supporting different src and tgt.")
                asserted_equal = True

            sample_features = self.tokenizer(source_text, truncation=True)
            sample_targets = self.tokenizer(target_text, truncation=True)

            features_batch.append({"input_ids": sample_features.input_ids,
                                   "attention_mask": sample_features.attention_mask,
                                   "labels": sample_targets.input_ids})
            if len(features_batch) == self.batch_size:
                yield self.collator(features_batch)
                features_batch = []

        if features_batch:
            # yield last nonempty residual batch
            yield self.collator(features_batch)

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
