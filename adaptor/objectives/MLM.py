from typing import Iterable, Dict, Union, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, BatchEncoding

from ..objectives.objective_base import UnsupervisedObjective
from ..utils import Head


class MaskedLanguageModeling(UnsupervisedObjective):
    """
    TODO: a support for pre-training a new model
    TODO: will require an option of initialization without pre-trained tokenizer and lang_module
    """
    compatible_head = Head.MLM

    def __init__(self,
                 *args,
                 masking_application_prob: float = 0.15,
                 full_token_masking: bool = False,
                 **kwargs):
        """
        :param masking_application_prob: A probability of applying mask at each token.
        :param full_token_masking: Whether to apply mask on the whole (multi-subword) words, or just subwords.
        """
        super().__init__(*args, **kwargs)
        if full_token_masking:
            self.collator = DataCollatorForWholeWordMask(self.tokenizer,
                                                         mlm_probability=masking_application_prob,
                                                         pad_to_multiple_of=8)
        else:
            self.collator = DataCollatorForLanguageModeling(self.tokenizer,
                                                            mlm_probability=masking_application_prob,
                                                            pad_to_multiple_of=8)

    def _mask_some_tokens(self, texts: Iterable[str]) -> Iterable[Dict[str, torch.LongTensor]]:
        """
        Encodes input texts and applies collator that masks some tokens from inputs.
        :param texts: Input texts.
        :return: Encodings with masks on labels.
        """
        batch_features = []
        for text in texts:
            input_features = self.tokenizer(text, truncation=True)
            batch_features.append(input_features)

            # maybe yield a batch
            if len(batch_features) == self.batch_size:
                # selection of masked tokens, padding and labeling is provided by transformers.DataCollatorForLM
                yield self.collator(batch_features)
                batch_features = []
        # yield remaining texts in collected batch
        if batch_features:
            yield self.collator(batch_features)

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Applies masks on input iterator from a path or a list of strings.
        :param split: Data split. `train` or `eval`.
        :return:
        """
        texts_iter = self._per_split_iterator_single(split)
        collated_iter = self._mask_some_tokens(texts_iter)
        return collated_iter

    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Masked language modeling, as implemented by BERT.

        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw LM model outputs
        :param labels: ids of expected outputs.

        :return: loss value with grad_fn.
        """
        # token classification loss, from transformers.BertForMaskedLM
        loss_fct = CrossEntropyLoss()
        vocab_size = logit_outputs.size()[-1]
        masked_lm_loss = loss_fct(logit_outputs.view(-1, vocab_size), labels.view(-1))
        return masked_lm_loss
