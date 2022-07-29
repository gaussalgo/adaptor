import abc
from typing import List, Optional, Iterable, Dict, Iterator, Callable, Any, Union

import torch
from transformers import DataCollatorForSeq2Seq, BatchEncoding

from ..lang_module import LangModule
from ..objectives.objective_base import SupervisedObjective, Objective
from ..utils import Head


class SequentialMixin(Objective, abc.ABC):

    collator: Callable[[List[Dict[str, torch.FloatTensor]]], List[Dict[str, torch.FloatTensor]]]
    source_lang_id: Optional[str]
    target_lang_id: Optional[str]

    def __init__(self, *args,
                 source_lang_id: Optional[str] = None,
                 target_lang_id: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.source_lang_id = source_lang_id
        self.target_lang_id = target_lang_id

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
        for source_text, target_text in zip(source_texts, target_texts):
            self.tokenizer.src_lang = self.source_lang_id
            self.tokenizer.tgt_lang = self.target_lang_id
            sample_features = self.tokenizer(source_text, truncation=True)

            with self.tokenizer.as_target_tokenizer():
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

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Creates a default iterator over encodings with aligned input and output texts.
        :param split: Data split. `train` or `eval`.
        :return: Iterator of model input encodings.
        """
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        collated_iter = self._get_seq2seq_collated_iterator(source_texts_iter, target_texts_iter)

        return collated_iter


class Sequence2SequenceMixin(SequentialMixin, abc.ABC):

    compatible_head: Head = Head.SEQ2SEQ
    collator: Callable[[List[Dict[str, torch.FloatTensor]]], List[Dict[str, torch.FloatTensor]]]

    def __init__(self, *args, **kwargs):
        """
        Refer to the documentation of superclass.
        """
        # adjust only default max_samples_per_*log, since generative evaluation is much slower
        # but stick to user selection, if there is any
        if "max_samples_per_log" not in kwargs:
            kwargs["max_samples_per_log"] = 20
        if "max_samples_per_eval_log" not in kwargs:
            kwargs["max_samples_per_eval_log"] = 100

        super().__init__(*args, **kwargs)

        # if this is translation objective, tokenization of source and target might vary (can include lang_token_id)
        # if it does not, this will just set unused attribute of tokenizer
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model, pad_to_multiple_of=8)

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Computes sequence2sequence loss
        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw outputs of language modeling head model
        :param labels: Token ids of expected outputs.
        :return: Single value of the loss, with grad_fn.
        """
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations at some cases
        loss_fct = torch.nn.CrossEntropyLoss()
        # vocab-agnostic loss circumvents incorrectly-set vocab_size of some models (e.g. mt5)
        lm_loss = loss_fct(lm_logit_outputs.flatten(end_dim=1), labels.flatten())

        return lm_loss

    def register_compatible_head_model(self,
                                       lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:

        head_module = super().register_compatible_head_model(lang_module, other_objective,
                                                             objective_args_for_head_config, preloaded_module)
        assert hasattr(head_module, "prepare_decoder_input_ids_from_labels"), \
            "No head of the loaded LangModule is compatible with %s objective! " \
            "\nNote that the module compatible with " \
            "Sequence2SequenceMixin \nmust have `prepare_decoder_input_ids_from_labels` method, " \
            "see e.g. \ntransformers.BartModel." % self

        return head_module


class Sequence2Sequence(Sequence2SequenceMixin, SupervisedObjective):

    pass
