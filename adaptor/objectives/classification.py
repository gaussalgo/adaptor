from typing import Dict, Iterable, Optional, Union

import torch
from transformers import DataCollatorForTokenClassification, BatchEncoding

from ..objectives.objective_base import SupervisedObjective
from ..utils import Head


class TokenClassification(SupervisedObjective):

    compatible_head = Head.TOKEN_CLASSIFICATION

    def _wordpiece_token_label_alignment(self,
                                         texts: Iterable[str],
                                         labels: Iterable[str],
                                         label_all_tokens: bool = True,
                                         ignore_label_idx: int = -100) -> Iterable[Dict[str, torch.LongTensor]]:
        """
        Decompose given space-separated labels and words into subword-aligned input ids and label ids,
        Performs batching and collation and return resulting encodings.

        NOTE: be aware that due to the segmentation by spaces, tokenization might differ between the training
        and inference for the models using space-including tokenizers, such as sentencepiece.
        We tested this objective only with commonly-used Encoders (BERT, RoBERTa) utilizing pre-tokenized WPiece & BPE.

        For an example of expected inputs, see tests/mock_data/supervised_texts.txt
        and texts/mock_data/supervised_texts_token_labels.txt

        :param texts: Sentence-level input texts.
        :param labels: Sentence-level input labels, aligned with input words by spaces.
        :param label_all_tokens: Whether to assign consistent label to all wordpieces of labeled tokens,
        or only to the first wordpiece, giving `ignore_label_idx` to the following wordpieces.
        :param ignore_label_idx: a label assigned to the wordpieces assigned no labels.

        :return: Aligned, batched encodings.
        """
        collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        batch_features = []

        # special tokens identification: general heuristic
        ids1 = self.tokenizer("X").input_ids
        ids2 = self.tokenizer("Y").input_ids

        special_bos_tokens = []
        for i in range(len(ids1)):
            if ids1[i] == ids2[i]:
                special_bos_tokens.append(ids1[i])
            else:
                break

        special_eos_tokens = []
        for i in range(1, len(ids1)):
            if ids1[-i] == ids2[-i]:
                special_eos_tokens.append(ids1[-i])
            else:
                break
        special_eos_tokens = list(reversed(special_eos_tokens))

        # per-sample iteration
        for text, text_labels in zip(texts, labels):
            tokens = text.split()
            labels = text_labels.split()

            assert len(tokens) == len(labels), \
                "A number of tokens in the first line is different than a number of labels. " \
                "Text: %s \nLabels: %s" % (text, text_labels)

            tokens_ids = self.tokenizer(tokens, truncation=True, add_special_tokens=False).input_ids

            wpiece_ids = special_bos_tokens.copy()

            # labels of BoS and EoS are always "other"
            out_label_ids = [ignore_label_idx] * len(special_bos_tokens)

            for token_ids, label in zip(tokens_ids, labels):
                # chain the wordpieces without the special symbols for each token
                wpiece_ids.extend(token_ids)
                if label_all_tokens:
                    # label all wordpieces
                    out_label_ids.extend([self.labels_map[label]] * len(token_ids))
                else:
                    # label only the first wordpiece
                    out_label_ids.append(self.labels_map[label])
                    # ignore the predictions over other token's wordpieces from the loss
                    out_label_ids.extend([ignore_label_idx] * (len(token_ids) - 1))

            out_label_ids.extend([ignore_label_idx] * len(special_eos_tokens))
            wpiece_ids.extend(special_eos_tokens.copy())

            assert len(out_label_ids) == len(wpiece_ids), "We found misaligned labels in sample: '%s'" % text

            if self.tokenizer.model_max_length is None:
                truncated_size = len(out_label_ids)
            else:
                truncated_size = min(self.tokenizer.model_max_length, len(out_label_ids))

            batch_features.append({"input_ids": wpiece_ids[:truncated_size],
                                   "attention_mask": [1] * truncated_size,
                                   "labels": out_label_ids[:truncated_size]})
            # maybe yield a batch
            if len(batch_features) == self.batch_size:
                yield collator(batch_features)
                batch_features = []
        if batch_features:
            yield collator(batch_features)

        # check that the number of outputs of the selected compatible head matches the just-parsed data set
        num_outputs = list(self.compatible_head_model.parameters())[-1].shape[0]
        num_labels = len(self.labels_map)
        assert num_outputs == num_labels, "A number of the outputs for the selected %s head (%s) " \
                                          "does not match a number of token labels (%s)" \
                                          % (self.compatible_head, num_outputs, num_labels)

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Constructs input encodings for token classification using Transformers.
        :param split: Selected data split. `train` or `eval`.
        :return: Encodings compatible with self.compatible_model_head.
        """

        texts_iter, labels_iter = self._per_split_iterators(split)

        aligned_collated_iter = self._wordpiece_token_label_alignment(texts_iter, labels_iter)

        return aligned_collated_iter

    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Computes a loss for model outputs on a single token classification batch.
        :param logit_outputs: Token Classification model raw outputs.
        :param labels: Expected labels.
        :param attention_mask: Mask of the tokens to compute loss on.
        :return: loss value with grad_fn.
        """
        # generic token classification loss, originally implemented e.g. in transformers.BertForTokenClassification

        loss_fct = torch.nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logit_outputs.view(-1, len(self.labels_map))
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logit_outputs.view(-1, len(self.labels_map)), labels.view(-1))

        return loss


class SequenceClassification(SupervisedObjective):

    compatible_head = Head.SEQ_CLASSIFICATION

    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Computes a loss for model outputs on a single sequence classification batch.
        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Sequence Classification model raw outputs.
        :param labels: Expected labels.
        :return: loss value with grad_fn.
        """
        # based on transformers.modeling_bert.BertForSequenceClassification
        if labels.dim() == 1:
            # single-label classification
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logit_outputs.view(-1, logit_outputs.shape[-1]), labels.view(-1))
        else:
            # multi-label classification
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logit_outputs, labels)

        return loss
