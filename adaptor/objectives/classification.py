from typing import Dict, Iterable, Optional, Union

import torch
from transformers import DataCollatorForTokenClassification, BatchEncoding

from ..objectives.objective_base import SupervisedObjective
from ..utils import Head


class TokenClassification(SupervisedObjective):

    compatible_head = Head.TOKEN_CLASSIFICATION

    def _wordpiece_token_label_alignment(self, texts: Iterable[str],
                                         labels: Iterable[str]) -> Iterable[Dict[str, torch.LongTensor]]:
        """
        Decompose given space-separated labels and words into subword-aligned input ids and label ids,
        Performs batching and collation and return resulting encodings.
        :param texts: Sentence-level input texts.
        :param labels: Sentence-level input labels, aligned with input words by spaces.
        For an example of expected inputs, see tests/mock_data/supervised_texts.txt
        and texts/mock_data/supervised_texts_token_labels.txt

        :return: Aligned encodings.
        """
        collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        batch_features = []

        for text, text_labels in zip(texts, labels):
            tokens = text.split()
            labels = text_labels.split()

            tokenizer_encodings = self.tokenizer(text, truncation=True)
            # attention mask is lang_module-specific
            attention_mask = tokenizer_encodings.attention_mask
            wpiece_ids = tokenizer_encodings.input_ids
            wordpieces = self.tokenizer.batch_decode(wpiece_ids)

            out_label_ids = []

            # next token lookup - avoid out-of-index, and exclude from token labels
            tokens.append(wordpieces[-1])
            labels.append("O")

            assert len(tokens) == len(labels), \
                "A number of tokens in the first line is different than a number of labels. " \
                "Text: %s \nLabels: %s" % (text, text_labels)

            # assign current label to current wordpiece until the current_token is fully iterated-over
            current_token, current_label = tokens.pop(0), labels.pop(0)  # noqa F401: current_token unused
            for wpiece_id, wpiece in zip(wpiece_ids, wordpieces):
                next_token = tokens[0]
                if next_token.startswith(wpiece):
                    # if the next token starts with a current wordpiece, move to the next token + label
                    current_token, current_label = tokens.pop(0), labels.pop(0)  # noqa F401: current_token unused
                out_label_ids.append(self.labels_map[current_label])

            batch_features.append({"input_ids": wpiece_ids,
                                   "attention_mask": attention_mask,
                                   "labels": out_label_ids})
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
