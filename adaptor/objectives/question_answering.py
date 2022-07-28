from typing import Dict, Optional, Tuple, Union, Iterator, List

import torch
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from ..objectives.objective_base import SupervisedObjective
from ..utils import Head


class ExtractiveQA(SupervisedObjective):
    compatible_head: Head = Head.QA

    @staticmethod
    def _find_start_end_position(answer_ids: List[int], context_ids: List[int]) -> Tuple[int, int]:
        """
        Returns first occurrence of subsequence (answer_ids) in the sequence (context_ids).
        If no match is found, (0, 0) is returned.
        """
        start_positions = [i for i in range(len(context_ids) - len(answer_ids))
                           if context_ids[i: i + len(answer_ids)] == answer_ids]
        if not start_positions:
            return 0, 0
        else:
            first_start_position, first_end_position = start_positions[0], start_positions[0] + len(answer_ids)
            return first_start_position, first_end_position

    def _get_inputs_iterator(self, split: str) -> Iterator:
        """
        Batches and encodes input texts, text pairs and corresponding labels.
        :param split: Selected data split. `train` or `eval`.
        :return: Iterator over batch encodings.
        """

        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8,
                                           return_tensors="pt", padding='max_length')

        batch_features = []

        for src_text, text_pair, label in zip(*self._per_split_iterators(split)):
            out_sample = self.tokenizer(src_text, text_pair=text_pair, truncation=True, padding='max_length')
            tokenized_label = self.tokenizer(label, truncation=True, padding='max_length')
            label_wo_padding = self.tokenizer(label)
            start_position, end_position = self._find_start_end_position(label_wo_padding["input_ids"][1:-1],
                                                                         out_sample["input_ids"])
            out_sample["label"] = tokenized_label["input_ids"]
            out_sample["start_position"] = start_position
            out_sample["end_position"] = end_position

            batch_features.append(out_sample)
            if len(batch_features) == self.batch_size:
                yield collator(batch_features)
                batch_features = []

        if batch_features:
            # yield residual batch
            yield collator(batch_features)

    def _compute_loss(self,
                      model_outputs: QuestionAnsweringModelOutput,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Computes a loss for model outputs on a single question answering batch.
        :param model_outputs: QuestionAnsweringModelOutput.
        :param labels: Expected labels.
        :param attention_mask: Mask of the tokens to compute loss on.
        :return: loss value with grad_fn.
        """
        loss_fct = torch.nn.CrossEntropyLoss()

        # following keys need to be present in the model output
        start_logits = model_outputs["start_logits"]
        end_logits = model_outputs["end_logits"]

        start_positions = inputs["start_position"]
        end_positions = inputs["end_position"]

        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss
