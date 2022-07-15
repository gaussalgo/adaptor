from adaptor.objectives.objective_base import SupervisedObjective
from adaptor.utils import Head
from typing import Dict, Optional, Union, Iterator
import torch
from transformers import DataCollatorWithPadding, BatchEncoding

from ..objectives.objective_base import SupervisedObjective
from ..utils import Head

MAX_LENGTH = 512

class ExtractiveQA(SupervisedObjective):
    compatible_head: Head = Head.QA

    def _get_inputs_iterator(self, split: str) -> Iterator:
            """
            Batches and encodes input texts and corresponding labels.
            :param split: Selected data split. `train` or `eval`.
            :return: Iterator over batch encodings.
            """

            collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8,return_tensors = "pt", max_length=MAX_LENGTH,padding='max_length')

            batch_features = []
            
            for src_text, text_pair, label in zip(*self._per_split_iterators_text_pair(split)):
                out_sample = self.tokenizer(src_text, text_pair=text_pair, max_length=MAX_LENGTH, truncation=True,padding='max_length')
                tokenized_label = self.tokenizer(label, max_length=MAX_LENGTH, truncation=True,padding='max_length')
                label_wo_padding = self.tokenizer(label)
                #find indexes for answer:
                if set(tokenized_label["input_ids"][1:-1]).issubset(set(out_sample["input_ids"])): 
                    start_position=out_sample["input_ids"].index(tokenized_label["input_ids"][1:-1][0])
                    answer_length=len(label_wo_padding["input_ids"][1:-1])
                else:
                    start_position=0
                    answer_length=0
                end_position = start_position+answer_length
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
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Computes a loss for model outputs on a single question answering batch.
        :param logit_outputs: Token Classification model raw outputs.
        :param labels: Expected labels.
        :param attention_mask: Mask of the tokens to compute loss on.
        :return: loss value with grad_fn.
        """
        loss_fct = torch.nn.CrossEntropyLoss()
        #split concatonated outputs for start and end:
        start_logits = logit_outputs["start_logits"]
        end_logits = logit_outputs["end_logits"]

        start_positions = inputs["start_position"]
        end_positions = inputs["end_position"]

        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss



 