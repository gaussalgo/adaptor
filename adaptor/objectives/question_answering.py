from adaptor.objectives.objective_base import SupervisedObjective
from adaptor.utils import Head
from typing import Dict, Optional, Union, Iterator
import torch
from transformers import DataCollatorWithPadding, BatchEncoding

from ..objectives.objective_base import SupervisedObjective
from ..utils import Head

class ExtractiveQA(SupervisedObjective):
    compatible_head: Head = Head.QA

    def _get_inputs_iterator(self, split: str) -> Iterator:
            """
            Batches and encodes input texts and corresponding labels.
            :param split: Selected data split. `train` or `eval`.
            :return: Iterator over batch encodings.
            """

            collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

            batch_features = []
            
            for src_text, text_pair, label in zip(*self._per_split_iterators_text_pair(split)):
                max_length = 384 #TODO check what max length and where it should be set
                out_sample = self.tokenizer(src_text, text_pair=text_pair)
                tokenized_label = self.tokenizer(label)

                #find indexes for answer:
                if set(tokenized_label["input_ids"][1:-1]).issubset(set(out_sample["input_ids"])): 
                    start_position=out_sample["input_ids"].index(tokenized_label["input_ids"][1:-1][0])
                    answer_length=len(tokenized_label["input_ids"][1:-1])
                else:
                    start_position=-1
                    answer_length=0
                end_position = start_position+answer_length
                out_sample["label"] = tokenized_label
                out_sample["start_position"] = torch.tensor(start_position)
                out_sample["end_position"] = torch.tensor(end_position)

                batch_features.append(out_sample)
                if len(batch_features) == self.batch_size:
                    print(batch_features)
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

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        #split concatonated outputs for start and end:
        start_logits, end_logits = torch.split(logit_outputs, dim = 0)

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return total_loss



 