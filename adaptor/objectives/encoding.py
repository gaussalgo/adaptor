import abc
from typing import Tuple, Union, List, Dict, Iterable, Optional

import torch
from torch import Tensor
from transformers import BatchEncoding

from adaptor.objectives.objective_base import SupervisedObjective, Objective
from adaptor.utils import Head, AdaptationDataset


class Encoding(Objective, abc.ABC):
    compatible_head = Head.ENCODING

    # loss_function is sentence-transformers.losses.*Loss class type
    # -- sentence-transformers do not have any inheritance hierarchy
    loss_function: torch.nn.Module

    def smart_batching_collate(self,
                               batch_texts: List[List[str]],
                               batch_labels: List[int]) -> Tuple[List[BatchEncoding], Tensor]:
        """
        Adjusted from sentence_transformers.SentenceTransformer.smart_batching_collate

        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        :param batch_texts: texts within the batch. Tuples of Triples
        :param batch_labels: labels within the batch
        :return: a batch of tensors for the model
        """
        sentence_features = [self.tokenizer(texts, padding=True, truncation=True) for texts in batch_texts]
        labels = torch.tensor(batch_labels)
        return sentence_features, labels

    def _get_inputs_iterator(self, split: str) -> Iterable:
        """
        Batches and encodes input texts, text pairs and corresponding labels.
        :param split: Selected data split. `train` or `eval`.
        :return: Iterator over batch encodings.
        """

        batch_texts = []
        batch_labels = []

        for sample in zip(*self._per_split_iterators(split)):
            input_texts, labels = sample[:-1], sample[-1]
            batch_texts.append(input_texts)
            batch_labels.append(labels)
            if len(batch_texts) == self.batch_size:
                yield self.smart_batching_collate(batch_texts, batch_labels)
                batch_texts = []
                batch_labels = []

        if batch_texts:
            # yield residual batch
            yield self.smart_batching_collate(batch_texts, batch_labels)

    def _compute_loss(self,
                      inputs: Union[BatchEncoding, Dict[str, torch.Tensor]],
                      labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Computes a loss for model outputs on a single question answering batch.
        :param inputs: encoded inputs for a sentence_transformers model
        :param labels: Assigned labels.
        :return: loss value with grad_fn.
        """
        self.loss_function.model = self.compatible_head_model
        loss = self.loss_function(inputs, labels)
        return loss


class PairEncodingObjective(SupervisedObjective, Encoding):

    def __init__(self,
                 *args,
                 text_pair_or_path: Union[str, List[str]],
                 val_text_pair_or_path: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        super().__init__(*args, text_pair_or_path=text_pair_or_path,
                         val_text_pair_or_path=val_text_pair_or_path, **kwargs)


class TripletEncodingObjective(SupervisedObjective, Encoding):
    negative_text_pair: Optional[List[str]] = None
    negative_text_pair_path: Optional[str] = None

    val_negative_text_pair: Optional[List[str]] = None
    val_negative_text_pair_path: Optional[str] = None

    def __init__(self,
                 *args,
                 positive_text_pair_or_path: Union[str, List[str]],
                 val_positive_text_pair_or_path: Optional[Union[str, List[str]]] = None,
                 negative_text_pair_or_path: Union[str, List[str]],
                 val_negative_text_pair_or_path: Optional[Union[str, List[str]]] = None,
                 loss_function: torch.nn.Module,
                 **kwargs):
        super().__init__(*args, text_pair_or_path=positive_text_pair_or_path,
                         val_text_pair_or_path=val_positive_text_pair_or_path, **kwargs)
        if negative_text_pair_or_path is not None:
            if isinstance(negative_text_pair_or_path, str):
                self.text_pair_path = negative_text_pair_or_path
            else:
                self.text_pair = negative_text_pair_or_path

        if val_negative_text_pair_or_path is not None:
            if isinstance(val_negative_text_pair_or_path, str):
                self.val_negative_text_pair_path = val_negative_text_pair_or_path
            else:
                self.val_negative_text_pair = val_negative_text_pair_or_path

        self.negative_text_pair_or_path = negative_text_pair_or_path

        self.loss_function = loss_function

    def _per_split_iterators(self, split: str) -> Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
        queries_iter, positives_iter, targets_iter = super()._per_split_iterators(split)
        if split == "train":
            if self.text_pair is not None:
                negative_iter = iter(self.text_pair)
            elif self.text_pair_path is not None:
                negative_iter = AdaptationDataset.iter_text_file_per_line(self.text_pair_path)
            else:
                raise ValueError("Triplet objective requires you pass `negative_text_pair_or_path`"
                                 "with either a list of negative texts or a path to a .txt file containing these.")

        elif split == "eval":
            if self.val_text_pair is not None:
                negative_iter = iter(self.val_text_pair)
            elif self.val_text_pair_path is not None:
                negative_iter = AdaptationDataset.iter_text_file_per_line(self.val_text_pair_path)
            else:
                raise ValueError("You asked for validation, "
                                 "but the objective %s did not get any negative texts for validation. :( "
                                 "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, "
                                 "or set Objective(val_labels) param." % self)
        else:
            raise ValueError("Unrecognized split: %s" % split)

        return queries_iter, positives_iter, negative_iter, targets_iter
