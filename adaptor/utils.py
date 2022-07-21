import abc
from enum import Enum
from typing import Dict, Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import BatchEncoding, TrainingArguments


class Head(Enum):
    SEQ_CLASSIFICATION = 1
    TOKEN_CLASSIFICATION = 2
    SEQ2SEQ = 4
    CLM = 5
    MLM = 6
    QA = 7
    UNKNOWN = 8


class StoppingStrategy(Enum):
    FIRST_OBJECTIVE_CONVERGED = 1
    ALL_OBJECTIVES_CONVERGED = 2
    FIRST_OBJECTIVE_NUM_EPOCHS = 3
    ALL_OBJECTIVES_NUM_EPOCHS = 4
    ALL_OBJECTIVES_NUM_STEPS = 5
    NUM_STEPS_TOTAL = 6
    MANUAL = 7


class AdaptationDataset(IterableDataset, abc.ABC):
    """
    United dataset for both sequence and token training, and both supervised and unsupervised objectives.
    """

    def __init__(self, length: Optional[int] = None):
        self.length = length

    def __getitem__(self, index: int) -> BatchEncoding:
        raise ValueError("We shouldn't ever get here?")

    def __len__(self):
        return self.length

    @staticmethod
    def iter_text_file_per_line(path: str) -> Iterable[str]:
        with open(path) as f:
            for line in f:
                yield line.strip()


class TransformerAdaptationDataset(AdaptationDataset):
    def __init__(
        self,
        batch_encoding_params: Iterable[Dict[str, torch.LongTensor]],
        length: Optional[int] = None,
    ):
        """
        :param batch_encoding_params: Arguments to be passed to BatchEncoding (input_ids, attention_mask, labels)
        """
        super().__init__(length)
        self.batch_encoding_params = batch_encoding_params

    def __iter__(self) -> Iterator[Dict[str, torch.LongTensor]]:
        """
        Iterates over collated items of the dataset. The items are already collated by the specific Objective,
        so that Schedules can perform item-level sampling.
        :return: iterator over the samples of the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()

        for i, encoded_sample in enumerate(self.batch_encoding_params):
            if worker_info is not None:
                # multi-gpu DataParallel
                if (i - worker_info.id) % worker_info.num_workers == 0:
                    # sample modulo number of all workers match this worker rank
                    yield encoded_sample
            else:
                yield encoded_sample


class AdaptationArguments(TrainingArguments):

    fixed_adaptation_args = {
        "per_device_train_batch_size": 1,  # batching is done by Objective, no two distinct batches
        "per_device_eval_batch_size": 1,  # should be present in a single infer batch
        "per_gpu_train_batch_size": None,  # aggregation over multiple objectives can be done using
        "per_gpu_eval_batch_size": None,  # `gradient_accumulation_steps` > 1
        "do_predict": False,  # we do not want to mangle with multi-objective reports here,
        # models are separately reloadable
        "disable_tqdm": True,  # scheduler takes care of top-level terminal monitoring
        "dataloader_pin_memory": False,  # does not necessarily match the shapes in multi-objective training
        "remove_unused_columns": False,  # from transformers 4.19.x, this would remove batches' control attributes
    }

    def __init__(
        self,
        stopping_strategy: StoppingStrategy,
        stopping_patience: Optional[int] = 10,
        also_log_converged_objectives: Optional[bool] = True,
        **kwargs
    ):

        # novel arguments, w.r.t. original TrainingArguments
        self.stopping_strategy = stopping_strategy
        self.stopping_patience = stopping_patience
        self.log_converged_objectives = also_log_converged_objectives

        # adjustments of the defaults expected by Scheduler
        unexpected_adjusted_args = [
            arg for arg in kwargs.keys() if arg in self.fixed_adaptation_args.keys()
        ]
        if unexpected_adjusted_args:
            raise ValueError(
                "You should not set these TrainingArgs for Adaptation: %s"
                % unexpected_adjusted_args
            )

        # set default values to fixed args
        kwargs = {**kwargs, **self.fixed_adaptation_args}
        super().__init__(**kwargs)
