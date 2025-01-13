import abc
import logging
from enum import Enum
from typing import Dict, Iterable, Iterator, Optional
import os

import torch
import peft
from torch.utils.data import IterableDataset
import transformers
from transformers import BatchEncoding, TrainingArguments


logger = logging.getLogger()


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


class SavingStrategy(Enum):
    ALL_OBJECTIVES = 1
    FIRST_OBJECTIVE = 2
    FINISHED_OBJECTIVES = 3


HEAD_TO_MODEL_CLS = {
    Head.SEQ_CLASSIFICATION: {"full": transformers.AutoModelForSequenceClassification,
                              "peft": peft.PeftModelForSequenceClassification},
    Head.TOKEN_CLASSIFICATION: {"full": transformers.AutoModelForTokenClassification,
                                "peft": peft.PeftModelForTokenClassification},
    Head.SEQ2SEQ: {"full": transformers.AutoModelForSeq2SeqLM,
                   "peft": peft.PeftModelForSeq2SeqLM},
    Head.CLM: {"full": transformers.AutoModelForCausalLM,
               "peft": peft.PeftModelForCausalLM},
    Head.MLM: {"full": transformers.AutoModelForMaskedLM,
               "peft": NotImplemented},
    Head.QA: {"full": transformers.AutoModelForQuestionAnswering,
              "peft": peft.PeftModelForQuestionAnswering}
}

PEFT_BASE_MODEL_CHECKPOINT_SUBDIR = "base_model"


class AdaptationDataset(IterableDataset, abc.ABC):
    """
    United dataset for both sequence and token training, and both supervised and unsupervised objectives.
    """

    def __init__(self, length: Optional[int] = None):
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size > 1:
            logger.warning("World size for data sampling: %s" % self.world_size)
            self.length = length // self.world_size
        else:
            self.length = length

    def __getitem__(self, index: int) -> BatchEncoding:
        raise ValueError("We shouldn't ever get here?")

    def __len__(self):
        return self.length

    @staticmethod
    def iter_text_file_per_line(path: str) -> Iterable[str]:
        """
        Iterate over the lines of a file on a given path.
        At this point, `path` is checked to be of a supported format.
        :param path: file path
        """
        if path.endswith(".gz"):
            import gzip
            import io
            with io.TextIOWrapper(io.BufferedReader(gzip.open(path))) as file:  # type: ignore
                for line in file:
                    yield line.strip()
        else:
            # assumes plain, newline-separated text file
            with open(path) as f:
                for line in f:
                    yield line.strip()


class TransformerAdaptationDataset(AdaptationDataset):
    def __init__(self,
                 batch_encoding_params: Iterable[Dict[str, torch.LongTensor]],
                 length: Optional[int] = None,
                 offset: int = 0):
        """
        :param batch_encoding_params: Arguments to be passed to BatchEncoding (input_ids, attention_mask, labels)
        """
        super().__init__(length)
        self.batch_encoding_params = batch_encoding_params
        self.offset = offset

    def __iter__(self) -> Iterator[Dict[str, torch.LongTensor]]:
        """
        Iterates over collated items of the dataset. The items are already collated by the specific Objective,
        so that Schedules can perform item-level sampling.
        :return: iterator over the samples of the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()

        for i, encoded_sample in enumerate(self.batch_encoding_params):
            # fast-forward the self.offset steps in continued training
            if i < self.offset:
                continue

            if self.world_size > 1 and worker_info is not None:
                # multi-gpu DataParallel
                if i % self.world_size == worker_info.id:
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

    def __init__(self,
                 stopping_strategy: StoppingStrategy,
                 stopping_patience: Optional[int] = 10,
                 saving_strategy: SavingStrategy = SavingStrategy.ALL_OBJECTIVES,
                 also_log_converged_objectives: Optional[bool] = True,
                 save_peft_base_model: bool = False,
                 **kwargs):
        """
        Adds Adaptor-specific arguments to standard HF's TrainingArguments
        :param stopping_strategy: A strategy to decide whether to stop training, based on the states of all objectives
        :param stopping_patience: How many global steps to wait before stopping the training
        :param saving_strategy: A strategy to choose the objectives for which we persist the models in checkpoints.
        :param also_log_converged_objectives: Whether to perform evaluations also for already stopped objectives
        :param save_peft_base_model: Whether to also persist the base model when training some objective(s) with PEFT.
        """
        # novel arguments, w.r.t. original TrainingArguments
        self.stopping_strategy = stopping_strategy
        self.stopping_patience = stopping_patience
        self.saving_strategy = saving_strategy
        self.log_converged_objectives = also_log_converged_objectives
        self.save_peft_base_model = save_peft_base_model

        # adjustments of the defaults expected by Scheduler
        unexpected_adjusted_args = [arg for arg in kwargs.keys() if arg in self.fixed_adaptation_args.keys()]
        if unexpected_adjusted_args:
            raise ValueError("You should not set these TrainingArgs for Adaptation: %s" % unexpected_adjusted_args)

        # set default values to fixed args
        kwargs = {**kwargs, **self.fixed_adaptation_args}
        super().__init__(**kwargs)
