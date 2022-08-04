import abc
import logging
from typing import List, Iterable, Dict, Any, Tuple, Iterator, Union, Optional

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, BatchEncoding

from adaptor.objectives.objective_base import Objective
from adaptor.utils import TransformerAdaptationDataset, StoppingStrategy, AdaptationArguments

logger = logging.getLogger()


class Schedule(abc.ABC):
    """
    Schedule instance decides on ordering of application of given objectives.
    It primarily keeps track of yielded model inputs and afterwards the corresponding objectives for computing loss.
    Additionally, schedule does the following:
    1. modifies a flow of the training process, by triggering termination on selected condition
    2. Aggregates logs of the train/eval objectives
    3. Constructs IterableDataset of objectives' batch encodings, passed to the Trainer by Adapter.
    """

    label: str
    objectives: Dict[str, Dict[int, Objective]]
    objectives_outputs_queue: List[Tuple[str, int]]
    converged_objectives: List[Objective]
    should_stop: bool

    def __init__(self,
                 objectives: List[Objective],
                 args: AdaptationArguments,
                 extra_eval_objectives: Iterable[Objective] = ()):
        """
        Initialises queues of objectives outputs and training flow modification parameters.
        :param objectives: Training objectives to be scheduled
        :param args: Adaptation arguments, extending HF TrainingArguments for multi-objective-specific params.
        :param extra_eval_objectives: Objectives used only for evaluation, in addition to training `objectives`.
                                      Make sure to `share_other_objective_head` for evaluation Objectives.
        """

        # eval objectives = train + eval => train objectives are evaluated implicitly
        self.objectives = {"train": {id(o): o for o in objectives},
                           "eval": {id(o): o for o in objectives + list(extra_eval_objectives)}}

        # initially, let the user know the total number of samples that will be used for training and evaluation
        for split in ["train", "eval"]:
            num_samples = 0
            for oid, objective in self.objectives[split].items():
                num_samples += objective.dataset_length[split]

            logger.warning("Total number of %s samples: %s", split, num_samples)
            if not num_samples:
                logger.warning("Make sure that you do not want to pass any %s samples!", split)

        self.objectives_outputs_queue = []
        self.converged_objectives = []
        self.should_stop = False

        self.args = args

    @abc.abstractmethod
    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        """
        Constructs an iterable determining an ordering of sampling objectives.
        Override only this method to implement custom Schedule.
        See the examples in SequentialSchedule and ParallelSchedule below.

        :param split: Data split to sample objectives for.
        :return: Iterator of Objectives determining order in which to sample given objectives.
                 Note that sampled objectives might be different for `train` and `eval`.
        """
        pass

    def objectives_log(self, split: str) -> Dict[str, float]:
        """
        Collects logs of all the objectives:
        train objectives for split == `train`, train+extra_eval_objectives for split == `eval`
        :param split: data split to construct logs for
        :return: a combined dictionary of logs of all objectives.
        """
        out_logs = {}
        for objective in self.objectives[split].values():
            out_logs = {**out_logs, **objective.per_objective_log(split)}

        return out_logs

    def _objective_passed_epochs(self, oid: int) -> bool:
        return self.objectives["train"][oid].epoch > self.args.num_train_epochs

    def _should_stop(self) -> Tuple[bool, StoppingStrategy]:
        """
        Decides whether the training should be terminated on next re-evaluation.

        :return: a tuple of [<if to stop>, <reason why to stop>]
        """
        # a number of epochs per all objectives is an upper-bound of the training duration
        obj_passed_epochs = [oid for oid in self.objectives["train"].keys() if self._objective_passed_epochs(oid)]
        if len(obj_passed_epochs) == len(self.objectives["train"]):
            logger.warning("Scheduler reached the given maximum number of epochs for all objectives. "
                           "Triggering termination.")
            return True, StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS
        # if the upper bound does not apply, check for the user-selected stopping strategy

        # strategies based on objectives' convergence
        if self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                           StoppingStrategy.ALL_OBJECTIVES_CONVERGED):
            self.converged_objectives = [obj for obj in self.objectives["train"].values()
                                         if obj.is_finished(convergence_patience=self.args.stopping_patience)]
            logger.warning("Converged objectives: %s" % [str(o) for o in self.converged_objectives])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_CONVERGED:
                return len(self.converged_objectives) > 0, self.args.stopping_strategy
            else:
                return len(self.converged_objectives) == len(self.objectives["train"]), self.args.stopping_strategy

        # strategies based on objectives' number of epochs
        elif self.args.stopping_strategy in (StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                             StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS):
            logger.warning("Objectives that passed max_epochs: %s" % [str(self.objectives["train"][o])
                                                                      for o in obj_passed_epochs])
            if self.args.stopping_strategy == StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS:
                return len(obj_passed_epochs) > 0, self.args.stopping_strategy
            else:
                return len(obj_passed_epochs) == len(self.objectives["train"]), self.args.stopping_strategy

        # strategies based on a number of steps
        elif self.args.stopping_strategy == StoppingStrategy.NUM_STEPS_TOTAL:
            total_steps = sum(o.num_steps for o in self.objectives["train"].values())
            if total_steps >= self.args.max_steps:
                return True, StoppingStrategy.NUM_STEPS_TOTAL

        elif self.args.stopping_strategy == StoppingStrategy.ALL_OBJECTIVES_NUM_STEPS:
            max_steps_objectives = [o for o in self.objectives["train"].values() if o.num_steps >= self.args.max_steps]
            logger.warning("Objectives that passed max_steps: %s" % [str(o) for o in max_steps_objectives])

            return len(max_steps_objectives) == len(self.objectives["train"]), StoppingStrategy.ALL_OBJECTIVES_NUM_STEPS

        return False, self.args.stopping_strategy

    def should_stop_check_callback(self) -> TrainerCallback:
        """
        Constructs HF Callback object that triggers a check for the stopping stategy on selected event (on_log).
        Is added among training callbacks by Adapter.
        :return: TrainerCallback checking stopping condition.
        """

        class AdaptationStoppingCallback(TrainerCallback):

            def on_log(cls, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
                """ Event called by Trainer after the given `logging_steps`."""
                self.remember_if_should_stop()

        return AdaptationStoppingCallback()

    def remember_if_should_stop(self):
        """
        Changes Schedule state according to a result of evaluation of stopping strategy.
        Logs the possible reason of stopping.
        """
        self.should_stop, stopping_strategy = self._should_stop()
        if self.should_stop:
            logger.warning("Scheduler reached a termination condition: %s" % stopping_strategy.name)

    def compute_loss(self,
                     logit_outputs: torch.FloatTensor,
                     labels: torch.Tensor,
                     inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Retrieves a loss from the corresponding objective.

        :param logit_outputs: Raw model outputs.
        :param labels: Corresponding expected outputs.
        :param inputs: Input sample corresponding to given model output (logits) and ground truth (labels).

        :return: loss scalar of corresponding objective, with grad_fn.
        """
        split, oid = self.objectives_outputs_queue.pop(0)

        # the objective loss arrives aggregated into a single item
        loss = self.objectives[split][oid].compute_loss(logit_outputs, labels, inputs, split)

        return loss

    def _one_round_eval_objective_sampler(self, objective: Objective, obj_i: int) -> Iterator[Dict[str, Any]]:
        """
        Default evaluation data sampling strategy: constructs a single-round iterator
        over evaluation dataset of selected objective.

        :param objective: Objective to iterate dataset from.
        :param obj_i: Rank of the objective. Used to construct non-interleaving progressbars in multi-objective sampling
        :return: Iterator over evaluation samples
        """
        dataset = objective.get_dataset("eval", obj_i, self.args.device)
        for sample in dataset:
            self.objectives_outputs_queue.append(("eval", sample["oid"]))
            yield sample

    def _infinite_train_objective_sampler(self, objective: Objective, obj_i: int) -> Iterator[Dict[str, Any]]:
        """
        Default training data sampling strategy: constructs infinite iterator
        over training dataset of selected objective.

        :param objective: Objective to iterate dataset from.
        :param obj_i: Rank of the objective. Used to construct non-interleaving progressbars in multi-objective sampling
        :return: Iterator over evaluation samples.
        """
        while True:
            # check for stopping conditions at the beginning of every epoch's objective
            self.remember_if_should_stop()

            dataset = objective.get_dataset("train", obj_i, self.args.device)
            for sample in dataset:
                self.objectives_outputs_queue.append(("train", sample["oid"]))
                yield sample

    def _sample_objective_dataset(self, objective: Objective, obj_i: int, split: str) -> Iterator[Dict[str, Any]]:
        if split == "train":
            # infinite iteration of the training resources, until the termination condition apply
            return self._infinite_train_objective_sampler(objective, obj_i)
        else:
            # single-round sampling - we do not want to iterate the evaluation forever
            return self._one_round_eval_objective_sampler(objective, obj_i)

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        Constructs combined iterator over the datasets of all objectives,
        according to the implemented `_sample_objectives`.
        This main training iteration is upper-bound by a `num_epochs` over a full data set.
        :param split: data split to iterate.
        :return: Iterator over samples of selected split.
        """
        if split == "train":
            objective_sampler = self._sample_objectives(split)
        else:
            # evaluation split uses simple, sequential evaluation over objectives
            objective_sampler = SequentialSchedule.single_iteration_eval_sampling(self.objectives["eval"].values())

        objectives_data_samplers = {obj: self._sample_objective_dataset(obj, obj_i, split)
                                    for obj_i, obj in enumerate(self.objectives[split].values())}
        for i, objective in enumerate(objective_sampler):
            try:
                yield next(objectives_data_samplers[objective])
            except StopIteration:
                # TODO: evaluation routine was reported to have raised StopIteration, we should find out why
                # logger.warning("Scheduler %s + Objective %s raised StopIteration.", self, objective)
                continue
            # stop on next requested batch, if we're in the should_stop state from on_log event
            if self.should_stop:
                return

    def iterable_dataset(self, split: str) -> TransformerAdaptationDataset:
        """
        Constructs IterableDataset from the samples of schedule's objective.
        :param split: data split to iterate. `train` or `eval`.
        :return: AdaptationDataset combined according to this Schedule.
        """
        length_combined = int(sum((o.dataset_length[split] // o.batch_size) for o in self.objectives[split].values()))
        if split == "train":
            length_combined *= int(self.args.num_train_epochs)

        return TransformerAdaptationDataset(self._combine_datasets(split), length_combined)


class SequentialSchedule(Schedule):

    label = "sequential"

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in a sequential order - each objective is sampled for its `dataset_length` steps.

        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the references to objectives.
        """
        # infinite loop - termination is determined by _should_stop() + _combine_datasets()
        while True:
            for objective in self.objectives[split].values():
                for _ in range(objective.dataset_length[split]):
                    if objective in self.converged_objectives and not self.args.log_converged_objectives:
                        continue
                    yield objective

    @staticmethod
    def single_iteration_eval_sampling(objectives: Iterable[Objective]) -> Iterable[Objective]:
        """
        Simple finite, single iteration over all objectives. Used by base Schedule for evaluation.
        :param objectives: Objectives to schedule.
        :return:  Iterator over the given to objectives.
        """
        for objective in objectives:
            for _ in range(objective.dataset_length["eval"]):
                yield objective


class ParallelSchedule(Schedule):

    label = "parallel"

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in parallel - choose objectives in Round Robin fashion.
        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return:
        """
        while True:
            for objective in self.objectives[split].values():
                if objective in self.converged_objectives and not self.args.log_converged_objectives:
                    continue
                yield objective
