import abc
import itertools
import logging
from typing import List, Union, Optional, Iterable, Tuple, Dict, Sequence, Any

import torch
from tqdm import trange
from transformers import BatchEncoding

from ..evaluators.evaluator_base import EvaluatorBase
from ..lang_module import LangModule
from ..utils import AdaptationDataset, Head, TransformerAdaptationDataset


logger = logging.getLogger()


class Objective(abc.ABC):
    """
    Functionality and attributes inherited in all implemented objectives.
    """

    compatible_head: Head
    given_id: Optional[str]
    epoch: int
    num_steps: int

    texts: Optional[List[str]]
    texts_path: Optional[str]

    val_texts_path: Optional[str]
    val_texts: Optional[List[str]]

    dataset_length: Dict[str, int]
    loss_history: Dict[str, List[float]]
    inputs_history: Dict[str, List[Dict[str, torch.LongTensor]]]
    outputs_history: Dict[str, List[Tuple[torch.FloatTensor, torch.LongTensor]]]
    evaluations_history: Dict[str, Dict[Union[str, EvaluatorBase], List[float]]]
    progressbar: Dict[str, trange]
    evaluators: Dict[str, List[EvaluatorBase]]

    num_samples_per_log: Dict[str, int]

    def __init__(self,
                 lang_module: LangModule,
                 batch_size: int,
                 texts_or_path: Union[str, List[str]],
                 val_texts_or_path: Optional[Union[str, List[str]]] = None,
                 train_evaluators: Sequence[EvaluatorBase] = (),
                 val_evaluators: Sequence[EvaluatorBase] = (),
                 share_other_objective_head: Optional["Objective"] = None,
                 objective_module: Optional[torch.nn.Module] = None,
                 objective_id: Optional[str] = "",
                 loss_weight: Optional[float] = 1,
                 max_samples_per_log: int = 1000,
                 max_samples_per_eval_log: int = 10000):
        """
        Shared initialisation logic of every Objective.
        Registers a compatible model for this objective to given `lang_module`,
        initialises state variables for logging, registers evaluators,
        initialises data set inputs and labels either from path to .txt files, or a lists of strings.

        :param lang_module: LangModule to register a model of this objective into.
        :param batch_size: Sample batch size to be retrieved from this objective.
        :param texts_or_path: A path to a .txt file, or a list of strings that will be used as training inputs.
        :param val_texts_or_path: A path to a .txt file, or a list of strings that will be used as validation inputs.
        :param train_evaluators: Evaluators to be called on every logging step on a subset of training outputs.
        :param val_evaluators: Evaluators to be called on every evaluation step on validation outputs.
        :param share_other_objective_head: If given, this objective will share module with other given objective.
        :param objective_module: If given, this module will be registered for this objective.
        :param objective_id: Identifier of this objective, used in logging and checkpoints persistence.
        Necessary, if you train with multiple objectives of the same type, otherwise they might override each other.
        :param loss_weight: A scalar of the loss of this objective in multi-objective training.
        :param max_samples_per_log: Maximum number of training outputs this objective will remember for logging.
        Reduces memory consumption of the training process.
        :param max_samples_per_eval_log: Maximum number of evaluation outputs this objective will remember for logging.
        """

        self.batch_size = batch_size
        self.tokenizer = lang_module.tokenizer
        self.given_id = objective_id
        self.loss_weight = loss_weight
        self.num_steps = 0

        self.compatible_head_model = self.register_compatible_head_model(lang_module,
                                                                         share_other_objective_head,
                                                                         {},
                                                                         objective_module)
        self.epoch = 0
        self.dataset_length = {"train": 0, "eval": 0}
        self.loss_history = {"train": [], "eval": []}  # loss is treated differently than other outputs
        self.inputs_history = {"train": [], "eval": []}
        self.outputs_history = {"train": [], "eval": []}
        self.evaluators = {"train": [], "eval": []}
        self.evaluations_history = {"train": {}, "eval": {}}
        self.max_samples_per_log = {"train": max_samples_per_log, "eval": max_samples_per_eval_log}

        self.progressbar = {}

        self.texts = None
        self.val_texts = None
        self.texts_path = None
        self.val_texts_path = None

        if type(texts_or_path) == str:
            self.texts_path = texts_or_path
            with open(self.texts_path) as f:
                self.dataset_length["train"] = len(f.readlines())
        else:
            self.texts = texts_or_path
            self.dataset_length["train"] = len(self.texts)
        assert self.dataset_length, \
            "Objective %s was initialized with texts_or_path of zero length, this wouldn't work :("
        for split, given_evaluators in zip(("train", "eval"), (train_evaluators, val_evaluators)):
            for given_evaluator in given_evaluators:
                if given_evaluator.compatible_head != self.compatible_head:
                    raise ValueError("%s got incompatible evaluator: %s" % (self, given_evaluator))
                self.evaluators[split].append(given_evaluator)
                self.evaluations_history[split][given_evaluator] = []

            # loss is objective-dependent, hence we do not delegate it to a separate Evaluator object
            self.evaluations_history[split]["loss"] = []

        if val_texts_or_path is not None:
            if type(val_texts_or_path) == str:
                self.val_texts_path = val_texts_or_path
                with open(self.val_texts_path) as f:
                    self.dataset_length["eval"] = len(f.readlines())
            else:
                self.val_texts = val_texts_or_path
                self.dataset_length["eval"] = len(self.val_texts)

    def per_objective_log(self, split: str) -> Dict[str, float]:
        """
        Generates a log of this objective for a given split, using Evaluators of selected split.
        :param split: Split of the log. Either `train` or `eval`.
        :return: Dict of the format {split + objective_name + evaluator_name: evaluator_value}
        """
        out_logs = {}
        # aggregate per-progress_bar-steps, or per-evaluation-steps, keep the results of unprocessed evaluations
        logger.warning("Constructing %s logs based on %s samples" % (split, len(self.outputs_history[split])))
        if self.outputs_history[split]:
            # if nonempty (last evaluation)
            # aggregate recent losses into the report, clear out losses cache
            mean_loss = sum(self.loss_history[split]) / len(self.loss_history[split])
            self.evaluations_history[split]["loss"].append(mean_loss)

            out_logs["%s_%s_loss" % (split, self)] = mean_loss
            out_logs["%s_%s_num_batches" % (split, self)] = len(self.outputs_history[split])
            for evaluator in self.evaluators[split]:
                n_last_inputs = self.inputs_history[split]
                n_last_logits = [logits for logits, labels in self.outputs_history[split]]
                n_last_labels = [labels for logits, labels in self.outputs_history[split]]

                # evaluator should already return an aggregated value, so unlike loss, we don't average it
                evaluator_value = evaluator(n_last_inputs, self.compatible_head_model,
                                            n_last_logits, n_last_labels, self.tokenizer)
                self.evaluations_history[split][evaluator].append(evaluator_value)
                out_logs["%s_%s_%s" % (split, self, evaluator)] = evaluator_value

            # LM logits, each of shape (batch_size, n_tokens, vocab_size) can consume a lot of memory
            # we erase the raw outputs after the progress_bar, to save space, but we remember the values of Evaluators
            self.inputs_history[split] = []
            self.outputs_history[split] = []
        return out_logs

    def is_finished(self, convergence_patience: Optional[int] = None, max_steps: Optional[int] = None) -> bool:
        """
        Decides on whether this objective has passed a finishing strategy: either convergence by its Evaluators,
        or a maximum number of steps.

        :param convergence_patience: if given, a patience for which the objective did not improve.
        :param max_steps: if given, a maximum number of steps to optimize by this objective.
        :return: true, if any of these condition hold, false otherwise
        """

        if convergence_patience is not None:
            return self.has_converged(convergence_patience)
        elif max_steps is not None:
            return self.num_steps < max_steps
        else:
            return False

    def has_converged(self, patience: int) -> bool:
        """
        Reports if this objective converged, according to the given Evaluators.
        :param patience: number of steps not to improve to be considered converged.
        :return: True, if Objective did not improve for `patience` steps, False otherwise
        """
        convergence_evaluators = [e for e in self.evaluators['eval']
                                  if isinstance(e, EvaluatorBase) and e.determines_convergence]
        if convergence_evaluators:
            stopping_evaluator = convergence_evaluators[0]
        else:
            stopping_evaluator = "loss"

        # the objective was not active in the recent progress_bar interval -> it should not be marked converged
        if not any(self.evaluations_history["train"][e] for e in self.evaluators['train']):
            return False

        passed_patience_evals = len(self.evaluations_history["eval"][stopping_evaluator]) > patience
        if not passed_patience_evals:
            # less than `patience` evaluations has passed so far
            return False
        last_n = self.evaluations_history["eval"][stopping_evaluator][-patience:]
        previous = self.evaluations_history["eval"][stopping_evaluator][:-patience]
        if stopping_evaluator == "loss" or stopping_evaluator.smaller_is_better:
            did_not_improve = min(previous) <= min(last_n)
        else:
            did_not_improve = max(previous) >= max(last_n)

        if did_not_improve:
            logger.warning("Objective `%s` convergence metric `%s` did not improve for %s eval steps. History: %s" %
                           (self, stopping_evaluator, patience, last_n))

        return passed_patience_evals and did_not_improve

    def _register_outputs(self, split: str, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> None:
        """
        Adds model outputs to the memory. Will be later used for generating logs by Evaluators.
        :param split: Data split. Either `train` or `eval`.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.
        """
        self.outputs_history[split].append((logit_outputs.detach().cpu(), labels.detach().cpu()))

        # memory saving cleanup - outputs of 100+ Language modeling outputs can span 100GB+ of memory
        self.inputs_history[split] = self.inputs_history[split][-self.max_samples_per_log[split]:]
        self.outputs_history[split] = self.outputs_history[split][-self.max_samples_per_log[split]:]

    @abc.abstractmethod
    def _compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        An implementation of the loss computation for a given objective.
        Override this, or inherit it from other suitable objective when implementing custom objective.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.
        :return: a single-item torch tensor with registered grad_fn.
        """
        pass

    def compute_loss(self, logit_outputs: torch.FloatTensor, labels: torch.LongTensor, split: str) -> torch.FloatTensor:
        """
        Shared wrapper of objective-specific loss computation. Additionally, it registers model outputs, and labels
        for logging and updates this objective progress bar.
        :param logit_outputs:
        :param labels:
        :param split:
        :return:
        """
        self._register_outputs(split, logit_outputs, labels)
        loss = self._compute_loss(logit_outputs, labels)
        self.loss_history[split].append(loss.item())
        self.num_steps += 1

        self.progressbar[split].set_postfix(refresh=False, split=split, loss=loss.item(), epoch=self.epoch)
        self.progressbar[split].update(1)

        return loss * self.loss_weight

    @abc.abstractmethod
    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Returns an iterator over the encoded batch inputs compatible with a model of this objective.
        Override this, or inherit it from other suitable objective when implementing custom objective.
        If implementing custom iterator, check out for the implementations of similar objectives.
        Beware that encoding and data iteration is a primary bottleneck of training on high-performing GPUs.

        :param split: A split to retrieve encoded inputs for. `train` or `eval`.
        :return: Iterable over the encoded inputs.
        """
        pass

    def get_dataset(self, split: str, objective_i: int, device: Union[str, torch.device]) -> AdaptationDataset:
        """
        Default logic for wrapping the inputs iterator into torch.IterableDataset, used in Trainer.train_dataloaer.
        :param split: A split of the retrieved dataset. `train` or `eval`.
        :param objective_i: Rank of this objective in schedule. Used only to properly set up progress bar.
        :param device: Device to transfer this data set to.

        :return: AdaptationDataset wrapping a data set of this objective.
        """
        self.epoch += 1 if split == "train" else 0

        self.progressbar[split] = trange(self.dataset_length[split] // self.batch_size,
                                         desc=str(self),
                                         unit="batches",
                                         position=objective_i,
                                         leave=True)
        self.progressbar[split].set_postfix(refresh=False, split=split, epoch=self.epoch, loss=-1)

        inputs_iter = self._get_inputs_iterator(split)

        def _sample_to_device(sample: Dict[str, torch.LongTensor]) -> Dict[str, torch.LongTensor]:
            return {k: v.to(device) if k != "oid" else v for k, v in sample.items()}

        def _add_oid(sample: Dict[str, Union[int, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            sample["oid"] = id(self)
            return sample

        def _register_input_sample(sample: Dict[str, Union[int, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            self.inputs_history[split].append(sample)
            return sample

        device_inputs_iter = map(_sample_to_device, inputs_iter)
        device_inputs_iter = map(_add_oid, device_inputs_iter)
        device_inputs_iter = map(_register_input_sample, device_inputs_iter)

        return TransformerAdaptationDataset(device_inputs_iter, self.dataset_length[split])

    @abc.abstractmethod
    def _per_split_iterators(self, split: str) -> Union[Iterable[str], Tuple[Iterable[str], Iterable[str]]]:
        """
        Implementations of shared (un/)supervised iterations in (Un/)SupervisedObjective.
        Not meant to be overriden when implementing custom data set. Instead choose to inherit either
        from SupervisedObjective, or UnsupervisedObjective (or their ancestors).

        :param split: Data split to iterate over

        :return: A pair of [inputs_iterator, labels_iterator]
        """
        pass

    def register_compatible_head_model(self, lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Resolves a model of this objective in given lang_module. Either requests LangModule to provide model with
        self.compatible_head, or asks to register custom model (if `preloaded_module` is given).

        :param lang_module: LangModule instance to register objetive into.
        :param other_objective: if given, an objective to share the model with.
        :param objective_args_for_head_config: if given, and no `preloaded_module` given, additional configuration
        parameters for AutoModel initialisation of compatible_model in LangModule.
        :param preloaded_module: if given, a model to be registered in lang_module: no AutoModel initialisation
        would be called, but `preloaded_module`'s parameters will still be shared within given `lang_module`.

        :return: a module registered in `lang_module` for this objective.
        """
        head_config = objective_args_for_head_config if objective_args_for_head_config is not None else {}

        if other_objective is not None:
            logger.warning("Objective %s will use %s head of %s objective",
                           self, self.compatible_head.name, other_objective)
            preloaded_module = other_objective.compatible_head_model

        return lang_module.load_training_head(self.compatible_head, str(id(self)), head_config, preloaded_module)

    def __str__(self) -> str:
        """
        Default pretty print of this objective. Identification used also in the logs.
        :return: string identifier of this objective.
        """
        if self.given_id:
            return str("%s-%s" % (self.given_id, self.__class__.__name__))
        else:
            return self.__class__.__name__


class UnsupervisedObjective(Objective, abc.ABC):

    def _per_split_iterator_single(self, split: str) -> Iterable[str]:
        """
        An iterator over unsupervised texts.
        :param split: split to iterate data over
        :return: Iterable of input texts.
        """
        if split == "train":
            if self.texts is not None:
                sources_iter = iter(self.texts)
            else:
                sources_iter = AdaptationDataset.iter_text_file_per_line(self.texts_path)
        elif split == "eval":
            if self.val_texts is not None:
                sources_iter = iter(self.val_texts)
            elif self.val_texts_path is not None:
                sources_iter = AdaptationDataset.iter_text_file_per_line(self.val_texts_path)
            else:
                raise ValueError("Objective %s did not get any validation texts :( "
                                 "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, "
                                 "or set Objective(val_texts) param." % self)
        else:
            raise ValueError("Unrecognized split: %s" % split)

        return sources_iter

    def _per_split_iterators(self, split: str) -> Tuple[Iterable[str], Iterable[str]]:
        """
        Default inputs iterator for unsupervised objectives. Returns input texts as both inputs and labels.
        Not meant to be overriden when implementing custom data set. Instead choose to inherit either
        :param split: Data split to iterate over
        :return: a pair of identical [inputs_iterator, inputs_iterator]
        """
        return self._per_split_iterator_single(split), self._per_split_iterator_single(split)


class SupervisedObjective(UnsupervisedObjective, abc.ABC):

    labels_path: Optional[str] = None
    labels: Optional[List[str]] = None

    val_labels_path: Optional[str] = None
    val_labels: Optional[List[str]] = None

    labels_map: Dict[str, int] = {}

    def __init__(self,
                 *args,
                 labels_or_path: Union[str, List[str]],
                 val_labels_or_path: Optional[Union[str, List[str]]] = None,
                 **kwargs):

        if type(labels_or_path) == str:
            self.labels_path = labels_or_path
        else:
            self.labels = labels_or_path

        if val_labels_or_path is not None:
            if type(val_labels_or_path) == str:
                self.val_labels_path = val_labels_or_path
            else:
                self.val_labels = val_labels_or_path

        # init will call register_compatible_head_model, which resolves num_labels for new head config from self.labels
        super().__init__(*args, **kwargs)

    def register_compatible_head_model(self, lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Additionally adds labels into a configuration of this objective's model in lang_module.
        Refer further to the documentation of the superclass.
        """
        # supervised objective additionally keeps track of labels persistence in config
        if self.compatible_head in (Head.TOKEN_CLASSIFICATION, Head.SEQ_CLASSIFICATION):
            if self.labels is not None:
                all_labels = self.labels
            else:
                all_labels = [l.strip() for l in AdaptationDataset.iter_text_file_per_line(self.labels_path)]
            if self.val_labels is not None:
                all_labels += self.val_labels
            elif self.val_labels_path is not None:
                all_labels += [l.strip() for l in AdaptationDataset.iter_text_file_per_line(self.val_labels_path)]

            if self.compatible_head == Head.TOKEN_CLASSIFICATION:
                all_labels = set(itertools.chain(*(token_labels_str.split() for token_labels_str in all_labels)))

            self.labels_map = {val: i for i, val in enumerate(sorted(set(all_labels)))}

            objective_args_for_head_config = {"num_labels": len(all_labels),
                                              "label2id": self.labels_map,
                                              "id2label": {v: k for k, v in self.labels_map.items()},
                                              **objective_args_for_head_config}

        return super().register_compatible_head_model(lang_module, other_objective,
                                                      objective_args_for_head_config, preloaded_module)

    def _per_split_iterators(self, split: str) -> Tuple[Iterable[str], Iterable[str]]:
        """
        Default inputs iterator for supervised objectives. Returns a pair of iterators, over input texts and labels.
        Not meant to be overriden when implementing custom data set. Instead choose to inherit either
        :param split: Data split to iterate over
        :return: a pair of identical [inputs_iterator, labels_iterator]
        """
        sources_iter, _ = super(SupervisedObjective, self)._per_split_iterators(split)

        if split == "train":
            if self.texts is not None:
                targets_iter = iter(self.labels)
            else:
                targets_iter = AdaptationDataset.iter_text_file_per_line(self.labels_path)
        elif split == "eval":

            if self.val_labels is not None:
                targets_iter = iter(self.val_labels)
            elif self.val_labels_path is not None:
                targets_iter = AdaptationDataset.iter_text_file_per_line(self.val_labels_path)
            else:
                raise ValueError("Objective %s did not get any validation labels :( "
                                 "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, "
                                 "or set Objective(val_labels) param." % self)
        else:
            raise ValueError("Unrecognized split: %s" % split)

        return sources_iter, targets_iter
