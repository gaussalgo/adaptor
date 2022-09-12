import abc
import itertools
import logging
from typing import List, Union, Optional, Iterable, Tuple, Dict, Sequence, Any, Iterator

import torch
from tqdm import trange
from transformers import BatchEncoding, DataCollatorWithPadding

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
    last_input: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]]

    texts: Optional[List[str]]
    texts_path: Optional[str]

    val_texts_path: Optional[str]
    val_texts: Optional[List[str]]

    dataset_length: Dict[str, int]
    loss_history: Dict[str, List[float]]
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
                 max_samples_per_eval_log: int = 10000,
                 remember_last_input: Optional[bool] = False):
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
        :param max_samples_per_log: Maximum number batches that this objective will compute for logging.
        :param max_samples_per_eval_log: Maximum number batches that this objective will compute for evaluation logging.
        :param remember_last_input: Debugging feature: whether the objective should remember the last input
        to its compatible model. Useful for debugging a development of the new objective;
        If the training fails (in the interactive - `-i` mode), the last, possibly error input can be retrieved
        from `this_objective.last_input`.
        """

        self.batch_size = batch_size
        self.tokenizer = lang_module.tokenizer
        self.given_id = objective_id
        self.loss_weight = loss_weight
        self.num_steps = 0
        self.remember_last_input = remember_last_input
        self.last_input = None

        self.compatible_head_model = self.register_compatible_head_model(lang_module,
                                                                         share_other_objective_head,
                                                                         {},
                                                                         objective_module)
        self.epoch = 0
        self.dataset_length = {"train": 0, "eval": 0}
        self.loss_history = {"train": [], "eval": []}  # loss is treated differently than other outputs
        self.evaluators = {"train": [], "eval": []}
        self.evaluations_history = {"train": {}, "eval": {}}
        self.max_samples_per_log = {"train": max_samples_per_log, "eval": max_samples_per_eval_log}

        self.progressbar = {}

        self.texts = None
        self.val_texts = None
        self.texts_path = None
        self.val_texts_path = None

        if isinstance(texts_or_path, str):
            self.texts_path = texts_or_path
            with open(self.texts_path) as f:
                self.dataset_length["train"] = len(f.readlines())
        else:
            self.texts = texts_or_path
            self.dataset_length["train"] = len(self.texts)

        for split, given_evaluators in zip(("train", "eval"), (train_evaluators, val_evaluators)):
            for given_evaluator in given_evaluators:
                if self.compatible_head not in given_evaluator.compatible_heads:
                    raise ValueError("%s got incompatible evaluator: %s" % (self, given_evaluator))
                self.evaluators[split].append(given_evaluator)
                self.evaluations_history[split][given_evaluator] = []

            # loss is objective-dependent, hence we do not delegate it to a separate Evaluator object
            self.evaluations_history[split]["loss"] = []

        if val_texts_or_path is not None:
            if isinstance(val_texts_or_path, str):
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
        loss_history = self.loss_history[split][-self.max_samples_per_log[split]:]
        mean_loss = sum(loss_history) / len(loss_history) if len(loss_history) else 0
        self.evaluations_history[split]["loss"].append(mean_loss)

        out_logs["%s_%s_loss" % (split, self)] = mean_loss
        out_logs["%s_%s_num_batches" % (split, self)] = len(loss_history)

        for evaluator in self.evaluators[split]:
            dataset = self.get_dataset(split, 0, self.compatible_head_model.device,
                                       firstn=self.max_samples_per_log[split],
                                       add_oid=False,
                                       is_training_dataset=False)
            # evaluator should already return an aggregated value, so unlike loss, we don't average it
            evaluator_value = evaluator(self.compatible_head_model, self.tokenizer, dataset)
            self.evaluations_history[split][evaluator].append(evaluator_value)
            out_logs["%s_%s_%s" % (split, self, evaluator)] = evaluator_value

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

    @abc.abstractmethod
    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        An implementation of the loss computation for a given objective.
        Override this, or inherit it from other suitable objective when implementing custom objective.
        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.
        :return: a single-item torch tensor with registered grad_fn.
        """
        pass

    def compute_loss(self,
                     logit_outputs: torch.FloatTensor,
                     labels: torch.LongTensor,
                     inputs: Union[BatchEncoding, Dict[str, torch.Tensor]] = None,
                     split: Optional[str] = "") -> torch.FloatTensor:
        """
        Shared wrapper of objective-specific loss computation. Additionally, it registers model outputs, and labels
        for logging and updates this objective progress bar.
        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.
        :param split: Dataset split. `train` or `eval`.
        :return: a single-item torch tensor with registered grad_fn.
        """
        loss = self._compute_loss(logit_outputs, labels, inputs)
        self.loss_history[split].append(loss.item())
        self.num_steps += 1

        if self.progressbar[split] is not None:
            self.progressbar[split].set_postfix(refresh=False, split=split, loss=loss.item(), epoch=self.epoch)

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

    def get_dataset(self,
                    split: str,
                    objective_i: int,
                    device: Union[str, torch.device],
                    firstn: Optional[int] = None,
                    add_oid: bool = True,
                    is_training_dataset: bool = True,
                    show_progressbar: bool = True) -> TransformerAdaptationDataset:
        """
        Default logic for wrapping the inputs iterator into torch.IterableDataset, used in Trainer.train_dataloaer.
        :param split: A split of the retrieved dataset. `train` or `eval`.
        :param objective_i: Rank of this objective in schedule. Used only to properly set up progress bar.
        :param device: Device to transfer this data set to.
        :param firstn: If given, a number of the retrieved items from the dataset.
        :param add_oid: Whether to append objective id to the match. Required for forward pass over LangModule.
        :param is_training_dataset: Whether this dataset is used for training -> if to update the epochs counter.

        :return: TransformerAdaptationDataset wrapping a data set of this objective.
        """
        if split == "train" and is_training_dataset:
            # increment epoch only for train split and only for the dataset used as training
            # - get_dataset is also called from self.per_objective_log, or specific objectives
            self.epoch += 1 if split == "train" else 0

        if show_progressbar:
            self.progressbar[split] = trange(self.dataset_length[split] // self.batch_size,
                                             desc=str(self),
                                             unit="batches",
                                             position=objective_i,
                                             leave=True)
            self.progressbar[split].set_postfix(refresh=False, split=split, epoch=self.epoch, loss=-1)
        else:
            # we do not update loss, if no progress bar is pertained
            self.progressbar[split] = None

        inputs_iter = self._get_inputs_iterator(split)

        def _sample_to_device(sample: Union[BatchEncoding, Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            return {k: v.to(device) if k != "oid" else v for k, v in sample.items()}

        def _add_oid(sample: Union[BatchEncoding, Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            sample["oid"] = id(self)
            return sample

        def _remember_input(sample: Union[BatchEncoding, Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            self.last_input = sample
            return sample

        def _update_pbar(sample: Union[BatchEncoding, Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            self.progressbar[split].update(1)
            return sample

        device_inputs_iter = map(_sample_to_device, inputs_iter)

        if add_oid:
            device_inputs_iter = map(_add_oid, device_inputs_iter)

        if firstn is not None and firstn < self.dataset_length[split]:
            device_inputs_iter = itertools.islice(device_inputs_iter, firstn)

        if self.remember_last_input:
            device_inputs_iter = map(_remember_input, device_inputs_iter)

        if show_progressbar:
            device_inputs_iter = map(_update_pbar, device_inputs_iter)

        return TransformerAdaptationDataset(device_inputs_iter, self.dataset_length[split])

    def compute_loss_on_last_sample(self) -> torch.FloatTensor:
        """
        This method aims to reproduce an error of calling the `objective.compatible_head_model`
        on the `objective.last_input`. Useful for debugging new objective(s) in interactive (`-i`) mode.
        """
        if not self.remember_last_input:
            raise ValueError("This objective does not remember its last output. "
                             "For debugging, initialize the objective with `remember_last_input=True`.")

        logger.warning("Reproducing loss computation on the last sample")
        logger.warning("The last sample can be retrieved from `this_objective_instance.last_input`")
        labels = self.last_input["labels"]

        logger.warning("Computing model output")
        model_inputs = {k: v for k, v in self.last_input.items() if k not in ("oid", "labels")}
        logits = self.compatible_head_model(**model_inputs).logits

        logger.warning("Computing loss")
        loss = self._compute_loss(logits, labels, self.last_input)

        logger.warning("Loss computation on the recent sample successful. Loss value: %s", loss.item())
        return loss

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

    def register_compatible_head_model(self,
                                       lang_module: LangModule,
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

    text_pair_path: Optional[str] = None
    text_pair: Optional[List[str]] = None

    val_text_pair_path: Optional[str] = None
    val_text_pair: Optional[List[str]] = None

    labels_map: Dict[str, int] = {}

    def __init__(self,
                 *args,
                 labels_or_path: Union[str, List[str]],
                 val_labels_or_path: Optional[Union[str, List[str]]] = None,
                 text_pair_or_path: Optional[Union[str, List[str]]] = None,
                 val_text_pair_or_path: Optional[Union[str, List[str]]] = None,
                 **kwargs):

        if isinstance(labels_or_path, str):
            self.labels_path = labels_or_path
        else:
            self.labels = labels_or_path

        if val_labels_or_path is not None:
            if isinstance(val_labels_or_path, str):
                self.val_labels_path = val_labels_or_path
            else:
                self.val_labels = val_labels_or_path

        if text_pair_or_path is not None:
            if isinstance(text_pair_or_path, str):
                self.text_pair_path = text_pair_or_path
            else:
                self.text_pair = text_pair_or_path

        if val_text_pair_or_path is not None:
            if isinstance(val_text_pair_or_path, str):
                self.val_text_pair_path = val_text_pair_or_path
            else:
                self.val_text_pair = val_text_pair_or_path

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
                all_labels = [line.strip() for line in AdaptationDataset.iter_text_file_per_line(self.labels_path)]
            if self.val_labels is not None:
                all_labels += self.val_labels
            elif self.val_labels_path is not None:
                all_labels += [line.strip() for line in AdaptationDataset.iter_text_file_per_line(self.val_labels_path)]

            if self.compatible_head == Head.TOKEN_CLASSIFICATION:
                all_labels = set(itertools.chain(*(token_labels_str.split() for token_labels_str in all_labels)))

            self.labels_map = {val: i for i, val in enumerate(sorted(set(all_labels)))}

            objective_args_for_head_config = {"num_labels": len(all_labels),
                                              "label2id": self.labels_map,
                                              "id2label": {v: k for k, v in self.labels_map.items()},
                                              **objective_args_for_head_config}

        return super().register_compatible_head_model(lang_module, other_objective,
                                                      objective_args_for_head_config, preloaded_module)

    def _get_inputs_iterator(self, split: str) -> Iterator[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Batches and encodes input texts and corresponding labels.
        :param split: Selected data split. `train` or `eval`.
        :return: Iterator over batch encodings.
        """

        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        classifying_pairs = None

        batch_features = []
        for source_target_tuple in zip(*self._per_split_iterators(split)):
            # check from the first sample
            if classifying_pairs is None:
                # if the input texts are tab-separated we will tokenize them as pairs
                classifying_pairs = len(source_target_tuple) > 2
                if classifying_pairs:
                    assert len(source_target_tuple) == 3, "Expecting tuples of (source, source_pair, target) texts"
            if classifying_pairs:
                text, text_pair, label = source_target_tuple
                out_sample = self.tokenizer(text, text_pair=text_pair, truncation=True)
            else:
                text, label = source_target_tuple
                out_sample = self.tokenizer(text, truncation=True)

            out_sample["label"] = torch.tensor(self.labels_map[label])

            batch_features.append(out_sample)
            if len(batch_features) == self.batch_size:
                yield collator(batch_features)
                batch_features = []

        if batch_features:
            # yield residual batch
            yield collator(batch_features)

    def _per_split_iterators(self, split: str) -> Union[Tuple[Iterable[str], Iterable[str]],
                                                        Tuple[Iterable[str], Iterable[str], Iterable[str]]]:
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
            if self.text_pair is not None:
                source_pairs_iter = iter(self.text_pair)
            elif self.text_pair_path is not None:
                source_pairs_iter = AdaptationDataset.iter_text_file_per_line(self.text_pair_path)
            else:
                source_pairs_iter = None

        elif split == "eval":
            if self.val_labels is not None:
                targets_iter = iter(self.val_labels)
            elif self.val_labels_path is not None:
                targets_iter = AdaptationDataset.iter_text_file_per_line(self.val_labels_path)
            else:
                raise ValueError("Objective %s did not get any validation labels :( "
                                 "Hint: pass `AdaptationArgs(do_eval=False)` to avoid evaluation, "
                                 "or set Objective(val_labels) param." % self)

            if self.val_text_pair is not None:
                source_pairs_iter = iter(self.val_text_pair)
            elif self.val_text_pair_path is not None:
                source_pairs_iter = AdaptationDataset.iter_text_file_per_line(self.val_text_pair_path)
            else:
                source_pairs_iter = None
        else:
            raise ValueError("Unrecognized split: %s" % split)
        if source_pairs_iter is not None:
            return sources_iter, source_pairs_iter, targets_iter
        else:
            return sources_iter, targets_iter
