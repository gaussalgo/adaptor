import copy
import itertools
import logging
import os
from typing import List, Dict, Tuple, Union, Optional

from peft import PeftModel
from transformers import WEIGHTS_NAME, TrainerState
import torch
from transformers import Trainer, BatchEncoding
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINER_STATE_NAME

from .lang_module import LangModule
from .schedules import Schedule
from .utils import AdaptationArguments, SavingStrategy, PEFT_BASE_MODEL_CHECKPOINT_SUBDIR

logger = logging.getLogger()


class Adapter(Trainer):
    """
    Adapter instance is a lightweigt wrapper of HuggingFace Trainer.
    1. It performs mapping of IterableDatasets constructed in Schedule, to Trainer(*dataset)
    2. For user convenience, it re-evaluates arguments sanity for (multi-)objective adaptation.
    3. It propagates computation of loss to schedule, which distributes them to corresponding Objectives.
    4. It extends training logs (created in events `on_log` and `on_evaluate`) with objective-specific logs.
    5. It extends model persistence on checkpoints and after the training to a separate model for each Objective.
    """

    permitted_args = ["args", "tokenizer", "callbacks", "optimizers"]
    eval_metrics_prefix = "eval"
    args: AdaptationArguments

    def __init__(self, lang_module: LangModule, schedule: Schedule, args: AdaptationArguments, **kwargs):
        """
        Initialises Adapter, used in the same way as HuggingFace Trainer, refer to its documentation for more features.
        :param lang_module: Wrapper of multi-head model with registered heads for each objective of `schedule`.
        :param schedule: Adaptor's Schedule. Determines ordering of applying training Objectives and other.
        :param args: Positional arguments to be passed to HF Trainer.
        :param kwargs: Keyword arguments to be checked and passed to HF Trainer.
        """
        unexpected_args = [k for k in kwargs.keys() if k not in self.permitted_args]
        if unexpected_args:
            raise ValueError("Adapter(**kwargs) got these unexpected kwargs: %s" % unexpected_args)

        self.schedule = schedule

        orig_callbacks = [] if "callbacks" not in kwargs else kwargs.pop("callbacks")

        all_objectives_ids = list(map(str, self.schedule.objectives["train"].values()))
        if len(set(all_objectives_ids)) < len(all_objectives_ids):
            duplicates = [identifier for identifier in all_objectives_ids if all_objectives_ids.count(identifier) > 1]
            raise ValueError("These objectives have identical identifiers: %s; This would cause "
                             "incorrect persistence of checkpoints for your objectives." % set(duplicates))
        lang_module.finalize()

        super().__init__(model=lang_module,
                         args=args,
                         train_dataset=self.schedule.iterable_dataset(split="train"),
                         eval_dataset=self.schedule.iterable_dataset(split="eval"),
                         data_collator=self.flattened_collator,
                         compute_metrics=None,  # logged metrics are handled by Objectives
                         callbacks=orig_callbacks + [schedule.should_stop_check_callback()],
                         **kwargs)

    @staticmethod
    def flattened_collator(features: List[BatchEncoding]) -> BatchEncoding:
        """
        Objectives take care of their own data collation, so this collator just flattens the outputs of batch_size=1.

        :return: loss and a placeholder of unused outputs, for compatibility
        """
        assert len(features) == 1, "Sorry, for multi-GPU training, we only support DistributedDataParallel for now."

        return features[0]

    def compute_loss(self,
                     model: LangModule,
                     inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, None]]:
        labels = inputs["labels"] if "labels" in inputs else inputs["label"]

        outputs = model(**inputs)
        if self.label_smoother is not None:
            raise NotImplementedError()  # objective-dependent label smoothing is custom
            # loss = self.label_smoother(outputs, labels)
        else:
            loss = self.schedule.compute_loss(outputs, labels, inputs)

        mock_outputs = torch.tensor([-1, -1])
        return (loss, mock_outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        is_eval_log = any(self.eval_metrics_prefix in log_key for log_key in logs)
        extended_logs = self.schedule.objectives_log(split="eval" if is_eval_log else "train")
        return super().log({**logs, **extended_logs})

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        logger.warning("Evaluating...")
        out = super(Adapter, self).evaluate(*args, **kwargs)
        if "metric_key_prefix" in kwargs:
            self.eval_metrics_prefix = kwargs["metric_key_prefix"]

        # refresh exhausted evaluation iteration for possible next evaluation
        self.eval_dataset = self.schedule.iterable_dataset("eval")

        return out

    def _save_module(self, module: torch.nn.Module, output_module_path: str) -> None:
        # simple wrapper to save an arbitrary model to a directory in a standard format
        # for each objective, we also persist a shared tokenizer to make each Objective independently loadable
        self.model.tokenizer.save_pretrained(output_module_path)

        if hasattr(module, "save_pretrained") or hasattr(unwrap_model(module), "save_pretrained"):
            # if the head module has "save_pretrained" method, it will be called for persistence
            module.save_pretrained(output_module_path, use_diff=False, safe_serialization=False)
        else:
            # otherwise, we persist only a raw pytorch module
            torch.save(module.state_dict(), os.path.join(output_module_path, WEIGHTS_NAME))

    def save_model(self, output_dir: Optional[str] = None, **kwargs) -> None:
        # HF native reload compatibility
        all_objectives = set(itertools.chain(self.schedule.objectives["train"].values(),
                                             self.schedule.objectives["eval"].values()))

        objectives_counter = {str(obj): 0 for obj in all_objectives}

        os.makedirs(output_dir, exist_ok=True)

        # also save the base model, if any of our objectives are peft models
        if (self.args.save_peft_base_model and any(
                isinstance(o.compatible_head_model, PeftModel) for o in self.schedule.objectives["train"].values())):
            # For simplicity, we assume that base models for all pefts are the same
            # -- this might be violated only if the user passes custom model_head to Objective
            # and additionally creates a peft module on it.
            # With this assumption, we retrieve a base model from an arbitrary (i.e. the first) peft-model objective
            peft_obj = next(o for o in self.schedule.objectives["train"].values()
                            if isinstance(o.compatible_head_model, PeftModel))

            orig_model = copy.deepcopy(peft_obj.compatible_head_model)
            while isinstance(orig_model, PeftModel):
                # we find cases where unload() does not return the base model on the first call
                orig_model = orig_model.unload()

            base_model_path = os.path.join(output_dir, PEFT_BASE_MODEL_CHECKPOINT_SUBDIR)
            self._save_module(orig_model, base_model_path)
            logger.info(f"Base model for PEFT objectives saved in {base_model_path}")

        for objective in all_objectives:
            if not objective.save_objective_module:
                logger.warning("Skipping objective %s from saving objectives' modules.", objective)
                continue
            module = objective.compatible_head_model
            if (self.args.saving_strategy == SavingStrategy.FINISHED_OBJECTIVES
                    and self.objective not in self.schedule.converged_objectives):
                logger.warning("Not saving model for %s as SavingStrategy is set to FINISHED_OBJECTIVES.", objective)
                continue

            output_module_path = os.path.join(output_dir, str(objective))

            # if the objective of this id was already persisted, we'll index the configs of the next ones
            if objectives_counter[str(objective)] != 0:
                output_module_path += "_{}".format(objectives_counter[str(objective)])
                objectives_counter[str(objective)] += 1

            # training args are shared and persisted in the output_dir root
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if isinstance(module, PeftModel) and self.args.save_peft_base_model:
                base_model_path = os.path.abspath(os.path.join(output_dir, "base_model"))
                module.peft_config['default'].base_model_name_or_path = base_model_path
                logger.warning("Base model for PEFT objective %s set to %s", objective, base_model_path)

            self._save_module(module, output_module_path)
            logger.warning(f"Model of objective {str(objective)} saved in {output_module_path}")
            if self.args.saving_strategy == SavingStrategy.FIRST_OBJECTIVE:
                logger.warning("Skipping other objectives from saving as the chosen SavingStrategy is FIRST_OBJECTIVE.")
                break

    def _load_optimizer_and_scheduler(self, checkpoint: str) -> None:
        # Customizations to support continued training

        # If the training already State exists, it overrides newly-initialized TrainerState (initialized in HF.train())
        possible_state_path = os.path.join(self.model.model_name_or_path, TRAINER_STATE_NAME)
        if os.path.exists(possible_state_path):
            self.state = TrainerState.load_from_json(possible_state_path)
            logger.warning("Restoring training on global step %s", self.state.global_step)

        # in case of continued training, optimizer exists on model.model_name_or_path
        # if the optimizer.pt does not exist, the `super()._load_optimizer_and_scheduler` does not do anything
        return super()._load_optimizer_and_scheduler(checkpoint=self.model.model_name_or_path)
