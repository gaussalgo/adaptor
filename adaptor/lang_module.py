import logging
import inspect
from typing import List, Dict, Any, Optional

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
    AutoModelForMaskedLM, AutoModelForQuestionAnswering

from .utils import Head

logger = logging.getLogger()


class LangModule(torch.nn.Module):
    """
    Module wrapping models of selected objectives, sharing their common parameters.
    With objectives of different `compatible_head`, LangModule is a multi-head transformer model.
    LangModule instance takes care of:
    1. Merging the shared parameters of objectives' models
    2. Distributing inputs to the heads of corresponding objectives.

    All objectives are expected to share the same tokenizer of HuggingFace PreTrainedTokenizer type.
    """

    tokenizer: PreTrainedTokenizer
    trainable_models: torch.nn.ModuleDict
    heads_output_sizes: Dict[str, int] = {}

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # head_kwargs = head_kwargs if head_kwargs is not None else [{}] * len(head_types)
        # self._load_pretrained_with_heads(model_name_or_path, head_types, head_kwargs)
        self.trainable_models = torch.nn.ModuleDict()

    @staticmethod
    def load_head(model_name_or_path: str,
                  head_type: Head,
                  head_kwargs: Dict[str, Any]) -> torch.nn.Module:
        """
        Returns transformers model with a head of the requested type.
        :param model_name_or_path: base model identifier
        :param head_type: type of the requested head
        :param head_kwargs: additional initialization arguments, adjusting its default, or persisted config
        :return: transformer with a gead of requested type
        """

        if head_type == Head.SEQ_CLASSIFICATION:
            new_head = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **head_kwargs)
        elif head_type == Head.TOKEN_CLASSIFICATION:
            new_head = AutoModelForTokenClassification.from_pretrained(model_name_or_path, **head_kwargs)
        elif head_type == Head.SEQ2SEQ:
            new_head = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **head_kwargs)
        elif head_type == Head.CLM:
            new_head = AutoModelForCausalLM.from_pretrained(model_name_or_path, **head_kwargs)
        elif head_type == Head.MLM:
            new_head = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **head_kwargs)
        elif head_type == Head.QA:
            new_head = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, **head_kwargs)
        else:
            new_head = torch.load(model_name_or_path, **head_kwargs)

        return new_head

    def load_training_head(self,
                           head_type: Head,
                           objective_id: str,
                           head_kwargs: Optional[Dict[str, Any]] = None,
                           new_head: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Registers a selected model to this LangModule, i.e. merges its weights with first one of self.trainable_models,
        and registers new model into self.trainable_models[objective_id].
        :param head_type: if no `new_head` is given, a transformer of self.model_name_or_path
        with a head of `head_type` will be registered.
        :param objective_id: key of the new_head model.
        :param head_kwargs: if transformer is automatically resolved, additional init args of the transformer,
        passed to AutoModelForXY.from_pretrained()
        :param new_head: if given, this would be a selected model to be registered.
        :return:
        """
        # manually-initialized head chosen for this objective will also be merged with other objectives and registered
        if head_kwargs is None:
            head_kwargs = {}
        if new_head is None:
            new_head = self.load_head(self.model_name_or_path, head_type, head_kwargs)

        # this applies to the 2nd+ -added models: they adopt the shared parameters of the first lang_module
        if len(self.trainable_models) >= 1:
            unmatched_modules = self._partially_merge_models(list(self.trainable_models.values())[0], new_head)
            # this can contain a deep stack of layers, hence in general, it can not be checked automatically
            logger.warning("These layers of the loaded %s were not merged: %s" % (head_type.name, unmatched_modules))
        self.trainable_models[objective_id] = new_head

        return new_head

    @staticmethod
    def _partially_merge_models(orig_model: torch.nn.Module,
                                new_model: torch.nn.Module,
                                top_level: bool = True) -> List[str]:
        """
        Matches and merges shared parameters of the models.
        Presumes that a vocabulary (tokenizer) of the both models does match (assured by shared self.tokenizer).
        :param orig_model: lang_module to merge to
        :param new_model: lang_module to merge
        :return: unmatched submodules
        """
        unmatched_modules = []
        children = dict(new_model.named_children())

        if not children:
            # leaf with possibly-shareable parameters

            # Use `share_other_objective_head` if you want to share objectives' complete models,
            # or set the values of the parameters that you want to have merged to the same value.
            for new_param_key, orig_model_param in new_model.named_parameters():
                if new_param_key in dict(orig_model.named_parameters()):
                    # param present in the model to merge new_model into
                    new_model_param = getattr(new_model, new_param_key)
                    orig_model_param = getattr(orig_model, new_param_key)
                    if orig_model_param.shape == new_model_param.shape and torch.all(
                            orig_model_param == new_model_param):
                        setattr(new_model, new_param_key, orig_model_param)
                        assert id(getattr(orig_model, new_param_key)) == id(getattr(new_model, new_param_key))
        else:
            # non-leaf node -> merge in a separate branch
            for child_attr, child_module in children.items():
                if hasattr(orig_model, child_attr):
                    unmatched_modules += LangModule._partially_merge_models(getattr(orig_model, child_attr),
                                                                            getattr(new_model, child_attr),
                                                                            top_level=False)
                else:
                    unmatched_modules += list(dict(getattr(new_model, child_attr).named_parameters()).keys())
        # check that merge-able modules now refer to the same physical address
        if top_level:
            for i, (new_param_key, orig_model_param) in enumerate(orig_model.named_parameters()):
                if new_param_key in dict(new_model.named_parameters()):
                    new_model_param = new_model.get_parameter(new_param_key)
                    if orig_model_param.shape == new_model_param.shape and \
                            torch.all(orig_model_param == new_model_param):
                        assert id(new_model_param) == id(orig_model_param)

        return unmatched_modules

    def forward(self, **inputs) -> torch.LongTensor:
        """
        Performs forward pass over the head identified by the sample's `oid`.
        :param inputs: given head input arguments with corresponding values.
        :return: Raw model outputs (logits).
        """
        try:
            selected_head_model = self.trainable_models[str(inputs["oid"])]
        except KeyError:
            raise ValueError("Requesting inference with the objective having no registered head."
                             "If you are using `extra_eval_objectives`, "
                             "do not forget to fill in their `share_other_objective_head`.")
        # include only correct inputs for a specific model
        list_of_model_specific_inputs = inspect.getfullargspec(selected_head_model.forward).args
        model_specific_inputs = {k: v for k, v in inputs.items() if k in list_of_model_specific_inputs}

        # including labels cause the loss to be computed twice - by objective + by HF models forward()
        # but labels are also used to infer decoder_input_ids of some models, so we need to pass it
        selected_head_output = selected_head_model(**model_specific_inputs)
        # HF models produce special Output objects instead of a raw output
        logits = selected_head_output.logits if hasattr(selected_head_output, "logits") else selected_head_output

        return logits

    def reinitialize(self, seed: int = 42) -> None:
        """
        Resets the trainable weights of all trainable_models.
        :param seed: Seed value for deterministic reinitialization.
        """
        def reinit_model_weights(m: torch.nn.Module):
            if hasattr(m, "children"):
                for m_child in m.children():
                    if hasattr(m_child, "reset_parameters"):
                        m_child.reset_parameters()
                    reinit_model_weights(m_child)

        torch.manual_seed(seed)
        for head, head_model in self.trainable_models.items():
            head_model.apply(reinit_model_weights)
