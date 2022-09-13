import abc
import inspect
from typing import Optional, Union, Dict, Any, Tuple

import torch
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from torch.nn.functional import log_softmax, softmax
from transformers import BatchEncoding, PreTrainedModel
from transformers.utils import ModelOutput

from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective


class Distillation(Objective, abc.ABC):

    def __init__(self, *args,
                 distilled_model: PreTrainedModel,
                 temperature: int = 1,
                 logits_ce_loss_weight: int = 1,
                 hidden_cossim_loss_weight: int = 1,
                 add_hidden_states_loss: bool = False,
                 restrict_loss_to_mask: bool = False,
                 **kwargs):
        self.teacher_model = distilled_model

        self.temperature = temperature
        self.logits_ce_loss_weight = logits_ce_loss_weight
        self.hidden_cossim_loss_weight = hidden_cossim_loss_weight

        self.restrict_loss_to_mask = restrict_loss_to_mask
        self.add_hidden_states_loss = add_hidden_states_loss

        if add_hidden_states_loss:
            # in this case, we'll need the teacher to yield the hidden states in the output
            self.teacher_model.config.output_hidden_states = True

        super().__init__(*args, **kwargs)

    def register_compatible_head_model(self,
                                       lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        if self.add_hidden_states_loss:
            # if the loss is computed also from the hidden_states, we make sure they are actually requested
            if objective_args_for_head_config is not None:
                objective_args_for_head_config["output_hidden_states"] = True
            else:
                objective_args_for_head_config = {"output_hidden_states": True}
        return super(Distillation, self).register_compatible_head_model(lang_module,
                                                                        other_objective,
                                                                        objective_args_for_head_config,
                                                                        preloaded_module)

    def _loss_for_hidden_states(self,
                                student_hidden: Tuple[torch.FloatTensor],
                                teacher_hidden: Tuple[torch.FloatTensor],
                                attn_mask: torch.LongTensor,
                                teacher_select_method: str = "alternate") -> torch.FloatTensor:
        assert student_hidden[0].shape[-1] == teacher_hidden[0].shape[-1], \
            "If adding loss of the hidden states, student and teacher must have embeddings of the same dimension."

        cosine_loss = CosineEmbeddingLoss(reduction="mean")

        # student_hidden = torch.vstack(student_hidden)
        # teacher_hidden = torch.vstack(teacher_hidden)

        if teacher_select_method == "alternate":
            teacher_student_ratio = len(teacher_hidden) / len(student_hidden)
            assert teacher_student_ratio >= 1.0, "Number of teacher's hidden states (%s) " \
                                                 "must bigger than the student's: (%s)" \
                                                 % (len(teacher_hidden), len(student_hidden))

            # select every n-th hidden state of the teacher, as in DistilBERT implementation
            # if the former size is not the multiplier of the latter, we select approximately proportional layers
            selected_teacher_hs = torch.arange(0, len(teacher_hidden), teacher_student_ratio).long()
            teacher_hidden_selected = [hidden for i, hidden in enumerate(teacher_hidden) if i in selected_teacher_hs]
        else:
            raise ValueError("Unknown teacher_select_method: %s" % teacher_select_method)

        student_hidden = torch.vstack([h.unsqueeze(0) for h in student_hidden])
        teacher_hidden_selected = torch.vstack([h.unsqueeze(0) for h in teacher_hidden_selected])

        if self.restrict_loss_to_mask:
            attn_mask_reshaped = attn_mask.unsqueeze(-1).unsqueeze(0).expand_as(student_hidden).bool()

            student_hidden_flat = torch.masked_select(student_hidden, attn_mask_reshaped)
            teacher_hidden_selected_flat = torch.masked_select(teacher_hidden_selected, attn_mask_reshaped)

            student_hidden_unbatched = student_hidden_flat.reshape(-1, student_hidden.shape[-1])
            teacher_hidden_unbatched = teacher_hidden_selected_flat.reshape(-1, student_hidden.shape[-1])
        else:
            student_hidden_unbatched = student_hidden.reshape(-1, student_hidden.shape[-1])
            teacher_hidden_unbatched = teacher_hidden_selected.reshape(-1, student_hidden.shape[-1])

        similarity_or_distance_loss = torch.ones(student_hidden_unbatched.shape[0])

        return cosine_loss(student_hidden_unbatched, teacher_hidden_unbatched, similarity_or_distance_loss)

    def _hidden_states_loss(self,
                            student_outputs: ModelOutput,
                            teacher_outputs: ModelOutput,
                            attn_mask: torch.LongTensor) -> torch.FloatTensor:
        if hasattr(teacher_outputs, "hidden_states"):
            assert hasattr(student_outputs, "hidden_states"), "Student and teacher must be of the same type"
            student_hidden = student_outputs.hidden_states
            teacher_hidden = teacher_outputs.hidden_states

            loss = self._loss_for_hidden_states(student_hidden, teacher_hidden, attn_mask)

        elif hasattr(teacher_outputs, "encoder_hidden_states") and hasattr(teacher_outputs, "decoder_hidden_states"):
            assert hasattr(student_outputs, "encoder_hidden_states") \
                   and hasattr(teacher_outputs, "decoder_hidden_states"), "Student and teacher must be of the same type"

            student_encoder_hidden = student_outputs.encoder_hidden_states
            teacher_encoder_hidden = teacher_outputs.encoder_hidden_states
            student_decoder_hidden = student_outputs.decoder_hidden_states
            teacher_decoder_hidden = teacher_outputs.decoder_hidden_states

            loss = (self._loss_for_hidden_states(student_encoder_hidden, teacher_encoder_hidden, attn_mask) +
                    self._loss_for_hidden_states(student_decoder_hidden, teacher_decoder_hidden, attn_mask))
        else:
            raise ValueError("Please initialize teacher model with `output_hidden_states=True`")

        return loss

    def _compute_loss(self,
                      student_logits: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        assert inputs is not None, "Distillation loss requires model inputs to be passed"

        ce_loss = CrossEntropyLoss()

        teacher_inputs = inspect.getfullargspec(self.teacher_model.forward).args
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**{k: v for k, v in inputs.items() if k in teacher_inputs})
            teacher_logits = teacher_outputs.logits

        if self.restrict_loss_to_mask:
            # pick only the predictions of tokens on the attended positions (i.e. ignore the others)
            attn_mask_reshaped = inputs["attention_mask"].unsqueeze(-1).expand_as(student_logits).bool()

            student_logits_flat = torch.masked_select(student_logits, attn_mask_reshaped)
            student_logits_unbatched = student_logits_flat.reshape(-1, student_logits.shape[-1])

            teacher_logits_flat = torch.masked_select(teacher_logits, attn_mask_reshaped)
            teacher_logits_unbatched = teacher_logits_flat.reshape(-1, student_logits.shape[-1])
        else:
            # we flatten the batch, to get the class scores & probabilities to the 2nd dimension
            student_logits_unbatched = student_logits.flatten(end_dim=1)
            teacher_logits_unbatched = teacher_logits.flatten(end_dim=1)

        distil_loss = ce_loss(log_softmax(student_logits_unbatched / self.temperature, dim=-1),
                              softmax(teacher_logits_unbatched / self.temperature, dim=-1)) * self.temperature ** 2
        distil_loss = self.logits_ce_loss_weight * distil_loss

        if self.add_hidden_states_loss:
            student_inputs = inspect.getfullargspec(self.compatible_head_model.forward).args
            student_outputs = self.compatible_head_model(**{k: v for k, v in inputs.items() if k in student_inputs})

            hidden_loss = self._hidden_states_loss(student_outputs, teacher_outputs, inputs["attention_mask"])
            hidden_loss_scaled = self.hidden_cossim_loss_weight * hidden_loss

            distil_loss = distil_loss + hidden_loss_scaled

        return distil_loss
