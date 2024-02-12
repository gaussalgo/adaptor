from adaptor.lang_module import LangModule
from adaptor.objectives.distillation import Distillation
from adaptor.objectives.objective_base import Objective
from utils import test_base_models, paths


def assert_module_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    loss = objective.compute_loss(dataset_sample, dataset_sample["labels"], split)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def test_distillation_seq():
    from adaptor.objectives.seq2seq import Sequence2Sequence
    from transformers import AutoModelForSeq2SeqLM

    class DistilledSeq2Seq(Distillation, Sequence2Sequence):
        # this is a full implementation of distillation within other objective
        pass

    lang_module = LangModule(test_base_models["translation_mono"])
    distilled_model = AutoModelForSeq2SeqLM.from_pretrained(test_base_models["translation_mono"])

    objective = DistilledSeq2Seq(lang_module,
                                 teacher_model=distilled_model,
                                 texts_or_path=paths["texts"]["translation"],
                                 labels_or_path=paths["labels"]["translation"],
                                 batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_distillation_mlm():
    from adaptor.objectives.MLM import MaskedLanguageModeling
    from transformers import AutoModelForMaskedLM

    class DistilledMLM(Distillation, MaskedLanguageModeling):
        pass

    lang_module = LangModule(test_base_models["MLM_student"])
    distilled_model = AutoModelForMaskedLM.from_pretrained(test_base_models["MLM"])

    objective = DistilledMLM(lang_module,
                             teacher_model=distilled_model,
                             texts_or_path=paths["texts"]["unsup"],
                             batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_distillation_mlm_incl_hidden_states():
    from adaptor.objectives.MLM import MaskedLanguageModeling
    from transformers import AutoModelForMaskedLM

    class DistilledMLM(Distillation, MaskedLanguageModeling):
        pass

    lang_module = LangModule(test_base_models["MLM_student"])
    distilled_model = AutoModelForMaskedLM.from_pretrained(test_base_models["MLM"])

    objective = DistilledMLM(lang_module,
                             teacher_model=distilled_model,
                             add_hidden_states_loss=True,
                             texts_or_path=paths["texts"]["unsup"],
                             batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_distillation_mlm_restrict_to_attention():
    from adaptor.objectives.MLM import MaskedLanguageModeling
    from transformers import AutoModelForMaskedLM

    class DistilledMLM(Distillation, MaskedLanguageModeling):
        pass

    lang_module = LangModule(test_base_models["MLM_student"])
    distilled_model = AutoModelForMaskedLM.from_pretrained(test_base_models["MLM"])

    objective = DistilledMLM(lang_module,
                             teacher_model=distilled_model,
                             add_hidden_states_loss=True,
                             restrict_loss_to_mask=True,
                             texts_or_path=paths["texts"]["unsup"],
                             batch_size=4)

    assert_module_objective_ok(lang_module, objective)
