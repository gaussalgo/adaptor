from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import CausalLanguageModeling
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.backtranslation import BackTranslation, BackTranslator
from adaptor.objectives.classification import TokenClassification
from adaptor.objectives.distillation import Distillation
from adaptor.objectives.denoising import DenoisingObjective
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.question_answering import ExtractiveQA
from adaptor.objectives.seq2seq import Sequence2Sequence
from utils import paths, test_base_models


def assert_module_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    # outputs = lang_module(**dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(dataset_sample, dataset_sample["labels"], split)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def test_token_classification_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = TokenClassification(lang_module,
                                    texts_or_path=paths["texts"]["ner"],
                                    labels_or_path=paths["labels"]["ner"],
                                    batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_mlm_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = MaskedLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_unsup_objective_bert():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = CausalLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_unsup_objective_marian():
    lang_module = LangModule(test_base_models["translation_mono"])
    objective = CausalLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_denoising_objective():
    lang_module = LangModule(test_base_models["translation_mono"])
    objective = DenoisingObjective(lang_module, texts_or_path=paths["texts"]["unsup"], batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_backtranslation_objective():
    # we are adapting cs->en translator with back-translation
    lang_module = LangModule(test_base_models["translation_mono"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-en-cs")

    objective = BackTranslation(lang_module,
                                back_translator=translator,
                                texts_or_path=paths["texts"]["unsup"],
                                batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_backtranslation_objective_mbart():
    # we are adapting cs->en translator with back-translation
    lang_module = LangModule(test_base_models["translation_multi"]["model"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-en-cs")

    objective = BackTranslation(lang_module,
                                back_translator=translator,
                                texts_or_path=paths["texts"]["unsup"],
                                batch_size=4,
                                source_lang_id=test_base_models["translation_multi"]["test_src_lang"],
                                target_lang_id=test_base_models["translation_multi"]["test_tgt_lang"])

    assert_module_objective_ok(lang_module, objective)


def test_supervised_seq2seq_objective():
    lang_module = LangModule(test_base_models["translation_mono"])
    objective = Sequence2Sequence(lang_module,
                                  texts_or_path=paths["texts"]["translation"],
                                  labels_or_path=paths["labels"]["translation"],
                                  batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_supervised_seq2seq_objective_mbart():
    # we are adapting cs->en translator with back-translation
    lang_module = LangModule(test_base_models["translation_multi"]["model"])

    objective = Sequence2Sequence(lang_module,
                                  texts_or_path=paths["texts"]["translation"],
                                  labels_or_path=paths["labels"]["translation"],
                                  batch_size=4,
                                  source_lang_id=test_base_models["translation_multi"]["test_src_lang"],
                                  target_lang_id=test_base_models["translation_multi"]["test_tgt_lang"])

    assert_module_objective_ok(lang_module, objective)


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


def test_supervised_QA_objective():
    lang_module = LangModule(test_base_models["extractive_QA"])

    objective = ExtractiveQA(lang_module,
                             texts_or_path=paths["texts"]["QA"],
                             text_pair_or_path=paths["text_pair"]["QA"],
                             labels_or_path=paths["labels"]["QA"],
                             batch_size=4)

    assert_module_objective_ok(lang_module, objective)


# def test_search_objective():
#     lang_module = LangModule(test_base_models["extractive_QA"])
#
#     objective = Encoding(lang_module,
#                          texts_or_path=paths["texts"]["QA"],
#                          text_pair_or_path=paths["text_pair"]["QA"],
#                          labels_or_path=paths["labels"]["QA"],
#                          batch_size=4)
#
#     assert_module_objective_ok(lang_module, objective)
