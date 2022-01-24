from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import CausalLanguageModeling
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.backtranslation import BackTranslation, BackTranslator
from adaptor.objectives.classification import TokenClassification
from adaptor.objectives.denoising import DenoisingObjective
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.seq2seq import Sequence2Sequence
from utils import paths, test_base_models

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"


def assert_module_objective_ok(lang_module: LangModule, objective: Objective, split: str = "train"):
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(**dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"], split)

    # check that retrieved loss has a backward_fn
    loss.backward()

    assert True


def test_token_classification_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = TokenClassification(lang_module,
                                    texts_or_path=paths["texts"]["target_domain"]["ner"],
                                    labels_or_path=paths["labels"]["target_domain"]["ner"],
                                    batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_mlm_objective():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = MaskedLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_unsup_objective_bert():
    lang_module = LangModule(test_base_models["token_classification"])
    objective = CausalLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_clm_unsup_objective_marian():
    lang_module = LangModule(test_base_models["translation"])
    objective = CausalLanguageModeling(lang_module,
                                       texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                       batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_denoising_objective():
    lang_module = LangModule(test_base_models["translation"])
    objective = DenoisingObjective(lang_module, texts_or_path=paths["texts"]["target_domain"]["unsup"], batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_backtranslation_objective():
    # we are adapting cs->en translator with back-translation
    lang_module = LangModule(test_base_models["translation"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-en-cs")

    objective = BackTranslation(lang_module,
                                back_translator=translator,
                                texts_or_path=paths["texts"]["target_domain"]["unsup"],
                                batch_size=4)

    assert_module_objective_ok(lang_module, objective)


def test_supervised_seq2seq_objective():
    lang_module = LangModule(test_base_models["translation"])
    objective = Sequence2Sequence(lang_module,
                                  texts_or_path=paths["texts"]["target_domain"]["translation"],
                                  labels_or_path=paths["labels"]["target_domain"]["translation"],
                                  batch_size=4,
                                  source_lang_id="cs",
                                  target_lang_id="en")

    assert_module_objective_ok(lang_module, objective)
