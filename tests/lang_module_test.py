from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.classification import TokenClassification
from utils import paths


def test_lang_module():
    lang_module = LangModule("bert-base-multilingual-cased")
    assert lang_module


def test_register_head():
    lang_module = LangModule("bert-base-multilingual-cased")

    objective = TokenClassification(lang_module,
                                    texts_or_path=paths["texts"]["ner"],
                                    labels_or_path=paths["labels"]["ner"],
                                    batch_size=4)
    assert objective.compatible_head_model


def test_merge_objectives():
    import torch
    lang_module = LangModule("bert-base-multilingual-cased")

    objective_base = TokenClassification(lang_module,
                                         texts_or_path=paths["texts"]["ner"],
                                         labels_or_path=paths["labels"]["ner"],
                                         batch_size=4)
    objective_new = MaskedLanguageModeling(lang_module,
                                           texts_or_path=paths["texts"]["unsup"],
                                           batch_size=4)

    # check that merge-able modules now refer to the same physical address
    for i, (new_param_key, orig_model_param) in enumerate(objective_base.compatible_head_model.named_parameters()):
        if new_param_key in dict(objective_new.compatible_head_model.named_parameters()):
            new_model_param = objective_new.compatible_head_model.get_parameter(new_param_key)
            if orig_model_param.shape == new_model_param.shape and \
                    torch.all(orig_model_param == new_model_param):
                assert id(new_model_param) == id(orig_model_param)
