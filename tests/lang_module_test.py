from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.classification import TokenClassification
from adaptor.utils import Head
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


def test_fresh_lang_module_mt():
    new_lang_module = LangModule.from_data(texts_or_path=paths["texts"]["translation"],
                                           vocab_size=29,  # vocab_size(s) must comply with sentencepiece constraints
                                           tokenizer_type="sentencepiece",
                                           tokenizer_kwargs={"source_lang": "en", "target_lang": "cs"},
                                           model_type="marian",
                                           model_dir="adaptation_output_dir")

    new_lang_module.load_training_head(Head.SEQ2SEQ, "mock_objective")

    input_text = "A piece of text."
    output_text = "Kousek textu."
    sample = new_lang_module.tokenizer(input_text, return_tensors="pt")
    sample["labels"] = new_lang_module.tokenizer(output_text, return_tensors="pt")["input_ids"]
    sample["oid"] = "mock_objective"

    new_model_outputs = new_lang_module(**sample)

    assert new_lang_module.tokenizer.batch_decode(new_model_outputs.argmax(-1))


def test_fresh_lang_module_multilingual_mt():
    from transformers import MBart50Tokenizer
    new_lang_module = LangModule.from_data(texts_or_path=paths["texts"]["translation"],
                                           vocab_size=29,  # vocab_size(s) must comply with sentencepiece constraints
                                           tokenizer_type="sentencepiece",
                                           tokenizer_hf_class=MBart50Tokenizer,
                                           tokenizer_kwargs={"source_lang": "en", "target_lang": "cs"},
                                           model_type="marian",
                                           model_dir="adaptation_output_dir")

    new_lang_module.load_training_head(Head.SEQ2SEQ, "mock_objective")

    input_text = "A piece of text."
    output_text = "Kousek textu."
    sample = new_lang_module.tokenizer(input_text, return_tensors="pt")
    sample["labels"] = new_lang_module.tokenizer(output_text, return_tensors="pt")["input_ids"]
    sample["oid"] = "mock_objective"

    new_model_outputs = new_lang_module(**sample)

    assert new_lang_module.tokenizer.batch_decode(new_model_outputs.argmax(-1))


def test_fresh_lang_module_ner():
    new_lang_module = LangModule.from_data(texts_or_path=paths["texts"]["ner"],
                                           vocab_size=29,  # vocab_size(s) must comply with sentencepiece constraints
                                           tokenizer_type="sentencepiece",
                                           model_type="bert",
                                           model_dir="adaptation_output_dir")

    new_lang_module.load_training_head(Head.TOKEN_CLASSIFICATION, "mock_objective")

    input_text = "A piece of text with King George in it."
    sample = new_lang_module.tokenizer(input_text, return_tensors="pt")
    sample["oid"] = "mock_objective"

    new_model_outputs = new_lang_module(**sample)

    assert new_lang_module.tokenizer.batch_decode(new_model_outputs.argmax(-1))
