from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification, \
    AutoModelForSequenceClassification

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.evaluators.token_classification import MeanFScore, AverageAccuracy
from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.backtranslation import BackTranslation, BackTranslator
from adaptor.objectives.classification import SequenceClassification, TokenClassification
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule, SequentialSchedule
from utils import training_arguments, paths, test_base_models

unsup_target_domain_texts = paths["texts"]["unsup"]
sup_target_domain_texts = paths["texts"]["ner"]
sup_target_domain_labels = paths["labels"]["ner"]

ner_model_out_dir = "entity_detector_model"


def test_adaptation_ner():
    # 1. pick the models - randomly pre-initialize the appropriate heads
    lang_module = LangModule(test_base_models["token_classification"])

    # 2. pick objectives
    # Objectives take either List[str] for in-memory iteration, or a source file path for streamed iteration
    objectives = [MaskedLanguageModeling(lang_module,
                                         batch_size=1,
                                         texts_or_path=paths["texts"]["unsup"]),
                  TokenClassification(lang_module,
                                      batch_size=1,
                                      texts_or_path=paths["texts"]["ner"],
                                      labels_or_path=paths["labels"]["ner"])]

    # 4. pick a schedule of the selected objectives
    schedule = SequentialSchedule(objectives, training_arguments)

    # 5. Run the training using Adapter, similarly to running HF.Trainer, only adding `schedule`
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 6. save the trained lang_module (with all heads)
    adapter.save_model(ner_model_out_dir)

    # 7. reload and use it like any other Hugging Face model
    ner_model = AutoModelForTokenClassification.from_pretrained("%s/TokenClassification" % ner_model_out_dir)
    tokenizer = AutoTokenizer.from_pretrained("%s/TokenClassification" % ner_model_out_dir)

    inputs = tokenizer("Is there any Abraham Lincoln here?", return_tensors="pt")
    outputs = ner_model(**inputs)
    ner_tags = [ner_model.config.id2label[label_id.item()] for label_id in outputs.logits[0].argmax(-1)]

    assert ner_tags


def test_adaptation_translation():
    # 1. pick the models - randomly pre-initialize the appropriate heads
    lang_module = LangModule(test_base_models["translation_mono"])

    # (optional) pick train and validation evaluators for the objectives
    seq2seq_evaluators = [BLEU(use_generate=True, decides_convergence=True)]

    # 2. pick objectives
    objectives = [BackTranslation(lang_module,
                                  back_translator=BackTranslator("Helsinki-NLP/opus-mt-cs-en"),
                                  batch_size=1,
                                  texts_or_path=paths["texts"]["unsup"]),
                  Sequence2Sequence(lang_module, batch_size=1,
                                    texts_or_path=paths["texts"]["translation"],
                                    val_evaluators=seq2seq_evaluators,
                                    labels_or_path=paths["labels"]["translation"])]
    # 3. pick a schedule of the selected objectives
    # this one will shuffle the batches of both objectives
    schedule = ParallelSchedule(objectives, training_arguments)

    # 4. train using Adapter
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()

    # 5. save the trained (multi-headed) lang_module
    adapter.save_model("translator_model")

    # 6. reload and use it like any other Hugging Face model
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("translator_model/Sequence2Sequence")
    tokenizer = AutoTokenizer.from_pretrained("translator_model/Sequence2Sequence")
    tokenizer.src_lang, tokenizer.tgt_lang = "en", "cs"

    inputs = tokenizer("A piece of text to translate.", return_tensors="pt")
    output_ids = translator_model.generate(**inputs)
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(output_text)


def test_evaluation_ner():
    # first, create a model to evaluate:
    test_adaptation_ner()

    # reload LangModule from this directory
    lang_module = LangModule("%s/TokenClassification" % ner_model_out_dir)

    evaluators = [MeanFScore(), AverageAccuracy()]
    # evaluate the result again through the Objective, that takes care of labels alignment
    eval_ner_objective = TokenClassification(lang_module,
                                             batch_size=1,
                                             texts_or_path=[],
                                             labels_or_path=[],
                                             val_texts_or_path=paths["texts"]["ner"],
                                             val_labels_or_path=paths["labels"]["ner"],
                                             val_evaluators=evaluators)

    evaluation = eval_ner_objective.per_objective_log("eval")
    for evaluator in evaluators:
        assert "eval_%s_%s" % (eval_ner_objective, evaluator) in evaluation


def test_adaptation_classification():

    lang_module = LangModule(test_base_models["sequence_classification"])

    mlm = MaskedLanguageModeling(lang_module=lang_module,
                                 texts_or_path=paths["texts"]["unsup"],
                                 batch_size=1)

    classification = SequenceClassification(lang_module=lang_module,
                                            texts_or_path=paths["texts"]["classification"],
                                            labels_or_path=paths["labels"]["classification"],
                                            batch_size=1)

    parallel_schedule = ParallelSchedule(objectives=[mlm, classification],
                                         args=training_arguments)
    # 4. train using Adapter
    adapter = Adapter(lang_module=lang_module,
                      schedule=parallel_schedule,
                      args=training_arguments)
    adapter.train()

    # 5. save the trained (multi-headed) lang_module
    adapter.save_model("output_models")

    # 6. reload and use it like any other Hugging Face model
    classifier = AutoModelForSequenceClassification.from_pretrained("output_models/SequenceClassification")
    tokenizer = AutoTokenizer.from_pretrained("output_models/SequenceClassification")

    inputs = tokenizer("A piece of text to translate.", return_tensors="pt")
    output = classifier(**inputs)
    output_label_id = output.logits.argmax(-1)[0].item()
    print("Prediction: %s" % classifier.config.id2label[output_label_id])
