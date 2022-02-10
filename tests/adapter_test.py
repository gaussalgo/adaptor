from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.backtranslation import BackTranslation, BackTranslator
from adaptor.objectives.classification import TokenClassification
from adaptor.objectives.denoising import DenoisingObjective
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import SequentialSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from utils import paths, test_base_models


training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=2,
                                         log_level="critical",
                                         logging_steps=1,
                                         num_train_epochs=3)


def run_adaptation(adapter: Adapter, trained_model_output_dir: str = "adaptation_output_dir/finished"):
    adapter.train()
    adapter.save_model(trained_model_output_dir)


def test_ner_adaptation():
    lang_module = LangModule(test_base_models["token_classification"])
    objectives = [MaskedLanguageModeling(lang_module,
                                         texts_or_path=paths["texts"]["unsup"],
                                         batch_size=1),
                  TokenClassification(lang_module,
                                      texts_or_path=paths["texts"]["ner"],
                                      labels_or_path=paths["labels"]["ner"],
                                      batch_size=1)]

    schedule = SequentialSchedule(objectives, training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)

    run_adaptation(adapter)


def test_mt_adaptation():
    lang_module = LangModule(test_base_models["translation_mono"])
    objectives = [DenoisingObjective(lang_module,
                                     texts_or_path=paths["texts"]["unsup"],
                                     batch_size=1),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1)]

    schedule = SequentialSchedule(objectives, training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)

    run_adaptation(adapter)


def test_mt_adaptation_bt():
    lang_module = LangModule(test_base_models["translation_mono"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-cs-en")
    objectives = [BackTranslation(lang_module,
                                  back_translator=translator,
                                  texts_or_path=paths["texts"]["unsup"],
                                  batch_size=4),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1)]

    schedule = SequentialSchedule(objectives, training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)

    run_adaptation(adapter)
