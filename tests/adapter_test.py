import os

from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.backtranslation import BackTranslation, BackTranslator
from adaptor.objectives.classification import TokenClassification
from adaptor.objectives.denoising import DenoisingObjective
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import SequentialSchedule, ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from utils import paths, test_base_models

SAVE_STEPS = 1


training_args_map = {
    "output_dir": "adaptation_output_dir",
    "stopping_strategy": StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
    "do_train": True,
    "do_eval": True,
    "gradient_accumulation_steps": 2,
    "log_level": "critical",
    "logging_steps": 1,
    "num_train_epochs": 3,
    "no_cuda": True,
}


training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=2,
                                         log_level="critical",
                                         logging_steps=1,
                                         num_train_epochs=3,
                                         eval_steps=SAVE_STEPS,
                                         save_steps=SAVE_STEPS,
                                         )


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

    args = AdaptationArguments(**training_args_map)

    schedule = SequentialSchedule(objectives, args)
    adapter = Adapter(lang_module, schedule, args=args)

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

    args = AdaptationArguments(**training_args_map)

    schedule = SequentialSchedule(objectives, args)
    adapter = Adapter(lang_module, schedule, args=args)

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

    args = AdaptationArguments(**training_args_map)

    schedule = SequentialSchedule(objectives, args)
    adapter = Adapter(lang_module, schedule, args=args)

    run_adaptation(adapter)


def test_continued_training():
    lang_module = LangModule(test_base_models["translation_mono"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-cs-en")
    objectives = [BackTranslation(lang_module,
                                  back_translator=translator,
                                  texts_or_path=paths["texts"]["unsup"],
                                  batch_size=2,
                                  peft_objective=False),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1,
                                    peft_objective=False)]

    args = AdaptationArguments(**{**training_args_map, **{"eval_steps": SAVE_STEPS, "save_steps": SAVE_STEPS}})

    schedule = SequentialSchedule(objectives, args)
    adapter = Adapter(lang_module, schedule, args=args)

    # first training iteration
    run_adaptation(adapter)

    # second training iteration - continue from the checkpoints persisted for each objective
    lang_module = LangModule(os.path.join(training_arguments.output_dir, "checkpoint-%s" % SAVE_STEPS))
    objectives = [BackTranslation(lang_module,
                                  back_translator=translator,
                                  texts_or_path=paths["texts"]["unsup"],
                                  batch_size=2),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1)]

    schedule = ParallelSchedule(objectives, training_arguments)
    adapter = Adapter(lang_module, schedule, args=training_arguments)
    run_adaptation(adapter)


def test_continued_training_peft():
    from peft import LoraConfig, TaskType

    lang_module = LangModule(test_base_models["translation_mono"])
    translator = BackTranslator("Helsinki-NLP/opus-mt-cs-en")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules=["k_proj", "q_proj", "v_proj", "fc1", "fc2"]  # Marian should have default
    )

    objectives = [BackTranslation(lang_module,
                                  back_translator=translator,
                                  texts_or_path=paths["texts"]["unsup"],
                                  batch_size=2,
                                  objective_args_for_head_config={"peft_config": peft_config},
                                  peft_objective=True),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1,
                                    objective_args_for_head_config={"peft_config": peft_config},
                                    peft_objective=True)]

    args = AdaptationArguments(**{**training_args_map, **{"eval_steps": SAVE_STEPS, "save_steps": SAVE_STEPS}})

    schedule = SequentialSchedule(objectives, args)
    adapter = Adapter(lang_module, schedule, args=args)

    # first training iteration
    run_adaptation(adapter)

    # second training iteration - continue from the checkpoints persisted for each objective
    lang_module = LangModule(os.path.join(training_arguments.output_dir, "checkpoint-%s" % SAVE_STEPS))
    objectives = [BackTranslation(lang_module,
                                  back_translator=translator,
                                  texts_or_path=paths["texts"]["unsup"],
                                  batch_size=2,
                                  objective_args_for_head_config={"peft_config": peft_config},
                                  peft_objective=True),
                  Sequence2Sequence(lang_module,
                                    texts_or_path=paths["texts"]["translation"],
                                    labels_or_path=paths["labels"]["translation"],
                                    batch_size=1,
                                    objective_args_for_head_config={"peft_config": peft_config},
                                    peft_objective=True)]

    schedule = ParallelSchedule(objectives, training_arguments)
    adapter = Adapter(lang_module, schedule, args=training_arguments)
    run_adaptation(adapter)
