from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.objectives.classification import TokenClassification
from adaptor.objectives.denoising import DenoisingObjective
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import SequentialSchedule, Schedule, ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from utils import test_base_models

unsup_target_domain_texts = "mock_data/domain_unsup.txt"
sup_target_domain_texts = "mock_data/ner_texts_sup.txt"
sup_target_domain_labels = "mock_data/ner_texts_sup_labels.txt"

sup_translation_texts_src = "mock_data/seq2seq_sources.txt"
sup_translation_texts_tgt = "mock_data/seq2seq_targets.txt"

args = AdaptationArguments(output_dir="adaptation_output_dir",
                           stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                           do_train=True,
                           do_eval=True,
                           gradient_accumulation_steps=2,
                           log_level="critical",
                           logging_steps=1,
                           eval_steps=1,
                           num_train_epochs=10)


def assert_schedule(lang_module: LangModule, schedule: Schedule):

    for batch in iter(schedule.iterable_dataset("train")):

        logit_outputs = lang_module(**batch)

        loss_combined = schedule.compute_loss(logit_outputs, batch["labels"])
        loss_combined.backward()

    # every objective has some key in its logs
    train_logs = schedule.objectives_log("train")
    assert all(any(str(obj) for log_key, _ in train_logs.items()) for obj in schedule.objectives["train"].keys())

    for batch in iter(schedule.iterable_dataset("eval")):

        logit_outputs = lang_module(**batch)

        loss_combined = schedule.compute_loss(logit_outputs, batch["labels"])
        loss_combined.backward()

    eval_logs = schedule.objectives_log("eval")
    assert all(any(str(obj) for log_key, _ in eval_logs.items()) for obj in schedule.objectives["eval"].keys())

    assert True


def ner_da_schedule(schedule_type):
    lang_module = LangModule(test_base_models["token_classification"])

    lm_adaptation = MaskedLanguageModeling(lang_module,
                                           texts_or_path=unsup_target_domain_texts,
                                           val_texts_or_path=unsup_target_domain_texts,
                                           batch_size=1)
    token_classification = TokenClassification(lang_module,
                                               texts_or_path=sup_target_domain_texts,
                                               labels_or_path=sup_target_domain_labels,
                                               val_texts_or_path=sup_target_domain_texts,
                                               val_labels_or_path=sup_target_domain_labels,
                                               batch_size=1)

    assert_schedule(lang_module, schedule_type(objectives=[lm_adaptation, token_classification], args=args))


def test_ner_da_schedule_sequential():
    ner_da_schedule(SequentialSchedule)


def test_ner_da_schedule_strided():
    ner_da_schedule(ParallelSchedule)


def test_mt_da_schedule():
    lang_module = LangModule(test_base_models["translation"])
    denoising_adaptation = DenoisingObjective(lang_module,
                                              texts_or_path=unsup_target_domain_texts,
                                              val_texts_or_path=unsup_target_domain_texts,
                                              batch_size=1)
    clm_finetuning = Sequence2Sequence(lang_module,
                                       texts_or_path=sup_translation_texts_src,
                                       labels_or_path=sup_translation_texts_tgt,
                                       val_texts_or_path=sup_translation_texts_src,
                                       val_labels_or_path=sup_translation_texts_tgt,
                                       source_lang_id="en",
                                       target_lang_id="cs", batch_size=1)

    assert_schedule(lang_module, SequentialSchedule(objectives=[denoising_adaptation, clm_finetuning], args=args))
