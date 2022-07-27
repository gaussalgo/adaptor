from adaptor.evaluators.generative import GenerativeEvaluator
from adaptor.evaluators.question_answering import ExtractiveQAEvaluator
from adaptor.evaluators.sequence_classification import SeqClassificationEvaluator
from adaptor.evaluators.token_classification import TokenClassificationEvaluator
from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.seq2seq import Sequence2Sequence
from utils import paths, test_base_models


def assert_evaluator_logs(lang_module: LangModule, objective: Objective, split: str) -> None:
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(**dataset_sample)

    # request objective for its loss
    loss = objective.compute_loss(outputs, dataset_sample["labels"], dataset_sample, split)
    assert loss.item()

    log = objective.per_objective_log(split)

    # assert that objective's id can be found in each key of the logs
    assert all(str(objective) in k for k in log.keys())

    for split_evaluator in objective.evaluators[split]:
        # assert that each evaluator of given split was logged and has a value of expected type
        assert any(str(split_evaluator) in k and isinstance(v, float) for k, v in log.items())


gen_lang_module = LangModule(test_base_models["translation_mono"])
gen_lang_module_multi = LangModule(test_base_models["translation_multi"]["model"])


def assert_gen_evaluator_logs(evaluator: GenerativeEvaluator, split: str) -> None:
    gen_objective = Sequence2Sequence(gen_lang_module,
                                      texts_or_path=paths["texts"]["translation"],
                                      labels_or_path=paths["labels"]["translation"],
                                      batch_size=1,
                                      train_evaluators=[evaluator],
                                      val_evaluators=[evaluator])

    assert_evaluator_logs(gen_lang_module, gen_objective, split)


def assert_gen_evaluator_logs_mbart(evaluator: GenerativeEvaluator, split: str) -> None:
    gen_objective = Sequence2Sequence(gen_lang_module_multi,
                                      texts_or_path=paths["texts"]["translation"],
                                      labels_or_path=paths["labels"]["translation"],
                                      batch_size=1,
                                      train_evaluators=[evaluator],
                                      val_evaluators=[evaluator],
                                      source_lang_id=test_base_models["translation_multi"]["test_src_lang"],
                                      target_lang_id=test_base_models["translation_multi"]["test_tgt_lang"])

    assert_evaluator_logs(gen_lang_module_multi, gen_objective, split)


def assert_ner_evaluator_logs(evaluator: TokenClassificationEvaluator, split: str) -> None:
    from adaptor.objectives.classification import TokenClassification
    lang_module = LangModule(test_base_models["token_classification"])

    gen_objective = TokenClassification(lang_module,
                                        texts_or_path=paths["texts"]["ner"],
                                        labels_or_path=paths["labels"]["ner"],
                                        batch_size=1,
                                        train_evaluators=[evaluator],
                                        val_evaluators=[evaluator])

    assert_evaluator_logs(lang_module, gen_objective, split)


def assert_classification_evaluator_logs(evaluator: SeqClassificationEvaluator, split: str) -> None:
    from adaptor.objectives.classification import SequenceClassification
    lang_module = LangModule(test_base_models["sequence_classification"])

    gen_objective = SequenceClassification(lang_module,
                                           texts_or_path=paths["texts"]["classification"],
                                           labels_or_path=paths["labels"]["classification"],
                                           batch_size=1,
                                           train_evaluators=[evaluator],
                                           val_evaluators=[evaluator])

    assert_evaluator_logs(lang_module, gen_objective, split)


def assert_qa_evaluator_logs(evaluator: ExtractiveQAEvaluator, split: str) -> None:
    from adaptor.objectives.question_answering import ExtractiveQA
    lang_module = LangModule(test_base_models["extractive_QA"])

    qa_objective = ExtractiveQA(lang_module,
                                texts_or_path=paths["texts"]["QA"],
                                text_pair_or_path=paths["text_pair"]["QA"],
                                labels_or_path=paths["labels"]["QA"],
                                batch_size=2,
                                train_evaluators=[evaluator],
                                val_evaluators=[evaluator])

    assert_evaluator_logs(lang_module, qa_objective, split)


def test_bleu():
    from adaptor.evaluators.generative import BLEU
    assert_gen_evaluator_logs(BLEU(use_generate=True, decides_convergence=True), "train")


def test_bleu_mbart():
    from adaptor.evaluators.generative import BLEU
    assert_gen_evaluator_logs_mbart(BLEU(use_generate=True, decides_convergence=True), "train")


def test_rouge():
    from adaptor.evaluators.generative import ROUGE
    assert_gen_evaluator_logs(ROUGE(use_generate=False, decides_convergence=True), "train")


def test_bertscore():
    from adaptor.evaluators.generative import BERTScore
    assert_gen_evaluator_logs(BERTScore(use_generate=False, decides_convergence=True), "train")


def test_meteor():
    from adaptor.evaluators.generative import METEOR
    assert_gen_evaluator_logs(METEOR(decides_convergence=True), "train")


def test_prism():
    """
    PRISM downloads relatively big model, we omit that by default.
    """
    # from adaptor.evaluators.generative import PRISM
    # assert_gen_evaluator_logs(PRISM(use_cuda=False, language="en", decides_convergence=True), "train")


def test_divergence():
    """
    Default JS_Divergence uses PRISM - note that this test will download PRISM model
    """
    # from adaptor.evaluators.generative import JS_Divergence
    # assert_gen_evaluator_logs(JS_Divergence(decides_convergence=True), "train")


def test_token_fscore():
    from adaptor.evaluators.token_classification import MeanFScore
    assert_ner_evaluator_logs(MeanFScore(decides_convergence=True), "train")


def test_sequence_accuracy():
    from adaptor.evaluators.sequence_classification import SequenceAccuracy
    assert_classification_evaluator_logs(SequenceAccuracy(decides_convergence=False), "train")


def test_QA_exact_match():
    from adaptor.evaluators.question_answering import ExactMatch
    assert_qa_evaluator_logs(ExactMatch(), "train")


def test_QA_fscore():
    from adaptor.evaluators.question_answering import F1ScoreForQA
    assert_qa_evaluator_logs(F1ScoreForQA(), "train")


def test_QA_BLEU():
    from adaptor.evaluators.question_answering import BLEUForQA
    assert_qa_evaluator_logs(BLEUForQA(), "train")
