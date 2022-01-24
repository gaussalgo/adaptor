from adaptor.evaluators.generative import BLEU, GenerativeEvaluator, ROUGE, BERTScore, PRISM, METEOR
from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.seq2seq import Sequence2Sequence
from utils import paths, test_base_models


def assert_evaluator_logs(lang_module: LangModule, objective: Objective, split: str) -> None:
    # dataset iteration test
    dataset_sample = next(iter(objective.get_dataset(split, objective_i=0, device="cpu")))

    # providing labels makes HF lang_module to compute its own loss, which is in DA redundantly done by Objective
    outputs = lang_module(**dataset_sample)

    # loss computation test, possible label smoothing is performed by Adapter
    loss = objective.compute_loss(outputs, dataset_sample["labels"], split)

    # make sure loss is actually computed
    loss.item()

    log = objective.per_objective_log(split)

    assert all(str(objective) in k for k in log.keys())


gen_lang_module = LangModule(test_base_models["translation"])


def assert_gen_evaluator_logs(evaluator: GenerativeEvaluator, split: str) -> None:
    global gen_lang_module

    gen_objective = Sequence2Sequence(gen_lang_module,
                                      texts_or_path=paths["texts"]["target_domain"]["translation"],
                                      labels_or_path=paths["labels"]["target_domain"]["translation"],
                                      batch_size=1,
                                      source_lang_id="en",
                                      target_lang_id="cs",
                                      train_evaluators=[evaluator],
                                      val_evaluators=[evaluator])

    assert_evaluator_logs(gen_lang_module, gen_objective, split)


def test_bleu():
    assert_gen_evaluator_logs(BLEU(use_generate=True, decides_convergence=True), "train")


def test_rouge():
    assert_gen_evaluator_logs(ROUGE(use_generate=False, decides_convergence=True), "train")


def test_bertscore():
    assert_gen_evaluator_logs(BERTScore(use_generate=False, decides_convergence=True), "train")


def test_prism():
    assert_gen_evaluator_logs(PRISM(use_cuda=False, language="en", decides_convergence=True), "train")


def test_meteor():
    assert_gen_evaluator_logs(METEOR(decides_convergence=True), "train")
