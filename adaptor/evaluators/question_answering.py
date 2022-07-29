import abc
import itertools

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head, AdaptationDataset
from typing import List, Sequence
import torch
from transformers import PreTrainedTokenizer


class ExtractiveQAEvaluator(EvaluatorBase, abc.ABC):
    """
    Base evaluator for extractive QA Evaluations.
    Providing the prediction routine and compatible head.
    """

    compatible_heads: List[Head] = [Head.QA]

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> float:
        """
        Extracts resulting answers and compares their BLEU score to the expected answers as a reference.
        Refer to the superclass documentation.
        """
        expected_str = []
        actual_str = []

        for batch in dataset:
            with torch.no_grad():
                model_outputs = model(**{k: v for k, v in batch.items() if k not in ["oid", "labels",
                                                                                     "start_position", "end_position"]})
                actual_start_pos = model_outputs.start_logits.argmax(-1)
                actual_end_pos = model_outputs.end_logits.argmax(-1)

            expected_str.extend(tokenizer.batch_decode(batch["labels"]))

            for i, (act_start, act_end) in enumerate(zip(actual_start_pos, actual_end_pos)):
                actual_str.append(tokenizer.decode(batch["input_ids"][i, act_start: act_end]))

        # make sure that reference is not empty:
        expected_not_empty = [" " if string == "" else string for string in expected_str]

        assert len(expected_not_empty) == len(actual_str), \
            "A number of entries does not match. Expected: %s, actual: %s" % (len(expected_not_empty), len(actual_str))

        return self.evaluate_str(list(expected_not_empty), actual_str)

    @abc.abstractmethod
    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        pass


class ExactMatch(ExtractiveQAEvaluator):
    """
    Exact Match metric for question answering.
    Computes accuracy of retrieved answers compared to the reference.
    """

    smaller_is_better: bool = False

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        matches = sum(e.strip() == a.strip() for e, a in zip(expected_list, actual_list))
        all = len(expected_list)

        return matches / all


class F1ScoreForQA(ExtractiveQAEvaluator):
    """
    Token-level F1-Score for question answering evaluation.
    Computes mean f-score over the predicted outputs by tokens segmented by whitespaces.
    """

    smaller_is_better: bool = False

    @staticmethod
    def _per_sample_f1(expected_answers: List[str], actual_answer: str) -> float:
        expected_answers_set = set(itertools.chain(*[a.split() for a in expected_answers]))
        actual_answer_set = actual_answer.split()

        true_positives = sum(a_word in expected_answers_set for a_word in actual_answer_set)
        false_positives = sum(a_word not in expected_answers_set for a_word in actual_answer_set)
        false_negatives = sum(e_word not in actual_answer_set for e_word in expected_answers_set)

        return true_positives / (true_positives + 0.5 * (false_positives + false_negatives))

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        f_scores = [self._per_sample_f1([e], a) for e, a in zip(expected_list, actual_list)]

        mean_f_score = sum(f_scores) / len(f_scores)
        return mean_f_score


class BLEUForQA(ExtractiveQAEvaluator):
    """
    BLEU evaluator for question answering.
    Computes standard corpus-level BLEU score between the retrieved and expected answers
    """

    smaller_is_better: bool = False

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        from sacrebleu import corpus_bleu
        return corpus_bleu(actual_list, [[e] for e in expected_list]).score
