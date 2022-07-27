from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head, AdaptationDataset
from typing import List, Sequence
from sacrebleu import corpus_bleu
import torch
from transformers import PreTrainedTokenizer


class BLEUForQA(EvaluatorBase):
    """
    BLEU evaluator for question answering.
    Computes standard corpus-level BLEU score between the retrieved and expected answers
    """

    compatible_heads: List[Head] = [Head.QA]
    smaller_is_better: bool = False

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> float:
        """
        Extracts resulting answers and copares their BLEU score to the expected answers as a reference.
        Refer to the superclass documentation.
        """
        expected_str = []
        actual_str = []

        for batch in dataset:
            with torch.no_grad():
                expected_tokens_start = batch["start_position"].flatten().tolist()[0]
                expected_tokens_end = batch["end_position"].flatten().tolist()[0]

                start_position = model(batch["input_ids"]).start_logits.argmax().flatten().tolist()[0]
                end_position = model(batch["input_ids"]).end_logits.argmax().flatten().tolist()[0]

            expected_str.append(tokenizer.decode(batch["input_ids"].flatten().tolist()[expected_tokens_start:
                                                                                       expected_tokens_end]))

            actual_str.append(tokenizer.decode(batch["input_ids"].flatten().tolist()[start_position:end_position]))

            # check that reference is not empty:
            expected_not_empty = [" " if string == "" else string for string in expected_str]

        return self.evaluate_str(list(expected_not_empty), actual_str)

    @staticmethod
    def evaluate_str(expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        return corpus_bleu(actual_list, [[e] for e in expected_list]).score
