import abc
from functools import lru_cache
from typing import List, Sequence, Optional, Dict, Iterator, Union

import numpy as np
import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from transformers import PreTrainedTokenizer, BatchEncoding

from .evaluator_base import EvaluatorBase
from .prism import Prism
from ..utils import Head, AdaptationDataset


class GenerativeEvaluator(EvaluatorBase, abc.ABC):

    compatible_heads: List[Head] = [Head.SEQ2SEQ]

    def __init__(self,
                 use_generate: bool = True,
                 progress_bar: Optional[bool] = True,
                 decides_convergence: Optional[bool] = False,
                 additional_sep_char: Optional[str] = None):
        super().__init__(decides_convergence)

        self.additional_sep_char = additional_sep_char
        self.use_generate = use_generate
        self.progress_bar = progress_bar

    @staticmethod
    @lru_cache(maxsize=1000)
    def _autoregressive_predict_one(input_ids: torch.LongTensor,
                                    attention_mask: torch.LongTensor,
                                    model: torch.nn.Module) -> torch.LongTensor:
        return model.generate(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()

    def _autoregressive_predict(self,
                                model: torch.nn.Module,
                                inputs_batch: Dict[str, torch.LongTensor]) -> Iterator[torch.LongTensor]:
        assert hasattr(model, "generate"), "If Evaluator(use_generate=True), " \
                                           "evaluated model must have its generate() method."

        return self._autoregressive_predict_one(inputs_batch["input_ids"], inputs_batch["attention_mask"], model)

    @staticmethod
    def _argmax_predict(model: torch.nn.Module,
                        inputs: Union[BatchEncoding, Dict[str, torch.FloatTensor]]) -> torch.Tensor:
        outputs = model(**inputs).logits
        return torch.argmax(outputs, -1)

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        """
        Refer to the superclass documentation.
        """
        expected_str = []
        actual_str = []

        for batch in dataset:
            with torch.no_grad():
                if self.use_generate:
                    output_tokens = self._autoregressive_predict(model, batch)
                else:
                    output_tokens = self._argmax_predict(model, batch)
            # replace -100 labels (excluded from the loss), otherwise encoded as unknown tokens
            batch["labels"][batch["labels"] < 0] = tokenizer.pad_token_id

            expected_str.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
            actual_str.extend(tokenizer.batch_decode(output_tokens, skip_special_tokens=True))

        if self.additional_sep_char is not None:
            expected_str = [" ".join(expected_one.split(self.additional_sep_char)) for expected_one in expected_str]
            actual_str = [" ".join(actual_one.split(self.additional_sep_char)) for actual_one in actual_str]

        return self.evaluate_str(list(expected_str), actual_str)

    @abc.abstractmethod
    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        """
        Evaluation of expected and actually-generated strings.
        This method can be used separately, in standalone in test evaluation.
        When implementing evaluation of generative language model, you can implement only this method.
        See other GenerativeEvaluators (e.g. BLEU, BERTScore) for examples.

        :param expected_list: A sequence of reference texts, that model is expected to generate.
        :param actual_list: A sequence of actually-generated texts by the model.
        """
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__) if not self.use_generate else str(self.__class__.__name__) + "-gen"


class BLEU(GenerativeEvaluator):
    smaller_is_better: bool = False

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:

        return corpus_bleu(actual_list, [list(expected_list)]).score


class ROUGE(GenerativeEvaluator):
    """
    Computes mean ROUGE-L score.
    """

    smaller_is_better: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        all_scores = [self.scorer.score(expected, actual)['rougeL'].recall
                      for expected, actual in zip(expected_list, actual_list)]
        return sum(all_scores) / len(expected_list)


class BERTScore(GenerativeEvaluator):
    """
    Compute BERTScore for a set of translations.
    Refer to https://github.com/Tiiiger/bert_score
    """

    smaller_is_better: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scorer = BERTScorer(lang="any", model_type="bert-base-multilingual-cased")

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        b_prec, b_rec, b_f_scores = self.scorer.score(expected_list, actual_list)
        return b_f_scores.mean().cpu().item()


class PRISM(GenerativeEvaluator):
    """
    Computes PRISM score for a set of translations or paraphrases.
    Refer to https://github.com/thompsonb/prism
    """

    def __init__(self,
                 language: str,
                 use_cuda: Optional[bool] = None,
                 probability: Optional[bool] = False,
                 model_dir: str = "prism/model_dir",
                 **kwargs):
        # language must be set, see prism.py: MODELS['langs'] for a list of supported langs
        super().__init__(**kwargs)
        self.probability = probability
        self.scorer = Prism(model_dir, lang=language, use_cuda=use_cuda)

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        # returns model score (-inf, 0), if par probability = True -> returns probability.
        # cand: candidate is the system output
        # ref: reference is the human reference
        if self.probability:
            return float(np.exp(self.scorer.score(cand=actual_list, ref=expected_list)))
        else:
            return float(self.scorer.score(cand=actual_list, ref=expected_list))


class METEOR(GenerativeEvaluator):
    """
    Computes METEOR score over a set of translations.
    """

    def __init__(self, *args, **kwargs):
        import nltk

        nltk.download('wordnet')
        nltk.download('omw-1.4')

        super().__init__(*args, **kwargs)

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str],
                     alpha: float = 0.9,  beta: float = 3, gamma: float = 0.1) -> float:
        from nltk.translate.meteor_score import meteor_score

        expected_list_tokenized = [item.split() for item in expected_list]
        actual_list_tokenized = [item.split() for item in actual_list]
        all_scores = [meteor_score([list(expected)], actual, alpha=alpha, beta=beta, gamma=gamma)
                      for expected, actual in zip(expected_list_tokenized, actual_list_tokenized)]

        return float(sum(all_scores) / len(all_scores))
