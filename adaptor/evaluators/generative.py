import abc
from functools import lru_cache

from bert_score import BERTScorer
import itertools
from typing import List, Sequence, Optional, Dict, Iterator, Union

import torch
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding

from .evaluator_base import EvaluatorBase
from ..utils import Head
import os
from .prism import Prism
import subprocess
from pathlib import Path
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')


class GenerativeEvaluator(EvaluatorBase, abc.ABC):
    compatible_head: Head = Head.SEQ2SEQ

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

    def _autoregressive_predict(self, inputs: List[Dict[str, torch.LongTensor]],
                                model: torch.nn.Module) -> Iterator[torch.LongTensor]:
        assert hasattr(model, "generate"), "If Evaluator(use_generate=True), " \
                                           "evaluated model must have its generate() method."

        if self.progress_bar:
            inputs = tqdm(inputs, desc="%s: Evaluating with generate()" % self)

        for inputs_batch in inputs:
            yield self._autoregressive_predict_one(inputs_batch["input_ids"], inputs_batch["attention_mask"], model)

    @staticmethod
    def _argmax_predict(logit_outputs: List[torch.FloatTensor]) -> Iterator[torch.LongTensor]:
        for outputs_batch in logit_outputs:
            yield torch.argmax(outputs_batch, -1)

    def __call__(self,
                 inputs: Optional[List[Union[Dict[str, torch.LongTensor], BatchEncoding]]] = None,
                 model: Optional[torch.nn.Module] = None,
                 logit_outputs: Optional[List[torch.FloatTensor]] = None,
                 labels: Optional[List[torch.LongTensor]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None) -> float:

        if labels is None or tokenizer is None:
            raise ValueError("Evaluator %s always needs 'labels' and 'tokenizer' arguments specified.")

        for label_t in labels:
            label_t[label_t < 0] = 0

        expected_str = itertools.chain(*(tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
                                         for batch_labels in labels))
        with torch.no_grad():
            if self.use_generate:
                output_tokens_gen = itertools.chain(*self._autoregressive_predict(inputs, model))
            else:
                output_tokens_gen = itertools.chain(*self._argmax_predict(logit_outputs))

            actual_str = tokenizer.batch_decode(output_tokens_gen, skip_special_tokens=True)
            if self.additional_sep_char is not None:
                expected_str = [" ".join(expected_one.split(self.additional_sep_char)) for expected_one in expected_str]
                actual_str = [" ".join(actual_one.split(self.additional_sep_char)) for actual_one in actual_str]

        return self.evaluate_str(list(expected_str), actual_str)

    @abc.abstractmethod
    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__) if not self.use_generate else str(self.__class__.__name__) + "-gen"


class BLEU(GenerativeEvaluator):
    smaller_is_better: bool = False

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        return corpus_bleu(actual_list, [list(expected_list)]).score


class ROUGE(GenerativeEvaluator):
    smaller_is_better: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        all_scores = [self.scorer.score(expected, actual)['rougeL'].recall
                      for expected, actual in zip(expected_list, actual_list)]
        return sum(all_scores) / len(expected_list)


class BERTScore(GenerativeEvaluator):
    smaller_is_better: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scorer = BERTScorer(lang="any", model_type="bert-base-multilingual-cased")

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str]) -> float:
        b_prec, b_rec, b_f_scores = self.scorer.score(expected_list, actual_list)
        return b_f_scores.mean().cpu().item()


class PRISM(GenerativeEvaluator):

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

    def evaluate_str(self, expected_list: Sequence[str], actual_list: Sequence[str],
                     parameters: List[float] = (0.9, 3, 0.1)) -> float:
        expected_list_tokenized = [item.split() for item in expected_list]
        actual_list_tokenized = [item.split() for item in actual_list]
        all_scores = [
            meteor_score([list(expected)], actual, alpha=parameters[0], beta=parameters[1], gamma=parameters[2])
            for expected, actual in zip(expected_list_tokenized, actual_list_tokenized)]
        return float(sum(all_scores) / len(all_scores))
