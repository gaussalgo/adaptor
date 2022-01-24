import logging
from typing import List, Iterator, Iterable

import torch
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM

from .seq2seq import Sequence2SequenceMixin
from ..objectives.objective_base import UnsupervisedObjective

logger = logging.getLogger()


class BackTranslator(torch.nn.Module):
    """
    Back-translation interface that can be used out-of-box in BackTranslation Objective.
    """

    def __init__(self, model_name_or_path: str, device: str = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)

    def translate(self, texts: List[str]) -> List[str]:
        """
        Translates input texts using the given translation model.
        :param texts: texts to be translated.
        :return: translated texts.
        """

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        outputs = self.translator.generate(**inputs.to(self.device))
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return translations


class BackTranslation(Sequence2SequenceMixin, UnsupervisedObjective):
    """
    BackTranslation Objective can be used for unsupervised adaptation of translator to *Target* domain.
    """

    def __init__(self,
                 *args,
                 back_translator: BackTranslator,
                 **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("%s expects target-language texts as input_texts_or_path. This is not further checked. " % self)

        self.translator = back_translator
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model)

    def _construct_batch(self, target_texts_batch: List[str]):
        translated_source_texts = self.translator.translate(target_texts_batch)
        features_batch = []
        for src_text, tgt_text in zip(translated_source_texts, target_texts_batch):
            sample_features = self.tokenizer(src_text, truncation=True, padding="longest")
            with self.tokenizer.as_target_tokenizer():
                sample_targets = self.tokenizer(tgt_text, truncation=True, padding="longest")
            features_batch.append({"input_ids": sample_features.input_ids,
                                   "attention_mask": sample_features.attention_mask,
                                   "labels": sample_targets.input_ids})
        return self.collator(features_batch)

    def _get_seq2seq_collated_iterator(self, source_texts: Iterable[str], target_texts: Iterable[str]) -> Iterator:
        """
        Constructs collated inputs from the target-side monolingual texts, using self.translator.
        :param source_texts: Unsupervised input texts -> Must match target texts.
        :param target_texts: Target texts of the trained translator.
        :return: collated batches for self.compatible_model_head.
        """
        targets_batch = []

        for source_text, target_text in zip(source_texts, target_texts):
            if source_text != target_text:
                logger.error("%s: source and target texts must match. Source: \n%s\nTarget:\n%s",
                             self, source_text, target_text)
            targets_batch.append(target_text)

            if len(targets_batch) == self.batch_size:
                # we want to use the batches of this Objective's size for back-translation as well
                yield self._construct_batch(targets_batch)
                targets_batch = []

        if targets_batch:
            # yield last nonempty residual batch
            yield self._construct_batch(targets_batch)
