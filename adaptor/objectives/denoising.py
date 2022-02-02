import abc
import collections
import itertools
import random
from typing import List, Tuple, Optional, Iterator

from transformers import BatchEncoding

from .seq2seq import Sequence2SequenceMixin
from ..objectives.objective_base import UnsupervisedObjective


class NoisingStrategy(abc.ABC):
    """
    Implementations of noising strategies inspired by https://arxiv.org/abs/1910.13461
    """
    special_char_tokens: Tuple[str, ...] = (".", ",", "?", "!", "-", "/")
    sentence_sep_chars: Tuple[str, ...] = (".", "?", "!")

    def __init__(self, application_ratio: float):
        self.application_ratio = application_ratio

    def _should_be_applied(self, eps: float = 1e-5):
        return random.randint(0, int(1 / eps)) < self.application_ratio / eps

    def _split_on_tokens(self, text: str) -> List[str]:
        # separate all special chars with space, so that they are parsed as separate tokens
        sep_text = text
        for special_char in self.special_char_tokens:
            sep_text = sep_text.replace(special_char, " %s " % special_char)

        return sep_text.split()

    def _correct_special_chars_spacing(self, text: str) -> str:
        # reconstruct spaces around special characters natively, as in the input
        out_text = text
        for special_char in self.special_char_tokens:
            out_text = out_text.replace(" " + special_char, special_char)
        return out_text

    def _call_per_sentence(self, text: str) -> str:
        """
        Splits the input text to sentences, succeeded with its separator
        """
        out_sents = [text]
        for sep in self.sentence_sep_chars:
            # split each sentence in the list and chain the result, omit the empty sentences
            out_sents = list(itertools.chain(*[[s+sep if s and sep in sent else s for s in sent.split(sep)]
                                               for sent in out_sents if sent]))

        # separators should keep their positions after the noising
        seps = [sent[-1] if sent[-1] in self.sentence_sep_chars else "" for sent in out_sents if sent]
        processed_sents = [sent[:-1] if sep else sent for sent, sep in zip(out_sents, seps) if sent]
        out_text = " ".join([self(sent, apply_per_sentence=False) + sep for sent, sep in zip(processed_sents, seps)])
        return out_text

    @abc.abstractmethod
    def __call__(self, text: str, apply_per_sentence: bool) -> str:
        pass


class Shuffle(NoisingStrategy):

    def __call__(self, text: str, apply_per_sentence: bool) -> str:
        if apply_per_sentence:
            return self._call_per_sentence(text)
        else:
            if self._should_be_applied():
                tokens = self._split_on_tokens(text)
                random.shuffle(tokens)

                out_text = self._correct_special_chars_spacing(" ".join(tokens))
            else:
                out_text = text

            return out_text


class Rotate(NoisingStrategy):

    def __call__(self, text: str, apply_per_sentence: bool) -> str:
        if apply_per_sentence:
            return self._call_per_sentence(text)
        else:
            if self._should_be_applied():
                tokens = self._split_on_tokens(text)
                middle_token_idx = random.randint(0, len(tokens))
                new_tokens = collections.OrderedDict()
                for i, token in enumerate(tokens):
                    assigned_pos = - (middle_token_idx - i)
                    new_tokens[assigned_pos] = token

                out_text = self._correct_special_chars_spacing(" ".join(new_tokens.values()))
            else:
                out_text = text
            return out_text


class Infilling(NoisingStrategy):
    pass


class Deletion(NoisingStrategy):
    pass


class Permutation(NoisingStrategy):
    pass


class Masking(NoisingStrategy):
    pass


class DenoisingObjective(Sequence2SequenceMixin, UnsupervisedObjective):

    def __init__(self,
                 *args,
                 noising_strategies: Optional[List[NoisingStrategy]] = None,
                 noising_prob: float = 0.6,
                 noising_per_sentence: bool = True,
                 **kwargs):
        """
        Noising strategy implementing token permutation, as introduced by BART in https://arxiv.org/abs/1910.13461
        :param noising_strategies: Stratiegies to noise text with. BART experiment with multiple strategies,
        but does not find all of them helpful
        :param noising_prob: A probability of noising each sentence if input.
        :param noising_per_sentence: Whether to apply noise separately for each sentence.
        """
        super().__init__(*args, **kwargs)

        if noising_strategies is None:
            self.noising_strategies = [Shuffle(application_ratio=noising_prob)]
        else:
            self.noising_strategies = noising_strategies

        self.noising_per_sentence = noising_per_sentence

    def _apply_noise(self, text: str) -> str:
        """
        Applies selected noising strategies
        :param text: text to be noised.
        :return: noised text.
        """
        out_text = text
        for noising_fn in self.noising_strategies:
            out_text = noising_fn(out_text, self.noising_per_sentence)
        return out_text

    def _get_inputs_iterator(self, split: str) -> Iterator[BatchEncoding]:
        """
        Generates labels by applying selected noising strategies on inputs.
        :param split: Data split. `train` or `eval`.
        :return: an iterator of seq2seq model encodings.
        """
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)

        input_texts_noised = (self._apply_noise(text) for text in source_texts_iter)
        collated_iter = self._get_seq2seq_collated_iterator(input_texts_noised, target_texts_iter)

        return collated_iter
