import json
import logging
from typing import List, Dict, Union, Any, Optional, Type, Tuple

from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig, CONFIG_MAPPING, AutoModel, \
    AutoTokenizer

SUPPORTED_INIT_TOKENIZERS = ["sentencepiece"]
BASE_CHECKPOINT_NAME = "init_model"

# Shared attributes from `PreTrainedTokenizer.SPECIAL_TOKENS_ATTRIBUTES`
_DEFAULT_SPECIAL_TOKEN_KEYS = ("bos", "eos", "unk", "sep", "pad", "cls", "mask")
DEFAULT_SPECIAL_VOCAB_NAMES = {t + "_token": "<%s>" % t for t in _DEFAULT_SPECIAL_TOKEN_KEYS}
DEFAULT_SPECIAL_VOCAB_IDS = {t + "_token_id": idx for idx, t in enumerate(_DEFAULT_SPECIAL_TOKEN_KEYS)}
DEFAULT_SPECIAL_VOCAB = {"<%s>" % t: idx for idx, t in enumerate(_DEFAULT_SPECIAL_TOKEN_KEYS)}


logger = logging.getLogger()


def _concat_save_vocabs(base_vocab: Dict[str, int], joint_vocab_paths: List[str], out_vocab_file: str) -> None:
    all_vocab = base_vocab.copy()

    for vocab_path in joint_vocab_paths:
        with open(vocab_path) as src_v:
            for token_score in src_v.readlines():
                token, score = token_score.split("\t")
                all_vocab[token] = len(all_vocab)

    with open(out_vocab_file, "w") as output_vocab_f:
        json.dump(all_vocab, output_vocab_f)


def tokenizer_config_from_data(tokenizer_type: str,
                               tokenizer_cls: Type[PreTrainedTokenizer],
                               model_type: str,
                               texts: List[str],
                               vocab_size: int,
                               save_tokenizer_dir: str,
                               tokenizer_kwargs: Dict[str, Any] = (),
                               config_kwargs: Dict[str, Any] = (),
                               sentencepiece_kwargs: Dict[str, Any] = ()) -> Tuple[PreTrainedTokenizer,
                                                                                   PretrainedConfig]:
    if tokenizer_type == "sentencepiece":
        import sentencepiece as spm

        spm.SentencePieceTrainer.Train(sentence_iterator=iter(texts),
                                       model_prefix='src_tokenizer',
                                       vocab_size=vocab_size,
                                       **dict(sentencepiece_kwargs))
        if tokenizer_cls is None:
            from transformers import XLMRobertaTokenizer
            tokenizer_cls = XLMRobertaTokenizer
            logger.warning("No `tokenizer_cls` given. Will default to %s." % tokenizer_cls)

        # We pick XLMRobertaTokenizer implementation as a generic sentencepiece-based tokenizer,
        # but note that this might not correspond to implementation of some models from their authors
        spm.SentencePieceTrainer.Train(sentence_iterator=iter(texts),
                                       model_prefix='tokenizer',
                                       vocab_size=vocab_size,
                                       **dict(sentencepiece_kwargs))
        try:
            new_tokenizer = tokenizer_cls("tokenizer.model",
                                          **DEFAULT_SPECIAL_VOCAB_NAMES,
                                          **DEFAULT_SPECIAL_VOCAB_IDS,
                                          **dict(tokenizer_kwargs))
        except Exception:
            raise ValueError("Given `tokenizer_cls=%s` does not accept sentencepiece model as first argument."
                             "Is this tokenizer supporting sentencepiece?" % tokenizer_cls)
    else:
        raise ValueError("Unknown tokenizer_type: %s" % tokenizer_type)
    # it seems that special vocab in tokenizer objects is not fully persisted between save and reload,
    # but we don't really mind about specific ids of special vocab when training from scratch,
    # we just need to reassure consistence

    new_tokenizer.save_pretrained(save_tokenizer_dir)
    # this is how lang_module resolves tokenizer, so we make sure that persisted config is consistent with it
    new_tokenizer = AutoTokenizer.from_pretrained(save_tokenizer_dir)

    tokenizer_special_vocab_keys = new_tokenizer.SPECIAL_TOKENS_ATTRIBUTES

    special_vocab_tokens = {k: getattr(new_tokenizer, k) for k in tokenizer_special_vocab_keys if
                            getattr(new_tokenizer, k) is not None}
    corresponding_ids_attrs = [k + "_id" if "tokens" not in k else k + "_ids" for k in special_vocab_tokens.keys()]
    special_vocab_ids = {k: getattr(new_tokenizer, k) for k in corresponding_ids_attrs}

    # this is the best we can do, but if tokenizer has no vocab attribute (No-FastTokenizers) and vocab_size is wrong,
    # this would cause IndexError in forward pass. Then tokenizer.vocab_size needs to be corrected (no known cases).
    resolved_vocab_size = len(new_tokenizer.vocab) if hasattr(new_tokenizer, "vocab") else new_tokenizer.vocab_size

    init_kwargs = {"vocab_size": resolved_vocab_size,
                   "decoder_start_token_id": new_tokenizer.sep_token_id,
                   **special_vocab_tokens, **special_vocab_ids}

    all_init_kwargs = {**init_kwargs, **dict(config_kwargs)}
    new_config = CONFIG_MAPPING[model_type](**all_init_kwargs)

    return new_tokenizer, new_config


def model_from_config(config: PretrainedConfig, model_kwargs: Dict[str, Any] = ()) -> PreTrainedModel:
    model = AutoModel.from_config(config, **dict(model_kwargs))
    model.resize_token_embeddings(config.vocab_size)
    return model


def texts_or_path_to_list(texts_or_path: Union[str, List[str]]) -> List[str]:
    if isinstance(texts_or_path, str):
        with open(texts_or_path) as src_f:
            return [l.strip() for l in src_f.readlines()]
    else:
        return texts_or_path