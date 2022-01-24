import logging
import os
from typing import Tuple, List, Optional

from examples.data_utils import DataSource

# these can be easily exchanged with links to other domains and languages in MOSES format
# see a list for your language pair in https://opus.nlpl.eu/

OPUS_RESOURCES_URLS = {
    "WikiMatrix": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/%s-%s.txt.zip",
    "wikimedia": "https://object.pouta.csc.fi/OPUS-wikimedia/v20210402/moses/%s-%s.txt.zip",
    "OpenSubtitles": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/%s-%s.txt.zip",
    "Bible": "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/%s-%s.txt.zip"
}

# asserting the set, we make sure that no duplicates are loaded among the resources
loaded_srcs = set()


logger = logging.getLogger()


class OPUSDataset(DataSource):

    def __init__(self, domain: str, split: str, src_lang: str, tgt_lang: str,
                 data_dir: str, firstn: Optional[int] = None, skip_seen_duplicates: bool = True):

        assert domain in OPUS_RESOURCES_URLS, "I do not recognize %s OPUS domain, sorry. Pick one of: %s." % \
                                              (domain, OPUS_RESOURCES_URLS.keys())
        self.domain_label = domain
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # in OPUS, lang pair identifier is sorted alphabetically
        self.ordered_lang_pair = tuple(sorted([self.src_lang, self.tgt_lang]))

        self.split = split
        self.data_dir = data_dir
        self.skip_seen_duplicates = skip_seen_duplicates

        source_texts, target_texts = self._load_translation_pairs(firstn)

        self.source = source_texts
        self.target = target_texts

    @staticmethod
    def _preproc(text: str):
        return text.strip()

    def _deduplicate(self, src_texts: List[str], tgt_texts: List[str]) -> Tuple[List[str], List[str]]:
        out_src_texts = []
        out_tgt_texts = []
        duplicates = 0
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            if src_text not in loaded_srcs:
                out_src_texts.append(src_text)
                out_tgt_texts.append(tgt_text)
                loaded_srcs.add(src_text)
            duplicates += 1
        if duplicates:
            logger.warning("Dropping %s duplicates from %s %s split." % (duplicates, self.domain_label, self.split))
        return out_src_texts, out_tgt_texts

    def _load_translation_pairs(self, firstn: int) -> Tuple[List[str], List[str]]:
        src_file, tgt_file = self._maybe_download_unzip()
        with open(src_file, "r") as f:
            src_lines = [self._preproc(l) for l in f.readlines()]
        with open(tgt_file, "r") as f:
            tgt_lines = [self._preproc(l) for l in f.readlines()]

        src_lines = self._get_in_split(src_lines)
        tgt_lines = self._get_in_split(tgt_lines)

        if self.skip_seen_duplicates:
            src_lines, tgt_lines = self._deduplicate(src_lines, tgt_lines)

        if firstn is not None:
            src_lines = src_lines[:firstn]
            tgt_lines = tgt_lines[:firstn]

        return src_lines, tgt_lines

    def _maybe_download_unzip(self) -> Tuple[str, str]:
        src_suffix = "%s-%s.%s" % (self.ordered_lang_pair[0], self.ordered_lang_pair[1], self.src_lang)
        tgt_suffix = "%s-%s.%s" % (self.ordered_lang_pair[0], self.ordered_lang_pair[1], self.tgt_lang)
        from zipfile import ZipFile
        from urllib.request import urlopen
        from io import BytesIO

        out_srcs = [os.path.join(self.data_dir, fpath) for fpath in os.listdir(self.data_dir)
                    if self.domain_label.lower() in fpath.lower() and src_suffix in fpath.lower()]
        out_tgts = [os.path.join(self.data_dir, fpath) for fpath in os.listdir(self.data_dir)
                    if self.domain_label.lower() in fpath.lower() and tgt_suffix in fpath]

        # resources are not yet downloaded
        if not out_srcs or not out_tgts:
            print("Downloading %s" % self.domain_label)
            url = OPUS_RESOURCES_URLS[self.domain_label] % self.ordered_lang_pair
            resp = urlopen(url)
            with ZipFile(BytesIO(resp.read())) as zipfile:
                files_in_zip = zipfile.NameToInfo.keys()
                src_zip_path = [zipfile for zipfile in files_in_zip if src_suffix in zipfile][0]
                tgt_zip_path = [zipfile for zipfile in files_in_zip if tgt_suffix in zipfile][0]
                for cached_f in [src_zip_path, tgt_zip_path]:
                    zipfile.extract(cached_f, path=self.data_dir)
                    assert os.path.exists(os.path.join(self.data_dir, cached_f))

            out_src_f = os.path.join(self.data_dir, src_zip_path)
            out_tgt_f = os.path.join(self.data_dir, tgt_zip_path)
        else:
            out_src_f = out_srcs[0]
            out_tgt_f = out_tgts[0]

        return out_src_f, out_tgt_f

