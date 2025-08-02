from __future__ import annotations

"""
Tokenizer library for Myanmar (Burmese) NLP.

Usage example
-------------
>>> from myTokenize import WordTokenizer
>>> tk = WordTokenizer(engine="LSTM")
>>> tk.tokenize("မြန်မာနိုင်ငံ")
['မြန်မာ', 'နိုင်ငံ']
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pycrfsuite
from cached_path import cached_path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TF INFO logs
tf.get_logger().setLevel("ERROR")

# ---------- Local helpers / constants -------------------------------------------------
HF_TOKENIZER_REPO = "hf://LULab/myNLP-Tokenization/Models"
HF_myWord_DICT_REPO = "hf://LULab/myNLP-Tokenization/Models/dict_ver1"

# Local import of syllable/phrase Viterbi utilities
from .myWord import phrase_segment as phr
from .myWord import word_segment as wseg

# -------------------------------------------------------------------------------------
class SyllableTokenizer:
    """
    Syllable Tokenizer using Sylbreak for Myanmar language.
    Author: Ye Kyaw Thu
    Link: https://github.com/ye-kyaw-thu/sylbreak
    :Example:
    from myNLP.tokenize import Tokenizer
    tokenizer = Tokenizer.SyllableTokenizer()
    syllables = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
    print(syllables)
    # ['မြန်', 'မာ', 'နိုင်', 'ငံ', '။']
    """

    def __init__(self) -> None:
        self._my_consonant = r"က-အ"
        self._en_char = r"a-zA-Z0-9"
        self._other_char = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\[-`{-~\s"
        self._ss_symbol = "္"
        self._a_that = "်"
        pattern = (
            rf"((?<!.{self._ss_symbol})["  # negative‑lookbehind for stacked conso.
            rf"{self._my_consonant}"          # any Burmese consonant
            rf"](?![{self._a_that}{self._ss_symbol}])"  # not followed by virama
            rf"|[{self._en_char}{self._other_char}])"
        )
        self._break_pattern: re.Pattern[str] = re.compile(pattern)

    # ------------------------------------------------------------------
    def tokenize(self, raw_text: str) -> List[str]:
        """Return a list of syllables for *raw_text*."""
        lined_text = re.sub(self._break_pattern, r" \1", raw_text)
        return lined_text.split()


# -------------------------------------------------------------------------------------
class WordTokenizer(SyllableTokenizer):
    """
    Word Tokenization class: myWord, CRFs and, BiLSTM available.
    :Example:
    from myNLP.tokenize import Tokenizer
    tokenizer = Tokenizer.WordTokenizer()
    words = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
    print(words)
    # ['မြန်မာ', 'နိုင်ငံ', '။']
    """

    def __init__(self, engine: str = "CRF") -> None:
        super().__init__()
        self.engine = engine

        if engine == "CRF":
            self._init_crf()
        elif engine == "LSTM":
            self._init_lstm()
        elif engine == "myWord":
            self._init_myWord()
        else:
            raise ValueError(f"No {engine}. engine must be one of: CRF | LSTM | myWord")

    # ------------------------- CRF -----------------------------------
    def _init_crf(self) -> None:
        model_path = Path(__file__).with_suffix("").parent / "CRFTokenizer" / "wordseg_c2_crf.crfsuite"
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(model_path))

    # ------------------------- LSTM ----------------------------------
    def _init_lstm(self) -> None:
        # Fetch remote artifacts once and reuse locally afterwards.
        self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/myWseg-s4-bilstm-v2.h5")
        self.tag_path = cached_path(f"{HF_TOKENIZER_REPO}/tag_map_s4.json")
        self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/vocab_s4.json")

        # Load params into memory.
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        with open(self.tag_path, "r", encoding="utf-8") as f:
            self.tag_map: dict[str, int] = json.load(f)
        self.model = load_model(self.model_path)

    # ------------------------- myWord --------------------------------
    def _init_myWord(self) -> None:
        # Pull binary dictionaries via HF – only first call hits the network.
        self.unigram_word_bin = cached_path(f"{HF_myWord_DICT_REPO}/unigram-word.bin")
        self.bigram_word_bin = cached_path(f"{HF_myWord_DICT_REPO}/bigram-word.bin")

    # ------------------------- Feature engineering -------------------
    @staticmethod
    def _word2features(sent: str, i: int) -> dict[str, object]:
        word = sent[i]
        feats: dict[str, object] = {"number": word.isdigit()}
        if i > 0:
            prev = sent[i - 1]
            feats.update(
                {
                    "prev_word.lower()": prev.lower(),
                    "prev_number": prev.isdigit(),
                    "bigram": prev.lower() + "_" + word.lower(),
                }
            )
        else:
            feats["BOS"] = True
        if i < len(sent) - 1:
            nxt = sent[i + 1]
            feats.update({"next_word.lower()": nxt.lower(), "next_number": nxt.isdigit()})
        else:
            feats["EOS"] = True
        if i > 1:
            feats["trigram_1"] = sent[i - 2].lower() + "_" + sent[i - 1].lower() + "_" + word.lower()
        if i < len(sent) - 2:
            feats["trigram_2"] = word.lower() + "_" + sent[i + 1].lower() + "_" + sent[i + 2].lower()
        return feats

    # Helper to vectorise a sentence for CRF
    def _crf_features(self, sent: str):
        return [self._word2features(sent, i) for i in range(len(sent))]

    # ------------------------------------------------------------------
    def tokenize(self, raw_text: str) -> List[str]:
        engine = self.engine
        if engine == "CRF":
            sent = raw_text.replace(" ", "")
            preds = self.tagger.tag(self._crf_features(sent))
            merged = "".join(
                char + ("_" if tag == "|" else "") for char, tag in zip(sent, preds)
            )
            return merged.split("_")[:-1]

        if engine == "LSTM":
            syllables = super().tokenize(raw_text)
            idx_seq = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in syllables]
            logits = self.model.predict(pad_sequences([idx_seq], padding="post"), verbose=0)
            tag_ids = logits.argmax(axis=-1)[0]
            inv_tag = {v: k for k, v in self.tag_map.items()}
            segmented: list[str] = []
            for syl, tag_id in zip(syllables, tag_ids):
                tag = inv_tag[tag_id]
                segmented.append(syl + (" " if tag == "|" else ""))
            return "".join(segmented).split()

        if engine == "myWord":
            wseg.P_unigram = wseg.ProbDist(self.unigram_word_bin, True)
            wseg.P_bigram = wseg.ProbDist(self.bigram_word_bin, False)
            _, tokens = wseg.viterbi(raw_text.replace(" ", "").strip())
            return tokens

        raise RuntimeError("Unknown engine: " + engine)


# -------------------------------------------------------------------------------------
class PhraseTokenizer(WordTokenizer):
    """
    NPMI based Unsupervised Phrase Segmentation
    Author: Ye Kyaw Thu
    Link: https://github.com/ye-kyaw-thu/myWord/blob/main/phrase_segment.py
    Experiment Note by Assoc. Prof. Daichi Mochihashi: http://chasen.org/~daiti-m/diary/
    Statistically recognize long phrases with Normalized PMI: http://chasen.org/~daiti-m/diary/misc/phraser.py
    https://courses.engr.illinois.edu/cs440/fa2018/lectures/lect36.html
    https://courses.engr.illinois.edu/cs447/fa2018/Slides/Lecture17HO.pdf
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    https://stackoverflow.com/questions/6589814/what-is-the-difference-between-dict-and-collections-defaultdict
    https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    https://stackoverflow.com/questions/47606995/python3-change-dictionary-key-from-string-to-tuple-of-strings
    """
    def __init__(self, threshold: float = 0.1, minfreq: int = 2) -> None:
        # Download phrase dictionaries once.
        self.unigram_phrase_bin = cached_path(f"{HF_myWord_DICT_REPO}/unigram-phrase.bin")
        self.bigram_phrase_bin = cached_path(f"{HF_myWord_DICT_REPO}/bigram-phrase.bin")
        self.threshold = threshold
        self.minfreq = minfreq
        super().__init__(engine="myWord")

    # ------------------------------------------------------------------
    def tokenize(self, raw_text: str) -> List[str]:
        unigram = phr.read_dict(self.unigram_phrase_bin)
        bigram = phr.read_dict(self.bigram_phrase_bin)
        phrases = phr.compute_phrase(unigram, bigram, self.threshold, self.minfreq)
        words = super().tokenize(raw_text)
        if not words:
            return []
        collocated = phr.collocate(words, phrases)
        return collocated


# -------------------------------------------------------------------------------------
class SentenceTokenizer:
    """
    Bi-LSTM based Sentence Tokenization model.
    :Example:
    from myNLP.tokenize import Tokenizer
    tokenizer = Tokenizer.SentenceTokenizer()
    sentences = tokenizer.tokenize("ညာဘက်ကိုယူပြီးတော့တည့်တည့်သွားပါခင်ဗျားငါးမိနစ်လောက်ကြာလိမ့်မယ်")
    print(sentences)
    #   [['ညာ', 'ဘက်', 'ကို', 'ယူ', 'ပြီး', 'တော့', 'တည့်တည့်', 'သွား', 'ပါ'],
    #    ['ခင်ဗျား', 'ငါး', 'မိနစ်', 'လောက်', 'ကြာ', 'လိမ့်', 'မယ်']]
    """
    def __init__(self) -> None:
        self.word_tokenizer = WordTokenizer(engine="LSTM")
        self._init_lstm()

    # ---------------------- model & assets ----------------------------
    def _init_lstm(self) -> None:
        self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/mySentence-bilstm-v2.h5")
        self.tag_path = cached_path(f"{HF_TOKENIZER_REPO}/tag_map_mySentence.json")
        self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/vocab_mySentence.json")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(self.tag_path, "r", encoding="utf-8") as f:
            self.tag_map = json.load(f)
        self.model = load_model(self.model_path)

    # ------------------------------------------------------------------
    def tokenize(self, raw_text: str) -> List[List[str]]:
        words = self.word_tokenizer.tokenize(raw_text)
        idx_seq = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in words]
        logits = self.model.predict(pad_sequences([idx_seq], padding="post"), verbose=0)
        tag_ids = logits.argmax(axis=-1)[0]
        inv_tag = {v: k for k, v in self.tag_map.items()}

        sentences: List[List[str]] = []
        current: List[str] = []
        for word, tag_id in zip(words, tag_ids):
            tag = inv_tag[tag_id]
            current.append(word)
            if tag == "E":
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        return sentences


# -------------------------------------------------------------------------------------
__all__ = [
    "SyllableTokenizer",
    "WordTokenizer",
    "PhraseTokenizer",
    "SentenceTokenizer",
]
