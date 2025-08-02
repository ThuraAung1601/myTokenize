from __future__ import annotations
import re
import json
from typing import List
import numpy as np
import pycrfsuite
import tensorflow as tf
import sentencepiece as sp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cached_path import cached_path
from .myWord import word_segment as wseg, phrase_segment as phr

tf.get_logger().setLevel("ERROR")

# ---------- Local helpers / constants -------------------------------------------------
HF_TOKENIZER_REPO = "hf://LULab/myNLP-Tokenization/Models"
HF_myWord_DICT_REPO = "hf://LULab/myNLP-Tokenization/Models/dict_ver1"


class SyllableTokenizer:
    def __init__(self) -> None:
        self.myConsonant: str = r"က-အ"
        self.enChar: str = r"a-zA-Z0-9"
        self.otherChar: str = r"၎ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s"
        self.ssSymbol: str = r"္"
        self.aThat: str = r"်"
        self.BreakPattern: re.Pattern = re.compile(
            r"((?<!" + self.ssSymbol + r")[" + self.myConsonant + r"](?![" + self.aThat + self.ssSymbol + r"])"
            + r"|[" + self.enChar + self.otherChar + r"])"
        )

    def tokenize(self, raw_text: str) -> List[str]:
        line: str = re.sub(self.BreakPattern, " " + r"\1", raw_text)
        return line.split()


class BPETokenizer:
    def __init__(self, model_folder: str = "BPE_v1") -> None:
        self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/{model_folder}/BPE_v1.spm")
        self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/{model_folder}/BPE_v1.vocab")

    def tokenize(self, raw_text: str) -> List[str]:
        sp_model = sp.SentencePieceProcessor()
        sp_model.load(self.model_path)
        return sp_model.encode(raw_text, out_type=str)


class UnigramTokenizer:
    def __init__(self, model_folder: str = "Unigram_v1") -> None:
        self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/{model_folder}/Unigram_v1.spm")
        self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/{model_folder}/Unigram_v1.vocab")

    def tokenize(self, raw_text: str) -> List[str]:
        sp_model = sp.SentencePieceProcessor()
        sp_model.load(self.model_path)
        return sp_model.encode(raw_text, out_type=str)


class WordTokenizer(SyllableTokenizer):
    def __init__(self, engine: str = "myWord") -> None:
        super().__init__()
        self.engine = engine

        if engine == "CRF":
            self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/wordseg_c2_crf.crfsuite")
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_path)

        elif engine == "LSTM":
            self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/myWseg-s4-bilstm-v2.h5")
            self.tag_map_path = cached_path(f"{HF_TOKENIZER_REPO}/tag_map_s4.json")
            self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/vocab_s4.json")

            with open(self.tag_map_path, "r", encoding="utf-8") as f:
                self.tag_map = json.load(f)
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

            self.model = load_model(self.model_path)

        elif engine == "myWord":
            self.unigram_word_bin = cached_path(f"{HF_myWord_DICT_REPO}/unigram-word.bin")
            self.bigram_word_bin = cached_path(f"{HF_myWord_DICT_REPO}/bigram-word.bin")

        else:
            raise ValueError(f"No {engine}. engine must be one of: CRF | LSTM | myWord")

    @staticmethod
    def word2features(sent: str, i: int) -> dict[str, object]:
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

    def _crf_features(self, sent: str) -> List[dict[str, object]]:
        return [self.word2features(sent, i) for i in range(len(sent))]

    def tokenize(self, raw_text: str) -> List[str]:
        engine = self.engine
        if engine == "CRF":
            sent = raw_text.replace(" ", "")
            preds = self.tagger.tag(self._crf_features(sent))
            merged = "".join(char + ("_" if tag == "|" else "") for char, tag in zip(sent, preds))
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


class PhraseTokenizer(WordTokenizer):
    def __init__(self, threshold: float = 0.1, minfreq: int = 2) -> None:
        self.unigram_phrase_bin = cached_path(f"{HF_myWord_DICT_REPO}/unigram-phrase.bin")
        self.bigram_phrase_bin = cached_path(f"{HF_myWord_DICT_REPO}/bigram-phrase.bin")
        self.threshold = threshold
        self.minfreq = minfreq
        super().__init__(engine="myWord")

    def tokenize(self, raw_text: str) -> List[str]:
        unigram = phr.read_dict(self.unigram_phrase_bin)
        bigram = phr.read_dict(self.bigram_phrase_bin)
        phrases = phr.compute_phrase(unigram, bigram, self.threshold, self.minfreq)
        words = super().tokenize(raw_text)
        if not words:
            return []
        collocated = phr.collocate(words, phrases)
        return collocated


class SentenceTokenizer:
    def __init__(self) -> None:
        self.word_tokenizer = WordTokenizer(engine="LSTM")
        self.model_path = cached_path(f"{HF_TOKENIZER_REPO}/mySentence-bilstm-v2.h5")
        self.tag_path = cached_path(f"{HF_TOKENIZER_REPO}/tag_map_mySentence.json")
        self.vocab_path = cached_path(f"{HF_TOKENIZER_REPO}/vocab_mySentence.json")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(self.tag_path, "r", encoding="utf-8") as f:
            self.tag_map = json.load(f)
        self.model = load_model(self.model_path)

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


__all__ = [
    "SyllableTokenizer",
    "BPETokenizer",
    "UnigramTokenizer",
    "WordTokenizer",
    "PhraseTokenizer",
    "SentenceTokenizer",
]
