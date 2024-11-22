import os
from typing import List, Tuple
import re
import json
import numpy as np
import pycrfsuite
import tensorflow as tf
import sentencepiece as sp
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .myWord import word_segment as wseg, phrase_segment as phr

import git
from git import GitCommandError

def clone_repository(repo_url, local_path):
    # Clone the repository
    repo = git.Repo.clone_from(repo_url, local_path)
    return repo

def pull_lfs_file(repo, lfs_file_path):
    # pull_lfs_file(repo, lfs_file_path)
    try:
        # Fetch the specific LFS file
        repo.git.checkout(lfs_file_path)
    except GitCommandError as e:
        print("LFS Not Found at {}".format(lfs_file_path))

class SyllableTokenizer:
    """
    Syllable Tokenizer using Sylbreak for Myanmar language.
    Author: Ye Kyaw Thu
    Link: https://github.com/ye-kyaw-thu/sylbreak
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.SyllableTokenizer()
    syllables = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
    print(syllables)
    # ['မြန်', 'မာ', 'နိုင်', 'ငံ', '။']
    """
    def __init__(self) -> None:
        self.myConsonant: str = r"က-အ"
        self.enChar: str = r"a-zA-Z0-9"
        self.otherChar: str = r"၎ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s"
        self.ssSymbol: str = r'္'
        self.aThat: str = r'်'
        self.BreakPattern: re.Pattern = re.compile(r"((?<!" + self.ssSymbol + r")["+ self.myConsonant + r"](?![" + self.aThat + self.ssSymbol + r"])" + r"|[" + self.enChar + self.otherChar + r"])")

    def tokenize(self, raw_text: str) -> List[str]:
        """
        Tokenizes the input text into syllables.
        :param raw_text: Input text in Myanmar language.
        :return: List of syllables.
        """
        self.raw_text: str = raw_text
        self.data: str = ""
        line: str = re.sub(self.BreakPattern, " "+r"\1", self.raw_text)
        self.data += line
        return self.data.split()

class BPETokenizer:
    """
    BPE Tokenizer using Sentencepiece
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.BPETokenizer()
    tokens = tokenizer.tokenize("ရွေးကောက်ပွဲမှာနိုင်ထားတဲ့ဒေါ်နယ်ထရမ့်")
    print(tokens)
    # ['▁ရွေးကောက်ပွဲ', 'မှာ', 'နိုင်', 'ထား', 'တဲ့', 'ဒေါ်', 'နယ်', 'ထ', 'ရ', 'မ့်']
    """
    def __init__(self, model_path: str = os.path.join(os.path.dirname(__file__), "SentencePiece", "bpe_sentencepiece_model.model")) -> None:
        self.model_path = model_path
    def tokenize(self, raw_text: str) -> List[str]:

        sp_model = sp.SentencePieceProcessor()
        sp_model.load(self.model_path)
        encoded = sp_model.encode(raw_text, out_type=str)
        return encoded

class UnigramTokenizer:
    """
    Unigram Tokenizer using Sentencepiece
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.UnigramTokenizer()
    tokens = tokenizer.tokenize("ရွေးကောက်ပွဲမှာနိုင်ထားတဲ့ဒေါ်နယ်ထရမ့်")
    print(tokens)
    # ['▁ရွေးကောက်ပွဲ', 'မှာ', 'နိုင်', 'ထား', 'တဲ့', 'ဒေါ', '်', 'နယ်', 'ထ', 'ရ', 'မ', '့်']
    """
    def __init__(self, model_path: str = os.path.join(os.path.dirname(__file__), "SentencePiece", "unigram_sentencepiece_model.model")) -> None:
        self.model_path = model_path
    def tokenize(self, raw_text: str) -> List[str]:

        sp_model = sp.SentencePieceProcessor()
        sp_model.load(self.model_path)
        encoded = sp_model.encode(raw_text, out_type=str)
        return encoded
    
class WordTokenizer(SyllableTokenizer):
    """
    Word Tokenization class: myWord, CRFs and, BiLSTM available.
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.WordTokenizer()
    words = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
    print(words)
    # ['မြန်မာ', 'နိုင်ငံ', '။']
    """
    def __init__(self, engine="CRF") -> None:
        super().__init__()
        self.engine = engine
        if self.engine == "CRF":
            self.model_path = os.path.join(os.path.dirname(__file__), "CRFTokenizer", "wordseg_c2_crf.crfsuite")
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_path)
        elif self.engine == "LSTM":
            self._load_lstm_model()
        elif self.engine == "myWord":
            self._load_myWord_dict()
        else:
            raise ValueError("Invalid choice. Available models are CRF: Conditional Random Fields, LSTM: BiLSTM")

    def _load_lstm_model(self) -> None:
        _neural_path = os.path.join(os.path.dirname(__file__), "NeuralTokenizer")
        _repo = "https://huggingface.co/LULab/myNLP-Tokenization"
        _local_path = os.path.join(os.path.dirname(__file__), "NeuralTokenizer")
        _lfs_wseg_file_path = "Models/myWseg-s2-bilstm.h5"
        _lfs_sseg_file_path = "Models/mySentence-bilstm-v2.h5"

        if not os.path.exists(_neural_path):
            print("Downloading neural network models ...")
            repo = clone_repository(_repo, _local_path)
            pull_lfs_file(repo, _lfs_wseg_file_path)
            pull_lfs_file(repo, _lfs_sseg_file_path)
            print("Download Completed.")

        self.bilstm_s4_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "myWseg-s4-bilstm-v2.h5")
        self.tag_s4_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "tag_map_s4.json")
        self.vocab_s4_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "vocab_s4.json")
        
        with open(self.vocab_s4_path, 'r') as f:
            self.vocab: dict = json.load(f)
        with open(self.tag_s4_path, 'r') as f:
            self.tag_map: dict = json.load(f)
        self.model: load_model = load_model(self.bilstm_s4_path)
    
    def _load_myWord_dict(self) -> None:
        # https://github.com/ye-kyaw-thu/myWord
        _dict_path = os.path.join(os.path.dirname(__file__), "Dict")
        _repo = "https://huggingface.co/datasets/LULab/myWord-dict"
        _local_path = os.path.join(os.path.dirname(__file__), "Dict")
        _lfs_unigram_word_bin = "dict_ver1/unigram-word.bin"
        _lfs_bigram_word_bin = "dict_ver1/bigram-word.bin"
        _lfs_unigram_phrase = "dict_ver1/unigram-phrase.bin"
        _lfs_bigram_phrase = "dict_ver1/bigram-phrase.bin"

        if not os.path.exists(_dict_path):
            print("Downloading myWord dictionary ...")
            repo = clone_repository(_repo, _local_path)
            pull_lfs_file(repo, _lfs_unigram_word_bin)
            pull_lfs_file(repo, _lfs_bigram_word_bin)
            pull_lfs_file(repo, _lfs_unigram_phrase)
            pull_lfs_file(repo, _lfs_bigram_phrase)
            print("Download Completed.")

        self.unigram_word_bin: str = os.path.join(os.path.dirname(__file__), "Dict", "dict_ver1", "unigram-word.bin")
        self.bigram_word_bin: str = os.path.join(os.path.dirname(__file__), "Dict", "dict_ver1", "bigram-word.bin")
        
    @staticmethod
    def word2features(sent, i):
        word = sent[i]
        features = {'number': word.isdigit()}
        if i > 0:
            word_prev = sent[i - 1]
            features.update({
                'prev_word.lower()': word_prev.lower(),
                'prev_number': word_prev.isdigit(),
                'bigram': word_prev.lower() + '_' + word.lower()
            })
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            word_next = sent[i + 1]
            features.update({
                'next_word.lower()': word_next.lower(),
                'next_number': word_next.isdigit(),
            })
        else:
            features['EOS'] = True
        if i > 1:
            word_prev_prev = sent[i - 2]
            features['trigram_1'] = word_prev_prev.lower() + '_' + word_prev.lower() + '_' + word.lower()
        if i < len(sent) - 2:
            word_next_next = sent[i + 2]
            features['trigram_2'] = word.lower() + '_' + word_next.lower() + '_' + word_next_next.lower()
        return features
    
    def get_features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def tokenize(self, raw_text: str) -> List[str]:
        if self.engine == "CRF":
            sent = raw_text.replace(" ", "")
            predictions = self.tagger.tag(self.get_features(sent))
            complete = ""
            for i, p in enumerate(predictions):
                if p == "|":
                    complete += sent[i] + "_"
                else:
                    complete += sent[i]
            return complete.split("_")[:-1]
        elif self.engine == "LSTM":
            if self.model is None:
                raise ValueError("Model is not loaded. Load the model first.")
            self.raw_text: str = raw_text
            try:
                self.tokens: List[str] = super().tokenize(self.raw_text)
                indexed_tokens: List[int] = [self.vocab.get(token, self.vocab['<UNK>']) for token in self.tokens]
                padded_sequence: np.ndarray = pad_sequences([indexed_tokens], padding='post')
                
                # Reshape to (batch_size, timesteps, features)
                padded_sequence = np.expand_dims(padded_sequence, -1)  # Shape becomes (1, sequence_length, 1)
                
                predictions: np.ndarray = self.model.predict(padded_sequence, verbose=0)
                predicted_tags: np.ndarray = np.argmax(predictions, axis=-1)[0]
                reverse_tag_map: dict = {v: k for k, v in self.tag_map.items()}
                
                segmented_sentence: List[str] = []
                for i, (syl, tag_index) in enumerate(zip(self.tokens, predicted_tags)):
                    tag: str = reverse_tag_map[tag_index]
                    if tag == '|':
                        segmented_sentence.append(syl + ' ')
                    else:
                        segmented_sentence.append(syl)
                return ''.join(segmented_sentence).split()
            except tf.errors.InvalidArgumentError as e:
                print("Error occurred:", e)
                pass
                return []
        elif self.engine == "myWord":
            wseg.P_unigram = wseg.ProbDist(self.unigram_word_bin, True)
            wseg.P_bigram = wseg.ProbDist(self.bigram_word_bin, False)
            listString = wseg.viterbi(raw_text.replace(" ", "").strip())  # remove space between words and pass to viterbi()
            # print("listString: " + str(listString))
            wordStr = ' '.join(listString[1])  # Concatenate the list of segmented words
            return wordStr.split()  # Split the wordStr into a list of words
        else:
            raise ValueError("Error")

class PhraseTokenizer(WordTokenizer):
    """
    NPMI based Unsupervised Phrase Segmentation
    Author: Ye Kyaw Thu
    Link: https://github.com/ye-kyaw-thu/myWord/blob/main/phrase_segment.py 
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.PhraseTokenizer()
    words = tokenizer.tokenize("ညာဘက်ကိုယူပြီးတော့တည့်တည့်သွားပါ")
    print(words)
    # ['ညာဘက်_ကို', 'ယူ', 'ပြီး_တော့', 'တည့်တည့်', 'သွား_ပါ']
    """
    def __init__(self, threshold=0.1, minfreq=2) -> None:
        self.unigram_phrase_bin: str = os.path.join(os.path.dirname(__file__), "Dict", "dict_ver1", "unigram-phrase.bin")
        self.bigram_phrase_bin: str = os.path.join(os.path.dirname(__file__), "Dict", "dict_ver1", "bigram-phrase.bin")
        self.threshold = threshold
        self.minfreq = minfreq
        super().__init__()
    def tokenize(self, raw_text: str) -> List[str]:
        unigram = phr.read_dict(self.unigram_phrase_bin)   
        bigram = phr.read_dict(self.bigram_phrase_bin)  
        phrases = phr.compute_phrase (unigram, bigram, self.threshold, self.minfreq)
        words: List[str] = super().tokenize(raw_text)
        if len(words) > 0:
                sentence = phr.collocate(words, phrases)
                sentence = ' '.join(sentence)
        return sentence.split()

class SentenceTokenizer:
    """
    Bi-LSTM based Sentence Tokenization model.
    :Example:
    from myTokenizer import Tokenizer
    tokenizer = Tokenizer.SentenceTokenizer()
    sentences = tokenizer.tokenize("ညာဘက်ကိုယူပြီးတော့တည့်တည့်သွားပါခင်ဗျားငါးမိနစ်လောက်ကြာလိမ့်မယ်")
    print(sentences)
    #   [['ညာ', 'ဘက်', 'ကို', 'ယူ', 'ပြီး', 'တော့', 'တည့်တည့်', 'သွား', 'ပါ'],
    #    ['ခင်ဗျား', 'ငါး', 'မိနစ်', 'လောက်', 'ကြာ', 'လိမ့်', 'မယ်']]
    """
    def __init__(self) -> None:
        self.word_tokenizer: WordTokenizer = WordTokenizer()
        self._load_lstm_model()

    def _load_lstm_model(self) -> None:
        self.bilstm_mySentence_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "mySentence-bilstm-v2.h5")
        self.tag_mySentence_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "tag_map_mySentence.json")
        self.vocab_mySentence_path: str = os.path.join(os.path.dirname(__file__), "NeuralTokenizer", "Models", "vocab_mySentence.json")

        with open(self.vocab_mySentence_path, 'r') as f:
            self.vocab: dict = json.load(f)
        with open(self.tag_mySentence_path, 'r') as f:
            self.tag_map: dict = json.load(f)
        self.model: load_model = load_model(self.bilstm_mySentence_path)

    def tokenize(self, raw_text: str) -> List[List[str]]:
        if self.model is None:
            raise ValueError("Model is not loaded. Load the model first.")
        word_tokens: List[str] = self.word_tokenizer.tokenize(raw_text)
        indexed_tokens: List[int] = [self.vocab.get(token, self.vocab['<UNK>']) for token in word_tokens]
        padded_sequence: np.ndarray = pad_sequences([indexed_tokens], padding='post')
        predictions: np.ndarray = self.model.predict(padded_sequence, verbose=0)
        predicted_tags: np.ndarray = np.argmax(predictions, axis=-1)[0]
        reverse_tag_map: dict = {v: k for k, v in self.tag_map.items()}
        segmented_sentence: List[List[str]] = []
        sublist: List[Tuple[str, str]] = []
        for i, (word, tag_index) in enumerate(zip(word_tokens, predicted_tags)):
            tag: str = reverse_tag_map[tag_index]
            sublist.append((word, tag))
            if tag == 'E':
                segmented_sentence.append(sublist)
                sublist = []
        if sublist:
            segmented_sentence.append(sublist)
        words_only: List[List[str]] = [[word for word, _ in sentence] for sentence in segmented_sentence]
        return words_only
