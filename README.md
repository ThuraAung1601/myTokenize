# myTokenize

**myTokenize** is a Python library that tokenizes Myanmar text into syllables, words, phrases, and sentences. It supports multiple tokenization techniques using rule-based, statistical, and neural network-based approaches.

## Features

- **Syllable Tokenization**: Break text into syllables using regex rules.
- **Word Tokenization**: Segment text into words using:
  - `myWord`: Dictionary-based tokenization.
  - `CRF`: Conditional Random Fields-based tokenization.
  - `BiLSTM`: Neural network-based tokenization.
- **Phrase Tokenization**: Identify phrases in text using normalized pointwise mutual information (NPMI).
- **Sentence Tokenization**: Use a BiLSTM model to segment text into sentences.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ThuraAung1601/myTokenize.git
   cd myTokenize
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the library:
   ```bash
   pip install .
   ```

## Usage

### Syllable Tokenizer
```python
from myTokenize import SyllableTokenizer

tokenizer = SyllableTokenizer()
syllables = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
print(syllables)  # ['မြန်', 'မာ', 'နိုင်', 'ငံ', '။']
```

### Word Tokenizer
```python
from myTokenize import WordTokenizer

tokenizer = WordTokenizer(engine="CRF")  # Use "myWord", "CRF", or "LSTM"
words = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
print(words)  # ['မြန်မာ', 'နိုင်ငံ', '။']
```

### Phrase Tokenizer
```python
from myTokenize import PhraseTokenizer

tokenizer = PhraseTokenizer()
phrases = tokenizer.tokenize("ညာဘက်ကိုယူပြီးတော့တည့်တည့်သွားပါ")
print(phrases)  # ['ညာဘက်_ကို', 'ယူ', 'ပြီး_တော့', 'တည့်တည့်', 'သွား_ပါ']
```

### Sentence Tokenizer
```python
from myTokenize import SentenceTokenizer

tokenizer = SentenceTokenizer()
sentences = tokenizer.tokenize("ညာဘက်ကိုယူပြီးတော့တည့်တည့်သွားပါခင်ဗျားငါးမိနစ်လောက်ကြာလိမ့်မယ်")
print(sentences)  # [['ညာ', 'ဘက်', 'ကို', 'ယူ', 'ပြီး', 'တော့', 'တည့်တည့်', 'သွား', 'ပါ'], ['ခင်ဗျား', 'ငါး', 'မိနစ်', 'လောက်', 'ကြာ', 'လိမ့်', 'မယ်']]
```

## Folder Structure

```
./myTokenize/
├── CRFTokenizer
│   └── wordseg_c2_crf.crfsuite
├── Tokenizer.py
└── myWord
    ├── phrase_segment.py
    └── word_segment.py
```

## Dependencies

- Python 3.8
- TensorFlow 2.13
- SentencePiece
- pycrfsuite
- Numpy
- cached_path 1.5.0 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- **Ye Kyaw Thu** 
- **Thura Aung**

## References
- myWord: Syllable, Word and Phrase Segmenter for Burmese, Ye Kyaw Thu, Sept 2021, GitHub Link: https://github.com/ye-kyaw-thu/myWord
- sylbreak: Syllable segmentation tool for Myanmar language (Burmese), Ye Kyaw Thu, GitHub Link: https://github.com/ye-kyaw-thu/sylbreak
- mySentence: Corpus and Models for Burmese (Myanmar language) Sentence Tokenization, GitHub Link: https://github.com/ye-kyaw-thu/mySentence
