from setuptools import setup, find_packages
import os

# Load the README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="myTokenize",
    version="0.1.1",
    description="Comprehensive tokenization library for Myanmar language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ye Kyaw Thu, Thura Aung",
    author_email="ykt.nlp.ai@gmail.com, thuraaung.ai.mdy@gmail.com",
    url="https://github.com/ThuraAung1601/myTokenizer",
    packages=["myTokenize"],  # Explicitly specify the package name
    include_package_data=True,
    package_data={
        "myTokenize": [  # Correct package name here
            "CRFTokenizer/*.crfsuite",
            "SentencePiece/*.model",
            "SentencePiece/*.vocab",
            "myWord/*.py",
        ]
    },
    install_requires=[
        "numpy",
        "tensorflow==2.13",
        "python-crfsuite",
        "sentencepiece",
        "matplotlib",
        "GitPython"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
