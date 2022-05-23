import re
import nltk
from abc import ABC, abstractmethod

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class PreprocessingNode(ABC):

    @abstractmethod
    def process(self, inp: str):
        pass


class PunctuationNode(PreprocessingNode):

    def process(self, inp: str):
        return re.sub(r"[^a-zA-Z0-9]", " ", inp.lower())


class StopWordsNode(PreprocessingNode):

    def __init__(self, lang: str):
        self.__stop_words = nltk.corpus.stopwords.words(lang)

    def process(self, inp: str):
        return ' '.join([i for i in inp.split() if i not in self.__stop_words])


class StemmingNode(PreprocessingNode):

    def __init__(self):
        self.__stemmer = PorterStemmer()

    def process(self, inp: str):
        return ' '.join([self.__stemmer.stem(i) for i in inp.split()])


class LemmatizationNode(PreprocessingNode):

    def __init__(self):
        self.__lemmatizer = WordNetLemmatizer()

    def process(self, inp: str):
        return ' '.join([self.__lemmatizer.lemmatize(i) for i in inp.split()])


class PreprocessingPipeline(PreprocessingNode):
    __nodes = []

    def process(self, inp: str):
        res = inp

        for node in self.__nodes:
            res = node.process(res)

        return res

    def add(self, node):
        self.__nodes.append(node)
