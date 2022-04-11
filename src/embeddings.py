import re
import math
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod


def remove_punctuation(inp: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", " ", inp.lower())


class Embedding(ABC):

    @abstractmethod
    def to_vec(self, inp: List[str]):
        pass


class BagOfWords(Embedding):

    def __init__(self, all_sentences: List[str]):
        word_set = set()

        for sentence in all_sentences:
            word_set = word_set.union(sentence.split())

        self.vocabulary = list(word_set)

    def create_word_dict(self, inp: str):
        encoding = dict.fromkeys(self.vocabulary, 0)

        for word in set(inp.split()):
            encoding[word] = inp.count(word)

        return encoding

    def to_vec(self, inp: List[str]):
        return [list(self.create_word_dict(x).values()) for x in inp]


class TFIDF(Embedding):

    def __init__(self, all_sentences: List[str]):
        self.__bag_model = BagOfWords(all_sentences)

        word_dicts = []

        for sentence in all_sentences:
            word_dicts.append(self.__bag_model.create_word_dict(sentence))

        self.__idf_dict = dict.fromkeys(word_dicts[0].keys(), 0)

        for sentence_dict in word_dicts:
            for word, v in sentence_dict.items():
                if v > 0:
                    self.__idf_dict[word] += 1

        for word, v in self.__idf_dict.items():
            self.__idf_dict[word] = math.log10(len(word_dicts) / float(v))

    def to_vec(self, inp: List[str]):
        result = []

        for sent in inp:
            w_dict = self.__bag_model.create_word_dict(sent)
            tfidf_dict = dict.fromkeys(self.__bag_model.vocabulary, 0)

            for word, v in w_dict.items():
                tfidf_dict[word] = v * self.__idf_dict[word]

            result.append(list(tfidf_dict.values()))

        return result


class SentenceBert(Embedding):

    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.__model = SentenceTransformer(model_name)

    def to_vec(self, inp: List[str]):
        return self.__model.encode(inp)


class CachedEmbedding(Embedding):
    cache: Dict[str, list] = {}

    def __init__(self, wrapped: Embedding):
        self.wrapped = wrapped

    def to_vec(self, inp: List[str]):
        missing = [x for x in inp if x not in self.cache]
        missing_results = self.wrapped.to_vec(missing)

        for key, val in zip(missing, missing_results):
            self.cache[key] = val

        return [self.cache[x] for x in self.cache.keys()]
