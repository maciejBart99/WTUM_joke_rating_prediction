from abc import ABC, abstractmethod

import numpy as np

from src.embeddings import remove_punctuation, Embedding


class TextSimilarityModel(ABC):

    @abstractmethod
    def get_similarity(self, a: str, b: str):
        pass


class JaccardModel(TextSimilarityModel):

    def get_similarity(self, a: str, b: str):
        a_s = set(remove_punctuation(a).split())
        b_s = set(remove_punctuation(b).split())

        intersection = a_s.intersection(b_s)
        union = a_s.union(b_s)

        return len(intersection) / len(union)


class KMeansModel(TextSimilarityModel):

    def get_similarity(self, a: str, b: str):
        pass


class CosineSimilarity(TextSimilarityModel):

    def __init__(self, embedding: Embedding):
        self.__embedding = embedding

    def get_similarity(self, a: str, b: str):
        e = self.__embedding.to_vec([a, b])

        np_e_a = np.array(list(e[0]))
        np_e_b = np.array(list(e[1]))

        return np.divide(np.dot(np_e_a, np_e_b), np.multiply(np.sqrt(np.sum(np.power(np_e_a, 2))), np.sqrt(np.sum(np.power(np_e_b, 2)))))


class LatentSemanticIndexing(TextSimilarityModel):

    def get_similarity(self, a: str, b: str):
        pass


class WordMoverDistance(TextSimilarityModel):

    def get_similarity(self, a: str, b: str):
        pass
