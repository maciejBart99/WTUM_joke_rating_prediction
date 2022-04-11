from typing import List

from src.embeddings import TFIDF, SentenceBert, CachedEmbedding
from src.preprocessing import PreprocessingPipeline, PunctuationNode, StopWordsNode, LemmatizationNode
from src.text_similarity_models import TextSimilarityModel, CosineSimilarity, CachedSimilarityModel


class FullPreprocessingTFIDFCosineModel(TextSimilarityModel):

    def __init__(self, text_set: List[str]):
        self.__preprocessing_pipeline = PreprocessingPipeline()
        self.__preprocessing_pipeline.add(PunctuationNode())
        self.__preprocessing_pipeline.add(StopWordsNode('english'))
        self.__preprocessing_pipeline.add(LemmatizationNode())

        self.__embedding = CachedEmbedding(TFIDF([self.__preprocessing_pipeline.process(t) for t in text_set]))

        self.__cosine = CachedSimilarityModel(CosineSimilarity(self.__embedding))

    def get_similarity(self, a: str, b: str):
        pre_a = self.__preprocessing_pipeline.process(a)
        pre_b = self.__preprocessing_pipeline.process(b)

        return self.__cosine.get_similarity(pre_a, pre_b)


class BertCosineModel(TextSimilarityModel):

    def __init__(self):
        self.__embedding = CachedEmbedding(SentenceBert())

        self.__cosine = CachedSimilarityModel(CosineSimilarity(self.__embedding))

    def get_similarity(self, a: str, b: str):
        return self.__cosine.get_similarity(a, b)
