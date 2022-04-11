import numpy as np
import pandas as pd

from src.core import Model
from src.util import compute_rating_matrix
from src.text_similarity_models import TextSimilarityModel


class ContentBaseModel(Model):
    ratings_matrix = None

    def __init__(self, size: int, similarity: TextSimilarityModel):
        self.similarity = similarity
        self.size = size
        self.similarity_matrix = np.ones((size, size))

    def fit(self, data, jokes):
        self.ratings_matrix = compute_rating_matrix(data)

        for i in range(self.size):
            for j in range(0, i):
                self.similarity_matrix[i, j] = self.similarity.get_similarity(jokes[i], jokes[j])

    def predict(self, data_x, jokes):
        def predict_single(row):
            joke_id = row['joke_id']
            user_id = row['user_id']

            sm = 0
            wg = 0
            C = self.ratings_matrix[:, user_id - 1].toarray()[:, 0]
            D = C.nonzero()[0]

            for ind in D:
                sim = self.similarity_matrix[joke_id - 1, ind]
                sm += sim * self.ratings_matrix[ind, user_id - 1]
                wg += sim

            return sm / wg if wg > 0 else 0

        d = {'Rating': [predict_single(r[1]) for r in data_x.iterrows()]}
        predicted = pd.DataFrame(data=d)
        return predicted
