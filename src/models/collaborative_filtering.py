import numpy as np
import pandas as pd

from src.core import Model
from src.util import compute_rating_matrix


THRESHOLD = 0.1


class CollaborativeFilteringModel(Model):
    ratings_matrix = None

    def fit(self, data: pd.DataFrame, jokes):
        self.ratings_matrix = compute_rating_matrix(data)

    def predict(self, data_x, jokes):
        M = self.ratings_matrix.transpose().toarray()
        M_presence = M != 0

        def predict_single(row):
            joke_id = row['joke_id'] - 1
            user_id = row['user_id'] - 1

            sm = 0
            wg = 0

            row = M[user_id, :]
            row_presence = row != 0

            P = (M_presence | np.repeat(row_presence.reshape((1, row.shape[0])), M.shape[0], 0)).astype(int)
            D = np.sum(M * np.repeat(row.reshape((1, row.shape[0])), M.shape[0], 0), 1)
            N = np.linalg.norm(M * P, 2, 1) * np.linalg.norm(row * row_presence, 2)
            C_map = np.abs(D / N)
            C_map[np.isnan(C_map)] = 0

            C = self.ratings_matrix[joke_id, :].toarray()[0, :]
            D = C.nonzero()[0]

            for ind in D:
                sim = C_map[ind]

                if sim > THRESHOLD:
                    sm += sim * M[ind, joke_id]
                    wg += sim

            return sm / wg if wg > 0 else 0

        d = {'Rating': [predict_single(r[1]) for r in data_x.iterrows()]}
        predicted = pd.DataFrame(data=d)
        return predicted
