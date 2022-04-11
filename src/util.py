import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def compute_rating_matrix(df: pd.DataFrame):
    N = len(df['user_id'].unique())
    M = len(df['joke_id'].unique())

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["joke_id"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["joke_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    movie_index = [movie_mapper[i] for i in df['joke_id']]

    return csr_matrix((df["Rating"], (movie_index, user_index)), shape=(M, N))
