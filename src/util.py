import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def compute_rating_matrix(df: pd.DataFrame):
    N = len(df['user_id'].unique())
    M = len(df['joke_id'].unique())

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    joke_mapper = dict(zip(np.unique(df["joke_id"]), list(range(M))))

    user_index = [user_mapper[i] for i in df['user_id']]
    movie_index = [joke_mapper[i] for i in df['joke_id']]

    return csr_matrix((df["Rating"], (movie_index, user_index)), shape=(M, N))
