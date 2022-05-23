import pandas as pd

from src.core import Model
from src.embeddings import Embedding
from sklearn.cluster import SpectralClustering


class CategoryBasedModel(Model):
    category_aggregated = None
    general_aggregated = None
    clusters = None

    def __init__(self, embedding: Embedding, cat: int):
        self.embedding = embedding
        self.cat = cat

    def fit(self, data, jokes):
        embedded = self.embedding.to_vec(jokes)

        kmeans = SpectralClustering(n_clusters=self.cat)
        kmeans.fit(embedded)
        self.clusters = kmeans.labels_

        self.category_aggregated = data
        self.category_aggregated['category'] = self.category_aggregated.apply(lambda x: self.clusters[x['joke_id'] - 1],
                                                                              axis=1)
        self.general_aggregated = self.category_aggregated.groupby('category').mean()['Rating'].to_numpy()

    def predict(self, data_x, jokes):
        def predict_single(row):
            joke_id = row['joke_id']
            user_id = row['user_id']
            category = self.clusters[joke_id - 1]

            return self.category_aggregated.loc[user_id, category]['Rating'] \
                if (user_id, category) in self.category_aggregated.index \
                else self.general_aggregated[category]

        d = {'Rating': [predict_single(r[1]) for r in data_x.iterrows()]}
        predicted = pd.DataFrame(data=d)
        return predicted
