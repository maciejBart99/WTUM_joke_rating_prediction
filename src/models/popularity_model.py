import numpy as np
import pandas as pd

from src.core import Model


class PopularityModel(Model):
    avg: np.array = None

    def fit(self, data: pd.DataFrame, jokes):
        self.avg = data.groupby('joke_id').mean()['Rating'].to_numpy()

    def predict(self, data_x, jokes):
        d = {'Rating': [self.avg[r[1]['joke_id'] - 1] for r in data_x.iterrows()]}
        predicted = pd.DataFrame(data=d)
        return predicted
