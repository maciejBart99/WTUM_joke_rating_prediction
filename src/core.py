from os import path
from abc import abstractmethod
import pandas as pd
import sklearn.metrics as sklmetrics

DATA_PATH = '../data'

DEFAULT_JOKES_FILENAME = path.join(DATA_PATH, 'jokes.csv')
DEFAULT_RATING_FILENAME = path.join(DATA_PATH, 'rating.csv')


class Model:
    @abstractmethod
    def fit(self, data_x, data_y):
        pass

    @abstractmethod
    def predict(self, data_x):
        pass


class Environment:
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model

    def run(self, model_trainable=True):
        if model_trainable:
            rating_train_df = self.data_loader.train_data
            train_x, train_y = rating_train_df.get(["id", "user_id", "joke_id"]), rating_train_df.get("Rating")
            self.model.fit(train_x, train_y)
        rating_test_df = self.data_loader.test_data
        test_x, test_y = rating_test_df.get(["id", "user_id", "joke_id"]), rating_test_df.get("Rating")
        predicted_y = self.model.predict(test_x)
        print("Explained variance score: " + str(sklmetrics.explained_variance_score(test_y, predicted_y)))
        print("Mean absolute error: " + str(sklmetrics.mean_absolute_error(test_y, predicted_y)))
        print("Mean squared error: " + str(sklmetrics.mean_squared_error(test_y, predicted_y)))
        print("Median absolute error: " + str(sklmetrics.median_absolute_error(test_y, predicted_y)))
        print("R2 coefficient: " + str(sklmetrics.r2_score(test_y, predicted_y)))


class DataLoader:
    def __init__(self, jokes_filename=DEFAULT_JOKES_FILENAME, rating_filename=DEFAULT_RATING_FILENAME):
        self._jokes_df = pd.read_csv(jokes_filename)
        self._rating_df = pd.read_csv(rating_filename)
        self._rating_train_df = None
        self._rating_test_df = None

        self.resample()

    def resample(self, test_frac=.2, per_user=False):
        if test_frac == 0:
            self._rating_train_df = self._rating_df
            self._rating_test_df = self._rating_df.head(0)

        if per_user:
            df = self._rating_df
            test_frames = [self.for_user(user_id, df=df).sample(frac=test_frac) for user_id in df.user_id.unique()]
            self._rating_test_df = pd.concat(test_frames)
        else:
            self._rating_test_df = self._rating_df.sample(frac=test_frac)

        self._rating_train_df = self._rating_df.drop(self._rating_test_df.index)

    @property
    def train_data(self):
        return self._rating_train_df

    @property
    def test_data(self):
        return self._rating_test_df

    @property
    def jokes(self):
        return self._jokes_df

    def for_user(self, user_id, df=None):
        if df is None:
            df = self.train_data

        return df.loc[df['user_id'] == user_id]

    def for_joke(self, joke_id, df=None):
        if df is None:
            df = self.train_data

        return df.loc[df['joke_id'] == joke_id]

    def joke_text(self, joke_id):
        return self.jokes.loc[self.jokes['joke_id'] == joke_id].iloc[0]['joke_text']
