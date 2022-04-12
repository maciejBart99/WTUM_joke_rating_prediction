import numpy as np
import pandas as pd

from src.core import Model

class UsersSimilarity(Model):
    def __init__(self):
        pass
    def fit(self, data, jokes):
        self.train_data = data
    def predict(self, data_x, jokes):
        predicted = pd.DataFrame(columns=['Rating'])
        to_predict = data_x.get(['user_id', 'joke_id'])
        for i, r in to_predict.iterrows():
            joke_id = r['joke_id']
            user_id = r['user_id']
            data = self.train_data
            data_user = data[data['user_id'] == user_id] # selected_by_this_user
            data_joke = data[data['joke_id'] == joke_id] # selected_by_joke and only other users
            data_joke = data_joke[data_joke['user_id'] != user_id]
            other_users = data_joke['user_id'].unique()
            user_sim = {}
            for current_user in other_users:
                current_user_data = data[data['user_id'] == current_user]
                common_jokes = pd.merge(current_user_data, data_user, how ='inner', on =['joke_id'])
                d_sum = 0
                jokes_number = common_jokes.shape[0]
                if jokes_number == 0:
                    continue
                for i, row in common_jokes.iterrows():
                    d_sum = d_sum + abs(row['Rating_x']-row['Rating_y'])
                user_sim[current_user] = d_sum/jokes_number
            dt = np.dtype([('user_id', np.int64), ('distance', np.float64)])
            sorted_user_sim = np.array([(k, v) for k, v in sorted(user_sim.items(), key=lambda item: item[1])], dtype=dt)
            num = min(len(sorted_user_sim), 15)
            sum = 0
            for i in range(0, num):
                sum = sum + data_joke.loc[data_joke.user_id == sorted_user_sim[i]['user_id']]['Rating'].iloc[0]
            result = sum / num
            predicted = predicted.append({'Rating': result}, ignore_index=True)
        return predicted
