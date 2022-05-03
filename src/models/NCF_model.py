from typing import List

import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.embeddings import Embedding, SentenceBert
from src.core import Model


class JokeRatingDataset(Dataset):

    def __init__(self, ratings):
        self.users, self.items, self.labels = self.get_dataset(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['user_id'], ratings['joke_id'], ratings['Rating']))

        for u, i, r in user_item_set:
            users.append(u)
            items.append(i)
            labels.append((r + 10) / 20)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):

    def __init__(self, num_users: int, embedding: Embedding, jokes: List[str], ratings):
        super().__init__()
        self.user_embedding_1 = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=12)
        self.user_embedding_2 = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=12)

        self.embedded_jokes = torch.Tensor(embedding.to_vec(jokes))
        self.fc_red = nn.Linear(in_features=self.embedded_jokes.shape[1], out_features=12)
        self.fc1 = nn.Linear(in_features=24, out_features=64)
        self.fc1_drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=16)
        self.output = nn.Linear(in_features=28, out_features=1)
        self.ratings = ratings

    def forward(self, user_input, joke_ids):
        user_embedded = self.user_embedding_1(user_input)
        user_embedded_2 = self.user_embedding_2(user_input)
        joke_embedded = self.embedded_jokes[joke_ids - 1, :]
        joke_red = nn.ReLU()(self.fc_red(joke_embedded))

        vector = torch.cat([user_embedded, joke_red], dim=-1)
        vector2 = torch.mul(user_embedded_2, joke_red)

        vector = nn.ReLU()(self.fc1(vector))
        vector = self.fc1_drop(vector)
        vector = nn.ReLU()(self.fc2(vector))
        vector = self.fc2_drop(vector)
        vector = nn.ReLU()(self.fc3(vector))
        vector = torch.cat([vector, vector2], dim=-1)

        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, joke_input, labels = batch
        predicted_labels = self(user_input, joke_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(JokeRatingDataset(self.ratings), batch_size=512, num_workers=4)


class NCFModel(Model):
    model = None

    def __init__(self, epochs: int):
        self.epochs = epochs

    def fit(self, data, jokes):
        num_users = len(data['user_id'].unique()) + 1
        self.model = NCF(num_users, SentenceBert(), jokes, data)

        trainer = pl.Trainer(max_epochs=self.epochs, progress_bar_refresh_rate=50, logger=False,
                             checkpoint_callback=False)
        trainer.fit(self.model)

    def predict(self, data_x, jokes):
        predicted = self.model(torch.LongTensor(data_x['user_id'].to_list()),
                               torch.LongTensor(data_x['joke_id'].to_list())) * 20 - 10

        d = {'Rating': predicted.detach().numpy()[:, 0].tolist()}
        predicted = pd.DataFrame(data=d)
        return predicted
