import pandas as pd
import numpy as np
from surprise.prediction_algorithms import KNNBaseline
from surprise import Dataset, Reader, accuracy
import logging
import datetime
import fire
import pickle

logging.basicConfig(level=logging.INFO, filename="./data/logs.log", filemode="w")
logger = logging.getLogger('model')


class AdvancedModel:
    def __init__(self):
        self.baseline = KNNBaseline(random_state=45)

    def train(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        ratings = ratings[['UserID', 'MovieID', 'Rating']]
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings, reader)
        train_set = data.build_full_trainset()
        self.baseline.fit(train_set)

        filename = './model/finalized_model.sav'
        pickle.dump(self.baseline, open(filename, 'wb'))

    def evaluate(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        ratings = ratings[['UserID', 'MovieID', 'Rating']]
        test_set = ratings.values.tolist()
        filename = './model/finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        baseline_predictions = loaded_model.test(test_set)

        return accuracy.rmse(baseline_predictions)


if __name__ == '__main__':
    fire.Fire(AdvancedModel)
