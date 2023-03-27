import pandas as pd
import numpy as np
from surprise.prediction_algorithms import KNNBaseline
from surprise import Dataset, Reader, accuracy
import logging
import datetime
import fire

logging.basicConfig(level=logging.INFO, filename="./data/logs.log", filemode="w")
logger = logging.getLogger('model')


class AdvancedModel:
    def __init__(self):
        self.model = None
        self.baseline = KNNBaseline(random_state=45)

    def train(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings, reader)
        trainset = data.build_full_trainset()
        self.baseline.fit(trainset)

    def evaluate(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        test_set = ratings.values.tolist()
        baseline_predictions = self.baseline.test(test_set)

        return accuracy.rmse(baseline_predictions)


if __name__ == '__main__':
    fire.Fire(AdvancedModel)
