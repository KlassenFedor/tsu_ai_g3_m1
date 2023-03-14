import fire
import pandas as pd
import numpy as np
import logging


class My_Rec_Model:
    def __init__(self):
        self.model = None

    def train(self, dataset):
        ratings = pd.read_csv(dataset, sep='::', header=None, engine='python')
        ratings = ratings.rename(columns={
            0: 'UserID',
            1: 'MovieID',
            2: 'Rating',
            3: 'Timestamp'
        })
        user_movie_matrix = ratings.pivot(index='MovieID', columns='UserID', values='Rating')
        user_movie_matrix = user_movie_matrix.T
        user_movie_matrix = user_movie_matrix.apply(
            lambda x: x - x.mean()
        )
        user_movie_matrix.fillna(0, inplace=True)
        svd_decomposition = np.linalg.svd(user_movie_matrix.to_numpy())
        sum(svd_decomposition[1][:500] ** 2) / sum(svd_decomposition[1][:] ** 2)
        factors_number = 500
        u = svd_decomposition[0][:, :factors_number]
        e = np.diag(svd_decomposition[1][:factors_number])
        v = svd_decomposition[2][:factors_number, :]
        r = np.matmul(u, v)
        pd.to_excel('./model/model.xlsx')

    def evaluate(self, dataset):
        ratings = pd.read_csv(dataset, sep='::', header=None, engine='python')
        model = pd.read_csv('./model/model.xlsx')
        ratings = ratings.rename(columns={
            0: 'UserID',
            1: 'MovieID',
            2: 'Rating',
            3: 'Timestamp'
        })
        user_movie_matrix = ratings.pivot(index='MovieID', columns='UserID', values='Rating')
        user_movie_matrix = user_movie_matrix.T
        user_movie_matrix = user_movie_matrix.apply(
            lambda x: x - x.mean()
        )
        user_movie_matrix.fillna(0, inplace=True)

        pass

    def predict(self, movies_ratings, count=5):
        pass

    def warmup(self):
        if self.model is None:
            self.model = pd.read_csv('./model/model.xlsx')

    def find_similar(self, movie_id, count=5):
        pass


if __name__ == '__main__':
    fire.Fire(My_Rec_Model)
