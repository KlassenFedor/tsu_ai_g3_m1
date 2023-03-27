import fire
import pandas as pd
import numpy as np
import logging
import datetime
logging.basicConfig(level=logging.INFO, filename="./data/logs.log", filemode="w")
logger = logging.getLogger('model')


class My_Rec_Model:
    def __init__(self):
        self.model = None
        self.ratings = None
        self.n_latent_factors = 20
        self.movies_dict = pd.read_csv(
            './dataset/movies.dat',
            sep='::',
            header=None,
            engine='python',
            encoding="ISO-8859-1",
            names=['MovieID', 'Title', 'Genres']
        )
        self.movies_dict = dict(zip(self.movies_dict['MovieID'], self.movies_dict['Title']))
        self.warmup()

    def warmup(self):
        self.model = pd.read_csv('./model/model.csv', index_col='UserID')
        logger.info(f'time: {datetime.datetime.now()}, warmup')

    def __get_movie_name_by_id(self, movie_id):
        return self.movies_dict[movie_id]

    def train(self, dataset):
        logger.info(f'time: {datetime.datetime.now()}, training started')

        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )

        # creating matrix for all users and movies in train set
        user_movie_matrix = ratings.pivot(index='MovieID', columns='UserID', values='Rating')
        user_movie_matrix = user_movie_matrix.T
        user_movie_matrix_orig = user_movie_matrix

        # matrix normalization
        user_movie_matrix = (user_movie_matrix - np.nanmean(user_movie_matrix, axis=1).reshape(-1, 1)) / np.nanstd(
            user_movie_matrix, axis=1).reshape(-1, 1)
        user_movie_matrix.fillna(0, inplace=True)

        # svd algorithm using numpy
        svd_decomposition = np.linalg.svd(user_movie_matrix.to_numpy())
        u = svd_decomposition[0][:, :self.n_latent_factors]
        e = np.diag(svd_decomposition[1][:self.n_latent_factors])
        v = svd_decomposition[2][:self.n_latent_factors, :]

        # creating predictions matrix
        ratings_predict = np.matmul(u, e)
        ratings_predict = np.matmul(ratings_predict, v)
        ratings_predict = pd.DataFrame(ratings_predict)
        ratings_predict.columns = user_movie_matrix_orig.columns
        ratings_predict.index = user_movie_matrix_orig.index

        # returning data to its normal form
        ratings_predict = ratings_predict * np.nanstd(user_movie_matrix_orig, axis=1).reshape(-1, 1) + \
                          np.nanmean(user_movie_matrix_orig, axis=1).reshape(-1, 1)
        ratings_predict[ratings_predict > 5] = 5
        ratings_predict[ratings_predict < 1] = 1

        # saving matrix
        ratings_predict.to_csv('./model/model.csv')

        logger.info(f'time: {datetime.datetime.now()}, training ended')

    def evaluate(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        model = pd.read_csv('./model/model.csv') if self.model is None else self.model

        # receiving predicts for data in test set
        predicts = []
        for i in range(ratings.shape[0]):
            predicts.append(model[model['UserID'] == ratings.iloc[i]['UserID']][str(ratings.iloc[i]['MovieID'])])
        ratings['predict'] = predicts

        # rmse calculation
        rmse = self.__calculate_rmse(ratings)

        logger.info(f'time: {datetime.datetime.now()}, evaluated with rmse: {rmse}')

        return rmse

    def predict(self, movies_ratings, count=5):
        logger.info(f'time: {datetime.datetime.now()}, prediction started')

        # data preparation
        movies_ids = [str(item) for item in movies_ratings[0]]
        ratings = movies_ratings[1]
        self.warmup()
        predicted_ratings = self.model

        # search most similar users
        predicted_ratings['similarity'] = [
            self.__find_vectors_similarity(
                predicted_ratings.loc[i][movies_ids],
                ratings
            ) for i in predicted_ratings.index
        ]
        predicted_ratings = predicted_ratings.sort_values(by='similarity').iloc[:3]
        predicted_ratings = predicted_ratings.drop(columns=movies_ids)

        # movies ratings computations and sorting
        mean_ratings = predicted_ratings.mean(axis=0).to_dict()
        best_movies = list(
            {
                k: v for k, v in sorted(mean_ratings.items(), key=lambda x: x[1], reverse=True)
            }.items()
        )[:count]

        logger.info(f'time: {datetime.datetime.now()}, prediction ended')

        return [[item[0] for item in best_movies], [item[1] for item in best_movies]]

    def find_similar(self, movie_id, count=5):
        logger.info(f'time: {datetime.datetime.now()}, similar searching started')

        # data preparation
        movie_id = str(movie_id)
        predict = pd.read_csv('./model/model.csv') if self.model is None else self.model
        similarity = dict()

        # computations for movies similarity
        for column in predict.columns[1:]:
            similarity[column] = self.__find_vectors_similarity(list(predict[movie_id]), list(predict[column]))

        # movies similarity sorting
        rated = list({k: v for k, v in sorted(similarity.items(), key=lambda x: x[1])}.items())[:count]
        movies_ids = [item[0] for item in rated]
        movies_names = [self.__get_movie_name_by_id(int(movie_id)) for movie_id in movies_ids]

        logger.info(f'time: {datetime.datetime.now()}, similar searching ended')

        return [movies_ids, movies_names]

    def __calculate_rmse(self, data):
        data['difference'] = data['predict'] - data['Rating']
        rmse = (sum(data['difference']) / data.shape[0]) ** 0.5

        return rmse

    def __find_vectors_similarity(self, first_vector, second_vector):
        return np.sum(np.abs(np.array(first_vector) - np.array(second_vector))) / len(first_vector)

    def __get_mean_rating_for_users(self, user_ids, movie_id):
        ratings = self.model.iloc[user_ids][str(movie_id)]
        return np.mean(ratings)


if __name__ == '__main__':
    fire.Fire(My_Rec_Model)
