import fire
import pandas as pd
import numpy as np


class My_Rec_Model:
    def __init__(self):
        self.model = None
        self.movies_dict = dict()
        movies = pd.read_csv('./dataset/movies.dat', sep='::', header=None, engine='python', encoding="ISO-8859-1")
        movies = movies.rename(columns={
            0: 'MovieID',
            1: 'Title',
            2: 'Genres'
        })
        for i in range(movies.shape[0]):
            self.movies_dict[movies.iloc[i]['MovieID']] = movies.iloc[i]['Title']

    def get_movie_name_by_id(self, movie_id):
        return self.movies_dict[movie_id]

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
        latent_factors = 20
        user_bias = np.zeros(user_movie_matrix.shape[0])
        movie_bias = np.zeros(user_movie_matrix.shape[1])
        common_bias = np.mean(np.mean(user_movie_matrix))
        user_movie_matrix.fillna(0, inplace=True)
        users_factors = np.random.normal(scale=1. / latent_factors, size=(user_movie_matrix.shape[0], latent_factors))
        movies_factors = np.random.normal(scale=1. / latent_factors, size=(user_movie_matrix.shape[1], latent_factors))
        reg = 0.01
        learning_rate = 0.1
        counter = 1
        iter_num = 10
        sample_row, sample_col = user_movie_matrix.to_numpy().nonzero()

        while counter <= iter_num:
            training_indices = np.arange(user_movie_matrix.shape[0])
            np.random.shuffle(training_indices)

            for index in training_indices:
                u = sample_row[index]
                i = sample_col[index]

                bias = common_bias + user_bias[u] + movie_bias[i]
                prediction = users_factors[u, :].dot(movies_factors[i, :].T) + bias

                e = (user_movie_matrix.iloc[u, i] - prediction)

                user_bias[u] += learning_rate * (e - reg * user_bias[u])
                movie_bias[i] += learning_rate * (e - reg * movie_bias[i])

                users_factors[u, :] += learning_rate * (e * movies_factors[i, :] - reg * users_factors[u, :])
                movies_factors[i, :] += learning_rate * (e * users_factors[u, :] - reg * movies_factors[i, :])

            counter += 1

        predictions = np.zeros((users_factors.shape[0], movies_factors.shape[0]))
        for u in range(users_factors.shape[0]):
            for i in range(movies_factors.shape[0]):
                bias = common_bias + user_bias[u] + movie_bias[i]
                predictions[u, i] = users_factors[u, :].dot(movies_factors[i, :].T) + bias
                if np.any(np.isnan(predictions[u, i])):
                    predictions[u, i] = 0

        predict = pd.DataFrame(predictions)
        predict.index = user_movie_matrix.index
        predict.columns = user_movie_matrix.columns
        predict.to_csv('./model/model.csv')

    def evaluate(self, dataset):
        ratings = pd.read_csv(dataset, sep='::', header=None, engine='python')
        model = pd.read_csv('./model/model.csv') if self.model is None else self.model
        ratings = ratings.rename(columns={
            0: 'UserID',
            1: 'MovieID',
            2: 'Rating',
            3: 'Timestamp'
        })
        rmse = 0
        for i in range(ratings.shape[0]):
            user_id = ratings.iloc[i]['MovieID']
            movie_id = ratings.iloc[i]['MovieID']
            rating = ratings.iloc[i]['Rating']
            rmse += (model.loc[user_id][movie_id] - rating) ** 2
        rmse = (rmse / ratings.shape[0]) ** 0.5
        return rmse

    def predict(self, movies_ratings, count=5):
        ids = movies_ratings[0]
        ratings = movies_ratings[1]
        vector = dict()
        for i in range(len(ids)):
            vector[ids[i]] = ratings[i]
        vector = {k: v for k, v in sorted(vector.items(), key=lambda x: x[1])}

        similarity = dict()
        predict = pd.read_csv('./model/model.csv') if self.model is None else self.model
        pred = predict[ids]
        for index in predict.index:
            similarity[index] = (
                    np.dot(pred.iloc[index - 1, :], [v for k, v in vector.items()]) /
                    (np.linalg.norm(pred.iloc[index - 1, :]) * np.linalg.norm([v for k, v in vector.items()]))
            )
        r = {k: v for k, v in sorted(similarity.items(), key=lambda x: x[1], reverse=True)}
        p = predict.drop(labels=ids, axis=1)
        res = pd.DataFrame(p.iloc[list(r.items())[0][0] - 1]).sort_values(by=[list(r.items())[0][0]]).index
        return [list(res)[-count:], []]

    def warmup(self):
        if self.model is None:
            self.model = pd.read_csv('./model/model.csv')

    def find_similar(self, movie_id, count=5):
        predict = pd.read_csv('./model/model.csv') if self.model is None else self.model
        similarity = dict()
        for column in predict.columns:
            similarity[column] = (
                    np.dot(predict.iloc[:][movie_id], predict.iloc[:][column]) /
                    (np.linalg.norm(predict.iloc[:][movie_id]) * np.linalg.norm(predict.iloc[:][column]))
            )

        rated = list({k: v for k, v in sorted(similarity.items(), key=lambda x: x[1])}.items())[-count:]
        movies_ids = [item[0] for item in rated]
        movies_names = [self.get_movie_name_by_id(id) for id in movies_ids]
        return [movies_ids, movies_names]


if __name__ == '__main__':
    fire.Fire(My_Rec_Model)
