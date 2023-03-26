import fire
import pandas as pd
import numpy as np
import scipy.spatial.distance as ds


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

    def warmup(self):
        self.ratings = pd.read_csv(dataset, sep='::', header=None, engine='python')

    def get_movie_name_by_id(self, movie_id):
        return self.movies_dict[movie_id]

    def train(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )

        user_movie_matrix = ratings.pivot(index='MovieID', columns='UserID', values='Rating')
        user_movie_matrix = user_movie_matrix.T
        user_movie_matrix_orig = user_movie_matrix

        user_movie_matrix = (user_movie_matrix - np.nanmean(user_movie_matrix, axis=1).reshape(-1, 1)) / np.nanstd(
            user_movie_matrix, axis=1).reshape(-1, 1)
        user_movie_matrix.fillna(0, inplace=True)

        svd_decomposition = np.linalg.svd(user_movie_matrix.to_numpy())
        u = svd_decomposition[0][:, :self.n_latent_factors]
        e = np.diag(svd_decomposition[1][:self.n_latent_factors])
        v = svd_decomposition[2][:self.n_latent_factors, :]

        ratings_predict = np.matmul(u, e)
        ratings_predict = np.matmul(ratings_predict, v)
        ratings_predict = pd.DataFrame(ratings_predict)

        ratings_predict.columns = user_movie_matrix_orig.columns
        ratings_predict.index = user_movie_matrix_orig.index

        ratings_predict = ratings_predict * np.nanstd(user_movie_matrix_orig, axis=1).reshape(-1, 1) + \
                          np.nanmean(user_movie_matrix_orig, axis=1).reshape(-1, 1)
        ratings_predict[ratings_predict > 5] = 5
        ratings_predict[ratings_predict < 1] = 1

        ratings_predict.to_csv('./model/model.csv')

    def evaluate(self, dataset):
        ratings = pd.read_csv(
            dataset, sep='::',
            header=None,
            engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        model = pd.read_csv('./model/model.csv') if self.model is None else self.model

        predicts = []
        for i in range(ratings.shape[0]):
            print(model[model['UserID'] == ratings.iloc[i]['UserID']][str(ratings.iloc[i]['MovieID'])])
            predicts.append(model[model['UserID'] == ratings.iloc[i]['UserID']][str(ratings.iloc[i]['MovieID'])])
        ratings['predict'] = predicts

        rmse = self.calculate_rmse(ratings)

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
            similarity[index] = ds.cosine(pred.iloc[index - 1, :], [v for k, v in vector.items()])

        r = {k: v for k, v in sorted(similarity.items(), key=lambda x: x[1], reverse=True)}
        p = predict.drop(labels=ids, axis=1)
        res = pd.DataFrame(p.iloc[list(r.items())[0][0] - 1]).sort_values(by=[list(r.items())[0][0]]).index
        return [list(res)[-count:], []]

    def find_similar(self, movie_id, count=5):
        movie_id = str(movie_id)
        predict = pd.read_csv('./model/model.csv') if self.model is None else self.model
        similarity = dict()
        for column in predict.columns[1:]:
            similarity[column] = self.find_vectors_similarity(list(predict[movie_id]), list(predict[column]))

        rated = list({k: v for k, v in sorted(similarity.items(), key=lambda x: x[1])}.items())[:count]
        movies_ids = [item[0] for item in rated]
        movies_names = [self.get_movie_name_by_id(int(id)) for id in movies_ids]
        return [movies_ids, movies_names]

    def calculate_rmse(self, data):
        data['difference'] = data['predict'] - data['Rating']
        rmse = (sum(data['difference']) / data.shape[0]) ** 0.5

        return rmse

    def test(self):
        return 4

    def find_vectors_similarity(self, first_vector, second_vector):
        return np.sum(np.abs(np.array(first_vector) - np.array(second_vector))) / len(first_vector)


if __name__ == '__main__':
    fire.Fire(My_Rec_Model)
