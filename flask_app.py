from flask import Flask, request
from fuzzywuzzy import fuzz
from model import My_Rec_Model
import json

app = Flask(__name__)
model = My_Rec_Model()


@app.route('/api/predict')
def predict():
    request_data = request.get_json()
    print(request_data)
    movies = list(request_data[0])
    ratings = list(request_data[1])
    result = model.predict(
        [
            [find_movie_id(movie, model.movies_dict) for movie in movies],
            ratings
        ],
        5
    )
    return [[model.movies_dict[int(movie)] for movie in result[0]], result[1]]


@app.route('/api/log')
def log():
    logs = []
    with open('./logging/logs.log') as file:
        for line in (file.readlines()[-20:]):
            logs.append(line)
    return logs


@app.route('/api/info')
def info():
    credentials = 'Klassen Fedor, 3rd grade'
    logs = ''
    with open('./logging/logs.log') as file:
        for line in (file.readlines()[-20:]):
            logs += line
    result = json.dumps(
        {
            'credentials': credentials
        }
    )
    return result


@app.route('/api/reload')
def reload():
    global model
    model = My_Rec_Model()


@app.route('/api/similar')
def similar():
    request_data = request.get_json()
    global model
    print(request_data)
    movie_id = find_movie_id(request_data['movie_name'], model.movies_dict)
    similar_movies = model.find_similar(movie_id)
    print(similar_movies)
    return [model.movies_dict[int(movie)] for movie in similar_movies[0]]


def find_movie_id(name, movies_dict):
    max_estimate = 0
    movie_id = 1
    for k, v in movies_dict.items():
        current_estimate = fuzz.WRatio(name, v)
        if current_estimate > max_estimate:
            max_estimate = current_estimate
            movie_id = k
    return movie_id


if __name__ == '__main__':
    app.run(host='0.0.0.0')
