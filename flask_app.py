from flask import Flask, request, make_response, jsonify
from fuzzywuzzy import fuzz
from model import My_Rec_Model
import logging
import datetime
import json

logging.basicConfig(level=logging.INFO, filename="./data/logs.log", filemode="w")
logger = logging.getLogger('model')

app = Flask(__name__)
model = My_Rec_Model()


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
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
    except Exception as e:
        logger.error(f'time: {datetime.datetime.now()}, error: {e}')
        return make_response(jsonify({'error': 'Unexpected error'}), 500)


@app.route('/api/log', methods=['GET'])
def log():
    try:
        logs = []
        with open('./data/logs.log') as file:
            for line in (file.readlines()[-20:]):
                logs.append(line)
        return logs
    except Exception as e:
        logger.error(f'time: {datetime.datetime.now()}, error: {e}')
        return make_response(jsonify({'error': 'Unexpected error'}), 500)


@app.route('/api/info', methods=['GET'])
def info():
    try:
        data = []
        with open('./info.txt') as file:
            for line in (file.readlines()):
                data.append(line)
        result = json.dumps(
            {
                'docker-build-time': data[0],
                'credentials': data[1]
            }
        )
        logger.info(f'time: {datetime.datetime.now()}, info')
    except Exception as e:
        logger.error(f'time: {datetime.datetime.now()}, error: {e}')
        return make_response(jsonify({'error': 'Unexpected error'}), 500)

    return result


@app.route('/api/reload', methods=['GET'])
def reload():
    try:
        global model
        model = My_Rec_Model()
        model.warmup()
        return 'success'
    except Exception as e:
        logger.error(f'time: {datetime.datetime.now()}, error: {e}')
        return make_response(jsonify({'error': 'Unexpected error'}), 500)


@app.route('/api/similar', methods=['POST'])
def similar():
    try:
        request_data = request.get_json()
        global model
        print(request_data)
        movie_id = find_movie_id(request_data['movie_name'], model.movies_dict)
        similar_movies = model.find_similar(movie_id)
        print(similar_movies)
        return [model.movies_dict[int(movie)] for movie in similar_movies[0]]
    except Exception as e:
        logger.error(f'time: {datetime.datetime.now()}, error: {e}')
        return make_response(jsonify({'error': 'Unexpected error'}), 500)


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
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
