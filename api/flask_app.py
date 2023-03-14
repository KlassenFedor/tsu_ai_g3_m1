from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!2'


@app.route('/api/predict')
def hello():
    return 'Hello, World!2'


@app.route('/api/log')
def hello():
    return 'Hello, World!2'


@app.route('/api/info')
def hello():
    return 'Hello, World!2'


@app.route('/api/reload')
def hello():
    return 'Hello, World!2'


@app.route('/api/similar')
def hello():
    return 'Hello, World!2'
