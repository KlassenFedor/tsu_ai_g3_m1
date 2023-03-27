FROM python:3.10

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y python3-pip

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

RUN mkdir -p /app/data
RUN mkdir -p /app/model

RUN echo "$(date)" > ./info.txt
RUN echo "Klassen Fedor, HITS 3rd grade" >> ./info.txt

CMD ["python3", "/app/flask_app.py"]