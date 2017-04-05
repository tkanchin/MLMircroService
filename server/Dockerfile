FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install -y python-pip python-dev build-essential

COPY ./app /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]

EXPOSE 8080