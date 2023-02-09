FROM python:3.10-slim

WORKDIR /app
ADD . /app

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y
RUN apt-get install -y ffmpeg
RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
