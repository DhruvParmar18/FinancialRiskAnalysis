FROM python:3.12-rc-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .

RUN mkdir models

CMD ["python", "train.py"]