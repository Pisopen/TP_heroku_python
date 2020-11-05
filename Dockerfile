FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt

Expose 8000

CMD  python algo.py | python api.py