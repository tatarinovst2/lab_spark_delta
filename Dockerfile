FROM bitnami/spark:latest

USER root

RUN apt-get update && \
    apt-get install -y cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYSPARK_PIN_THREAD=false

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "src/etl.py"]
