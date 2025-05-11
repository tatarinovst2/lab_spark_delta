#!/bin/bash
set -e

echo "Running ETL Pipeline..."
docker-compose run --rm --build spark-app python src/etl.py

echo "Running ML Training Pipeline..."
docker-compose run --rm --build spark-app python src/ml.py

echo "Pipeline executed successfully."
