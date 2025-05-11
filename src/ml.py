import mlflow
import mlflow.pyspark.ml
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from delta import configure_spark_with_delta_pip


def create_spark_session():
    builder = SparkSession.builder \
        .appName("Mushroom ML Pipeline") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def run_ml():
    spark = create_spark_session()
    print("Spark session created for ML training.")

    mlflow.set_tracking_uri("http://mlflow:5000")  # Update this URI as per your MLflow server
    mlflow.set_experiment("Mushroom_Classification")

    silver_path = "data/silver/mushrooms"

    print(f"Loading Silver data from {silver_path}...")
    silver_df = spark.read.format("delta").load(silver_path)
    silver_df.printSchema()
    silver_df.show(2, truncate=False)

    if silver_df.rdd.isEmpty():
        print("Error: The Silver DataFrame is empty. Exiting the ML pipeline.")
        spark.stop()
        return

    label_indexer = StringIndexer(inputCol="class", outputCol="label", handleInvalid="skip")
    label_model = label_indexer.fit(silver_df)
    ml_df = label_model.transform(silver_df)

    ml_ready_df = ml_df.select("features", "label")

    print("Prepared ML DataFrame:")
    ml_ready_df.printSchema()
    ml_ready_df.show(2, truncate=False)

    print("Splitting data into 80% training and 20% testing sets...")
    train_df, test_df = ml_ready_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training Data Count: {train_df.count()}")
    print(f"Testing Data Count: {test_df.count()}")

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.01)
    print("Initialized Logistic Regression model.")

    mlflow.pyspark.ml.autolog()

    with mlflow.start_run():
        print("Starting MLflow run and training the model...")
        model = lr.fit(train_df)
        print("Model training completed.")

        predictions = model.transform(test_df)
        print("Generated predictions on the test set.")

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        print(f"Test AUC: {auc}")

        mlflow.log_metric("test_auc", auc)

    spark.stop()
    print("ML pipeline completed and Spark session stopped.")


if __name__ == "__main__":
    run_ml()
