from pathlib import Path
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler


def create_spark_session():
    builder = SparkSession.builder \
        .appName("Mushroom ETL Pipeline") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]")
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def run_etl():
    spark = create_spark_session()
    print("Spark session created.")

    project_root = Path(__file__).resolve().parent.parent
    bronze_path = project_root / "data" / "bronze" / "mushrooms"
    silver_path = project_root / "data" / "silver" / "mushrooms"
    raw_csv_path = project_root / "data" / "mushrooms.csv"

    bronze_path_str = str(bronze_path)
    silver_path_str = str(silver_path)
    raw_csv_path_str = str(raw_csv_path)

    print("Reading raw CSV data from:", raw_csv_path_str)
    raw_df = spark.read.format("csv").option("header", True).load(raw_csv_path_str)
    if raw_df.rdd.isEmpty():
        print("Warning: raw_df is empty!")
    else:
        print("raw_df count:", raw_df.count())

    raw_df.write.format("delta") \
        .mode("overwrite") \
        .partitionBy("class") \
        .save(bronze_path_str)
    print(f"Written Bronze table to {bronze_path_str}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS bronze_mushrooms USING DELTA LOCATION '{bronze_path_str}'")
    try:
        spark.sql("OPTIMIZE bronze_mushrooms")
        print("Optimized Bronze table successfully.")
    except Exception as e:
        print(f"Could not optimize Bronze table: {e}")

    print("Building Silver table from Bronze layer...")
    processed_df = spark.read.format("delta").load(bronze_path_str)

    processed_df = processed_df.drop("veil-type")  # Drop veil-type column as it has only one type

    numerical_columns = ["cap-diameter", "stem-height", "stem-width"]
    categorical_columns = [column for column in processed_df.columns if column not in numerical_columns + ['class']]

    print("Numerical Columns:", numerical_columns)
    print("Categorical Columns:", categorical_columns)

    for col_name in numerical_columns:
        processed_df = processed_df.withColumn(col_name, col(col_name).cast("double"))

    fill_categorical = {col_name: "unknown" for col_name in categorical_columns}
    processed_df = processed_df.fillna(fill_categorical)

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_indexed", handleInvalid="keep")
        for column in categorical_columns
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{column}_indexed" for column in categorical_columns],
        outputCols=[f"{column}_ohe" for column in categorical_columns]
    )

    assembler = VectorAssembler(
        inputCols=encoder.getOutputCols() + numerical_columns,
        outputCol="features_unscaled"
    )

    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=False)

    pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])
    model = pipeline.fit(processed_df)
    processed_df = model.transform(processed_df)

    silver_df = processed_df.select("class", "features")

    if "class" in silver_df.columns:
        silver_df = silver_df.repartition("class")

    sample = silver_df.head(1)
    if sample:
        print("Sample row from silver DataFrame:", sample)

    silver_df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("class") \
        .option("overwriteSchema", "true") \
        .save(silver_path_str)
    print(f"Written Silver table to {silver_path_str}")

    spark.sql(f"CREATE TABLE IF NOT EXISTS silver_mushrooms USING DELTA LOCATION '{silver_path_str}'")
    try:
        spark.sql("OPTIMIZE silver_mushrooms")
        print("Optimized Silver table successfully.")
    except Exception as e:
        print(f"Could not optimize Silver table: {e}")

    spark.stop()
    print("ETL pipeline completed and Spark session stopped.")


if __name__ == "__main__":
    run_etl()
