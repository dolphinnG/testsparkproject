import logging
import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from mlflow.models import infer_signature
from pyspark.sql.functions import col
from pyspark.ml.functions import array_to_vector
import mlflow.pyfunc
import pandas as pd

logger = logging.getLogger(__name__)

from another_file import AnotherClass
a = AnotherClass()
print(a.another_method())

def main():
    print(a.another_method())
    mlflow.set_experiment("SPARK-TEST-5")
    
    logger.log(logging.INFO, "Starting the spark session")

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("gggg wwwww") \
        .getOrCreate()
        # .config("spark.driver.host", os.environ.get("POD_IP")) \
        # .config("spark.driver.host", 'host.docker.internal') \

    # Create a pandas DataFrame
    pandas_df = pd.DataFrame({
        "features": [[3.0, 4.0], [5.0, 6.0]],
        "label": [0, 1]
    })

    # Convert pandas DataFrame to Spark DataFrame
    train_df = spark.createDataFrame(pandas_df).select(array_to_vector("features").alias("features"), col("label"))

    lor = LogisticRegression(maxIter=2)
    lor.setPredictionCol("").setProbabilityCol("prediction")
    lor_model = lor.fit(train_df)

    # Log parameters
    mlflow.log_param("maxIter", lor.getMaxIter())
    mlflow.log_param("regParam", lor.getRegParam())

    test_df = train_df.select("features")
    prediction_df = lor_model.transform(train_df)
    prediction_df.printSchema() # prediction result is a vector of probabilities of the label classes summing up to 1
    
    res = prediction_df.select("prediction")

    # Log metrics
    accuracy = res.filter(res["prediction"] == train_df["label"]).count() / float(train_df.count())
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(test_df, prediction_df)

    with mlflow.start_run() as run:
        model_info = mlflow.spark.log_model(
            lor_model,
            "model",
            # signature=signature,
            dfs_tmpdir="/opt/bitnami/spark/tmp/" # MUST
        )
        # Register the model
        mlflow.register_model(model_info.model_uri, "LogisticRegressionModel")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()

