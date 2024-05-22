import pytest
from pyspark.sql import SparkSession
from script_name import clean_articulacion, clean_localizacion

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("Test ETL").getOrCreate()

def test_clean_articulacion(spark):
    data = [("rodilla"), ("rodila"), ("espalda")]
    df = spark.createDataFrame(data, StringType())
    result = clean_articulacion(df)
    expected_data = [("rodilla"), ("rodilla"), ("espalda")]
    expected_df = spark.createDataFrame(expected_data, StringType())
    assert result.collect() == expected_df.collect()

def test_clean_localizacion(spark):
    # Similar a test_clean_articulacion, pero adaptado para clean_localizacion
    pass
