from pyspark.sql import SparkSession
import os, boto3

REGION  = "us-west-1"
PROFILE = os.environ.get("AWS_PROFILE", "default")

print("ENV AWS_PROFILE=", PROFILE)
print("ENV AWS_REGION =", os.environ.get("AWS_REGION"))

# Prove the Python side sees the right account/region before Spark starts
sts = boto3.client("sts", region_name=REGION)
print("STS identity:", sts.get_caller_identity())
ddb = boto3.client("dynamodb", region_name=REGION)
print("DDB tables:", ddb.list_tables()["TableNames"])

spark = (
    SparkSession.builder
    .appName("NBA_Injury_Analysis")
    .config(
        "spark.jars.packages",
        "com.audienceproject:spark-dynamodb_2.12:1.1.2,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.775"
    )
    # Make the Java (SDK v1) side use your profile + region explicitly
    .config("spark.driver.extraJavaOptions", f"-Daws.profile={PROFILE} -Daws.region={REGION}")
    .config("spark.executor.extraJavaOptions", f"-Daws.profile={PROFILE} -Daws.region={REGION}")
    # Belt-and-suspenders: set both keys some connectors read
    .config("spark.dynamodb.region", REGION)
    .config("spark.hadoop.dynamodb.region", REGION)
    .getOrCreate()
)

spark.range(3).show()

# Try reading the table; exact name and case
df = (
    spark.read
    .format("dynamodb")
    .option("tableName", "nba_injury_data")
    # Some builds honor an explicit region option; harmless if ignored
    .option("region", REGION)
    .load()
)

df.printSchema()
df.show(5, truncate=False)
