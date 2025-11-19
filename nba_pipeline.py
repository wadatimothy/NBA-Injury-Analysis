from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.dynamodb import DynamoDBHook
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import json
from decimal import Decimal


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


# nba_injury_data, nba_pbp_2018, nba_stats_2018


def extract_from_dynamodb(table_name, tmp_filename):
    hook = DynamoDBHook(aws_conn_id="aws_conn")
    table = hook.get_conn().Table(table_name)

    response = table.scan()
    items = response["Items"]

    # Convert Decimals to float for CSV
    for item in items:
        for k, v in item.items():
            if isinstance(v, Decimal):
                item[k] = float(v)

    # Convert to DataFrame
    df = pd.DataFrame(items)

    # Save as CSV
    df.to_csv(f"/Users/timmywada/data/{tmp_filename}.csv", index=False)


def transform_data():
    spark = SparkSession.builder.appName("NBAInjuryTransform").getOrCreate()
    injuries = spark.read.option("header", True).csv(
        "/Users/timmywada/data/nba_injuries.csv"
    )
    pbp = spark.read.option("header", True).csv(
        "/Users/timmywada/data/nba_stats_2018.csv"
    )
    games = spark.read.option("header", True).csv(
        "/Users/timmywada/data/nba_pbp_2018.csv"
    )

    # ---------------------------------------------------------
    # 3. CLEAN INJURIES DATASET
    # ---------------------------------------------------------
    # Goal: Standardize player name → FirstInitial. Lastname

    def std_name(full_name):
        # Example: "LeBron James" → "L. James"
        if full_name is None:
            return None
        parts = full_name.split(" ")
        first, last = parts[0], parts[-1]
        return f"{first[0]}. {last}"

    std_name_udf = udf(std_name, StringType())

    injuries_clean = injuries.withColumn(
        "playerNameFormatted", std_name_udf(col("player_name"))
    ).select("playerNameFormatted", "Team", "Notes", "Injury_Type", "date")

    injuries_clean = injuries_clean.filter(
        (col("date") >= "2018-10-01") & (col("date") <= "2019-05-01")
    )

    injuries_clean.write.mode("overwrite").parquet("output/injuries_2018_clean.parquet")
    # ---------------------------------------------------------
    # 4. CLEAN PLAY-BY-PLAY
    # ---------------------------------------------------------
    # Keep only needed columns
    pbp_cols_to_keep = [
        "gameId",
        "playerNameI",
        "teamId",
        "teamTricode",
        "period",
        "clock",
        "actionType",
        "description",
        "shotResult",
        "pointsTotal",
    ]

    pbp_clean = pbp.select(*pbp_cols_to_keep)

    # Filter only events for injured players
    injured_players = injuries_clean.select("playerNameFormatted").distinct()

    pbp_filtered = pbp_clean.join(
        injured_players,
        pbp_clean.playerNameI == injured_players.playerNameFormatted,
        "inner",
    )

    # ---------------------------------------------------------
    # 5. CLEAN GAME METADATA
    # ---------------------------------------------------------
    games_clean = games.select(col("GAMEID").alias("gameId"), "gameDate").withColumn(
        "gameDate", to_date(col("gameDate"), "yyyy-MM-dd")
    )

    # ---------------------------------------------------------
    # 6. JOIN ALL PBP DATASETS
    # ---------------------------------------------------------
    final_df = pbp_filtered.join(games_clean, "gameId", "left")

    # ---------------------------------------------------------
    # 7. WRITE FINAL CLEANED OUTPUT
    # ---------------------------------------------------------
    final_df.write.mode("overwrite").parquet("output/nba_analysis_ready.parquet")


with DAG(
    "nba_dynamodb_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    extract_injuries = PythonOperator(
        task_id="extract_injuries",
        python_callable=extract_from_dynamodb,
        op_kwargs={
            "table_name": "nba_injury_data",
            "tmp_filename": "nba_injuries",
        },
    )

    extract_playbyplay = PythonOperator(
        task_id="extract_playbyplay",
        python_callable=extract_from_dynamodb,
        op_kwargs={"table_name": "nba_stats_2018", "tmp_filename": "nba_stats_2018"},
    )

    extract_dates = PythonOperator(
        task_id="extract_dates",
        python_callable=extract_from_dynamodb,
        op_kwargs={
            "table_name": "nba_pbp_2018",
            "tmp_filename": "nba_pbp_2018",
        },
    )

    transform_task = PythonOperator(
        task_id="transform_data", python_callable=transform_data
    )

    [extract_injuries, extract_playbyplay, extract_dates] >> transform_task
