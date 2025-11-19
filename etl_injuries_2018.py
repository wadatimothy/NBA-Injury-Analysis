import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    upper,
    lower,
    when,
    regexp_extract,
    regexp_replace,
    coalesce,
    to_date,
    year,
    month,
    dayofweek,
    lpad,
    concat_ws,
    sha2,
    lit,
)
from pyspark.sql.types import IntegerType, StringType


def build_spark(app="NBA Injuries Clean 2018", region="us-west-1"):
    return (
        SparkSession.builder.appName(app)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
        .getOrCreate()
    )


def normalize_cols(df):
    # Trim and standardize common columns that often appear in Kaggle injury CSVs
    cols = {c: trim(col(c)).alias(c) for c in df.columns}
    df = df.select(*cols.values())

    # Standardize column names if variants exist
    rename_map = {
        "player_name": "Player_Name",
        "player": "Player_Name",
        "team_abbrev": "Team",
        "team_code": "Team",
        "injury": "Injury_Type",
        "injurytype": "Injury_Type",
        "notes": "Notes",
        "date": "Date",
        "year": "Year",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.withColumnRenamed(src, dst)

    # Ensure required columns exist
    for req in ["Player_Name", "Team", "Date", "Injury_Type", "Notes"]:
        if req not in df.columns:
            df = df.withColumn(req, lit(None).cast(StringType()))

    # Clean date, year
    # Try multiple patterns then coalesce
    df = df.withColumn(
        "Date",
        coalesce(
            to_date(col("Date"), "yyyy-MM-dd"),
            to_date(col("Date"), "MM/dd/yyyy"),
            to_date(col("Date"), "dd/MM/yyyy"),
        ),
    )

    # If Year column missing, derive from Date
    if "Year" not in df.columns:
        df = df.withColumn("Year", year(col("Date")).cast(StringType()))
    else:
        df = df.withColumn("Year", regexp_replace(col("Year"), r"\.0$", ""))

    # Basic cleanup on strings
    df = df.withColumn("Player_Name", regexp_replace(col("Player_Name"), r"\s+", " "))
    df = df.withColumn("Team", upper(col("Team")))
    df = df.withColumn("Injury_Type", regexp_replace(col("Injury_Type"), r"\s+", " "))
    df = df.withColumn("Notes", col("Notes"))
    return df


def derive_injury_region_and_severity(df):
    text = lower(concat_ws(" ", col("Injury_Type"), col("Notes")))

    # Injury region taxonomy (simple regex buckets)
    df = df.withColumn(
        "injury_region",
        when(text.rlike(r"\b(concussion|head|skull)\b"), "HEAD")
        .when(text.rlike(r"\b(neck|cervical)\b"), "NECK")
        .when(text.rlike(r"\b(shoulder|rotator cuff|clavicle)\b"), "SHOULDER")
        .when(text.rlike(r"\b(elbow|ulna|humerus)\b"), "ELBOW")
        .when(text.rlike(r"\b(wrist|carpal)\b"), "WRIST")
        .when(text.rlike(r"\b(hand|thumb|finger|metacarp)\b"), "HAND")
        .when(text.rlike(r"\b(chest|rib|sternum)\b"), "CHEST")
        .when(text.rlike(r"\b(back|lumbar|spine)\b"), "BACK")
        .when(text.rlike(r"\b(hip|pelvis)\b"), "HIP")
        .when(text.rlike(r"\b(groin|adductor)\b"), "GROIN")
        .when(text.rlike(r"\b(hamstring)\b"), "HAMSTRING")
        .when(text.rlike(r"\b(quad|quadricep)\b"), "QUAD")
        .when(text.rlike(r"\b(calf|achilles)\b"), "CALF")
        .when(text.rlike(r"\b(knee|acl|mcl|meniscus|patella)\b"), "KNEE")
        .when(text.rlike(r"\b(ankle)\b"), "ANKLE")
        .when(text.rlike(r"\b(foot|toe|plantar)\b"), "FOOT")
        .otherwise("OTHER"),
    )

    # Severity heuristic
    df = df.withColumn(
        "injury_severity",
        when(
            text.rlike(r"\b(surgery|repair|fracture|broken|rupture|torn|tear)\b"),
            "HIGH",
        )
        .when(
            text.rlike(r"\b(grade\s*2|grade\s*iii|partial tear|dislocation)\b"),
            "MEDIUM",
        )
        .when(
            text.rlike(r"\b(sprain|strain|soreness|contusion|bruise|tightness)\b"),
            "LOW",
        )
        .otherwise("UNKNOWN"),
    )

    # Injury category summary (type-level rollup)
    df = df.withColumn(
        "injury_category",
        when(text.rlike(r"\b(concussion)\b"), "CONCUSSION")
        .when(text.rlike(r"\b(fracture|broken)\b"), "FRACTURE")
        .when(text.rlike(r"\b(sprain|strain)\b"), "SPRAIN_STRAIN")
        .when(text.rlike(r"\b(tear|torn|rupture)\b"), "TEAR_RUPTURE")
        .when(text.rlike(r"\b(contusion|bruise)\b"), "CONTUSION")
        .otherwise("OTHER"),
    )
    return df


def build_dims_and_fact(df):
    # Filter to 2018 only
    df18 = df.filter(col("Year") == "2018").filter(col("Date").isNotNull())

    # Surrogate keys via stable hashes
    dim_player = (
        df18.select("Player_Name")
        .dropDuplicates()
        .withColumn("player_key", sha2(upper(col("Player_Name")), 256))
    )
    dim_team = (
        df18.select("Team")
        .dropDuplicates()
        .withColumn("team_key", sha2(col("Team"), 256))
    )

    dim_injury = (
        df18.select("injury_category", "injury_region", "injury_severity")
        .fillna(
            {
                "injury_category": "OTHER",
                "injury_region": "OTHER",
                "injury_severity": "UNKNOWN",
            }
        )
        .dropDuplicates()
        .withColumn(
            "injury_key",
            sha2(
                concat_ws(
                    "|",
                    col("injury_category"),
                    col("injury_region"),
                    col("injury_severity"),
                ),
                256,
            ),
        )
    )

    # Date dimension: YYYYMMDD key
    dim_date = (
        df18.select("Date")
        .dropDuplicates()
        .withColumn("year", year(col("Date")).cast(IntegerType()))
        .withColumn("month", month(col("Date")).cast(IntegerType()))
        .withColumn("day_of_week", dayofweek(col("Date")).cast(IntegerType()))
        .withColumn(
            "date_key",
            concat_ws(
                "",
                col("year"),
                lpad(col("month").cast(StringType()), 2, "0"),
                lpad(col("Date").cast("date").substr(9, 2), 2, "0"),
            ),
        )
        .withColumn("date_key", col("date_key").cast(IntegerType()))
        .select("date_key", "Date", "year", "month", "day_of_week")
    )

    # Join to create fact
    fact = (
        df18.join(dim_player, "Player_Name", "left")
        .join(dim_team, "Team", "left")
        .join(
            dim_injury, ["injury_category", "injury_region", "injury_severity"], "left"
        )
        .join(dim_date, "Date", "left")
        .withColumn(
            "fact_injury_key",
            sha2(
                concat_ws(
                    "|",
                    col("player_key"),
                    col("team_key"),
                    col("injury_key"),
                    col("date_key").cast(StringType()),
                    coalesce(col("Notes"), lit("")),
                ),
                256,
            ),
        )
        .select(
            "fact_injury_key",
            "player_key",
            "team_key",
            "injury_key",
            "date_key",
            "Player_Name",
            "Team",
            "injury_category",
            "injury_region",
            "injury_severity",
            "Injury_Type",
            "Notes",
        )
    )

    return dim_player, dim_team, dim_injury, dim_date, fact


def write_parquet(df, path, partitions=None, mode="overwrite"):
    if partitions:
        df.write.mode(mode).partitionBy(*partitions).parquet(path)
    else:
        df.write.mode(mode).parquet(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-injuries", required=True, help="s3a path to raw injuries CSV(s)"
    )
    p.add_argument(
        "--clean-prefix",
        required=True,
        help="s3a prefix for clean outputs, e.g. s3a://bucket/clean/nba/",
    )
    p.add_argument("--region", default="us-west-1")
    args = p.parse_args()

    spark = build_spark(region=args.region)

    # Read raw injuries
    injuries = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(args.raw_injuries)
    )

    injuries = normalize_cols(injuries)
    injuries = derive_injury_region_and_severity(injuries)

    dim_player, dim_team, dim_injury, dim_date, fact = build_dims_and_fact(injuries)

    # Write clean zone (Parquet). Partition fact by team and year for efficient pruning.
    base = args.clean_prefix.rstrip("/")
    write_parquet(dim_player.select("player_key", "Player_Name"), f"{base}/dim_player")
    write_parquet(dim_team.select("team_key", "Team"), f"{base}/dim_team")
    write_parquet(
        dim_injury.select(
            "injury_key", "injury_category", "injury_region", "injury_severity"
        ),
        f"{base}/dim_injury",
    )
    write_parquet(dim_date, f"{base}/dim_date")
    # Add a Year column to help partitioning
    fact_out = fact.withColumn("Year", lit(2018).cast(IntegerType()))
    write_parquet(fact_out, f"{base}/fact_injury", partitions=["Team", "Year"])

    print("Wrote clean tables to:", base)


if __name__ == "__main__":
    main()
