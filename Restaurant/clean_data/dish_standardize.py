import argparse
from pyspark.sql import SparkSession
from clean_data.base_standardize import DataFrameStandardizer


class DishStandardizer(DataFrameStandardizer):
    def _get_text_columns(self):
        # dish_name là text, default "unknown"
        return {"dish_name": "unknown"}

    def _get_numeric_columns(self):
        # price là số, default = 0
        return {
            "price": 0
        }

    def _get_timestamp_columns(self):
        # không có timestamp
        return []

    def _get_accent_columns(self):
        # dish_name cần bỏ dấu
        return ["dish_name"]

    def _get_id_columns(self):
        # dish_id là ID → giữ nguyên
        return ["dish_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize dish table")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file (single file)")
    parser.add_argument("--input_format", default="csv", help="Input format (csv/parquet)")
    parser.add_argument("--output_format", default="csv", help="Output format (csv/parquet)")
    args = parser.parse_args()

    # Khởi tạo Spark
    spark = (
        SparkSession.builder
        .appName("DishStandardize")
        .getOrCreate()
    )

    std = DishStandardizer()
    std.standardize(
        spark,
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        output_format=args.output_format
    )

    spark.stop()
