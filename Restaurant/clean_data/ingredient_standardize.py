import argparse
from pyspark.sql import SparkSession
from clean_data.base_standardize import DataFrameStandardizer


class IngredientStandardizer(DataFrameStandardizer):
    def _get_text_columns(self):
        # ingredient_name là text
        return {
            "ingredient_name": "unknown"
        }

    def _get_numeric_columns(self):
        # không có cột numeric ngoài id, ingredient_id (giữ nguyên int)
        return {}

    def _get_timestamp_columns(self):
        # không có timestamp
        return []

    def _get_accent_columns(self):
        # bỏ dấu cho ingredient_name
        return ["ingredient_name"]

    def _get_id_columns(self):
        # id, ingredient_id giữ nguyên
        return ["id", "ingredient_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize ingredient table")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file (single file)")
    parser.add_argument("--input_format", default="csv", help="Input format (csv/parquet)")
    parser.add_argument("--output_format", default="csv", help="Output format (csv/parquet)")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("IngredientStandardize")
        .getOrCreate()
    )

    std = IngredientStandardizer()
    std.standardize(
        spark,
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        output_format=args.output_format
    )

    spark.stop()
