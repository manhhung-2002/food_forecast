import argparse
from pyspark.sql import SparkSession
from clean_data.base_standardize import DataFrameStandardizer


class OrderStandardizer(DataFrameStandardizer):
    def _get_text_columns(self):
        # Bảng order không có text column
        return {}

    def _get_numeric_columns(self):
        # Các cột số + default value (int)
        return {
            "total_price": 0,
            "quantity": 0
        }

    def _get_timestamp_columns(self):
        # Cột thời gian
        return ["time"]

    def _get_accent_columns(self):
        # Không có cột cần bỏ dấu
        return []

    def _get_id_columns(self):
        # order_id là ID => giữ nguyên
        return ["order_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize order table")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file (single file)")
    parser.add_argument("--input_format", default="csv", help="Input format (csv/parquet)")
    parser.add_argument("--output_format", default="csv", help="Output format (csv/parquet)")
    args = parser.parse_args()

    # Khởi tạo Spark
    spark = (
        SparkSession.builder
        .appName("OrderStandardize")
        .getOrCreate()
    )

    std = OrderStandardizer()
    std.standardize(
        spark,
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        output_format=args.output_format
    )

    spark.stop()
