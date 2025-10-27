import logging
import re
import unicodedata
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


class DataFrameStandardizer(ABC):
    """
    Abstract class for standardizing Spark DataFrames:
    - text cleaning
    - numeric cleaning (cast về int/bigint)
    - timestamp formatting
    - accent removal
    - save as single cleaned file
    """

    # ================= ABSTRACT METHODS =================
    @abstractmethod
    def _get_text_columns(self) -> Dict[str, str]:
        """Define text columns with default values."""
        return {}

    @abstractmethod
    def _get_numeric_columns(self) -> Dict[str, Union[int, float]]:
        """Define numeric columns with default values."""
        return {}

    @abstractmethod
    def _get_timestamp_columns(self) -> List[str]:
        """Define timestamp columns."""
        return []

    @abstractmethod
    def _get_accent_columns(self) -> List[str]:
        """Define columns to remove accents."""
        return []

    @abstractmethod
    def _get_id_columns(self) -> List[str]:
        """Define identifier columns (giữ nguyên, không cast)."""
        return []

    # ================= DATA LOADING =================
    def load_input(self, spark: SparkSession, input_path: str, file_format: str = "csv") -> DataFrame:
        if file_format == "csv":
            df = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(input_path)
            )
        elif file_format == "parquet":
            df = spark.read.parquet(input_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        return df

    # ================= CLEANING METHODS =================
    def _process_text_columns(self, df: DataFrame) -> DataFrame:
        text_columns = self._get_text_columns()
        for column, default_value in text_columns.items():
            if column in df.columns:
                df = df.withColumn(
                    column,
                    F.when(
                        F.col(column).isNull() | (F.trim(F.col(column)) == ""),
                        F.lit(default_value)
                    ).otherwise(F.lower(F.trim(F.col(column))))
                )
        return df

    def _process_numeric_columns(self, df: DataFrame) -> DataFrame:
        numeric_columns = self._get_numeric_columns()
        id_columns = set(self._get_id_columns())

        for column, default_value in numeric_columns.items():
            if column in df.columns and column not in id_columns:
                df = df.withColumn(
                    column,
                    F.when(
                        (F.col(column).isNull()) | (F.trim(F.col(column)) == "") |
                        (F.expr(f"try_cast({column} as bigint)").isNull()),
                        F.lit(default_value)
                    ).otherwise(F.expr(f"try_cast({column} as bigint)"))
                )
        return df

    def _process_timestamp_columns(self, df: DataFrame) -> DataFrame:
        timestamp_columns = self._get_timestamp_columns()
        for column in timestamp_columns:
            if column in df.columns:
                df = df.withColumn(column, F.to_timestamp(F.col(column), "yyyy-MM-dd HH:mm:ss"))
                df = df.withColumn(f"{column}_date", F.to_date(F.col(column)))
                df = df.withColumn(f"{column}_hour", F.hour(F.col(column)))
        return df

    @staticmethod
    def normalize_text_value(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        text = unicodedata.normalize("NFD", text)
        text = re.sub(r"[\u0300-\u036f]", "", text)  # remove accents
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
        text = re.sub(r"\s+", " ", text)  # collapse spaces
        return text.strip().lower()

    def _remove_accents(self, df: DataFrame) -> DataFrame:
        accent_columns = self._get_accent_columns()
        if not accent_columns:
            return df

        normalize_udf = F.udf(self.normalize_text_value, StringType())
        for column in accent_columns:
            if column in df.columns:
                df = df.withColumn(column, normalize_udf(F.col(column)))
        return df

    # ================= SAVE RESULT =================
    def _save_result(self, df: DataFrame, output_path: str, file_format: str = "csv"):
        """
        Save DataFrame as a single file (e.g. order.csv) instead of Spark folder.
        """
        dir_name = os.path.dirname(output_path)
        os.makedirs(dir_name, exist_ok=True)
        tmp_path = os.path.join(dir_name, f"_tmp_output_{uuid.uuid4().hex}")

        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        if file_format == "csv":
            (
                df.coalesce(1)
                .write.mode("overwrite")
                .option("header", True)
                .csv(tmp_path)
            )
            part_file = None
            for f in os.listdir(tmp_path):
                if f.startswith("part-") and f.endswith(".csv"):
                    part_file = os.path.join(tmp_path, f)
                    break
            if part_file is None:
                raise RuntimeError("❌ Không tìm thấy file part-*.csv sau khi ghi Spark")

            shutil.copy(part_file, output_path)
            shutil.rmtree(tmp_path)

        elif file_format == "parquet":
            (
                df.coalesce(1)
                .write.mode("overwrite")
                .parquet(tmp_path)
            )
            part_file = None
            for f in os.listdir(tmp_path):
                if f.startswith("part-") and f.endswith(".parquet"):
                    part_file = os.path.join(tmp_path, f)
                    break
            if part_file is None:
                raise RuntimeError("❌ Không tìm thấy file part-*.parquet sau khi ghi Spark")

            shutil.copy(part_file, output_path)
            shutil.rmtree(tmp_path)

        else:
            raise ValueError(f"Unsupported format: {file_format}")

    # ================= MAIN PIPELINE =================
    def standardize(self, spark: SparkSession, input_path: str, output_path: str,
                    input_format: str = "csv", output_format: str = "csv") -> DataFrame:
        try:
            df = self.load_input(spark, input_path, input_format)
            df = df.dropDuplicates()
            df = self._process_text_columns(df)
            df = self._process_numeric_columns(df)
            df = self._process_timestamp_columns(df)
            df = self._remove_accents(df)

            df.show(10, truncate=False)
            self._save_result(df, output_path, output_format)
            print(f"✅ Standardization finished. Output saved to: {output_path}")
            return df
        except Exception as e:
            logging.error(f"Standardization failed: {e}")
            raise
