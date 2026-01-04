import logging
from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Setup logger
logger = logging.getLogger(__name__)


def basic_df_profile(df: DataFrame) -> Dict:
    """Lấy thông tin cơ bản của DataFrame"""
    logger.info("Thông tin cơ bản của df")
    return {
        "num_rows": df.count(),
        "num_columns": len(df.columns),
        "columns": df.columns,
        "dtype": df.dtypes,
    }


def check_missing_percentage(df: DataFrame) -> Dict[str, float]:
    """Kiểm tra tỷ lệ missing values cho từng cột"""
    logger.info("Checking missing percentage per column")
    total_rows = df.count()
    result = {}
    for c in df.columns:
        null_cnt = df.filter(F.col(c).isNull()).count()
        result[c] = round(null_cnt / total_rows * 100, 2)
    return result


def profile_numerical_col(df: DataFrame) -> Optional[DataFrame]:
    """Phân tích các cột số (min, max, avg, std)"""
    logger.info("Checking numerical column")
    num_col = []
    for field in df.schema.fields:
        if isinstance(field.dataType, (T.IntegerType, T.LongType, T.DoubleType, T.FloatType)):
            num_col.append(field.name)

    if len(num_col) == 0:
        logger.warning("DataFrame không có cột numerical")
        return None

    agg_exp = []
    for column_name in num_col:
        agg_exp.append(F.min(column_name).alias(f"{column_name}_minimum"))
        agg_exp.append(F.max(column_name).alias(f"{column_name}_maximum"))
        agg_exp.append(F.avg(column_name).alias(f"{column_name}_avg"))
        agg_exp.append(F.stddev(column_name).alias(f"{column_name}_std"))

    numerical_profile_dataframe = df.agg(*agg_exp)
    return numerical_profile_dataframe


def profile_categorical_col(df: DataFrame) -> Optional[Dict[str, int]]:
    """Đếm số lượng giá trị unique cho các cột categorical"""
    logger.info("Checking categorical column")
    cat_col = []
    for field in df.schema.fields:
        if isinstance(field.dataType, T.StringType):
            cat_col.append(field.name)

    if len(cat_col) == 0:
        logger.warning("Dataframe không có cột categorical")
        return None

    results = {}
    for column_name in cat_col:
        results[column_name] = df.select(column_name).distinct().count()

    return results


def profile_correlations(df: DataFrame) -> Dict[str, float]:
    """Tính correlation giữa các cột số"""
    logger.info("Calculating correlations")
    numeric_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (T.IntegerType, T.LongType, T.DoubleType, T.FloatType))
    ]

    corr_results = {}
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            corr_results[f"{c1}__{c2}"] = df.stat.corr(c1, c2)

    return corr_results