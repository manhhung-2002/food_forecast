import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List
from pyspark.sql import Window
import os
import argparse

# ========================= LOGGER =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeatureStore")


# ========================= HÀM CLEAN =========================
def build_clean_dataset(order_path: str, order_detail_path: str, dish_path: str,
                        spark: SparkSession = None) -> DataFrame:
    """
    Đọc và join 3 bảng order, order_detail, dish -> clean dataset

    Args:
        order_path: path tới order.csv
        order_detail_path: path tới order_detail.csv
        dish_path: path tới dish.csv
        spark: SparkSession (nếu None thì tự tạo)

    Returns:
        DataFrame clean (granularity = order_id, dish_id)
    """
    if spark is None:
        spark = SparkSession.builder.appName("BuildCleanDataset").getOrCreate()

    logger.info("Đọc dữ liệu từ CSV...")
    df_order = spark.read.option("header", True).option("inferSchema", True).csv(order_path)
    df_order_detail = spark.read.option("header", True).option("inferSchema", True).csv(order_detail_path)
    df_dish = spark.read.option("header", True).option("inferSchema", True).csv(dish_path)

    logger.info("Chuẩn hóa bảng order...")
    df_order = (
        df_order
        .drop("time")  # bỏ timestamp gốc
        .withColumnRenamed("quantity", "order_quantity")
        .withColumnRenamed("total_price", "order_total_price")
    )

    logger.info("Chuẩn hóa bảng order_detail...")
    df_order_detail = (
        df_order_detail
        .withColumnRenamed("quantity", "dish_quantity")
        .withColumnRenamed("price", "price_total_dish")
        .drop("id")
        .drop("dish_name")   # tránh trùng với dish_name từ bảng dish
    )

    logger.info("Chuẩn hóa bảng dish...")
    df_dish = df_dish.withColumnRenamed("price", "price_one_dish")

    logger.info("Join 3 bảng...")
    df_clean = (
        df_order_detail
        .join(df_order, on="order_id", how="inner")
        .join(
            df_dish.select("dish_id", "dish_name", "price_one_dish"),
            on="dish_id",
            how="left"
        )
    )

    logger.info(f"Schema sau khi join: {df_clean.columns}")
    return df_clean


# ========================= HÀM FEATURE =========================
def gen_feature_for_dish(df_clean, dish_name: str, horizons: List[int] = [3, 7, 14, 30]):
    """
    Sinh feature store cho 1 dish từ df_clean đã chuẩn hóa
    """
    # ----- CONTEXT LEVEL: toàn quán (chưa filter dish) -----
    df_context = (
        df_clean.groupBy("time_date")
        .agg(
            F.countDistinct("order_id").alias("day_order_count_1d"),
            F.sum("order_quantity").alias("day_total_quantity_1d"),
            F.sum("order_total_price").alias("day_total_revenue_1d"),
            F.countDistinct("dish_id").alias("day_order_variety_1d"),
            (F.sum("order_quantity") / F.countDistinct("order_id")).alias("avg_order_size_1d")
        )
    )

    # ----- DISH LEVEL: filter theo món rồi mới tính -----
    df_dish_level = (
        df_clean.filter(F.col("dish_name") == dish_name)
        .groupBy("time_date")
        .agg(
            F.sum("dish_quantity").alias("y_sales"),
            F.countDistinct("order_id").alias("dish_count_in_order_1d"),
            (F.sum("dish_quantity") / F.countDistinct("order_id")).alias("dish_quantity_in_order_avg_1d"),
            F.first("price_one_dish").alias("dish_base_price"),
            F.sum("price_total_dish").alias("dish_revenue_day_1d"),
            F.sum("dish_quantity").alias("dish_total_quantity_1d"),
        )
    )

    # ----- SHARE LEVEL: join dish_level với context -----
    df_share_level = (
        df_dish_level
        .join(df_context, on="time_date", how="left")
        .withColumn(
            "dish_sales_ratio_daily_1d",
            F.col("dish_total_quantity_1d") / F.col("day_total_quantity_1d")
        )
        .withColumn(
            "dish_order_ratio_daily_1d",
            F.col("dish_count_in_order_1d") / F.col("day_order_count_1d")
        )
        .withColumn(
            "dish_revenue_ratio_daily_1d",
            F.col("dish_revenue_day_1d") / F.col("day_total_revenue_1d")
        )
        .withColumn(
            "dish_order_size_ratio_1d",
            F.col("dish_quantity_in_order_avg_1d") / F.col("avg_order_size_1d")
        )
    )

    # Sort trước khi rolling
    df_sorted_for_rolling = df_share_level.orderBy("time_date")

    # Rolling theo horizon
    for h in horizons:
        w = (
            Window
            .orderBy("time_date")
            .rowsBetween(-(h - 1), 0)   # dùng rowsBetween thay rangeBetween
        )

        cnt_days = F.count(F.lit(1)).over(w)

        # ===== Context (rolling strict) =====
        sum_day_order_count = F.sum("day_order_count_1d").over(w)
        sum_day_total_qty = F.sum("day_total_quantity_1d").over(w)
        sum_day_total_rev = F.sum("day_total_revenue_1d").over(w)
        sum_day_variety = F.sum("day_order_variety_1d").over(w)

        df_sorted_for_rolling = (
            df_sorted_for_rolling
            .withColumn(f"day_order_count_{h}d",
                        F.when(cnt_days == h, sum_day_order_count).otherwise(F.lit(0)))
            .withColumn(f"day_total_quantity_{h}d",
                        F.when(cnt_days == h, sum_day_total_qty).otherwise(F.lit(0)))
            .withColumn(f"day_total_revenue_{h}d",
                        F.when(cnt_days == h, sum_day_total_rev).otherwise(F.lit(0)))
            .withColumn(f"avg_order_size_{h}d",
                        F.when((cnt_days == h) & (sum_day_order_count != 0),
                               (sum_day_total_qty / sum_day_order_count).cast("double"))
                        .otherwise(F.lit(0.0)))
            .withColumn(f"day_order_variety_{h}d",
                        F.when(cnt_days == h, sum_day_variety).otherwise(F.lit(0)))
        )

        # ===== Dish-level (rolling strict) =====
        sum_dish_order_count = F.sum("dish_count_in_order_1d").over(w)
        sum_dish_total_qty = F.sum("dish_total_quantity_1d").over(w)
        sum_dish_rev = F.sum("dish_revenue_day_1d").over(w)

        df_sorted_for_rolling = (
            df_sorted_for_rolling
            .withColumn(f"dish_count_in_order_{h}d",
                        F.when(cnt_days == h, sum_dish_order_count).otherwise(F.lit(0)))
            .withColumn(f"dish_quantity_in_order_avg_{h}d",
                        F.when((cnt_days == h) & (sum_dish_order_count != 0),
                               (sum_dish_total_qty / sum_dish_order_count).cast("double"))
                        .otherwise(F.lit(0.0)))
        )

        # ===== Share-level (rolling strict) =====
        df_sorted_for_rolling = (
            df_sorted_for_rolling
            .withColumn(f"dish_sales_ratio_daily_{h}d",
                        F.when((cnt_days == h) & (sum_day_total_qty != 0),
                               (sum_dish_total_qty / sum_day_total_qty).cast("double"))
                        .otherwise(F.lit(0.0)))
            .withColumn(f"dish_order_ratio_daily_{h}d",
                        F.when((cnt_days == h) & (sum_day_order_count != 0),
                               (sum_dish_order_count / sum_day_order_count).cast("double"))
                        .otherwise(F.lit(0.0)))
            .withColumn(f"dish_revenue_ratio_daily_{h}d",
                        F.when((cnt_days == h) & (sum_day_total_rev != 0),
                               (sum_dish_rev / sum_day_total_rev).cast("double"))
                        .otherwise(F.lit(0.0)))
            .withColumn(f"dish_order_size_ratio_{h}d",
                        F.when(cnt_days == h, F.avg("dish_order_size_ratio_1d").over(w))
                        .otherwise(F.lit(0.0)))
        )

    return df_sorted_for_rolling


# ========================= MAIN =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh feature cho tất cả món ăn")
    parser.add_argument("--input_dir", default="data/cleaned_data",
                        help="Thư mục input chứa order.csv, order_detail.csv, dish.csv")
    parser.add_argument("--output_dir", default="data/raw_data/train_dataset",
                        help="Thư mục output để lưu features")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FeatureStoreAllDishes").getOrCreate()

    # 1. Đọc & chuẩn hoá
    order_path = os.path.join(args.input_dir, "order.csv")
    order_detail_path = os.path.join(args.input_dir, "order_detail.csv")
    dish_path = os.path.join(args.input_dir, "dish.csv")

    df_clean = build_clean_dataset(order_path, order_detail_path, dish_path, spark)

    # 2. Lấy danh sách tất cả dish
    dish_names = [row["dish_name"] for row in df_clean.select("dish_name").distinct().collect()]
    logger.info(f"📌 Tổng số dish = {len(dish_names)}")

    # 3. Sinh feature cho từng dish
    os.makedirs(args.output_dir, exist_ok=True)
    for dish_name in dish_names:
        logger.info(f"🚀 Sinh feature cho dish: {dish_name}")
        df_features = gen_feature_for_dish(df_clean, dish_name)

        safe_name = dish_name.replace(" ", "_")  # tránh lỗi tên file
        output_path = os.path.join(args.output_dir, f"{safe_name}.csv")

        df_features.toPandas().to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Done: {output_path}")

    spark.stop()
