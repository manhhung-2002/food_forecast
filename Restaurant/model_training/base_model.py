""""
Các bước:
1. Đọc dữ liệu CSV
2. Sắp xếp theo thời gian
3. Tạo các feature phụ từ thời gian
4. Chia dữ liệu 60% train, 20% valid, 20% test
5. Tách feature (X) và target (y) cho từng tập
6. Train mô hình XGBoost
7. Đánh giá hiệu năng và lưu model ra file .pkl
"""

import argparse
import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_split_time_series(df_path, target_col="y_sales", time_col="time_date"):
    df = pd.read_csv(df_path)

    df[time_col] = pd.to_datetime(df[time_col])
    df["month"] = df["time_date"].dt.month
    df["day_of_year"] = df["time_date"].dt.dayofyear

    total_rows = len(df)
    num_train = int(total_rows * 0.6)
    num_valid = int(total_rows * 0.2)

    train_df = df.iloc[:num_train]
    valid_df = df.iloc[num_train:num_train+num_valid]
    test_df = df.iloc[num_train+num_valid:]


    print("=======================================")
    print(f"Tổng số dòng dữ liệu: {total_rows}")
    print(f"  - Tập Train : {len(train_df)} dòng")
    print(f"  - Tập Valid : {len(valid_df)} dòng")
    print(f"  - Tập Test  : {len(test_df)} dòng")
    print("=======================================")


    all_columns = df.columns.tolist()
    feature_columns = []
    for column_name in all_columns:
        if column_name != target_col and column_name != time_col:
            feature_columns.append(column_name)

    # Bước 5.3: Chọn phần feature (X) và target (y) cho từng tập
    X_train = train_df[feature_columns]  # Feature của tập train
    y_train = train_df[target_col]  # Target của tập train

    X_valid = valid_df[feature_columns]  # Feature của tập validation
    y_valid = valid_df[target_col]  # Target của tập validation

    X_test = test_df[feature_columns]  # Feature của tập test
    y_test = test_df[target_col]  # Target của tập test

    # Bước 5.4: Trả về toàn bộ 6 tập dữ liệu
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_xgb_time_series(X_train, y_train, X_valid, y_valid, params):
    """
    Huấn luyện mô hình xgboost

    """
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=True
    )
    return model

def evaluate_and_save(model, X_test, y_test, out_dir, dish_name):
    """
    Đánh giá hiệu năng của mô hình trên tập test và lưu model ra file.
    """
    # Bước 1: Dự đoán giá trị y_pred từ X_test
    y_pred = model.predict(X_test)

    # Bước 2: Tính các chỉ số đánh giá
    mae_value = mean_absolute_error(y_test, y_pred)
    rmse_value = mean_squared_error(y_test, y_pred, squared=False)
    r2_value = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae_value,
        "RMSE": rmse_value,
        "R2": r2_value
    }

    # Bước 3: Lưu mô hình ra file .pkl
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"xgb_{dish_name}.pkl")
    joblib.dump(model, out_path)

    # Bước 4: In kết quả ra màn hình
    print("\n==============================")
    print(f" Đã train xong mô hình cho món: {dish_name}")
    print(f" Model được lưu tại: {out_path}")
    print(f" Hiệu năng mô hình:")
    print(f"   - MAE  : {mae_value:.4f}")
    print(f"   - RMSE : {rmse_value:.4f}")
    print(f"   - R2   : {r2_value:.4f}")
    print("==============================\n")

    # Bước 5: Trả về các chỉ số đánh giá
    return metrics

def main():
    """
    Pipeline chính: đọc dữ liệu → chia theo thời gian → train → evaluate → save
    """

    # Bước 1: Định nghĩa và đọc tham số đầu vào
    parser = argparse.ArgumentParser(description="Train XGBoost cho dữ liệu time series")
    parser.add_argument("--df_path", required=True, help="Đường dẫn file CSV của món ăn")
    parser.add_argument("--target", default="y_sales", help="Tên cột target (nhãn cần dự đoán)")
    parser.add_argument("--dish_name", required=True, help="Tên món ăn để đặt tên file model")
    parser.add_argument("--out_dir", default="./models", help="Thư mục lưu model sau khi train")

    # Tham số mô hình
    parser.add_argument("--n_estimators", type=int, default=800, help="Số lượng cây trong mô hình")
    parser.add_argument("--max_depth", type=int, default=7, help="Độ sâu tối đa của mỗi cây")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="Tốc độ học")
    parser.add_argument("--subsample", type=float, default=0.9, help="Tỷ lệ mẫu dùng cho mỗi cây")
    parser.add_argument("--colsample_bytree", type=float, default=0.9, help="Tỷ lệ cột dùng cho mỗi cây")

    args = parser.parse_args()

    # Bước 2: Đọc và chia dữ liệu
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_split_time_series(
        df_path=args.df_path,
        target_col=args.target
    )

    # Bước 3: Cấu hình tham số mô hình
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": 42,
    }

    # Bước 4: Huấn luyện mô hình
    model = train_xgb_time_series(X_train, y_train, X_valid, y_valid, params)

    # Bước 5: Đánh giá và lưu kết quả
    evaluate_and_save(model, X_test, y_test, args.out_dir, args.dish_name)


# =====================================================
# 5️⃣ ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()


