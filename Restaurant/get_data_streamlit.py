import os
import pandas as pd
import joblib
from glob import glob
import os
print("🧭 Current working dir:", os.getcwd())
print("📁 Data dir exists:", os.path.exists("data/train_dataset"))


# ========================= MAIN =========================
def main():
    MODEL_DIR = "models/"
    DATA_DIR = "data/train_dataset/"
    OUT_DIR = "data/predictions/"
    os.makedirs(OUT_DIR, exist_ok=True)

    models ={}
    for model_path in glob(os.path.join(MODEL_DIR, "*.pkl")):
        dish = os.path.basename(model_path).replace(".pkl", "").replace("xgb_", "")
        models[dish] = joblib.load(model_path)

    merged_df = None  # sẽ merge các món theo time_date

    for dish_name, model in models.items():
        data_path = os.path.join(DATA_DIR, f"{dish_name}.csv")
        if not os.path.exists(data_path):
            print(f"Bỏ qua {dish_name} : không có {data_path}")
            continue

        df = pd.read_csv(data_path)
        if "time_date" not in df.columns:
            raise ValueError(f" File {data_path} thiếu cột 'time_date'")

        df["month"] = pd.to_datetime(df["time_date"]).dt.month
        df["day_of_year"] = pd.to_datetime(df["time_date"]).dt.dayofyear
        # Giữ lại time_date
        time_col = df[["time_date"]]

        X = df.drop(columns=["time_date","y_sales"], errors = "ignore")
        # Prdict
        y_pred = model.predict(X)
        pred_df = pd.DataFrame({
            "time_date": time_col["time_date"],
            dish_name: y_pred
        })

        if merged_df is None:
            merged_df = pred_df
        else:
            merged_df = pd.merge(merged_df, pred_df, on="time_date", how="outer")

        print(f" Đã predict xong món: {dish_name} ({len(pred_df):,} dòng)")

    if merged_df is not None:
        merged_df["month"] = pd.to_datetime(merged_df["time_date"]).dt.strftime("%Y%m")
        for month, group in merged_df.groupby("month"):
            output_path = os.path.join(OUT_DIR, f"predict_{month}.csv")
            group.sort_values("time_date").to_csv(output_path, index=False)
        print("Đã hoành thành dự đoán cho các tháng")
    else:
        print("Không có dữ liệu nào để dự đoán")



if __name__ == "__main__":
    main()






