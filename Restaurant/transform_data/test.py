import os
import pandas as pd
import joblib
from glob import glob

# ========================= MAIN =========================
def main():
    MODEL_DIR = "models/"
    DATA_DIR = "data_predict/"
    OUT_DIR = "predictions/"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1️⃣ Load tất cả mô hình
    models = {}
    for model_path in glob(os.path.join(MODEL_DIR, "*.pkl")):
        dish = os.path.basename(model_path).replace(".pkl", "").replace("xgb_", "")
        models[dish] = joblib.load(model_path)
    print(f"✅ Đã load {len(models)} mô hình:", ", ".join(models.keys()))

    merged_df = None  # sẽ merge các món theo time_date

    # 2️⃣ Predict từng món
    for dish_name, model in models.items():
        data_path = os.path.join(DATA_DIR, f"{dish_name}.csv")
        if not os.path.exists(data_path):
            print(f"⚠️ Bỏ qua {dish_name}: không thấy {data_path}")
            continue

        df = pd.read_csv(data_path)
        if "time_date" not in df.columns:
            raise ValueError(f"❌ File {data_path} thiếu cột 'time_date'")

        # Giữ lại time_date
        time_col = df[["time_date"]]

        # Drop y_sales + time_date (model chỉ cần features)
        X = df.drop(columns=["y_sales", "time_date"], errors="ignore")

        # Predict
        y_pred = model.predict(X)
        pred_df = pd.DataFrame({
            "time_date": time_col["time_date"],
            dish_name: y_pred
        })

        # Merge vào DataFrame tổng
        if merged_df is None:
            merged_df = pred_df
        else:
            merged_df = pd.merge(merged_df, pred_df, on="time_date", how="outer")

        print(f"✅ Đã predict xong món: {dish_name} ({len(pred_df):,} dòng)")

    # 3️⃣ Sau khi merge hết → thêm cột tháng
    if merged_df is not None:
        merged_df["month"] = pd.to_datetime(merged_df["time_date"]).dt.strftime("%Y%m")

        # 4️⃣ Gộp và lưu theo tháng
        for month, group in merged_df.groupby("month"):
            out_path = os.path.join(OUT_DIR, f"predict_{month}.csv")
            group.sort_values("time_date").to_csv(out_path, index=False)
            print(f"📦 Đã lưu file: {out_path} ({len(group):,} dòng)")

        print("\n🎉 Hoàn tất inference toàn năm.")
    else:
        print("⚠️ Không có dữ liệu nào để dự đoán.")

# ========================= ENTRYPOINT =========================
if __name__ == "__main__":
    main()
