import os
import pandas as pd
import mysql.connector

# Config DB
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "12345678",
    "database": "Restaurant"
}

OUTPUT_DIR = "../data"

def export_all_tables():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(f"SHOW TABLES FROM {DB_CONFIG['database']}")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"📂 Found tables: {tables}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for table in tables:
        print(f"➡ Exporting {table} ...")
        query = f"SELECT * FROM `{table}`"
        df = pd.read_sql(query, conn)

        # Chuẩn hóa các cột datetime/time
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    print(f"⚠️ Could not convert {col}: {e}")

        # Đường dẫn file CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{table}.csv")

        # Nếu file đã tồn tại thì xoá trước
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"🗑️ Removed old {csv_path}")

        # Ghi CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved {csv_path} ({len(df)} rows)")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    export_all_tables()
