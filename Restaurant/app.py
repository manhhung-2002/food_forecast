import streamlit as st
import pandas as pd
import os

# ==================== CONFIG ====================
OUTPUT_DIR = "data/predictions"   # đường dẫn thư mục chứa predict_YYYYMM.csv

st.set_page_config(page_title="🍜 Restaurant Forecast — Daily Overview", layout="wide")
st.title("🍜 Restaurant Forecast — Theo ngày & theo món")

# ==================== LOAD FILES ====================
files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")])
if not files:
    st.warning("⚠️ Không có file dự đoán nào trong thư mục predictions/")
    st.stop()

# Lấy danh sách tháng (ví dụ: 202401 → 2024-01)
month_list = [f.replace("predict_", "").replace(".csv", "") for f in files]

selected_month = st.selectbox("🗓️ Chọn tháng:", month_list)

# Load dữ liệu tháng được chọn
file_path = os.path.join(OUTPUT_DIR, f"predict_{selected_month}.csv")
df = pd.read_csv(file_path)
df["time_date"] = pd.to_datetime(df["time_date"])

# ==================== UI: CHỌN NGÀY ====================
day_list = sorted(df["time_date"].dt.date.unique())
selected_day = st.selectbox("📅 Chọn ngày:", day_list, format_func=lambda x: x.strftime("%Y-%m-%d"))

# Lấy dữ liệu ngày được chọn
df_day = df[df["time_date"].dt.date == selected_day]

# Nếu không có dòng nào thì cảnh báo
if df_day.empty:
    st.warning("Không có dữ liệu dự đoán cho ngày này.")
    st.stop()

# Bỏ cột time_date (vì đã chọn 1 ngày)
dish_columns = [c for c in df.columns if c != "time_date"]

# ==================== TÍNH TRUNG BÌNH THÁNG ====================
df_avg = df[dish_columns].mean().to_dict()

st.divider()
st.markdown(f"## 📆 Kết quả ngày **{selected_day.strftime('%Y-%m-%d')}** — Tháng **{selected_month}**")

# ==================== HIỂN THỊ CARD CHO MỖI MÓN ====================
st.markdown("""
<style>
.card {
  border: 1px solid #eee; border-radius: 12px; padding: 14px; margin-bottom: 12px;
  background: #fffaf3;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.metric {font-size: 28px; font-weight: 700; margin: 6px 0 0;}
.subtle {color:#555; font-size:14px;}
</style>
""", unsafe_allow_html=True)

cols = st.columns(3)

for i, dish in enumerate(dish_columns):
    pred_today = float(df_day[dish].iloc[0])
    avg_month = float(df_avg[dish])

    with cols[i % 3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"## 🍽️ {dish.replace('_',' ').title()}")
        st.markdown(f'<div class="metric">{pred_today:,.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtle">Dự tính bán được trong ngày đã chọn</div>', unsafe_allow_html=True)
        st.markdown(f"**Trung bình/tháng:** {avg_month:,.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(f"Dữ liệu dự đoán đã xử lý sẵn. Tháng {selected_month} — Ngày {selected_day.strftime('%Y-%m-%d')}.")
