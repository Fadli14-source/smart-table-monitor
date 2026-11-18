import os
import time
import cv2
import pandas as pd
import streamlit as st
from src.detector import TableMonitorEngine

# Konfigurasi Halaman
st.set_page_config(
    page_title="AI Restaurant Table Occupancy & Operations",
    page_icon="🍽️",
    layout="wide",
)

# Path Absolut Otomatis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(BASE_DIR, "assets", "cctv_sample.mp4")
ZONES_PATH = os.path.join(BASE_DIR, "config", "zones.json")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Styling CSS ringan untuk kartu status
st.markdown(
    """
    <style>
    .table-card {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: white;
        font-family: sans-serif;
    }
    .status-FREE { background-color: #1e8449; border-left: 6px solid #2ecc71; }
    .status-OCCUPIED { background-color: #922b21; border-left: 6px solid #e74c3c; }
    .status-NEEDS_CLEANING { background-color: #b7950b; border-left: 6px solid #f1c40f; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍽️ AI Smart Table Occupancy & Operations Monitor")
st.caption("Real-time Computer Vision Floor Monitoring, Queue Turnover Prediction & Staffing Analytics")

# Sidebar Kontrol
with st.sidebar:
    st.header("⚙️ Pengaturan Sistem")
    run_stream = st.toggle("▶️ Jalankan Monitoring CCTV", value=True)
    loop_video = st.checkbox("🔁 Ulangi Video Otomatis (Loop)", value=True)
    st.divider()
    st.subheader("Threshold Simulasi")
    sit_sec = st.slider("Konfirmasi Duduk (detik)", 0.5, 5.0, 1.2, 0.1)
    waiter_sec = st.slider("Alert Pelayan (detik)", 2.0, 15.0, 4.0, 0.5)
    clean_sec = st.slider("Konfirmasi Bersih (detik)", 1.0, 10.0, 2.0, 0.5)
    st.divider()
    st.info("💡 **Tips Demo:** Karena video sampel berdurasi 10 detik, nilai threshold dipercepat agar siklus status & alert langsung terlihat.")

# Validasi File Zona
if not os.path.exists(ZONES_PATH):
    st.error(f"File konfigurasi zona tidak ditemukan di `{ZONES_PATH}`. Jalankan `src/calibrate_zones.py` terlebih dahulu.")
    st.stop()

# Navigasi Tab Utama
tab_live, tab_analytics = st.tabs(["🎥 Live Floor Operations", "📊 Peak-Hour Analytics & Staffing"])

with tab_analytics:
    st.subheader("Analisis Jam Sibuk (Peak Hours) & Rekomendasi Staf")
    st.markdown("Agregasi data historis okupansi meja untuk membantu manajer menyusun jadwal *shift* pelayan secara optimal.")

    # Data simulasi historis berbasis pola restoran harian
    hourly_data = pd.DataFrame({
        "Jam": ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00"],
        "Okupansi (%)": [25, 55, 92, 88, 45, 30, 35, 60, 85, 95, 90, 50],
        "Rata-rata Durasi (Menit)": [22, 30, 42, 40, 28, 25, 25, 32, 45, 48, 44, 30],
    })

    col_chart, col_rec = st.columns([2, 1])
    with col_chart:
        st.bar_chart(hourly_data.set_index("Jam")["Okupansi (%)"], color="#e74c3c")

    with col_rec:
        st.markdown("### 📋 Rekomendasi Shift Staf")
        st.warning("**Peak Lunch (12:00 – 13:30):** Okupansi > 85%. Disarankan **4 Pelayan + 1 Busser** khusus pembersihan cepat.")
        st.error("**Peak Dinner (18:30 – 20:30):** Okupansi mencapai 95%. Disarankan **5 Pelayan + 1 Greeter** untuk manajemen antrean.")
        st.success("**Off-Peak (14:00 – 16:30):** Okupansi < 45%. Cukup **2 Pelayan** untuk efisiensi biaya operasional.")

with tab_live:
    # Placeholder Top KPI Metrics
    kpi_placeholder = st.empty()

    col_video, col_panel = st.columns([1.8, 1.2])
    with col_video:
        st.subheader("📹 Live CCTV Feed (CAM 04)")
        video_placeholder = st.empty()

    with col_panel:
        st.subheader("🚨 Operational Alerts")
        alert_placeholder = st.empty()

        st.subheader("🪑 Status Meja Real-Time")
        table_grid_placeholder = st.empty()

    # Jalankan pemrosesan video jika toggle aktif
    if run_stream:
        cap = cv2.VideoCapture(DEFAULT_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Gunakan yolov8n.pt lokal jika ada, jika tidak ultralytics akan otomatis mengunduh
        model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt"
        engine = TableMonitorEngine(zones_path=ZONES_PATH, model_path=model_file, fps=fps)

        while cap.isOpened() and run_stream:
            ret, frame = cap.read()
            if not ret:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    engine = TableMonitorEngine(zones_path=ZONES_PATH, model_path=model_file, fps=fps)
                    continue
                else:
                    break

            # Update parameter threshold secara real-time dari slider sidebar
            engine.SIT_CONFIRM_SEC = sit_sec
            engine.UNATTENDED_ALERT_SEC = waiter_sec
            engine.CLEAN_CONFIRM_SEC = clean_sec

            annotated_frame, table_states, summary = engine.process_frame(frame)

            # 1. Update KPI Cards
            with kpi_placeholder.container():
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Occupancy Rate", f"{summary['occupancy_rate']}%")
                k2.metric("🟢 Free", f"{summary['free']} Meja")
                k3.metric("🔴 Occupied", f"{summary['occupied']} Meja")
                k4.metric("🟡 Needs Cleaning", f"{summary['needs_cleaning']} Meja")
                k5.metric("👥 Active Guests", f"{summary['total_guests']} Orang")
                k6.metric("⏳ Next Available", summary["next_available_table"])

            # 2. Update Video Frame (BGR -> RGB)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # 3. Update Alerts Panel
            with alert_placeholder.container():
                if len(summary["alerts"]) == 0:
                    st.success("✅ Semua operasional meja berjalan normal.")
                else:
                    for al in summary["alerts"]:
                        if al["type"] == "CLEANING":
                            st.warning(f"🧹 {al['message']}")
                        else:
                            st.error(f"🔔 {al['message']}")

            # 4. Update Grid Kartu Status Meja (2 kolom)
            with table_grid_placeholder.container():
                t_cols = st.columns(2)
                for idx, (tid, tstate) in enumerate(table_states.items()):
                    status = tstate["status"]
                    duration_txt = f"{int(tstate['occupied_seconds'])}s" if status == "OCCUPIED" else "-"
                    guest_txt = f"{tstate['guest_count']} Tamu" if status == "OCCUPIED" else "0 Tamu"

                    card_html = f"""
                    <div class="table-card status-{status}">
                        <strong>{tid} — {status.replace('_', ' ')}</strong><br/>
                        <small>👥 {guest_txt} &nbsp;|&nbsp; ⏱️ Durasi: {duration_txt}</small>
                    </div>
                    """
                    t_cols[idx % 2].markdown(card_html, unsafe_allow_html=True)

            # Sedikit jeda agar UI Streamlit mulus
            time.sleep(0.01)

        cap.release()
    else:
        st.info("Aktifkan tombol **▶️ Jalankan Monitoring CCTV** di sidebar kiri untuk memulai pemrosesan.")