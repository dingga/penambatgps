import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os
import time
import folium
from streamlit_folium import st_folium
from scipy.spatial import KDTree

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Inspeksi Penambat Rel ITB", 
    page_icon="🛤️", 
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
if 'gallery_hilang' not in st.session_state:
    st.session_state.gallery_hilang = []
if 'laporan_final' not in st.session_state:
    st.session_state.laporan_final = []

# --- FUNGSI HELPER ---
def load_database(df_piket):
    """Membuat KDTree untuk pencarian koordinat tercepat."""
    coords = df_piket[['latitude', 'longitude']].values
    tree = KDTree(coords)
    return tree

# --- SIDEBAR & MODEL LOADING ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/ae/Logo_Institut_Teknologi_Bandung.svg/1200px-Logo_Institut_Teknologi_Bandung.svg.png", width=80)
st.sidebar.title("⚙️ Panel Kontrol")

# Memuat model otomatis dari GitHub
MODEL_PATH = 'best.pt'
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    st.sidebar.success("✅ Model 'best.pt' aktif")
else:
    st.sidebar.error("❌ File 'best.pt' tidak ditemukan di repositori!")
    st.stop()

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.20)
video_file = st.sidebar.file_uploader("1. Upload Video (.mp4)", type=['mp4'])
gps_file = st.sidebar.file_uploader("2. Upload GPS Rekayasa (.csv)", type=['csv'])
piket_file = st.sidebar.file_uploader("3. Upload Database Patok (.csv)", type=['csv'])

if st.sidebar.button("🗑️ Reset Analisis"):
    st.session_state.gallery_hilang = []
    st.session_state.laporan_final = []
    st.rerun()

# --- LAYOUT UTAMA ---
st.title("🛤️ Dashboard Deteksi & Geomapping Penambat Rel")
st.write("Integrasi Computer Vision YOLOv8 dengan Koordinat GPS dan Kilometer (KM) Rel.")

if video_file and gps_file and piket_file:
    # Load Data
    df_gps_video = pd.read_csv(gps_file)
    df_piket = pd.read_csv(piket_file)
    piket_tree = load_database(df_piket)

    # Simpan video sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Ref (Sesuai parameter thesis Anda)
    roi_points = np.array([
        [int(0.35 * width), int(0.70 * height)], [int(0.40 * width), int(0.10 * height)],
        [int(0.60 * width), int(0.10 * height)], [int(0.65 * width), int(0.70 * height)]
    ], np.int32)
    y_ref = int(0.5 * height)

    col_vid, col_stat = st.columns([2, 1])
    frame_window = col_vid.empty()
    stat_window = col_stat.empty()

    if st.button("🚀 Jalankan Inspeksi Jalur"):
        # Jalankan tracking
        results = model.track(source=video_path, persist=True, imgsz=1024, stream=True, conf=conf_thresh)
        
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter({"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0})

        for f_idx, res in enumerate(results):
            frame = res.orig_img
            
            # 1. Sinkronisasi Spasial (GPS ke KM)
            try:
                row_gps = df_gps_video[df_gps_video['frame_id'] == f_idx].iloc[0]
                lat = row_gps['reconstructed_lat']
                lon = row_gps['reconstructed_lon']
                dist, p_idx = piket_tree.query([lat, lon], k=1)
                piket = df_piket.iloc[p_idx]
                km_now = piket['kd_kmhm']
                lintas = piket.get('route_code', 'N/A')
            except:
                lat, lon, km_now, lintas = 0, 0, "Searching...", "N/A"

            # 2. Deteksi & Tracking
            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    # Visualisasi
                    color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"ID:{tid} {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Logika Counting & Laporan
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            if final_label == "Hilang":
                                # Simpan untuk Peta & CSV Laporan
                                data_temuan = {
                                    "Track_ID": tid, "Status": "HILANG", "KM_Lokasi": km_now,
                                    "Latitude": lat, "Longitude": lon, "Lintas": lintas, "Frame": f_idx
                                }
                                st.session_state.laporan_final.append(data_temuan)
                                st.session_state.gallery_hilang.append({
                                    "id": tid, "km": km_now, "img": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                })

            # 3. Update Visual Dashboard
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            with stat_window.container():
                st.subheader("📊 Monitoring")
                st.info(f"📍 Posisi: **{km_now}**")
                st.metric("⚠️ Total Penambat Hilang", summary_counts["Hilang"])
                st.write(f"DE Clip: {summary_counts['DE CLIP']} | E-Clip: {summary_counts['E Clip']}")

    # --- FITUR DOWNLOAD & PETA ---
    if st.session_state.laporan_final:
        st.divider()
        df_laporan = pd.DataFrame(st.session_state.laporan_final)
        
        col_map, col_dl = st.columns([2, 1])
        
        with col_dl:
            st.subheader("📥 Unduh Laporan")
            csv = df_laporan.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Laporan Detail", data=csv, file_name='Laporan_Inspeksi_Penambat.csv', mime='text/csv')
            st.dataframe(df_laporan.tail(5)) # Menampilkan 5 temuan terakhir

        with col_map:
            st.subheader("🗺️ Sebaran Titik Kerusakan")
            m = folium.Map(location=[df_laporan['Latitude'].iloc[0], df_laporan['Longitude'].iloc[0]], zoom_start=15)
            # Gambar rute jika data GPS tersedia
            folium.PolyLine(df_gps_video[['reconstructed_lat', 'reconstructed_lon']].values, color="blue", weight=2, opacity=0.5).add_to(m)
            # Tambahkan Marker Kerusakan
            for _, row in df_laporan.iterrows():
                folium.Marker(
                    [row['Latitude'], row['Longitude']], 
                    popup=f"KM: {row['KM_Lokasi']} (ID: {row['Track_ID']})",
                    icon=folium.Icon(color='red', icon='warning-sign')
                ).add_to(m)
            st_folium(m, width=700, height=400)

    # --- GALERI BUKTI VISUAL ---
    if st.session_state.gallery_hilang:
        st.divider()
        st.subheader("📸 Galeri Bukti Penambat Hilang")
        cols = st.columns(4)
        for idx, item in enumerate(reversed(st.session_state.gallery_hilang)):
            with cols[idx % 4]:
                st.image(item['img'], caption=f"ID: {item['id']} | KM: {item['km']}")

else:
    st.info("Silakan unggah Video, GPS Rekayasa, dan Database Patok pada sidebar untuk memulai analisis.")
