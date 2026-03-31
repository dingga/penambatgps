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
    page_title="Sistem Inspeksi Penambat Rel - ITB", 
    page_icon="🛤️", 
    layout="wide"
)

# --- FUNGSI HELPER ---
def load_database(path_piket):
    """Memuat database patok KM untuk sinkronisasi spasial."""
    df_piket = pd.read_csv(path_piket)
    coords = df_piket[['latitude', 'longitude']].values
    tree = KDTree(coords)
    return df_piket, tree

# --- INITIALIZE SESSION STATE ---
if 'gallery_hilang' not in st.session_state:
    st.session_state.gallery_hilang = []
if 'map_markers' not in st.session_state:
    st.session_state.map_markers = []

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/ae/Logo_Institut_Teknologi_Bandung.svg/1200px-Logo_Institut_Teknologi_Bandung.svg.png", width=80)
st.sidebar.title("⚙️ Panel Kontrol")

model_file = st.sidebar.file_uploader("1. Upload Model (.pt)", type=['pt'])
video_file = st.sidebar.file_uploader("2. Upload Video (.mp4)", type=['mp4'])
gps_file = st.sidebar.file_uploader("3. Upload GPS Rekayasa (.csv)", type=['csv'])
piket_file = st.sidebar.file_uploader("4. Upload Database Patok (.csv)", type=['csv'])

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

if st.sidebar.button("🗑️ Reset Analisis"):
    st.session_state.gallery_hilang = []
    st.session_state.map_markers = []
    st.rerun()

# --- LAYOUT UTAMA ---
st.title("🛤️ Sistem Deteksi & Pemetaan Penambat Rel")
st.write("Visualisasi Real-time Deteksi YOLOv8 dengan Sinkronisasi GPS & Kilometer (KM).")

if model_file and video_file and gps_file and piket_file:
    # Load Databases
    df_gps_video = pd.read_csv(gps_file)
    df_piket, piket_tree = load_database(piket_file)

    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
        tmp_model.write(model_file.read()); model_path = tmp_model.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read()); video_path = tmp_video.name

    # Load YOLO
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI (Sesuai kode Colab Anda)
    roi_points = np.array([
        [int(0.35 * width), int(0.70 * height)], [int(0.40 * width), int(0.10 * height)],
        [int(0.60 * width), int(0.10 * height)], [int(0.65 * width), int(0.70 * height)]
    ], np.int32)
    y_ref = int(0.5 * height)

    # UI Columns
    col_vid, col_stat = st.columns([2, 1])
    frame_window = col_vid.empty()
    stat_window = col_stat.empty()

    if st.button("🚀 Jalankan Inspeksi Jalur"):
        results = model.track(source=video_path, persist=True, imgsz=1024, stream=True, conf=conf_thresh)
        
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter({"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0})

        for f_idx, res in enumerate(results):
            frame = res.orig_img
            
            # 1. Sinkronisasi GPS & KM
            try:
                row_gps = df_gps_video[df_gps_video['frame_id'] == f_idx].iloc[0]
                lat, lon = row_gps['reconstructed_lat'], row_gps['reconstructed_lon']
                dist, p_idx = piket_tree.query([lat, lon], k=1)
                piket = df_piket.iloc[p_idx]
                km_now = piket['kd_kmhm']
            except:
                lat, lon, km_now = 0, 0, "N/A"

            # 2. Pemrosesan Box
            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    # Visualisasi di Video
                    color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"ID:{tid} {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Logika Counting & Geotagging
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            if final_label == "Hilang":
                                # Simpan data untuk Peta & Galeri
                                st.session_state.map_markers.append({"lat": lat, "lon": lon, "km": km_now, "id": tid})
                                st.session_state.gallery_hilang.append({
                                    "id": tid, "km": km_now, "coord": f"{lat:.6f}, {lon:.6f}",
                                    "img": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                })

            # 3. Update UI Real-time
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)
            
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            with stat_window.container():
                st.subheader("📊 Statistik Live")
                st.info(f"📍 KM Saat Ini: **{km_now}**")
                c1, c2 = st.columns(2)
                c1.metric("E-Clip", summary_counts["E Clip"])
                c1.metric("DE-Clip", summary_counts["DE CLIP"])
                c2.metric("KA-Clip", summary_counts["KA Clip"])
                c2.metric("⚠️ Hilang", summary_counts["Hilang"], delta_color="inverse")

    # --- BAGIAN PETA INTERAKTIF ---
    st.divider()
    st.subheader("🗺️ Sebaran Titik Penambat Hilang")
    if st.session_state.map_markers:
        m = folium.Map(location=[st.session_state.map_markers[0]['lat'], st.session_state.map_markers[0]['lon']], zoom_start=16)
        # Gambar Rute
        folium.PolyLine(df_gps_video[['reconstructed_lat', 'reconstructed_lon']].values, color="blue", weight=2).add_to(m)
        # Tambah Marker Merah
        for mark in st.session_state.map_markers:
            folium.Marker(
                [mark['lat'], mark['lon']], 
                popup=f"ID: {mark['id']} | KM: {mark['km']}",
                icon=folium.Icon(color='red', icon='warning-sign')
            ).add_to(m)
        st_folium(m, width=1400, height=500)

    # --- GALERI TEMUAN ---
    st.divider()
    st.subheader("📸 Bukti Visual Penambat Hilang")
    cols = st.columns(3)
    for idx, item in enumerate(reversed(st.session_state.gallery_hilang)):
        with cols[idx % 3]:
            st.image(item['img'], caption=f"ID: {item['id']} | KM: {item['km']} | {item['coord']}")

else:
    st.info("Harap lengkapi semua upload (Model, Video, GPS Rekayasa, dan Database Patok) untuk memulai.")
