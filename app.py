import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os
import folium
from streamlit_folium import st_folium
from scipy.spatial import KDTree
import base64
from io import BytesIO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Inspeksi Penambat Rel ITB", 
    page_icon="🛤️", 
    layout="wide"
)

# --- FUNGSI HELPER GAMBAR & DATA ---
def get_base64_encoded_image(img_rgb):
    """Mengonversi array gambar RGB ke string base64 untuk popup HTML peta."""
    img = Image.fromarray(img_rgb)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_database(df_piket):
    """Membuat KDTree untuk pencarian koordinat spasial cepat."""
    coords = df_piket[['latitude', 'longitude']].values
    tree = KDTree(coords)
    return tree

# --- INITIALIZE SESSION STATE ---
if 'gallery_hilang' not in st.session_state:
    st.session_state.gallery_hilang = []
if 'laporan_final' not in st.session_state:
    st.session_state.laporan_final = []
if 'rekap_aset' not in st.session_state:
    st.session_state.rekap_aset = {"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0}

# --- SIDEBAR & AUTO-LOAD MODEL ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/ae/Logo_Institut_Teknologi_Bandung.svg/1200px-Logo_Institut_Teknologi_Bandung.svg.png", width=80)
st.sidebar.title("⚙️ Panel Kontrol")

MODEL_PATH = 'best.pt'
if os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
        st.sidebar.success("✅ Model 'best.pt' Aktif")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
        st.stop()
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
    st.session_state.rekap_aset = {"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0}
    st.rerun()

# --- LAYOUT UTAMA ---
st.title("🛤️ Dashboard Deteksi & Geomapping Penambat Rel")
st.write("Analisis lengkap aset penambat rel dengan integrasi pemetaan spasial dan bukti visual.")

if video_file and gps_file and piket_file:
    df_gps_video = pd.read_csv(gps_file)
    df_piket = pd.read_csv(piket_file)
    piket_tree = load_database(df_piket)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    roi_points = np.array([
        [int(0.35 * width), int(0.70 * height)], [int(0.40 * width), int(0.10 * height)],
        [int(0.60 * width), int(0.10 * height)], [int(0.65 * width), int(0.70 * height)]
    ], np.int32)
    y_ref = int(0.5 * height)

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
            
            try:
                row_gps = df_gps_video[df_gps_video['frame_id'] == f_idx].iloc[0]
                lat, lon = row_gps['reconstructed_lat'], row_gps['reconstructed_lon']
                dist, idxs = piket_tree.query([lat, lon], k=2)
                piket1, piket2 = df_piket.iloc[idxs[0]], df_piket.iloc[idxs[1]]
                km_list = sorted([str(piket1['kd_kmhm']), str(piket2['kd_kmhm'])])
                km_range = f"{km_list[0]} / {km_list[1]}"
            except:
                lat, lon, km_range = 0, 0, "Mencari..."

            if res.boxes is not None and res.boxes.id is not None:
                boxes, ids, clss = res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"ID:{tid} {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1
                            st.session_state.rekap_aset = dict(summary_counts)

                            # Log Temuan ke Laporan (Semua Jenis Aset)
                            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            data_temuan = {
                                "Track_ID": tid, "Status": final_label, "KM_Rentang": km_range,
                                "Latitude": lat, "Longitude": lon, "Frame": f_idx,
                                "b64_img": get_base64_encoded_image(img_rgb) if final_label == "Hilang" else None
                            }
                            st.session_state.laporan_final.append(data_temuan)

                            if final_label == "Hilang":
                                st.session_state.gallery_hilang.append({"id": tid, "km": km_range, "img": img_rgb})

            cv2.polylines(frame, [roi_points], True, (0, 0, 0), 2) # ROI Hitam
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            with stat_window.container():
                st.subheader("📊 Statistik Aset Terdeteksi")
                st.info(f"📍 Posisi: **{km_range}**")
                c1, c2 = st.columns(2)
                c1.metric("E-Clip", summary_counts["E Clip"])
                c1.metric("DE-Clip", summary_counts["DE CLIP"])
                c2.metric("KA-Clip", summary_counts["KA Clip"])
                c2.metric("⚠️ Hilang", summary_counts["Hilang"])

    # --- BAGIAN LAPORAN & PETA ---
    if st.session_state.laporan_final:
        st.divider()
        df_laporan = pd.DataFrame(st.session_state.laporan_final)
        
        col_map, col_dl = st.columns([2, 1])
        
        with col_dl:
            st.subheader("📥 Rekapitulasi Akhir")
            # Tabel Ringkasan Total
            rekap_df = pd.DataFrame([st.session_state.rekap_aset]).T.reset_index()
            rekap_df.columns = ['Jenis Aset', 'Total Unit']
            st.table(rekap_df)
            
            csv = df_laporan.drop(columns=['b64_img']).to_csv(index=False).encode('utf-8')
            st.download_button("Download Laporan Inventaris (.csv)", data=csv, file_name='Laporan_Aset_Rel.csv', mime='text/csv')

        with col_map:
            st.subheader("🗺️ Peta Sebaran Kerusakan")
            df_hilang_map = df_laporan[df_laporan['Status'] == 'Hilang']
            if not df_hilang_map.empty:
                m = folium.Map(location=[df_hilang_map['Latitude'].mean(), df_hilang_map['Longitude'].mean()], zoom_start=15)
                folium.PolyLine(df_gps_video[['reconstructed_lat', 'reconstructed_lon']].values, color="blue", weight=2, opacity=0.4).add_to(m)
                
                for _, row in df_hilang_map.iterrows():
                    html = f'''<div style="font-family: Arial; width: 180px;">
                                <b style="color: red;">HILANG (ID: {row['Track_ID']})</b><br>
                                KM: {row['KM_Rentang']}<br>
                                <img src="data:image/jpeg;base64,{row['b64_img']}" style="width: 100%; margin-top:5px;">
                               </div>'''
                    folium.Marker([row['Latitude'], row['Longitude']], popup=folium.Popup(html, max_width=200), icon=folium.Icon(color='red')).add_to(m)
                st_folium(m, width=700, height=400)

    if st.session_state.gallery_hilang:
        st.divider()
        st.subheader("📸 Galeri Bukti Temuan Hilang")
        cols = st.columns(4)
        for idx, item in enumerate(reversed(st.session_state.gallery_hilang)):
            with cols[idx % 4]:
                st.image(item['img'], caption=f"ID: {item['id']} | KM: {item['km']}")
