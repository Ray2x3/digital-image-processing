"""Interactive Streamlit front end for Image Lab."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import streamlit as st

# Streamlit may put the script folder, rather than the project root, on sys.path.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.imaging import apply_kernel, decode_image, detect_faces, psnr, view_channels


def configure_chart_font() -> bool:
    """Load a local CJK font when available; return whether Chinese labels are safe."""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_CJK_FONT = configure_chart_font()

st.set_page_config(page_title="互動式影像處理與人臉分析", page_icon="🖼️", layout="wide")
st.title("互動式影像處理與人臉分析工具")
st.caption("影像載入・RGB/YUV 色彩分析・自訂卷積核・PSNR・Haar 人臉偵測")
uploaded = st.file_uploader("上傳 PNG、JPG/JPEG 或 BMP 影像", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is None:
    st.info("請先上傳一張影像以開始操作。")
    st.stop()
try:
    rgb, info = decode_image(uploaded.getvalue())
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("影像資訊")
st.write(f"解析度：{info.width} × {info.height} px　｜　色彩維度：{info.channels}（RGB）　｜　原始模式：{info.mode}　｜　格式：{info.format}　｜　大小：{info.file_size:,} bytes")
tab_color, tab_filter, tab_face = st.tabs(["色彩與直方圖", "自訂濾波與 PSNR", "人臉偵測"])
with tab_color:
    mode = st.selectbox("顯示模式", ["Original", "R", "G", "B", "Y", "U", "V"])
    shown, channels = view_channels(rgb, mode)
    left, right = st.columns(2)
    left.image(shown, caption=f"{mode} 顯示", use_container_width=True, clamp=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, plane in channels:
        ax.hist(plane.ravel(), bins=256, range=(0, 256), alpha=0.55, label=name)
    if HAS_CJK_FONT:
        xlabel, ylabel, title = "像素強度（0–255）", "像素數量", f"{mode} 直方圖"
    else:
        xlabel, ylabel, title = "Pixel intensity (0–255)", "Pixel count", f"{mode} Histogram"
    ax.set(xlim=(0, 255), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); ax.grid(alpha=0.2)
    right.pyplot(fig, clear_figure=True)

with tab_filter:
    preset = st.selectbox("濾波器範例", ["自訂", "Identity", "Gaussian 3×3", "Sharpen 3×3", "Emboss 3×3"])
    presets = {"Identity": [[0,0,0],[0,1,0],[0,0,0]], "Gaussian 3×3": [[1,2,1],[2,4,2],[1,2,1]],
               "Sharpen 3×3": [[0,-1,0],[-1,5,-1],[0,-1,0]], "Emboss 3×3": [[-2,-1,0],[-1,1,1],[0,1,2]]}
    kernel_text = st.text_area("輸入奇數邊長方陣，每列以逗號分隔", value="\n".join(", ".join(map(str, row)) for row in presets.get(preset, presets["Identity"])), height=110)
    normalize = st.checkbox("依卷積核總和正規化（總和為 0 時拒絕）", value=preset == "Gaussian 3×3")
    if st.button("執行濾波", type="primary"):
        try:
            rows = [[float(v.strip()) for v in line.split(",")] for line in kernel_text.strip().splitlines()]
            filtered = apply_kernel(rgb, rows, normalize)
            score = psnr(rgb, filtered)
            a, b = st.columns(2)
            a.image(rgb, caption="原圖", use_container_width=True)
            b.image(filtered, caption="濾波結果", use_container_width=True)
            st.metric("PSNR", "∞ dB（影像完全相同）" if np.isinf(score) else f"{score:.2f} dB")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))

with tab_face:
    if st.button("偵測人臉", type="primary"):
        try:
            marked, faces = detect_faces(rgb)
            st.image(marked, caption=f"偵測到 {len(faces)} 張人臉", use_container_width=True)
            st.write(f"人臉數量：{len(faces)}")
            st.caption("OpenCV Haar Cascade 不提供信心度分數，因此不呈現不實數值。")
            if faces: st.json(faces)
        except RuntimeError as exc:
            st.error(str(exc))
