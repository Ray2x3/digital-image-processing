# 互動式影像處理與人臉分析

以 Python 3.11、Streamlit、FastAPI、NumPy、OpenCV 與 Pillow 建立的影像處理專題。支援影像中繼資料、RGB/YUV 分量與直方圖、自訂二維卷積核、PSNR，以及 YuNet 人臉框與逐臉信心度。

## 啟動

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
```

預設影像為 Audrey Hepburn JPG；下拉選單可切換同圖 PNG/BMP、Star Trek 多人臉測試圖，或上傳其他影像。預設素材與 YuNet 模型放在 `assets/`，會隨 repository 部署；本機原始測試資料夾不參與執行。

FastAPI 可另開終端啟動：

```powershell
uv run uvicorn app.api:app --reload
```

## 架構

```text
影像來源 → Streamlit 介面 ─┐
                           ├→ app/imaging.py → Pillow / NumPy / OpenCV
HTTP 請求 → FastAPI ───────┘
```

`app/streamlit_app.py` 負責介面、圖表和互動；`app/api.py` 提供 HTTP 入口；`app/imaging.py` 集中實作讀檔、色彩轉換、卷積、品質指標和人臉偵測，供兩種介面共用。

## 影像處理原理

### 色彩分量與直方圖

解碼後將影像表示成 `H × W × 3` 的 8-bit RGB 陣列。R/G/B 檢視保留選定分量並把另兩個分量設為 0；因此 R 呈紅色、G 呈綠色、B 呈藍色。各通道直方圖統計像素強度 `k∈[0,255]` 的次數：

`h(k) = #{p | I(p) = k}`

YUV 將亮度與色度分開。OpenCV 的 RGB→YUV 轉換可概略表示為 `Y≈0.299R+0.587G+0.114B`，U/V 是相對於亮度的色差分量（顯示值以 128 為中心）。各分量以灰階強度圖與對應直方圖呈現。

### 空間卷積與邊界

對每個像素取周圍鄰域，依核權重加權求和：

`O(x,y) = ΣᵢΣⱼ K(i,j) I(x−i,y−j)`

核需為奇數邊長方陣，最大 15×15。程式以複製邊界像素處理影像外側，結果裁切至 0–255。高斯核常以權重總和正規化（例如 3×3 核總和為 16，除以 16 後可大致保留亮度）；銳化和浮雕核則保留其差分/增強權重，不應任意正規化。OpenCV `filter2D` 計算相關運算，因此程式翻轉核後實作數學卷積。

### PSNR

以原圖 `I` 與濾波結果 `K` 的所有 RGB 樣本計算均方誤差：

`MSE = (1/N) Σ(Iᵢ−Kᵢ)²`

對 8-bit 影像，`MAX=255`，故：

`PSNR = 10 log₁₀(MAX²/MSE) dB`

PSNR 越高代表像素差異通常越小；它不完全等同主觀視覺品質。完全相同時 MSE 為 0，PSNR 為 `∞ dB`。

### 人臉偵測

使用 OpenCV Zoo 的 YuNet 輕量 CNN 人臉偵測器，回傳每張臉的邊界框與模型分數；本專題固定以 0.50 分數門檻篩選，使用者按下偵測即可得到總數、逐臉框和信心度。此分數是模型輸出，不代表校準後的真實正確機率。模型檔為 `assets/models/face_detection_yunet_2026may.onnx`，其動態輸入尺寸支援不同解析度；YuNet 可偵測約 10×10 至 300×300 像素範圍內的人臉。[OpenCV Zoo 模型說明與 MIT 授權](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。

## FastAPI 路由

- `GET /health`：服務狀態
- `POST /image/info`：影像中繼資料
- `POST /image/filter`：卷積與 PSNR
- `POST /image/faces`：偵測人臉、框座標與逐臉信心度

## 專案資料

- `assets/samples/`：四張 Demo 預設圖（Audrey JPG/PNG/BMP、Star Trek 多人臉）
- `assets/models/`：OpenCV Zoo YuNet ONNX 模型
- `app/`：介面、API 與影像核心
- `scripts/convert_image.py`：將 JPEG 轉成 PNG/BMP，並逐像素檢查輸出
- `pyproject.toml`、`uv.lock`：Python 相依套件與鎖定版本
- `docs/report_outline.md`：報告章節與實測紀錄大綱
