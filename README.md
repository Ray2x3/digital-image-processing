# 互動式影像處理與人臉分析工具

以 Python、Streamlit 和 FastAPI 製作的影像處理期中專題。支援 PNG/JPG/JPEG/BMP、RGB/YUV 通道與直方圖、自訂空間卷積核、PSNR，以及 OpenCV Haar Cascade 人臉偵測。

## 專案規劃與架構

```text
使用者影像 → Streamlit 互動介面（上傳、顯示、參數、圖表）
                 ↓ 共用影像核心
       app/imaging.py ←→ FastAPI（/health、/image/info、/image/filter、/image/faces）
       Pillow：解碼與中繼資訊；NumPy/OpenCV：色彩、卷積、人臉；Matplotlib：直方圖
```

先完成本機影像處理，再補上操作截圖與測試紀錄，接著推送 GitHub、部署公開展示，最後整理完整 PDF 報告與 AI 協作歷程。報告需填入真實公開網址及本人實測結果。

## 環境建置

使用 Python 3.11 或更新版本。已安裝 uv 時：

```powershell
uv sync
uv run streamlit run app/streamlit_app.py
uv run uvicorn app.api:app --reload
```

若尚未安裝 uv，可先用 Python venv：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
uv sync
uv run streamlit run app/streamlit_app.py
```

## Git 與 GitHub 入門

Git 是電腦上的版本記錄；GitHub 是分享程式碼的遠端網站。先在 GitHub 建立空白 **Public** repository，再在本資料夾執行：

```powershell
git init
git add .
git commit -m "建立影像處理專題初始版本"
git branch -M main
git remote add origin https://github.com/你的帳號/你的repository.git
git push -u origin main
```

每完成一小段功能，使用 `git status` 查看改動、`git add .` 加入版本、`git commit -m "描述這次完成的事"` 留存，再用 `git push` 上傳。將範例網址換成自己的 GitHub repository；不要把密碼或 token 放進程式碼。

## PSNR

`MSE = (1/N) Σ(I − K)²`；`PSNR = 10 log10(MAX²/MSE)`，本專題 `MAX=255`。原圖與結果完全相同時 `MSE=0`，PSNR 為無限大。

## JPEG 轉 PNG/BMP

使用 Pillow 的驗證式轉檔工具：

```powershell
uv run python scripts/convert_image.py "test_img/audrey hepburn.JPG"
```

預設將 PNG 和 BMP 輸出至來源資料夾。程式不覆寫既有檔案，並在留下輸出前重新解碼、逐像素比對 RGB；若格式不符、讀取失敗或像素不一致，會停止並清除暫存/部分輸出。JPEG 原有的有損壓縮無法由轉檔還原，但轉出的 PNG/BMP 像素應與「解碼後的 JPG」完全相同。

## 已知限制與後續項目

Haar Cascade 能回傳人臉框與數量，但模型本身不提供可信的 confidence score；如評分必須有信心度，需改用輸出 confidence 的模型。報告檔名需為 `學號_姓名_期中專題.pdf`，並補入本人操作截圖、邊界測試、AI prompt 歷程、GitHub 與公開展示網址。
