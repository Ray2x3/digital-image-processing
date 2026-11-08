# 期中專題報告草稿大綱

> 請以本人實際資料、執行結果與截圖補齊方括號欄位。勿將規劃內容寫成已完成的部署或測試成果。

## Chapter 1：專案概述與環境建置

- 題目：基於 AI 輔助之互動式影像處理與人臉分析工具
- 語言：[Python 版本；目前開發環境為 Python 3.11.4]
- 介面：Streamlit；API：FastAPI；影像處理：Pillow、NumPy、OpenCV、Matplotlib
- 相依套件及用途：見 `requirements.txt`
- 建置方式、執行畫面：[補上實際操作截圖]

## Chapter 2：可執行連結

- Live Demo：[部署後填入公開網址]
- Source Code：[GitHub 公開 repository 網址]
- 驗證方式：[以未登入/公開視窗確認可檢閱]

## Chapter 3：程式功能與操作說明

### 架構與資料流

使用者上傳 PNG/JPG/BMP → Pillow 解碼並擷取尺寸/格式/檔案大小 → `app/imaging.py` 提供 RGB/YUV、卷積、PSNR、人臉偵測 → Streamlit 顯示圖片、直方圖與指標；FastAPI 提供對應 API。

### 功能實測紀錄

| 功能 | 測試輸入 | 預期/實際結果 | 證據 |
|---|---|---|---|
| 影像載入 | PNG、JPG、BMP；錯誤檔案 | [填寫] | [截圖] |
| RGB/YUV 與直方圖 | 彩色測試影像 | [填寫] | [截圖] |
| 卷積與 PSNR | Identity、高斯、銳化、零總和正規化 | [填寫] | [截圖] |
| 人臉偵測 | Audrey 單人、Star Trek 七人（含眼部遮擋與外星角色） | Audrey：1 張，0.93；Star Trek：7 張，0.86–0.94 | [補操作截圖] |

Star Trek 模型比較：Haar `minNeighbors=5` 偵測 6 張，降至 3 偵測 7 張；YuNet 固定分數門檻 0.50 偵測 7 張，逐框視覺檢查皆落在臉部區域。YuNet 的信心度分數不是校準後的正確機率。此結果只代表本測試圖，不代表所有影像上的整體準確率。

### PSNR

對 N 個像素樣本，`MSE = (1/N) Σ(Iᵢ − Kᵢ)²`。8-bit 影像峰值 `MAX=255`，故 `PSNR = 10 log₁₀(MAX²/MSE) dB`。若 `MSE=0`，PSNR 為 `∞ dB`。PSNR 衡量像素差異，不能單獨代表主觀視覺品質。

### 人臉偵測模型與限制

本專題使用 OpenCV Zoo YuNet 輕量 CNN，輸出邊界框及逐臉模型信心度；固定門檻為 0.50。信心度是模型分數，不等於校準後的真實正確機率。請以 Audrey 單人、Star Trek 多人及無人臉影像實測，填寫實際偵測數與分數。

## Chapter 4：AI 協作與 Prompt 歷程

逐次記錄實際使用的 AI 產品/模型名稱、日期、原始 prompt、AI 建議、採用或修改內容、錯誤與 refinement。不可補造未發生的 prompt 或模型版本。初始規格由使用者提供，AI 協作可如實記錄本次需求拆解與初始架構產出；後續逐步補記實作對話。
