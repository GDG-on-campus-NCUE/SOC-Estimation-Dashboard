# SOC Estimation Dashboard

以資料驅動的電池剩餘電量 (State of Charge, SOC) 估測與模擬平台，整合 PyTorch 訓練流程、Flask 即時服務與高互動儀表板，協助電池管理系統 (BMS) 團隊快速驗證演算法與情境。

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Socket.IO](https://img.shields.io/badge/Socket.IO-010101?logo=socketdotio)](https://socket.io/)
[![Chart.js](https://img.shields.io/badge/Chart.js-F5788D?logo=chartdotjs)](https://www.chartjs.org/)

---

## 目錄
- [專案概觀](#專案概觀)
- [核心功能](#核心功能)
- [系統架構與模組](#系統架構與模組)
- [技術棧](#技術棧)
- [環境建置](#環境建置)
  - [本地開發環境](#本地開發環境)
  - [使用 Docker](#使用-docker)
- [啟動與操作流程](#啟動與操作流程)
- [資料與模型管理](#資料與模型管理)
- [主要檔案對照表](#主要檔案對照表)
- [開發建議](#開發建議)
- [疑難排解](#疑難排解)

---

## 專案概觀
此儀表板針對實務電池管理需求打造，提供以下能力：

- 以 BP 類神經網路為核心的 SOC 估測模型，可由操作人員即時調整訓練超參數。
- 透過 Socket.IO 實現前後端雙向即時溝通，支援訓練過程監控、模型狀態回饋與模擬推論。
- 前端採暗色系的互動式儀表板，可視化訓練曲線、SOC 預測、誤差、延遲等指標。
- 允許對輸入電壓/電流/溫度加入雜訊並調整模擬速度，用以壓力測試模型穩健度。

---

## 核心功能
- **模型訓練流程**：在瀏覽器內輸入 Window Size、隱藏層配置、學習率與 Epochs，後端即時啟動 PyTorch 訓練並回傳 Loss、RMSE。
- **推論模擬儀表板**：即時呈現預測 SOC 與實際 SOC、電壓曲線與估測誤差，並提供圖例開關、資料點裁切等視覺化控制。
- **互動提示**：所有圖表與控制項皆附有說明提示，滑鼠懸浮即可閱讀專業指引。
- **資料匯出**：模擬完成後可一鍵匯出最新一次預測結果為 CSV，便於後續分析與稽核。
- **系統狀態監看**：側邊欄狀態燈以顏色區分待命、訓練、模擬與錯誤情境，同步顯示在事件日誌中。

---

## 系統架構與模組
```
SOC-Estimation-Dashboard/
├── app.py                 # Flask + Socket.IO 服務入口，統籌訓練與模擬流程
├── train.py               # 獨立離線訓練腳本，支援完整模型訓練與評估
├── model.py               # 建立 BP 神經網路結構的工廠函式
├── data_utils.py          # 資料載入、滑動窗口特徵與正規化流程
├── templates/
│   └── index.html         # 單頁式儀表板模板 (Tailwind CSS + 自訂樣式)
├── static/
│   └── js/
│       └── dashboard.js   # 前端互動邏輯、圖表渲染與 Socket.IO 事件處理
├── data/                  # 訓練與測試用電池量測資料 (VCT/SOC)
├── saved_model/           # 模型權重、Scaler 與設定檔輸出位置
└── requirements.txt       # Python 依賴套件清單
```

> 註：`saved_model/` 會在首次訓練完成後自動建立，無需手動新增。

---

## 技術棧
| 領域 | 技術 | 用途 |
| :--- | :--- | :--- |
| **後端** | Python 3.9+ / Flask | 提供 REST/SSE 頁面服務與 Socket.IO 事件處理 |
| | Flask-SocketIO | 管理前後端即時雙向通訊 |
| **機器學習** | PyTorch | 建立與訓練 BP 類神經網路 |
| | NumPy / Pandas / scikit-learn | 資料處理、正規化與評估 |
| **前端** | Tailwind CSS | 快速打造響應式暗色儀表板 |
| | Chart.js | 即時繪製訓練與模擬圖表 |
| | Vanilla JavaScript (ES6) | 控制畫面互動、提示、事件邏輯 |
| **開發支援** | Docker (可選) | 建立一致的部署與開發環境 |

---

## 環境建置
以下步驟以 macOS/Linux Bash 與 Windows PowerShell 為例，請依作業系統調整指令。

### 本地開發環境
1. **取得原始碼**
   ```bash
   git clone https://github.com/<YOUR-ORG>/SOC-Estimation-Dashboard.git
   cd SOC-Estimation-Dashboard
   ```
2. **建立虛擬環境**
   ```bash
   # 建議使用 .venv 作為虛擬環境名稱
   python -m venv .venv

   # 啟動虛擬環境
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   # .\.venv\Scripts\Activate.ps1
   ```
3. **更新 pip 並安裝依賴**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **驗證環境**
   ```bash
   python -m pip list | grep -E "flask|torch|socketio"
   ```
   確認主要套件均已正確安裝後即可進入啟動流程。

### 使用 Docker
> 若需在 CI/CD 或伺服器端部署，建議使用容器化方式。

1. 建立映像檔：
   ```bash
   docker build -t soc-estimation-dashboard:latest .
   ```
2. 啟動容器：
   ```bash
   docker run --rm -p 5000:5000 --name soc-dashboard soc-estimation-dashboard:latest
   ```
   服務啟動後同樣可於 `http://127.0.0.1:5000` 存取儀表板。

---

## 啟動與操作流程
1. **啟動後端**
   ```bash
   python app.py
   ```
   伺服器預設監聽於 `http://127.0.0.1:5000`，終端機會輸出模型載入狀態。
2. **開啟瀏覽器**：於 Chrome/Edge/Firefox 造訪上述網址。
3. **首次使用建議流程**：
   - 切換至「訓練流程」頁籤，設定 Window Size、隱藏層與學習率。
   - 按下「重新訓練模型」，等待訓練完成並觀察曲線圖。
   - 訓練完成後，系統會自動載入最新模型並回到儀表板。
4. **模擬操作**：
   - 在「儀表板」頁面可調整速度倍率與雜訊。
   - 點擊「開始模擬」後即時觀察 SOC、電壓曲線與延遲指標。
   - 模擬中可隨時按「停止」或導出 CSV 檔案以保存結果。

---

## 資料與模型管理
- `data/` 內建 CASE_4 系列 VCT/SOC 原始資料，可替換為自有數據，惟需維持相同欄位順序與轉置格式。
- 訓練後的權重 (`soc_model_weights.pth`)、正規化器 (`x_scaler.pkl`) 與模型結構 (`model_config.json`) 會存放於 `saved_model/`，Flask 服務啟動時會嘗試載入。
- 若需批次訓練，可直接執行：
  ```bash
  python train.py
  ```
  完成後同樣會在 `saved_model/` 生成最新模型與訓練監看圖 (`training_monitoring.png`)。

---

## 主要檔案對照表
| 檔案/資料夾 | 說明 |
| :--- | :--- |
| `app.py` | Web 服務主程式，包含 Socket.IO 事件、訓練與模擬執行緒。 |
| `static/js/dashboard.js` | 前端核心腳本：Chart.js 設定、提示定位、模擬資料處理。 |
| `templates/index.html` | 儀表板 UI 範本，整合 Tailwind CSS 與元件樣式。 |
| `data_utils.py` | 資料重塑、時間序列特徵建立與正規化。 |
| `model.py` | 產生 BP 類神經網路的工廠函式。 |
| `train.py` | CLI 訓練流程，可輸出訓練曲線與評估指標。 |
| `requirements.txt` | Python 依賴套件列表。 |

---

## 開發建議
- 修改前端程式碼後可直接重新整理瀏覽器，Socket.IO 會重新連線並載入最新腳本。
- 若新增 Python 模組，請同步更新 `requirements.txt` 並於虛擬環境中重新安裝。
- 建議在提交程式碼前執行：
  ```bash
  python -m compileall .
  ```
  以快速檢查 Python 語法錯誤。

---

## 疑難排解
| 問題 | 可能原因 | 解決方案 |
| :--- | :--- | :--- |
| 瀏覽器無法連線 | 伺服器未啟動或防火牆阻擋 | 確認 `python app.py` 已執行，必要時開放 5000 連接埠。 |
| 訓練未開始 | 仍有前一次訓練執行緒 | 檢查終端是否顯示「Training completed」，必要時重新啟動服務。 |
| 模擬按鈕為灰色 | 模型尚未載入 | 需先完成訓練或放置有效的模型檔於 `saved_model/`。 |
| CSV 匯出失敗 | 尚未有模擬紀錄 | 需先執行一次完整模擬流程再匯出。 |

---

如需更多資訊或進一步整合，歡迎於 Issue 區回報或提出建議。
