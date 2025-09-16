# 基於BP類神經網路的電池SOC估測與分析平台

**版本：2.0 (網頁應用程式)**
**日期：2025年9月16日**

[![zh-TW](https://img.shields.io/badge/語言-繁體中文-blue.svg)](README.md)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)](https://www.docker.com/)

本專案是一個先進的電池剩餘電量 (State of Charge, SOC) 估測與分析平台。它從一份嚴謹的技術規格書概念出發，透過現代化的軟體工程方法，最終實現為一個功能豐富、互動性強的網頁應用程式。使用者不僅能在瀏覽器中動態設定模型參數、即時監看訓練過程，還能在訓練完成後，於一個高擬真度的儀表板上進行模擬推論與參數擾動分析。

![應用程式主介面](https://i.imgur.com/8Q0b1yB.png)

---

## I. 核心功能與特色

- **現代化使用者介面**: 採用 Flask、Socket.IO 與 Tailwind CSS 技術棧，打造出專業、美觀且能適應不同裝置的暗色主題儀表板。
- **獨立的作業流程**: 應用程式內建**模型訓練**與**即時推論**兩種模式，使用者可無縫切換，符合專業機器學習的開發維運流程。
- **智慧化模型管理**: 系統啟動時會自動偵測並載入已訓練好的模型，同時提供介面讓使用者隨時能以新的參數重新訓練。
- **即時訓練監控**: 透過 WebSocket 技術，前端介面可即時、逐週期地視覺化呈現訓練損失、驗證損失和驗證集均方根誤差，讓訓練過程的細節完全透明。
- **高互動模擬儀表板**:
    - **速度調控**: 可透過滑桿自由調整模擬數據的更新速率。
    - **雜訊注入**: 可手動為電壓、電流、溫度等觀測值加入微小擾動，用以測試模型的穩定性與強健性。
- **專業化工程實踐**:
    - **環境隔離**: 透過 `venv` 虛擬環境進行本地開發，並提供 `Dockerfile` 實現與雲端一致的、可重現的訓練環境。
    - **雲端就緒**: 訓練腳本經過特殊設計，可直接部署至 Google Cloud Vertex AI 等雲端平台進行規模化訓練。

---

## II. 系統架構與技術選型

本系統的架構嚴格遵循初始規格書的設計理念，將**離線訓練**與**線上推論**兩個工作流程，優雅地整合在一個統一的網頁應用程式中。

### A. 專案結構

SOC_Estimation_System/
├── data/                 # 存放原始 CSV 數據集
├── saved_model/          # 存放訓練完成的模型產物
├── static/
│   └── js/
│       └── dashboard.js  # 前端互動邏輯核心
├── templates/
│   └── index.html      # 網頁 UI 主介面
├── app.py                # Flask 後端主程式 (整合訓練與推論)
├── data_utils.py         # 資料管線與預處理模組
├── model.py              # PyTorch 模型工廠
├── train.py              # (可選) 獨立的離線/雲端訓練腳本
├── Dockerfile            # Docker 容器設定檔
└── requirements.txt      # Python 依賴套件列表


### B. 技術選型 (Technology Stack)

| 類別 | 技術 | 說明 |
| :--- | :--- | :--- |
| **後端** | Python 3.9+ | 主要開發語言。 |
| | Flask | 輕量級網頁伺服器框架，提供服務接口。 |
| | Flask-SocketIO | 實現後端與前端之間的即時雙向通訊。 |
| **機器學習** | PyTorch | 核心深度學習框架，用於建構與訓練神經網路。 |
| | Pandas | 強大的資料處理與分析工具。 |
| | NumPy | 高效能科學計算函式庫。 |
| | Scikit-learn | 用於資料預處理 (如 `MinMaxScaler`) 和模型評估。 |
| **前端** | HTML5 | 網頁結構。 |
| | Tailwind CSS | 現代化的樣式工具，快速打造專業介面。 |
| | JavaScript (ES6+) | 處理前端所有互動邏輯。 |
| | Chart.js | 功能強大且美觀的圖表繪製函式庫。 |
| **容器化** | Docker | 建立標準化、可移植的應用程式執行環境。 |

---

## III. 演算法與模型細節

### A. 資料預處理流程

資料管線確保數據在進入模型前的每一階段都經過嚴格且一致的處理。

1.  **資料載入與重塑**:
    - 從 `*_transposed.csv` 檔案讀取一維數據流。
    - 將 VCT (電壓、電流、溫度) 數據重塑為 `(N, 3)` 的矩陣，其中 N 為樣本總數。
    - 透過斷言驗證 VCT 與 SOC 數據的樣本數是否一致。
2.  **特徵工程 (滑動窗口)**:
    - 建立一個大小為 $k$ 的「回看窗口」。
    - 對於時間點 $t$，將從 $t-k+1$ 到 $t$ 的所有 VCT 數據攤平，形成一個長度為 $3 \times k$ 的特徵向量。
    - 最終產生特徵矩陣 $X$ (形狀 `(N-k+1, 3*k)`) 與目標向量 $y$ (形狀 `(N-k+1, 1)`)。
3.  **資料正規化**:
    - **輸入特徵 (X)**: 使用 `MinMaxScaler` 將所有特徵縮放到 `[0, 1]` 區間。正規化模型僅在訓練集上學習，再應用於所有數據集。
    - **目標變數 (y)**: 將 SOC 百分比 (`0-100`) 直接除以 100，縮放到 `[0, 1]` 區間，以匹配模型輸出層的 Sigmoid 激活函數。

### B. BP 類神經網路模型

模型架構採用「模型工廠」設計模式，可由使用者在介面上動態配置。

- **輸入層**: 神經元數量由回看窗口大小 $k$ 動態決定，為 $3 \times k$。
- **隱藏層**: 可由使用者自訂層數與每層的神經元數量。隱藏層之間的激活函數使用 **ReLU (修正線性單元)**。
- **輸出層**: 由一個線性層和一個 **Sigmoid** 激活函數構成，輸出一介於 `(0, 1)` 之間的數值，代表正規化後的 SOC。

### C. 核心演算法與公式

- **損失函數 (Loss Function)**: 採用**均方誤差 (Mean Squared Error, MSE)**，衡量預測值與真實值之間的差距。
  $$
  \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2
  $$
- **優化器 (Optimizer)**: 使用 **Adam**，這是一種高效且穩健的自適應學習率優化演算法。
- **評估指標 (Evaluation Metric)**: 除了 MSE，系統還計算並顯示**均方根誤差 (Root Mean Squared Error, RMSE)**，其單位與目標變數相同（%），更具解釋性。
  $$
  \text{RMSE} = \sqrt{\text{MSE}}
  $$

---

## IV. 如何重現 (Step-by-Step Guide)

請遵循以下步驟在您的本地環境中設置並運行此專案。

### A. 環境準備

1. **安裝 Python**: 確保您的系統已安裝 Python 3.9 或更高版本的 **64 位元** 版本。
2. **安裝 Git**: 用於從版本庫中複製專案。
3. **(可選) 安裝 Docker Desktop**: 如果您希望使用容器化的方式進行訓練。

### B. 專案設置

1.  **複製專案**:
    ```bash
    git clone [您的專案 Git URL]
    cd SOC_Estimation_System
    ```

2.  **建立並啟動虛擬環境**:
    ```bash
    # 建立 venv
    python -m venv venv

    # 啟動 venv (Windows PowerShell)
    .\venv\Scripts\activate

    # 啟動 venv (macOS/Linux)
    # source venv/bin/activate
    ```
    成功啟動後，您的終端機提示符前會出現 `(venv)`。

3.  **安裝依賴套件**:
    ```bash
    pip install -r requirements.txt
    ```

### C. 運行應用程式

1.  **啟動後端伺服器**:
    ```bash
    python app.py
    ```
    伺服器啟動後，會自動檢查是否存在已訓練好的模型。

2.  **打開瀏覽器**:
    - 在您的 Chrome, Edge 或 Firefox 瀏覽器中，訪問 `http://127.0.0.1:5000`。

3.  **操作流程**:
    - **首次運行**: 系統會提示模型不存在。請切換到 "Training" 頁面，使用預設參數或自訂參數，點擊「Re-train Model」按鈕進行訓練。
    - **訓練完成後**: 介面會自動跳轉回 "Dashboard" 頁面。
    - **進行模擬**: 在 "Dashboard" 頁面，點擊「Start」按鈕開始即時推論。您可以透過左側的滑桿和輸入框即時調整模擬速度與觀測雜訊。

### D. (可選) 使用 Docker 進行本地訓練

如果您希望在一個隔離的環境中進行訓練，可以依照 `DOCKER_TRAINING.md` (需自行建立) 中的指南進行操作。

---

## V. 未來展望

基於目前穩固的架構，未來的發展方向包括：

- **演算法增強**: 引入 LSTM 或 GRU 等更擅長處理時間序列的循環神經網路模型，並在介面上提供模型選擇功能。
- **即時資料整合**: 開發一個可以透過序列埠 (Serial Port) 或網路通訊端 (Socket) 直接從硬體（如電池充放電機）接收即時數據的模組。
- **雲端部署與推論**: 將訓練好的模型部署到 GCP Vertex AI Endpoint，並讓網頁應用直接呼叫雲端 API 進行推論，實現真正的雲端一體化。
- **模型管理系統**: 建立一個更完善的系統來管理不同版本、不同參數訓練出的模型，並追蹤它們的性能指標。