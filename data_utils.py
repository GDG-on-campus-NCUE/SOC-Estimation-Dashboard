# data_utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_and_reshape_data(vct_path: str, soc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    根據規格書 II.A 規範，載入並重塑資料。
    - 載入 VCT 和 SOC 檔案。
    - 將 VCT 從一維 (3*N) 重塑為二維 (N, 3)。
    - 驗證 VCT 和 SOC 的樣本數是否一致。

    Args:
        vct_path (str): VCT CSV 檔案路徑。
        soc_path (str): SOC CSV 檔案路徑。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 重塑後的 VCT 矩陣和 SOC 向量。
    """
    print(f"正在載入資料: {vct_path} 與 {soc_path}")
    
    # 1. 載入資料
    soc_df = pd.read_csv(soc_path, header=None)
    vct_df = pd.read_csv(vct_path, header=None)

    # 2. 資料重塑與對齊
    soc_array = soc_df.values
    # 將長度為 3*N 的一維陣列重塑為 (N, 3) 的矩陣
    vct_array = vct_df.values.reshape(-1, 3)
    
    # 3. 資料驗證
    assert len(soc_array) == len(vct_array), \
        f"資料驗證失敗：VCT 樣本數 ({len(vct_array)}) 與 SOC 樣本數 ({len(soc_array)}) 不匹配！"
    
    print("資料載入與重塑成功。")
    return vct_array, soc_array


def create_time_series_features(vct_data: np.ndarray, soc_data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    根據規格書 II.B 規範，使用滑動窗口建立時間序列特徵。
    對於時間點 t，使用從 t-k+1 到 t 的數據作為特徵。

    Args:
        vct_data (np.ndarray): 形狀為 (N, 3) 的 VCT 矩陣。
        soc_data (np.ndarray): 形狀為 (N, 1) 的 SOC 向量。
        window_size (int): 回看窗口大小 (k)。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 特徵矩陣 X (形狀 N-k+1, 3*k) 和目標向量 y (形狀 N-k+1, 1)。
    """
    print(f"正在建立時間序列特徵，窗口大小 k={window_size}...")
    
    X, y = [], []
    num_samples = len(vct_data)
    
    # 從第 k-1 個時間點開始遍歷
    for i in range(window_size - 1, num_samples):
        # 提取從 t-k+1 到 t 的窗口數據
        window_features = vct_data[i - window_size + 1 : i + 1]
        
        # 將 (k, 3) 的窗口數據攤平成為一個長度為 3*k 的向量
        flattened_features = window_features.flatten()
        X.append(flattened_features)
        
        # 對應的目標是時間點 t 的 SOC 值
        y.append(soc_data[i])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"特徵建立完成。輸入特徵 X 形狀: {X.shape}, 目標 y 形狀: {y.shape}")
    return X, y

def normalize_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    根據規格書 II.B 規範，對資料進行正規化。
    - 輸入特徵 (X): 使用 MinMaxScaler，僅在訓練集上 fit。
    - 目標變數 (y): 除以 100，縮放到 [0, 1] 區間。

    Args:
        X_train, X_test, y_train, y_test: 訓練集與測試集的特徵和目標。

    Returns:
        Tuple: 正規化後的數據 (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled) 和已擬合的 scaler 物件。
    """
    print("正在進行資料正規化...")
    
    # 1. 正規化輸入特徵 (X)
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # 2. 正規化目標變數 (y)
    y_train_scaled = y_train / 100.0
    y_test_scaled = y_test / 100.0
    
    print("正規化完成。")
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler