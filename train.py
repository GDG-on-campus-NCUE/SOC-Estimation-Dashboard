import os
import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data_utils import load_and_reshape_data, create_time_series_features, normalize_data
from model import create_bp_model

# --- 模型超參數配置 (對應規格書表 3.1) ---
class TrainingConfig:
    WINDOW_SIZE = 5
    HIDDEN_LAYERS = [64, 32]
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 128
    VALIDATION_SPLIT = 0.2 # 從訓練資料中切分 20% 作為驗證集
    
    # 檔案路徑設定
    TRAIN_VCT_PATH = 'data/CASE_4_Train_VCT_transposed.csv'
    TRAIN_SOC_PATH = 'data/CASE_4_Train_SOC_transposed.csv'
    TEST_VCT_PATH = 'data/CASE_4_Test_VCT_transposed.csv'
    TEST_SOC_PATH = 'data/CASE_4_Test_SOC_transposed.csv'
    
    # 模型儲存路徑
    MODEL_SAVE_DIR = 'saved_model'
    MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, 'soc_model_weights.pth')
    MODEL_CONFIG_PATH = os.path.join(MODEL_SAVE_DIR, 'model_config.json')
    SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'x_scaler.pkl')
    # 監看結果圖表的儲存路徑
    MONITORING_PLOT_PATH = os.path.join(MODEL_SAVE_DIR, 'training_monitoring.png')

def main():
    """主訓練流程"""
    cfg = TrainingConfig()
    
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # --- 1. 資料管線 ---
    # 載入完整的訓練和測試資料
    train_vct_full, train_soc_full = load_and_reshape_data(cfg.TRAIN_VCT_PATH, cfg.TRAIN_SOC_PATH)
    test_vct, test_soc = load_and_reshape_data(cfg.TEST_VCT_PATH, cfg.TEST_SOC_PATH)

    # 建立時間序列特徵
    X_full, y_full = create_time_series_features(train_vct_full, train_soc_full, cfg.WINDOW_SIZE)
    X_test, y_test = create_time_series_features(test_vct, test_soc, cfg.WINDOW_SIZE)
    
    # <--- 將完整訓練集切分為新的訓練集和驗證集 ---
    print(f"將訓練資料以 {1-cfg.VALIDATION_SPLIT:.0%}:{cfg.VALIDATION_SPLIT:.0%} 的比例切分為訓練集與驗證集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=cfg.VALIDATION_SPLIT, random_state=42, shuffle=True
    )
    
    # 正規化: Scaler 只在新的、較小的訓練集上 fit
    X_train_s, X_val_s, y_train_s, y_val_s, x_scaler = normalize_data(X_train, X_val, y_train, y_val)
    # 測試集也用同一個 scaler 轉換
    X_test_s = x_scaler.transform(X_test)
    y_test_s = y_test / 100.0

    # --- 2. 轉換為 PyTorch Tensors 並建立 DataLoader ---
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # --- 3. 建立模型 ---
    input_size = X_train.shape[1]
    output_size = 1
    layer_config = [input_size] + cfg.HIDDEN_LAYERS + [output_size]
    
    model = create_bp_model(layer_config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # <--- 用於記錄監看數據的列表 ---
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []
    }

    # --- 4. 訓練迴圈 ---
    print("\n--- 開始模型訓練 ---")
    # <--- 使用 tqdm 建立進度條 ---
    for epoch in tqdm(range(cfg.EPOCHS), desc="整體訓練進度"):
        # --- 訓練模式 ---
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_train_loss = np.mean(batch_losses)
        
        # --- 驗證模式 ---
        model.eval()
        val_batch_losses = []
        all_val_preds = []
        all_val_true = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_outputs = model(batch_X_val)
                val_loss = criterion(val_outputs, batch_y_val)
                val_batch_losses.append(val_loss.item())
                all_val_preds.append(val_outputs.numpy())
                all_val_true.append(batch_y_val.numpy())
        
        epoch_val_loss = np.mean(val_batch_losses)
        
        # 每 10 個 epoch 記錄一次數據並印出
        if (epoch + 1) % 10 == 0:
            # 反正規化以計算真實世界的誤差
            val_preds_rescaled = np.concatenate(all_val_preds) * 100
            val_true_rescaled = np.concatenate(all_val_true) * 100
            
            val_rmse = np.sqrt(np.mean((val_preds_rescaled - val_true_rescaled)**2))
            val_mae = np.mean(np.abs(val_preds_rescaled - val_true_rescaled))
            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['val_rmse'].append(val_rmse)
            history['val_mae'].append(val_mae)

            tqdm.write(f"Epoch [{epoch+1}/{cfg.EPOCHS}] | "
                       f"Train Loss: {epoch_train_loss:.6f} | "
                       f"Val Loss: {epoch_val_loss:.6f} | "
                       f"Val RMSE: {val_rmse:.4f}% | "
                       f"Val MAE: {val_mae:.4f}%")

    print("--- 訓練完成 ---\n")

    # --- 5. 在最終的「測試集」上評估模型 ---
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)
        predictions_rescaled = predictions.numpy() * 100
        y_test_rescaled = y_test_t.numpy() * 100
        
        rmse = np.sqrt(np.mean((predictions_rescaled - y_test_rescaled)**2))
        mae = np.mean(np.abs(predictions_rescaled - y_test_rescaled))

        print("--- 最終模型評估結果 (在獨立測試集上) ---")
        print(f"均方根誤差 (RMSE): {rmse:.4f} %")
        print(f"平均絕對誤差 (MAE): {mae:.4f} %")
        print("-----------------------------------------")

    # <--- 繪製並儲存監看圖表 ---
    print(f"正在繪製監看圖表並儲存至 {cfg.MONITORING_PLOT_PATH}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('訓練過程監看圖表', fontsize=16)

    # 圖一：訓練損失 vs 驗證損失
    ax1.plot(history['epoch'], history['train_loss'], 'b-o', label='訓練損失 (Training Loss)')
    ax1.plot(history['epoch'], history['val_loss'], 'r-o', label='驗證損失 (Validation Loss)')
    ax1.set_yscale('log')
    ax1.set_title('損失函數變化曲線 (Loss Curve)')
    ax1.set_ylabel('損失 (MSE, log scale)')
    ax1.legend()
    ax1.grid(True)

    # 圖二：驗證集的 RMSE 和 MAE
    ax2.plot(history['epoch'], history['val_rmse'], 'g-s', label='均方根誤差 (RMSE %)')
    ax2.plot(history['epoch'], history['val_mae'], 'm-^', label='平均絕對誤差 (MAE %)')
    ax2.set_title('驗證集效能指標 (Validation Metrics)')
    ax2.set_xlabel('訓練週期 (Epochs)')
    ax2.set_ylabel('誤差百分比 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(cfg.MONITORING_PLOT_PATH)
    plt.show()

    # --- 6. 儲存模型產物 ---
    print(f"\n正在儲存模型至 '{cfg.MODEL_SAVE_DIR}' 資料夾...")
    torch.save(model.state_dict(), cfg.MODEL_WEIGHTS_PATH)
    with open(cfg.MODEL_CONFIG_PATH, 'w') as f:
        json.dump(layer_config, f)
    joblib.dump(x_scaler, cfg.SCALER_PATH)
    print("模型儲存成功！")


if __name__ == '__main__':
    main()