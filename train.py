# bpnn_train_test.py
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 中文字體
plt.rcParams['axes.unicode_minus'] = False  # 負號顯示
# ======================
# 0) 參數
# ======================
DATA_CSV = "battery_data.csv"   # 輸入的電池資料 (需含 Voltage, Current, Temperature, SOC)
MODEL_PKL = "bpnn_model.pkl"
SCALER_PKL = "bpnn_scaler.pkl"
PRED_CSV  = "bpnn_test_predictions.csv"
RANDOM_SEED = 42

start = time.time()

# ======================
# 1) 讀資料
# ======================
df = pd.read_csv(DATA_CSV)

# 確認必要欄位
required = ["Voltage", "Current", "Temperature", "SOC"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少欄位：{missing}，請確認檔案內容。")

df = df.dropna(subset=required).copy()

# 特徵 (輸入)
X = df[["Voltage", "Current", "Temperature"]].values
# 標籤 (輸出)
y = df["SOC"].values

# ======================
# 2) 8:2 切分
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)

# ======================
# 3) 標準化
# ======================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ======================
# 4) BPNN (MLPRegressor + logistic)
# ======================
bpnn = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),  # 可調
    activation="logistic",            # sigmoid
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=3000,
    early_stopping=True,
    n_iter_no_change=30,
    random_state=RANDOM_SEED,
    verbose=False
)

bpnn.fit(X_train_s, y_train)

# ======================
# 5) 評估
# ======================
def eval_and_print(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"- MSE : {mse:.6f}")
    print(f"- RMSE: {rmse:.6f}")
    print(f"- MAE : {mae:.6f}")
    print(f"- R²  : {r2:.6f}")
    return mse, rmse, mae, r2

y_pred_train = bpnn.predict(X_train_s)
y_pred_test  = bpnn.predict(X_test_s)

eval_and_print("訓練集", y_train, y_pred_train)
eval_and_print("測試集",  y_test,  y_pred_test)

# ======================
# 6) 匯出測試集預測 CSV
# ======================
pred_df = pd.DataFrame(X_test, columns=["Voltage", "Current", "Temperature"])
pred_df["Actual_SOC"]     = y_test
pred_df["Predicted_SOC"]  = y_pred_test
pred_df["Error"]          = pred_df["Predicted_SOC"] - pred_df["Actual_SOC"]
pred_df["AbsError"]       = pred_df["Error"].abs()
pred_df.to_csv(PRED_CSV, index=False)
print(f"\n✅ 已輸出測試集預測：{PRED_CSV}")

# ======================
# 7) 存模型與標準化器
# ======================
joblib.dump(bpnn, MODEL_PKL)
joblib.dump(scaler, SCALER_PKL)
print(f"✅ 已儲存模型：{MODEL_PKL}")
print(f"✅ 已儲存Scaler：{SCALER_PKL}")

# ======================
# 8) 繪圖
# ======================

# (a) Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, label="Test")
minv = min(y_test.min(), y_pred_test.min(), 0.0)
maxv = max(y_test.max(), y_pred_test.max(), 100.0)  # SOC 通常 0~100
plt.plot([minv, maxv], [minv, maxv], 'r--', label="Ideal")
plt.xlabel("Actual SOC")
plt.ylabel("Predicted SOC")
plt.title("BPNN (MLP, logistic) - Actual vs Predicted (Test)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# (b) Loss Curve
if hasattr(bpnn, "loss_curve_"):
    plt.figure(figsize=(8, 4))
    plt.plot(bpnn.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("BPNN Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# (c) SOC 曲線圖 (實際 vs 預測)
plt.figure(figsize=(10, 5))
num_cycles = 1  # 假設 6 個循環
x_axis = np.linspace(0, num_cycles, len(y_test))

plt.plot(x_axis, y_test, label="實際值", color="blue", linewidth=1)
plt.plot(x_axis, y_pred_test, label="預測值", color="red", linestyle="--", linewidth=1)

plt.xlabel("循環次數")
plt.ylabel("SOC (%)")
plt.title("實際值 vs 預測值 曲線圖 (BPNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\n⏱️ 執行時間：{time.time() - start:.2f} 秒")