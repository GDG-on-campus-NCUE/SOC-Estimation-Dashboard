import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.scalers = {}

    def load_and_prepare_data(self, data_key):
        if data_key not in self.file_paths:
            raise FileNotFoundError(f"找不到數據集金鑰: {data_key}")

        paths = self.file_paths[data_key]
        try:
            df_soc = pd.read_csv(paths["soc"], header=None, names=['SOC'])
            df_vct = pd.read_csv(paths["vct"], header=None)
            df_vct = pd.DataFrame(df_vct.values.reshape(-1, 3), columns=['電壓', '電流', '溫度'])
            
            if len(df_soc) != len(df_vct):
                raise ValueError("SOC 和 VCT 檔案行數不一致")

            df_combined = pd.concat([df_vct, df_soc], axis=1)
            
            # --- 數據標準化 ---
            # 儲存 scaler 以便反向轉換
            self.scalers[data_key] = {
                '電壓': {'max': df_combined['電壓'].max(), 'min': df_combined['電壓'].min()},
                '電流': {'max': df_combined['電流'].max(), 'min': df_combined['電流'].min()},
                '溫度': {'max': df_combined['溫度'].max(), 'min': df_combined['溫度'].min()},
                'SOC': {'max': df_combined['SOC'].max(), 'min': df_combined['SOC'].min()}
            }
            
            for col in ['電壓', '電流', '溫度', 'SOC']:
                max_val = self.scalers[data_key][col]['max']
                min_val = self.scalers[data_key][col]['min']
                df_combined[f'標準化_{col}'] = (df_combined[col] - min_val) / (max_val - min_val)

            # --- 加入歷史數據特徵 ---
            for col in ['標準化_電壓', '標準化_電流', '標準化_溫度']:
                for i in [1, 2, 3, 4, 5]: # 建立 t-1 到 t-5 的特徵
                    df_combined[f'{col}_t-{i}'] = df_combined[col].shift(i)

            df_combined.dropna(inplace=True)
            df_combined.reset_index(drop=True, inplace=True)
            
            return df_combined

        except FileNotFoundError as e:
            raise FileNotFoundError(f"找不到數據檔案: {e.filename}")
        except Exception as e:
            raise RuntimeError(f"讀取或處理數據時發生錯誤: {e}")