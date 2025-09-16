# model.py

import torch
import torch.nn as nn
from typing import List

def create_bp_model(layer_config: List[int]) -> nn.Sequential:
    """
    根據規格書 III.A 的「模型工廠」設計模式，動態建構 BP 類神經網路。

    Args:
        layer_config (List[int]): 定義網路結構的整數列表。
                                  例如: [input_size, hidden_1_size, hidden_2_size, output_size]
                                  範例: [15, 64, 32, 1]

    Returns:
        nn.Sequential: 一個 PyTorch 的序列模型。
    """
    layers = []
    
    # 遍歷設定檔，建立隱藏層
    for i in range(len(layer_config) - 2):
        layers.append(nn.Linear(layer_config[i], layer_config[i+1]))
        layers.append(nn.ReLU()) # 隱藏層使用 ReLU 活化函數
    
    # 建立輸出層
    # 規格書 III.A 要求：輸出層為一個 Linear 層，緊跟一個 Sigmoid
    layers.append(nn.Linear(layer_config[-2], layer_config[-1]))
    layers.append(nn.Sigmoid())
    
    model = nn.Sequential(*layers)
    print("--- BP 神經網路模型已建立 ---")
    print(model)
    print("--------------------------")
    
    return model