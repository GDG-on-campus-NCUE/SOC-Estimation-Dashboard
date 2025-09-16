import numpy as np

class NeuralNetwork:
    def __init__(self, layer_config):
        self.layer_config = layer_config
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(len(self.layer_config) - 1):
            input_dim = self.layer_config[i]
            output_dim = self.layer_config[i+1]
            # He Initialization
            weight = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            bias = np.zeros((1, output_dim))
            self.weights.append(weight)
            self.biases.append(bias)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _relu(self, x):
        return np.maximum(0, x)
        
    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._relu(z) # 隱藏層使用 ReLU
            activations.append(a)
        
        # 輸出層
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_out = self._sigmoid(z_out) # 輸出層使用 Sigmoid
        activations.append(a_out)
        
        return activations

    def train(self, X, y, epochs, learning_rate, progress_callback=None):
        history = {'loss': []}
        for i in range(epochs):
            # 前向傳播
            activations = self.forward(X)
            y_pred = activations[-1]
            
            # 計算損失
            loss = np.mean((y_pred - y) ** 2)
            history['loss'].append(loss)
            
            # --- 反向傳播 ---
            # 輸出層誤差
            error = y_pred - y
            delta = error * self._sigmoid_derivative(y_pred)
            
            # 迭代更新隱藏層
            for j in reversed(range(len(self.weights))):
                activation_prev = activations[j]
                
                # 計算梯度
                d_weights = np.dot(activation_prev.T, delta)
                d_biases = np.sum(delta, axis=0, keepdims=True)
                
                # 更新權重和偏差
                self.weights[j] -= learning_rate * d_weights
                self.biases[j] -= learning_rate * d_biases
                
                if j > 0:
                    # 傳遞誤差到前一層
                    delta = np.dot(delta, self.weights[j].T) * self._relu_derivative(activations[j])

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, epochs, loss)
        
        return history

    def predict(self, X):
        return self.forward(X)[-1]