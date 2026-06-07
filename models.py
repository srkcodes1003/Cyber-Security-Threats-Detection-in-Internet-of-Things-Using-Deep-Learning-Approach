import numpy as np

# Mimic Keras history callback output
class PyTorchHistory:
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

class Layer:
    def forward(self, input):
        raise NotImplementedError
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        # Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias
        
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Gradient clip for stability
        np.clip(weights_gradient, -1.0, 1.0, out=weights_gradient)
        np.clip(bias_gradient, -1.0, 1.0, out=bias_gradient)
        
        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
        
    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Sigmoid(Layer):
    def forward(self, input_data):
        self.output = 1.0 / (1.0 + np.exp(-np.clip(input_data, -15, 15)))
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.output * (1.0 - self.output)

class Softmax(Layer):
    def forward(self, input_data):
        exp_shifted = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        return output_gradient

class SimpleConv1D(Layer):
    def __init__(self, input_dim: int, num_filters: int, kernel_size: int):
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.out_dim = input_dim - kernel_size + 1
        
        # Initialize filters and biases
        self.weights = np.random.randn(num_filters, kernel_size) * np.sqrt(2.0 / (kernel_size + num_filters))
        self.biases = np.zeros(num_filters)
        
    def forward(self, input_data):
        self.input = input_data
        batch_size = input_data.shape[0]
        out = np.zeros((batch_size, self.num_filters, self.out_dim))
        
        for f in range(self.num_filters):
            for t in range(self.out_dim):
                out[:, f, t] = np.sum(input_data[:, t:t+self.kernel_size] * self.weights[f], axis=1) + self.biases[f]
                
        return out.reshape(batch_size, -1)
        
    def backward(self, output_gradient, learning_rate):
        batch_size = self.input.shape[0]
        grad_reshaped = output_gradient.reshape(batch_size, self.num_filters, self.out_dim)
        
        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.weights)
        biases_gradient = np.zeros_like(self.biases)
        
        for f in range(self.num_filters):
            biases_gradient[f] = np.sum(grad_reshaped[:, f, :])
            for t in range(self.out_dim):
                weights_gradient[f] += np.sum(self.input[:, t:t+self.kernel_size] * grad_reshaped[:, f, t:t+1], axis=0)
                input_gradient[:, t:t+self.kernel_size] += grad_reshaped[:, f, t:t+1] * self.weights[f]
                
        np.clip(weights_gradient, -1.0, 1.0, out=weights_gradient)
        np.clip(biases_gradient, -1.0, 1.0, out=biases_gradient)
        
        self.weights -= learning_rate * (weights_gradient / batch_size)
        self.biases -= learning_rate * (biases_gradient / batch_size)
        
        return input_gradient

class SimpleRNN(Layer):
    def __init__(self, input_len: int, hidden_size: int):
        self.input_len = input_len
        self.hidden_size = hidden_size
        
        # Weights
        self.W_xh = np.random.randn(1, hidden_size) * np.sqrt(2.0 / (1 + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.b_h = np.zeros((1, hidden_size))
        
    def forward(self, input_data):
        self.input = input_data
        batch_size = input_data.shape[0]
        self.h = np.zeros((self.input_len + 1, batch_size, self.hidden_size))
        
        for t in range(self.input_len):
            x_t = input_data[:, t:t+1]
            self.h[t+1] = np.tanh(np.dot(x_t, self.W_xh) + np.dot(self.h[t], self.W_hh) + self.b_h)
            
        return self.h[-1]
        
    def backward(self, output_gradient, learning_rate):
        batch_size = self.input.shape[0]
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        
        dh_t = output_gradient
        
        for t in reversed(range(self.input_len)):
            dtanh = dh_t * (1.0 - self.h[t+1]**2)
            x_t = self.input[:, t:t+1]
            dW_xh += np.dot(x_t.T, dtanh)
            dW_hh += np.dot(self.h[t].T, dtanh)
            db_h += np.sum(dtanh, axis=0, keepdims=True)
            dh_t = np.dot(dtanh, self.W_hh.T)
            
        np.clip(dW_xh, -1.0, 1.0, out=dW_xh)
        np.clip(dW_hh, -1.0, 1.0, out=dW_hh)
        np.clip(db_h, -1.0, 1.0, out=db_h)
        
        self.W_xh -= learning_rate * (dW_xh / batch_size)
        self.W_hh -= learning_rate * (dW_hh / batch_size)
        self.b_h -= learning_rate * (db_h / batch_size)
        
        return np.zeros_like(self.input)

class NumPyModel:
    def __init__(self, layers, num_classes: int, mode: str):
        self.layers = layers
        self.num_classes = num_classes
        self.mode = mode
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        if self.num_classes == 2:
            return self.sigmoid.forward(out)
        else:
            return self.softmax.forward(out)
            
    def predict(self, X, verbose=0):
        return self.forward(X)
        
    def fit(self, X_train, y_train, validation_data=None, epochs=10, batch_size=64, verbose=1):
        X_val, y_val = validation_data
        history = PyTorchHistory()
        
        # Adaptive learning rate based on task
        lr = 0.01 if self.mode == "binary" else 0.02
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                if self.num_classes == 2:
                    target = batch_y.reshape(-1, 1)
                else:
                    target = np.zeros((batch_y.shape[0], self.num_classes))
                    target[np.arange(batch_y.shape[0]), batch_y] = 1.0
                    
                pred = self.forward(batch_X)
                grad = pred - target
                
                # Compute loss
                if self.num_classes == 2:
                    loss = -np.mean(target * np.log(pred + 1e-15) + (1.0 - target) * np.log(1.0 - pred + 1e-15))
                else:
                    loss = -np.mean(np.sum(target * np.log(pred + 1e-15), axis=1))
                epoch_loss += loss * (end_idx - start_idx)
                
                # Backpropagation
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
                    
            epoch_train_loss = epoch_loss / num_samples
            
            # Evaluate train metrics
            train_pred = self.forward(X_train)
            if self.num_classes == 2:
                train_acc = np.mean((train_pred > 0.5).astype(int).flatten() == y_train)
            else:
                train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)
                
            # Evaluate val metrics
            val_pred = self.forward(X_val)
            if self.num_classes == 2:
                val_target = y_val.reshape(-1, 1)
                val_loss = -np.mean(val_target * np.log(val_pred + 1e-15) + (1.0 - val_target) * np.log(1.0 - val_pred + 1e-15))
                val_acc = np.mean((val_pred > 0.5).astype(int).flatten() == y_val)
            else:
                val_target = np.zeros((y_val.shape[0], self.num_classes))
                val_target[np.arange(y_val.shape[0]), y_val] = 1.0
                val_loss = -np.mean(np.sum(val_target * np.log(val_pred + 1e-15), axis=1))
                val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
                
            history.history['loss'].append(epoch_train_loss)
            history.history['accuracy'].append(train_acc)
            history.history['val_loss'].append(val_loss)
            history.history['val_accuracy'].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
                
        return history

def build_ann(input_dim: int, num_classes: int) -> NumPyModel:
    mode = "binary" if num_classes == 2 else "multiclass"
    layers = [
        Dense(input_dim, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 1 if num_classes == 2 else num_classes)
    ]
    return NumPyModel(layers, num_classes, mode)

def build_cnn(input_dim: int, num_classes: int) -> NumPyModel:
    mode = "binary" if num_classes == 2 else "multiclass"
    # Simplified Conv1D with 8 filters and 3 kernel_size
    conv_layer = SimpleConv1D(input_dim, num_filters=8, kernel_size=3)
    out_features = 8 * (input_dim - 3 + 1)
    layers = [
        conv_layer,
        ReLU(),
        Dense(out_features, 32),
        ReLU(),
        Dense(32, 1 if num_classes == 2 else num_classes)
    ]
    return NumPyModel(layers, num_classes, mode)

def build_lstm(input_dim: int, num_classes: int) -> NumPyModel:
    mode = "binary" if num_classes == 2 else "multiclass"
    layers = [
        SimpleRNN(input_len=input_dim, hidden_size=16),
        Dense(16, 1 if num_classes == 2 else num_classes)
    ]
    return NumPyModel(layers, num_classes, mode)
