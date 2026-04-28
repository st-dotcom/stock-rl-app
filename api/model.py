import numpy as np

class NumPyQNetwork:
    """
    DQNのためのシンプルなQネットワーク (NumPy実装)
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        
        # He Initialization
        self.W1 = np.random.randn(state_size, 64) * np.sqrt(2. / state_size)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 64) * np.sqrt(2. / 64)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, action_size) * np.sqrt(2. / 64)
        self.b3 = np.zeros(action_size)
        
        # Adam Optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), 
                  np.zeros_like(self.W2), np.zeros_like(self.b2), 
                  np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), 
                  np.zeros_like(self.W2), np.zeros_like(self.b2), 
                  np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.t = 0
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        return (x > 0).astype(float)
        
    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3
        
        return self.a3
        
    def backward(self, d_out):
        m = self.x.shape[0]
        
        dW3 = np.dot(self.a2.T, d_out) / m
        db3 = np.sum(d_out, axis=0) / m
        
        d_a2 = np.dot(d_out, self.W3.T)
        d_z2 = d_a2 * self.relu_derivative(self.z2)
        
        dW2 = np.dot(self.a1.T, d_z2) / m
        db2 = np.sum(d_z2, axis=0) / m
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        
        dW1 = np.dot(self.x.T, d_z1) / m
        db1 = np.sum(d_z1, axis=0) / m
        
        self._update_weights([dW1, db1, dW2, db2, dW3, db3])
        
    def _update_weights(self, grads):
        self.t += 1
        weights = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for i in range(len(weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights
        
    def get_weights(self):
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy()
        }
        
    def set_weights(self, weights_dict):
        self.W1 = weights_dict['W1'].copy()
        self.b1 = weights_dict['b1'].copy()
        self.W2 = weights_dict['W2'].copy()
        self.b2 = weights_dict['b2'].copy()
        self.W3 = weights_dict['W3'].copy()
        self.b3 = weights_dict['b3'].copy()
