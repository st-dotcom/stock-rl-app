import random
import numpy as np
from collections import deque
from model import NumPyQNetwork

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 割引率
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = NumPyQNetwork(state_size, action_size, learning_rate=self.learning_rate)
        self.target_model = NumPyQNetwork(state_size, action_size, learning_rate=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, train=True):
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        action_values = self.model.forward(state)
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Target values using target network
        next_q_values = self.target_model.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Current predictions
        q_values = self.model.forward(states)
        
        # Calculate gradients for MSE loss
        d_out = np.zeros_like(q_values)
        for i in range(batch_size):
            d_out[i, actions[i]] = q_values[i, actions[i]] - targets[i]
            
        self.model.backward(d_out)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        weights = self.model.get_weights()
        np.savez(name, **weights)

    def load(self, name):
        weights_data = np.load(name)
        weights_dict = {key: weights_data[key] for key in weights_data.files}
        self.model.set_weights(weights_dict)
        self.update_target_model()
