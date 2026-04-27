import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    株式取引のカスタム環境
    """
    def __init__(self, df, window_size=10):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.window_size = window_size
        self.prices = self.df['Close'].values
        
        # 行動空間: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # 状態空間: [過去N日分の価格推移(正規化)] + [保有フラグ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0  # 0: 未保有, 1: 保有
        self.buy_price = 0
        self.total_reward = 0
        self.history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # ウィンドウ期間の価格を取得して正規化（現在の価格で割る）
        window_prices = self.prices[self.current_step - self.window_size : self.current_step]
        normalized_prices = window_prices / self.prices[self.current_step - 1]
        
        # 観測値 = [過去の正規化価格] + [保有フラグ]
        obs = np.append(normalized_prices, [float(self.position)])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        done = False
        
        # 行動の実行
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
        elif action == 2:  # Sell
            if self.position == 1:
                self.position = 0
                # 報酬は売却益
                reward = (current_price - self.buy_price) / self.buy_price
        
        # 未売却時の評価損益を報酬に少し加える（継続的な学習のため）
        if self.position == 1:
            reward += (current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True
            
        self.total_reward += reward
        
        # 次の状態、報酬、終了フラグを返す
        obs = self._get_observation()
        
        return obs, reward, done, False, {}
