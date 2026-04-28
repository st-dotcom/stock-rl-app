from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from utils import fetch_stock_data, preprocess_data
from env import StockTradingEnv
from agent import DQNAgent

app = FastAPI(title="Stock RL Trading API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル保存用ディレクトリ
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 学習ステータス管理（簡易的なインメモリ保持）
training_status = {}

class TrainRequest(BaseModel):
    ticker: str
    episodes: int = 20
    window_size: int = 10
    mode: str = "normal"  # "normal" or "quick"

@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    ticker = request.ticker.upper()
    
    # 簡易モードの設定
    episodes = 5 if request.mode == "quick" else request.episodes
    
    training_status[ticker] = {"status": "training", "progress": 0}
    
    background_tasks.add_task(run_training, ticker, episodes, request.window_size)
    
    return {"message": f"Training started for {ticker}", "ticker": ticker}

def run_training(ticker, episodes, window_size):
    try:
        df = fetch_stock_data(ticker)
        if df is None:
            training_status[ticker] = {"status": "error", "message": "Failed to fetch data"}
            return
            
        df = preprocess_data(df)
        env = StockTradingEnv(df, window_size=window_size)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = DQNAgent(state_size, action_size)
        batch_size = 32
        
        for e in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            for time in range(500): # max steps per episode
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    agent.update_target_model()
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            
            progress = int(((e + 1) / episodes) * 100)
            training_status[ticker]["progress"] = progress
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
            
        # モデルの保存
        model_path = os.path.join(MODEL_DIR, f"{ticker}_dqn.npz")
        agent.save(model_path)
        
        training_status[ticker] = {"status": "completed", "progress": 100, "model_path": model_path}
    except Exception as e:
        training_status[ticker] = {"status": "error", "message": str(e)}

@app.get("/api/status/{ticker}")
async def get_status(ticker: str):
    return training_status.get(ticker.upper(), {"status": "not_started"})

@app.get("/api/predict/{ticker}")
async def predict_action(ticker: str):
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_dqn.npz")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Please train first.")
        
    df = fetch_stock_data(ticker, period="1mo") # 最近のデータを取得
    df = preprocess_data(df)
    window_size = 10
    env = StockTradingEnv(df, window_size=window_size)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # 最近のデータでシミュレーションを実行して全期間のシグナルを返す
    state, _ = env.reset()
    results = []
    prices = df['Close'].tolist()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    for i in range(len(env.prices) - window_size):
        action = agent.act(state, train=False)
        action_name = ["HOLD", "BUY", "SELL"][action]
        
        # 現在のステップの日付と価格を取得
        # env.current_step は内部で更新される
        current_idx = env.current_step
        results.append({
            "date": dates[current_idx],
            "price": prices[current_idx],
            "action": action_name
        })
        
        next_state, _, done, _, _ = env.step(action)
        state = next_state
        if done: break
        
    return {
        "ticker": ticker,
        "history": results
    }

@app.get("/api/history/{ticker}")
async def get_history(ticker: str):
    df = fetch_stock_data(ticker.upper())
    if df is None:
        raise HTTPException(status_code=404, detail="Stock data not found")
        
    df = preprocess_data(df)
    prices = df['Close'].tolist()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    history = [{"date": d, "price": p} for d, p in zip(dates, prices)]
    return {"ticker": ticker.upper(), "history": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
