# Stock RL Trader

強化学習（DQN）を使用して株価データから売買戦略を学習するフルスタックWebアプリケーションです。

## 🚀 セットアップ手順

### 1. バックエンド (FastAPI)
Python 3.8以上が必要です。

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. フロントエンド (Next.js)
Node.jsが必要です。

```bash
cd frontend
npm install
npm run dev
```

ブラウザで `http://localhost:3000` を開きます。

## 🧠 技術スタック
- **Backend**: FastAPI, PyTorch (DQN), Gymnasium, yfinance
- **Frontend**: Next.js (TypeScript), Recharts, Tailwind CSS, Lucide React
- **ML Algorithm**: Deep Q-Network (DQN)
  - 状態: 過去10日間の終値（正規化） + ポジション状態
  - 行動: Hold (維持), Buy (購入), Sell (売却)
  - 報酬: ポートフォリオ価値の変化

## 📁 ディレクトリ構成
- `/backend`: 学習環境、DQNモデル、APIサーバー
- `/frontend`: Next.js ダッシュボード、可視化チャート

## 🌐 デプロイ
- **Frontend**: Vercel
  - 環境変数 `NEXT_PUBLIC_API_URL` にバックエンドのURLを設定してください。
- **Backend**: Render, Railway, または AWS/GCP
  - `torch` と `yfinance` が動作する環境が必要です。
