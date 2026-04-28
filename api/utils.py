import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, period="1y", interval="1d"):
    """
    Yahoo Financeから株価データを取得
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def preprocess_data(df):
    """
    データのクリーニングと前処理
    """
    # 欠損値の削除
    df = df.dropna()
    return df
