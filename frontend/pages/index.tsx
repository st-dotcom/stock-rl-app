import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import axios from 'axios';
import StockChart from '../components/StockChart';
import { Search, Play, Activity, TrendingUp, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [ticker, setTicker] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<any[]>([]);
  const [mode, setMode] = useState('normal');

  const fetchHistory = async (symbol: string) => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/history/${symbol}`);
      setHistory(res.data.history);
    } catch (err) {
      console.error(err);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/api/train`, { 
        ticker, 
        mode,
        episodes: mode === 'quick' ? 5 : 20 
      });
      pollStatus();
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  const pollStatus = () => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/status/${ticker}`);
        setStatus(res.data);
        if (res.data.status === 'completed' || res.data.status === 'error') {
          clearInterval(interval);
          setLoading(false);
          if (res.data.status === 'completed') handlePredict();
        }
      } catch (err) {
        clearInterval(interval);
        setLoading(false);
      }
    }, 2000);
  };

  const handlePredict = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/predict/${ticker}`);
      setPrediction(res.data.history);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHistory('AAPL');
  }, []);

  return (
    <div className="min-h-screen p-8 text-slate-200">
      <Head>
        <title>Stock RL Trader | AI Trading Agent</title>
      </Head>

      <main className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-8">
          <div>
            <h1 className="text-4xl font-extrabold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              Stock RL Trader
            </h1>
            <p className="text-slate-400 mt-2">DQNを用いた強化学習株式取引エージェント</p>
          </div>
          
          <div className="flex items-center gap-2 bg-slate-900/80 p-2 rounded-2xl border border-slate-700">
            <Search className="text-slate-500 ml-2" size={20} />
            <input 
              type="text" 
              value={ticker} 
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="bg-transparent border-none outline-none px-2 py-1 w-24 text-lg font-bold"
              placeholder="AAPL"
            />
            <button 
              onClick={() => fetchHistory(ticker)}
              className="bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-xl transition-all font-semibold"
            >
              Fetch
            </button>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Chart */}
          <div className="lg:col-span-2 space-y-6">
            <StockChart 
              data={prediction.length > 0 ? prediction : history} 
              ticker={ticker} 
            />
            
            {/* Stats Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <Activity size={16} />
                  <span className="text-sm">Status</span>
                </div>
                <div className="text-lg font-bold">
                  {status?.status === 'training' ? (
                    <span className="text-amber-400 flex items-center gap-2">
                      <Loader2 size={18} className="animate-spin" /> Training {status.progress}%
                    </span>
                  ) : status?.status === 'completed' ? (
                    <span className="text-emerald-400 flex items-center gap-2">
                      <CheckCircle2 size={18} /> Ready
                    </span>
                  ) : (
                    <span className="text-slate-500">Idle</span>
                  )}
                </div>
              </div>
              
              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <TrendingUp size={16} />
                  <span className="text-sm">Current Price</span>
                </div>
                <div className="text-2xl font-bold">
                  ${history[history.length - 1]?.price.toFixed(2) || '0.00'}
                </div>
              </div>

              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <AlertCircle size={16} />
                  <span className="text-sm">Signals</span>
                </div>
                <div className="text-lg font-bold">
                  {prediction.length > 0 ? (
                    <div className="flex gap-2">
                      <span className="text-emerald-400">{prediction.filter(x => x.action === 'BUY').length} Buy</span>
                      <span className="text-rose-400">{prediction.filter(x => x.action === 'SELL').length} Sell</span>
                    </div>
                  ) : 'No signals'}
                </div>
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            <div className="bg-slate-900/80 backdrop-blur-xl rounded-2xl p-6 border border-slate-700 shadow-xl space-y-6">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <Play className="text-indigo-400" size={24} />
                Control Panel
              </h2>
              
              <div className="space-y-4">
                <label className="block text-sm font-medium text-slate-400">Training Mode</label>
                <div className="grid grid-cols-2 gap-2">
                  <button 
                    onClick={() => setMode('normal')}
                    className={`py-2 rounded-xl border transition-all ${mode === 'normal' ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-400'}`}
                  >
                    Normal (20 ep)
                  </button>
                  <button 
                    onClick={() => setMode('quick')}
                    className={`py-2 rounded-xl border transition-all ${mode === 'quick' ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-400'}`}
                  >
                    Quick (5 ep)
                  </button>
                </div>
              </div>

              <button 
                onClick={handleTrain}
                disabled={loading}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:from-slate-700 disabled:to-slate-700 py-4 rounded-xl font-bold text-lg shadow-lg shadow-indigo-500/20 transition-all flex items-center justify-center gap-2"
              >
                {loading ? <Loader2 className="animate-spin" /> : 'Start AI Training'}
              </button>

              <div className="p-4 bg-indigo-900/20 rounded-xl border border-indigo-500/30 text-sm text-indigo-200">
                <p>エージェントは指定された銘柄の過去1年間のデータを分析し、最適な売買タイミングを強化学習で習得します。</p>
              </div>
            </div>
            
            {/* Log / Recent Actions */}
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800 h-[300px] overflow-y-auto">
              <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Trading Log</h3>
              <div className="space-y-3">
                {prediction.slice().reverse().slice(0, 10).map((item, i) => (
                  <div key={i} className="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                    <span className="text-slate-400">{item.date}</span>
                    <span className={item.action === 'BUY' ? 'text-emerald-400' : item.action === 'SELL' ? 'text-rose-400' : 'text-slate-500'}>
                      {item.action}
                    </span>
                    <span className="font-mono">${item.price.toFixed(2)}</span>
                  </div>
                ))}
                {prediction.length === 0 && <p className="text-slate-600 text-center mt-12">No recent logs</p>}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
