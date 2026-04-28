import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ComposedChart
} from 'recharts';

interface HistoryItem {
  date: string;
  price: number;
  action?: string;
}

interface StockChartProps {
  data: HistoryItem[];
  ticker: string;
}

const StockChart: React.FC<StockChartProps> = ({ data, ticker }) => {
  // Buy/Sellシグナルのデータを抽出
  const buySignals = data.filter(item => item.action === 'BUY').map(item => ({
    date: item.date,
    price: item.price * 0.98, // チャート上で少し下に表示
  }));

  const sellSignals = data.filter(item => item.action === 'SELL').map(item => ({
    date: item.date,
    price: item.price * 1.02, // チャート上で少し上に表示
  }));

  return (
    <div className="w-full h-[400px] bg-slate-900/50 backdrop-blur-md rounded-xl p-4 border border-slate-700 shadow-2xl">
      <h3 className="text-xl font-bold mb-4 text-indigo-400">{ticker} RL Analysis</h3>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8" 
            fontSize={12}
            tickFormatter={(str) => str.split('-').slice(1).join('/')}
          />
          <YAxis 
            stroke="#94a3b8" 
            fontSize={12} 
            domain={['auto', 'auto']}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f1f5f9' }}
            itemStyle={{ color: '#818cf8' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#6366f1" 
            strokeWidth={2} 
            dot={false}
            name="Stock Price"
          />
          {/* Buy シグナル */}
          <Scatter 
            name="BUY Signal" 
            data={buySignals} 
            fill="#10b981" 
          />
          {/* Sell シグナル */}
          <Scatter 
            name="SELL Signal" 
            data={sellSignals} 
            fill="#ef4444" 
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockChart;
