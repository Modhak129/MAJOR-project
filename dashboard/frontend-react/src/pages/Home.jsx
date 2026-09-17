import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Home({ socket }) {
  const [metrics, setMetrics] = useState({ round: 0, map: 0, loss: 0, savings_bytes: 0 });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch('/api/metrics').then(r => r.json()).then(data => setMetrics(data));
    fetch('/api/metrics/history').then(r => r.json()).then(data => setHistory(data));
    
    if (!socket) return;
    
    socket.on('metrics_update', (data) => setMetrics(data));
    socket.on('metrics_history_update', (data) => setHistory(data));
  }, [socket]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">FL Round</CardTitle></CardHeader>
          <CardContent><div className="text-3xl font-bold">{metrics.round}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Global mAP</CardTitle></CardHeader>
          <CardContent><div className="text-3xl font-bold">{Number(metrics.map).toFixed(4)}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Global Loss</CardTitle></CardHeader>
          <CardContent><div className="text-3xl font-bold">{Number(metrics.loss).toFixed(4)}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Data Saved</CardTitle></CardHeader>
          <CardContent><div className="text-3xl font-bold">{(metrics.savings_bytes / (1024*1024)).toFixed(2)} MB</div></CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Global mAP Progression</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="round" stroke="#888" label={{ value: 'Round', position: 'insideBottomRight', offset: -5 }} />
                <YAxis stroke="#888" domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                <Legend />
                <Line type="monotone" dataKey="map" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Global Loss Progression</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="round" stroke="#888" label={{ value: 'Round', position: 'insideBottomRight', offset: -5 }} />
                <YAxis stroke="#888" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
