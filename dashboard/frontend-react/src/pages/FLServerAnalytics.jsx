import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import DeviceManager from '../components/DeviceManager';

export default function FLServerAnalytics({ socket }) {
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState({ round: 0, map: 0, loss: 0, savings_bytes: 0 });

  useEffect(() => {
    fetch('/api/db/fl-rounds').then(r => r.json()).then(data => {
        // Reverse because db returns desc
        setHistory(data.reverse());
        if(data.length > 0) setMetrics(data[data.length-1]);
    });
    
    if (!socket) return;
    socket.on('metrics_update', (data) => setMetrics(data));
    socket.on('metrics_history_update', (data) => setHistory(data));
  }, [socket]);

  return (
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Col: Device Manager & Controls */}
        <div className="col-span-1 space-y-6">
            <DeviceManager />
            <Card className="border-red-500/20 bg-red-500/5">
                <CardHeader><CardTitle className="text-red-400">Fault Tolerance Tester</CardTitle></CardHeader>
                <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground">Terminate an edge node mid-round to verify FLServer resilience (FLM-11).</p>
                    <button onClick={() => fetch('/api/containers/edge_node_1/stop', {method:'POST'})} className="w-full bg-red-600/20 hover:bg-red-600/40 text-red-400 px-4 py-2 rounded transition-colors">Kill Node 1 (Constrained)</button>
                    <button onClick={() => fetch('/api/containers/edge_node_2/stop', {method:'POST'})} className="w-full bg-red-600/20 hover:bg-red-600/40 text-red-400 px-4 py-2 rounded transition-colors">Kill Node 2 (Capable)</button>
                </CardContent>
            </Card>
        </div>

        {/* Right Col: Charts */}
        <div className="col-span-2 space-y-6">
            
            <div className="grid grid-cols-3 gap-4">
                <Card className="bg-white/5 border-white/10 backdrop-blur">
                    <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Global mAP</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold text-emerald-400">{Number(metrics.map || 0).toFixed(4)}</div></CardContent>
                </Card>
                <Card className="bg-white/5 border-white/10 backdrop-blur">
                    <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Global Loss</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold text-blue-400">{Number(metrics.loss || 0).toFixed(4)}</div></CardContent>
                </Card>
                <Card className="bg-white/5 border-white/10 backdrop-blur">
                    <CardHeader className="pb-2"><CardTitle className="text-sm text-muted-foreground">Total Sparsification Savings</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold text-purple-400">{((metrics.savings_bytes || 0) / (1024*1024)).toFixed(2)} MB</div></CardContent>
                </Card>
            </div>

            <Card className="border-white/10 bg-white/5 backdrop-blur">
              <CardHeader><CardTitle>Global Model Progression</CardTitle></CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                    <XAxis dataKey="round" stroke="#888" />
                    <YAxis stroke="#888" yAxisId="left" />
                    <YAxis stroke="#888" yAxisId="right" orientation="right" />
                    <Tooltip contentStyle={{ backgroundColor: '#000000dd', border: '1px solid #333', borderRadius: '8px' }} />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="map" stroke="#34d399" strokeWidth={3} dot={{r: 4, fill: '#34d399'}} name="mAP @ 0.5" />
                    <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#60a5fa" strokeWidth={3} dot={{r: 4, fill: '#60a5fa'}} name="Val Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="border-white/10 bg-white/5 backdrop-blur">
              <CardHeader><CardTitle>Communication Efficiency (Bytes Saved %)</CardTitle></CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                    <XAxis dataKey="round" stroke="#888" />
                    <YAxis stroke="#888" domain={[0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: '#000000dd', border: '1px solid #333', borderRadius: '8px' }} />
                    <Bar dataKey="saved_pct" fill="#c084fc" radius={[4, 4, 0, 0]} name="Savings %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

        </div>
      </div>
    </div>
  );
}
