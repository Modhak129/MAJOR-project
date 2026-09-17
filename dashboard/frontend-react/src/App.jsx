import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import LiveMonitoring from './pages/LiveMonitoring';
import FLServerAnalytics from './pages/FLServerAnalytics';
import DeviceProfiles from './pages/DeviceProfiles';
import { Activity, Server, Cpu } from 'lucide-react';

function Sidebar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Live Monitoring', icon: Activity },
    { path: '/fl-server', label: 'FL Server Analytics', icon: Server },
    { path: '/devices', label: 'Edge Nodes', icon: Cpu },
  ];

  return (
    <div className="w-64 bg-black/40 backdrop-blur-xl border-r border-white/10 h-full min-h-screen flex flex-col fixed left-0 top-0 z-50">
      <div className="p-6 border-b border-white/10">
        <h1 className="text-xl font-bold tracking-tight text-white">Edge-AI Safety</h1>
        <p className="text-xs text-blue-400 font-mono mt-1">Federated V2.0</p>
      </div>
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map(item => (
          <Link 
            key={item.path} 
            to={item.path}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${
              location.pathname === item.path ? 'bg-blue-500/20 text-blue-400 shadow-[inset_0_0_12px_rgba(59,130,246,0.2)]' : 'text-muted-foreground hover:bg-white/5 hover:text-white'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium text-sm">{item.label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}

function App() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = io();
    setSocket(newSocket);
    return () => newSocket.close();
  }, []);

  return (
    <Router>
      <div className="flex min-h-screen bg-slate-950 text-slate-50 selection:bg-blue-500/30">
        <Sidebar />
        <main className="flex-1 ml-64 p-8 overflow-y-auto h-screen">
          <Routes>
            <Route path="/" element={<LiveMonitoring socket={socket} />} />
            <Route path="/fl-server" element={<FLServerAnalytics socket={socket} />} />
            <Route path="/devices" element={<DeviceProfiles socket={socket} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
