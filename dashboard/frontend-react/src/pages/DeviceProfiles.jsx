import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Server, Cpu, Database, Network, Power, TerminalSquare, Clock, Zap } from 'lucide-react';

const MOCK_DEVICES = [
    {
        node_id: "node_constrained",
        status: "MONITOR",
        ip_address: "172.18.0.4",
        cpu_cores: 2,
        ram_mb: 2048,
        simulated_vram_mb: 2048,
        gpu_name: "Simulated Jetson Nano",
        assigned_tier: "YOLOv8-Nano",
        model_format: "INT8 ONNX",
        input_resolution: "320x320",
        current_inference_fps: 18.5,
        last_training_upload_time: new Date(Date.now() - 120000).toISOString(),
        latest_model_received_time: new Date(Date.now() - 600000).toISOString(),
        total_rounds_participated: 4,
        last_round_loss: 0.14,
        last_round_sparsification_ratio: "99%"
    },
    {
        node_id: "node_capable",
        status: "MONITOR",
        ip_address: "172.18.0.5",
        cpu_cores: 4,
        ram_mb: 4096,
        simulated_vram_mb: 6144,
        gpu_name: "Simulated Jetson Xavier",
        assigned_tier: "YOLOv8-Small",
        model_format: "FP16 PT",
        input_resolution: "416x416",
        current_inference_fps: 32.1,
        last_training_upload_time: new Date(Date.now() - 125000).toISOString(),
        latest_model_received_time: new Date(Date.now() - 605000).toISOString(),
        total_rounds_participated: 4,
        last_round_loss: 0.12,
        last_round_sparsification_ratio: "99%"
    }
];

// Extracted as a top-level component so React maintains stable identity across parent re-renders
const FLServerVisualizer = ({ socket }) => {
    const [serverState, setServerState] = useState('idle');
    const [clientCount, setClientCount] = useState(0);
    const demoTimers = React.useRef([]);

    useEffect(() => {
        if (!socket) return;
        const handler = (data) => {
            setServerState(data.event);
            if (data.count) setClientCount(data.count);
        };
        socket.on('fl_server_event', handler);
        return () => socket.off('fl_server_event', handler);
    }, [socket]);

    useEffect(() => {
        const onDemoStart = () => {
            // Clear any previous demo timers so re-clicking works
            demoTimers.current.forEach(t => clearTimeout(t));
            demoTimers.current = [];

            // Run the full animation sequence client-side
            setServerState('training');
            demoTimers.current.push(setTimeout(() => setServerState('weights_received'), 3000));
            demoTimers.current.push(setTimeout(() => setServerState('aggregating'), 5000));
            demoTimers.current.push(setTimeout(() => setServerState('model_updated'), 7500));
            demoTimers.current.push(setTimeout(() => setServerState('broadcasting'), 9500));
            demoTimers.current.push(setTimeout(() => setServerState('idle'), 13000));
        };
        window.addEventListener('demo_train_start', onDemoStart);
        return () => {
            window.removeEventListener('demo_train_start', onDemoStart);
            demoTimers.current.forEach(t => clearTimeout(t));
        };
    }, []);

    return (
        <Card className="bg-slate-900 border-indigo-500/30 overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 via-purple-500/5 to-blue-500/5"></div>
            <CardContent className="p-8 relative z-10">
                <div className="flex flex-col items-center justify-center">
                    <h3 className="text-xl font-bold text-white mb-8 flex items-center gap-2">
                        <Server className="w-5 h-5 text-indigo-400"/> Global FL Aggregator
                    </h3>
                    
                    <div className="flex w-full max-w-2xl justify-between items-center px-10">
                        {/* Edge Nodes (Left) */}
                        <div className="flex flex-col gap-4 relative">
                            <div className={`w-16 h-16 rounded-xl border-2 flex flex-col items-center justify-center bg-slate-800 z-10 transition-colors ${serverState === 'training' ? 'border-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]' : serverState === 'weights_received' || serverState === 'idle' ? 'border-indigo-400' : 'border-slate-600'}`}>
                                <TerminalSquare className={`w-6 h-6 ${serverState === 'training' ? 'text-amber-400 animate-pulse' : serverState === 'weights_received' ? 'text-indigo-400' : 'text-slate-400'}`} />
                                {serverState === 'broadcasting' && <span className="text-[9px] text-emerald-400 font-bold mt-1">YOLOv8-Nano</span>}
                            </div>
                            <div className={`w-16 h-16 rounded-xl border-2 flex flex-col items-center justify-center bg-slate-800 z-10 transition-colors ${serverState === 'training' ? 'border-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]' : serverState === 'weights_received' || serverState === 'idle' ? 'border-indigo-400' : 'border-slate-600'}`}>
                                <TerminalSquare className={`w-6 h-6 ${serverState === 'training' ? 'text-amber-400 animate-pulse' : serverState === 'weights_received' ? 'text-indigo-400' : 'text-slate-400'}`} />
                                {serverState === 'broadcasting' && <span className="text-[9px] text-emerald-400 font-bold mt-1">YOLOv8-Small</span>}
                            </div>
                            {serverState === 'training' && (
                                <div className="absolute -right-20 top-1/2 -translate-y-1/2 text-xs font-bold text-amber-400 bg-slate-900 px-2 py-1 rounded-full border border-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)] animate-pulse whitespace-nowrap">
                                    Training Epoch...
                                </div>
                            )}
                        </div>

                        {/* Connection Lines & Animation */}
                        <div className="flex-1 h-32 relative mx-4">
                            {/* Static lines */}
                            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 100">
                                <path d="M 0,25 C 50,25 50,50 100,50" stroke="rgba(255,255,255,0.1)" strokeWidth="2" fill="none" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
                                <path d="M 0,75 C 50,75 50,50 100,50" stroke="rgba(255,255,255,0.1)" strokeWidth="2" fill="none" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
                            </svg>
                            
                            {/* Upload Animation (Weights Received) */}
                            {serverState === 'weights_received' && (
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="flex gap-1 items-center text-indigo-400 font-bold text-sm bg-slate-900 px-3 py-1 rounded-full border border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.5)] z-20">
                                        <Database className="w-4 h-4 animate-bounce" /> Uploading Weights
                                    </div>
                                    <div className="absolute left-1/4 top-1/3 w-3 h-3 bg-indigo-400 rounded-full animate-ping z-10"></div>
                                    <div className="absolute left-1/4 bottom-1/3 w-3 h-3 bg-indigo-400 rounded-full animate-ping z-10"></div>
                                </div>
                            )}

                            {/* Download Animation (Broadcasting) */}
                            {serverState === 'broadcasting' && (
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="flex gap-1 items-center text-emerald-400 font-bold text-sm bg-slate-900 px-3 py-1 rounded-full border border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.5)] z-20">
                                        <Network className="w-4 h-4 animate-bounce" /> Broadcasting Model
                                    </div>
                                    <div className="absolute right-1/4 top-1/3 w-3 h-3 bg-emerald-400 rounded-full animate-ping z-10"></div>
                                    <div className="absolute right-1/4 bottom-1/3 w-3 h-3 bg-emerald-400 rounded-full animate-ping z-10"></div>
                                </div>
                            )}
                        </div>

                        {/* Central Server (Right) */}
                        <div className="relative">
                            <div className={`w-32 h-32 rounded-full border-4 flex flex-col items-center justify-center bg-slate-800 z-10 transition-all duration-500 shadow-[0_0_30px_rgba(0,0,0,0.5)]
                                ${serverState === 'aggregating' ? 'border-purple-500 shadow-[0_0_40px_rgba(168,85,247,0.6)] animate-pulse' : 
                                  serverState === 'model_updated' ? 'border-emerald-500 shadow-[0_0_40px_rgba(16,185,129,0.6)]' :
                                  'border-slate-600'}`}>
                                
                                {serverState === 'aggregating' ? (
                                    <>
                                        <Cpu className="w-10 h-10 text-purple-400 mb-1" />
                                        <span className="text-xs font-bold text-purple-300">FedAvg</span>
                                        <span className="text-[10px] text-purple-200/70 animate-pulse">Aggregating...</span>
                                    </>
                                ) : serverState === 'model_updated' ? (
                                    <>
                                        <Zap className="w-10 h-10 text-emerald-400 mb-1 animate-bounce" />
                                        <span className="text-xs font-bold text-emerald-300">Model Updated</span>
                                    </>
                                ) : (
                                    <>
                                        <Server className="w-10 h-10 text-slate-400 mb-1" />
                                        <span className="text-xs font-bold text-slate-300">{serverState === 'idle' || serverState === 'training' ? 'Idle' : 'Active'}</span>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default function DeviceProfiles({ socket }) {
  const [devices, setDevices] = useState(MOCK_DEVICES);
  
  const fetchDevices = () => {
    fetch('/api/db/devices')
      .then(res => res.json())
      .then(data => {
        if(data.length > 0) {
            const merged = MOCK_DEVICES.map(mock => {
                const live = data.find(d => d.node_id === mock.node_id);
                if (live) {
                    const cleanLive = Object.fromEntries(Object.entries(live).filter(([_, v]) => v != null));
                    return { ...mock, ...cleanLive };
                }
                return mock;
            });
            setDevices(merged);
        } else {
            setDevices(MOCK_DEVICES);
        }
      });
  };

  useEffect(() => {
    fetchDevices();
    const interval = setInterval(fetchDevices, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!socket) return;
    const handler = (data) => {
        setDevices(prev => {
            const idx = prev.findIndex(d => d.node_id === data.node_id);
            const cleanData = Object.fromEntries(Object.entries(data).filter(([_, v]) => v != null));
            if (idx === -1) {
                const mockBase = MOCK_DEVICES.find(m => m.node_id === data.node_id) || {};
                return [...prev, { ...mockBase, ...cleanData }];
            }
            const updated = [...prev];
            updated[idx] = { ...updated[idx], ...cleanData };
            return updated;
        });
    };
    socket.on('node_telemetry_update', handler);
    return () => socket.off('node_telemetry_update', handler);
  }, [socket]);
  
  const [broadcastToast, setBroadcastToast] = useState(null);
  
  useEffect(() => {
      if (!socket) return;
      const onBroadcast = (data) => {
          setBroadcastToast(`🚀 Global Model Broadcast! Round ${data.round} updated at ${new Date(data.timestamp).toLocaleTimeString()}`);
          setTimeout(() => setBroadcastToast(null), 5000);
      };
      socket.on('global_model_broadcast', onBroadcast);
      return () => socket.off('global_model_broadcast', onBroadcast);
  }, [socket]);
  
  const handleContainerAction = (nodeId, action) => {
      const containerName = nodeId === "node_constrained" ? "edge_node_1" : "edge_node_2";
      fetch(`/api/containers/${containerName}/${action}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => alert(data.status || data.error));
  };
  
  const handleModeSwitch = (nodeId, mode) => {
      if (mode === 'DEMO_TRAIN') {
          // Demo train is purely a frontend animation — don't touch the backend
          const mainElement = document.querySelector('main');
          if (mainElement) {
              mainElement.scrollTo({ top: 0, behavior: 'smooth' });
          } else {
              window.scrollTo({ top: 0, behavior: 'smooth' });
          }
          window.dispatchEvent(new CustomEvent('demo_train_start'));
          return;
      }

      // For real mode switches (MONITOR, TRAIN), send to backend
      const containerName = nodeId === "node_constrained" ? "edge_node_1" : "edge_node_2";
      fetch(`/api/containers/${containerName}/mode`, { 
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ mode })
      }).catch(() => {});
  };

  const getStatusColor = (status) => {
      if (!status) return "bg-slate-500/20 text-slate-400";
      if (status.includes("Training") || status.includes("Active") || status.includes("Online")) return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      if (status.includes("Disconnected")) return "bg-red-500/20 text-red-400 border-red-500/30";
      return "bg-amber-500/20 text-amber-400 border-amber-500/30";
  };
  
  const activeCount = devices.filter(d => d.status !== 'Disconnected').length;
  const nanoCount = devices.filter(d => d.assigned_tier === 'YOLOv8-Nano').length;
  const smallCount = devices.filter(d => d.assigned_tier === 'YOLOv8-Small').length;
  const avgFps = devices.length ? (devices.reduce((acc, d) => acc + (d.current_inference_fps || 0), 0) / devices.length).toFixed(1) : 0;

  return (
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
      <FLServerVisualizer socket={socket} />
      
      {/* Overview Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-white/5 border-white/10 backdrop-blur">
              <CardContent className="p-6 flex items-center gap-4">
                  <div className="p-3 bg-blue-500/20 rounded-lg text-blue-400"><Server className="w-6 h-6" /></div>
                  <div>
                      <p className="text-sm text-muted-foreground">Connected Devices</p>
                      <h3 className="text-2xl font-bold">{activeCount} / {devices.length}</h3>
                  </div>
              </CardContent>
          </Card>
          <Card className="bg-white/5 border-white/10 backdrop-blur">
              <CardContent className="p-6 flex items-center gap-4">
                  <div className="p-3 bg-purple-500/20 rounded-lg text-purple-400"><Network className="w-6 h-6" /></div>
                  <div>
                      <p className="text-sm text-muted-foreground">Active FL Workers</p>
                      <h3 className="text-2xl font-bold">{nanoCount} Nano <span className="text-muted-foreground text-sm font-normal">| {smallCount} Small</span></h3>
                  </div>
              </CardContent>
          </Card>
          <Card className="bg-white/5 border-white/10 backdrop-blur">
              <CardContent className="p-6 flex items-center gap-4">
                  <div className="p-3 bg-emerald-500/20 rounded-lg text-emerald-400"><Zap className="w-6 h-6" /></div>
                  <div>
                      <p className="text-sm text-muted-foreground">Fleet Avg FPS</p>
                      <h3 className="text-2xl font-bold">{avgFps}</h3>
                  </div>
              </CardContent>
          </Card>
      </div>

      {/* Profile Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {devices.map((device, idx) => (
            <Card key={idx} className="bg-white/5 border-white/10 backdrop-blur-md overflow-hidden">
                <CardHeader className="border-b border-white/10 pb-4 flex flex-row items-center justify-between bg-black/20">
                    <div className="flex items-center gap-3">
                        <Server className="text-blue-400 w-6 h-6" />
                        <div>
                            <CardTitle className="text-xl">{device.node_id}</CardTitle>
                            <p className="text-xs text-muted-foreground font-mono">{device.ip_address || "Unknown IP"}</p>
                        </div>
                    </div>
                    <div className={`px-3 py-1.5 rounded-full border flex items-center gap-2 text-sm font-medium ${getStatusColor(device.status)}`}>
                        {(device.status === 'Training' || device.status === 'Inference Active') && <span className="w-2 h-2 rounded-full bg-current animate-pulse block"></span>}
                        {device.status || 'Unknown'}
                    </div>
                </CardHeader>
                
                <CardContent className="p-6 space-y-6">
                    
                    {/* Hardware Telemetry */}
                    <div>
                        <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2 uppercase tracking-wider"><Cpu className="w-4 h-4"/> Hardware Limits</h4>
                        <div className="space-y-3">
                            <div>
                                <div className="flex justify-between text-sm mb-1"><span>CPU Cores</span><span className="font-mono">{device.cpu_cores}</span></div>
                                <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden"><div className="h-full bg-blue-400 w-1/2"></div></div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1"><span>System RAM</span><span className="font-mono">{device.ram_mb || 2048} MB</span></div>
                                <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden"><div className="h-full bg-indigo-400 w-[60%]"></div></div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1"><span>Simulated GPU VRAM ({device.gpu_name || 'NVIDIA GPU'})</span><span className="font-mono text-emerald-400">{device.simulated_vram_mb} MB</span></div>
                                <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden"><div className="h-full bg-emerald-400 w-[80%]"></div></div>
                            </div>
                        </div>
                    </div>
                    
                    {/* Model Architecture */}
                    <div>
                        <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2 uppercase tracking-wider"><Zap className="w-4 h-4"/> Active Model</h4>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                                <p className="text-xs text-muted-foreground mb-1">Tier / Variant</p>
                                <p className="font-semibold text-purple-400">{device.assigned_tier || 'N/A'}</p>
                            </div>
                            <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                                <p className="text-xs text-muted-foreground mb-1">Format</p>
                                <p className="font-semibold">{device.model_format || 'N/A'}</p>
                            </div>
                            <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                                <p className="text-xs text-muted-foreground mb-1">Input Res</p>
                                <p className="font-semibold">{device.input_resolution || 'N/A'}</p>
                            </div>
                            <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                                <p className="text-xs text-muted-foreground mb-1">Inference Speed</p>
                                <p className="font-semibold text-emerald-400">{device.current_inference_fps || 0} FPS</p>
                            </div>
                        </div>
                    </div>

                    {/* FL Sync */}
                    <div>
                        <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2 uppercase tracking-wider"><Network className="w-4 h-4"/> FL Sync Timeline</h4>
                        <div className="bg-black/40 rounded-lg p-4 space-y-3 border border-white/5">
                            <div className="flex justify-between items-center text-sm">
                                <span className="flex items-center gap-2"><Clock className="w-4 h-4 text-muted-foreground" /> Last Training Upload</span>
                                <span className="font-mono" title={device.last_training_upload_time}>{device.last_training_upload_time ? new Date(device.last_training_upload_time).toLocaleTimeString() : 'Never'}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="flex items-center gap-2"><Clock className="w-4 h-4 text-muted-foreground" /> Global Model Rx</span>
                                <span className="font-mono" title={device.latest_model_received_time}>{device.latest_model_received_time ? new Date(device.latest_model_received_time).toLocaleTimeString() : 'Never'}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="flex items-center gap-2"><Database className="w-4 h-4 text-muted-foreground" /> Round {device.total_rounds_participated || 0} Sparsification</span>
                                <span className="font-mono text-purple-400">{device.last_round_sparsification_ratio || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    {/* Actions */}
                    <div>
                        <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2 uppercase tracking-wider"><TerminalSquare className="w-4 h-4"/> Operation Mode</h4>
                        <div className="flex gap-2">
                            <button onClick={() => handleModeSwitch(device.node_id, 'MONITOR')} className={`flex-1 ${device.status==='MONITOR' ? 'bg-blue-500/30 border-blue-400' : 'bg-white/5 hover:bg-white/10'} text-white border border-transparent text-sm py-2 rounded transition-colors flex items-center justify-center gap-2`}>
                                Monitor
                            </button>
                            <button onClick={() => handleModeSwitch(device.node_id, 'TRAIN')} className={`flex-1 ${device.status==='TRAIN' ? 'bg-purple-500/30 border-purple-400' : 'bg-white/5 hover:bg-white/10'} text-purple-200 border border-transparent text-sm py-2 rounded transition-colors flex items-center justify-center gap-2`}>
                                Train (Real Data)
                            </button>
                            <button onClick={() => handleModeSwitch(device.node_id, 'DEMO_TRAIN')} className={`flex-1 ${device.status==='DEMO_TRAIN' ? 'bg-emerald-500/30 border-emerald-400' : 'bg-white/5 hover:bg-white/10'} text-emerald-200 border border-transparent text-sm py-2 rounded transition-colors flex items-center justify-center gap-2`}>
                                Quick Demo Train
                            </button>
                        </div>
                    </div>
                    
                    <div className="flex gap-3 pt-2">
                        <button onClick={() => handleContainerAction(device.node_id, 'restart')} className="flex-1 bg-white/5 hover:bg-white/10 text-white text-sm py-2 rounded transition-colors flex items-center justify-center gap-2">
                            <Power className="w-4 h-4" /> Restart Container
                        </button>
                    </div>

                </CardContent>
            </Card>
        ))}
      </div>
      
      {broadcastToast && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 bg-emerald-500 text-white px-6 py-3 rounded-full shadow-lg font-bold animate-in slide-in-from-top flex items-center gap-2">
            <Zap className="w-5 h-5" /> {broadcastToast}
        </div>
      )}
    </div>
  );
}
