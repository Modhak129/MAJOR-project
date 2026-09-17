import React, { useRef, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeofenceDrawer from "../components/GeofenceDrawer";
import AlertFeed from "../components/AlertFeed";
import { Server, Cpu, Database, Activity } from 'lucide-react';

export default function LiveMonitoring({ socket }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [processedImg, setProcessedImg] = useState(null);
  const [streamActive, setStreamActive] = useState(false);
  const [polygon, setPolygon] = useState([]);
  const [devices, setDevices] = useState([]);
  const [dbDevices, setDbDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [targetNode, setTargetNode] = useState("node_constrained");

  useEffect(() => {
    const getDevices = async () => {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devs.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0) setSelectedDevice(videoDevices[0].deviceId);
    };
    getDevices();
    
    fetch('/api/geofence').then(r => r.json()).then(data => {
        if(data.polygon) setPolygon(data.polygon);
    });

    fetch('/api/db/devices').then(r => r.json()).then(data => setDbDevices(data));
  }, []);

  useEffect(() => {
    if (!selectedDevice) return;
    let stream = null;
    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { deviceId: selectedDevice ? { exact: selectedDevice } : undefined } 
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStreamActive(true);
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };
    startWebcam();
    return () => { if (stream) stream.getTracks().forEach(track => track.stop()); };
  }, [selectedDevice]);

  const isProcessingRef = useRef(false);

  useEffect(() => {
    if (!streamActive || !socket) return;
    const intervalId = setInterval(() => {
      if (videoRef.current && canvasRef.current && !isProcessingRef.current) {
        isProcessingRef.current = true;
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.5);
        socket.emit('frontend_frame', { image: dataUrl, target_node: targetNode });
        
        // Prevent permanent lockup if edge node drops the frame
        setTimeout(() => {
            isProcessingRef.current = false;
        }, 1000);
      }
    }, 33); 
    return () => clearInterval(intervalId);
  }, [streamActive, socket, targetNode]);

  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  useEffect(() => {
    if (!socket) return;
    const handler = (data) => {
        isProcessingRef.current = false;
        setProcessedImg(data);
        frameCountRef.current += 1;
        const now = Date.now();
        if (now - lastTimeRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastTimeRef.current = now;
        }
    };
    socket.on('processed_frame_result', handler);
    return () => socket.off('processed_frame_result', handler);
  }, [socket]);

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      
      {/* Top row: Video Feed & Alerts */}
      <div className="flex flex-col gap-6">
        <Card className="shadow-lg border-white/10 bg-white/5 backdrop-blur-md">
          <CardHeader>
            <CardTitle className="flex justify-between items-center w-full">
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">Live Edge Vision</span>
              <span className="text-sm font-mono text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-3 py-1 rounded-md">{fps} FPS</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex gap-4">
              <select className="bg-slate-800 text-white text-sm rounded-md px-3 py-2 border border-white/20 outline-none" value={selectedDevice} onChange={(e) => setSelectedDevice(e.target.value)}>
                  {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.substring(0,5)}`}</option>)}
              </select>
              <select className="bg-slate-800 text-white text-sm rounded-md px-3 py-2 border border-white/20 outline-none" value={targetNode} onChange={(e) => setTargetNode(e.target.value)}>
                  <option value="node_constrained">Edge Node 1 (Constrained)</option>
                  <option value="node_capable">Edge Node 2 (Capable)</option>
              </select>
            </div>
            
            <video ref={videoRef} autoPlay playsInline muted className="hidden" />
            <canvas ref={canvasRef} width="320" height="240" className="hidden" />

            <div className="w-full border border-white/20 rounded-lg overflow-hidden bg-black/60 relative aspect-video flex items-center justify-center">
              {processedImg ? (
                <>
                  <img src={processedImg} alt="Processed feed" className="absolute inset-0 w-full h-full object-cover" />
                  <div className="absolute inset-0 z-10"><GeofenceDrawer onSave={(pts) => setPolygon(pts)} initialPolygon={polygon} /></div>
                </>
              ) : (
                <span className="text-muted-foreground animate-pulse">Awaiting Edge processing...</span>
              )}
            </div>
          </CardContent>
        </Card>
        
        <div className="w-full">
           <AlertFeed socket={socket} />
        </div>
      </div>
    </div>
  );
}
