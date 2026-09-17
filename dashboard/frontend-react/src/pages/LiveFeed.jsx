import React, { useRef, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeofenceDrawer from "../components/GeofenceDrawer";

export default function LiveFeed({ socket }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [processedImg, setProcessedImg] = useState(null);
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const [streamActive, setStreamActive] = useState(false);
  const [polygon, setPolygon] = useState([]);
  
  // Camera and Node Selection
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [targetNode, setTargetNode] = useState("node_constrained");
  const [stream, setStream] = useState(null);

  useEffect(() => {
    const getDevices = async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0) setSelectedDevice(videoDevices[0].deviceId);
    };
    getDevices();
    
    fetch('/api/geofence').then(r => r.json()).then(data => {
        if(data.polygon) setPolygon(data.polygon);
    }).catch(e => console.error(e));
  }, []);

  useEffect(() => {
    if (!selectedDevice) return;
    
    const startWebcam = async () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
      
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({ 
            video: { deviceId: selectedDevice ? { exact: selectedDevice } : undefined } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = newStream;
        }
        setStream(newStream);
        setStreamActive(true);
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };
    startWebcam();

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [selectedDevice]);

  useEffect(() => {
    if (!streamActive || !socket) return;

    const intervalId = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.5);
        socket.emit('frontend_frame', { image: dataUrl, target_node: targetNode });
      }
    }, 100); 

    return () => clearInterval(intervalId);
  }, [streamActive, socket, targetNode]);

  useEffect(() => {
    if (!socket) return;
    const handler = (data) => {
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
    <Card className="col-span-1 lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <span>Live Inference Feed</span>
          <span className="text-sm font-normal text-muted-foreground">FPS: {fps}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        
        <div className="flex gap-4">
            <select 
                className="bg-secondary text-secondary-foreground text-sm rounded-md px-3 py-2 border border-input outline-none"
                value={selectedDevice} 
                onChange={(e) => setSelectedDevice(e.target.value)}
            >
                {devices.map(d => (
                    <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.substring(0,5)}`}</option>
                ))}
            </select>
            
            <select 
                className="bg-secondary text-secondary-foreground text-sm rounded-md px-3 py-2 border border-input outline-none"
                value={targetNode} 
                onChange={(e) => setTargetNode(e.target.value)}
            >
                <option value="node_constrained">Edge Node 1 (Constrained)</option>
                <option value="node_capable">Edge Node 2 (Capable)</option>
            </select>
        </div>

        <video ref={videoRef} autoPlay playsInline muted className="hidden" />
        <canvas ref={canvasRef} width="320" height="240" className="hidden" />

        <div className="w-full max-w-3xl mx-auto border rounded bg-black/50 aspect-video relative flex items-center justify-center overflow-hidden">
          {processedImg ? (
            <>
              <img src={processedImg} alt="Processed feed" className="absolute inset-0 w-full h-full object-cover" />
              {/* Overlay geofence drawing tool but read-only / update via original tool */}
              <div className="absolute inset-0 z-10 pointer-events-none">
                 <GeofenceDrawer initialPolygon={polygon} />
              </div>
            </>
          ) : (
            <span className="text-muted-foreground">Waiting for edge node processing...</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
