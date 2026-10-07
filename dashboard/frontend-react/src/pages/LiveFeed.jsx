import React, { useRef, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeofenceDrawer from "../components/GeofenceDrawer";
import { UploadCloud, Play, Pause, RotateCcw, Camera, Film } from "lucide-react";

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

  // New states for Video Upload Mode
  const [sourceType, setSourceType] = useState("webcam"); // "webcam" | "upload"
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle, uploading, success, error
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [demoState, setDemoState] = useState("idle"); // idle, playing, paused, finished

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
    if (sourceType !== "webcam" || !selectedDevice) {
      if (stream) stream.getTracks().forEach(track => track.stop());
      setStreamActive(false);
      return;
    }
    
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
  }, [selectedDevice, sourceType]);

  // Tell edge node to switch source when sourceType changes
  useEffect(() => {
    if (sourceType === "webcam") {
      fetch('/api/edge/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: targetNode, source_type: "webcam" })
      });
      setDemoState("idle");
    }
  }, [sourceType, targetNode]);

  // Frame sender for webcam mode only
  useEffect(() => {
    if (!streamActive || !socket || sourceType !== "webcam") return;

    const intervalId = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.5);
        socket.emit('frontend_frame', { image: dataUrl, target_node: targetNode });
      }
    }, 100); 

    return () => clearInterval(intervalId);
  }, [streamActive, socket, targetNode, sourceType]);

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
    const finishedHandler = (data) => {
        if (data.node_id && (data.node_id === targetNode || (targetNode === "node_constrained" && data.node_id === "edge_node_1") || (targetNode === "node_capable" && data.node_id === "edge_node_2"))) {
            setDemoState("finished");
        }
    };
    socket.on('processed_frame_result', handler);
    socket.on('demo_finished', finishedHandler);
    return () => {
        socket.off('processed_frame_result', handler);
        socket.off('demo_finished', finishedHandler);
    };
  }, [socket, targetNode]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);

    setUploadStatus("uploading");
    setUploadProgress(0);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/video/upload", true);
    
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(percent);
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        setUploadedFile(response);
        setUploadStatus("success");
      } else {
        setUploadStatus("error");
      }
    };

    xhr.onerror = () => {
      setUploadStatus("error");
    };

    xhr.send(formData);
  };

  const startDemoAnalysis = () => {
    if (!uploadedFile) return;
    fetch('/api/edge/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        node_id: targetNode, 
        source_type: "upload",
        video_path: uploadedFile.file_path 
      })
    }).then(() => {
        sendDemoControl("play");
        setDemoState("playing");
    });
  };

  const sendDemoControl = (action) => {
    fetch('/api/edge/demo/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: targetNode, action })
    });
  };

  const togglePlayPause = () => {
      if (demoState === "playing") {
          sendDemoControl("pause");
          setDemoState("paused");
      } else {
          sendDemoControl("play");
          setDemoState("playing");
      }
  };

  const handleReset = () => {
      sendDemoControl("reset");
      setDemoState("paused");
      setProcessedImg(null); // Clear last frame
  };

  return (
    <Card className="col-span-1 lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="flex items-center gap-4">
             <span>Live Inference Feed</span>
             
             {/* Source Toggle */}
             <div className="flex bg-secondary p-1 rounded-md text-sm border border-input">
                <button 
                  onClick={() => setSourceType("webcam")}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded transition-all ${sourceType === "webcam" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                >
                    <Camera className="w-4 h-4" /> Live Camera
                </button>
                <button 
                  onClick={() => setSourceType("upload")}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded transition-all ${sourceType === "upload" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                >
                    <Film className="w-4 h-4" /> Demo Video Upload
                </button>
             </div>
          </div>
          
          <span className="text-sm font-normal text-muted-foreground">FPS: {fps}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        
        {/* Connection Tools */}
        <div className="flex gap-4 items-center">
            {sourceType === "webcam" && (
                <select 
                    className="bg-secondary text-secondary-foreground text-sm rounded-md px-3 py-2 border border-input outline-none"
                    value={selectedDevice} 
                    onChange={(e) => setSelectedDevice(e.target.value)}
                >
                    {devices.map(d => (
                        <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.substring(0,5)}`}</option>
                    ))}
                </select>
            )}
            
            <select 
                className="bg-secondary text-secondary-foreground text-sm rounded-md px-3 py-2 border border-input outline-none"
                value={targetNode} 
                onChange={(e) => setTargetNode(e.target.value)}
            >
                <option value="node_constrained">Edge Node 1 (Constrained)</option>
                <option value="node_capable">Edge Node 2 (Capable)</option>
            </select>
        </div>

        {sourceType === "upload" && (
            <div className="w-full bg-slate-900/50 p-6 rounded-lg border border-slate-700">
                {uploadStatus === "idle" || uploadStatus === "error" ? (
                    <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-600 rounded-lg hover:border-indigo-500 transition-colors cursor-pointer" onClick={() => document.getElementById('video-upload').click()}>
                        <UploadCloud className="w-12 h-12 text-slate-400 mb-4" />
                        <p className="text-slate-300 font-medium mb-1">Click to Upload Demo Video</p>
                        <p className="text-xs text-slate-500">MP4, AVI, or MOV files</p>
                        <input type="file" id="video-upload" accept="video/mp4,video/x-m4v,video/*" className="hidden" onChange={handleFileUpload} />
                        {uploadStatus === "error" && <p className="text-red-400 text-sm mt-4">Upload failed. Please try again.</p>}
                    </div>
                ) : uploadStatus === "uploading" ? (
                    <div className="flex flex-col items-center p-6">
                        <div className="w-full max-w-md bg-slate-800 rounded-full h-2.5 mb-2">
                          <div className="bg-indigo-500 h-2.5 rounded-full transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
                        </div>
                        <p className="text-sm text-slate-400">Uploading... {uploadProgress}%</p>
                    </div>
                ) : (
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3 text-emerald-400">
                            <span className="text-xl">✓</span>
                            <div>
                                <p className="font-semibold">{uploadedFile.filename} uploaded successfully</p>
                            </div>
                        </div>
                        {demoState === "idle" && (
                            <button onClick={startDemoAnalysis} className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded font-medium transition-colors">
                                Start Demo Analysis
                            </button>
                        )}
                    </div>
                )}

                {/* Demo Playback Controls */}
                {demoState !== "idle" && uploadStatus === "success" && (
                    <div className="mt-6 flex items-center justify-between border-t border-slate-700 pt-4">
                        <div className="flex items-center gap-4">
                            <button onClick={togglePlayPause} className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded text-sm text-white">
                                {demoState === "playing" ? <Pause className="w-4 h-4"/> : <Play className="w-4 h-4"/>} 
                                {demoState === "playing" ? "Pause" : "Play"}
                            </button>
                            <button onClick={handleReset} className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded text-sm text-white">
                                <RotateCcw className="w-4 h-4"/> Replay / Reset
                            </button>
                        </div>
                        <div className="text-sm text-slate-400 flex items-center gap-2">
                            Status: 
                            <span className={demoState === "playing" ? "text-emerald-400" : demoState === "finished" ? "text-amber-400" : "text-amber-400"}>
                                {demoState === "playing" ? "Playing" : demoState === "finished" ? "Demo Finished" : "Paused"}
                            </span>
                        </div>
                    </div>
                )}
            </div>
        )}

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
            <span className="text-muted-foreground">{sourceType === "webcam" ? "Waiting for edge node processing..." : "Upload and start demo to view inference"}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
