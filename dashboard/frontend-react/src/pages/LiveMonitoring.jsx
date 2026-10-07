import React, { useRef, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeofenceDrawer from "../components/GeofenceDrawer";
import AlertFeed from "../components/AlertFeed";
import { Server, Cpu, Database, Activity, UploadCloud, Play, Pause, RotateCcw, Camera, Film } from 'lucide-react';

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

  // New Demo Video States
  const [sourceMode, setSourceMode] = useState("webcam"); // "webcam" | "upload"
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle, uploading, success, error
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [demoState, setDemoState] = useState("idle"); // idle, playing, paused, finished

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
    if (sourceMode !== "webcam" || !selectedDevice) {
      setStreamActive(false);
      return;
    }
    
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
  }, [selectedDevice, sourceMode]);

  useEffect(() => {
    if (sourceMode === "webcam") {
      fetch('/api/edge/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: targetNode, source_type: "webcam" })
      }).catch(e => console.error(e));
      setDemoState("idle");
    }
  }, [sourceMode, targetNode]);

  const isProcessingRef = useRef(false);

  useEffect(() => {
    if (!streamActive || !socket || sourceMode !== "webcam") return;
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
  }, [streamActive, socket, targetNode, sourceMode]);

  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  const [demoSessionId, setDemoSessionId] = useState(null);

  useEffect(() => {
    if (!socket) return;
    const handler = (data) => {
        isProcessingRef.current = false;
        
        // Handle new dictionary format or old string format
        const imgStr = typeof data === 'string' ? data : data.image;
        const msgSessionId = typeof data === 'string' ? null : data.session_id;
        
        // Prevent stale frames from old sessions
        if (sourceMode === "upload" && demoSessionId && msgSessionId && msgSessionId !== demoSessionId) {
            return;
        }
        
        setProcessedImg(imgStr);
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
            if (sourceMode === "upload" && demoSessionId && data.session_id && data.session_id !== demoSessionId) {
                return; // Stale event
            }
            setDemoState("finished");
        }
    };
    const startedHandler = (data) => {
        if (sourceMode === "upload" && demoSessionId && data.session_id === demoSessionId) {
            setDemoState("playing");
        }
    };
    const errorHandler = (data) => {
        if (sourceMode === "upload" && demoSessionId && data.session_id === demoSessionId) {
            setDemoState("error");
            alert("Edge inference failed to start or process video.");
        }
    };
    
    socket.on('processed_frame_result', handler);
    socket.on('demo_finished', finishedHandler);
    socket.on('demo_started', startedHandler);
    socket.on('demo_error', errorHandler);
    
    return () => {
        socket.off('processed_frame_result', handler);
        socket.off('demo_finished', finishedHandler);
        socket.off('demo_started', startedHandler);
        socket.off('demo_error', errorHandler);
    };
  }, [socket, targetNode, sourceMode, demoSessionId]);

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
    const newSessionId = crypto.randomUUID();
    setDemoSessionId(newSessionId);
    setDemoState("starting");
    
    fetch('/api/edge/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        node_id: targetNode, 
        source_type: "upload",
        video_path: uploadedFile.file_path,
        session_id: newSessionId
      })
    }).then(() => {
        sendDemoControl("play");
    }).catch(() => {
        alert("Unable to start demo analysis.");
        setDemoState("error");
    });
  };

  const sendDemoControl = (action) => {
    fetch('/api/edge/demo/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: targetNode, action })
    }).catch(e => console.error(e));
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
      const newSessionId = crypto.randomUUID();
      setDemoSessionId(newSessionId);
      setDemoState("starting");
      setProcessedImg(null); // Clear last frame
      
      fetch('/api/edge/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          node_id: targetNode, 
          source_type: "upload",
          video_path: uploadedFile.file_path,
          session_id: newSessionId
        })
      }).then(() => {
          sendDemoControl("play");
      }).catch(() => {
          alert("Unable to replay demo.");
          setDemoState("error");
      });
  };

  const sendDemoControlForNode = (nodeId, action) => {
    fetch('/api/edge/demo/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: nodeId, action })
    }).catch(e => console.error(e));
  };

  const handleModelChange = (newNodeId) => {
      // Always stop old node cleanly
      fetch('/api/edge/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ node_id: targetNode, source_type: "stopped" })
      });
      
      setProcessedImg(null); // Clear UI to prevent frozen frames
      
      if (sourceMode === "upload" && demoState !== "idle" && uploadedFile) {
          setTargetNode(newNodeId);
          
          const newSessionId = crypto.randomUUID();
          setDemoSessionId(newSessionId);
          setDemoState("starting");
          
          fetch('/api/edge/source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              node_id: newNodeId, 
              source_type: "upload",
              video_path: uploadedFile.file_path,
              session_id: newSessionId
            })
          }).then(() => {
              sendDemoControlForNode(newNodeId, "play");
          });
      } else {
          setTargetNode(newNodeId);
      }
  };

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
            
            {/* Source Selector */}
            <div className="flex flex-col gap-2">
                <span className="text-sm text-slate-400 font-medium">Video Source</span>
                <div className="flex bg-slate-900/50 p-1 rounded-md text-sm border border-white/10 w-fit">
                    <button 
                      onClick={() => setSourceMode("webcam")}
                      className={`flex items-center gap-2 px-4 py-2 rounded transition-all ${sourceMode === "webcam" ? "bg-slate-700 text-white shadow" : "text-slate-400 hover:text-white"}`}
                    >
                        <Camera className="w-4 h-4" /> Live Camera
                    </button>
                    <button 
                      onClick={() => setSourceMode("upload")}
                      className={`flex items-center gap-2 px-4 py-2 rounded transition-all ${sourceMode === "upload" ? "bg-slate-700 text-white shadow" : "text-slate-400 hover:text-white"}`}
                    >
                        <Film className="w-4 h-4" /> Demo Video Upload
                    </button>
                </div>
            </div>

            <div className="flex gap-4">
              {sourceMode === "webcam" && (
                  <select className="bg-slate-800 text-white text-sm rounded-md px-3 py-2 border border-white/20 outline-none" value={selectedDevice} onChange={(e) => setSelectedDevice(e.target.value)}>
                      {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.substring(0,5)}`}</option>)}
                  </select>
              )}
              <select className="bg-slate-800 text-white text-sm rounded-md px-3 py-2 border border-white/20 outline-none" value={targetNode} onChange={(e) => handleModelChange(e.target.value)}>
                  <option value="node_constrained">Edge Node 1 (Constrained)</option>
                  <option value="node_capable">Edge Node 2 (Capable)</option>
              </select>
            </div>

            {sourceMode === "upload" && (
                <div className="w-full bg-black/40 p-6 rounded-lg border border-white/10">
                    {uploadStatus === "idle" || uploadStatus === "error" ? (
                        <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-slate-600 rounded-lg hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all cursor-pointer" onClick={() => document.getElementById('video-upload').click()}>
                            <UploadCloud className="w-12 h-12 text-slate-400 mb-4" />
                            <p className="text-white font-medium mb-2 text-lg">Upload Demo Video</p>
                            <p className="text-sm text-slate-400 mb-4">Drag & drop your video here or <span className="text-emerald-400">Choose Video</span></p>
                            <p className="text-xs text-slate-500 font-mono">MP4 • AVI • MOV</p>
                            <input type="file" id="video-upload" accept=".mp4,.avi,.mov" className="hidden" onChange={handleFileUpload} />
                            {uploadStatus === "error" && <p className="text-red-400 text-sm mt-4">Video upload failed. Please try again.</p>}
                        </div>
                    ) : uploadStatus === "uploading" ? (
                        <div className="flex flex-col items-center p-8 border border-slate-700 rounded-lg">
                            <p className="text-sm text-slate-300 mb-4 font-mono">Uploading video...</p>
                            <div className="w-full max-w-md bg-slate-800 rounded-full h-2.5 mb-2 overflow-hidden">
                              <div className="bg-emerald-500 h-full transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
                            </div>
                            <p className="text-sm text-emerald-400 font-mono">{uploadProgress}%</p>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center justify-between bg-slate-800/50 p-4 rounded-lg border border-emerald-500/20">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 text-lg">✓</div>
                                    <div>
                                        <p className="font-medium text-emerald-400 text-sm">Uploaded successfully</p>
                                        <p className="text-white font-mono text-sm">{uploadedFile.filename}</p>
                                    </div>
                                </div>
                                {demoState === "idle" && (
                                    <button onClick={startDemoAnalysis} className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded text-sm font-medium transition-colors shadow-lg shadow-emerald-900/20">
                                        <Play className="w-4 h-4" /> Start Demo Analysis
                                    </button>
                                )}
                            </div>

                            {/* Demo Playback Controls */}
                            {demoState !== "idle" && uploadStatus === "success" && (
                                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border border-white/10 bg-slate-800/30 p-4 rounded-lg gap-4">
                                    <div className="flex items-center gap-3">
                                        <button onClick={togglePlayPause} className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded text-sm text-white font-medium transition-colors">
                                            {demoState === "playing" ? <Pause className="w-4 h-4"/> : <Play className="w-4 h-4"/>} 
                                            {demoState === "playing" ? "Pause" : "Play"}
                                        </button>
                                        <button onClick={handleReset} className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded text-sm text-white font-medium transition-colors">
                                            <RotateCcw className="w-4 h-4"/> Replay
                                        </button>
                                    </div>
                                    <div className="flex items-center gap-2 font-mono text-sm">
                                        <div className={`w-2 h-2 rounded-full ${demoState === "playing" ? "bg-emerald-400 animate-pulse" : demoState === "starting" ? "bg-blue-400 animate-pulse" : demoState === "finished" ? "bg-slate-400" : demoState === "error" ? "bg-red-500" : "bg-amber-400"}`}></div>
                                        <span className={demoState === "playing" ? "text-emerald-400" : demoState === "starting" ? "text-blue-400" : demoState === "finished" ? "text-slate-300" : demoState === "error" ? "text-red-500" : "text-amber-400"}>
                                            {demoState === "playing" ? "DEMO PLAYING" : demoState === "starting" ? "DEMO STARTING" : demoState === "finished" ? "DEMO FINISHED" : demoState === "error" ? "DEMO ERROR" : "DEMO PAUSED"}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
            
            <video ref={videoRef} autoPlay playsInline muted className="hidden" />
            <canvas ref={canvasRef} width="320" height="240" className="hidden" />

            <div className="w-full border border-white/20 rounded-lg overflow-hidden bg-black/60 relative aspect-video flex items-center justify-center">
              {processedImg ? (
                <>
                  <img src={processedImg} alt="Processed feed" className="absolute inset-0 w-full h-full object-cover" />
                  <div className="absolute inset-0 z-10"><GeofenceDrawer onSave={(pts) => setPolygon(pts)} initialPolygon={polygon} /></div>
                </>
              ) : (
                <span className="text-muted-foreground animate-pulse">
                    {sourceMode === "webcam" ? "Awaiting Edge processing..." : demoState === "starting" ? "Starting Edge inference..." : "Upload and Start Demo to view inference"}
                </span>
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
