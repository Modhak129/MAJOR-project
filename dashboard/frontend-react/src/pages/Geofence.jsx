import React, { useRef, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import GeofenceDrawer from "../components/GeofenceDrawer";

export default function Geofence() {
  const videoRef = useRef(null);
  const [polygon, setPolygon] = useState([]);
  const [streamActive, setStreamActive] = useState(false);

  useEffect(() => {
    let stream;
    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStreamActive(true);
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };
    startWebcam();
    
    fetch('/api/geofence').then(r => r.json()).then(data => {
      if(data.polygon) setPolygon(data.polygon);
    });

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Interactive Geofencing</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground mb-4">Click on the webcam feed below to draw a region of interest. Any detection within this polygon will trigger a Zone Intrusion alert.</p>
        <div className="relative w-full max-w-2xl mx-auto border rounded bg-black overflow-hidden aspect-video">
           {!streamActive && <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">Waiting for camera...</div>}
           <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
           <div className="absolute inset-0 z-10">
             <GeofenceDrawer onSave={(pts) => setPolygon(pts)} initialPolygon={polygon} />
           </div>
        </div>
      </CardContent>
    </Card>
  );
}
