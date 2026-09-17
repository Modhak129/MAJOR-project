import React, { useRef, useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";

export default function GeofenceDrawer({ onSave, initialPolygon = [] }) {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    if (initialPolygon && initialPolygon.length > 0) {
      setPoints(initialPolygon);
    }
  }, [initialPolygon]);

  useEffect(() => {
    drawPolygon();
  }, [points]);

  const drawPolygon = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (points.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    
    // Fill if we have at least 3 points
    if (points.length > 2) {
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
      ctx.fill();
    }
    
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw dots
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.stroke();
    });
  };

  const handleCanvasClick = (e) => {
    if (!isDrawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    
    // Calculate scaling ratios since canvas visual size differs from internal size
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    setPoints([...points, [x, y]]);
  };

  const handleSave = () => {
    setIsDrawing(false);
    fetch('/api/geofence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ polygon: points })
    });
    if (onSave) onSave(points);
  };

  const handleClear = () => {
    setPoints([]);
    setIsDrawing(true);
  };

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        width="640" 
        height="480"
        className={`w-full h-full ${isDrawing ? 'cursor-crosshair' : 'cursor-default'}`}
        onClick={handleCanvasClick}
      />
      <div className="absolute top-2 right-2 flex gap-2">
        {!isDrawing ? (
          <Button size="sm" variant="secondary" onClick={() => setIsDrawing(true)}>Edit Geofence</Button>
        ) : (
          <>
            <Button size="sm" variant="destructive" onClick={handleClear}>Clear</Button>
            <Button size="sm" onClick={handleSave}>Save Geofence</Button>
          </>
        )}
      </div>
    </div>
  );
}
