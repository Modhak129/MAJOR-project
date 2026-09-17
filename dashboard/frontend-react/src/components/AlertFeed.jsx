import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle } from 'lucide-react';

export default function AlertFeed({ socket }) {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetch('/api/alerts').then(r => r.json()).then(data => setAlerts(data));
  }, []);

  useEffect(() => {
    if (!socket) return;
    socket.on('new_alert', (data) => {
      setAlerts(prev => [data, ...prev].slice(0, 50));
    });
  }, [socket]);

  return (
    <Card className="col-span-1 h-[400px] flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-destructive">
          <AlertTriangle className="w-5 h-5" /> Live Safety Alerts
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Node</TableHead>
              <TableHead>Worker ID</TableHead>
              <TableHead>Violation</TableHead>
              <TableHead>Conf</TableHead>
              <TableHead>Time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {alerts.map((a, i) => (
              <TableRow key={i}>
                <TableCell className="font-medium text-xs">{a.client_id}</TableCell>
                <TableCell className="text-xs font-mono">{a.worker_id != null ? `#${a.worker_id}` : 'N/A'}</TableCell>
                <TableCell className="text-xs">
                  <div className="text-destructive font-semibold">{a.type}</div>
                  {a.details && typeof a.details === 'object' && a.details.missing_items && (
                    <div className="flex gap-1 mt-1 flex-wrap">
                      {a.details.missing_items.map((item, idx) => (
                        <Badge key={idx} variant="destructive" className="text-[10px] px-1 py-0 h-4">
                          No {item}
                        </Badge>
                      ))}
                    </div>
                  )}
                  {a.details && typeof a.details === 'string' && (
                    <div className="text-muted-foreground text-[10px] mt-1">{a.details}</div>
                  )}
                </TableCell>
                <TableCell className="text-xs">{a.confidence.toFixed(2)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{new Date(a.timestamp).toLocaleTimeString()}</TableCell>
              </TableRow>
            ))}
            {alerts.length === 0 && (
                <TableRow>
                    <TableCell colSpan={5} className="text-center text-muted-foreground">No alerts active.</TableCell>
                </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
