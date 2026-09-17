import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Play, Square, Power, RefreshCw, BrainCircuit } from 'lucide-react';

export default function DeviceManager() {
  const [containers, setContainers] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState("Paused");

  const fetchContainers = async () => {
    try {
      const res = await fetch('/api/containers');
      const data = await res.json();
      setContainers(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchContainers();
    const int = setInterval(fetchContainers, 5000);
    return () => clearInterval(int);
  }, []);

  const handleAction = async (id, action) => {
    await fetch(`/api/containers/${id}/${action}`, { method: 'POST' });
    fetchContainers();
  };

  const controlEdge = async (client_id, command) => {
    await fetch(`/api/edge/${client_id}/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command })
    });
    if (command === "start_training") setTrainingStatus("Running");
    if (command === "stop_training") setTrainingStatus("Paused");
  };

  return (
    <Card className="col-span-1">
      <CardHeader>
        <CardTitle>Edge Device Management</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Node</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Docker</TableHead>
              <TableHead>Controls</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {containers.map(c => {
              let clientId = "node_constrained";
              if (c.name.includes("edge_node_2")) clientId = "node_capable";
              return (
              <TableRow key={c.id}>
                <TableCell className="font-medium text-xs">{c.name.substring(0,30)}</TableCell>
                <TableCell>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${c.status === 'running' || c.status === 'native' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                    {c.status}
                  </span>
                </TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button size="icon" variant="ghost" onClick={() => handleAction(c.id, 'start')} title="Start Container"><Play className="w-4 h-4 text-green-500" /></Button>
                      <Button size="icon" variant="ghost" onClick={() => handleAction(c.id, 'stop')} title="Stop Container"><Square className="w-4 h-4 text-red-500" /></Button>
                      <Button size="icon" variant="ghost" onClick={() => handleAction(c.id, 'restart')} title="Restart Container"><RefreshCw className="w-4 h-4 text-blue-500" /></Button>
                    </div>
                  </TableCell>
                <TableCell>
                   <div className="flex flex-col gap-2">
                      <div className="flex gap-1">
                        <Button size="sm" variant="outline" className="text-xs w-full" onClick={() => controlEdge(clientId, 'start_inference')}>Start Inference</Button>
                        <Button size="sm" variant="outline" className="text-xs w-full" onClick={() => controlEdge(clientId, 'stop_inference')}>Stop Inference</Button>
                      </div>
                   </div>
                </TableCell>
              </TableRow>
            )})}
          </TableBody>
        </Table>

        <div className="mt-6 border-t pt-4 border-border">
            <h4 className="text-sm font-semibold mb-2">Federated Learning Controls</h4>
            <div className="flex flex-wrap gap-2 items-center">
                <span className="text-sm text-muted-foreground mr-2">Schedule: <span className="text-foreground">{trainingStatus}</span></span>
                <Button size="sm" variant="secondary" onClick={() => controlEdge('node_constrained', 'start_training')}><Play className="w-4 h-4 mr-1"/> Resume FL</Button>
                <Button size="sm" variant="secondary" onClick={() => controlEdge('node_constrained', 'stop_training')}><Square className="w-4 h-4 mr-1"/> Pause FL</Button>
                <Button size="sm" onClick={() => controlEdge('node_constrained', 'trigger_training_now')}><BrainCircuit className="w-4 h-4 mr-1"/> Force Train Now</Button>
            </div>
        </div>
      </CardContent>
    </Card>
  );
}
