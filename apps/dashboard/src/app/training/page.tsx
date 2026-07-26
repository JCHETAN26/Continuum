"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from "react";

const generateMockLoss = () => {
  const data = [];
  let loss = 2.5;
  for (let i = 0; i < 50; i++) {
    loss = loss * 0.95 + (Math.random() * 0.1);
    data.push({
      step: i * 10,
      loss: Math.max(0.2, loss)
    });
  }
  return data;
};

export default function TrainingPage() {
  const [lossData, setLossData] = useState<any[]>([]);

  useEffect(() => {
    // Animate the line drawing in by adding data points over time
    const fullData = generateMockLoss();
    let index = 0;
    
    const interval = setInterval(() => {
      if (index < fullData.length) {
        setLossData(prev => [...prev, fullData[index]]);
        index++;
      } else {
        clearInterval(interval);
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Training Monitor</h1>
        <p className="text-muted-foreground">Live telemetry for background LoRA fine-tuning jobs.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>Active Job Details</CardTitle>
            <CardDescription>Job ID: 2026.07.26-1a2b</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Status</span>
              <Badge className="bg-blue-500/20 text-blue-500 animate-pulse">TRAINING</Badge>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Base Model</span>
              <span className="font-medium">sentence-transformers/all-MiniLM-L6-v2</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Technique</span>
              <span className="font-medium">PEFT LoRA (rank=8, alpha=16)</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Dataset Size</span>
              <span className="font-medium">4,250 examples</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle>Training Loss</CardTitle>
            <CardDescription>Multiple Negatives Ranking (MNR) Loss</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px] w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis 
                    dataKey="step" 
                    stroke="hsl(var(--muted-foreground))" 
                    fontSize={12} 
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    domain={[0, 3]} 
                    stroke="hsl(var(--muted-foreground))" 
                    fontSize={12} 
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                    itemStyle={{ color: 'hsl(var(--foreground))' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="hsl(var(--primary))" 
                    strokeWidth={3} 
                    dot={false}
                    activeDot={{ r: 6, fill: 'hsl(var(--primary))', strokeWidth: 0 }}
                    animationDuration={300}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
