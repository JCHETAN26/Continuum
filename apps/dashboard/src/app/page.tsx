"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from "react";
import { format } from "date-fns";

// Mock data to simulate the API
const generateMockData = () => {
  const data = [];
  let score = 0.2;
  const now = new Date();
  for (let i = 24; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 5 * 60000);
    score += (Math.random() - 0.4) * 0.1;
    if (score < 0.1) score = 0.1;
    if (score > 0.9) score = 0.9;
    
    data.push({
      time: format(time, "HH:mm"),
      driftScore: score,
      threshold: 0.75
    });
  }
  return data;
};

export default function Home() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    setData(generateMockData());
    
    // Simulate real-time updates
    const interval = setInterval(() => {
      setData(prev => {
        const newData = [...prev.slice(1)];
        const lastScore = prev[prev.length - 1].driftScore;
        let newScore = lastScore + (Math.random() - 0.4) * 0.1;
        if (newScore < 0.1) newScore = 0.1;
        if (newScore > 0.9) newScore = 0.9;
        
        newData.push({
          time: format(new Date(), "HH:mm"),
          driftScore: newScore,
          threshold: 0.75
        });
        return newData;
      });
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const currentDrift = data.length > 0 ? data[data.length - 1].driftScore : 0;
  const isBreached = currentDrift > 0.75;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">Monitor real-time semantic drift across your embeddings.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm transition-all hover:shadow-md">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Current Drift Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter flex items-center gap-3">
              {currentDrift.toFixed(3)}
              {isBreached ? (
                <Badge variant="destructive" className="animate-pulse">Breached</Badge>
              ) : (
                <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20">Healthy</Badge>
              )}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm transition-all hover:shadow-md">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Model</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold tracking-tight truncate">
              2026.07.26-1a2b
            </div>
            <p className="text-xs text-muted-foreground mt-1">all-MiniLM-L6-v2 (LoRA)</p>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm transition-all hover:shadow-md">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Training Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter">
              {isBreached ? '1' : '0'}
            </div>
            {isBreached && <p className="text-xs text-amber-500 mt-1 animate-pulse">Adaptation in progress...</p>}
          </CardContent>
        </Card>
      </div>

      <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm overflow-hidden">
        <CardHeader>
          <CardTitle>Semantic Drift History (5m windows)</CardTitle>
          <CardDescription>Cosine distance between current window centroid and baseline.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis 
                  domain={[0, 1]} 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(val) => val.toFixed(1)}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                  itemStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="threshold" 
                  stroke="hsl(var(--destructive))" 
                  strokeWidth={2} 
                  strokeDasharray="5 5" 
                  dot={false} 
                  activeDot={false}
                  name="Threshold"
                />
                <Line 
                  type="monotone" 
                  dataKey="driftScore" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={3} 
                  dot={false}
                  activeDot={{ r: 6, fill: 'hsl(var(--primary))', strokeWidth: 0 }}
                  name="Drift Score"
                  animationDuration={300}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
