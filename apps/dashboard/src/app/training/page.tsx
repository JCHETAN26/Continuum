'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const TRAINER_API = process.env.NEXT_PUBLIC_TRAINER_API_URL ?? 'http://localhost:8003';

interface TrainingJob {
  id: string;
  status: string;
  trigger: string;
  sampleCount: number | null;
  lossHistory: { step: number; loss: number }[] | null;
  queuedAt: string;
}

interface TrainingEventPayload {
  jobs: TrainingJob[];
}

export default function TrainingPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);

  useEffect(() => {
    const refreshSnapshot = async () => {
      const response = await fetch(`${TRAINER_API}/v1/training/jobs`);
      if (response.ok) {
        setJobs((await response.json()) as TrainingJob[]);
      }
    };

    void refreshSnapshot();

    if (typeof EventSource === 'undefined') {
      const interval = setInterval(() => {
        void refreshSnapshot();
      }, 3000);
      return () => {
        clearInterval(interval);
      };
    }

    const trainingEvents = new EventSource(`${TRAINER_API}/v1/training/events`);
    trainingEvents.addEventListener('training', (event) => {
      const payload = JSON.parse(event.data as string) as TrainingEventPayload;
      setJobs(payload.jobs);
    });

    return () => {
      trainingEvents.close();
    };
  }, []);

  const activeJob = jobs.length > 0 ? jobs[0] : null;
  const lossData = activeJob?.lossHistory ?? [];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Training Monitor</h1>
        <p className="text-muted-foreground">Telemetry from drift-triggered adaptation jobs.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>Latest Job</CardTitle>
            <CardDescription>{activeJob?.id ?? 'Waiting for drift alert'}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Status</span>
              <Badge>{activeJob?.status ?? 'IDLE'}</Badge>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Trigger</span>
              <span className="font-medium">{activeJob?.trigger ?? 'none'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Technique</span>
              <span className="font-medium">Demo LoRA gate</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border/50">
              <span className="text-muted-foreground">Dataset Size</span>
              <span className="font-medium">{activeJob?.sampleCount ?? 0} examples</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle>Training Loss</CardTitle>
            <CardDescription>Recorded by the local training worker.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px] w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="step"
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    domain={[0, 2]}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      borderColor: 'hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="hsl(var(--primary))"
                    strokeWidth={3}
                    dot={false}
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
