'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { format } from 'date-fns';
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

const DRIFT_API = process.env.NEXT_PUBLIC_DRIFT_API_URL ?? 'http://localhost:8001';
const TRAINER_API = process.env.NEXT_PUBLIC_TRAINER_API_URL ?? 'http://localhost:8003';

interface DriftWindow {
  id: string;
  windowStart: string;
  documentCount: number;
  driftScore: number;
  threshold: number;
  breached: boolean;
}

interface DriftSummary {
  documentCount: number;
  embeddingCount: number;
  latestDriftScore: number;
  breached: boolean;
  threshold: number;
}

interface ModelVersion {
  version: string;
  status: string;
  baseModel: string;
}

interface ProjectionPoint {
  x: number;
  y: number;
  source: string;
  label: string;
}

interface DriftEventPayload {
  windows: DriftWindow[];
  summary: DriftSummary;
  projection: { points: ProjectionPoint[] };
}

interface TrainingEventPayload {
  models: ModelVersion[];
}

export default function Home() {
  const [windows, setWindows] = useState<DriftWindow[]>([]);
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [points, setPoints] = useState<ProjectionPoint[]>([]);

  useEffect(() => {
    const refreshSnapshot = async () => {
      const [windowRes, summaryRes, modelRes, projectionRes] = await Promise.allSettled([
        fetch(`${DRIFT_API}/v1/drift/status`),
        fetch(`${DRIFT_API}/v1/drift/summary`),
        fetch(`${TRAINER_API}/v1/models`),
        fetch(`${DRIFT_API}/v1/embeddings/projection`),
      ]);

      if (windowRes.status === 'fulfilled' && windowRes.value.ok) {
        const body = (await windowRes.value.json()) as DriftWindow[];
        setWindows(body.reverse());
      }
      if (summaryRes.status === 'fulfilled' && summaryRes.value.ok) {
        setSummary((await summaryRes.value.json()) as DriftSummary);
      }
      if (modelRes.status === 'fulfilled' && modelRes.value.ok) {
        setModels((await modelRes.value.json()) as ModelVersion[]);
      }
      if (projectionRes.status === 'fulfilled' && projectionRes.value.ok) {
        const body = (await projectionRes.value.json()) as { points: ProjectionPoint[] };
        setPoints(body.points);
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

    const driftEvents = new EventSource(`${DRIFT_API}/v1/drift/events`);
    const trainingEvents = new EventSource(`${TRAINER_API}/v1/training/events`);

    driftEvents.addEventListener('drift', (event) => {
      const payload = JSON.parse(event.data as string) as DriftEventPayload;
      setWindows(payload.windows.reverse());
      setSummary(payload.summary);
      setPoints(payload.projection.points);
    });

    trainingEvents.addEventListener('training', (event) => {
      const payload = JSON.parse(event.data as string) as TrainingEventPayload;
      setModels(payload.models);
    });

    return () => {
      driftEvents.close();
      trainingEvents.close();
    };
  }, []);

  const activeModel = models.find((model) => model.status === 'ACTIVE');
  const latestScore = summary?.latestDriftScore ?? windows.at(-1)?.driftScore ?? 0;
  const threshold = summary?.threshold ?? windows.at(-1)?.threshold ?? 0.35;
  const isBreached = summary?.breached ?? latestScore > threshold;
  const chartData = windows.map((window) => ({
    time: format(new Date(window.windowStart), 'HH:mm:ss'),
    driftScore: window.driftScore,
    threshold: window.threshold,
  }));

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">
          Monitor real semantic drift from the local pipeline.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-4">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Drift Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter flex items-center gap-3">
              {latestScore.toFixed(3)}
              <Badge variant={isBreached ? 'destructive' : 'secondary'}>
                {isBreached ? 'Breached' : 'Healthy'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter">{summary?.documentCount ?? 0}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {summary?.embeddingCount ?? 0} embedded
            </p>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Active Model
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold tracking-tight truncate">
              {activeModel?.version ?? 'none'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {activeModel?.baseModel ?? 'waiting for registry'}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Model Versions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter">{models.length}</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle>Semantic Drift History</CardTitle>
            <CardDescription>
              Cosine distance between rolling centroid and baseline.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[360px] w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
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
                    dataKey="threshold"
                    stroke="hsl(var(--destructive))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Threshold"
                  />
                  <Line
                    type="monotone"
                    dataKey="driftScore"
                    stroke="hsl(var(--primary))"
                    strokeWidth={3}
                    dot={false}
                    name="Drift Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>Embedding Projection</CardTitle>
            <CardDescription>First two normalized embedding dimensions by source.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative h-[360px] rounded-md border border-border/60 bg-background overflow-hidden">
              {points.map((point, index) => (
                <span
                  key={`${point.source}-${String(index)}`}
                  title={point.label}
                  className={`absolute size-2 rounded-full ${point.source.includes('medical') ? 'bg-rose-400' : 'bg-cyan-400'}`}
                  style={{
                    left: `${Math.max(4, Math.min(96, 50 + point.x * 140)).toFixed(2)}%`,
                    top: `${Math.max(4, Math.min(96, 50 - point.y * 140)).toFixed(2)}%`,
                  }}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
