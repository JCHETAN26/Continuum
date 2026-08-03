'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
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
const LINGUISTIC_API = process.env.NEXT_PUBLIC_LINGUISTIC_API_URL ?? 'http://localhost:8004';
const TRAINER_API = process.env.NEXT_PUBLIC_TRAINER_API_URL ?? 'http://localhost:8003';
const INGEST_API = process.env.NEXT_PUBLIC_INGEST_API_URL ?? 'http://localhost:8000';

interface DriftWindow {
  id: string;
  windowStart: string;
  documentCount: number;
  driftScore: number;
  threshold: number;
  breached: boolean;
}

interface DemoStatus {
  phase: string;
  ingested: number;
  total: number;
  error: string | null;
}

interface DriftSummary {
  documentCount: number;
  embeddingCount: number;
  latestDriftScore: number;
  breached: boolean;
  threshold: number;
}

interface LinguisticWindow {
  id: string;
  windowStart: string;
  documentCount: number;
  compositeScore: number;
  threshold: number;
  breached: boolean;
  newEntities: { text: string; label: string; count: number }[];
  emergingTopics: { label: string; share: number; top_terms?: string[] }[];
  emergingTerms: { term: string; baseline_count: number; window_count: number; score: number }[];
}

interface LinguisticSummary {
  latestCompositeScore: number;
  breached: boolean;
  threshold: number;
  windowCount: number;
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

interface LinguisticEventPayload {
  windows: LinguisticWindow[];
  summary: LinguisticSummary;
}

interface TrainingEventPayload {
  models: ModelVersion[];
}

export default function Home() {
  const [windows, setWindows] = useState<DriftWindow[]>([]);
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [linguisticWindows, setLinguisticWindows] = useState<LinguisticWindow[]>([]);
  const [linguisticSummary, setLinguisticSummary] = useState<LinguisticSummary | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [points, setPoints] = useState<ProjectionPoint[]>([]);
  const [demo, setDemo] = useState<DemoStatus | null>(null);

  // Phases the controller reports while a run is in flight. Anything else -- idle,
  // complete, failed, cancelled -- means the button is free again.
  const demoRunning =
    demo !== null &&
    ['starting', 'loading corpus', 'baseline', 'settling', 'drift'].includes(demo.phase);

  const startDemo = async () => {
    try {
      const response = await fetch(`${INGEST_API}/v1/demo/seed`, { method: 'POST' });
      if (response.ok) {
        setDemo((await response.json()) as DemoStatus);
      }
    } catch {
      // The ingest service is not up. The button simply does nothing rather than
      // throwing, which matches how the rest of this page treats a missing backend.
    }
  };

  useEffect(() => {
    if (!demoRunning) {
      return;
    }
    const poll = setInterval(() => {
      void (async () => {
        try {
          const response = await fetch(`${INGEST_API}/v1/demo/status`);
          if (response.ok) {
            setDemo((await response.json()) as DemoStatus);
          }
        } catch {
          // Transient while the stack settles; the next tick retries.
        }
      })();
    }, 1000);
    return () => {
      clearInterval(poll);
    };
  }, [demoRunning]);

  useEffect(() => {
    const refreshSnapshot = async () => {
      const [
        windowRes,
        summaryRes,
        linguisticWindowRes,
        linguisticSummaryRes,
        modelRes,
        projectionRes,
      ] = await Promise.allSettled([
        fetch(`${DRIFT_API}/v1/drift/status`),
        fetch(`${DRIFT_API}/v1/drift/summary`),
        fetch(`${LINGUISTIC_API}/v1/linguistic/status`),
        fetch(`${LINGUISTIC_API}/v1/linguistic/summary`),
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
      if (linguisticWindowRes.status === 'fulfilled' && linguisticWindowRes.value.ok) {
        const body = (await linguisticWindowRes.value.json()) as LinguisticWindow[];
        setLinguisticWindows(body.reverse());
      }
      if (linguisticSummaryRes.status === 'fulfilled' && linguisticSummaryRes.value.ok) {
        setLinguisticSummary((await linguisticSummaryRes.value.json()) as LinguisticSummary);
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
    const linguisticEvents = new EventSource(`${LINGUISTIC_API}/v1/linguistic/events`);
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

    linguisticEvents.addEventListener('linguistic', (event) => {
      const payload = JSON.parse(event.data as string) as LinguisticEventPayload;
      setLinguisticWindows(payload.windows.reverse());
      setLinguisticSummary(payload.summary);
    });

    return () => {
      driftEvents.close();
      linguisticEvents.close();
      trainingEvents.close();
    };
  }, []);

  const activeModel = models.find((model) => model.status === 'ACTIVE');
  const latestScore = summary?.latestDriftScore ?? windows.at(-1)?.driftScore ?? 0;
  const threshold = summary?.threshold ?? windows.at(-1)?.threshold ?? 0.35;
  const isBreached = summary?.breached ?? latestScore > threshold;
  const latestLinguisticScore =
    linguisticSummary?.latestCompositeScore ?? linguisticWindows.at(-1)?.compositeScore ?? 0;
  const linguisticThreshold =
    linguisticSummary?.threshold ?? linguisticWindows.at(-1)?.threshold ?? 0.65;
  const linguisticBreached =
    linguisticSummary?.breached ?? latestLinguisticScore > linguisticThreshold;
  const latestLinguisticWindow = linguisticWindows.at(-1);
  const chartData = windows.map((window) => ({
    time: format(new Date(window.windowStart), 'HH:mm:ss'),
    driftScore: window.driftScore,
    threshold: window.threshold,
  }));
  const linguisticChartData = linguisticWindows.map((window) => ({
    time: format(new Date(window.windowStart), 'HH:mm:ss'),
    compositeScore: window.compositeScore,
    threshold: window.threshold,
  }));

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Overview</h1>
          <p className="text-muted-foreground">
            Monitor real semantic drift from the local pipeline.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Button
            onClick={() => {
              void startDemo();
            }}
            disabled={demoRunning}
          >
            {demoRunning ? 'Running demo…' : 'Run demo'}
          </Button>
          {demo && demo.phase !== 'idle' && (
            <p className="text-xs text-muted-foreground">
              {demo.phase}
              {demo.total > 0 && ` · ${String(demo.ingested)}/${String(demo.total)} documents`}
            </p>
          )}
          {demo?.error && <p className="text-xs text-destructive">{demo.error}</p>}
        </div>
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
              Linguistic Drift
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-4xl font-bold tracking-tighter flex items-center gap-3">
              {latestLinguisticScore.toFixed(3)}
              <Badge variant={linguisticBreached ? 'destructive' : 'secondary'}>
                {linguisticBreached ? 'Breached' : 'Stable'}
              </Badge>
            </div>
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

      <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle>Linguistic Drift History</CardTitle>
            <CardDescription>Entity, topic, and vocabulary shift composite score.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={linguisticChartData}
                  margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                >
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
                    dataKey="compositeScore"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={3}
                    dot={false}
                    name="Composite Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>Emerging Language</CardTitle>
            <CardDescription>Newest entities and terms from the latest window.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Terms
              </div>
              <div className="flex flex-wrap gap-2">
                {(latestLinguisticWindow?.emergingTerms ?? []).slice(0, 8).map((term) => (
                  <Badge key={term.term} variant="outline">
                    {term.term} x{term.window_count}
                  </Badge>
                ))}
                {(latestLinguisticWindow?.emergingTerms ?? []).length === 0 ? (
                  <span className="text-sm text-muted-foreground">No term surge yet</span>
                ) : null}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Entities
              </div>
              <div className="flex flex-wrap gap-2">
                {(latestLinguisticWindow?.newEntities ?? []).slice(0, 8).map((entity) => (
                  <Badge key={`${entity.label}-${entity.text}`} variant="secondary">
                    {entity.text}
                  </Badge>
                ))}
                {(latestLinguisticWindow?.newEntities ?? []).length === 0 ? (
                  <span className="text-sm text-muted-foreground">No new entities yet</span>
                ) : null}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Topics
              </div>
              <div className="space-y-2">
                {(latestLinguisticWindow?.emergingTopics ?? []).slice(0, 5).map((topic) => (
                  <div
                    key={topic.label}
                    className="flex items-center justify-between gap-3 text-sm"
                  >
                    <span className="truncate">{topic.label}</span>
                    <span className="font-mono text-muted-foreground">
                      {(topic.share * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
                {(latestLinguisticWindow?.emergingTopics ?? []).length === 0 ? (
                  <span className="text-sm text-muted-foreground">No topic surge yet</span>
                ) : null}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
