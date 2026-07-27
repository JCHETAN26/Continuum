# Continuum E2E Demo

This guide walks you through the local demo of Continuum detecting semantic drift and autonomously registering an adapted model candidate.

## Prerequisites

Ensure Docker is installed and running on your machine.

## 1. Start the Platform

Boot up the entire infrastructure and all 5 Continuum microservices:

```bash
docker compose up --build -d
```

_Wait ~30 seconds for Kafka and the services to initialize._

## 2. Open the Mission Control Dashboard

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.
You will see the **Dashboard Overview**. The dashboard streams drift, trainer, and registry updates over SSE with REST fallbacks.

## 3. Inject Baseline Data

In a new terminal window, run the seed script to start streaming normal software engineering documents:

```bash
uv run scripts/seed.py
```

_Observe the dashboard._ The drift chart will remain stable since the documents match the baseline distribution.

## 4. Watch the Drift Spike (Automated in script)

After a few seconds, the script automatically injects a burst of out-of-distribution **Healthcare** documents.

1. Look at the Dashboard chart: The drift score will rapidly climb.
2. Once it crosses the configured threshold, the status turns to **Breached**.
3. A drift-triggered training job appears in the trainer API.

## 5. Monitor the Adaptation

Click on the **Training Monitor** tab on the left sidebar.
You will see telemetry for the deterministic demo adaptation job. Watch the recorded loss curve descend as the job completes.

## 6. Verify and Activate

Click on the **Model Registry** tab.

1. You will see a new model version listed with a status of **PASSED** and a positive MRR delta.
2. Click the **Activate** button to hot-swap it into the Serving Engine.

## 7. Prove the Improvement

Run the benchmark script to query the Serving API directly:

```bash
uv run eval/benchmark.py
```

The benchmark reports retrieval quality for the active local embedding service.

To check the entire narrative in one command, run:

```bash
pnpm demo:verify
```

The verifier waits for document/embedding counts, drift breach, a completed training job,
an active or passed model, and an active-model MRR lift over the baseline.

## Clean Up

```bash
docker compose down -v
```
