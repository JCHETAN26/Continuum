# Continuum E2E Demo

This guide walks you through the local demo of Continuum detecting semantic drift and autonomously registering an adapted model candidate.

## Prerequisites

Ensure Docker is installed and running on your machine.

## 1. Start the Platform

Boot up the infrastructure, APIs, workers, serving engine, and dashboard:

```bash
docker compose up --build -d
```

Compose healthchecks coordinate startup, so app services wait for their required infrastructure and the dashboard waits for the drift and trainer APIs.

Confirm the host can reach every exposed service:

```bash
pnpm stack:health
```

## 2. Open the Mission Control Dashboard

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.
You will see the **Dashboard Overview**. The dashboard streams drift, trainer, and registry updates over SSE with REST fallbacks.

## 3. Inject Baseline Data

In a new terminal window, run the seed script to start streaming normal software engineering documents:

```bash
uv run scripts/seed.py
```

The default seed injects 1,000 software documents followed by 500 healthcare documents.
For a faster smoke run, lower the counts with `--baseline-docs`, `--drift-docs`, and `--delay`.
The script exits non-zero if ingestion fails.

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

1. You will see a new model version listed with a status of **ACTIVE** or **PASSED** and a positive retrieval-quality delta.
2. Use the **Activate** button to hot-swap a passed model manually; successful drift-triggered jobs also auto-activate the new model.

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
an active or passed model, and active-model retrieval-quality improvement over the baseline.

## Clean Up

```bash
docker compose down -v
```
