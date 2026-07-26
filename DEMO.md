# Continuum E2E Demo

This guide walks you through the 2-minute demo of Continuum detecting semantic drift and autonomously adapting the embedding model.

## Prerequisites
Ensure Docker is installed and running on your machine.

## 1. Start the Platform
Boot up the entire infrastructure and all 5 Continuum microservices:
```bash
docker compose up --build -d
```
*Wait ~30 seconds for Kafka and the services to initialize.*

## 2. Open the Mission Control Dashboard
Navigate to [http://localhost:3000](http://localhost:3000) in your browser.
You will see the **Dashboard Overview**. Notice the "Current Drift Score" is low, and the "Active Model" is the baseline `all-MiniLM-L6-v2`.

## 3. Inject Baseline Data
In a new terminal window, run the seed script to start streaming normal software engineering documents:
```bash
uv run scripts/seed.py
```
*Observe the dashboard.* The drift chart will remain stable since the documents match the baseline distribution.

## 4. Watch the Drift Spike (Automated in script)
After a few seconds, the script automatically injects a burst of out-of-distribution **Healthcare** documents.
1. Look at the Dashboard chart: The drift score will rapidly climb.
2. Once it crosses the `0.75` threshold, the status turns to **Breached**.
3. The **Active Training Jobs** counter will tick to `1`.

## 5. Monitor the Adaptation
Click on the **Training Monitor** tab on the left sidebar.
You will see the live training telemetry for the LoRA fine-tuning job. Watch the loss curve descend as the model adapts to the new healthcare vocabulary!

## 6. Verify and Activate
Click on the **Model Registry** tab.
1. You will see a new model version listed with a status of **PASSED** and a positive MRR Delta (e.g., `+12.4%`).
2. Click the **Activate** button to hot-swap it into the Serving Engine.

## 7. Prove the Improvement
Run the benchmark script to query the Serving API directly. It will evaluate the embeddings of the baseline vs. the new hot-swapped model:
```bash
uv run eval/benchmark.py
```
Notice the MRR (Mean Reciprocal Rank) jumps significantly for the healthcare queries!

## Clean Up
```bash
docker compose down -v
```
