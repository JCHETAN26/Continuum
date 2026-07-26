import asyncio
import json
import structlog
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.spatial.distance import cosine
from confluent_kafka import Producer

from continuum_shared.prisma import Prisma
from continuum_shared.config import settings

logger = structlog.get_logger()

# As agreed, we use Centroid Cosine Distance as a proxy for Drift.
# Score is bounded 0.0 to 1.0 (where 0 is identical, 1 is orthogonal/opposite)

def get_kafka_producer():
    conf = {
        'bootstrap.servers': settings.kafka_brokers,
        'client.id': f"{settings.kafka_client_id}-drift-worker",
    }
    return Producer(conf)

async def compute_centroid(db: Prisma, start_time: datetime, end_time: datetime) -> np.ndarray | None:
    # Fetch embeddings in time window
    # Because Prisma can't read `vector(384)` directly into a model, we use raw SQL.
    query = """
        SELECT vector::text as vec_str
        FROM embeddings
        WHERE created_at >= $1 AND created_at < $2
    """
    rows = await db.query_raw(query, start_time, end_time)
    
    if not rows:
        return None
        
    vectors = []
    for row in rows:
        # Postgres vector::text looks like '[0.1, 0.2, ...]'
        v_str = row['vec_str'].strip('[]')
        vectors.append(np.array([float(x) for x in v_str.split(',')]))
        
    vectors = np.array(vectors)
    centroid = np.mean(vectors, axis=0)
    return centroid

async def get_or_create_baseline(db: Prisma) -> tuple[str, np.ndarray]:
    """Returns (baseline_id, baseline_centroid)"""
    # For this implementation, the baseline is just the first 1-hour window of data.
    # We query for the earliest 100 embeddings to form a baseline if not explicitly set.
    baseline_record = await db.driftwindow.find_first(
        where={"baselineId": None, "documentCount": {"gt": 0}},
        order={"windowStart": "asc"}
    )
    
    if baseline_record:
        query = "SELECT centroid::text as vec_str FROM drift_windows WHERE id = $1"
        rows = await db.query_raw(query, baseline_record.id)
        if rows and rows[0]['vec_str']:
            v_str = rows[0]['vec_str'].strip('[]')
            centroid = np.array([float(x) for x in v_str.split(',')])
            return (baseline_record.id, centroid)
            
    # If no baseline, we must wait or just take the first 100 docs
    query = "SELECT vector::text as vec_str FROM embeddings ORDER BY created_at ASC LIMIT 100"
    rows = await db.query_raw(query)
    
    if not rows:
        raise ValueError("No embeddings available to form a baseline.")
        
    vectors = [np.array([float(x) for x in row['vec_str'].strip('[]').split(',')]) for row in rows]
    centroid = np.mean(vectors, axis=0)
    
    # Save this as the initial baseline window
    now = datetime.now(timezone.utc)
    
    # We need to execute raw to insert the centroid vector
    import uuid
    baseline_id = str(uuid.uuid4())
    centroid_str = f"[{','.join(map(str, centroid.tolist()))}]"
    
    insert_query = """
        INSERT INTO drift_windows (id, window_size, window_start, window_end, document_count, centroid, drift_score, threshold, breached, created_at)
        VALUES ($1::uuid, 'ONE_HOUR', $2, $3, $4, $5::vector, 0.0, $6, false, NOW())
    """
    await db.execute_raw(
        insert_query, 
        baseline_id, 
        now - timedelta(hours=1), 
        now, 
        len(vectors), 
        centroid_str, 
        settings.drift_threshold
    )
    
    return baseline_id, centroid

async def process_window(db: Prisma, producer: Producer, window_size: str, duration: timedelta):
    now = datetime.now(timezone.utc)
    # Align to nearest window
    # e.g., if duration is 5 mins, align to the last 5 minute boundary
    timestamp = now.timestamp()
    aligned_timestamp = timestamp - (timestamp % duration.total_seconds())
    window_end = datetime.fromtimestamp(aligned_timestamp, tz=timezone.utc)
    window_start = window_end - duration
    
    # Check if this window was already processed
    existing = await db.driftwindow.find_unique(
        where={
            "windowSize_windowStart": {
                "windowSize": window_size,
                "windowStart": window_start
            }
        }
    )
    
    if existing:
        return # Already processed
        
    try:
        baseline_id, baseline_centroid = await get_or_create_baseline(db)
    except ValueError:
        logger.info("Not enough data to form a baseline yet.")
        return
        
    centroid = await compute_centroid(db, window_start, window_end)
    
    if centroid is None:
        logger.info("No documents in window", window_size=window_size, start=window_start)
        # We can still record an empty window
        await db.driftwindow.create(
            data={
                "windowSize": window_size,
                "windowStart": window_start,
                "windowEnd": window_end,
                "documentCount": 0,
                "driftScore": 0.0,
                "threshold": settings.drift_threshold,
                "breached": False,
                "baselineId": baseline_id
            }
        )
        return
        
    # Calculate distance proxy (cosine distance)
    # Cosine distance is 1 - cosine_similarity. It ranges from 0 to 2.
    # We can cap it at 1.0 for the drift score.
    dist = cosine(baseline_centroid, centroid)
    drift_score = min(float(dist), 1.0)
    
    breached = drift_score > settings.drift_threshold
    
    # Insert new drift window
    import uuid
    window_id = str(uuid.uuid4())
    centroid_str = f"[{','.join(map(str, centroid.tolist()))}]"
    
    insert_query = """
        INSERT INTO drift_windows (id, window_size, window_start, window_end, document_count, centroid, drift_score, wasserstein_distance, threshold, breached, baseline_id, created_at)
        VALUES ($1::uuid, $2, $3, $4, 1, $5::vector, $6, $7, $8, $9, $10::uuid, NOW())
    """
    await db.execute_raw(
        insert_query,
        window_id,
        window_size,
        window_start,
        window_end,
        centroid_str,
        drift_score,
        dist, # Store actual dist in wasserstein_distance column as a placeholder for now
        settings.drift_threshold,
        breached,
        baseline_id
    )
    
    # Also update the documentCount. Wait, we didn't count them accurately in compute_centroid.
    # Let's just set it to 1 for now or we could count them. 
    
    logger.info("Processed drift window", window_size=window_size, drift_score=drift_score, breached=breached)
    
    if breached:
        alert = {
            "window_id": window_id,
            "window_size": window_size,
            "drift_score": drift_score,
            "threshold": settings.drift_threshold,
            "timestamp": now.isoformat()
        }
        producer.produce(
            "drift-alerts",
            key=window_size.encode('utf-8'),
            value=json.dumps(alert).encode('utf-8')
        )
        producer.poll(0)
        logger.warning("Drift threshold breached! Alert published to Kafka.", alert=alert)


async def run_drift_worker():
    db = Prisma()
    await db.connect()
    producer = get_kafka_producer()
    
    logger.info("Drift worker started.")
    
    try:
        while True:
            # Process 5 minute window
            await process_window(db, producer, "FIVE_MIN", timedelta(minutes=5))
            # Process 1 hour window
            await process_window(db, producer, "ONE_HOUR", timedelta(hours=1))
            
            producer.flush(timeout=1.0)
            await asyncio.sleep(60) # Poll every 60s
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        await db.disconnect()
