import json
import random

from locust import HttpUser, between, task


class EmbeddingUser(HttpUser):
    wait_time = between(0.05, 0.2)

    # We simulate an array of strings
    vocab = [
        "hello",
        "world",
        "this",
        "is",
        "a",
        "load",
        "test",
        "for",
        "continuum",
        "embeddings",
        "performance",
    ]

    @task
    def embed_batch(self):
        # Generate batch of 1 to 32 items
        batch_size = random.randint(1, 32)
        texts = [" ".join(random.choices(self.vocab, k=5)) for _ in range(batch_size)]

        headers = {"Content-Type": "application/json", "x-api-key": "continuum-secret-key"}

        payload = {"texts": texts}

        with self.client.post(
            "/v1/embed", data=json.dumps(payload), headers=headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}: {response.text}")
