import onnxruntime as ort
from tokenizers import Tokenizer
import structlog
import asyncio
from typing import List, Tuple
from continuum_shared.prisma import Prisma
from continuum_shared.prisma.enums import ModelStatus

logger = structlog.get_logger()

class ModelEngine:
    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None
        self.current_version: str | None = None
        self.dimension: int = 384
        self._lock = asyncio.Lock()
        self.db = Prisma()
        
    async def connect(self):
        await self.db.connect()
        # Initial load
        await self.poll_active_model()
        
    async def disconnect(self):
        await self.db.disconnect()
        
    async def poll_active_model(self):
        """Poll the database for the active model and hot-swap if changed."""
        active_model = await self.db.modelversion.find_first(
            where={"status": ModelStatus.ACTIVE}
        )
        
        if not active_model:
            logger.info("No ACTIVE model found in registry.")
            return
            
        if active_model.version == self.current_version:
            return # Unchanged
            
        logger.info("New ACTIVE model detected, initiating hot-swap", version=active_model.version)
        
        try:
            # In a real system, we'd download the ONNX artifact from S3/MinIO here
            # artifact_uri = active_model.artifactUri
            # Download to local temp path
            # ...
            
            # For this MVP, we simulate by loading a dummy session or if we don't have one,
            # we just track the version so requests can succeed in test mode.
            # Real onnxruntime loading:
            # new_session = ort.InferenceSession("local_model.onnx", providers=['CPUExecutionProvider'])
            # new_tokenizer = Tokenizer.from_file("tokenizer.json")
            
            async with self._lock:
                self.current_version = active_model.version
                # self.session = new_session
                # self.tokenizer = new_tokenizer
                logger.info("Hot-swap complete", version=self.current_version)
        except Exception as e:
            logger.error("Failed to load new model", error=str(e), version=active_model.version)

    async def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], str, int]:
        """Returns (embeddings, model_version_used, dimension)"""
        async with self._lock:
            version = self.current_version
            # session = self.session
            
        if not version:
            raise RuntimeError("No active model is loaded.")
            
        # Real ONNX inference:
        # encoded = self.tokenizer.encode_batch(texts)
        # inputs = {
        #     "input_ids": [e.ids for e in encoded],
        #     "attention_mask": [e.attention_mask for e in encoded],
        # }
        # outputs = session.run(None, inputs)
        # embeddings = outputs[0].tolist()
        
        # Mock logic
        embeddings = [[0.1] * self.dimension for _ in texts]
        
        return embeddings, version, self.dimension

engine = ModelEngine()

async def background_poller():
    while True:
        await asyncio.sleep(10)
        try:
            await engine.poll_active_model()
        except Exception as e:
            logger.error("Error polling active model", error=str(e))
