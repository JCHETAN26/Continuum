import grpc
import structlog

from continuum_server.engine import engine
from continuum_server.grpc_gen import embed_pb2, embed_pb2_grpc

logger = structlog.get_logger()


class EmbedServiceServicer(embed_pb2_grpc.EmbedServiceServicer):
    async def EmbedBatch(  # noqa: N802
        self, request, context: grpc.aio.ServicerContext
    ) -> embed_pb2.EmbedBatchResponse:
        texts = list(request.texts)
        if len(texts) > 32:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Batch size exceeds maximum of 32")

        try:
            embeddings_list, version, dim = await engine.embed_batch(
                texts, model_version=request.model_version or "auto"
            )
        except RuntimeError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))

        response = embed_pb2.EmbedBatchResponse()
        response.model_version_used = version
        response.dimension = dim

        for vec in embeddings_list:
            emb = response.embeddings.add()
            emb.vector.extend(vec)

        return response


async def serve_grpc():
    server = grpc.aio.server()
    embed_pb2_grpc.add_EmbedServiceServicer_to_server(EmbedServiceServicer(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("Starting gRPC server", addr=listen_addr)
    await server.start()
    await server.wait_for_termination()
