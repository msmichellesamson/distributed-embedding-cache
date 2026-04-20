import asyncio
import logging
import signal
import sys
from concurrent import futures
from typing import AsyncIterator, Dict, List, Optional, Tuple

import grpc
import numpy as np
import structlog
from grpc import aio
from grpc_reflection.v1alpha import reflection
from grpc_status import rpc_status
from google.rpc import code_pb2, status_pb2
from prometheus_client import Counter, Histogram, start_http_server

from ..core.cache_engine import CacheEngine
from ..core.embedding_predictor import EmbeddingPredictor
from ..core.similarity_router import SimilarityRouter
from ..exceptions import (
    CacheEngineError,
    EmbeddingNotFoundError,
    PredictorError,
    RouterError
)
from ..proto import embedding_cache_pb2, embedding_cache_pb2_grpc

# Metrics
REQUEST_COUNT = Counter('grpc_requests_total', 'Total gRPC requests', ['method', 'status'])
REQUEST_DURATION = Histogram('grpc_request_duration_seconds', 'Request duration', ['method'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')
SIMILARITY_SEARCHES = Counter('similarity_searches_total', 'Similarity searches performed')

logger = structlog.get_logger(__name__)


class EmbeddingCacheServicer(embedding_cache_pb2_grpc.EmbeddingCacheServiceServicer):
    """High-performance gRPC service for distributed embedding cache operations."""
    
    def __init__(
        self,
        cache_engine: CacheEngine,
        predictor: EmbeddingPredictor,
        router: SimilarityRouter,
        max_batch_size: int = 1000,
        similarity_threshold: float = 0.85
    ) -> None:
        """Initialize the gRPC servicer.
        
        Args:
            cache_engine: Distributed cache engine instance
            predictor: Embedding predictor for precomputation
            router: Similarity router for semantic routing
            max_batch_size: Maximum batch size for operations
            similarity_threshold: Threshold for similarity matching
        """
        self.cache_engine = cache_engine
        self.predictor = predictor
        self.router = router
        self.max_batch_size = max_batch_size
        self.similarity_threshold = similarity_threshold
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "EmbeddingCacheServicer initialized",
            max_batch_size=max_batch_size,
            similarity_threshold=similarity_threshold
        )

    async def StoreEmbedding(
        self,
        request: embedding_cache_pb2.StoreEmbeddingRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.StoreEmbeddingResponse:
        """Store a single embedding in the cache."""
        method_name = "StoreEmbedding"
        start_time = asyncio.get_event_loop().time()
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                # Validate request
                if not request.key:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Embedding key cannot be empty"
                    )
                
                if len(request.embedding) == 0:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Embedding vector cannot be empty"
                    )
                
                # Convert to numpy array
                embedding = np.array(request.embedding, dtype=np.float32)
                
                # Store in cache with metadata
                metadata = {
                    "timestamp": request.metadata.timestamp,
                    "model_version": request.metadata.model_version,
                    "dimensions": len(embedding),
                    "ttl_seconds": request.ttl_seconds if request.ttl_seconds > 0 else None
                }
                
                success = await self.cache_engine.store_embedding(
                    key=request.key,
                    embedding=embedding,
                    metadata=metadata,
                    ttl_seconds=request.ttl_seconds if request.ttl_seconds > 0 else None
                )
                
                if not success:
                    await context.abort(
                        grpc.StatusCode.INTERNAL,
                        "Failed to store embedding in cache"
                    )
                
                # Update similarity router
                try:
                    await self.router.add_embedding(request.key, embedding)
                except RouterError as e:
                    logger.warning("Failed to update similarity router", error=str(e), key=request.key)
                
                # Trigger predictive precomputation
                asyncio.create_task(self._trigger_precomputation(request.key, embedding))
                
                REQUEST_COUNT.labels(method=method_name, status="success").inc()
                
                logger.info(
                    "Embedding stored successfully",
                    key=request.key,
                    dimensions=len(embedding),
                    duration=asyncio.get_event_loop().time() - start_time
                )
                
                return embedding_cache_pb2.StoreEmbeddingResponse(
                    success=True,
                    key=request.key
                )
                
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error storing embedding", error=str(e), key=request.key)
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def RetrieveEmbedding(
        self,
        request: embedding_cache_pb2.RetrieveEmbeddingRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.RetrieveEmbeddingResponse:
        """Retrieve a single embedding from the cache."""
        method_name = "RetrieveEmbedding"
        start_time = asyncio.get_event_loop().time()
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                if not request.key:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Embedding key cannot be empty"
                    )
                
                # Try cache first
                try:
                    embedding, metadata = await self.cache_engine.retrieve_embedding(request.key)
                    CACHE_HITS.inc()
                    
                    # Convert metadata
                    response_metadata = embedding_cache_pb2.EmbeddingMetadata(
                        timestamp=metadata.get("timestamp", 0),
                        model_version=metadata.get("model_version", ""),
                        dimensions=metadata.get("dimensions", len(embedding))
                    )
                    
                    REQUEST_COUNT.labels(method=method_name, status="success").inc()
                    
                    logger.info(
                        "Embedding retrieved from cache",
                        key=request.key,
                        dimensions=len(embedding),
                        duration=asyncio.get_event_loop().time() - start_time
                    )
                    
                    return embedding_cache_pb2.RetrieveEmbeddingResponse(
                        found=True,
                        embedding=embedding.tolist(),
                        metadata=response_metadata
                    )
                    
                except EmbeddingNotFoundError:
                    CACHE_MISSES.inc()
                    
                    # Try similarity-based retrieval if enabled
                    if request.use_similarity_fallback:
                        similar_embeddings = await self._find_similar_embeddings(
                            request.key, 
                            request.similarity_threshold or self.similarity_threshold
                        )
                        
                        if similar_embeddings:
                            # Return the most similar embedding
                            best_match = similar_embeddings[0]
                            embedding, metadata = await self.cache_engine.retrieve_embedding(best_match["key"])
                            
                            response_metadata = embedding_cache_pb2.EmbeddingMetadata(
                                timestamp=metadata.get("timestamp", 0),
                                model_version=metadata.get("model_version", ""),
                                dimensions=metadata.get("dimensions", len(embedding))
                            )
                            
                            REQUEST_COUNT.labels(method=method_name, status="similarity_match").inc()
                            
                            logger.info(
                                "Embedding retrieved via similarity match",
                                requested_key=request.key,
                                matched_key=best_match["key"],
                                similarity_score=best_match["score"],
                                duration=asyncio.get_event_loop().time() - start_time
                            )
                            
                            return embedding_cache_pb2.RetrieveEmbeddingResponse(
                                found=True,
                                embedding=embedding.tolist(),
                                metadata=response_metadata,
                                similarity_match=True,
                                matched_key=best_match["key"],
                                similarity_score=best_match["score"]
                            )
                    
                    REQUEST_COUNT.labels(method=method_name, status="not_found").inc()
                    
                    logger.info(
                        "Embedding not found",
                        key=request.key,
                        duration=asyncio.get_event_loop().time() - start_time
                    )
                    
                    return embedding_cache_pb2.RetrieveEmbeddingResponse(
                        found=False
                    )
                    
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error retrieving embedding", error=str(e), key=request.key)
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def BatchStore(
        self,
        request: embedding_cache_pb2.BatchStoreRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.BatchStoreResponse:
        """Store multiple embeddings in a single batch operation."""
        method_name = "BatchStore"
        start_time = asyncio.get_event_loop().time()
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                if len(request.items) == 0:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Batch cannot be empty"
                    )
                
                if len(request.items) > self.max_batch_size:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Batch size {len(request.items)} exceeds maximum {self.max_batch_size}"
                    )
                
                results = []
                successful_stores = 0
                
                # Process items in parallel batches
                batch_size = min(100, len(request.items))
                for i in range(0, len(request.items), batch_size):
                    batch = request.items[i:i + batch_size]
                    batch_results = await asyncio.gather(
                        *[self._store_single_item(item) for item in batch],
                        return_exceptions=True
                    )
                    
                    for j, result in enumerate(batch_results):
                        item = batch[j]
                        if isinstance(result, Exception):
                            results.append(embedding_cache_pb2.BatchStoreResult(
                                key=item.key,
                                success=False,
                                error=str(result)
                            ))
                        else:
                            results.append(embedding_cache_pb2.BatchStoreResult(
                                key=item.key,
                                success=True
                            ))
                            successful_stores += 1
                
                REQUEST_COUNT.labels(method=method_name, status="success").inc()
                
                logger.info(
                    "Batch store completed",
                    total_items=len(request.items),
                    successful_stores=successful_stores,
                    failed_stores=len(request.items) - successful_stores,
                    duration=asyncio.get_event_loop().time() - start_time
                )
                
                return embedding_cache_pb2.BatchStoreResponse(
                    results=results,
                    total_processed=len(request.items),
                    successful_stores=successful_stores
                )
                
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error in batch store", error=str(e), batch_size=len(request.items))
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def BatchRetrieve(
        self,
        request: embedding_cache_pb2.BatchRetrieveRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.BatchRetrieveResponse:
        """Retrieve multiple embeddings in a single batch operation."""
        method_name = "BatchRetrieve"
        start_time = asyncio.get_event_loop().time()
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                if len(request.keys) == 0:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Key list cannot be empty"
                    )
                
                if len(request.keys) > self.max_batch_size:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Batch size {len(request.keys)} exceeds maximum {self.max_batch_size}"
                    )
                
                results = []
                cache_hits = 0
                
                # Process keys in parallel batches
                batch_size = min(100, len(request.keys))
                for i in range(0, len(request.keys), batch_size):
                    batch = request.keys[i:i + batch_size]
                    batch_results = await asyncio.gather(
                        *[self._retrieve_single_item(key) for key in batch],
                        return_exceptions=True
                    )
                    
                    for j, result in enumerate(batch_results):
                        key = batch[j]
                        if isinstance(result, Exception):
                            results.append(embedding_cache_pb2.BatchRetrieveResult(
                                key=key,
                                found=False,
                                error=str(result)
                            ))
                        elif result is None:
                            results.append(embedding_cache_pb2.BatchRetrieveResult(
                                key=key,
                                found=False
                            ))
                        else:
                            embedding, metadata = result
                            response_metadata = embedding_cache_pb2.EmbeddingMetadata(
                                timestamp=metadata.get("timestamp", 0),
                                model_version=metadata.get("model_version", ""),
                                dimensions=metadata.get("dimensions", len(embedding))
                            )
                            
                            results.append(embedding_cache_pb2.BatchRetrieveResult(
                                key=key,
                                found=True,
                                embedding=embedding.tolist(),
                                metadata=response_metadata
                            ))
                            cache_hits += 1
                
                REQUEST_COUNT.labels(method=method_name, status="success").inc()
                
                logger.info(
                    "Batch retrieve completed",
                    total_keys=len(request.keys),
                    cache_hits=cache_hits,
                    cache_misses=len(request.keys) - cache_hits,
                    duration=asyncio.get_event_loop().time() - start_time
                )
                
                return embedding_cache_pb2.BatchRetrieveResponse(
                    results=results,
                    total_processed=len(request.keys),
                    cache_hits=cache_hits
                )
                
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error in batch retrieve", error=str(e), batch_size=len(request.keys))
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def FindSimilar(
        self,
        request: embedding_cache_pb2.FindSimilarRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.FindSimilarResponse:
        """Find embeddings similar to a given query vector."""
        method_name = "FindSimilar"
        start_time = asyncio.get_event_loop().time()
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                if len(request.query_embedding) == 0:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Query embedding cannot be empty"
                    )
                
                if request.top_k <= 0:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "top_k must be positive"
                    )
                
                if request.top_k > 1000:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "top_k cannot exceed 1000"
                    )
                
                query_embedding = np.array(request.query_embedding, dtype=np.float32)
                threshold = request.threshold if request.threshold > 0 else self.similarity_threshold
                
                # Find similar embeddings using the router
                similar_items = await self.router.find_similar(
                    query_embedding=query_embedding,
                    top_k=request.top_k,
                    threshold=threshold
                )
                
                results = []
                for item in similar_items:
                    # Retrieve full embedding data from cache
                    try:
                        embedding, metadata = await self.cache_engine.retrieve_embedding(item["key"])
                        
                        response_metadata = embedding_cache_pb2.EmbeddingMetadata(
                            timestamp=metadata.get("timestamp", 0),
                            model_version=metadata.get("model_version", ""),
                            dimensions=metadata.get("dimensions", len(embedding))
                        )
                        
                        results.append(embedding_cache_pb2.SimilarityResult(
                            key=item["key"],
                            similarity_score=item["score"],
                            embedding=embedding.tolist(),
                            metadata=response_metadata
                        ))
                    except EmbeddingNotFoundError:
                        # Embedding was removed from cache but still in router
                        logger.warning(
                            "Embedding found in router but not in cache",
                            key=item["key"]
                        )
                        continue
                
                SIMILARITY_SEARCHES.inc()
                REQUEST_COUNT.labels(method=method_name, status="success").inc()
                
                logger.info(
                    "Similarity search completed",
                    query_dimensions=len(query_embedding),
                    top_k=request.top_k,
                    threshold=threshold,
                    results_found=len(results),
                    duration=asyncio.get_event_loop().time() - start_time
                )
                
                return embedding_cache_pb2.FindSimilarResponse(
                    results=results,
                    total_results=len(results)
                )
                
        except RouterError as e:
            REQUEST_COUNT.labels(method=method_name, status="router_error").inc()
            logger.error("Router error in similarity search", error=str(e))
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Similarity search error: {str(e)}"
            )
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error in similarity search", error=str(e))
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def GetCacheStats(
        self,
        request: embedding_cache_pb2.GetCacheStatsRequest,
        context: grpc.aio.ServicerContext
    ) -> embedding_cache_pb2.GetCacheStatsResponse:
        """Get cache statistics and health information."""
        method_name = "GetCacheStats"
        
        try:
            with REQUEST_DURATION.labels(method=method_name).time():
                stats = await self.cache_engine.get_stats()
                router_stats = await self.router.get_stats()
                
                REQUEST_COUNT.labels(method=method_name, status="success").inc()
                
                return embedding_cache_pb2.GetCacheStatsResponse(
                    total_embeddings=stats.get("total_embeddings", 0),
                    memory_usage_bytes=stats.get("memory_usage_bytes", 0),
                    cache_hit_rate=stats.get("hit_rate", 0.0),
                    average_retrieval_time_ms=stats.get("avg_retrieval_time_ms", 0.0),
                    cluster_nodes=stats.get("cluster_nodes", 1),
                    router_index_size=router_stats.get("index_size", 0),
                    predictor_queue_size=await self._get_predictor_queue_size()
                )
                
        except Exception as e:
            REQUEST_COUNT.labels(method=method_name, status="error").inc()
            logger.error("Error getting cache stats", error=str(e))
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Internal error: {str(e)}"
            )

    async def _store_single_item(self, item: embedding_cache_pb2.StoreEmbeddingRequest) -> bool:
        """Store a single embedding item (helper for batch operations)."""
        embedding = np.array(item.embedding, dtype=np.float32)
        
        metadata = {
            "timestamp": item.metadata.timestamp,
            "model_version": item.metadata.model_version,
            "dimensions": len(embedding),
            "ttl_seconds": item.ttl_seconds if item.ttl_seconds > 0 else None
        }
        
        success = await self.cache_engine.store_embedding(
            key=item.key,
            embedding=embedding,
            metadata=metadata,
            ttl_seconds=item.ttl_seconds if item.ttl_seconds > 0 else None
        )
        
        if success:
            try:
                await self.router.add_embedding(item.key, embedding)
            except RouterError as e:
                logger.warning("Failed to update router in batch", error=str(e), key=item.key)
            
            asyncio.create_task(self._trigger_precomputation(item.key, embedding))
        
        return success

    async def _retrieve_single_item(self, key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Retrieve a single embedding item (helper for batch operations)."""
        try:
            return await self.cache_engine.retrieve_embedding(key)
        except EmbeddingNotFoundError:
            return None

    async def _find_similar_embeddings(self, key: str, threshold: float) -> List[Dict]:
        """Find similar embeddings for a given key."""
        try:
            # For now, we can't find similar to a key that doesn't exist
            # This would require the original embedding or a different approach
            return []
        except Exception:
            return []

    async def _trigger_precomputation(self, key: str, embedding: np.ndarray) -> None:
        """Trigger predictive precomputation for related embeddings."""
        try:
            await self.predictor.predict_and_precompute(key, embedding)
        except PredictorError as e:
            logger.warning("Precomputation failed", error=str(e), key=key)
        except Exception as e:
            logger.error("Unexpected error in precomputation", error=str(e), key=key)

    async def _get_predictor_queue_size(self) -> int:
        """Get the current predictor queue size."""
        try:
            return await self.predictor.get_queue_size()
        except Exception:
            return 0

    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down gRPC service")
        self._shutdown_event.set()


class GRPCServer:
    """High-performance gRPC server for distributed embedding cache."""
    
    def __init__(
        self,
        servicer: EmbeddingCacheServicer,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 100,
        max_message_length: int = 100 * 1024 * 1024  # 100MB
    ) -> None:
        """Initialize the gRPC server.
        
        Args:
            servicer: The gRPC servicer instance
            host: Host to bind to
            port: Port to bind to
            max_workers: Maximum number of worker threads
            max_message_length: Maximum message length in bytes
        """
        self.servicer = servicer
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_message_length = max_message_length
        self.server: Optional[grpc.aio.Server] = None
        
        # Start Prometheus metrics server
        start_http_server(8080)
        logger.info("Prometheus metrics server started on port 8080")

    async def start(self) -> None:
        """Start the gRPC server."""
        # Server options for performance
        options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.max_receive_message_length', self.max_message_length),
            ('grpc.max_send_message_length', self.max_message_length),
            ('grpc.max_connection_idle_ms', 300000),
        ]
        
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=options
        )
        
        # Add servicer
        embedding_cache_pb2_grpc.add_EmbeddingCacheServiceServicer_to_server(
            self.servicer, self.server
        )
        
        # Enable reflection for development
        SERVICE_NAMES = (
            embedding_cache_pb2.DESCRIPTOR.services_by_name['EmbeddingCacheService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, self.server)
        
        # Bind to address
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        
        logger.info(
            "Starting gRPC server",
            host=self.host,
            port=self.port,
            max_workers=self.max_workers,
            max_message_length=self.max_message_length
        )
        
        await self.server.start()
        logger.info(f"gRPC server started on {listen_addr}")

    async def stop(self, grace_period: Optional[float] = 30.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info("Stopping gRPC server", grace_period=grace_period)
            await self.servicer.shutdown()
            await self.server.stop(grace_period)
            logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()


async def create_server(
    cache_engine: CacheEngine,
    predictor: EmbeddingPredictor,
    router: SimilarityRouter,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 100
) -> GRPCServer:
    """Create and configure a gRPC server instance."""
    servicer = EmbeddingCacheServicer(
        cache_engine=cache_engine,
        predictor=predictor,
        router=router
    )
    
    server = GRPCServer(
        servicer=servicer,
        host=host,
        port=port,
        max_workers=max_workers
    )
    
    return server


def setup_signal_handlers(server: GRPCServer) -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(server.stop())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Main entry point for the gRPC server."""
    # This would normally be called from src/main.py
    # but included here for completeness
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Import and setup components (would be done in main.py)
    # This is just for reference
    pass


if __name__ == "__main__":
    asyncio.run(main())