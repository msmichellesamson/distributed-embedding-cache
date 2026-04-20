import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import structlog
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

from ..exceptions import SimilarityRouterError, EmbeddingNotFoundError
from ..config import get_settings

logger = structlog.get_logger(__name__)

# Metrics
SIMILARITY_REQUESTS = Counter(
    'similarity_router_requests_total',
    'Total similarity routing requests',
    ['cache_status', 'similarity_method']
)

SIMILARITY_LATENCY = Histogram(
    'similarity_router_latency_seconds',
    'Similarity computation latency',
    ['similarity_method']
)

CACHE_HIT_RATIO = Gauge(
    'similarity_cache_hit_ratio',
    'Ratio of similarity cache hits'
)

EMBEDDING_DIMENSIONS = Gauge(
    'similarity_router_embedding_dimensions',
    'Dimensions of processed embeddings'
)


@dataclass
class SimilarityResult:
    """Result of similarity computation."""
    embedding_id: str
    similarity_score: float
    cache_key: str
    metadata: Dict[str, Any]
    computation_time_ms: float


@dataclass
class RouterConfig:
    """Configuration for similarity router."""
    similarity_threshold: float = 0.85
    max_candidates: int = 50
    use_approximate_search: bool = True
    precompute_batch_size: int = 100
    cache_ttl_seconds: int = 3600
    embedding_dimension: int = 768
    index_rebuild_interval: int = 300


class SimilarityIndex:
    """Efficient similarity index using approximate nearest neighbors."""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.index: Optional[NearestNeighbors] = None
        self.embedding_ids: List[str] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.last_rebuild: float = 0
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def rebuild_index(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Rebuild the similarity index with new embeddings."""
        if not embeddings:
            logger.warning("No embeddings provided for index rebuild")
            return
            
        async with self.lock:
            start_time = time.time()
            
            try:
                # Prepare data for index
                self.embedding_ids = list(embeddings.keys())
                self.embeddings_matrix = np.vstack(list(embeddings.values()))
                
                # Build index in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.index = await loop.run_in_executor(
                    self.executor,
                    self._build_index,
                    self.embeddings_matrix
                )
                
                self.last_rebuild = time.time()
                build_time = (self.last_rebuild - start_time) * 1000
                
                logger.info(
                    "Similarity index rebuilt",
                    embedding_count=len(self.embedding_ids),
                    build_time_ms=build_time,
                    dimensions=self.embeddings_matrix.shape[1]
                )
                
                EMBEDDING_DIMENSIONS.set(self.embeddings_matrix.shape[1])
                
            except Exception as e:
                logger.error(
                    "Failed to rebuild similarity index",
                    error=str(e),
                    embedding_count=len(embeddings)
                )
                raise SimilarityRouterError(f"Index rebuild failed: {e}")
    
    def _build_index(self, embeddings_matrix: np.ndarray) -> NearestNeighbors:
        """Build NearestNeighbors index (runs in thread pool)."""
        algorithm = 'ball_tree' if self.config.use_approximate_search else 'brute'
        metric = 'cosine'
        
        index = NearestNeighbors(
            n_neighbors=min(self.config.max_candidates, len(embeddings_matrix)),
            algorithm=algorithm,
            metric=metric,
            n_jobs=1  # Single job to avoid thread conflicts
        )
        
        index.fit(embeddings_matrix)
        return index
    
    async def find_similar(self, query_embedding: np.ndarray, k: int = None) -> List[Tuple[str, float]]:
        """Find k most similar embeddings."""
        if self.index is None or self.embeddings_matrix is None:
            raise SimilarityRouterError("Index not built")
            
        k = k or min(self.config.max_candidates, len(self.embedding_ids))
        
        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                distances, indices = await loop.run_in_executor(
                    self.executor,
                    self._query_index,
                    query_embedding.reshape(1, -1),
                    k
                )
                
                # Convert distances to similarities (cosine distance to similarity)
                similarities = 1 - distances[0]
                
                results = []
                for idx, similarity in zip(indices[0], similarities):
                    if similarity >= self.config.similarity_threshold:
                        embedding_id = self.embedding_ids[idx]
                        results.append((embedding_id, float(similarity)))
                
                return results
                
            except Exception as e:
                logger.error("Similarity search failed", error=str(e))
                raise SimilarityRouterError(f"Similarity search failed: {e}")
    
    def _query_index(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query the index (runs in thread pool)."""
        return self.index.kneighbors(query_embedding, n_neighbors=k)
    
    def needs_rebuild(self) -> bool:
        """Check if index needs rebuilding."""
        return (
            self.index is None or
            time.time() - self.last_rebuild > self.config.index_rebuild_interval
        )


class SimilarityRouter:
    """Semantic similarity router for embedding cache."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        config: Optional[RouterConfig] = None
    ):
        self.redis = redis_client
        self.config = config or RouterConfig()
        self.similarity_index = SimilarityIndex(self.config)
        
        # Metrics tracking
        self._total_requests = 0
        self._cache_hits = 0
        
        logger.info(
            "SimilarityRouter initialized",
            similarity_threshold=self.config.similarity_threshold,
            max_candidates=self.config.max_candidates,
            use_approximate_search=self.config.use_approximate_search
        )
    
    async def route_embedding(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[SimilarityResult]:
        """Route embedding to similar cached embeddings."""
        start_time = time.time()
        
        try:
            # Update metrics
            self._total_requests += 1
            
            # Check if index needs rebuilding
            if self.similarity_index.needs_rebuild():
                await self._rebuild_index_from_cache()
            
            # Find similar embeddings
            with SIMILARITY_LATENCY.labels('approximate' if self.config.use_approximate_search else 'exact').time():
                similar_embeddings = await self.similarity_index.find_similar(embedding)
            
            # Convert to results
            results = []
            computation_time = (time.time() - start_time) * 1000
            
            for embedding_id, similarity_score in similar_embeddings:
                cache_key = f"embedding:{embedding_id}"
                
                result = SimilarityResult(
                    embedding_id=embedding_id,
                    similarity_score=similarity_score,
                    cache_key=cache_key,
                    metadata=metadata or {},
                    computation_time_ms=computation_time
                )
                results.append(result)
            
            # Update cache hit metrics
            if results:
                self._cache_hits += 1
                SIMILARITY_REQUESTS.labels('hit', 'approximate' if self.config.use_approximate_search else 'exact').inc()
            else:
                SIMILARITY_REQUESTS.labels('miss', 'approximate' if self.config.use_approximate_search else 'exact').inc()
            
            CACHE_HIT_RATIO.set(self._cache_hits / self._total_requests)
            
            logger.info(
                "Similarity routing completed",
                similar_count=len(results),
                computation_time_ms=computation_time,
                best_similarity=results[0].similarity_score if results else 0.0
            )
            
            return results
            
        except Exception as e:
            SIMILARITY_REQUESTS.labels('error', 'unknown').inc()
            logger.error(
                "Similarity routing failed",
                error=str(e),
                embedding_shape=embedding.shape
            )
            raise SimilarityRouterError(f"Routing failed: {e}")
    
    async def add_embedding_to_index(
        self,
        embedding_id: str,
        embedding: np.ndarray,
        force_rebuild: bool = False
    ) -> None:
        """Add a new embedding to the similarity index."""
        try:
            # Store embedding in Redis for index rebuilding
            cache_key = f"index_embedding:{embedding_id}"
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            await self.redis.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                embedding_bytes
            )
            
            # Store metadata
            metadata_key = f"index_metadata:{embedding_id}"
            metadata = {
                'shape': embedding.shape,
                'dtype': str(embedding.dtype),
                'added_at': time.time()
            }
            
            await self.redis.hset(metadata_key, mapping=metadata)
            await self.redis.expire(metadata_key, self.config.cache_ttl_seconds)
            
            logger.debug(
                "Embedding added to index cache",
                embedding_id=embedding_id,
                shape=embedding.shape
            )
            
            # Trigger rebuild if forced or if enough time has passed
            if force_rebuild or self.similarity_index.needs_rebuild():
                await self._rebuild_index_from_cache()
                
        except Exception as e:
            logger.error(
                "Failed to add embedding to index",
                embedding_id=embedding_id,
                error=str(e)
            )
            raise SimilarityRouterError(f"Failed to add embedding: {e}")
    
    async def remove_embedding_from_index(self, embedding_id: str) -> None:
        """Remove embedding from the similarity index."""
        try:
            # Remove from Redis
            cache_key = f"index_embedding:{embedding_id}"
            metadata_key = f"index_metadata:{embedding_id}"
            
            await self.redis.delete(cache_key, metadata_key)
            
            logger.debug("Embedding removed from index cache", embedding_id=embedding_id)
            
            # Trigger index rebuild
            await self._rebuild_index_from_cache()
            
        except Exception as e:
            logger.error(
                "Failed to remove embedding from index",
                embedding_id=embedding_id,
                error=str(e)
            )
            raise SimilarityRouterError(f"Failed to remove embedding: {e}")
    
    async def _rebuild_index_from_cache(self) -> None:
        """Rebuild similarity index from cached embeddings."""
        try:
            # Get all embedding keys from Redis
            pattern = "index_embedding:*"
            keys = []
            
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key.decode())
            
            if not keys:
                logger.warning("No embeddings found in cache for index rebuild")
                return
            
            # Load embeddings
            embeddings = {}
            
            for key in keys:
                try:
                    # Get embedding data
                    embedding_bytes = await self.redis.get(key)
                    if not embedding_bytes:
                        continue
                    
                    # Get metadata
                    embedding_id = key.replace("index_embedding:", "")
                    metadata_key = f"index_metadata:{embedding_id}"
                    metadata = await self.redis.hgetall(metadata_key)
                    
                    if not metadata:
                        continue
                    
                    # Reconstruct embedding
                    shape_str = metadata.get(b'shape', b'').decode()
                    if shape_str:
                        shape = tuple(map(int, shape_str.strip('()').split(', ')))
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(shape)
                        embeddings[embedding_id] = embedding
                
                except Exception as e:
                    logger.warning(
                        "Failed to load embedding for index",
                        key=key,
                        error=str(e)
                    )
                    continue
            
            # Rebuild index
            await self.similarity_index.rebuild_index(embeddings)
            
        except Exception as e:
            logger.error("Failed to rebuild index from cache", error=str(e))
            raise SimilarityRouterError(f"Index rebuild failed: {e}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the similarity index."""
        try:
            stats = {
                'embedding_count': len(self.similarity_index.embedding_ids),
                'last_rebuild': self.similarity_index.last_rebuild,
                'needs_rebuild': self.similarity_index.needs_rebuild(),
                'total_requests': self._total_requests,
                'cache_hits': self._cache_hits,
                'hit_ratio': self._cache_hits / max(self._total_requests, 1),
                'config': {
                    'similarity_threshold': self.config.similarity_threshold,
                    'max_candidates': self.config.max_candidates,
                    'use_approximate_search': self.config.use_approximate_search,
                    'embedding_dimension': self.config.embedding_dimension
                }
            }
            
            if self.similarity_index.embeddings_matrix is not None:
                stats['embedding_dimensions'] = self.similarity_index.embeddings_matrix.shape[1]
                stats['index_size_bytes'] = self.similarity_index.embeddings_matrix.nbytes
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            raise SimilarityRouterError(f"Failed to get stats: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on similarity router."""
        try:
            # Check Redis connectivity
            await self.redis.ping()
            
            # Check index status
            index_healthy = (
                self.similarity_index.index is not None and
                len(self.similarity_index.embedding_ids) > 0
            )
            
            return {
                'status': 'healthy' if index_healthy else 'degraded',
                'redis_connected': True,
                'index_built': self.similarity_index.index is not None,
                'embedding_count': len(self.similarity_index.embedding_ids),
                'last_rebuild': self.similarity_index.last_rebuild,
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'error': str(e),
                'redis_connected': False,
                'index_built': False
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            self.similarity_index.executor.shutdown(wait=True)
            logger.info("SimilarityRouter closed successfully")
        except Exception as e:
            logger.error("Error closing SimilarityRouter", error=str(e))


async def create_similarity_router(redis_url: str, config: Optional[RouterConfig] = None) -> SimilarityRouter:
    """Factory function to create a similarity router."""
    try:
        redis_client = redis.from_url(redis_url, decode_responses=False)
        await redis_client.ping()
        
        router = SimilarityRouter(redis_client, config)
        router._start_time = time.time()
        
        logger.info("SimilarityRouter created successfully", redis_url=redis_url)
        return router
        
    except Exception as e:
        logger.error("Failed to create SimilarityRouter", error=str(e), redis_url=redis_url)
        raise SimilarityRouterError(f"Router creation failed: {e}")