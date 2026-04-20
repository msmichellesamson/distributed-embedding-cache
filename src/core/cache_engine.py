import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import structlog
import numpy as np
from redis.asyncio import Redis, RedisCluster
from redis.exceptions import RedisError, ConnectionError
import torch
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()

class CacheEngineError(Exception):
    """Base exception for cache engine operations"""
    pass

class NodeUnavailableError(CacheEngineError):
    """Raised when a cache node becomes unavailable"""
    pass

class EmbeddingNotFoundError(CacheEngineError):
    """Raised when requested embedding is not in cache"""
    pass

class InvalidEmbeddingError(CacheEngineError):
    """Raised when embedding data is invalid"""
    pass

@dataclass
class EmbeddingMetadata:
    """Metadata for cached embeddings"""
    embedding_id: str
    model_name: str
    dimension: int
    created_at: float
    last_accessed: float
    access_count: int
    similarity_cluster: Optional[str] = None
    precomputed_neighbors: Optional[List[str]] = None

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    precompute_hits: int
    avg_response_time: float
    node_health: Dict[str, bool]
    memory_usage: Dict[str, float]

class SemanticRouter:
    """Routes embeddings to optimal cache nodes based on semantic similarity"""
    
    def __init__(self, num_clusters: int = 16):
        self.num_clusters = num_clusters
        self.cluster_centroids: Optional[np.ndarray] = None
        self.embedding_clusters: Dict[str, int] = {}
        self._logger = logger.bind(component="semantic_router")
    
    async def initialize_clusters(self, sample_embeddings: List[Tuple[str, np.ndarray]]) -> None:
        """Initialize semantic clusters from sample embeddings"""
        try:
            if len(sample_embeddings) < self.num_clusters:
                self._logger.warning("insufficient_samples", 
                                   samples=len(sample_embeddings), 
                                   clusters=self.num_clusters)
                return
            
            embeddings_matrix = np.array([emb for _, emb in sample_embeddings])
            
            # K-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            self.cluster_centroids = kmeans.cluster_centers_
            
            # Map embeddings to clusters
            for i, (emb_id, _) in enumerate(sample_embeddings):
                self.embedding_clusters[emb_id] = cluster_labels[i]
            
            self._logger.info("clusters_initialized", clusters=self.num_clusters)
            
        except Exception as e:
            self._logger.error("cluster_initialization_failed", error=str(e))
            raise CacheEngineError(f"Failed to initialize clusters: {e}")
    
    def route_embedding(self, embedding: np.ndarray) -> int:
        """Route embedding to appropriate cluster/node"""
        if self.cluster_centroids is None:
            return hash(embedding.tobytes()) % self.num_clusters
        
        try:
            similarities = cosine_similarity([embedding], self.cluster_centroids)[0]
            return int(np.argmax(similarities))
        except Exception as e:
            self._logger.error("routing_failed", error=str(e))
            return hash(embedding.tobytes()) % self.num_clusters

class PredictivePrecomputer:
    """Handles predictive precomputation of similar embeddings"""
    
    def __init__(self, similarity_threshold: float = 0.8, max_neighbors: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self._logger = logger.bind(component="precomputer")
    
    async def compute_similarities(self, 
                                 target_embedding: np.ndarray,
                                 candidate_embeddings: Dict[str, np.ndarray]) -> List[str]:
        """Compute and return IDs of similar embeddings"""
        try:
            if not candidate_embeddings:
                return []
            
            similarities = {}
            target_norm = np.linalg.norm(target_embedding)
            
            for emb_id, embedding in candidate_embeddings.items():
                try:
                    similarity = np.dot(target_embedding, embedding) / (
                        target_norm * np.linalg.norm(embedding)
                    )
                    if similarity >= self.similarity_threshold:
                        similarities[emb_id] = similarity
                except (ValueError, ZeroDivisionError):
                    continue
            
            # Sort by similarity and return top neighbors
            sorted_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return [emb_id for emb_id, _ in sorted_neighbors[:self.max_neighbors]]
            
        except Exception as e:
            self._logger.error("similarity_computation_failed", error=str(e))
            return []
    
    async def should_precompute(self, metadata: EmbeddingMetadata) -> bool:
        """Decide if embedding should trigger precomputation"""
        # Precompute for frequently accessed embeddings
        return (
            metadata.access_count >= 5 or
            (time.time() - metadata.last_accessed) < 300  # Last 5 minutes
        )

class DistributedCacheEngine:
    """Main distributed cache engine coordinating all operations"""
    
    def __init__(self, 
                 redis_nodes: List[str],
                 embedding_dimension: int = 768,
                 ttl_seconds: int = 3600,
                 max_cache_size_mb: int = 1024):
        self.redis_nodes = redis_nodes
        self.embedding_dimension = embedding_dimension
        self.ttl_seconds = ttl_seconds
        self.max_cache_size_mb = max_cache_size_mb
        
        self.redis_clients: List[Redis] = []
        self.router = SemanticRouter()
        self.precomputer = PredictivePrecomputer()
        self._stats = CacheStats(0, 0, 0, 0, 0.0, {}, {})
        self._request_times: List[float] = []
        self._logger = logger.bind(component="cache_engine")
    
    async def initialize(self) -> None:
        """Initialize cache engine and all components"""
        try:
            # Initialize Redis connections
            for node in self.redis_nodes:
                host, port = node.split(':')
                client = Redis(host=host, port=int(port), decode_responses=False)
                await client.ping()
                self.redis_clients.append(client)
                self._stats.node_health[node] = True
            
            self._logger.info("cache_engine_initialized", 
                            nodes=len(self.redis_clients),
                            dimension=self.embedding_dimension)
            
            # Initialize semantic routing with sample data if available
            await self._initialize_routing()
            
        except Exception as e:
            self._logger.error("initialization_failed", error=str(e))
            raise CacheEngineError(f"Failed to initialize cache engine: {e}")
    
    async def _initialize_routing(self) -> None:
        """Initialize semantic routing from existing cache data"""
        try:
            sample_embeddings = []
            for client in self.redis_clients:
                try:
                    keys = await client.keys("emb:*")
                    for key in keys[:100]:  # Sample first 100 embeddings per node
                        data = await client.hgetall(key)
                        if data and b'embedding' in data:
                            embedding = np.frombuffer(data[b'embedding'], dtype=np.float32)
                            if len(embedding) == self.embedding_dimension:
                                emb_id = key.decode().split(':')[1]
                                sample_embeddings.append((emb_id, embedding))
                except Exception as e:
                    self._logger.warning("sample_collection_failed", error=str(e))
                    continue
            
            if sample_embeddings:
                await self.router.initialize_clusters(sample_embeddings)
            
        except Exception as e:
            self._logger.warning("routing_initialization_failed", error=str(e))
    
    def _get_embedding_key(self, embedding_id: str) -> str:
        """Generate Redis key for embedding"""
        return f"emb:{embedding_id}"
    
    def _get_metadata_key(self, embedding_id: str) -> str:
        """Generate Redis key for embedding metadata"""
        return f"meta:{embedding_id}"
    
    def _select_node(self, embedding_id: str, embedding: Optional[np.ndarray] = None) -> Redis:
        """Select optimal Redis node for embedding"""
        if embedding is not None:
            cluster_id = self.router.route_embedding(embedding)
        else:
            # Fallback to hash-based routing
            cluster_id = hash(embedding_id) % len(self.redis_clients)
        
        node_index = cluster_id % len(self.redis_clients)
        return self.redis_clients[node_index]
    
    async def store_embedding(self, 
                            embedding_id: str, 
                            embedding: np.ndarray, 
                            model_name: str,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store embedding in distributed cache with metadata"""
        start_time = time.time()
        
        try:
            if embedding.shape[0] != self.embedding_dimension:
                raise InvalidEmbeddingError(
                    f"Expected dimension {self.embedding_dimension}, got {embedding.shape[0]}"
                )
            
            # Select optimal node
            client = self._select_node(embedding_id, embedding)
            
            # Prepare data
            embedding_bytes = embedding.astype(np.float32).tobytes()
            current_time = time.time()
            
            embedding_metadata = EmbeddingMetadata(
                embedding_id=embedding_id,
                model_name=model_name,
                dimension=self.embedding_dimension,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1
            )
            
            if metadata:
                embedding_metadata.__dict__.update(metadata)
            
            # Store embedding and metadata atomically
            pipe = client.pipeline()
            pipe.hset(
                self._get_embedding_key(embedding_id),
                mapping={
                    'embedding': embedding_bytes,
                    'model_name': model_name.encode(),
                    'dimension': str(self.embedding_dimension).encode(),
                    'stored_at': str(current_time).encode()
                }
            )
            pipe.expire(self._get_embedding_key(embedding_id), self.ttl_seconds)
            pipe.hset(
                self._get_metadata_key(embedding_id),
                mapping={k: json.dumps(v).encode() for k, v in asdict(embedding_metadata).items()}
            )
            pipe.expire(self._get_metadata_key(embedding_id), self.ttl_seconds)
            
            await pipe.execute()
            
            # Trigger predictive precomputation if needed
            if await self.precomputer.should_precompute(embedding_metadata):
                asyncio.create_task(self._precompute_neighbors(embedding_id, embedding))
            
            response_time = time.time() - start_time
            self._update_stats(response_time, hit=False)
            
            self._logger.info("embedding_stored", 
                            embedding_id=embedding_id,
                            dimension=self.embedding_dimension,
                            response_time=response_time)
            return True
            
        except Exception as e:
            self._logger.error("store_failed", 
                             embedding_id=embedding_id,
                             error=str(e))
            raise CacheEngineError(f"Failed to store embedding: {e}")
    
    async def get_embedding(self, embedding_id: str) -> Tuple[np.ndarray, EmbeddingMetadata]:
        """Retrieve embedding and metadata from cache"""
        start_time = time.time()
        
        try:
            # Try all nodes (for fault tolerance)
            for client in self.redis_clients:
                try:
                    # Get embedding and metadata in parallel
                    embedding_future = client.hgetall(self._get_embedding_key(embedding_id))
                    metadata_future = client.hgetall(self._get_metadata_key(embedding_id))
                    
                    embedding_data, metadata_data = await asyncio.gather(
                        embedding_future, metadata_future, return_exceptions=True
                    )
                    
                    if isinstance(embedding_data, Exception) or isinstance(metadata_data, Exception):
                        continue
                    
                    if not embedding_data or b'embedding' not in embedding_data:
                        continue
                    
                    # Parse embedding
                    embedding = np.frombuffer(embedding_data[b'embedding'], dtype=np.float32)
                    
                    # Parse metadata
                    metadata = EmbeddingMetadata(**{
                        k: json.loads(v.decode()) for k, v in metadata_data.items()
                    }) if metadata_data else EmbeddingMetadata(
                        embedding_id=embedding_id,
                        model_name=embedding_data.get(b'model_name', b'unknown').decode(),
                        dimension=len(embedding),
                        created_at=float(embedding_data.get(b'stored_at', time.time())),
                        last_accessed=time.time(),
                        access_count=1
                    )
                    
                    # Update access metadata
                    metadata.last_accessed = time.time()
                    metadata.access_count += 1
                    
                    # Update metadata in background
                    asyncio.create_task(self._update_metadata(client, embedding_id, metadata))
                    
                    response_time = time.time() - start_time
                    self._update_stats(response_time, hit=True)
                    
                    self._logger.info("embedding_retrieved", 
                                    embedding_id=embedding_id,
                                    response_time=response_time)
                    
                    return embedding, metadata
                    
                except (RedisError, ConnectionError):
                    continue
            
            # Not found in any node
            response_time = time.time() - start_time
            self._update_stats(response_time, hit=False)
            raise EmbeddingNotFoundError(f"Embedding {embedding_id} not found in cache")
            
        except EmbeddingNotFoundError:
            raise
        except Exception as e:
            self._logger.error("retrieval_failed", 
                             embedding_id=embedding_id,
                             error=str(e))
            raise CacheEngineError(f"Failed to retrieve embedding: {e}")
    
    async def find_similar_embeddings(self, 
                                    query_embedding: np.ndarray, 
                                    limit: int = 10,
                                    threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar embeddings across all cache nodes"""
        start_time = time.time()
        
        try:
            all_similarities = []
            
            # Search across all nodes
            for client in self.redis_clients:
                try:
                    keys = await client.keys("emb:*")
                    
                    for key in keys:
                        embedding_data = await client.hget(key, 'embedding')
                        if not embedding_data:
                            continue
                        
                        cached_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                        
                        # Calculate cosine similarity
                        similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                        
                        if similarity >= threshold:
                            embedding_id = key.decode().split(':')[1]
                            all_similarities.append((embedding_id, float(similarity)))
                
                except Exception as e:
                    self._logger.warning("node_search_failed", error=str(e))
                    continue
            
            # Sort by similarity and return top results
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            
            response_time = time.time() - start_time
            self._logger.info("similarity_search_completed",
                            query_dimension=len(query_embedding),
                            results_found=len(all_similarities),
                            response_time=response_time)
            
            return all_similarities[:limit]
            
        except Exception as e:
            self._logger.error("similarity_search_failed", error=str(e))
            raise CacheEngineError(f"Failed to find similar embeddings: {e}")
    
    async def _precompute_neighbors(self, embedding_id: str, embedding: np.ndarray) -> None:
        """Precompute and cache similar embeddings"""
        try:
            # Get sample of embeddings for similarity computation
            candidate_embeddings = {}
            
            for client in self.redis_clients:
                try:
                    keys = await client.keys("emb:*")
                    for key in keys[:200]:  # Limit to prevent excessive computation
                        if key.decode().endswith(embedding_id):
                            continue  # Skip self
                        
                        embedding_data = await client.hget(key, 'embedding')
                        if embedding_data:
                            candidate_id = key.decode().split(':')[1]
                            candidate_emb = np.frombuffer(embedding_data, dtype=np.float32)
                            candidate_embeddings[candidate_id] = candidate_emb
                
                except Exception:
                    continue
            
            # Compute similarities
            similar_ids = await self.precomputer.compute_similarities(
                embedding, candidate_embeddings
            )
            
            if similar_ids:
                # Update metadata with precomputed neighbors
                client = self._select_node(embedding_id, embedding)
                metadata_key = self._get_metadata_key(embedding_id)
                
                await client.hset(
                    metadata_key,
                    'precomputed_neighbors',
                    json.dumps(similar_ids).encode()
                )
                
                self._logger.info("neighbors_precomputed",
                                embedding_id=embedding_id,
                                neighbors_count=len(similar_ids))
        
        except Exception as e:
            self._logger.error("precomputation_failed",
                             embedding_id=embedding_id,
                             error=str(e))
    
    async def _update_metadata(self, client: Redis, embedding_id: str, metadata: EmbeddingMetadata) -> None:
        """Update embedding metadata"""
        try:
            await client.hset(
                self._get_metadata_key(embedding_id),
                mapping={k: json.dumps(v).encode() for k, v in asdict(metadata).items()}
            )
        except Exception as e:
            self._logger.warning("metadata_update_failed", 
                               embedding_id=embedding_id,
                               error=str(e))
    
    def _update_stats(self, response_time: float, hit: bool) -> None:
        """Update cache performance statistics"""
        self._stats.total_requests += 1
        if hit:
            self._stats.cache_hits += 1
        else:
            self._stats.cache_misses += 1
        
        self._request_times.append(response_time)
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-1000:]
        
        self._stats.avg_response_time = sum(self._request_times) / len(self._request_times)
    
    async def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        try:
            # Update node health
            for i, client in enumerate(self.redis_clients):
                node = self.redis_nodes[i]
                try:
                    await client.ping()
                    self._stats.node_health[node] = True
                    
                    # Get memory usage
                    info = await client.info('memory')
                    used_memory_mb = info['used_memory'] / (1024 * 1024)
                    self._stats.memory_usage[node] = used_memory_mb
                    
                except Exception:
                    self._stats.node_health[node] = False
                    self._stats.memory_usage[node] = 0.0
            
            return self._stats
            
        except Exception as e:
            self._logger.error("stats_collection_failed", error=str(e))
            return self._stats
    
    async def cleanup_expired_embeddings(self) -> int:
        """Clean up expired embeddings and return count cleaned"""
        cleaned_count = 0
        
        try:
            current_time = time.time()
            
            for client in self.redis_clients:
                try:
                    # Get all metadata keys
                    metadata_keys = await client.keys("meta:*")
                    
                    for key in metadata_keys:
                        metadata_data = await client.hgetall(key)
                        if not metadata_data:
                            continue
                        
                        # Check if embedding is expired
                        created_at = json.loads(metadata_data[b'created_at'].decode())
                        if current_time - created_at > self.ttl_seconds:
                            embedding_id = key.decode().split(':')[1]
                            
                            # Delete embedding and metadata
                            pipe = client.pipeline()
                            pipe.delete(self._get_embedding_key(embedding_id))
                            pipe.delete(self._get_metadata_key(embedding_id))
                            await pipe.execute()
                            
                            cleaned_count += 1
                
                except Exception as e:
                    self._logger.warning("cleanup_failed_for_node", error=str(e))
                    continue
            
            self._logger.info("cleanup_completed", cleaned_count=cleaned_count)
            return cleaned_count
            
        except Exception as e:
            self._logger.error("cleanup_failed", error=str(e))
            return cleaned_count
    
    async def shutdown(self) -> None:
        """Shutdown cache engine and close all connections"""
        try:
            for client in self.redis_clients:
                await client.close()
            
            self._logger.info("cache_engine_shutdown")
            
        except Exception as e:
            self._logger.error("shutdown_failed", error=str(e))