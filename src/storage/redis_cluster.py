import asyncio
import json
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

import structlog
import redis.asyncio as redis
from redis.asyncio import RedisCluster
from redis.exceptions import RedisClusterException, ConnectionError, TimeoutError
import numpy as np

from ..core.exceptions import CacheError, ClusterError, SerializationError


logger = structlog.get_logger(__name__)


@dataclass
class ClusterNode:
    """Redis cluster node information."""
    host: str
    port: int
    node_id: str
    slots: List[Tuple[int, int]]
    is_master: bool
    is_replica: bool


@dataclass
class EmbeddingEntry:
    """Cached embedding entry with metadata."""
    embedding: np.ndarray
    model_version: str
    timestamp: float
    access_count: int
    similarity_threshold: float


class EmbeddingSerializer:
    """Handles serialization/deserialization of embeddings with compression."""
    
    @staticmethod
    def serialize(entry: EmbeddingEntry) -> bytes:
        """Serialize embedding entry to bytes."""
        try:
            data = {
                'embedding': entry.embedding.tobytes(),
                'shape': entry.embedding.shape,
                'dtype': str(entry.embedding.dtype),
                'model_version': entry.model_version,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'similarity_threshold': entry.similarity_threshold
            }
            return json.dumps(data).encode('utf-8')
        except Exception as e:
            raise SerializationError(f"Failed to serialize embedding: {e}")
    
    @staticmethod
    def deserialize(data: bytes) -> EmbeddingEntry:
        """Deserialize bytes to embedding entry."""
        try:
            parsed = json.loads(data.decode('utf-8'))
            embedding = np.frombuffer(
                parsed['embedding'], 
                dtype=np.dtype(parsed['dtype'])
            ).reshape(parsed['shape'])
            
            return EmbeddingEntry(
                embedding=embedding,
                model_version=parsed['model_version'],
                timestamp=parsed['timestamp'],
                access_count=parsed['access_count'],
                similarity_threshold=parsed['similarity_threshold']
            )
        except Exception as e:
            raise SerializationError(f"Failed to deserialize embedding: {e}")


class RedisClusterManager:
    """Manages Redis cluster operations for embedding cache."""
    
    def __init__(
        self,
        startup_nodes: List[Dict[str, Union[str, int]]],
        password: Optional[str] = None,
        decode_responses: bool = False,
        skip_full_coverage_check: bool = False,
        max_connections_per_node: int = 50,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 30.0,
        health_check_interval: int = 30
    ) -> None:
        """Initialize Redis cluster manager.
        
        Args:
            startup_nodes: List of initial cluster nodes
            password: Redis auth password
            decode_responses: Whether to decode responses
            skip_full_coverage_check: Skip cluster coverage validation
            max_connections_per_node: Max connections per node
            socket_timeout: Socket operation timeout
            socket_connect_timeout: Socket connection timeout
            health_check_interval: Health check interval in seconds
        """
        self.startup_nodes = startup_nodes
        self.password = password
        self.decode_responses = decode_responses
        self.skip_full_coverage_check = skip_full_coverage_check
        self.max_connections_per_node = max_connections_per_node
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.health_check_interval = health_check_interval
        
        self._cluster: Optional[RedisCluster] = None
        self._serializer = EmbeddingSerializer()
        self._health_check_task: Optional[asyncio.Task] = None
        self._cluster_nodes: Dict[str, ClusterNode] = {}
        
        self.log = logger.bind(component="redis_cluster")
    
    async def __aenter__(self) -> 'RedisClusterManager':
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection to Redis cluster."""
        try:
            self._cluster = RedisCluster(
                startup_nodes=self.startup_nodes,
                password=self.password,
                decode_responses=self.decode_responses,
                skip_full_coverage_check=self.skip_full_coverage_check,
                max_connections_per_node=self.max_connections_per_node,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout
            )
            
            # Test connection
            await self._cluster.ping()
            
            # Initialize cluster topology
            await self._refresh_cluster_nodes()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.log.info(
                "Connected to Redis cluster",
                nodes=len(self._cluster_nodes),
                startup_nodes=self.startup_nodes
            )
            
        except Exception as e:
            raise ClusterError(f"Failed to connect to Redis cluster: {e}")
    
    async def close(self) -> None:
        """Close Redis cluster connections."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cluster:
            await self._cluster.close()
            self._cluster = None
        
        self.log.info("Closed Redis cluster connections")
    
    async def _refresh_cluster_nodes(self) -> None:
        """Refresh cluster node information."""
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            nodes = await self._cluster.cluster_nodes()
            self._cluster_nodes.clear()
            
            for node_id, node_info in nodes.items():
                if isinstance(node_info, dict):
                    self._cluster_nodes[node_id] = ClusterNode(
                        host=node_info.get('host', ''),
                        port=node_info.get('port', 0),
                        node_id=node_id,
                        slots=node_info.get('slots', []),
                        is_master=node_info.get('flags', {}).get('master', False),
                        is_replica=node_info.get('flags', {}).get('replica', False)
                    )
            
            self.log.debug("Refreshed cluster topology", nodes=len(self._cluster_nodes))
            
        except Exception as e:
            self.log.warning("Failed to refresh cluster nodes", error=str(e))
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for cluster nodes."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_cluster_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.warning("Health check failed", error=str(e))
    
    async def _check_cluster_health(self) -> None:
        """Check health of cluster nodes."""
        if not self._cluster:
            return
        
        try:
            await self._cluster.ping()
            await self._refresh_cluster_nodes()
            
            healthy_masters = sum(1 for node in self._cluster_nodes.values() if node.is_master)
            self.log.debug("Cluster health check passed", healthy_masters=healthy_masters)
            
        except Exception as e:
            self.log.error("Cluster health check failed", error=str(e))
            raise ClusterError(f"Cluster unhealthy: {e}")
    
    def _get_cache_key(self, text: str, model_version: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{model_version}:{text_hash}"
    
    def _get_similarity_key(self, embedding_hash: str) -> str:
        """Generate key for similarity index."""
        return f"similarity:{embedding_hash}"
    
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        model_version: str,
        similarity_threshold: float = 0.85,
        ttl: Optional[int] = None
    ) -> bool:
        """Store embedding in cluster.
        
        Args:
            text: Original text
            embedding: Embedding vector
            model_version: Model version identifier
            similarity_threshold: Similarity threshold for routing
            ttl: Time-to-live in seconds
            
        Returns:
            True if stored successfully
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            key = self._get_cache_key(text, model_version)
            
            entry = EmbeddingEntry(
                embedding=embedding,
                model_version=model_version,
                timestamp=asyncio.get_event_loop().time(),
                access_count=0,
                similarity_threshold=similarity_threshold
            )
            
            serialized = self._serializer.serialize(entry)
            
            if ttl:
                await self._cluster.setex(key, ttl, serialized)
            else:
                await self._cluster.set(key, serialized)
            
            # Store in similarity index
            embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
            similarity_key = self._get_similarity_key(embedding_hash)
            await self._cluster.sadd(similarity_key, key)
            
            self.log.debug(
                "Stored embedding",
                key=key,
                embedding_shape=embedding.shape,
                model_version=model_version
            )
            
            return True
            
        except Exception as e:
            self.log.error("Failed to store embedding", error=str(e), text=text[:50])
            raise CacheError(f"Failed to store embedding: {e}")
    
    async def get_embedding(
        self,
        text: str,
        model_version: str
    ) -> Optional[EmbeddingEntry]:
        """Retrieve embedding from cluster.
        
        Args:
            text: Original text
            model_version: Model version identifier
            
        Returns:
            Embedding entry if found, None otherwise
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            key = self._get_cache_key(text, model_version)
            data = await self._cluster.get(key)
            
            if not data:
                return None
            
            entry = self._serializer.deserialize(data)
            
            # Increment access count
            entry.access_count += 1
            updated = self._serializer.serialize(entry)
            await self._cluster.set(key, updated)
            
            self.log.debug(
                "Retrieved embedding",
                key=key,
                access_count=entry.access_count,
                model_version=model_version
            )
            
            return entry
            
        except SerializationError:
            raise
        except Exception as e:
            self.log.error("Failed to get embedding", error=str(e), text=text[:50])
            raise CacheError(f"Failed to retrieve embedding: {e}")
    
    async def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float = 0.85,
        limit: int = 10
    ) -> List[Tuple[str, EmbeddingEntry, float]]:
        """Find similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of (key, entry, similarity_score) tuples
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            # Get all similarity index keys
            pattern = "similarity:*"
            similarity_keys = []
            async for key in self._cluster.scan_iter(match=pattern):
                similarity_keys.append(key)
            
            similar_entries = []
            
            for sim_key in similarity_keys:
                embedding_keys = await self._cluster.smembers(sim_key)
                
                for embedding_key in embedding_keys:
                    data = await self._cluster.get(embedding_key)
                    if not data:
                        continue
                    
                    try:
                        entry = self._serializer.deserialize(data)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, entry.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                        )
                        
                        if similarity >= similarity_threshold:
                            similar_entries.append((embedding_key, entry, float(similarity)))
                            
                    except SerializationError:
                        continue
            
            # Sort by similarity score descending
            similar_entries.sort(key=lambda x: x[2], reverse=True)
            
            self.log.debug(
                "Found similar embeddings",
                total_found=len(similar_entries),
                threshold=similarity_threshold,
                limit=limit
            )
            
            return similar_entries[:limit]
            
        except Exception as e:
            self.log.error("Failed to find similar embeddings", error=str(e))
            raise CacheError(f"Failed to find similar embeddings: {e}")
    
    async def bulk_store(
        self,
        entries: List[Tuple[str, np.ndarray, str]],
        ttl: Optional[int] = None
    ) -> int:
        """Store multiple embeddings in batch.
        
        Args:
            entries: List of (text, embedding, model_version) tuples
            ttl: Time-to-live in seconds
            
        Returns:
            Number of successfully stored entries
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        stored_count = 0
        
        try:
            pipe = self._cluster.pipeline()
            
            for text, embedding, model_version in entries:
                try:
                    key = self._get_cache_key(text, model_version)
                    
                    entry = EmbeddingEntry(
                        embedding=embedding,
                        model_version=model_version,
                        timestamp=asyncio.get_event_loop().time(),
                        access_count=0,
                        similarity_threshold=0.85
                    )
                    
                    serialized = self._serializer.serialize(entry)
                    
                    if ttl:
                        pipe.setex(key, ttl, serialized)
                    else:
                        pipe.set(key, serialized)
                    
                    # Add to similarity index
                    embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
                    similarity_key = self._get_similarity_key(embedding_hash)
                    pipe.sadd(similarity_key, key)
                    
                except Exception as e:
                    self.log.warning("Skipped entry in bulk store", error=str(e))
                    continue
            
            await pipe.execute()
            stored_count = len(entries)
            
            self.log.info("Bulk stored embeddings", count=stored_count)
            
        except Exception as e:
            self.log.error("Bulk store failed", error=str(e))
            raise CacheError(f"Bulk store failed: {e}")
        
        return stored_count
    
    async def evict_by_pattern(self, pattern: str) -> int:
        """Evict keys matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of evicted keys
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            keys_to_delete = []
            async for key in self._cluster.scan_iter(match=pattern):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                deleted = await self._cluster.delete(*keys_to_delete)
                self.log.info("Evicted keys", pattern=pattern, count=deleted)
                return deleted
            
            return 0
            
        except Exception as e:
            self.log.error("Failed to evict keys", error=str(e), pattern=pattern)
            raise CacheError(f"Failed to evict keys: {e}")
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics.
        
        Returns:
            Dictionary containing cluster stats
        """
        if not self._cluster:
            raise ClusterError("Not connected to cluster")
        
        try:
            info = await self._cluster.cluster_info()
            stats = {
                'cluster_state': info.get('cluster_state', 'unknown'),
                'cluster_slots_assigned': info.get('cluster_slots_assigned', 0),
                'cluster_slots_ok': info.get('cluster_slots_ok', 0),
                'cluster_slots_pfail': info.get('cluster_slots_pfail', 0),
                'cluster_slots_fail': info.get('cluster_slots_fail', 0),
                'cluster_known_nodes': info.get('cluster_known_nodes', 0),
                'cluster_size': info.get('cluster_size', 0),
                'nodes': {}
            }
            
            for node_id, node in self._cluster_nodes.items():
                stats['nodes'][node_id] = {
                    'host': node.host,
                    'port': node.port,
                    'is_master': node.is_master,
                    'is_replica': node.is_replica,
                    'slots_count': len(node.slots)
                }
            
            return stats
            
        except Exception as e:
            self.log.error("Failed to get cluster stats", error=str(e))
            raise CacheError(f"Failed to get cluster stats: {e}")