"""
Prometheus metrics exporter for distributed embedding cache.

Tracks cache performance, embedding operations, and system health metrics.
"""
import asyncio
import time
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)
from prometheus_client.exposition import MetricsHandler
import redis.asyncio as redis
from fastapi import HTTPException

from ..core.cache_engine import CacheEngine
from ..storage.redis_cluster import RedisClusterManager
from ..models.precompute_model import PrecomputeModel


logger = structlog.get_logger(__name__)


class MetricsExportError(Exception):
    """Raised when metrics export fails."""
    pass


class MetricsCollectionError(Exception):
    """Raised when metrics collection fails."""
    pass


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: datetime
    cache_hits: int
    cache_misses: int
    total_embeddings: int
    avg_similarity_score: float
    precompute_queue_size: int
    redis_memory_usage: int
    active_connections: int
    error_count: int
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and export."""
    port: int = 9090
    collection_interval: float = 30.0
    retention_days: int = 7
    redis_stats_enabled: bool = True
    detailed_timing_enabled: bool = True
    custom_labels: Dict[str, str] = field(default_factory=dict)


class PrometheusExporter:
    """
    Comprehensive Prometheus metrics exporter for the distributed embedding cache.
    
    Collects and exports:
    - Cache performance metrics (hits/misses, latency)
    - Embedding operation metrics (similarity, predictions)
    - System health metrics (memory, connections)
    - Error tracking and alerting metrics
    """
    
    def __init__(
        self,
        cache_engine: CacheEngine,
        redis_manager: RedisClusterManager,
        precompute_model: PrecomputeModel,
        config: Optional[MetricsConfig] = None
    ):
        self.cache_engine = cache_engine
        self.redis_manager = redis_manager
        self.precompute_model = precompute_model
        self.config = config or MetricsConfig()
        
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Cache operation metrics
        self.cache_operations = Counter(
            'embedding_cache_operations_total',
            'Total cache operations by type',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_latency = Histogram(
            'embedding_cache_latency_seconds',
            'Cache operation latency',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'embedding_cache_hit_rate',
            'Cache hit rate over time window',
            registry=self.registry
        )
        
        # Embedding metrics
        self.embedding_operations = Counter(
            'embedding_operations_total',
            'Total embedding operations',
            ['operation', 'model'],
            registry=self.registry
        )
        
        self.similarity_scores = Histogram(
            'embedding_similarity_scores',
            'Distribution of similarity scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.precompute_queue_size = Gauge(
            'embedding_precompute_queue_size',
            'Size of precomputation queue',
            registry=self.registry
        )
        
        # System metrics
        self.redis_memory_usage = Gauge(
            'redis_memory_usage_bytes',
            'Redis memory usage by node',
            ['node'],
            registry=self.registry
        )
        
        self.redis_connections = Gauge(
            'redis_connections_active',
            'Active Redis connections by node',
            ['node'],
            registry=self.registry
        )
        
        self.system_errors = Counter(
            'system_errors_total',
            'System errors by component',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Performance metrics
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.grpc_request_duration = Histogram(
            'grpc_request_duration_seconds',
            'gRPC request duration',
            ['service', 'method', 'status'],
            registry=self.registry
        )
        
        # Info metrics
        self.build_info = Info(
            'embedding_cache_build_info',
            'Build information',
            registry=self.registry
        )
        
        # Internal state
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._metrics_history: List[MetricSnapshot] = []
        self._last_snapshot: Optional[MetricSnapshot] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        
        logger.info(
            "Prometheus exporter initialized",
            port=self.config.port,
            collection_interval=self.config.collection_interval
        )
    
    async def start(self) -> None:
        """Start the metrics collection and HTTP server."""
        if self._running:
            raise MetricsExportError("Metrics exporter already running")
        
        try:
            # Set build info
            self.build_info.info({
                'version': '1.0.0',
                'python_version': '3.11',
                'build_date': datetime.utcnow().isoformat(),
                **self.config.custom_labels
            })
            
            # Start HTTP server for metrics endpoint
            start_http_server(
                self.config.port, 
                registry=self.registry,
                handler=self._create_metrics_handler()
            )
            
            # Start background collection
            self._running = True
            self._collection_task = asyncio.create_task(self._collection_loop())
            
            logger.info(
                "Metrics exporter started",
                metrics_url=f"http://localhost:{self.config.port}/metrics"
            )
            
        except Exception as e:
            logger.error("Failed to start metrics exporter", error=str(e))
            raise MetricsExportError(f"Failed to start exporter: {e}") from e
    
    async def stop(self) -> None:
        """Stop the metrics collection."""
        if not self._running:
            return
        
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("Metrics exporter stopped")
    
    def _create_metrics_handler(self) -> type:
        """Create custom metrics handler with additional endpoints."""
        registry = self.registry
        
        class CustomMetricsHandler(MetricsHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(generate_latest(registry))
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "healthy"}')
                else:
                    self.send_error(404)
        
        return CustomMetricsHandler
    
    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        logger.info("Starting metrics collection loop")
        
        while self._running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                self.system_errors.labels(
                    component="metrics_collector",
                    error_type=type(e).__name__
                ).inc()
                await asyncio.sleep(5.0)  # Brief pause on error
    
    async def _collect_all_metrics(self) -> None:
        """Collect all metrics from various sources."""
        collection_start = time.time()
        
        try:
            # Collect metrics concurrently
            await asyncio.gather(
                self._collect_cache_metrics(),
                self._collect_redis_metrics(),
                self._collect_embedding_metrics(),
                self._collect_system_health(),
                return_exceptions=True
            )
            
            # Create snapshot
            snapshot = await self._create_snapshot()
            self._update_history(snapshot)
            
            collection_duration = time.time() - collection_start
            logger.debug(
                "Metrics collection completed",
                duration=collection_duration,
                snapshot_time=snapshot.timestamp.isoformat()
            )
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
            raise MetricsCollectionError(f"Collection failed: {e}") from e
    
    async def _collect_cache_metrics(self) -> None:
        """Collect cache performance metrics."""
        try:
            # Get cache stats from engine
            stats = await self.cache_engine.get_stats()
            
            # Update counters
            for operation, count in stats.get('operations', {}).items():
                self.cache_operations.labels(
                    operation=operation,
                    status='success'
                )._value._value = count
            
            # Update hit rate
            hit_rate = stats.get('hit_rate', 0.0)
            self.cache_hit_rate.set(hit_rate)
            
            # Update latency histograms
            for operation, latencies in stats.get('latencies', {}).items():
                for latency in latencies:
                    self.cache_latency.labels(operation=operation).observe(latency)
            
        except Exception as e:
            logger.warning("Failed to collect cache metrics", error=str(e))
            self.system_errors.labels(
                component="cache",
                error_type=type(e).__name__
            ).inc()
    
    async def _collect_redis_metrics(self) -> None:
        """Collect Redis cluster metrics."""
        if not self.config.redis_stats_enabled:
            return
        
        try:
            nodes = await self.redis_manager.get_cluster_nodes()
            
            for node_info in nodes:
                node_id = node_info.get('id', 'unknown')
                
                # Memory usage
                memory_info = await self.redis_manager.get_node_memory_info(node_id)
                if memory_info:
                    self.redis_memory_usage.labels(node=node_id).set(
                        memory_info.get('used_memory', 0)
                    )
                
                # Connection count
                conn_info = await self.redis_manager.get_node_connection_info(node_id)
                if conn_info:
                    self.redis_connections.labels(node=node_id).set(
                        conn_info.get('connected_clients', 0)
                    )
            
        except Exception as e:
            logger.warning("Failed to collect Redis metrics", error=str(e))
            self.system_errors.labels(
                component="redis",
                error_type=type(e).__name__
            ).inc()
    
    async def _collect_embedding_metrics(self) -> None:
        """Collect embedding and ML model metrics."""
        try:
            # Precompute queue size
            queue_size = await self.precompute_model.get_queue_size()
            self.precompute_queue_size.set(queue_size)
            
            # Get recent similarity scores
            recent_scores = await self._get_recent_similarity_scores()
            for score in recent_scores:
                self.similarity_scores.observe(score)
            
        except Exception as e:
            logger.warning("Failed to collect embedding metrics", error=str(e))
            self.system_errors.labels(
                component="embeddings",
                error_type=type(e).__name__
            ).inc()
    
    async def _collect_system_health(self) -> None:
        """Collect system health metrics."""
        try:
            # Check component health
            components = [
                ('cache_engine', self.cache_engine),
                ('redis_manager', self.redis_manager),
                ('precompute_model', self.precompute_model)
            ]
            
            for component_name, component in components:
                try:
                    # Attempt health check
                    if hasattr(component, 'health_check'):
                        await component.health_check()
                except Exception as e:
                    self.system_errors.labels(
                        component=component_name,
                        error_type=type(e).__name__
                    ).inc()
            
        except Exception as e:
            logger.warning("Failed to collect system health", error=str(e))
    
    async def _get_recent_similarity_scores(self) -> List[float]:
        """Get recent similarity scores for metrics."""
        try:
            # This would typically come from the cache engine's recent operations
            return await self.cache_engine.get_recent_similarity_scores()
        except Exception:
            return []
    
    async def _create_snapshot(self) -> MetricSnapshot:
        """Create a snapshot of current metrics."""
        try:
            cache_stats = await self.cache_engine.get_stats()
            precompute_queue = await self.precompute_model.get_queue_size()
            
            return MetricSnapshot(
                timestamp=datetime.utcnow(),
                cache_hits=cache_stats.get('hits', 0),
                cache_misses=cache_stats.get('misses', 0),
                total_embeddings=cache_stats.get('total_embeddings', 0),
                avg_similarity_score=cache_stats.get('avg_similarity', 0.0),
                precompute_queue_size=precompute_queue,
                redis_memory_usage=await self._get_total_redis_memory(),
                active_connections=await self._get_total_redis_connections(),
                error_count=cache_stats.get('errors', 0)
            )
            
        except Exception as e:
            logger.error("Failed to create metrics snapshot", error=str(e))
            # Return empty snapshot
            return MetricSnapshot(
                timestamp=datetime.utcnow(),
                cache_hits=0, cache_misses=0, total_embeddings=0,
                avg_similarity_score=0.0, precompute_queue_size=0,
                redis_memory_usage=0, active_connections=0, error_count=0
            )
    
    async def _get_total_redis_memory(self) -> int:
        """Get total memory usage across Redis cluster."""
        try:
            nodes = await self.redis_manager.get_cluster_nodes()
            total_memory = 0
            
            for node_info in nodes:
                node_id = node_info.get('id', 'unknown')
                memory_info = await self.redis_manager.get_node_memory_info(node_id)
                if memory_info:
                    total_memory += memory_info.get('used_memory', 0)
            
            return total_memory
        except Exception:
            return 0
    
    async def _get_total_redis_connections(self) -> int:
        """Get total active connections across Redis cluster."""
        try:
            nodes = await self.redis_manager.get_cluster_nodes()
            total_connections = 0
            
            for node_info in nodes:
                node_id = node_info.get('id', 'unknown')
                conn_info = await self.redis_manager.get_node_connection_info(node_id)
                if conn_info:
                    total_connections += conn_info.get('connected_clients', 0)
            
            return total_connections
        except Exception:
            return 0
    
    def _update_history(self, snapshot: MetricSnapshot) -> None:
        """Update metrics history and clean old entries."""
        self._metrics_history.append(snapshot)
        self._last_snapshot = snapshot
        
        # Clean old entries
        cutoff_time = datetime.utcnow() - timedelta(days=self.config.retention_days)
        self._metrics_history = [
            s for s in self._metrics_history 
            if s.timestamp > cutoff_time
        ]
    
    # Public metric recording methods for use by other components
    
    def record_cache_operation(
        self, 
        operation: str, 
        status: str = 'success',
        duration: Optional[float] = None
    ) -> None:
        """Record a cache operation."""
        self.cache_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        if duration is not None:
            self.cache_latency.labels(operation=operation).observe(duration)
    
    def record_embedding_operation(
        self,
        operation: str,
        model: str = 'default'
    ) -> None:
        """Record an embedding operation."""
        self.embedding_operations.labels(
            operation=operation,
            model=model
        ).inc()
    
    def record_similarity_score(self, score: float) -> None:
        """Record a similarity score."""
        self.similarity_scores.observe(score)
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status: str,
        duration: float
    ) -> None:
        """Record an HTTP request."""
        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).observe(duration)
    
    def record_grpc_request(
        self,
        service: str,
        method: str,
        status: str,
        duration: float
    ) -> None:
        """Record a gRPC request."""
        self.grpc_request_duration.labels(
            service=service,
            method=method,
            status=status
        ).observe(duration)
    
    def record_error(self, component: str, error_type: str) -> None:
        """Record a system error."""
        self.system_errors.labels(
            component=component,
            error_type=error_type
        ).inc()
    
    def get_current_snapshot(self) -> Optional[MetricSnapshot]:
        """Get the most recent metrics snapshot."""
        return self._last_snapshot
    
    def get_metrics_history(self) -> List[MetricSnapshot]:
        """Get historical metrics snapshots."""
        return self._metrics_history.copy()
    
    async def export_metrics(self) -> str:
        """Export current metrics in Prometheus format."""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error("Failed to export metrics", error=str(e))
            raise MetricsExportError(f"Export failed: {e}") from e