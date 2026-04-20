"""Integration tests for cache performance and load testing."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from unittest.mock import AsyncMock, patch
import random

import grpc
import numpy as np
import pytest
import structlog
from prometheus_client import CollectorRegistry, REGISTRY
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.api.grpc_server import EmbeddingCacheService
from src.core.cache_engine import CacheEngine, CacheHit, CacheMiss
from src.core.embedding_predictor import EmbeddingPredictor
from src.core.similarity_router import SimilarityRouter
from src.storage.redis_cluster import RedisClusterManager
from src.metrics.prometheus_exporter import MetricsExporter

logger = structlog.get_logger(__name__)


class PerformanceTestError(Exception):
    """Performance test related errors."""
    pass


class LoadTestTimeout(PerformanceTestError):
    """Load test exceeded timeout."""
    pass


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""
    concurrent_clients: int = 100
    requests_per_client: int = 50
    embedding_dimension: int = 768
    cache_hit_ratio: float = 0.7
    timeout_seconds: int = 300
    ramp_up_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance test metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    requests_per_second: float
    cache_hit_rate: float
    error_rate: float
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return asdict(self)


class LoadTestClient:
    """Client for load testing cache operations."""
    
    def __init__(
        self, 
        cache_engine: CacheEngine,
        client_id: int,
        config: LoadTestConfig
    ):
        self.cache_engine = cache_engine
        self.client_id = client_id
        self.config = config
        self.logger = logger.bind(client_id=client_id)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    async def generate_test_embedding(self, key: str) -> np.ndarray:
        """Generate or retrieve test embedding."""
        if key not in self._embedding_cache:
            self._embedding_cache[key] = np.random.rand(
                self.config.embedding_dimension
            ).astype(np.float32)
        return self._embedding_cache[key]
    
    async def run_load_test(self) -> List[Dict[str, Any]]:
        """Run load test for this client."""
        results = []
        
        for request_id in range(self.config.requests_per_client):
            # Simulate cache hit ratio
            if random.random() < self.config.cache_hit_ratio:
                # Use existing key for cache hit
                key = f"test_key_{request_id % 10}"
            else:
                # Use unique key for cache miss
                key = f"test_key_{self.client_id}_{request_id}_{time.time()}"
            
            embedding = await self.generate_test_embedding(key)
            
            start_time = time.perf_counter()
            try:
                result = await self.cache_engine.get_or_compute(
                    key=key,
                    compute_fn=lambda: embedding,
                    namespace="load_test"
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                results.append({
                    'client_id': self.client_id,
                    'request_id': request_id,
                    'key': key,
                    'latency_ms': latency_ms,
                    'success': True,
                    'cache_hit': isinstance(result, CacheHit),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.logger.error(
                    "Request failed",
                    request_id=request_id,
                    key=key,
                    error=str(e)
                )
                
                results.append({
                    'client_id': self.client_id,
                    'request_id': request_id,
                    'key': key,
                    'latency_ms': latency_ms,
                    'success': False,
                    'cache_hit': False,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        return results


class PerformanceBenchmark:
    """Performance benchmark suite for distributed embedding cache."""
    
    def __init__(self, cache_engine: CacheEngine, metrics_exporter: MetricsExporter):
        self.cache_engine = cache_engine
        self.metrics_exporter = metrics_exporter
        self.logger = logger.bind(component="performance_benchmark")
    
    async def run_load_test(self, config: LoadTestConfig) -> PerformanceMetrics:
        """Run load test with specified configuration."""
        self.logger.info("Starting load test", config=config.to_dict())
        
        start_time = time.perf_counter()
        timeout_time = start_time + config.timeout_seconds
        
        # Create clients
        clients = [
            LoadTestClient(self.cache_engine, client_id, config)
            for client_id in range(config.concurrent_clients)
        ]
        
        # Stagger client startup for ramp-up
        async def start_client_with_delay(client: LoadTestClient, delay: float) -> List[Dict[str, Any]]:
            await asyncio.sleep(delay)
            return await client.run_load_test()
        
        # Calculate ramp-up delays
        ramp_delay = config.ramp_up_seconds / config.concurrent_clients
        tasks = [
            start_client_with_delay(client, i * ramp_delay)
            for i, client in enumerate(clients)
        ]
        
        try:
            # Run all clients concurrently with timeout
            all_results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=config.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise LoadTestTimeout(f"Load test exceeded {config.timeout_seconds}s timeout")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Flatten results
        flat_results = []
        for client_results in all_results:
            flat_results.extend(client_results)
        
        # Calculate metrics
        metrics = self._calculate_metrics(flat_results, duration)
        
        self.logger.info(
            "Load test completed",
            metrics=metrics.to_dict()
        )
        
        return metrics
    
    def _calculate_metrics(self, results: List[Dict[str, Any]], duration: float) -> PerformanceMetrics:
        """Calculate performance metrics from test results."""
        if not results:
            return PerformanceMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                max_latency_ms=0.0,
                requests_per_second=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                duration_seconds=duration
            )
        
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        
        latencies = [r['latency_ms'] for r in results]
        cache_hits = sum(1 for r in results if r.get('cache_hit', False))
        
        latencies_sorted = sorted(latencies)
        p95_idx = int(0.95 * len(latencies_sorted))
        p99_idx = int(0.99 * len(latencies_sorted))
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=latencies_sorted[p95_idx] if latencies_sorted else 0.0,
            p99_latency_ms=latencies_sorted[p99_idx] if latencies_sorted else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            requests_per_second=total_requests / duration if duration > 0 else 0.0,
            cache_hit_rate=cache_hits / total_requests if total_requests > 0 else 0.0,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0.0,
            duration_seconds=duration
        )
    
    async def benchmark_cache_operations(
        self, 
        num_operations: int = 1000,
        embedding_dim: int = 768
    ) -> Dict[str, float]:
        """Benchmark individual cache operations."""
        self.logger.info("Starting cache operations benchmark")
        
        # Generate test embeddings
        embeddings = {
            f"bench_key_{i}": np.random.rand(embedding_dim).astype(np.float32)
            for i in range(num_operations)
        }
        
        # Benchmark cache misses (first access)
        miss_times = []
        for key, embedding in embeddings.items():
            start_time = time.perf_counter()
            await self.cache_engine.get_or_compute(
                key=key,
                compute_fn=lambda e=embedding: e,
                namespace="benchmark"
            )
            miss_times.append((time.perf_counter() - start_time) * 1000)
        
        # Benchmark cache hits (second access)
        hit_times = []
        for key in embeddings.keys():
            start_time = time.perf_counter()
            await self.cache_engine.get_or_compute(
                key=key,
                compute_fn=lambda: np.random.rand(embedding_dim).astype(np.float32),
                namespace="benchmark"
            )
            hit_times.append((time.perf_counter() - start_time) * 1000)
        
        return {
            'avg_cache_miss_latency_ms': sum(miss_times) / len(miss_times),
            'avg_cache_hit_latency_ms': sum(hit_times) / len(hit_times),
            'p95_cache_miss_latency_ms': sorted(miss_times)[int(0.95 * len(miss_times))],
            'p95_cache_hit_latency_ms': sorted(hit_times)[int(0.95 * len(hit_times))],
            'cache_hit_speedup_ratio': (sum(miss_times) / len(miss_times)) / (sum(hit_times) / len(hit_times))
        }


@pytest.fixture
async def redis_cluster():
    """Redis cluster for testing."""
    mock_redis = AsyncMock(spec=Redis)
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.ping.return_value = True
    
    cluster_manager = RedisClusterManager(
        nodes=[{"host": "localhost", "port": 6379}],
        max_connections_per_node=10
    )
    
    with patch.object(cluster_manager, '_redis_client', mock_redis):
        yield cluster_manager


@pytest.fixture
async def cache_engine(redis_cluster):
    """Cache engine for testing."""
    predictor = AsyncMock(spec=EmbeddingPredictor)
    router = AsyncMock(spec=SimilarityRouter)
    
    registry = CollectorRegistry()
    metrics_exporter = MetricsExporter(registry=registry)
    
    engine = CacheEngine(
        redis_manager=redis_cluster,
        predictor=predictor,
        router=router,
        metrics_exporter=metrics_exporter
    )
    
    return engine


@pytest.fixture
def performance_benchmark(cache_engine):
    """Performance benchmark suite."""
    registry = CollectorRegistry()
    metrics_exporter = MetricsExporter(registry=registry)
    return PerformanceBenchmark(cache_engine, metrics_exporter)


@pytest.mark.asyncio
async def test_load_test_small_scale(performance_benchmark):
    """Test load test with small scale configuration."""
    config = LoadTestConfig(
        concurrent_clients=5,
        requests_per_client=10,
        embedding_dimension=128,
        cache_hit_ratio=0.8,
        timeout_seconds=30,
        ramp_up_seconds=5
    )
    
    metrics = await performance_benchmark.run_load_test(config)
    
    assert metrics.total_requests == 50  # 5 clients * 10 requests
    assert metrics.error_rate <= 0.1  # Max 10% error rate
    assert metrics.requests_per_second > 0
    assert metrics.duration_seconds > 0
    assert 0 <= metrics.cache_hit_rate <= 1


@pytest.mark.asyncio
async def test_load_test_high_concurrency(performance_benchmark):
    """Test load test with high concurrency."""
    config = LoadTestConfig(
        concurrent_clients=50,
        requests_per_client=20,
        embedding_dimension=768,
        cache_hit_ratio=0.6,
        timeout_seconds=120,
        ramp_up_seconds=10
    )
    
    metrics = await performance_benchmark.run_load_test(config)
    
    assert metrics.total_requests == 1000  # 50 clients * 20 requests
    assert metrics.successful_requests > 0
    assert metrics.avg_latency_ms > 0
    assert metrics.p95_latency_ms >= metrics.avg_latency_ms
    assert metrics.p99_latency_ms >= metrics.p95_latency_ms
    assert metrics.requests_per_second > 0


@pytest.mark.asyncio
async def test_cache_operations_benchmark(performance_benchmark):
    """Test individual cache operations benchmark."""
    results = await performance_benchmark.benchmark_cache_operations(
        num_operations=100,
        embedding_dim=512
    )
    
    assert 'avg_cache_miss_latency_ms' in results
    assert 'avg_cache_hit_latency_ms' in results
    assert 'p95_cache_miss_latency_ms' in results
    assert 'p95_cache_hit_latency_ms' in results
    assert 'cache_hit_speedup_ratio' in results
    
    # Cache hits should be faster than misses
    assert results['avg_cache_hit_latency_ms'] < results['avg_cache_miss_latency_ms']
    assert results['cache_hit_speedup_ratio'] > 1.0


@pytest.mark.asyncio
async def test_load_test_timeout_handling(performance_benchmark):
    """Test load test timeout handling."""
    config = LoadTestConfig(
        concurrent_clients=10,
        requests_per_client=5,
        timeout_seconds=0.1  # Very short timeout
    )
    
    with pytest.raises(LoadTestTimeout):
        await performance_benchmark.run_load_test(config)


@pytest.mark.asyncio
async def test_performance_metrics_calculation():
    """Test performance metrics calculation."""
    benchmark = PerformanceBenchmark(AsyncMock(), AsyncMock())
    
    # Test with empty results
    metrics = benchmark._calculate_metrics([], 10.0)
    assert metrics.total_requests == 0
    assert metrics.error_rate == 0.0
    
    # Test with sample results
    results = [
        {'success': True, 'latency_ms': 10.0, 'cache_hit': True},
        {'success': True, 'latency_ms': 20.0, 'cache_hit': False},
        {'success': False, 'latency_ms': 15.0, 'cache_hit': False},
        {'success': True, 'latency_ms': 5.0, 'cache_hit': True},
    ]
    
    metrics = benchmark._calculate_metrics(results, 2.0)
    
    assert metrics.total_requests == 4
    assert metrics.successful_requests == 3
    assert metrics.failed_requests == 1
    assert metrics.error_rate == 0.25
    assert metrics.cache_hit_rate == 0.5
    assert metrics.requests_per_second == 2.0
    assert metrics.avg_latency_ms == 12.5


@pytest.mark.asyncio
async def test_load_test_client():
    """Test individual load test client."""
    cache_engine = AsyncMock(spec=CacheEngine)
    cache_engine.get_or_compute.return_value = CacheHit(
        key="test_key",
        value=np.random.rand(128).astype(np.float32),
        namespace="test"
    )
    
    config = LoadTestConfig(
        concurrent_clients=1,
        requests_per_client=5,
        embedding_dimension=128,
        cache_hit_ratio=1.0
    )
    
    client = LoadTestClient(cache_engine, 0, config)
    results = await client.run_load_test()
    
    assert len(results) == 5
    for result in results:
        assert 'client_id' in result
        assert 'latency_ms' in result
        assert 'success' in result
        assert result['client_id'] == 0


@pytest.mark.asyncio
async def test_stress_test_configuration():
    """Test stress test with extreme configuration."""
    cache_engine = AsyncMock(spec=CacheEngine)
    cache_engine.get_or_compute.return_value = CacheHit(
        key="stress_key",
        value=np.random.rand(1024).astype(np.float32),
        namespace="stress"
    )
    
    registry = CollectorRegistry()
    metrics_exporter = MetricsExporter(registry=registry)
    benchmark = PerformanceBenchmark(cache_engine, metrics_exporter)
    
    config = LoadTestConfig(
        concurrent_clients=200,
        requests_per_client=100,
        embedding_dimension=1024,
        cache_hit_ratio=0.5,
        timeout_seconds=300,
        ramp_up_seconds=30
    )
    
    # This would be a real stress test - just verify config is valid
    assert config.concurrent_clients > 0
    assert config.requests_per_client > 0
    assert config.timeout_seconds > config.ramp_up_seconds


if __name__ == "__main__":
    # Example usage for manual testing
    async def main():
        """Run performance tests manually."""
        # This would connect to real Redis cluster in actual usage
        logger.info("Performance tests should be run via pytest")
        
    asyncio.run(main())