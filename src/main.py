from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import os
import asyncio

import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from .config import Settings, get_settings
from .cache.manager import CacheManager
from .models.requests import EmbeddingRequest, SearchRequest, BatchRequest
from .models.responses import EmbeddingResponse, SearchResponse, HealthResponse
from .exceptions import CacheException, EmbeddingException
from .metrics import MetricsCollector
from .middleware.tracing import TracingMiddleware
from .middleware.rate_limiting import RateLimitingMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global components
cache_manager: Optional[CacheManager] = None
metrics_collector: Optional[MetricsCollector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with proper startup/shutdown."""
    global cache_manager, metrics_collector
    
    settings = get_settings()
    logger.info("Starting distributed embedding cache service", 
                version=settings.VERSION, 
                environment=settings.ENVIRONMENT)
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager(settings)
        await cache_manager.initialize()
        logger.info("Cache manager initialized successfully")
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(cache_manager)
        
        # Start background tasks
        background_tasks = [
            asyncio.create_task(cache_manager.start_precomputation_worker()),
            asyncio.create_task(cache_manager.start_cleanup_worker()),
            asyncio.create_task(metrics_collector.start_collection())
        ]
        
        logger.info("All background services started")
        yield
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e), exc_info=True)
        raise
        
    finally:
        logger.info("Shutting down application")
        
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cleanup resources
        if cache_manager:
            await cache_manager.shutdown()
        
        if metrics_collector:
            await metrics_collector.shutdown()
            
        logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Distributed Embedding Cache",
    description="High-performance distributed cache for ML embeddings with predictive precomputation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TracingMiddleware)
app.add_middleware(RateLimitingMiddleware)

# Initialize Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app)

async def get_cache_manager() -> CacheManager:
    """Dependency to get cache manager instance."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    return cache_manager

async def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector instance."""
    if metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    return metrics_collector

@app.exception_handler(CacheException)
async def cache_exception_handler(request, exc: CacheException):
    """Handle cache-specific exceptions."""
    logger.error("Cache error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=503,
        content={"error": "Cache service unavailable", "detail": str(exc)}
    )

@app.exception_handler(EmbeddingException)
async def embedding_exception_handler(request, exc: EmbeddingException):
    """Handle embedding-specific exceptions."""
    logger.error("Embedding error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=422,
        content={"error": "Embedding processing failed", "detail": str(exc)}
    )

@app.get("/health", response_model=HealthResponse)
async def health_check(
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> HealthResponse:
    """Health check endpoint with detailed status."""
    try:
        # Check cache connectivity
        cache_healthy = await cache_mgr.health_check()
        
        # Get basic stats
        stats = await cache_mgr.get_stats()
        
        status = "healthy" if cache_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=asyncio.get_event_loop().time(),
            cache_nodes=stats.get("active_nodes", 0),
            total_embeddings=stats.get("total_embeddings", 0),
            cache_hit_rate=stats.get("hit_rate", 0.0)
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=asyncio.get_event_loop().time(),
            cache_nodes=0,
            total_embeddings=0,
            cache_hit_rate=0.0
        )

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embedding(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> EmbeddingResponse:
    """Get embedding with caching and precomputation."""
    try:
        logger.info("Processing embedding request", 
                   text_length=len(request.text),
                   model=request.model,
                   cache_key=request.cache_key)
        
        # Try cache first
        cached_result = await cache_mgr.get_embedding(
            text=request.text,
            model=request.model,
            cache_key=request.cache_key
        )
        
        if cached_result:
            logger.info("Cache hit", cache_key=request.cache_key)
            return EmbeddingResponse(
                embedding=cached_result["embedding"],
                model=cached_result["model"],
                cached=True,
                cache_key=cached_result["cache_key"],
                similarity_score=cached_result.get("similarity_score")
            )
        
        # Generate new embedding
        result = await cache_mgr.compute_and_cache_embedding(
            text=request.text,
            model=request.model,
            cache_key=request.cache_key,
            ttl=request.ttl
        )
        
        # Trigger precomputation for similar queries
        if request.enable_precomputation:
            background_tasks.add_task(
                cache_mgr.trigger_precomputation,
                request.text,
                request.model
            )
        
        logger.info("Embedding computed and cached", cache_key=result["cache_key"])
        
        return EmbeddingResponse(
            embedding=result["embedding"],
            model=result["model"],
            cached=False,
            cache_key=result["cache_key"]
        )
        
    except Exception as e:
        logger.error("Failed to process embedding request", 
                    error=str(e), 
                    text=request.text[:100])
        raise HTTPException(status_code=500, detail=f"Failed to process embedding: {str(e)}")

@app.post("/embeddings/batch", response_model=List[EmbeddingResponse])
async def get_embeddings_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> List[EmbeddingResponse]:
    """Process multiple embeddings in batch with optimized caching."""
    try:
        logger.info("Processing batch embedding request", 
                   batch_size=len(request.requests),
                   model=request.model)
        
        # Process batch with cache optimization
        results = await cache_mgr.process_batch_embeddings(
            requests=[{
                "text": req.text,
                "cache_key": req.cache_key,
                "ttl": req.ttl
            } for req in request.requests],
            model=request.model
        )
        
        responses = []
        for i, result in enumerate(results):
            responses.append(EmbeddingResponse(
                embedding=result["embedding"],
                model=result["model"],
                cached=result["cached"],
                cache_key=result["cache_key"],
                similarity_score=result.get("similarity_score")
            ))
            
            # Trigger precomputation for non-cached results
            if not result["cached"] and request.enable_precomputation:
                background_tasks.add_task(
                    cache_mgr.trigger_precomputation,
                    request.requests[i].text,
                    request.model
                )
        
        cache_hits = sum(1 for r in results if r["cached"])
        logger.info("Batch processing complete",
                   total=len(results),
                   cache_hits=cache_hits,
                   cache_hit_rate=cache_hits / len(results))
        
        return responses
        
    except Exception as e:
        logger.error("Failed to process batch embedding request", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> SearchResponse:
    """Perform semantic similarity search across cached embeddings."""
    try:
        logger.info("Processing semantic search", 
                   query_length=len(request.query),
                   model=request.model,
                   top_k=request.top_k)
        
        results = await cache_mgr.semantic_search(
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters
        )
        
        logger.info("Semantic search complete", 
                   results_count=len(results),
                   query=request.query[:50])
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0
        )
        
    except Exception as e:
        logger.error("Failed to process semantic search", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/cache/{cache_key}")
async def invalidate_cache_key(
    cache_key: str,
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """Invalidate specific cache key."""
    try:
        success = await cache_mgr.invalidate_key(cache_key)
        logger.info("Cache key invalidation", cache_key=cache_key, success=success)
        return {"cache_key": cache_key, "invalidated": success}
    except Exception as e:
        logger.error("Failed to invalidate cache key", cache_key=cache_key, error=str(e))
        raise HTTPException(status_code=500, detail=f"Invalidation failed: {str(e)}")

@app.post("/cache/warm")
async def warm_cache(
    requests: List[EmbeddingRequest],
    background_tasks: BackgroundTasks,
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """Warm cache with pre-computed embeddings."""
    try:
        logger.info("Starting cache warming", request_count=len(requests))
        
        background_tasks.add_task(
            cache_mgr.warm_cache,
            [(req.text, req.model, req.cache_key) for req in requests]
        )
        
        return {
            "message": "Cache warming started",
            "request_count": len(requests),
            "status": "processing"
        }
    except Exception as e:
        logger.error("Failed to start cache warming", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache warming failed: {str(e)}")

@app.get("/stats")
async def get_cache_stats(
    cache_mgr: CacheManager = Depends(get_cache_manager),
    metrics: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """Get detailed cache and performance statistics."""
    try:
        cache_stats = await cache_mgr.get_detailed_stats()
        performance_stats = await metrics.get_performance_stats()
        
        return {
            "cache": cache_stats,
            "performance": performance_stats,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": settings.LOG_LEVEL,
                "handlers": ["default"],
            },
        },
        access_log=True,
        reload=settings.ENVIRONMENT == "development"
    )