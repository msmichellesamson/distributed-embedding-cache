import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import structlog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.core.exceptions import PredictionError, ModelNotReadyError, TrainingError

logger = structlog.get_logger(__name__)

@dataclass
class AccessPattern:
    """Represents an embedding access pattern for training."""
    timestamp: float
    embedding_key: str
    vector_size: int
    similarity_score: Optional[float]
    access_frequency: int
    time_since_last_access: float
    cluster_id: Optional[str]

@dataclass
class PredictionResult:
    """Result of cache prediction."""
    embedding_key: str
    predicted_access_time: float
    confidence: float
    priority_score: float
    should_precompute: bool

class AccessPatternDataset(Dataset):
    """PyTorch dataset for access pattern training data."""
    
    def __init__(self, patterns: List[AccessPattern], scaler: Optional[StandardScaler] = None):
        self.patterns = patterns
        self.scaler = scaler or StandardScaler()
        self.features, self.targets = self._prepare_data()
        
    def _prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert access patterns to training features and targets."""
        features = []
        targets = []
        
        for pattern in self.patterns:
            feature_vector = [
                pattern.vector_size,
                pattern.similarity_score or 0.0,
                pattern.access_frequency,
                pattern.time_since_last_access,
                hash(pattern.cluster_id or "") % 1000,  # Simple cluster encoding
                time.time() - pattern.timestamp,  # Age of pattern
            ]
            features.append(feature_vector)
            # Target is time until next access (simplified)
            targets.append(pattern.timestamp)
        
        features_array = np.array(features, dtype=np.float32)
        if len(features_array) > 1:
            features_array = self.scaler.fit_transform(features_array)
        
        return torch.tensor(features_array), torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class EmbeddingPredictorModel(nn.Module):
    """Neural network for predicting embedding access patterns."""
    
    def __init__(self, input_size: int = 6, hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 32, 16]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))  # Output: predicted access time
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EmbeddingPredictor:
    """ML-powered predictor for embedding cache needs."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 training_window_hours: int = 24,
                 prediction_threshold: float = 0.7,
                 max_patterns: int = 10000):
        self.model: Optional[EmbeddingPredictorModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path = model_path
        self.training_window_hours = training_window_hours
        self.prediction_threshold = prediction_threshold
        self.max_patterns = max_patterns
        
        # In-memory pattern storage
        self.access_patterns: deque = deque(maxlen=max_patterns)
        self.embedding_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'access_count': 0,
            'last_access': 0.0,
            'average_interval': 0.0,
            'similarity_scores': deque(maxlen=100)
        })
        
        self.is_training = False
        self.last_training_time = 0.0
        self.training_interval = 3600  # Retrain every hour
        
        if model_path:
            self._load_model()
    
    async def record_access(self, 
                          embedding_key: str, 
                          vector_size: int,
                          similarity_score: Optional[float] = None,
                          cluster_id: Optional[str] = None) -> None:
        """Record an embedding access for pattern learning."""
        try:
            current_time = time.time()
            stats = self.embedding_stats[embedding_key]
            
            time_since_last = current_time - stats['last_access'] if stats['last_access'] else 0.0
            stats['access_count'] += 1
            
            # Update average interval
            if stats['access_count'] > 1:
                stats['average_interval'] = (
                    (stats['average_interval'] * (stats['access_count'] - 2) + time_since_last) 
                    / (stats['access_count'] - 1)
                )
            
            if similarity_score is not None:
                stats['similarity_scores'].append(similarity_score)
            
            stats['last_access'] = current_time
            
            # Create access pattern
            pattern = AccessPattern(
                timestamp=current_time,
                embedding_key=embedding_key,
                vector_size=vector_size,
                similarity_score=similarity_score,
                access_frequency=stats['access_count'],
                time_since_last_access=time_since_last,
                cluster_id=cluster_id
            )
            
            self.access_patterns.append(pattern)
            
            # Trigger retraining if needed
            if (current_time - self.last_training_time > self.training_interval 
                and len(self.access_patterns) > 100 
                and not self.is_training):
                asyncio.create_task(self._retrain_model())
                
        except Exception as e:
            logger.error("Error recording access pattern", 
                        embedding_key=embedding_key, error=str(e))
            raise PredictionError(f"Failed to record access: {e}")
    
    async def predict_access_needs(self, 
                                 embedding_keys: List[str],
                                 lookahead_hours: int = 4) -> List[PredictionResult]:
        """Predict which embeddings will be needed in the near future."""
        if not self.model or not self.scaler:
            raise ModelNotReadyError("Prediction model not trained yet")
        
        try:
            results = []
            current_time = time.time()
            
            for key in embedding_keys:
                stats = self.embedding_stats.get(key)
                if not stats:
                    continue
                
                # Prepare features for prediction
                avg_similarity = (
                    np.mean(list(stats['similarity_scores'])) 
                    if stats['similarity_scores'] else 0.0
                )
                
                features = np.array([[
                    1024,  # Default vector size
                    avg_similarity,
                    stats['access_count'],
                    current_time - stats['last_access'],
                    0,  # Default cluster ID
                    0   # Age (current)
                ]], dtype=np.float32)
                
                features_scaled = self.scaler.transform(features)
                
                with torch.no_grad():
                    prediction = self.model(torch.tensor(features_scaled)).item()
                
                # Calculate confidence based on historical patterns
                confidence = min(1.0, stats['access_count'] / 10.0)
                
                # Priority score combines prediction with frequency
                priority_score = prediction * confidence * np.log1p(stats['access_count'])
                
                # Decide if we should precompute
                should_precompute = (
                    prediction > self.prediction_threshold 
                    and confidence > 0.5
                    and stats['access_count'] > 2
                )
                
                results.append(PredictionResult(
                    embedding_key=key,
                    predicted_access_time=prediction,
                    confidence=confidence,
                    priority_score=priority_score,
                    should_precompute=should_precompute
                ))
            
            # Sort by priority score
            results.sort(key=lambda x: x.priority_score, reverse=True)
            
            logger.info("Generated access predictions", 
                       total_predictions=len(results),
                       precompute_recommendations=sum(1 for r in results if r.should_precompute))
            
            return results
            
        except Exception as e:
            logger.error("Error generating predictions", error=str(e))
            raise PredictionError(f"Failed to generate predictions: {e}")
    
    async def get_similar_embeddings(self, 
                                   reference_key: str, 
                                   limit: int = 10) -> List[str]:
        """Get embeddings similar to reference for predictive precomputation."""
        try:
            ref_stats = self.embedding_stats.get(reference_key)
            if not ref_stats:
                return []
            
            ref_pattern = self._get_access_pattern(reference_key)
            if not ref_pattern:
                return []
            
            similar = []
            
            for key, stats in self.embedding_stats.items():
                if key == reference_key:
                    continue
                
                pattern = self._get_access_pattern(key)
                if not pattern:
                    continue
                
                # Simple similarity based on access patterns
                similarity = self._calculate_pattern_similarity(ref_pattern, pattern)
                
                if similarity > 0.3:  # Threshold for similarity
                    similar.append((key, similarity))
            
            # Sort by similarity and return top matches
            similar.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in similar[:limit]]
            
        except Exception as e:
            logger.error("Error finding similar embeddings", 
                        reference_key=reference_key, error=str(e))
            return []
    
    def _get_access_pattern(self, embedding_key: str) -> Optional[AccessPattern]:
        """Get the most recent access pattern for an embedding."""
        for pattern in reversed(self.access_patterns):
            if pattern.embedding_key == embedding_key:
                return pattern
        return None
    
    def _calculate_pattern_similarity(self, 
                                    pattern1: AccessPattern, 
                                    pattern2: AccessPattern) -> float:
        """Calculate similarity between two access patterns."""
        try:
            # Simple similarity based on multiple factors
            factors = []
            
            # Vector size similarity
            size_sim = 1.0 - abs(pattern1.vector_size - pattern2.vector_size) / max(pattern1.vector_size, pattern2.vector_size)
            factors.append(size_sim)
            
            # Frequency similarity
            freq_sim = 1.0 - abs(pattern1.access_frequency - pattern2.access_frequency) / max(pattern1.access_frequency, pattern2.access_frequency)
            factors.append(freq_sim)
            
            # Cluster similarity (if available)
            if pattern1.cluster_id and pattern2.cluster_id:
                cluster_sim = 1.0 if pattern1.cluster_id == pattern2.cluster_id else 0.0
                factors.append(cluster_sim * 2)  # Weight cluster similarity higher
            
            return np.mean(factors) if factors else 0.0
            
        except Exception:
            return 0.0
    
    async def _retrain_model(self) -> None:
        """Retrain the prediction model with recent data."""
        if self.is_training:
            return
        
        self.is_training = True
        
        try:
            logger.info("Starting model retraining", 
                       pattern_count=len(self.access_patterns))
            
            # Filter recent patterns
            cutoff_time = time.time() - (self.training_window_hours * 3600)
            recent_patterns = [
                p for p in self.access_patterns 
                if p.timestamp > cutoff_time
            ]
            
            if len(recent_patterns) < 50:
                logger.warning("Insufficient data for retraining", 
                             pattern_count=len(recent_patterns))
                return
            
            # Create dataset
            dataset = AccessPatternDataset(recent_patterns)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Create new model
            self.model = EmbeddingPredictorModel()
            self.scaler = dataset.scaler
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            
            for epoch in range(10):  # Quick retraining
                epoch_loss = 0.0
                for features, targets in dataloader:
                    optimizer.zero_grad()
                    predictions = self.model(features).squeeze()
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                
                if epoch % 5 == 0:
                    logger.debug("Training progress", 
                               epoch=epoch, loss=epoch_loss)
            
            self.model.eval()
            self.last_training_time = time.time()
            
            # Save model if path provided
            if self.model_path:
                self._save_model()
            
            logger.info("Model retraining completed", 
                       final_loss=total_loss/10, 
                       training_samples=len(recent_patterns))
            
        except Exception as e:
            logger.error("Error during model retraining", error=str(e))
            raise TrainingError(f"Failed to retrain model: {e}")
        finally:
            self.is_training = False
    
    def _save_model(self) -> None:
        """Save the trained model and scaler."""
        try:
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'stats': dict(self.embedding_stats),
                'timestamp': time.time()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_state, f)
                
            logger.info("Model saved successfully", path=self.model_path)
            
        except Exception as e:
            logger.error("Error saving model", error=str(e))
    
    def _load_model(self) -> None:
        """Load a previously trained model."""
        try:
            with open(self.model_path, 'rb') as f:
                model_state = pickle.load(f)
            
            self.model = EmbeddingPredictorModel()
            self.model.load_state_dict(model_state['model_state_dict'])
            self.model.eval()
            
            self.scaler = model_state['scaler']
            
            # Restore stats if available
            if 'stats' in model_state:
                for key, stats in model_state['stats'].items():
                    self.embedding_stats[key] = defaultdict(lambda: {
                        'access_count': 0,
                        'last_access': 0.0,
                        'average_interval': 0.0,
                        'similarity_scores': deque(maxlen=100)
                    })
                    self.embedding_stats[key].update(stats)
            
            logger.info("Model loaded successfully", 
                       path=self.model_path,
                       model_age=time.time() - model_state.get('timestamp', 0))
            
        except FileNotFoundError:
            logger.warning("Model file not found, will train from scratch", 
                          path=self.model_path)
        except Exception as e:
            logger.error("Error loading model", error=str(e))
            raise ModelNotReadyError(f"Failed to load model: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        return {
            'total_patterns': len(self.access_patterns),
            'tracked_embeddings': len(self.embedding_stats),
            'model_trained': self.model is not None,
            'last_training_time': self.last_training_time,
            'is_training': self.is_training,
            'top_accessed_embeddings': sorted(
                [(k, v['access_count']) for k, v in self.embedding_stats.items()],
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }