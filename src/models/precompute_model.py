import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset

from ..exceptions import ModelError, PredictionError

logger = structlog.get_logger(__name__)


class UsagePattern(BaseModel):
    """Model for usage pattern data."""
    
    embedding_hash: str = Field(..., description="Hash of the embedding vector")
    access_count: int = Field(ge=0, description="Number of times accessed")
    last_access: float = Field(..., description="Unix timestamp of last access")
    access_frequency: float = Field(ge=0.0, description="Accesses per hour")
    similarity_group: int = Field(ge=0, description="Cluster ID for similar embeddings")
    vector_norm: float = Field(ge=0.0, description="L2 norm of embedding vector")
    dimension: int = Field(ge=1, description="Embedding dimension")


class UsageDataset(Dataset):
    """PyTorch dataset for usage patterns."""
    
    def __init__(self, patterns: List[UsagePattern]):
        self.patterns = patterns
        self._prepare_features()
    
    def _prepare_features(self) -> None:
        """Convert patterns to feature vectors."""
        self.features = []
        self.labels = []
        
        current_time = time.time()
        
        for pattern in self.patterns:
            # Time-based features
            hours_since_access = (current_time - pattern.last_access) / 3600.0
            
            # Feature vector: [access_count, hours_since_access, frequency, norm, dimension, group]
            feature = np.array([
                pattern.access_count,
                hours_since_access,
                pattern.access_frequency,
                pattern.vector_norm,
                pattern.dimension,
                pattern.similarity_group
            ], dtype=np.float32)
            
            # Label: probability of access in next hour (based on frequency)
            label = min(pattern.access_frequency, 1.0)
            
            self.features.append(feature)
            self.labels.append(label)
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class PrecomputeModel(nn.Module):
    """Neural network for predicting embedding access probability."""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModelTrainer:
    """Handles training and evaluation of the precompute model."""
    
    def __init__(
        self,
        model: PrecomputeModel,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.logger = structlog.get_logger(__name__).bind(component="trainer")
    
    async def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        early_stopping_patience: int = 5
    ) -> Dict[str, float]:
        """Train the model with optional validation and early stopping."""
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {"train_loss": [], "val_loss": []}
        
        self.logger.info("Starting model training", epochs=epochs)
        
        for epoch in range(epochs):
            # Training phase
            total_train_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            training_history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                val_loss = await self._validate(val_loader)
                training_history["val_loss"].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info(
                        "Early stopping triggered",
                        epoch=epoch,
                        best_val_loss=best_val_loss
                    )
                    break
            
            if epoch % 10 == 0:
                self.logger.info(
                    "Training progress",
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=val_loss
                )
        
        self.logger.info("Training completed", final_train_loss=avg_train_loss)
        return training_history
    
    async def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)
                
                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_labels)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches


class ONNXPredictor:
    """ONNX runtime predictor for production inference."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.logger = structlog.get_logger(__name__).bind(component="onnx_predictor")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model for inference."""
        try:
            if not self.model_path.exists():
                raise ModelError(f"Model file not found: {self.model_path}")
            
            # Configure ONNX runtime for CPU optimization
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers
            )
            
            # Get input name
            self.input_name = self.session.get_inputs()[0].name
            
            self.logger.info(
                "ONNX model loaded successfully",
                model_path=str(self.model_path),
                providers=providers
            )
            
        except Exception as e:
            raise ModelError(f"Failed to load ONNX model: {e}") from e
    
    async def predict_batch(
        self,
        features: np.ndarray,
        threshold: float = 0.5
    ) -> List[bool]:
        """Predict access probability for batch of features."""
        if self.session is None or self.input_name is None:
            raise PredictionError("Model not loaded")
        
        try:
            # Ensure correct input shape and type
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            features = features.astype(np.float32)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: features})
            probabilities = outputs[0].flatten()
            
            # Apply threshold to get binary predictions
            predictions = (probabilities > threshold).tolist()
            
            return predictions
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}") from e
    
    async def predict_single(
        self,
        pattern: UsagePattern,
        threshold: float = 0.5
    ) -> bool:
        """Predict if single embedding should be precomputed."""
        current_time = time.time()
        hours_since_access = (current_time - pattern.last_access) / 3600.0
        
        features = np.array([[
            pattern.access_count,
            hours_since_access,
            pattern.access_frequency,
            pattern.vector_norm,
            pattern.dimension,
            pattern.similarity_group
        ]], dtype=np.float32)
        
        predictions = await self.predict_batch(features, threshold)
        return predictions[0]


class ModelManager:
    """Manages model lifecycle: training, export, and serving."""
    
    def __init__(
        self,
        model_dir: Path,
        model_name: str = "precompute_model",
        device: str = "cpu"
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.device = device
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__).bind(component="model_manager")
        
        # File paths
        self.pytorch_path = self.model_dir / f"{model_name}.pth"
        self.onnx_path = self.model_dir / f"{model_name}.onnx"
        self.metadata_path = self.model_dir / f"{model_name}_metadata.pkl"
    
    async def train_and_export(
        self,
        usage_patterns: List[UsagePattern],
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50
    ) -> ONNXPredictor:
        """Complete training pipeline: train -> export -> load predictor."""
        if not usage_patterns:
            raise ModelError("No usage patterns provided for training")
        
        self.logger.info(
            "Starting training pipeline",
            num_patterns=len(usage_patterns),
            validation_split=validation_split
        )
        
        # Prepare datasets
        dataset = UsageDataset(usage_patterns)
        
        # Train/validation split
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_size > 0 else None
        
        # Initialize and train model
        model = PrecomputeModel()
        trainer = ModelTrainer(model, device=self.device)
        
        training_history = await trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs
        )
        
        # Save PyTorch model
        torch.save(model.state_dict(), self.pytorch_path)
        
        # Export to ONNX
        await self._export_to_onnx(model)
        
        # Save training metadata
        metadata = {
            "training_patterns": len(usage_patterns),
            "training_history": training_history,
            "model_architecture": {
                "input_size": 6,
                "hidden_size": 64
            },
            "export_timestamp": time.time()
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info("Training and export completed successfully")
        
        # Return ready-to-use predictor
        return ONNXPredictor(self.onnx_path)
    
    async def _export_to_onnx(self, model: PrecomputeModel) -> None:
        """Export PyTorch model to ONNX format."""
        model.eval()
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, 6, dtype=torch.float32)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(self.onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['features'],
                output_names=['probability'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'probability': {0: 'batch_size'}
                }
            )
            
            self.logger.info("Model exported to ONNX", onnx_path=str(self.onnx_path))
            
        except Exception as e:
            raise ModelError(f"ONNX export failed: {e}") from e
    
    def load_predictor(self) -> ONNXPredictor:
        """Load existing ONNX predictor."""
        if not self.onnx_path.exists():
            raise ModelError(f"No trained model found at {self.onnx_path}")
        
        return ONNXPredictor(self.onnx_path)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if not self.metadata_path.exists():
            return {"status": "no_model"}
        
        try:
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            return {
                "status": "ready",
                "model_path": str(self.onnx_path),
                "pytorch_path": str(self.pytorch_path),
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error("Failed to load model metadata", error=str(e))
            return {"status": "error", "error": str(e)}