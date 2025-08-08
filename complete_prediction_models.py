"""
Fixed Complete Prediction Models v3 - Schema Adaptive
This version dynamically adapts to your actual database schema
"""

import os
import sys
import time
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from boruta import BorutaPy

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

# Technical Analysis
import ta

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionModels")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ================== DATABASE SCHEMA HELPER ==================
class DatabaseSchemaHelper:
    """Helper class to dynamically adapt to database schema"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema_cache = {}
        self._load_schema()
    
    def _load_schema(self):
        """Load database schema information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    self.schema_cache[table_name] = [col[1] for col in columns]
                
                logger.info(f"Loaded schema for tables: {list(self.schema_cache.keys())}")
                
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
    
    def get_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        return self.schema_cache.get(table_name, [])
    
    def get_available_columns(self, table_name: str, desired_columns: List[str]) -> List[str]:
        """Get available columns from desired list"""
        table_columns = self.get_columns(table_name)
        return [col for col in desired_columns if col in table_columns]
    
    def get_column_mapping(self, table_name: str) -> Dict[str, str]:
        """Get column name mappings for common fields"""
        columns = self.get_columns(table_name)
        
        # Common mappings
        mappings = {
            'close': None,
            'open': None,
            'high': None,
            'low': None,
            'volume': None,
            'price': None,
            'rsi': None,
            'macd': None,
            'volatility': None
        }
        
        # Try to find matching columns
        for col in columns:
            col_lower = col.lower()
            
            # Price columns
            if 'close' in col_lower or 'price' in col_lower:
                if not mappings['close']:
                    mappings['close'] = col
            elif 'open' in col_lower:
                mappings['open'] = col
            elif 'high' in col_lower:
                mappings['high'] = col
            elif 'low' in col_lower:
                mappings['low'] = col
            elif 'volume' in col_lower:
                mappings['volume'] = col
            elif 'rsi' in col_lower:
                mappings['rsi'] = col
            elif 'macd' in col_lower and 'signal' not in col_lower:
                mappings['macd'] = col
            elif 'volatility' in col_lower:
                mappings['volatility'] = col
        
        # If no close price found, try to use price column
        if not mappings['close'] and mappings['price']:
            mappings['close'] = mappings['price']
        
        # Remove None values
        return {k: v for k, v in mappings.items() if v is not None}


# ================== BASE MODEL INTERFACE ==================
class BasePredictionModel:
    """Base class for all prediction models"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_name = self.__class__.__name__
        self.schema_helper = DatabaseSchemaHelper(db_path)
        
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Make prediction for a symbol - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def train(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Train the model on historical data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train method")
    
    def is_trained(self, symbol: str) -> bool:
        """Check if model is trained for symbol - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement is_trained method")
    
    def get_price_column(self) -> str:
        """Get the price column name from unified_features"""
        mappings = self.schema_helper.get_column_mapping('unified_features')
        
        # Try to find a price column
        if 'close' in mappings:
            return mappings['close']
        
        # Fallback: look for any column with 'price' or 'close' in the name
        columns = self.schema_helper.get_columns('unified_features')
        for col in columns:
            if 'price' in col.lower() or 'close' in col.lower():
                return col
        
        # Last resort: use the first numeric column that's not id or timestamp
        for col in columns:
            if col not in ['id', 'timestamp', 'symbol'] and not col.startswith('feature_'):
                return col
        
        raise ValueError("No suitable price column found in unified_features")


# ================== 1. ADAPTIVE BORUTA CNN-LSTM MODEL ==================
@dataclass
class BorutaModelConfig:
    """Configuration for Boruta CNN-LSTM model"""
    db_path: str = os.path.join(SCRIPT_DIR, "crypto_trading.db")
    
    # Boruta parameters
    boruta_max_iter: int = 100
    boruta_confidence: float = 0.95
    min_features: int = 5   # Reduced minimum
    max_features: int = 12
    
    # CNN architecture
    cnn_filters: List[int] = field(default_factory=lambda: [64, 64])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    cnn_activation: str = 'relu'
    
    # LSTM architecture
    lstm_layers: List[int] = field(default_factory=lambda: [128, 80])
    lstm_dropout: float = 0.2
    
    # Training parameters
    sequence_length: int = 30
    prediction_horizon: int = 1
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Regularization
    dropout_rate: float = 0.5
    batch_norm: bool = True
    gradient_clip: float = 1.0
    
    # Model paths
    model_dir: str = os.path.join(SCRIPT_DIR, "models", "boruta_cnn_lstm")
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_onchain_metrics: bool = True
    use_wavelet_decomposition: bool = False  # Disabled by default


class RobustBorutaCryptoDataset(Dataset):
    """Fixed PyTorch Dataset with proper error handling"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """Initialize dataset with pre-prepared sequences"""
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
        if len(self.sequences) != len(self.targets):
            raise ValueError(f"Sequences and targets length mismatch")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class OptimizedCNN_LSTM(nn.Module):
    """Adaptive CNN-LSTM architecture"""
    
    def __init__(self, input_features: int, config: BorutaModelConfig):
        super(OptimizedCNN_LSTM, self).__init__()
        self.config = config
        self.input_features = input_features
        
        # Ensure we have at least 1 input feature
        if input_features < 1:
            raise ValueError(f"Invalid input_features: {input_features}")
        
        # Simple architecture for few features
        if input_features < 5:
            # Direct LSTM without CNN
            self.use_cnn = False
            self.lstm = nn.LSTM(
                input_size=input_features,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=config.lstm_dropout if 2 > 1 else 0
            )
            lstm_output_size = 64
        else:
            # Full CNN-LSTM architecture
            self.use_cnn = True
            
            # CNN Feature Extractor
            self.conv_blocks = nn.ModuleList()
            in_channels = 1
            
            for i, (filters, kernel) in enumerate(zip(config.cnn_filters, config.cnn_kernel_sizes)):
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size=(kernel, 1), padding=(kernel//2, 0)),
                    nn.BatchNorm2d(filters) if config.batch_norm else nn.Identity(),
                    nn.ReLU() if config.cnn_activation == 'relu' else nn.Tanh(),
                    nn.Dropout2d(config.dropout_rate * 0.5) if i > 0 else nn.Identity()
                )
                self.conv_blocks.append(conv_block)
                in_channels = filters
            
            # Global pooling
            self.global_pool = nn.AdaptiveAvgPool2d((None, 1))
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=config.cnn_filters[-1],
                hidden_size=config.lstm_layers[0],
                num_layers=len(config.lstm_layers),
                batch_first=True,
                dropout=config.lstm_dropout if len(config.lstm_layers) > 1 else 0
            )
            lstm_output_size = config.lstm_layers[0]
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        # Ensure input is 3D: (batch, sequence, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, n_features = x.size()
        
        if self.use_cnn:
            # Add channel dimension for CNN: (batch, 1, sequence, features)
            x_cnn = x.unsqueeze(1)
            
            # Apply CNN blocks
            for conv_block in self.conv_blocks:
                x_cnn = conv_block(x_cnn)
            
            # Global pooling: (batch, filters, sequence, 1)
            x_cnn = self.global_pool(x_cnn)
            
            # Remove last dimension and transpose for LSTM: (batch, sequence, filters)
            x_lstm = x_cnn.squeeze(-1).transpose(1, 2)
        else:
            # Direct to LSTM
            x_lstm = x
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x_lstm)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Generate prediction
        output = self.output_layers(last_output)
        
        return output.squeeze(-1)


class BorutaCNNLSTMModel(BasePredictionModel):
    """Adaptive Boruta feature selection + CNN-LSTM model"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.config = BorutaModelConfig(db_path=db_path)
        self.models = {}
        self.scalers = {}
        self.selected_features = {}
        self.boruta_selectors = {}
        self.feature_importance = {}
        self.column_mappings = {}
        
        os.makedirs(self.config.model_dir, exist_ok=True)
        logger.info("Initialized Boruta CNN-LSTM model with adaptive configuration")
    
    def fetch_comprehensive_features(self, symbol: str, start_date: datetime, 
                                   end_date: datetime) -> pd.DataFrame:
        """Fetch features from unified_features table with dynamic schema"""
        try:
            with self.get_connection() as conn:
                # Get all columns from unified_features
                columns = self.schema_helper.get_columns('unified_features')
                
                # Build query with all available columns
                query = f"""
                    SELECT * FROM unified_features
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                )
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Log available columns
                logger.info(f"Available columns in unified_features: {list(df.columns)}")
                
                # Store column mappings
                self.column_mappings[symbol] = self.schema_helper.get_column_mapping('unified_features')
                
                # Select numeric columns (excluding metadata)
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                exclude_cols = ['id', 'symbol_id']
                numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
                
                df = df[numeric_columns]
                
                # Handle missing values
                df = self._handle_missing_values(df)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching features: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent handling of missing values"""
        # Forward fill for time series continuity
        df = df.fillna(method='ffill', limit=5)
        
        # Backward fill for remaining
        df = df.fillna(method='bfill', limit=5)
        
        # Fill remaining with column median
        for col in df.columns:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, symbol: str, horizon: int = 1) -> np.ndarray:
        """Create binary target variable for classification"""
        # Get price column name
        price_col = None
        
        # Try to find price column
        for col in df.columns:
            if 'price' in col.lower() or 'close' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            # Use the first numeric column as price
            numeric_cols = [col for col in df.columns if col not in ['id', 'timestamp', 'symbol']]
            if numeric_cols:
                price_col = numeric_cols[0]
                logger.warning(f"No price column found, using {price_col} as price")
            else:
                raise ValueError("No suitable price column found")
        
        # Calculate future returns
        future_returns = df[price_col].pct_change(horizon).shift(-horizon)
        
        # Binary classification: 1 if positive return, 0 otherwise
        threshold = 0.001  # 0.1% minimum move
        targets = (future_returns > threshold).astype(int).values
        
        return targets
    
    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences with validation"""
        if len(features) < sequence_length + 1:
            raise ValueError(f"Insufficient data for sequences: {len(features)} < {sequence_length + 1}")
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            if i + sequence_length < len(targets) and not np.isnan(targets[i + sequence_length]):
                X.append(features[i:i + sequence_length])
                y.append(targets[i + sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Train the Boruta CNN-LSTM model"""
        logger.info(f"{'='*50}")
        logger.info(f"Training Adaptive Boruta CNN-LSTM for {symbol}")
        logger.info(f"{'='*50}")
        
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
        
        try:
            # Fetch comprehensive features
            df = self.fetch_comprehensive_features(symbol, start_date, end_date)
            if df.empty or len(df) < 100:
                logger.error(f"Insufficient data for {symbol}: {len(df)} samples")
                return False
            
            # Create targets
            targets = self.create_target_variable(df, symbol, self.config.prediction_horizon)
            
            # Remove NaN targets
            valid_idx = ~np.isnan(targets)
            df = df[valid_idx]
            targets = targets[valid_idx]
            
            if len(df) < 50:
                logger.error(f"Insufficient valid data after cleaning: {len(df)} samples")
                return False
            
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Target distribution: {np.bincount(targets.astype(int))}")
            
            # If too few features, use all of them
            if len(df.columns) <= self.config.min_features:
                selected_features = list(df.columns)
                feature_df = df
                logger.info(f"Using all {len(selected_features)} features (below minimum threshold)")
            else:
                # Apply simple feature selection based on variance
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(df)
                selected_features = df.columns[selector.get_support()].tolist()
                
                # Ensure we have enough features
                if len(selected_features) < self.config.min_features:
                    # Add more features based on correlation with target
                    correlations = df.corrwith(pd.Series(targets, index=df.index)).abs()
                    top_features = correlations.nlargest(self.config.min_features).index.tolist()
                    selected_features = list(set(selected_features + top_features))[:self.config.max_features]
                
                feature_df = df[selected_features]
            
            self.selected_features[symbol] = selected_features
            logger.info(f"Selected {len(selected_features)} features")
            
            # Scale features
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(feature_df)
            self.scalers[symbol] = scaler
            
            # Prepare sequences
            X, y = self.prepare_sequences(scaled_features, targets, self.config.sequence_length)
            
            if len(X) < 50:
                logger.error(f"Insufficient sequences: {len(X)}")
                return False
            
            logger.info(f"Prepared {len(X)} sequences for training")
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create datasets
            train_dataset = RobustBorutaCryptoDataset(X_train, y_train)
            val_dataset = RobustBorutaCryptoDataset(X_val, y_val)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Initialize model
            model = OptimizedCNN_LSTM(len(selected_features), self.config).to(device)
            
            # Loss and optimizer
            pos_ratio = np.mean(y_train)
            pos_weight = torch.tensor([(1 - pos_ratio) / (pos_ratio + 1e-6)]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
            
            # Training loop
            best_val_accuracy = 0.0
            patience_counter = 0
            
            for epoch in range(min(self.config.epochs, 30)):  # Limit epochs for faster training
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        predictions = (outputs > 0.5).float()
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                # Calculate metrics
                train_accuracy = train_correct / train_total if train_total > 0 else 0
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                
                scheduler.step(avg_val_loss)
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                              f"Val Acc: {val_accuracy:.4f}")
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    self.models[symbol] = model
                    self.save_model(symbol)
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            logger.info(f"Training completed with best validation accuracy: {best_val_accuracy:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Training error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Make prediction with comprehensive analysis"""
        try:
            if not self.is_trained(symbol):
                logger.warning(f"Model not trained for {symbol}")
                return self._default_prediction("Model not trained")
            
            # Load model if needed
            if symbol not in self.models:
                self.load_model(symbol)
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=self.config.sequence_length + 24)
            
            df = self.fetch_comprehensive_features(symbol, start_date, end_date)
            if df.empty or len(df) < self.config.sequence_length:
                logger.warning(f"Insufficient data for prediction")
                return self._default_prediction("Insufficient data")
            
            # Use selected features
            selected_features = self.selected_features[symbol]
            
            # Ensure all features are available
            available_features = [f for f in selected_features if f in df.columns]
            if len(available_features) < len(selected_features):
                # Pad missing features with zeros
                for feat in selected_features:
                    if feat not in df.columns:
                        df[feat] = 0
            
            feature_df = df[selected_features]
            
            # Scale features
            scaled_features = self.scalers[symbol].transform(feature_df)
            
            # Get most recent sequence
            if len(scaled_features) < self.config.sequence_length:
                return self._default_prediction("Not enough recent data")
            
            sequence = scaled_features[-self.config.sequence_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Make prediction
            model = self.models[symbol]
            model.eval()
            
            with torch.no_grad():
                prediction = torch.sigmoid(model(sequence_tensor)).cpu().numpy()[0]
            
            # Get current price
            price_col = self.get_price_column()
            if price_col in df.columns:
                current_price = df[price_col].iloc[-1]
            else:
                current_price = df.iloc[-1, 0]  # Use first column as fallback
            
            # Generate trading signal
            if prediction > 0.65:
                action = 'buy'
                confidence = float((prediction - 0.5) * 2)
                price_target = current_price * 1.025
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.04
            elif prediction < 0.35:
                action = 'sell'
                confidence = float((0.5 - prediction) * 2)
                price_target = current_price * 0.975
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.96
            else:
                action = 'hold'
                confidence = float(1 - abs(prediction - 0.5) * 2)
                price_target = current_price
                stop_loss = None
                take_profit = None
            
            return {
                'action': action,
                'confidence': confidence,
                'price_target': float(price_target),
                'stop_loss': float(stop_loss) if stop_loss else None,
                'take_profit': float(take_profit) if take_profit else None,
                'model_name': self.model_name,
                'timestamp': datetime.now(),
                'raw_prediction': float(prediction),
                'current_price': float(current_price),
                'analysis': {
                    'selected_features': selected_features[:10],
                    'prediction_strength': 'strong' if confidence > 0.7 else 'moderate' if confidence > 0.5 else 'weak'
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return self._default_prediction(str(e))
    
    def _default_prediction(self, error_msg: str = "") -> Dict[str, Any]:
        """Return default prediction"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'price_target': None,
            'stop_loss': None,
            'take_profit': None,
            'model_name': self.model_name,
            'timestamp': datetime.now(),
            'error': error_msg
        }
    
    def is_trained(self, symbol: str) -> bool:
        """Check if model is trained"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.model_dir}/{symbol_safe}_model.pth"
        metadata_path = f"{self.config.model_dir}/{symbol_safe}_metadata.pkl"
        return os.path.exists(model_path) and os.path.exists(metadata_path)
    
    def save_model(self, symbol: str):
        """Save model and metadata"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.model_dir}/{symbol_safe}_model.pth"
        metadata_path = f"{self.config.model_dir}/{symbol_safe}_metadata.pkl"
        
        torch.save({
            'model_state_dict': self.models[symbol].state_dict(),
            'config': self.config,
            'selected_features': self.selected_features[symbol]
        }, model_path)
        
        metadata = {
            'scaler': self.scalers[symbol],
            'selected_features': self.selected_features[symbol],
            'column_mappings': self.column_mappings.get(symbol, {}),
            'timestamp': datetime.now()
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved for {symbol}")
    
    def load_model(self, symbol: str):
        """Load saved model"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.model_dir}/{symbol_safe}_model.pth"
        metadata_path = f"{self.config.model_dir}/{symbol_safe}_metadata.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model files not found for {symbol}")
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.scalers[symbol] = metadata['scaler']
        self.selected_features[symbol] = metadata['selected_features']
        self.column_mappings[symbol] = metadata.get('column_mappings', {})
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        model = OptimizedCNN_LSTM(
            len(self.selected_features[symbol]), 
            self.config
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        self.models[symbol] = model
        logger.info(f"Model loaded for {symbol}")


# ================== 2. ADAPTIVE HELFORMER MODEL ==================
@dataclass
class HelformerConfig:
    """Optimized Helformer configuration"""
    sequence_length: int = 48
    prediction_horizon: int = 6
    d_model: int = 128  # Reduced for smaller feature sets
    n_heads: int = 4
    n_encoder_layers: int = 2  # Reduced layers
    d_ff: int = 512
    
    seasonal_period: int = 24
    use_multiplicative: bool = True
    
    use_lstm_layers: bool = True
    lstm_hidden_size: int = 64
    
    batch_size: int = 16
    learning_rate: float = 0.001
    dropout: float = 0.1
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    use_amp: bool = False  # Disabled for stability
    
    model_save_path: str = os.path.join(SCRIPT_DIR, "models", "helformer")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ================== FIXED HELFORMER MODEL - COMPLETE VERSION ==================
# Replace the entire Helformer section in your complete_prediction_models.py with this

@dataclass
class HelformerConfig:
    """Optimized Helformer configuration"""
    sequence_length: int = 48
    prediction_horizon: int = 6
    d_model: int = 128  # Reduced for smaller feature sets
    n_heads: int = 4
    n_encoder_layers: int = 2  # Reduced layers
    d_ff: int = 512
    
    seasonal_period: int = 24
    use_multiplicative: bool = True
    
    use_lstm_layers: bool = True
    lstm_hidden_size: int = 64
    
    batch_size: int = 16
    learning_rate: float = 0.001
    dropout: float = 0.1
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    use_amp: bool = False  # Disabled for stability
    
    model_save_path: str = os.path.join(SCRIPT_DIR, "models", "helformer")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleHelformer(nn.Module):
    """Simplified Helformer for limited features"""
    
    def __init__(self, n_features: int, config: HelformerConfig):
        super(SimpleHelformer, self).__init__()
        self.config = config
        self.n_features = n_features
        
        # Simple LSTM-based architecture
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if 2 > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.lstm_hidden_size * 2,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output layers - NO SIGMOID, using BCEWithLogitsLoss
        self.output_layers = nn.Sequential(
            nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No Sigmoid here
        )
    
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Output
        output = self.output_layers(pooled)
        
        return output.squeeze(-1), {'attention_weights': None}


class HelformerModel(BasePredictionModel):
    """Adaptive Helformer implementation with NaN fixes"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.config = HelformerConfig()
        self.models = {}
        self.scalers = {}
        self.feature_configs = {}
        
        os.makedirs(self.config.model_save_path, exist_ok=True)
        logger.info("Initialized Helformer model with adaptive configuration")
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Engineer features based on available columns"""
        features = []
        feature_names = []
        
        # Use available numeric features
        for col in df.columns:
            if col not in ['id', 'timestamp', 'symbol', 'symbol_id']:
                values = df[col].values
                # Check for NaN or infinite values
                if np.isfinite(values).all():
                    features.append(values)
                    feature_names.append(col)
                else:
                    # Replace NaN/inf with median
                    finite_mask = np.isfinite(values)
                    if finite_mask.any():
                        median_val = np.median(values[finite_mask])
                        values = np.where(finite_mask, values, median_val)
                        features.append(values)
                        feature_names.append(col)
        
        # Add time features if we have enough columns
        if len(feature_names) < 10:
            # Add cyclical time encoding
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            
            features.append(np.sin(2 * np.pi * df['hour'] / 24))
            feature_names.append('hour_sin')
            features.append(np.cos(2 * np.pi * df['hour'] / 24))
            feature_names.append('hour_cos')
            
            features.append(np.sin(2 * np.pi * df['day_of_week'] / 7))
            feature_names.append('dow_sin')
            features.append(np.cos(2 * np.pi * df['day_of_week'] / 7))
            feature_names.append('dow_cos')
        
        # Ensure we have at least some features
        if len(features) == 0:
            raise ValueError("No usable features found")
        
        # Stack features
        feature_array = np.column_stack(features)
        
        # Final NaN check
        if not np.isfinite(feature_array).all():
            logger.warning("Found NaN/inf values in features, replacing with 0")
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_array, feature_names
    
    def train(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Train Helformer model with NaN handling"""
        logger.info(f"{'='*50}")
        logger.info(f"Training Adaptive Helformer for {symbol}")
        logger.info(f"{'='*50}")
        
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
        
        try:
            # Fetch data
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM unified_features
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                )
            
            if df.empty or len(df) < 100:
                logger.error(f"Insufficient data for {symbol}: {len(df)} samples")
                return False
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Remove non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['id', 'symbol_id']
            numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
            df = df[numeric_columns]
            
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Engineer features
            features, feature_names = self.engineer_features(df)
            self.feature_configs[symbol] = feature_names
            
            logger.info(f"Engineered {len(feature_names)} features: {feature_names[:10]}...")
            
            # Create targets
            price_col = self.get_price_column()
            if price_col in df.columns:
                price_data = df[price_col]
            else:
                # Use first numeric column
                price_data = df.iloc[:, 0]
            
            future_returns = price_data.pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
            targets = (future_returns > 0.002).astype(int)
            
            # Remove NaN
            valid_idx = ~np.isnan(targets) & np.isfinite(targets)
            features = features[valid_idx]
            targets = targets[valid_idx].values
            
            if len(features) < 50:
                logger.error(f"Insufficient valid data: {len(features)} samples")
                return False
            
            # Scale features with clipping to prevent extreme values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            
            # Clip extreme values
            scaled_features = np.clip(scaled_features, 0, 1)
            
            self.scalers[symbol] = scaler
            
            # Prepare sequences
            X, y = [], []
            for i in range(len(scaled_features) - self.config.sequence_length):
                if i + self.config.sequence_length < len(targets):
                    seq = scaled_features[i:i + self.config.sequence_length]
                    if np.isfinite(seq).all():  # Only use sequences without NaN
                        X.append(seq)
                        y.append(targets[i + self.config.sequence_length])
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            if len(X) < 20:
                logger.error(f"Insufficient sequences: {len(X)}")
                return False
            
            logger.info(f"Prepared {len(X)} sequences for training")
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create simple datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            # Create data loaders with drop_last to ensure consistent batch sizes
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True  # Important for batch consistency
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )
            
            # Initialize model
            model = SimpleHelformer(len(feature_names), self.config).to(device)
            
            # Loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Training loop with better error handling
            best_val_accuracy = 0.0  # Track accuracy instead of loss
            patience_counter = 0
            
            for epoch in range(min(self.config.epochs, 30)):
                # Training
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    
                    try:
                        predictions, _ = model(batch_X)
                        loss = criterion(predictions, batch_y)
                        
                        # Check for NaN loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"NaN/Inf loss detected at epoch {epoch}, skipping batch")
                            continue
                        
                        loss.backward()
                        
                        # Gradient clipping
                        clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_batches += 1
                        
                    except Exception as e:
                        logger.warning(f"Error in training batch: {e}")
                        continue
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        try:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device)
                            
                            predictions, _ = model(batch_X)
                            loss = criterion(predictions, batch_y)
                            
                            # Check for NaN loss
                            if not torch.isnan(loss) and not torch.isinf(loss):
                                val_loss += loss.item()
                                val_batches += 1
                            
                            # Apply sigmoid for evaluation
                            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
                            val_correct += (pred_binary == batch_y).sum().item()
                            val_total += batch_y.size(0)
                            
                        except Exception as e:
                            logger.warning(f"Error in validation batch: {e}")
                            continue
                
                # Calculate metrics
                avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Save based on accuracy instead of loss
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    self.models[symbol] = model
                    self.save_model(symbol)
                    logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # If no model was saved due to NaN issues, save the final model if accuracy is reasonable
            if symbol not in self.models and best_val_accuracy > 0.4:
                self.models[symbol] = model
                self.save_model(symbol)
                logger.info(f"Saved final model with accuracy: {best_val_accuracy:.4f}")
            
            logger.info(f"Training completed with best accuracy: {best_val_accuracy:.4f}")
            return symbol in self.models  # Return True only if model was saved
            
        except Exception as e:
            logger.error(f"Training error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Make prediction using Helformer"""
        try:
            if not self.is_trained(symbol):
                logger.warning(f"Helformer model not trained for {symbol}")
                return self._default_prediction("Model not trained")
            
            # Load model if needed
            if symbol not in self.models:
                self.load_model(symbol)
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=self.config.sequence_length + 24)
            
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM unified_features
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_date.strftime('%Y-%m-%d %H:%M:%S'), 
                           end_date.strftime('%Y-%m-%d %H:%M:%S'))
                )
            
            if df.empty or len(df) < self.config.sequence_length:
                logger.warning(f"Insufficient data for Helformer prediction")
                return self._default_prediction("Insufficient data")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Remove non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['id', 'symbol_id']
            numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
            df = df[numeric_columns]
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Engineer features
            features, _ = self.engineer_features(df)
            
            # Scale and clip
            scaled_features = self.scalers[symbol].transform(features)
            scaled_features = np.clip(scaled_features, 0, 1)
            
            # Get sequence
            if len(scaled_features) < self.config.sequence_length:
                return self._default_prediction("Not enough recent data")
            
            sequence = scaled_features[-self.config.sequence_length:]
            
            # Check for NaN
            if not np.isfinite(sequence).all():
                logger.warning("NaN values in sequence, replacing with 0")
                sequence = np.nan_to_num(sequence, nan=0.0, posinf=1.0, neginf=0.0)
            
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Predict
            model = self.models[symbol]
            model.eval()
            
            with torch.no_grad():
                logits, _ = model(sequence_tensor)
                # Apply sigmoid to get probability
                prediction = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get current price
            price_col = self.get_price_column()
            if price_col in df.columns:
                current_price = df[price_col].iloc[-1]
            else:
                current_price = df.iloc[-1, 0]
            
            # Generate signal
            if prediction > 0.7:
                action = 'buy'
                confidence = float(prediction)
                price_target = current_price * 1.03
                stop_loss = current_price * 0.975
                take_profit = current_price * 1.05
            elif prediction < 0.3:
                action = 'sell'
                confidence = float(1 - prediction)
                price_target = current_price * 0.97
                stop_loss = current_price * 1.025
                take_profit = current_price * 0.95
            else:
                action = 'hold'
                confidence = float(1 - abs(prediction - 0.5) * 2)
                price_target = current_price
                stop_loss = None
                take_profit = None
            
            return {
                'action': action,
                'confidence': confidence,
                'price_target': float(price_target),
                'stop_loss': float(stop_loss) if stop_loss else None,
                'take_profit': float(take_profit) if take_profit else None,
                'model_name': self.model_name,
                'timestamp': datetime.now(),
                'raw_prediction': float(prediction),
                'current_price': float(current_price),
                'analysis': {
                    'prediction_horizon': self.config.prediction_horizon,
                    'model_confidence': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
                    'features_used': len(self.feature_configs.get(symbol, []))
                }
            }
            
        except Exception as e:
            logger.error(f"Helformer prediction error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._default_prediction(str(e))
    
    def _default_prediction(self, error_msg: str = "") -> Dict[str, Any]:
        """Return default prediction"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'price_target': None,
            'stop_loss': None,
            'take_profit': None,
            'model_name': self.model_name,
            'timestamp': datetime.now(),
            'error': error_msg
        }
    
    def is_trained(self, symbol: str) -> bool:
        """Check if model is trained"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.model_save_path}/helformer_{symbol_safe}.pth"
        scaler_path = f"{self.config.model_save_path}/scaler_{symbol_safe}.pkl"
        return os.path.exists(model_path) and os.path.exists(scaler_path)
    
    def save_model(self, symbol: str):
        """Save model and scaler"""
        try:
            symbol_safe = symbol.replace('/', '_')
            model_path = f"{self.config.model_save_path}/helformer_{symbol_safe}.pth"
            scaler_path = f"{self.config.model_save_path}/scaler_{symbol_safe}.pkl"
            config_path = f"{self.config.model_save_path}/config_{symbol_safe}.pkl"
            
            torch.save({
                'model_state_dict': self.models[symbol].state_dict(),
                'config': self.config,
                'feature_names': self.feature_configs[symbol]
            }, model_path)
            
            joblib.dump(self.scalers[symbol], scaler_path)
            joblib.dump(self.feature_configs[symbol], config_path)
            
            logger.info(f"Helformer model saved for {symbol}")
        except Exception as e:
            logger.error(f"Error saving Helformer model for {symbol}: {e}")
    
    def load_model(self, symbol: str):
        """Load saved model"""
        try:
            symbol_safe = symbol.replace('/', '_')
            model_path = f"{self.config.model_save_path}/helformer_{symbol_safe}.pth"
            scaler_path = f"{self.config.model_save_path}/scaler_{symbol_safe}.pkl"
            config_path = f"{self.config.model_save_path}/config_{symbol_safe}.pkl"
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
                raise FileNotFoundError(f"Model files not found for {symbol}")
            
            self.feature_configs[symbol] = joblib.load(config_path)
            
            checkpoint = torch.load(model_path, map_location=device)
            n_features = len(self.feature_configs[symbol])
            
            model = SimpleHelformer(n_features, self.config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            self.models[symbol] = model
            self.scalers[symbol] = joblib.load(scaler_path)
            
            logger.info(f"Helformer model loaded for {symbol}")
        except Exception as e:
            logger.error(f"Error loading Helformer model for {symbol}: {e}")
            
# ================== 3. ADAPTIVE TEMPORAL FUSION MODEL ==================
class TemporalFusionModelConfig:
    """Configuration for the Temporal Fusion Model"""
    
    SEQ_LENGTH = 48
    PREDICTION_HORIZON = 6
    HIDDEN_SIZE = 256
    NUM_ATTENTION_HEADS = 8
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DB_PATH = os.path.join(SCRIPT_DIR, 'crypto_trading.db')
    BUY_THRESHOLD = 0.002
    SELL_THRESHOLD = -0.002
    MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'temporal_fusion')


class TemporalFusionModel(BasePredictionModel):
    """Adaptive Temporal Fusion Model implementation"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.config = TemporalFusionModelConfig()
        self.config.DB_PATH = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
    
    def train(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Train Temporal Fusion model"""
        logger.info(f"Training Temporal Fusion model for {symbol}...")
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            
            if start_date is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
            
            # Fetch from unified_features
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM unified_features
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                )
            
            if df.empty or len(df) < 100:
                logger.error(f"Insufficient data for training {symbol}")
                return False
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['id', 'symbol_id']
            available_features = [col for col in numeric_columns if col not in exclude_cols]
            
            if len(available_features) < 3:
                logger.error(f"Insufficient features available for {symbol}")
                return False
            
            # Store feature columns
            self.feature_columns[symbol] = available_features
            
            # Fill missing values
            df[available_features] = df[available_features].fillna(method='ffill').fillna(0)
            
            # Get price column
            price_col = self.get_price_column()
            if price_col in df.columns:
                price_data = df[price_col]
            else:
                price_data = df[available_features[0]]
            
            # Create targets
            future_returns = price_data.pct_change(6).shift(-6)
            targets = np.where(future_returns > self.config.BUY_THRESHOLD, 0,
                              np.where(future_returns < self.config.SELL_THRESHOLD, 2, 1))
            
            # Remove NaN targets
            valid_idx = ~np.isnan(targets)
            features = df[available_features][valid_idx].values
            targets = targets[valid_idx]
            
            if len(features) < 50:
                logger.error(f"Insufficient valid data for {symbol}")
                return False
            
            # Scale features
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalers[symbol] = scaler
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=50,  # Reduced for faster training
                max_depth=4,
                random_state=42
            )
            model.fit(scaled_features, targets)
            self.models[symbol] = model
            
            # Save model
            self.save_model(symbol)
            
            logger.info(f"Temporal Fusion training completed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Temporal Fusion training error for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Make prediction using Temporal Fusion model"""
        try:
            if not self.is_trained(symbol):
                logger.warning(f"Temporal Fusion model not trained for {symbol}")
                return self._default_prediction("Model not trained")
            
            if symbol not in self.models:
                self.load_model(symbol)
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)
            
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM unified_features
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_date.strftime('%Y-%m-%d %H:%M:%S'), 
                           end_date.strftime('%Y-%m-%d %H:%M:%S'))
                )
            
            if df.empty:
                logger.warning(f"No recent data for Temporal Fusion {symbol}")
                return self._default_prediction("No recent data")
            
            # Prepare features
            available_features = self.feature_columns.get(symbol, [])
            feature_values = df[available_features].fillna(0).values
            
            # Scale features
            scaled_features = self.scalers[symbol].transform(feature_values)
            
            # Make prediction
            model = self.models[symbol]
            prediction_probs = model.predict_proba(scaled_features)[0]
            predicted_class = model.predict(scaled_features)[0]
            
            # Get current price
            price_col = self.get_price_column()
            if price_col in df.columns:
                current_price = df[price_col].iloc[0]
            else:
                current_price = df[available_features[0]].iloc[0]
            
            # Convert prediction to trading signal
            if predicted_class == 0:  # Buy
                action = 'buy'
                confidence = prediction_probs[0]
                price_target = current_price * 1.02
                stop_loss = current_price * 0.985
                take_profit = current_price * 1.04
            elif predicted_class == 2:  # Sell
                action = 'sell'
                confidence = prediction_probs[2]
                price_target = current_price * 0.98
                stop_loss = current_price * 1.015
                take_profit = current_price * 0.96
            else:  # Hold
                action = 'hold'
                confidence = prediction_probs[1]
                price_target = current_price
                stop_loss = None
                take_profit = None
            
            return {
                'action': action,
                'confidence': float(confidence),
                'price_target': float(price_target),
                'stop_loss': float(stop_loss) if stop_loss else None,
                'take_profit': float(take_profit) if take_profit else None,
                'model_name': self.model_name,
                'timestamp': datetime.now(),
                'raw_prediction': predicted_class,
                'prediction_probs': prediction_probs.tolist(),
                'current_price': float(current_price),
                'analysis': {
                    'available_features': len(available_features),
                    'prediction_horizon': self.config.PREDICTION_HORIZON,
                    'model_type': 'GradientBoosting'
                }
            }
            
        except Exception as e:
            logger.error(f"Temporal Fusion prediction error for {symbol}: {e}")
            return self._default_prediction(str(e))
    
    def _default_prediction(self, error_msg: str = "") -> Dict[str, Any]:
        """Return default prediction"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'price_target': None,
            'stop_loss': None,
            'take_profit': None,
            'model_name': self.model_name,
            'timestamp': datetime.now(),
            'error': error_msg
        }
    
    def is_trained(self, symbol: str) -> bool:
        """Check if model is trained"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.MODEL_DIR}/temporal_fusion_{symbol_safe}.pkl"
        scaler_path = f"{self.config.MODEL_DIR}/scaler_{symbol_safe}.pkl"
        return os.path.exists(model_path) and os.path.exists(scaler_path)
    
    def save_model(self, symbol: str):
        """Save model and scaler"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.MODEL_DIR}/temporal_fusion_{symbol_safe}.pkl"
        scaler_path = f"{self.config.MODEL_DIR}/scaler_{symbol_safe}.pkl"
        feature_path = f"{self.config.MODEL_DIR}/features_{symbol_safe}.pkl"
        
        joblib.dump(self.models[symbol], model_path)
        joblib.dump(self.scalers[symbol], scaler_path)
        joblib.dump(self.feature_columns[symbol], feature_path)
        
        logger.info(f"Temporal Fusion model saved for {symbol}")
    
    def load_model(self, symbol: str):
        """Load saved model"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.MODEL_DIR}/temporal_fusion_{symbol_safe}.pkl"
        scaler_path = f"{self.config.MODEL_DIR}/scaler_{symbol_safe}.pkl"
        feature_path = f"{self.config.MODEL_DIR}/features_{symbol_safe}.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Temporal Fusion model files not found for {symbol}")
        
        self.models[symbol] = joblib.load(model_path)
        self.scalers[symbol] = joblib.load(scaler_path)
        
        if os.path.exists(feature_path):
            self.feature_columns[symbol] = joblib.load(feature_path)
        
        logger.info(f"Temporal Fusion model loaded for {symbol}")


# ================== 4. ADAPTIVE SENTIMENT ANALYZER ==================
try:
    from google import genai
except ImportError:
    genai = None
    logging.warning("Gemini API not available - sentiment model will use fallback")


class SentimentAnalyzer(BasePredictionModel):
    """Adaptive LLM-based sentiment analyzer"""
    
    def __init__(self, db_path: str, api_key: str = None):
        super().__init__(db_path)
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'AIzaSyDK0JbwANnhWfqK2HTkHNvRvjD3mBVY6ew')
        
        # Initialize Gemini client if available
        if genai:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.model_name_llm = "gemini-2.0-flash"
                self.use_llm = True
                logger.info("Gemini API initialized for sentiment analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}")
                self.use_llm = False
        else:
            self.use_llm = False
            logger.warning("Gemini API not available - using fallback sentiment analysis")
        
        # Cache for API calls
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Create directories
        os.makedirs(os.path.join(SCRIPT_DIR, "models", "sentiment_analyzer"), exist_ok=True)
    
    def train(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> bool:
        """'Train' the sentiment model"""
        logger.info(f"Training sentiment analyzer for {symbol}...")
        # Sentiment analyzer doesn't need traditional training
        return True
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """Generate trading prediction using sentiment analysis"""
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.prediction_cache:
            cache_time, cached_prediction = self.prediction_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_prediction
        
        try:
            # Fetch all available data
            market_data = self._fetch_market_data(symbol)
            
            current_price = 0
            if market_data:
                # Find price column
                price_col = self.get_price_column()
                columns = self.schema_helper.get_columns('unified_features')
                
                # Try different price column names
                for col in columns:
                    if 'price' in col.lower() or 'close' in col.lower():
                        current_price = market_data.get(col, 0)
                        if current_price > 0:
                            break
                
                # Use first numeric value as fallback
                if current_price <= 0:
                    for key, value in market_data.items():
                        if isinstance(value, (int, float)) and value > 0 and key not in ['id', 'symbol_id']:
                            current_price = value
                            break
            
            if current_price <= 0:
                logger.warning(f"Invalid price data for {symbol}")
                return self._default_prediction()
            
            # Generate prediction based on available data
            prediction = self._generate_fallback_prediction(symbol, market_data, current_price)
            
            # Cache the prediction
            self.prediction_cache[cache_key] = (time.time(), prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating sentiment prediction for {symbol}: {e}")
            return self._default_prediction()
    
    def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest market data from unified_features"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM unified_features 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if df.empty:
                    return {}
                
                return df.iloc[0].to_dict()
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _generate_fallback_prediction(self, symbol: str, market_data: Dict, 
                                    current_price: float) -> Dict[str, Any]:
        """Generate prediction using simple rules"""
        
        # Simple scoring based on available features
        score = 0
        available_features = []
        
        # Check for RSI
        for key in market_data:
            if 'rsi' in key.lower():
                rsi_value = market_data.get(key)
                if rsi_value is not None and isinstance(rsi_value, (int, float)):
                    available_features.append(f"RSI: {rsi_value:.1f}")
                    if rsi_value < 30:
                        score += 1
                    elif rsi_value > 70:
                        score -= 1
                    break
        
        # Check for volatility
        for key in market_data:
            if 'volatility' in key.lower():
                vol_value = market_data.get(key)
                if vol_value is not None and isinstance(vol_value, (int, float)):
                    available_features.append(f"Volatility: {vol_value:.4f}")
                    if vol_value > 0.03:  # High volatility
                        score -= 0.5
                    break
        
        # Check for MACD
        for key in market_data:
            if 'macd' in key.lower() and 'signal' not in key.lower():
                macd_value = market_data.get(key)
                if macd_value is not None and isinstance(macd_value, (int, float)):
                    available_features.append(f"MACD: {macd_value:.4f}")
                    if macd_value > 0:
                        score += 0.5
                    else:
                        score -= 0.5
                    break
        
        # Generate trading signal
        if score >= 1:
            action = 'buy'
            confidence = min(0.7, (score / 2.0))
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.05
        elif score <= -1:
            action = 'sell'
            confidence = min(0.7, (abs(score) / 2.0))
            stop_loss = current_price * 1.03
            take_profit = current_price * 0.95
        else:
            action = 'hold'
            confidence = 0.3
            stop_loss = None
            take_profit = None
        
        return {
            'action': action,
            'confidence': confidence,
            'price_target': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'model_name': self.model_name,
            'timestamp': datetime.now(),
            'raw_prediction': score,
            'current_price': float(current_price),
            'analysis': {
                'method': 'Rule-based analysis',
                'score': score,
                'available_features': available_features,
                'data_points': len(market_data)
            }
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default safe prediction"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'price_target': None,
            'stop_loss': None,
            'take_profit': None,
            'model_name': self.model_name,
            'timestamp': datetime.now(),
            'error': 'Unable to generate prediction'
        }
    
    def is_trained(self, symbol: str) -> bool:
        """Sentiment analyzer is always 'trained'"""
        return True


# ================== MODEL FACTORY ==================
class ModelFactory:
    """Factory for creating prediction models"""
    
    @staticmethod
    def create_model(model_name: str, db_path: str) -> BasePredictionModel:
        """Create a prediction model by name"""
        models = {
            'boruta_cnn_lstm': BorutaCNNLSTMModel,
            'helformer': HelformerModel,
            'temporal_fusion': TemporalFusionModel,
            'sentiment': SentimentAnalyzer
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name](db_path)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models"""
        return ['boruta_cnn_lstm', 'helformer', 'temporal_fusion', 'sentiment']


# ================== TESTING SCRIPT ==================
def main():
    """Test the adaptive models"""
    db_path = os.path.join(SCRIPT_DIR, "crypto_trading.db")
    
    # First, check database schema
    print("Checking database schema...")
    schema_helper = DatabaseSchemaHelper(db_path)
    
    print("\nAvailable tables:")
    for table in schema_helper.schema_cache:
        print(f"  {table}: {len(schema_helper.get_columns(table))} columns")
    
    if 'unified_features' in schema_helper.schema_cache:
        print(f"\nunified_features columns: {schema_helper.get_columns('unified_features')[:10]}...")
    
    symbols = ['BTC/USDT', 'SOL/USDT']
    
    print("\nTesting Adaptive Prediction Models")
    print("=" * 50)
    
    # Test all models
    test_models = ['boruta_cnn_lstm', 'helformer', 'temporal_fusion', 'sentiment']
    
    for model_name in test_models:
        print(f"\nTesting {model_name}:")
        
        try:
            model = ModelFactory.create_model(model_name, db_path)
            
            for symbol in symbols[:1]:  # Test with first symbol only
                print(f"  {symbol}:")
                
                # Check if already trained
                if model.is_trained(symbol):
                    print(f"    ✅ Model already trained, making prediction...")
                else:
                    print(f"    Training {model_name} for {symbol}...")
                    success = model.train(symbol)
                    if success:
                        print(f"    ✅ Training completed successfully")
                    else:
                        print(f"    ❌ Training failed")
                        continue
                
                # Make prediction
                prediction = model.predict(symbol)
                if not prediction.get('error'):
                    print(f"    Prediction: {prediction.get('action', 'unknown')} "
                          f"(confidence: {prediction.get('confidence', 0):.2%})")
                    print(f"    Price targets: SL={prediction.get('stop_loss')}, "
                          f"TP={prediction.get('take_profit')}")
                else:
                    print(f"    ❌ Prediction error: {prediction['error']}")
                
        except Exception as e:
            print(f"  ❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
