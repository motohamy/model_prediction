"""
Fixed Ensemble Manager for Crypto Trading System
Ensures all models are trained properly
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import joblib
import os
from dataclasses import dataclass

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from complete_prediction_models import ModelFactory, BasePredictionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleManager")

@dataclass
class EnsembleConfig:
    db_path: str = os.path.join(SCRIPT_DIR, "crypto_trading.db")
    model_dir: str = os.path.join(SCRIPT_DIR, "models", "ensemble")
    
    # XGBoost parameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_subsample: float = 0.8
    
    # Market regime parameters
    volatility_window: int = 24  # hours
    trend_window: int = 48       # hours
    
    # Model selection parameters
    min_confidence_threshold: float = 0.6
    lookback_days: int = 30      # Days to look back for performance
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    daily_loss_limit: float = 0.05  # 5% daily loss limit


class MarketRegimeAnalyzer:
    """Analyze current market conditions"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_market_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze current market regime
        Returns: {
            'regime': 'trending_up'/'trending_down'/'sideways'/'volatile',
            'volatility': float,
            'trend_strength': float,
            'volume_profile': float
        }
        """
        try:
            # Get recent market data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=72)
            
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timeframe = '1h'
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """
                
                df = pd.read_sql_query(
                    query, conn,
                    params=(symbol, start_time.strftime('%Y-%m-%d %H:%M:%S'), 
                           end_time.strftime('%Y-%m-%d %H:%M:%S'))
                )
            
            if df.empty or len(df) < 24:
                return self._default_regime()
            
            # Calculate metrics
            prices = df['close'].values
            volumes = df['volume'].values
            
            # Volatility (24h rolling std of returns)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns[-24:]) if len(returns) >= 24 else np.std(returns)
            
            # Trend strength (price change over 48h)
            if len(prices) >= 48:
                trend_strength = (prices[0] - prices[47]) / prices[47]
            else:
                trend_strength = (prices[0] - prices[-1]) / prices[-1]
            
            # Volume profile (current vs average)
            avg_volume = np.mean(volumes)
            current_volume = volumes[0] if len(volumes) > 0 else avg_volume
            volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine regime
            regime = self._classify_regime(volatility, trend_strength, volume_profile)
            
            return {
                'regime': regime,
                'volatility': float(volatility),
                'trend_strength': float(trend_strength),
                'volume_profile': float(volume_profile),
                'price_momentum': float(trend_strength),
                'market_stress': float(volatility * volume_profile)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return self._default_regime()
    
    def _classify_regime(self, volatility: float, trend_strength: float, volume_profile: float) -> str:
        """Classify market regime based on metrics"""
        # High volatility threshold
        if volatility > 0.03:  # 3% hourly volatility
            return 'volatile'
        
        # Strong trend thresholds
        if abs(trend_strength) > 0.05:  # 5% move over 48h
            if trend_strength > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # Otherwise sideways
        return 'sideways'
    
    def _default_regime(self) -> Dict[str, Any]:
        """Default regime when analysis fails"""
        return {
            'regime': 'sideways',
            'volatility': 0.02,
            'trend_strength': 0.0,
            'volume_profile': 1.0,
            'price_momentum': 0.0,
            'market_stress': 0.02
        }


class ModelPerformanceTracker:
    """Track performance of individual models"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_performance_table()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_performance_table(self):
        """Initialize performance tracking table"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_time TEXT NOT NULL,
                    predicted_action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actual_outcome TEXT,
                    profit_loss REAL,
                    market_regime TEXT,
                    accuracy REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_performance 
                ON model_performance(model_name, symbol, prediction_time)
            """)
    
    def log_prediction(self, model_name: str, symbol: str, prediction: Dict[str, Any], 
                      market_regime: str):
        """Log a model prediction"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO model_performance 
                (model_name, symbol, prediction_time, predicted_action, 
                 confidence, market_regime)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_name, symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                prediction.get('action', 'hold'), prediction.get('confidence', 0.0),
                market_regime
            ))
    
    def update_outcome(self, model_name: str, symbol: str, prediction_time: datetime,
                      actual_outcome: str, profit_loss: float):
        """Update the actual outcome of a prediction"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE model_performance 
                SET actual_outcome = ?, profit_loss = ?
                WHERE model_name = ? AND symbol = ? AND prediction_time = ?
            """, (actual_outcome, profit_loss, model_name, symbol, 
                 prediction_time.strftime('%Y-%m-%d %H:%M:%S')))
    
    def get_model_performance(self, model_name: str, symbol: str, 
                            days_back: int = 30) -> Dict[str, float]:
        """Get performance metrics for a model"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM model_performance 
                WHERE model_name = ? AND symbol = ? 
                AND prediction_time >= ?
                AND actual_outcome IS NOT NULL
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(model_name, symbol, cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
            )
        
        if df.empty:
            return {'accuracy': 0.5, 'avg_confidence': 0.5, 'profit_factor': 1.0, 'win_rate': 0.5}
        
        # Calculate metrics
        correct_predictions = df['predicted_action'] == df['actual_outcome']
        accuracy = correct_predictions.mean()
        avg_confidence = df['confidence'].mean()
        
        # Profit metrics
        profits = df[df['profit_loss'] > 0]['profit_loss'].sum()
        losses = abs(df[df['profit_loss'] < 0]['profit_loss'].sum())
        profit_factor = profits / max(losses, 0.001)
        win_rate = (df['profit_loss'] > 0).mean()
        
        return {
            'accuracy': float(accuracy),
            'avg_confidence': float(avg_confidence),
            'profit_factor': float(profit_factor),
            'win_rate': float(win_rate),
            'total_trades': len(df)
        }
    
    def get_regime_performance(self, model_name: str, symbol: str, 
                              regime: str, days_back: int = 30) -> Dict[str, float]:
        """Get performance for a specific market regime"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM model_performance 
                WHERE model_name = ? AND symbol = ? AND market_regime = ?
                AND prediction_time >= ?
                AND actual_outcome IS NOT NULL
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(model_name, symbol, regime, cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
            )
        
        if df.empty:
            return {'accuracy': 0.5, 'win_rate': 0.5}
        
        correct_predictions = df['predicted_action'] == df['actual_outcome']
        accuracy = correct_predictions.mean()
        win_rate = (df['profit_loss'] > 0).mean()
        
        return {
            'accuracy': float(accuracy),
            'win_rate': float(win_rate),
            'sample_size': len(df)
        }


class EnsembleManager:
    """Main ensemble manager that combines model predictions"""
    
    def __init__(self, db_path: str = os.path.join(SCRIPT_DIR, "crypto_trading.db")):
        self.config = EnsembleConfig(db_path=db_path)
        self.db_path = db_path
        
        # Initialize components
        self.regime_analyzer = MarketRegimeAnalyzer(db_path)
        self.performance_tracker = ModelPerformanceTracker(db_path)
        
        # Initialize models
        self.models = {}
        self._init_models()
        
        # XGBoost ensemble model
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        self.is_ensemble_trained = {}
        
        # Create model directory
        os.makedirs(self.config.model_dir, exist_ok=True)
    
    def _init_models(self):
        """Initialize all prediction models"""
        model_names = ModelFactory.get_available_models()
        
        for model_name in model_names:
            try:
                self.models[model_name] = ModelFactory.create_model(model_name, self.db_path)
                logger.info(f"Initialized {model_name} model")
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
    
    def collect_predictions(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Collect predictions from all models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(symbol)
                predictions[model_name] = prediction
                
                # Log prediction for performance tracking
                market_regime = self.regime_analyzer.get_market_regime(symbol)
                self.performance_tracker.log_prediction(
                    model_name, symbol, prediction, market_regime['regime']
                )
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                predictions[model_name] = {
                    'action': 'hold',
                    'confidence': 0.0,
                    'model_name': model_name,
                    'error': str(e)
                }
        
        return predictions
    
    # In create_ensemble_features method:
    def create_ensemble_features(self, symbol: str, predictions: Dict[str, Dict], 
                            market_regime: Dict[str, Any]) -> np.ndarray:
        features = []
        
        # Dynamic model handling instead of hard-coded
        available_models = list(predictions.keys())
        
        # Model predictions and confidences
        for model_name in available_models:
            pred = predictions.get(model_name, {})
            
            # Action encoding with better error handling
            action_encoding = {'buy': 1, 'sell': -1, 'hold': 0}
            action_value = action_encoding.get(pred.get('action', 'hold'), 0)
            
            # Validate confidence value
            confidence = pred.get('confidence', 0.0)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            features.extend([action_value, confidence])
        
        # Add market regime features with validation
        volatility = market_regime.get('volatility', 0.02)
        trend_strength = market_regime.get('trend_strength', 0.0)
        volume_profile = market_regime.get('volume_profile', 1.0)
        market_stress = market_regime.get('market_stress', 0.02)
        
        features.extend([volatility, trend_strength, volume_profile, market_stress])
        
        # Model performance features with validation
        for model_name in available_models:
            try:
                perf = self.performance_tracker.get_model_performance(model_name, symbol, days_back=7)
                accuracy = perf.get('accuracy', 0.5)
                features.append(max(0.0, min(1.0, accuracy)))  # Clamp to [0,1]
            except Exception as e:
                logger.warning(f"Error getting performance for {model_name}: {e}")
                features.append(0.5)  # Default
        
        # Ensure consistent feature vector size
        feature_array = np.array(features)
        if len(feature_array) == 0:
            feature_array = np.array([0.5] * 16)  # Default fallback
        
        return feature_array.reshape(1, -1)
    
    def train_ensemble(self, symbol: str, days_back: int = 90):
        """Train XGBoost ensemble model"""
        logger.info(f"Training ensemble model for {symbol}...")
        
        try:
            # Collect historical training data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get historical predictions (simulated for now)
            X_train, y_train = self._generate_training_data(symbol, start_date, end_date)
            
            if len(X_train) < 50:
                logger.warning(f"Insufficient training data for ensemble: {len(X_train)} samples")
                return False
            
            # Split data
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train_split)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train XGBoost
            self.ensemble_model = xgb.XGBClassifier(
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                subsample=self.config.xgb_subsample,
                random_state=42
            )
            
            self.ensemble_model.fit(X_train_scaled, y_train_split)
            
            # Evaluate
            y_pred = self.ensemble_model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            logger.info(f"Ensemble model trained for {symbol} with accuracy: {accuracy:.3f}")
            
            # Save model
            self._save_ensemble_model(symbol)
            self.is_ensemble_trained[symbol] = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble for {symbol}: {e}")
            return False
    
    def _generate_training_data(self, symbol: str, start_date: datetime, 
                              end_date: datetime) -> tuple:
        """Generate training data for ensemble (simplified version)"""
        # This is a simplified version - in production, you'd collect actual historical predictions
        
        # Simulate historical feature vectors
        n_samples = 200
        n_features = 16  # 4 models * 2 features + 4 regime features + 4 performance features
        
        X = np.random.randn(n_samples, n_features)
        
        # Simulate binary targets (profitable trade = 1, unprofitable = 0)
        y = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
        
        return X, y
    
    def make_final_decision(self, symbol: str) -> Dict[str, Any]:
        """Make final trading decision by combining all models"""
        try:
            # Get market regime
            market_regime = self.regime_analyzer.get_market_regime(symbol)
            
            # Collect predictions from all models
            predictions = self.collect_predictions(symbol)
            
            # Check if we have any valid predictions
            valid_predictions = {k: v for k, v in predictions.items() 
                               if not v.get('error') and v.get('confidence', 0) > 0.1}
            
            if not valid_predictions:
                logger.warning(f"No valid predictions for {symbol}")
                return self._default_decision(symbol)
            
            # Create ensemble features
            ensemble_features = self.create_ensemble_features(symbol, predictions, market_regime)
            
            # Use ensemble model if trained
            if self.is_ensemble_trained.get(symbol, False) and self.ensemble_model:
                try:
                    # Scale features
                    scaled_features = self.feature_scaler.transform(ensemble_features)
                    
                    # Get ensemble prediction
                    ensemble_prob = self.ensemble_model.predict_proba(scaled_features)[0]
                    ensemble_confidence = max(ensemble_prob)
                    ensemble_action = 'buy' if ensemble_prob[1] > 0.6 else ('sell' if ensemble_prob[0] > 0.6 else 'hold')
                    
                    # Get price targets from best performing model
                    best_model = self._get_best_model(symbol, market_regime['regime'])
                    best_prediction = predictions.get(best_model, {})
                    
                    return {
                        'symbol': symbol,
                        'action': ensemble_action,
                        'confidence': float(ensemble_confidence),
                        'size': self._calculate_position_size(ensemble_confidence),
                        'price_target': best_prediction.get('price_target'),
                        'stop_loss': best_prediction.get('stop_loss'),
                        'take_profit': best_prediction.get('take_profit'),
                        'market_regime': market_regime['regime'],
                        'contributing_models': list(valid_predictions.keys()),
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"Ensemble prediction failed: {e}")
                    # Fall back to simple majority vote
                    return self._majority_vote_decision(symbol, predictions, market_regime)
            
            else:
                # Train ensemble if not trained
                if not self.is_ensemble_trained.get(symbol, False):
                    self.train_ensemble(symbol)
                
                # Use majority vote as fallback
                return self._majority_vote_decision(symbol, predictions, market_regime)
            
        except Exception as e:
            logger.error(f"Error making final decision for {symbol}: {e}")
            return self._default_decision(symbol)
    
    def _majority_vote_decision(self, symbol: str, predictions: Dict[str, Dict], 
                              market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Simple majority vote fallback"""
        actions = []
        confidences = []
        
        for pred in predictions.values():
            if not pred.get('error') and pred.get('confidence', 0) > 0.3:
                actions.append(pred.get('action', 'hold'))
                confidences.append(pred.get('confidence', 0))
        
        if not actions:
            return self._default_decision(symbol)
        
        # Count votes
        buy_votes = actions.count('buy')
        sell_votes = actions.count('sell')
        hold_votes = actions.count('hold')
        
        # Determine action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            action = 'buy'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action = 'sell'
        else:
            action = 'hold'
        
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Get price targets from highest confidence model
        best_pred = max(predictions.values(), key=lambda x: x.get('confidence', 0))
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': float(avg_confidence),
            'size': self._calculate_position_size(avg_confidence),
            'price_target': best_pred.get('price_target'),
            'stop_loss': best_pred.get('stop_loss'),
            'take_profit': best_pred.get('take_profit'),
            'market_regime': market_regime['regime'],
            'method': 'majority_vote',
            'timestamp': datetime.now()
        }
    
    def _get_best_model(self, symbol: str, regime: str) -> str:
        """Get the best performing model for current regime"""
        best_model = 'boruta_cnn_lstm'  # default
        best_accuracy = 0.0
        
        for model_name in self.models.keys():
            perf = self.performance_tracker.get_regime_performance(model_name, symbol, regime)
            if perf['accuracy'] > best_accuracy and perf.get('sample_size', 0) > 5:
                best_accuracy = perf['accuracy']
                best_model = model_name
        
        return best_model
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        base_size = self.config.max_position_size
        
        # Scale position size by confidence
        if confidence < 0.6:
            return 0.0  # No trade
        elif confidence < 0.7:
            return base_size * 0.3
        elif confidence < 0.8:
            return base_size * 0.6
        else:
            return base_size
    
    def _default_decision(self, symbol: str) -> Dict[str, Any]:
        """Default decision when all else fails"""
        return {
            'symbol': symbol,
            'action': 'hold',
            'confidence': 0.0,
            'size': 0.0,
            'price_target': None,
            'stop_loss': None,
            'take_profit': None,
            'market_regime': 'unknown',
            'timestamp': datetime.now()
        }
    
    def _save_ensemble_model(self, symbol: str):
        """Save ensemble model"""
        # Fix the symbol path issue
        symbol_safe = symbol.replace('/', '_')  # Convert BTC/USDT to BTC_USDT
        
        model_path = f"{self.config.model_dir}/ensemble_{symbol_safe}.pkl"
        scaler_path = f"{self.config.model_dir}/scaler_{symbol_safe}.pkl"
        
        joblib.dump(self.ensemble_model, model_path)
        joblib.dump(self.feature_scaler, scaler_path)
    
    def _load_ensemble_model(self, symbol: str):
        """Load ensemble model"""
        symbol_safe = symbol.replace('/', '_')
        model_path = f"{self.config.model_dir}/ensemble_{symbol_safe}.pkl"
        scaler_path = f"{self.config.model_dir}/scaler_{symbol_safe}.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.ensemble_model = joblib.load(model_path)
            self.feature_scaler = joblib.load(scaler_path)
            self.is_ensemble_trained[symbol] = True
    
    def train_individual_models(self, symbol: str):
        """Train all individual models with enhanced error handling"""
        # Get available models in correct order
        model_order = ['boruta_cnn_lstm', 'helformer', 'temporal_fusion', 'sentiment']
        
        training_results = {}
        
        for model_name in model_order:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not initialized, skipping...")
                training_results[model_name] = False
                continue
                
            model = self.models[model_name]
            
            # Check if training is needed
            try:
                if hasattr(model, 'train') and hasattr(model, 'is_trained'):
                    if not model.is_trained(symbol):
                        logger.info(f"🔄 Training {model_name} for {symbol}...")
                        try:
                            success = model.train(symbol)
                            if success:
                                logger.info(f"✅ Successfully trained {model_name}")
                                training_results[model_name] = True
                            else:
                                logger.warning(f"⚠️ Failed to train {model_name} (returned False)")
                                training_results[model_name] = False
                        except Exception as e:
                            logger.error(f"❌ Error training {model_name}: {e}")
                            import traceback
                            logger.error(f"Full traceback for {model_name}: {traceback.format_exc()}")
                            training_results[model_name] = False
                    else:
                        logger.info(f"✅ {model_name} already trained for {symbol}")
                        training_results[model_name] = True
                elif model_name == 'sentiment':
                    # Sentiment model doesn't need traditional training
                    logger.info(f"✅ Sentiment analyzer ready for {symbol}")
                    training_results[model_name] = True
                else:
                    logger.warning(f"⚠️ {model_name} missing train/is_trained methods")
                    training_results[model_name] = False
                    
            except Exception as e:
                logger.error(f"❌ Critical error with {model_name}: {e}")
                import traceback
                logger.error(f"Full traceback for {model_name}: {traceback.format_exc()}")
                training_results[model_name] = False
        
        # Log summary
        successful_models = [k for k, v in training_results.items() if v]
        failed_models = [k for k, v in training_results.items() if not v]
        
        logger.info(f"Training summary for {symbol}:")
        logger.info(f"  ✅ Successful: {successful_models}")
        if failed_models:
            logger.warning(f"  ❌ Failed: {failed_models}")
        
        return training_results

# Test function
def main():
    """Test the ensemble manager"""
    ensemble = EnsembleManager()
    
    symbols = ['BTC/USDT', 'SOL/USDT']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing Ensemble for {symbol}")
        print(f"{'='*50}")
        
        # Train individual models
        ensemble.train_individual_models(symbol)
        
        # Train ensemble
        ensemble.train_ensemble(symbol)
        
        # Make decision
        decision = ensemble.make_final_decision(symbol)
        print(f"\nFinal Decision: {decision}")


if __name__ == "__main__":
    main()
