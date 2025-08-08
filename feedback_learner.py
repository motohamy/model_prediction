"""
Feedback Learning System for Crypto Trading Bot
Analyzes trade outcomes and continuously improves model performance
Implements adaptive learning with market regime awareness

Key Features:
- Track individual model performance across different market conditions
- Adaptive weight adjustment using multiple metrics (win rate, profit factor, Sharpe)
- Market regime-specific performance tracking
- Recency-weighted learning with exponential decay
- Safety mechanisms to prevent overfitting
- Comprehensive performance analytics
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeedbackLearner")

@dataclass
class TradeOutcome:
    """Structure for completed trade information"""
    trade_id: str
    position_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percentage: float
    duration_hours: float
    exit_reason: str
    model_source: str
    confidence: float
    market_regime: str
    
    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate actual risk/reward achieved"""
        if self.side == 'buy':
            risk = abs(self.entry_price * 0.02)  # Assumed 2% stop loss
            reward = self.exit_price - self.entry_price
        else:
            risk = abs(self.entry_price * 0.02)
            reward = self.entry_price - self.exit_price
        
        return reward / risk if risk > 0 else 0

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics for a model"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_duration_hours: float = 0.0
    best_regime: str = "unknown"
    worst_regime: str = "unknown"
    confidence_correlation: float = 0.0  # Correlation between confidence and profitability
    
    def update_from_trades(self, trades: List[TradeOutcome]):
        """Update metrics from a list of trades"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.is_profitable)
        self.losing_trades = self.total_trades - self.winning_trades
        
        # Calculate PnL metrics
        winning_pnls = [t.pnl for t in trades if t.is_profitable]
        losing_pnls = [t.pnl for t in trades if not t.is_profitable]
        
        self.total_pnl = sum(t.pnl for t in trades)
        self.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        self.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 1
        self.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_percentage for t in trades]
        if len(returns) > 1:
            self.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in sorted(trades, key=lambda x: x.exit_time)])
        if len(cumulative_pnl) > 0:
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (running_max - cumulative_pnl) / (running_max + 1e-8)
            self.max_drawdown = np.max(drawdown)
        
        # Average duration
        self.avg_duration_hours = np.mean([t.duration_hours for t in trades])
        
        # Confidence correlation
        if len(trades) > 2:
            confidences = [t.confidence for t in trades]
            profits = [1 if t.is_profitable else 0 for t in trades]
            self.confidence_correlation = np.corrcoef(confidences, profits)[0, 1]

@dataclass
class FeedbackConfig:
    """Configuration for feedback learning system"""
    # Database
    db_path: str = "data/crypto_trading.db"
    
    # Learning parameters
    learning_rate: float = 0.1  # How fast to adjust weights
    decay_factor: float = 0.95  # Exponential decay for old trades
    min_trades_for_update: int = 10  # Minimum trades before updating weights
    
    # Weight constraints
    min_model_weight: float = 0.05  # Minimum 5% weight
    max_model_weight: float = 0.40  # Maximum 40% weight
    
    # Performance thresholds
    poor_performance_threshold: float = 0.35  # Win rate below this is poor
    excellent_performance_threshold: float = 0.65  # Win rate above this is excellent
    
    # Regime-specific learning
    regime_weight_bonus: float = 0.1  # Extra weight for good regime performance
    
    # Recency parameters
    recent_window_days: int = 7  # Recent performance window
    medium_window_days: int = 30  # Medium-term window
    long_window_days: int = 90  # Long-term window
    
    # Safety parameters
    max_weight_change_per_update: float = 0.05  # Max 5% change per update
    consecutive_loss_threshold: int = 5  # Reduce weight after X consecutive losses
    
    # Analytics
    save_analytics: bool = True
    analytics_dir: str = "analytics/feedback"


class PerformanceAnalyzer:
    """Analyzes model performance across different dimensions"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def analyze_model_performance(self, model_name: str, window_days: int) -> Dict[str, Any]:
        """Comprehensive performance analysis for a model"""
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM completed_trades 
                WHERE model_source = ? 
                AND exit_time >= ?
                ORDER BY exit_time DESC
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(model_name, cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
            )
        
        if df.empty:
            return self._empty_analysis()
        
        # Convert to TradeOutcome objects
        trades = []
        for _, row in df.iterrows():
            trade = TradeOutcome(
                trade_id=row['id'],
                position_id=row['position_id'],
                symbol=row['symbol'],
                side=row['side'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                entry_time=pd.to_datetime(row['entry_time']),
                exit_time=pd.to_datetime(row['exit_time']),
                pnl=row['pnl'],
                pnl_percentage=row['pnl_percentage'],
                duration_hours=row['duration_minutes'] / 60,
                exit_reason=row['exit_reason'],
                model_source=row['model_source'],
                confidence=row['confidence'],
                market_regime=row['market_regime']
            )
            trades.append(trade)
        
        # Overall metrics
        overall_metrics = ModelPerformanceMetrics()
        overall_metrics.update_from_trades(trades)
        
        # Regime-specific analysis
        regime_performance = {}
        for regime in ['trending_up', 'trending_down', 'sideways', 'volatile']:
            regime_trades = [t for t in trades if t.market_regime == regime]
            if regime_trades:
                regime_metrics = ModelPerformanceMetrics()
                regime_metrics.update_from_trades(regime_trades)
                regime_performance[regime] = regime_metrics
        
        # Symbol-specific analysis
        symbol_performance = {}
        for symbol in df['symbol'].unique():
            symbol_trades = [t for t in trades if t.symbol == symbol]
            symbol_metrics = ModelPerformanceMetrics()
            symbol_metrics.update_from_trades(symbol_trades)
            symbol_performance[symbol] = symbol_metrics
        
        # Time-based analysis (performance over time)
        time_performance = self._analyze_time_performance(trades)
        
        # Exit reason analysis
        exit_reason_stats = df['exit_reason'].value_counts().to_dict()
        
        return {
            'model_name': model_name,
            'window_days': window_days,
            'overall': overall_metrics,
            'by_regime': regime_performance,
            'by_symbol': symbol_performance,
            'by_time': time_performance,
            'exit_reasons': exit_reason_stats,
            'recent_streak': self._calculate_streak(trades[-10:]) if len(trades) >= 10 else 0
        }
    
    def _analyze_time_performance(self, trades: List[TradeOutcome]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not trades:
            return {}
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda x: x.exit_time)
        
        # Calculate rolling metrics
        window_size = min(20, len(trades) // 3)
        if window_size < 5:
            return {}
        
        rolling_win_rates = []
        rolling_pnl = []
        
        for i in range(window_size, len(trades) + 1):
            window_trades = sorted_trades[i-window_size:i]
            wins = sum(1 for t in window_trades if t.is_profitable)
            win_rate = wins / len(window_trades)
            total_pnl = sum(t.pnl for t in window_trades)
            
            rolling_win_rates.append(win_rate)
            rolling_pnl.append(total_pnl)
        
        # Trend analysis
        if len(rolling_win_rates) > 1:
            win_rate_trend = np.polyfit(range(len(rolling_win_rates)), rolling_win_rates, 1)[0]
            pnl_trend = np.polyfit(range(len(rolling_pnl)), rolling_pnl, 1)[0]
        else:
            win_rate_trend = 0
            pnl_trend = 0
        
        return {
            'win_rate_trend': win_rate_trend,  # Positive = improving
            'pnl_trend': pnl_trend,
            'latest_win_rate': rolling_win_rates[-1] if rolling_win_rates else 0,
            'performance_stability': 1 - np.std(rolling_win_rates) if rolling_win_rates else 0
        }
    
    def _calculate_streak(self, trades: List[TradeOutcome]) -> int:
        """Calculate current winning/losing streak"""
        if not trades:
            return 0
        
        streak = 0
        is_winning = trades[-1].is_profitable
        
        for trade in reversed(trades):
            if trade.is_profitable == is_winning:
                streak += 1 if is_winning else -1
            else:
                break
        
        return streak
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'overall': ModelPerformanceMetrics(),
            'by_regime': {},
            'by_symbol': {},
            'by_time': {},
            'exit_reasons': {},
            'recent_streak': 0
        }


class AdaptiveWeightCalculator:
    """Calculates adaptive weights for models based on performance"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
    
    def calculate_weights(self, performance_data: Dict[str, Dict[str, Any]], 
                         current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate new model weights based on comprehensive performance data"""
        
        # Initialize scores for each model
        model_scores = {}
        
        for model_name, perf in performance_data.items():
            if not perf or 'overall' not in perf:
                model_scores[model_name] = 0.5  # Default score
                continue
            
            overall = perf['overall']
            
            # Base score from multiple metrics
            score = 0.0
            
            # Win rate component (40% weight)
            win_rate_score = overall.win_rate
            score += win_rate_score * 0.4
            
            # Profit factor component (30% weight)
            pf_score = min(overall.profit_factor / 2, 1.0)  # Normalize to 0-1
            score += pf_score * 0.3
            
            # Sharpe ratio component (20% weight)
            sharpe_score = min(max(overall.sharpe_ratio / 2, 0), 1.0)  # Normalize
            score += sharpe_score * 0.2
            
            # Consistency component (10% weight)
            if 'by_time' in perf and perf['by_time']:
                stability = perf['by_time'].get('performance_stability', 0.5)
                score += stability * 0.1
            else:
                score += 0.05  # Default consistency
            
            # Regime bonus
            if 'by_regime' in perf:
                regime_scores = []
                for regime, regime_perf in perf['by_regime'].items():
                    if regime_perf.win_rate > self.config.excellent_performance_threshold:
                        regime_scores.append(self.config.regime_weight_bonus)
                score += np.mean(regime_scores) if regime_scores else 0
            
            # Recency adjustment
            if 'by_time' in perf and perf['by_time']:
                trend = perf['by_time'].get('win_rate_trend', 0)
                if trend > 0:  # Improving performance
                    score *= 1.1
                elif trend < -0.01:  # Declining performance
                    score *= 0.9
            
            # Penalty for poor performance
            if overall.win_rate < self.config.poor_performance_threshold:
                score *= 0.7
            
            # Confidence correlation bonus
            if overall.confidence_correlation > 0.5:
                score *= 1.05  # Model's confidence is well-calibrated
            
            model_scores[model_name] = max(0.1, min(1.0, score))  # Bound between 0.1 and 1.0
        
        # Convert scores to weights
        new_weights = self._scores_to_weights(model_scores, current_weights)
        
        return new_weights
    
    def _scores_to_weights(self, scores: Dict[str, float], 
                          current_weights: Dict[str, float]) -> Dict[str, float]:
        """Convert performance scores to portfolio weights with safety constraints"""
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal weights if no scores
            num_models = len(scores)
            return {model: 1.0 / num_models for model in scores}
        
        # Calculate raw weights
        raw_weights = {model: score / total_score for model, score in scores.items()}
        
        # Apply learning rate to smooth changes
        new_weights = {}
        for model in raw_weights:
            current = current_weights.get(model, 0.25)  # Default 25% if new
            target = raw_weights[model]
            
            # Limit maximum change per update
            change = target - current
            max_change = self.config.max_weight_change_per_update
            if abs(change) > max_change:
                change = max_change if change > 0 else -max_change
            
            new_weight = current + change * self.config.learning_rate
            
            # Apply min/max constraints
            new_weight = max(self.config.min_model_weight, 
                           min(self.config.max_model_weight, new_weight))
            
            new_weights[model] = new_weight
        
        # Renormalize to ensure sum = 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {model: w / total_weight for model, w in new_weights.items()}
        
        return new_weights


class FeedbackLearner:
    """Main feedback learning system"""
    
    def __init__(self, db_path: str):
        self.config = FeedbackConfig(db_path=db_path)
        self.analyzer = PerformanceAnalyzer(db_path)
        self.weight_calculator = AdaptiveWeightCalculator(self.config)
        
        # Initialize database
        self._init_database()
        
        # Load current weights
        self.model_weights = self._load_weights()
        
        # Performance cache
        self.performance_cache = {}
        self.last_update_time = datetime.now()
        
        # Create analytics directory
        if self.config.save_analytics:
            os.makedirs(self.config.analytics_dir, exist_ok=True)
        
        logger.info("Feedback learner initialized")
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize feedback learning tables"""
        with self.get_connection() as conn:
            # Model weights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    weight REAL NOT NULL,
                    performance_score REAL,
                    last_update TEXT NOT NULL,
                    update_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name)
                )
            """)
            
            # Learning history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    old_weight REAL NOT NULL,
                    new_weight REAL NOT NULL,
                    performance_metrics TEXT,
                    trades_analyzed INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    window_days INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _load_weights(self) -> Dict[str, float]:
        """Load current model weights from database"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT model_name, weight 
                FROM model_weights 
                ORDER BY last_update DESC
            """)
            
            weights = {}
            for row in cursor.fetchall():
                weights[row['model_name']] = row['weight']
            
            # Initialize default weights if empty
            if not weights:
                default_models = ['boruta_cnn_lstm', 'helformer', 'temporal_fusion', 'sentiment']
                num_models = len(default_models)
                weights = {model: 1.0 / num_models for model in default_models}
                
                # Save default weights
                for model, weight in weights.items():
                    self._save_weight(model, weight, "initialization")
            
            return weights
    
    def _save_weight(self, model_name: str, weight: float, reason: str = "update"):
        """Save model weight to database"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_weights 
                (model_name, weight, last_update, update_reason)
                VALUES (?, ?, ?, ?)
            """, (model_name, weight, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), reason))
    
    def process_trade_outcome(self, outcome: TradeOutcome):
        """Process a completed trade outcome"""
        logger.info(f"Processing trade outcome: {outcome.model_source} - "
                   f"PnL: {outcome.pnl_percentage:.2f}%")
        
        # Clear performance cache to force recalculation
        self.performance_cache = {}
        
        # Check if we should update weights
        trades_since_update = self._get_trades_since_update()
        
        if trades_since_update >= self.config.min_trades_for_update:
            logger.info(f"Triggering weight update ({trades_since_update} trades since last update)")
            self.update_model_weights()
    
    def _get_trades_since_update(self) -> int:
        """Count trades since last weight update"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count 
                FROM trades 
                WHERE exit_time > ?
            """, (self.last_update_time.strftime('%Y-%m-%d %H:%M:%S'),))
            
            result = cursor.fetchone()
            return result['count'] if result else 0
    
    def update_model_weights(self, force: bool = False) -> Dict[str, float]:
        """Update model weights based on recent performance"""
        
        # Check if update is needed
        if not force and self._get_trades_since_update() < self.config.min_trades_for_update:
            logger.info("Not enough trades for weight update")
            return self.model_weights
        
        logger.info("Updating model weights...")
        
        # Analyze performance for each model
        performance_data = {}
        
        for model_name in self.model_weights.keys():
            # Multi-timeframe analysis
            recent_perf = self.analyzer.analyze_model_performance(
                model_name, self.config.recent_window_days
            )
            medium_perf = self.analyzer.analyze_model_performance(
                model_name, self.config.medium_window_days
            )
            long_perf = self.analyzer.analyze_model_performance(
                model_name, self.config.long_window_days
            )
            
            # Weighted combination (recent matters more)
            combined_perf = self._combine_performance_windows(
                recent_perf, medium_perf, long_perf
            )
            
            performance_data[model_name] = combined_perf
            
            # Save performance snapshot
            if self.config.save_analytics:
                self._save_performance_snapshot(model_name, combined_perf)
        
        # Calculate new weights
        old_weights = self.model_weights.copy()
        new_weights = self.weight_calculator.calculate_weights(
            performance_data, self.model_weights
        )
        
        # Save weight updates
        for model_name, new_weight in new_weights.items():
            old_weight = old_weights.get(model_name, 0)
            
            if abs(new_weight - old_weight) > 0.001:  # Significant change
                self._save_weight(model_name, new_weight, "performance_update")
                
                # Log learning history
                self._log_learning_history(
                    model_name, old_weight, new_weight, 
                    performance_data.get(model_name, {})
                )
        
        self.model_weights = new_weights
        self.last_update_time = datetime.now()
        
        # Log weight changes
        logger.info("Updated model weights:")
        for model, weight in sorted(new_weights.items(), key=lambda x: x[1], reverse=True):
            change = weight - old_weights.get(model, 0.25)
            change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
            logger.info(f"  {model}: {weight:.3f} ({change_str})")
        
        # Save analytics report
        if self.config.save_analytics:
            self._save_analytics_report(performance_data, old_weights, new_weights)
        
        return new_weights
    
    def _combine_performance_windows(self, recent: Dict, medium: Dict, long: Dict) -> Dict:
        """Combine multi-timeframe performance with recency weighting"""
        
        # Weights for different timeframes
        weights = {
            'recent': 0.5,   # 50% weight on last 7 days
            'medium': 0.3,   # 30% weight on last 30 days
            'long': 0.2      # 20% weight on last 90 days
        }
        
        combined = {}
        
        # Combine overall metrics
        if all('overall' in p for p in [recent, medium, long]):
            combined_metrics = ModelPerformanceMetrics()
            
            # Weighted averages
            combined_metrics.win_rate = (
                recent['overall'].win_rate * weights['recent'] +
                medium['overall'].win_rate * weights['medium'] +
                long['overall'].win_rate * weights['long']
            )
            
            combined_metrics.profit_factor = (
                recent['overall'].profit_factor * weights['recent'] +
                medium['overall'].profit_factor * weights['medium'] +
                long['overall'].profit_factor * weights['long']
            )
            
            combined_metrics.sharpe_ratio = (
                recent['overall'].sharpe_ratio * weights['recent'] +
                medium['overall'].sharpe_ratio * weights['medium'] +
                long['overall'].sharpe_ratio * weights['long']
            )
            
            # Use most recent values for some metrics
            combined_metrics.total_trades = recent['overall'].total_trades
            combined_metrics.confidence_correlation = recent['overall'].confidence_correlation
            
            combined['overall'] = combined_metrics
        
        # Pass through other analyses from recent window
        for key in ['by_regime', 'by_symbol', 'by_time', 'exit_reasons']:
            if key in recent:
                combined[key] = recent[key]
        
        return combined
    
    def _log_learning_history(self, model_name: str, old_weight: float, 
                            new_weight: float, performance: Dict):
        """Log weight update history"""
        with self.get_connection() as conn:
            metrics_json = json.dumps({
                'win_rate': performance.get('overall', {}).win_rate if 'overall' in performance else 0,
                'profit_factor': performance.get('overall', {}).profit_factor if 'overall' in performance else 0,
                'total_trades': performance.get('overall', {}).total_trades if 'overall' in performance else 0
            })
            
            conn.execute("""
                INSERT INTO learning_history 
                (timestamp, model_name, old_weight, new_weight, performance_metrics, trades_analyzed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                model_name, old_weight, new_weight, metrics_json,
                performance.get('overall', {}).total_trades if 'overall' in performance else 0
            ))
    
    def _save_performance_snapshot(self, model_name: str, performance: Dict):
        """Save performance snapshot for analysis"""
        with self.get_connection() as conn:
            metrics_json = json.dumps(performance, default=str)
            
            conn.execute("""
                INSERT INTO performance_snapshots 
                (timestamp, model_name, window_days, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                model_name,
                self.config.medium_window_days,
                metrics_json
            ))
    
    def _save_analytics_report(self, performance_data: Dict, 
                             old_weights: Dict, new_weights: Dict):
        """Save detailed analytics report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': {},
            'weight_changes': {},
            'recommendations': []
        }
        
        # Performance summary
        for model, perf in performance_data.items():
            if 'overall' in perf:
                overall = perf['overall']
                report['performance_summary'][model] = {
                    'win_rate': overall.win_rate,
                    'profit_factor': overall.profit_factor,
                    'total_trades': overall.total_trades,
                    'sharpe_ratio': overall.sharpe_ratio,
                    'best_regime': overall.best_regime,
                    'confidence_correlation': overall.confidence_correlation
                }
        
        # Weight changes
        for model in new_weights:
            old = old_weights.get(model, 0.25)
            new = new_weights[model]
            report['weight_changes'][model] = {
                'old_weight': old,
                'new_weight': new,
                'change': new - old,
                'change_percentage': ((new - old) / old * 100) if old > 0 else 0
            }
        
        # Generate recommendations
        for model, perf in performance_data.items():
            if 'overall' in perf:
                overall = perf['overall']
                
                if overall.win_rate < self.config.poor_performance_threshold:
                    report['recommendations'].append(
                        f"Consider retraining {model} - win rate {overall.win_rate:.1%} "
                        f"below threshold {self.config.poor_performance_threshold:.1%}"
                    )
                
                if overall.confidence_correlation < 0.2:
                    report['recommendations'].append(
                        f"{model} confidence scores poorly calibrated "
                        f"(correlation: {overall.confidence_correlation:.2f})"
                    )
        
        # Save report
        report_path = os.path.join(
            self.config.analytics_dir, 
            f"weight_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analytics report saved to {report_path}")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return self.model_weights.copy()
    
    def get_model_performance_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {}
        
        for model_name in self.model_weights.keys():
            perf = self.analyzer.analyze_model_performance(model_name, days_back)
            
            if 'overall' in perf and perf['overall'].total_trades > 0:
                overall = perf['overall']
                summary[model_name] = {
                    'weight': self.model_weights[model_name],
                    'win_rate': overall.win_rate,
                    'profit_factor': overall.profit_factor,
                    'total_trades': overall.total_trades,
                    'total_pnl': overall.total_pnl,
                    'sharpe_ratio': overall.sharpe_ratio,
                    'recent_streak': perf.get('recent_streak', 0)
                }
        
        return summary
    
    def get_learning_history(self, days_back: int = 7) -> pd.DataFrame:
        """Get recent learning history"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM learning_history 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),)
            )
        
        return df
    
    def reset_weights(self):
        """Reset all model weights to equal"""
        logger.warning("Resetting all model weights to equal distribution")
        
        num_models = len(self.model_weights)
        equal_weight = 1.0 / num_models
        
        for model in self.model_weights:
            self.model_weights[model] = equal_weight
            self._save_weight(model, equal_weight, "manual_reset")
        
        logger.info(f"Reset {num_models} models to {equal_weight:.3f} weight each")


# Testing and demonstration
def main():
    """Test the feedback learning system"""
    import random
    
    # Initialize feedback learner
    learner = FeedbackLearner("data/crypto_trading.db")
    
    print("Feedback Learning System Test")
    print("=" * 50)
    
    # Show current weights
    print("\nCurrent Model Weights:")
    weights = learner.get_model_weights()
    for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {weight:.3f}")
    
    # Get performance summary
    print("\nModel Performance Summary (30 days):")
    summary = learner.get_model_performance_summary(30)
    
    for model, metrics in summary.items():
        print(f"\n{model}:")
        print(f"  Weight: {metrics['weight']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Recent Streak: {metrics['recent_streak']}")
    
    # Simulate some trades for testing
    print("\n\nSimulating trades for demonstration...")
    
    models = list(weights.keys())
    for i in range(20):
        # Create fake trade outcome
        model = random.choice(models)
        is_profitable = random.random() > 0.45  # 55% win rate
        
        outcome = TradeOutcome(
            trade_id=f"test_{i}",
            position_id=f"pos_{i}",
            symbol=random.choice(['BTC/USDT', 'SOL/USDT']),
            side=random.choice(['buy', 'sell']),
            entry_price=50000 + random.uniform(-5000, 5000),
            exit_price=50000 + random.uniform(-5000, 5000),
            entry_time=datetime.now() - timedelta(hours=random.randint(1, 48)),
            exit_time=datetime.now(),
            pnl=random.uniform(-100, 200) if is_profitable else random.uniform(-200, -50),
            pnl_percentage=random.uniform(0.5, 3) if is_profitable else random.uniform(-3, -0.5),
            duration_hours=random.uniform(0.5, 24),
            exit_reason=random.choice(['STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL']),
            model_source=model,
            confidence=random.uniform(0.6, 0.9),
            market_regime=random.choice(['trending_up', 'volatile', 'sideways'])
        )
        
        print(f"  Trade {i+1}: {model} - {'WIN' if is_profitable else 'LOSS'}")
    
    # Force weight update
    print("\nUpdating weights based on simulated performance...")
    new_weights = learner.update_model_weights(force=True)
    
    print("\nUpdated Model Weights:")
    for model, weight in sorted(new_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {weight:.3f}")
    
    # Show learning history
    print("\nRecent Learning History:")
    history = learner.get_learning_history(7)
    if not history.empty:
        print(history[['timestamp', 'model_name', 'old_weight', 'new_weight']].head())


if __name__ == "__main__":
    main()