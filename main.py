"""
Fixed Integrated Crypto Trading System with Continuous Price Updates
- Continuously updates current prices for position management
- Properly checks exit conditions (TP/SL)
- Manages data and trains models automatically
- Fixed asyncio issues
"""
import os
import sys
import time
import signal
import threading
import logging
import json
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import shutil
import schedule

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import components
import ccxt.async_support as ccxt
import ccxt as ccxt_sync  # For synchronous price fetching
import aiohttp
import sqlite3
import ta
from pycoingecko import CoinGeckoAPI

# Import other system components
from ensemble_manager import EnsembleManager
from feedback_learner import FeedbackLearner, TradeOutcome
from complete_prediction_models import ModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "integrated_trading_system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegratedTradingSystem")


# Simple Webhook Sender Class (unchanged)
class SimpleWebhookSender:
    """Simple webhook sender with retry logic"""
    
    def __init__(self, btc_webhook_url: str = "https://api.primeautomation.ai/webhook/ChartPrime/bafbdc00-a670-48ad-9624-c7c059f2c385",
                 sol_webhook_url: str = "https://api.primeautomation.ai/webhook/ChartPrime/ca60dbfd-46b9-4a44-bdec-43c0a024a379"):
        self.webhook_urls = {
            'BTC/USDT': btc_webhook_url,
            'SOL/USDT': sol_webhook_url
        }
        self.max_retries = 3
        self.retry_delay = 1
        
    def send_trade_signal(self, symbol: str, action: str, price: float, 
                         confidence: float = 0.8, **kwargs) -> bool:
        """Send a trading signal webhook (buy/sell)"""
        if action not in ['buy', 'sell']:
            logger.error(f"Invalid action: {action}")
            return False
            
        webhook_url = self.webhook_urls.get(symbol)
        if not webhook_url:
            logger.error(f"No webhook URL configured for {symbol}")
            return False
        
        ticker = symbol.replace('/', '')
        
        payload = {
            "ticker": ticker,
            "action": action,
            "price": str(price),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": "11.2h",
            "confidence": f"{confidence:.2f}",
            "regime": kwargs.get("regime", "Ranging")
        }
        
        return self._send_with_retry(webhook_url, payload, f"{action.upper()} {symbol}")
    
    def send_exit_signal(self, symbol: str, action: str, price: float,
                        size: float = 1.0, sl: float = None, tp: float = None,
                        reason: str = "", per: float = None) -> bool:
        """Send an exit signal webhook"""
        if action not in ['exit_buy', 'exit_sell']:
            logger.error(f"Invalid exit action: {action}")
            return False
        
        webhook_url = self.webhook_urls.get(symbol)
        if not webhook_url:
            logger.error(f"No webhook URL configured for {symbol}")
            return False
        
        if per is None:
            if reason == 'tp_hit':
                per = 100
            elif reason == 'sl_hit':
                per = 0
            else:
                per = 50
        
        ticker = symbol.replace('/', '')
        
        payload = {
            "ticker": ticker,
            "action": action,
            "price": str(price),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size": str(size),
            "per": f"{per}%",
            "sl": str(sl) if sl else "",
            "tp": str(tp) if tp else ""
        }
        
        if reason:
            payload["reason"] = reason
        
        return self._send_with_retry(webhook_url, payload, f"{action.upper()} {symbol}")
    
    def _send_with_retry(self, webhook_url: str, payload: Dict[str, Any], 
                        description: str) -> bool:
        """Send webhook with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending webhook for {description} (attempt {attempt + 1}/{self.max_retries})")
                logger.debug(f"Payload: {payload}")
                
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Webhook sent successfully: {response.text}")
                    return True
                else:
                    logger.error(f"Webhook failed with status {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Webhook error (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to send webhook after {self.max_retries} attempts")
        return False


@dataclass
class IntegratedSystemConfig:
    """Complete system configuration with adjusted thresholds"""
    script_dir: str = SCRIPT_DIR
    
    # Database
    db_path: str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "crypto_trading.db"))
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "SOL/USDT"])
    
    # Data collection settings - MORE FREQUENT UPDATES
    market_data_interval: int = 30  # 30 seconds for market data
    price_update_interval: int = 10  # 10 seconds for price checks
    technical_indicators_interval: int = 60  # 1 minute
    onchain_interval: int = 300  # 5 minutes
    sentiment_interval: int = 300  # 5 minutes
    historical_days: int = 30
    
    # Trading intervals
    prediction_interval: int = 120  # 2 minutes between predictions
    position_check_interval: int = 10  # Check positions every 10 seconds
    health_check_interval: int = 60  # 1 minute health checks
    model_retrain_interval: int = 86400  # 24 hours model retraining
    
    # Risk management - ADJUSTED CONFIDENCE THRESHOLDss
    max_concurrent_positions: int = 2
    daily_loss_limit: float = 0.05  # 5%
    confidence_threshold: float = 0.90  # Lowered from 0.90 to 0.60
    
    # Simple webhook configuration
    btc_webhook_url: str = "https://api.primeautomation.ai/webhook/ChartPrime/bafbdc00-a670-48ad-9624-c7c059f2c385"
    sol_webhook_url: str = "https://api.primeautomation.ai/webhook/ChartPrime/ca60dbfd-46b9-4a44-bdec-43c0a024a379"
    enable_webhooks: bool = True
    
    # Features
    paper_trading: bool = True
    auto_training: bool = True
    enable_feedback_learning: bool = True
    auto_cleanup_days: int = 90  # Auto cleanup old data
    
    # API Keys
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    coingecko_api_key: str = field(default_factory=lambda: os.getenv('COINGECKO_API_KEY', ''))
    
    # Startup behavior
    collect_initial_data: bool = True
    train_on_startup: bool = True
    min_data_points_for_training: int = 100
    min_data_points_for_helformer: int = 500
    
    # Database migration
    backup_db: bool = True
    force_recreate_db: bool = False
    
    def __post_init__(self):
        """Ensure all directories use script directory as base"""
        self.data_dir = os.path.join(self.script_dir, "data")
        self.models_dir = os.path.join(self.script_dir, "models")
        self.analytics_dir = os.path.join(self.script_dir, "analytics")
        self.logs_dir = os.path.join(self.script_dir, "logs")


# Enhanced Price Manager for continuous updates
class LivePriceManager:
    """Manages live price updates using synchronous ccxt"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.prices = {}
        self.last_update = {}
        self.exchange = ccxt_sync.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.is_running = False
        self.update_thread = None
        
    def start(self):
        """Start price update thread"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Started live price manager")
        
    def stop(self):
        """Stop price updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
            
    def _update_loop(self):
        """Continuously update prices"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        self.prices[symbol] = ticker['last']
                        self.last_update[symbol] = datetime.now()
                        logger.debug(f"Updated {symbol} price: ${ticker['last']:,.2f}")
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {e}")
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Price update loop error: {e}")
                time.sleep(10)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        # Return cached price if recent (< 10 seconds old)
        if symbol in self.prices and symbol in self.last_update:
            age = (datetime.now() - self.last_update[symbol]).total_seconds()
            if age < 10:
                return self.prices[symbol]
        
        # Otherwise fetch fresh price
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.prices[symbol] = ticker['last']
            self.last_update[symbol] = datetime.now()
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return self.prices.get(symbol)


class DatabaseMigrator:
    """Handles database schema migration and fixes"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def backup_database(self):
        """Create backup of existing database"""
        if os.path.exists(self.db_path):
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            
    def check_database_exists(self) -> bool:
        """Check if database exists and has data"""
        if not os.path.exists(self.db_path):
            return False
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM market_data")
                count = cursor.fetchone()[0]
                return count > 0
        except:
            return False


class IntegratedDataCollector:
    """Enhanced data collector with live price updates"""
    
    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        self.is_running = False
        self.binance_exchange = None
        self.coingecko = CoinGeckoAPI(api_key=config.coingecko_api_key) if config.coingecko_api_key else None
        
        # Timing controls
        self.last_market_update = {}
        self.last_technical_update = {}
        self.last_onchain_update = {}
        self.last_sentiment_update = {}
        
        # Create necessary directories
        self._create_directories()
        
        # Handle database migration if needed
        self._handle_database_setup()
        
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            os.path.join(self.config.models_dir, "ensemble"),
            os.path.join(self.config.models_dir, "boruta_cnn_lstm"),
            os.path.join(self.config.models_dir, "helformer"),
            os.path.join(self.config.models_dir, "temporal_fusion"),
            os.path.join(self.config.models_dir, "sentiment_analyzer"),
            self.config.analytics_dir,
            os.path.join(self.config.analytics_dir, "feedback"),
            self.config.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Created directories in: {self.config.script_dir}")
    
    def _handle_database_setup(self):
        """Handle database migration and setup"""
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        migrator = DatabaseMigrator(self.config.db_path)
        
        if os.path.exists(self.config.db_path):
            if self.config.backup_db:
                migrator.backup_database()
        
        self._init_complete_database()
    
    def _init_complete_database(self):
        """Initialize ALL required database tables with correct schema"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            # All table creation code (same as before)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    trades_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    atr REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    volatility REAL,
                    adx REAL,
                    cci REAL,
                    roc REAL,
                    williams_r REAL,
                    obv REAL,
                    vwap REAL,
                    pivot REAL,
                    resistance_1 REAL,
                    support_1 REAL,
                    resistance_2 REAL,
                    support_2 REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS onchain_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    market_cap REAL,
                    total_volume REAL,
                    circulating_supply REAL,
                    max_supply REAL,
                    active_addresses INTEGER,
                    transaction_count INTEGER,
                    exchange_inflow REAL,
                    exchange_outflow REAL,
                    exchange_netflow REAL,
                    total_value_locked REAL,
                    defi_dominance REAL,
                    staking_ratio REAL,
                    hash_rate REAL,
                    mining_difficulty REAL,
                    network_fees REAL,
                    average_transaction_fee REAL,
                    gas_used REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    fear_greed_index REAL,
                    social_volume INTEGER,
                    social_dominance REAL,
                    news_volume INTEGER,
                    reddit_mentions INTEGER,
                    twitter_mentions INTEGER,
                    google_trends REAL,
                    market_momentum REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nft_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    total_market_cap REAL,
                    total_volume_24h REAL,
                    total_sales_24h INTEGER,
                    floor_price_trend REAL,
                    volume_momentum REAL,
                    chain_dominance REAL,
                    top_collections TEXT,
                    unique_buyers INTEGER,
                    unique_sellers INTEGER,
                    average_price REAL,
                    median_price REAL,
                    blue_chip_index REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, chain)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    bid_volume REAL,
                    ask_volume REAL,
                    spread REAL,
                    imbalance REAL,
                    depth_20_bid REAL,
                    depth_20_ask REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unified_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    volume_1h REAL,
                    rsi REAL,
                    macd REAL,
                    volatility_1h REAL,
                    orderbook_imbalance REAL,
                    buy_sell_ratio REAL,
                    nft_sentiment REAL,
                    exchange_netflow REAL,
                    sentiment_score REAL,
                    fear_greed_index REAL,
                    technical_rating REAL,
                    onchain_rating REAL,
                    feature_vector TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracked_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completed_trades (
                    id TEXT PRIMARY KEY,
                    position_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    duration_minutes REAL DEFAULT 0,
                    exit_reason TEXT,
                    model_source TEXT,
                    confidence REAL DEFAULT 0,
                    market_regime TEXT,
                    fees_paid REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_features_symbol_timestamp ON unified_features(symbol, timestamp)")
            
            conn.commit()
            logger.info("Database initialized with complete schema")
    
    async def initialize(self):
        """Initialize exchange connections"""
        try:
            self.binance_exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            await self.binance_exchange.load_markets()
            logger.info("Binance connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Binance: {e}")
            return False
        finally:
            # Always close the connection properly
            if self.binance_exchange:
                await self.binance_exchange.close()
    
    async def collect_historical_data(self, symbol: str, days: int):
        """Collect historical data for initial training"""
        exchange = None
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            await exchange.load_markets()
            
            logger.info(f"Collecting {days} days of historical data for {symbol}")
            
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            timeframe = '1h'
            
            all_ohlcv = []
            while True:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                if len(ohlcv) < 1000:
                    break
                    
                since = ohlcv[-1][0] + 1
                await asyncio.sleep(0.1)
            
            # Store in database
            with sqlite3.connect(self.config.db_path) as conn:
                for candle in all_ohlcv:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, timeframe, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'), timeframe,
                        candle[1], candle[2], candle[3], candle[4], candle[5]
                    ))
                
                conn.commit()
            
            logger.info(f"Collected {len(all_ohlcv)} candles for {symbol}")
            
            # Process all historical data
            await self.process_historical_data(symbol)
            
            return len(all_ohlcv)
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return 0
        finally:
            if exchange:
                await exchange.close()
    
    async def update_market_data(self, symbol: str):
        """Update latest market data with fresh prices"""
        exchange = None
        try:
            current_time = time.time()
            
            if symbol in self.last_market_update:
                if current_time - self.last_market_update[symbol] < self.config.market_data_interval:
                    return
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            await exchange.load_markets()
            
            # Get latest candles
            ohlcv = await exchange.fetch_ohlcv(symbol, '1m', limit=5)  # Get last 5 minutes
            
            if ohlcv:
                # Store all recent candles
                with sqlite3.connect(self.config.db_path) as conn:
                    for candle in ohlcv:
                        timestamp = datetime.fromtimestamp(candle[0] / 1000)
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO market_data 
                            (symbol, timestamp, timeframe, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'), '1m',
                            candle[1], candle[2], candle[3], candle[4], candle[5]
                        ))
                    
                    # Also update with current price as latest entry
                    ticker = await exchange.fetch_ticker(symbol)
                    current_timestamp = datetime.now()
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, timeframe, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, current_timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'current',
                        ticker['last'], ticker['high'], ticker['low'], ticker['last'], ticker['quoteVolume']
                    ))
                    
                    conn.commit()
                
                logger.debug(f"Updated {symbol} market data with current price: ${ticker['last']:,.2f}")
            
            self.last_market_update[symbol] = current_time
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
        finally:
            if exchange:
                await exchange.close()
    
    async def process_historical_data(self, symbol: str):
        """Process historical data to create indicators and features"""
        try:
            # Get all historical data
            with sqlite3.connect(self.config.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timeframe = '1h'
                    ORDER BY timestamp
                """, conn, params=(symbol,))
            
            if len(df) < 50:
                logger.warning(f"Not enough data to process for {symbol}")
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Calculate all technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            returns = df['close'].pct_change().dropna()
            df['volatility'] = returns.rolling(window=24).std() * np.sqrt(24)
            
            # Store all indicators and create unified features
            with sqlite3.connect(self.config.db_path) as conn:
                for idx, row in df.iterrows():
                    if pd.notna(row['rsi']):
                        timestamp_str = idx.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Store technical indicators
                        conn.execute("""
                            INSERT OR REPLACE INTO technical_indicators 
                            (symbol, timestamp, timeframe, rsi, macd, macd_signal, macd_histogram,
                             bb_upper, bb_middle, bb_lower, atr, sma_20, sma_50,
                             ema_12, ema_26, volatility)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, timestamp_str, '1h',
                            row['rsi'], row['macd'], row['macd_signal'], row['macd_histogram'],
                            row['bb_upper'], row['bb_middle'], row['bb_lower'], row['atr'],
                            row['sma_20'], row['sma_50'], row['ema_12'], row['ema_26'], row['volatility']
                        ))
                        
                        # Create unified features
                        conn.execute("""
                            INSERT OR REPLACE INTO unified_features 
                            (symbol, timestamp, price, volume_1h, rsi, macd, volatility_1h,
                             orderbook_imbalance, buy_sell_ratio, nft_sentiment, exchange_netflow,
                             sentiment_score, fear_greed_index, technical_rating, onchain_rating)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, timestamp_str, row['close'], row['volume'],
                            row['rsi'], row['macd'], row['volatility'],
                            0.0, 1.0, 0.0, 0.0, 0.0, 50.0,
                            self._calculate_technical_rating(row), 0.0
                        ))
                
                # Create sentiment and onchain data
                for i in range(0, len(df), 6):
                    timestamp_str = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO sentiment_data 
                        (symbol, timestamp, source, sentiment_score, sentiment_label,
                         fear_greed_index, social_volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timestamp_str, 'aggregate',
                        0.0, 'neutral', 50.0, 1000
                    ))
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO onchain_data 
                        (symbol, timestamp, market_cap, total_volume, active_addresses,
                         exchange_netflow, network_fees)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timestamp_str,
                        1000000000.0, 10000000.0, 50000, 0.0, 1.0
                    ))
                
                conn.commit()
            
            logger.info(f"Processed {len(df)} historical records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing historical data for {symbol}: {e}")
    
    def _calculate_technical_rating(self, row) -> float:
        """Calculate technical rating from indicators"""
        rating = 0.0
        
        if pd.notna(row.get('rsi')):
            if row['rsi'] < 30:
                rating += 1
            elif row['rsi'] > 70:
                rating -= 1
        
        if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
            if row['macd'] > row['macd_signal']:
                rating += 0.5
            else:
                rating -= 0.5
        
        return np.tanh(rating)
    
    async def process_latest_data(self, symbol: str):
        """Process latest data to create indicators and unified features"""
        try:
            # Get recent data
            with sqlite3.connect(self.config.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND (timeframe = '1h' OR timeframe = '1m' OR timeframe = 'current')
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, conn, params=(symbol,))
            
            if len(df) < 50:
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Calculate indicators for the latest data
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['volatility'] = df['close'].pct_change().rolling(24).std() * np.sqrt(24)
            
            # Store the latest indicators
            latest = df.iloc[-1]
            timestamp_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            with sqlite3.connect(self.config.db_path) as conn:
                # Create unified features with latest price
                conn.execute("""
                    INSERT OR REPLACE INTO unified_features 
                    (symbol, timestamp, price, volume_1h, rsi, macd, volatility_1h,
                     orderbook_imbalance, buy_sell_ratio, nft_sentiment, exchange_netflow,
                     sentiment_score, fear_greed_index, technical_rating, onchain_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timestamp_str, latest['close'], latest['volume'],
                    latest['rsi'] if pd.notna(latest['rsi']) else 50.0,
                    latest['macd'] if pd.notna(latest['macd']) else 0.0,
                    latest['volatility'] if pd.notna(latest['volatility']) else 0.02,
                    0.0, 1.0, 0.0, 0.0, 0.0, 50.0,
                    self._calculate_technical_rating(latest), 0.0
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error processing latest data for {symbol}: {e}")
    
    def check_data_availability(self, symbol: str) -> Dict[str, int]:
        """Check data availability across all tables"""
        counts = {}
        
        with sqlite3.connect(self.config.db_path) as conn:
            tables = ['market_data', 'technical_indicators', 'sentiment_data', 
                     'onchain_data', 'unified_features']
            
            for table in tables:
                cursor = conn.execute(f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE symbol = ?
                """, (symbol,))
                
                counts[table] = cursor.fetchone()[0]
        
        return counts
    
    async def check_and_collect_data(self, symbol: str) -> bool:
        """Check if we have enough data and collect if needed"""
        counts = self.check_data_availability(symbol)
        unified_count = counts.get('unified_features', 0)
        
        logger.info(f"{symbol} data availability: {counts}")
        
        required_data = self.config.min_data_points_for_helformer
        
        if unified_count < required_data:
            logger.info(f"Need {required_data - unified_count} more data points for {symbol}")
            
            current_data_days = unified_count / 24
            needed_days = max(30, int((required_data / 24) * 1.2))
            
            logger.info(f"Collecting {needed_days} days of historical data for {symbol}...")
            await self.collect_historical_data(symbol, needed_days)
            
            counts = self.check_data_availability(symbol)
            unified_count = counts.get('unified_features', 0)
            
        return unified_count >= self.config.min_data_points_for_training


class IntegratedTradingOrchestrator:
    """Main orchestrator with enhanced price tracking and position management"""
    
    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.data_collector = IntegratedDataCollector(config)
        self.ensemble_manager = None
        self.feedback_learner = None
        
        # Initialize live price manager
        self.price_manager = LivePriceManager(config.symbols)
        
        # Initialize webhook sender
        self.webhook_sender = SimpleWebhookSender(
            btc_webhook_url=config.btc_webhook_url,
            sol_webhook_url=config.sol_webhook_url
        )
        
        # Position tracking
        self.tracked_positions = {}
        
        # Timing control
        self.last_prediction_time = 0
        self.last_position_check_time = 0
        self.last_health_check_time = 0
        self.last_cleanup_time = time.time()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Integrated trading orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()
        self.shutdown()
    
    async def continuous_data_updater(self):
        """Continuously update market data"""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    await self.data_collector.update_market_data(symbol)
                    await self.data_collector.process_latest_data(symbol)
                
                await asyncio.sleep(self.config.market_data_interval)
                
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(30)
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing ensemble manager...")
            self.ensemble_manager = EnsembleManager(self.config.db_path)
            logger.info("✅ Ensemble manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ensemble manager: {e}")
            raise
            
        try:
            if self.config.enable_feedback_learning:
                logger.info("Initializing feedback learner...")
                self.feedback_learner = FeedbackLearner(self.config.db_path)
                logger.info("✅ Feedback learner initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize feedback learner: {e}")
            self.feedback_learner = None
        
        # Start live price manager
        self.price_manager.start()
        logger.info("✅ Live price manager started")
        
        logger.info("All components initialized successfully")
        
        # Train models if requested
        if self.config.train_on_startup:
            try:
                self.train_all_models()
            except Exception as e:
                logger.error(f"❌ Model training failed: {e}")
                logger.warning("Continuing without model training...")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("Starting model training...")
        
        for symbol in self.config.symbols:
            try:
                data_counts = self.data_collector.check_data_availability(symbol)
                unified_count = data_counts.get('unified_features', 0)
                
                if unified_count < self.config.min_data_points_for_training:
                    logger.warning(f"Insufficient data for training {symbol}: {unified_count} unified features")
                    continue
                
                logger.info(f"Training models for {symbol} with {unified_count} data points")
                
                try:
                    # Train individual models
                    logger.info(f"Training individual models for {symbol}...")
                    self.ensemble_manager.train_individual_models(symbol)
                    
                    # Train ensemble
                    logger.info(f"Training ensemble for {symbol}...")
                    ensemble_success = self.ensemble_manager.train_ensemble(symbol)
                    
                    if ensemble_success:
                        logger.info(f"✅ All training completed successfully for {symbol}")
                    else:
                        logger.warning(f"⚠️ Ensemble training failed for {symbol}, but continuing...")
                        
                except Exception as e:
                    logger.error(f"❌ Error training models for {symbol}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Error checking data for {symbol}: {e}")
                continue
        
        logger.info("Model training phase completed")
        
        # Update model weights
        try:
            if self.feedback_learner:
                logger.info("Updating model weights...")
                self.feedback_learner.update_model_weights(force=True)
                logger.info("✅ Model weights updated successfully")
        except Exception as e:
            logger.error(f"❌ Error updating model weights: {e}")
    
    def track_position(self, symbol: str, action: str, entry_price: float, confidence: float = 0.9):
        """Track position for exit signal generation"""
        if action == 'buy':
            stop_loss = entry_price * 0.99  # 3% SL
            take_profit = entry_price * 1.01  # 4% TP
        else:  # sell
            stop_loss = entry_price * 1.01  # 3% SL
            take_profit = entry_price * 0.99  # 4% TP
        
        self.tracked_positions[symbol] = {
            'side': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'confidence': confidence
        }
        
        # Save to database
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                conn.execute("""
                    INSERT INTO tracked_positions 
                    (symbol, side, entry_price, stop_loss, take_profit, entry_time, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, action, entry_price, stop_loss, take_profit,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'active'
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving tracked position: {e}")
    
    def remove_tracked_position(self, symbol: str):
        """Remove tracked position"""
        if symbol in self.tracked_positions:
            del self.tracked_positions[symbol]
            
            try:
                with sqlite3.connect(self.config.db_path) as conn:
                    conn.execute("""
                        UPDATE tracked_positions 
                        SET status = 'closed' 
                        WHERE symbol = ? AND status = 'active'
                    """, (symbol,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error updating tracked position: {e}")
    
    def check_exit_conditions(self, position_info: Dict, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if position_info['side'] == 'buy':
            if current_price >= position_info['take_profit']:
                return True, 'tp_hit'
            elif current_price <= position_info['stop_loss']:
                return True, 'sl_hit'
        else:  # sell
            if current_price <= position_info['take_profit']:
                return True, 'tp_hit'
            elif current_price >= position_info['stop_loss']:
                return True, 'sl_hit'
        
        return False, ''
    
    def check_exit_signals(self):
        """Check and send exit signals for tracked positions using LIVE prices"""
        current_time = time.time()
        
        # Check positions more frequently
        if current_time - self.last_position_check_time < self.config.position_check_interval:
            return
            
        try:
            positions_to_check = list(self.tracked_positions.items())
            
            for symbol, position_info in positions_to_check:
                try:
                    # Get LIVE price from price manager
                    current_price = self.price_manager.get_current_price(symbol)
                    
                    if not current_price:
                        logger.warning(f"Could not get current price for {symbol}")
                        continue
                    
                    # Log current status
                    logger.debug(f"{symbol} position check - Current: ${current_price:,.2f}, "
                               f"TP: ${position_info['take_profit']:,.2f}, "
                               f"SL: ${position_info['stop_loss']:,.2f}")
                    
                    # Check exit conditions
                    should_exit, reason = self.check_exit_conditions(position_info, current_price)
                    
                    if should_exit:
                        logger.info(f"📊 Exit signal detected for {symbol}: {reason} at ${current_price:,.2f}")
                        
                        # Determine exit action
                        if position_info['side'] == 'buy':
                            exit_action = 'exit_buy'
                        else:
                            exit_action = 'exit_sell'
                        
                        # Send exit webhook
                        if self.config.enable_webhooks:
                            success = self.webhook_sender.send_exit_signal(
                                symbol=symbol,
                                action=exit_action,
                                price=current_price,
                                sl=position_info.get('stop_loss'),
                                tp=position_info.get('take_profit'),
                                reason=reason,
                                per=100 if reason == 'tp_hit' else 0
                            )
                            
                            if success:
                                logger.info(f"✅ Exit signal sent for {symbol}")
                                self.remove_tracked_position(symbol)
                            else:
                                logger.warning(f"⚠️ Failed to send exit signal for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error checking exit for {symbol}: {e}")
                    continue
            
            self.last_position_check_time = current_time
                    
        except Exception as e:
            logger.error(f"Error in check_exit_signals: {e}")
    
    def run_prediction_cycle(self):
        """Run prediction cycle"""
        current_time = time.time()
        
        if current_time - self.last_prediction_time < self.config.prediction_interval:
            return
        
        try:
            logger.info("=" * 60)
            logger.info(f"🔄 Running prediction cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for symbol in self.config.symbols:
                try:
                    logger.info(f"\n📈 Processing {symbol}...")
                    
                    # Check if we already have a position
                    if symbol in self.tracked_positions:
                        logger.info(f"Already have position for {symbol}, skipping new predictions")
                        continue
                    
                    # Get ensemble decision
                    decision = self.ensemble_manager.make_final_decision(symbol)
                    
                    if decision is None:
                        logger.info(f"❌ No decision generated for {symbol}")
                        continue
                        
                    logger.info(f"📊 Decision for {symbol}: {decision.get('action', 'none')} "
                              f"(confidence: {decision.get('confidence', 0):.3f})")
                    
                    if decision.get('action') == 'hold':
                        logger.info(f"⏸️ HOLD signal for {symbol} - no action taken")
                        continue
                    
                    # Get LIVE price from price manager
                    current_price = self.price_manager.get_current_price(symbol)
                    
                    if not current_price:
                        logger.error(f"❌ Could not get current price for {symbol}")
                        continue
                    
                    # Calculate TP and SL
                    if decision['action'] == 'buy':
                        stop_loss = current_price * 0.99  # 3% SL
                        take_profit = current_price * 1.01  # 4% TP
                    else:  # sell
                        stop_loss = current_price * 1.01  # 3% SL
                        take_profit = current_price * 0.99  # 4% TP
                    
                    # Log trade details
                    logger.info(f"""
        📍 TRADE SIGNAL DETAILS for {symbol}:
        Action: {decision['action'].upper()}
        Price: ${current_price:,.2f}
        Stop Loss: ${stop_loss:,.2f} ({abs((stop_loss/current_price - 1) * 100):.1f}%)
        Take Profit: ${take_profit:,.2f} ({abs((take_profit/current_price - 1) * 100):.1f}%)
        Confidence: {decision.get('confidence', 0):.3f}
        Market Regime: {decision.get('market_regime', 'Ranging')}
                    """)
                    
                    # Check confidence threshold
                    if decision.get('confidence', 0) >= self.config.confidence_threshold:
                        logger.info(f"🎯 Signal meets confidence threshold ({self.config.confidence_threshold})")
                        
                        # Send webhook
                        if self.config.enable_webhooks:
                            success = self.webhook_sender.send_trade_signal(
                                symbol=symbol,
                                action=decision['action'],
                                price=current_price,
                                confidence=decision.get('confidence', 0.9),
                                regime=decision.get('market_regime', 'Ranging')
                            )
                            
                            if success:
                                logger.info(f"✅ Trade signal sent for {symbol}")
                                self.track_position(symbol, decision['action'], current_price, 
                                                  decision.get('confidence', 0.9))
                            else:
                                logger.warning(f"⚠️ Failed to send trade signal for {symbol}")
                    else:
                        logger.info(f"📊 Confidence {decision.get('confidence', 0):.3f} below "
                                  f"threshold {self.config.confidence_threshold} for {symbol}")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info("=" * 60)
            
            self.last_prediction_time = current_time
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
    
    def cleanup_old_data(self):
        """Cleanup old data from database"""
        try:
            current_time = time.time()
            
            # Run cleanup once per day
            if current_time - self.last_cleanup_time < 86400:
                return
            
            logger.info("Running database cleanup...")
            
            cutoff_date = datetime.now() - timedelta(days=self.config.auto_cleanup_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
            
            with sqlite3.connect(self.config.db_path) as conn:
                tables = ['market_data', 'orderbook_data', 'technical_indicators']
                
                for table in tables:
                    cursor = conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_str,))
                    logger.info(f"Cleaned {cursor.rowcount} old records from {table}")
                
                conn.commit()
            
            self.last_cleanup_time = current_time
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def scheduled_model_retrain(self):
        """Schedule model retraining"""
        try:
            logger.info("🔄 Starting scheduled model retraining...")
            self.train_all_models()
            logger.info("✅ Scheduled model retraining completed")
        except Exception as e:
            logger.error(f"Error during scheduled retraining: {e}")
    
    def load_tracked_positions(self):
        """Load tracked positions from database on startup"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, side, entry_price, stop_loss, take_profit, entry_time
                    FROM tracked_positions
                    WHERE status = 'active'
                """)
                
                for row in cursor.fetchall():
                    symbol = row[0]
                    self.tracked_positions[symbol] = {
                        'side': row[1],
                        'entry_price': row[2],
                        'stop_loss': row[3],
                        'take_profit': row[4],
                        'entry_time': datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
                    }
                    
                logger.info(f"Loaded {len(self.tracked_positions)} tracked positions")
                
        except Exception as e:
            logger.error(f"Error loading tracked positions: {e}")
    
    def start_trading(self):
        """Start the integrated trading system with continuous operation"""
        logger.info("🚀 Starting integrated crypto trading system...")
        
        try:
            # Load existing tracked positions
            self.load_tracked_positions()
            
            # Check if we need to collect initial data
            needs_data = False
            for symbol in self.config.symbols:
                counts = self.data_collector.check_data_availability(symbol)
                logger.info(f"{symbol} data available: {counts}")
                if counts.get('unified_features', 0) < self.config.min_data_points_for_training:
                    needs_data = True
                    break
            
            if needs_data:
                logger.info("📊 Need to collect initial market data...")
                
                # Run async data collection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def collect_data():
                    if not await self.data_collector.initialize():
                        raise Exception("Failed to initialize data collector")
                    
                    for symbol in self.config.symbols:
                        logger.info(f"Collecting historical data for {symbol}...")
                        collected = await self.data_collector.collect_historical_data(symbol, 60)
                        logger.info(f"Collected {collected} data points for {symbol}")
                    
                    return True
                
                try:
                    success = loop.run_until_complete(collect_data())
                    if not success:
                        raise Exception("Data collection failed")
                finally:
                    loop.close()
                
                logger.info("✅ Initial data collection completed")
            
            # Initialize all components
            self.initialize_system()
            
            # Start continuous data updater in background
            def run_async_updater():
                """Run async data updater in background"""
                update_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(update_loop)
                
                try:
                    update_loop.run_until_complete(self.continuous_data_updater())
                except Exception as e:
                    logger.error(f"Data updater error: {e}")
                finally:
                    update_loop.close()
            
            # Start data updater thread
            data_thread = threading.Thread(target=run_async_updater, daemon=True)
            data_thread.start()
            logger.info("📊 Started continuous data updater")
            
            # Schedule periodic tasks
            schedule.every(24).hours.do(self.scheduled_model_retrain)
            schedule.every(12).hours.do(self.cleanup_old_data)
            
            self.is_running = True
            
            logger.info("✅ All systems operational!")
            logger.info(f"📊 Monitoring {len(self.config.symbols)} symbols: {self.config.symbols}")
            logger.info(f"🎯 Confidence threshold: {self.config.confidence_threshold}")
            logger.info(f"🌐 BTC webhooks: {self.config.btc_webhook_url}")
            logger.info(f"🌐 SOL webhooks: {self.config.sol_webhook_url}")
            logger.info("Press Ctrl+C to stop...")
            
            # Main loop
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Run scheduled tasks
                    schedule.run_pending()
                    
                    # Run prediction cycle
                    self.run_prediction_cycle()
                    
                    # Check exit signals with live prices
                    self.check_exit_signals()
                    
                    # Cleanup old data periodically
                    self.cleanup_old_data()
                    
                    # Check for shutdown event
                    if self.shutdown_event.wait(timeout=5):
                        break
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(10)
            
            logger.info("Main loop ended, shutting down...")
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        if not self.is_running:
            return
            
        logger.info("🔽 Shutting down integrated trading system...")
        
        self.is_running = False
        
        # Stop price manager
        if self.price_manager:
            self.price_manager.stop()
        
        logger.info("✅ Shutdown complete")
        
        # Exit the program
        os._exit(0)


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     Integrated Crypto Trading System v3.0         ║
    ║    With Continuous Price Updates & Auto-Trading   ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # Create configuration with adjusted settings
    config = IntegratedSystemConfig()
    
    # Check if database exists
    migrator = DatabaseMigrator(config.db_path)
    if not migrator.check_database_exists():
        print("📊 Database not found. System will collect initial data...")
        print(f"📁 This may take a few minutes for {len(config.symbols)} symbols")
    
    # Create orchestrator
    orchestrator = IntegratedTradingOrchestrator(config)
    
    try:
        logger.info("🚀 Starting trading system with continuous price monitoring...")
        orchestrator.start_trading()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
