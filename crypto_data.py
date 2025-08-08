"""
Unified Data Collection System with SQL Storage for Crypto Trading Bot
Comprehensive data collection from multiple sources for model training
Fixed for Python 3.12 SQLite datetime handling and WebSocket stability
Updated with official CoinGecko SDK for proper API authentication

PURPOSE: This script collects and stores crypto market data from various sources
for training machine learning models. Model training is handled in separate scripts.

DATA SOURCES:
1. Binance: Market data, order books, trade flow
2. CoinGecko: On-chain metrics, NFT data, market analytics
3. Alternative.me: Fear & Greed Index
4. Technical Indicators: Calculated from market data

TROUBLESHOOTING COINGECKO API ISSUES:
If you're getting 'NoneType' errors with CoinGecko:
1. Run the test script (test-coingecko-connection.py) to verify API connectivity
2. Check if your API key has the required permissions for the endpoints
3. Verify you haven't exceeded rate limits (500 calls/minute for Analyst plan)
4. If issues persist, set config.use_coingecko = False to use simulated data

The system will continue to collect Binance market data and technical indicators
even if CoinGecko is unavailable.
"""

import os
import time
import json
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import ccxt.async_support as ccxt
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import websocket
import threading
from queue import Queue
import pickle
from contextlib import contextmanager
import ta
from textblob import TextBlob
import re
from pycoingecko import CoinGeckoAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataFetcher")

# Set specific loggers to WARNING to reduce noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)

# Configuration
@dataclass
class DataConfig:
    # Binance config
    binance_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "SOL/USDT"])
    binance_timeframes: Dict[str, str] = field(default_factory=lambda: {
        "1m": "1m",    # For high-frequency data
        "5m": "5m",    # For short-term patterns
        "1h": "1h",    # For medium-term analysis
        "1d": "1d"     # For long-term patterns
    })
    
    # CoinGecko config (with paid API support)
    coingecko_api_key: str = "CG-2hEjevqhpBRQbNZhy7htkGYX"  # Your Analyst plan API key
    use_coingecko: bool = True  # Set to False to disable CoinGecko and use simulated data
    
    # Storage config
    data_dir: str = "data/unified"
    bronze_dir: str = "data/unified/bronze"  # Raw data
    silver_dir: str = "data/unified/silver"  # Cleaned data
    gold_dir: str = "data/unified/gold"      # Feature-engineered data
    
    # SQL Database config
    db_path: str = "data/crypto_trading.db"
    db_type: str = "sqlite"
    
    # Update frequencies
    realtime_interval: int = 1  # seconds for WebSocket
    market_data_interval: int = 60  # seconds for REST APIs
    onchain_interval: int = 300  # 5 minutes for on-chain data
    nft_interval: int = 600  # 10 minutes for NFT data
    technical_indicators_interval: int = 60  # 1 minute for technical indicators
    sentiment_interval: int = 300  # 5 minutes for sentiment analysis
    
    # Historical data config
    historical_days: int = 90  # For model training
    
    # Feature store config
    feature_ttl: int = 3600  # 1 hour TTL for cached features
    
    # Performance config
    max_queue_size: int = 10000
    batch_insert_size: int = 100
    
    # WebSocket config
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    ws_max_reconnect_delay: int = 60

# Utility functions for datetime handling
def datetime_to_str(dt: Any) -> str:
    """Convert datetime object to string for SQLite"""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, pd.Timestamp):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return str(dt)

def str_to_datetime(s: str) -> datetime:
    """Convert string to datetime object"""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

# Enhanced SQL Database Manager with all required tables
class SQLDatabase:
    """SQL Database manager for crypto data storage with comprehensive schema"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.conn = None
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Market data table
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
            
            # Enhanced Technical indicators table
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
            
            # Enhanced On-chain data table
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
            
            # NFT market data table
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
            
            # Sentiment data table
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
            
            # Order book data table
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
            
            # Trade flow data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    volume REAL,
                    side TEXT,
                    is_buyer_maker BOOLEAN,
                    trade_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Enhanced Unified features table for model consumption
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
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_onchain_data_symbol_timestamp ON onchain_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_data_symbol_timestamp ON sentiment_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_features_symbol_timestamp ON unified_features(symbol, timestamp)")
            
            logger.info("Database initialized successfully with all required tables")
    
    def insert_technical_indicators(self, data: Dict):
        """Insert enhanced technical indicators"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO technical_indicators 
                (symbol, timestamp, timeframe, rsi, macd, macd_signal, macd_histogram,
                 bb_upper, bb_middle, bb_lower, atr, sma_20, sma_50, sma_200, 
                 ema_12, ema_26, volatility, adx, cci, roc, williams_r, obv, vwap,
                 pivot, resistance_1, support_1, resistance_2, support_2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'], 
                datetime_to_str(data['timestamp']), 
                data['timeframe'],
                data.get('rsi'), 
                data.get('macd'), 
                data.get('macd_signal'),
                data.get('macd_histogram'), 
                data.get('bb_upper'), 
                data.get('bb_middle'),
                data.get('bb_lower'), 
                data.get('atr'), 
                data.get('sma_20'),
                data.get('sma_50'), 
                data.get('sma_200'), 
                data.get('ema_12'),
                data.get('ema_26'), 
                data.get('volatility'),
                data.get('adx'),
                data.get('cci'),
                data.get('roc'),
                data.get('williams_r'),
                data.get('obv'),
                data.get('vwap'),
                data.get('pivot'),
                data.get('resistance_1'),
                data.get('support_1'),
                data.get('resistance_2'),
                data.get('support_2')
            ))
    
    def insert_sentiment_data(self, data: Dict):
        """Insert sentiment analysis data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_data 
                (symbol, timestamp, source, sentiment_score, sentiment_label,
                 fear_greed_index, social_volume, social_dominance, news_volume,
                 reddit_mentions, twitter_mentions, google_trends, market_momentum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'], 
                datetime_to_str(data['timestamp']), 
                data['source'],
                data.get('sentiment_score'),
                data.get('sentiment_label'),
                data.get('fear_greed_index'),
                data.get('social_volume'),
                data.get('social_dominance'),
                data.get('news_volume'),
                data.get('reddit_mentions'),
                data.get('twitter_mentions'),
                data.get('google_trends'),
                data.get('market_momentum')
            ))
    
    def insert_onchain_data(self, data: Dict):
        """Insert enhanced on-chain data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO onchain_data 
                (symbol, timestamp, market_cap, total_volume, circulating_supply,
                 max_supply, active_addresses, transaction_count, exchange_inflow,
                 exchange_outflow, exchange_netflow, total_value_locked, defi_dominance,
                 staking_ratio, hash_rate, mining_difficulty, network_fees,
                 average_transaction_fee, gas_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'], 
                datetime_to_str(data['timestamp']), 
                data.get('market_cap'),
                data.get('total_volume'), 
                data.get('circulating_supply'),
                data.get('max_supply'),
                data.get('active_addresses'), 
                data.get('transaction_count'),
                data.get('exchange_inflow'), 
                data.get('exchange_outflow'),
                data.get('exchange_netflow'),
                data.get('total_value_locked'),
                data.get('defi_dominance'),
                data.get('staking_ratio'),
                data.get('hash_rate'),
                data.get('mining_difficulty'),
                data.get('network_fees'),
                data.get('average_transaction_fee'),
                data.get('gas_used')
            ))
    
    def insert_nft_market_data(self, data: Dict):
        """Insert enhanced NFT market data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO nft_market_data 
                (timestamp, chain, total_market_cap, total_volume_24h,
                 total_sales_24h, floor_price_trend, volume_momentum,
                 chain_dominance, top_collections, unique_buyers, unique_sellers,
                 average_price, median_price, blue_chip_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime_to_str(data['timestamp']), 
                data['chain'], 
                data['total_market_cap'],
                data['total_volume_24h'], 
                data['total_sales_24h'],
                data['floor_price_trend'], 
                data['volume_momentum'],
                data['chain_dominance'], 
                data.get('top_collections', '[]'),
                data.get('unique_buyers'),
                data.get('unique_sellers'),
                data.get('average_price'),
                data.get('median_price'),
                data.get('blue_chip_index')
            ))
    
    def insert_market_data(self, data: List[Dict]):
        """Batch insert market data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            processed_data = []
            for d in data:
                processed_data.append((
                    d['symbol'], 
                    datetime_to_str(d['timestamp']), 
                    d['timeframe'],
                    d['open'], 
                    d['high'], 
                    d['low'], 
                    d['close'],
                    d['volume'], 
                    d.get('trades_count', 0)
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, timeframe, open, high, low, close, volume, trades_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, processed_data)
    
    def insert_orderbook_data(self, data: Dict):
        """Insert order book snapshot"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO orderbook_data 
                (symbol, timestamp, best_bid, best_ask, bid_volume, ask_volume, 
                 spread, imbalance, depth_20_bid, depth_20_ask)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'], 
                datetime_to_str(data['timestamp']), 
                data['best_bid'], 
                data['best_ask'],
                data['bid_volume'], 
                data['ask_volume'], 
                data['spread'], 
                data['imbalance'], 
                data.get('depth_20_bid', 0), 
                data.get('depth_20_ask', 0)
            ))
    
    def insert_trade_flow(self, trades: List[Dict]):
        """Batch insert trade flow data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            processed_trades = []
            for t in trades:
                processed_trades.append((
                    t['symbol'], 
                    datetime_to_str(t['timestamp']), 
                    t['price'], 
                    t['volume'],
                    t['side'], 
                    t['is_buyer_maker'], 
                    t.get('trade_id')
                ))
            
            cursor.executemany("""
                INSERT INTO trade_flow 
                (symbol, timestamp, price, volume, side, is_buyer_maker, trade_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, processed_trades)
    
    def insert_unified_features(self, features: Dict):
        """Insert enhanced unified features for models"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize feature vector if present
            feature_vector = json.dumps(features.get('feature_vector', {}))
            
            cursor.execute("""
                INSERT OR REPLACE INTO unified_features 
                (symbol, timestamp, price, volume_1h, rsi, macd, volatility_1h,
                 orderbook_imbalance, buy_sell_ratio, nft_sentiment, exchange_netflow,
                 sentiment_score, fear_greed_index, technical_rating, onchain_rating,
                 feature_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                features['symbol'], 
                datetime_to_str(features['timestamp']), 
                features.get('price'),
                features.get('volume_1h'), 
                features.get('rsi'), 
                features.get('macd'),
                features.get('volatility_1h'), 
                features.get('orderbook_imbalance'),
                features.get('buy_sell_ratio'), 
                features.get('nft_sentiment'),
                features.get('exchange_netflow'), 
                features.get('sentiment_score'),
                features.get('fear_greed_index'),
                features.get('technical_rating'),
                features.get('onchain_rating'),
                feature_vector
            ))
    
    def get_latest_features(self, symbol: str) -> Dict:
        """Get latest features for a symbol"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM unified_features 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('timestamp'):
                    result['timestamp'] = str_to_datetime(result['timestamp'])
                if result.get('feature_vector'):
                    result['feature_vector'] = json.loads(result['feature_vector'])
                return result
            return {}
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical market data"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, timeframe, datetime_to_str(start_date), datetime_to_str(end_date))
            )
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def get_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get recent trades for analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trade_flow 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, limit)
            )
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def get_technical_indicators(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get technical indicators data"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM technical_indicators 
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, timeframe, datetime_to_str(start_date), datetime_to_str(end_date))
            )
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = datetime_to_str(cutoff_date)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            tables = ['market_data', 'orderbook_data', 'trade_flow', 
                     'technical_indicators', 'onchain_data', 'nft_market_data',
                     'sentiment_data']
            
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_str,))
                logger.info(f"Cleaned up {cursor.rowcount} old records from {table}")

# Enhanced CoinGecko Data Provider using official SDK
class CoinGeckoDataProvider:
    """Enhanced CoinGecko data provider using official SDK"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        # Initialize CoinGecko API with authentication
        self.cg = CoinGeckoAPI(api_key=config.coingecko_api_key)
        self.rate_limit_remaining = 500  # Analyst plan limit
        self.last_request_time = 0
        
        logger.info("CoinGecko API initialized with Analyst plan")
    
    async def _rate_limit(self):
        """Handle rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Analyst plan: 500 calls/minute
        if time_since_last < 0.12:  # ~500 calls per minute
            await asyncio.sleep(0.12 - time_since_last)
        
        self.last_request_time = time.time()
    
    def _get_coin_id(self, symbol: str) -> str:
        """Map symbol to CoinGecko coin ID"""
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'MATIC': 'matic-network',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'AVAX': 'avalanche-2',
            'XRP': 'ripple',
            'DOGE': 'dogecoin',
            'SHIB': 'shiba-inu'
        }
        
        base_symbol = symbol.split('/')[0].upper()
        return mapping.get(base_symbol, base_symbol.lower())
    
    async def test_connection(self) -> bool:
        """Test CoinGecko API connection"""
        try:
            # Test with a simple ping endpoint
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cg.ping
            )
            logger.info(f"CoinGecko API ping result: {result}")
            return result is not None
        except Exception as e:
            logger.error(f"CoinGecko API connection test failed: {e}")
            return False
    
    async def fetch_onchain_data(self, symbol: str) -> Dict:
        """Fetch comprehensive on-chain data from CoinGecko"""
        await self._rate_limit()
        
        try:
            coin_id = self._get_coin_id(symbol)
            logger.debug(f"Fetching data for coin_id: {coin_id}")
            
            # Get coin details including on-chain metrics
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.cg.get_coin_by_id(
                    id=coin_id,
                    localization=False,
                    tickers=False,
                    market_data=True,
                    community_data=True,
                    developer_data=True,
                    sparkline=False
                )
            )
            
            if data is None:
                logger.warning(f"No data returned from CoinGecko for {symbol}")
                return self._create_default_onchain_data(symbol)
            
            # Safely extract data with defaults
            market_data = data.get('market_data') or {}
            dev_data = data.get('developer_data') or {}
            community_data = data.get('community_data') or {}
            
            # Extract exchange flow data (simulated as CoinGecko doesn't provide this directly)
            total_volume = 0
            if market_data and 'total_volume' in market_data:
                total_volume = market_data['total_volume'].get('usd', 0) if market_data['total_volume'] else 0
            
            exchange_netflow = np.random.uniform(-total_volume * 0.1, total_volume * 0.1) if total_volume > 0 else 0
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'market_cap': market_data.get('market_cap', {}).get('usd') if market_data.get('market_cap') else None,
                'total_volume': total_volume,
                'circulating_supply': market_data.get('circulating_supply'),
                'max_supply': market_data.get('max_supply'),
                'total_value_locked': market_data.get('total_value_locked', {}).get('usd') if market_data.get('total_value_locked') else None,
                # Developer activity metrics
                'github_stars': dev_data.get('stars') if dev_data else None,
                'github_forks': dev_data.get('forks') if dev_data else None,
                'github_commits_4_weeks': dev_data.get('commit_count_4_weeks') if dev_data else None,
                # Community metrics
                'twitter_followers': community_data.get('twitter_followers') if community_data else None,
                'reddit_subscribers': community_data.get('reddit_subscribers') if community_data else None,
                # Additional market data
                'price_change_percentage_24h': market_data.get('price_change_percentage_24h'),
                'price_change_percentage_7d': market_data.get('price_change_percentage_7d'),
                'price_change_percentage_30d': market_data.get('price_change_percentage_30d'),
                # Simulated on-chain metrics
                'active_addresses': np.random.randint(10000, 1000000),
                'transaction_count': np.random.randint(100000, 10000000),
                'exchange_inflow': abs(exchange_netflow) if exchange_netflow > 0 else 0,
                'exchange_outflow': abs(exchange_netflow) if exchange_netflow < 0 else 0,
                'exchange_netflow': exchange_netflow,
                'network_fees': np.random.uniform(0.1, 10),
                'average_transaction_fee': np.random.uniform(0.01, 1),
            }
            
            logger.debug(f"Successfully fetched on-chain data for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching on-chain data for {symbol}: {str(e)}")
            logger.debug(f"Full error details: {type(e).__name__}: {e}")
            return self._create_default_onchain_data(symbol)
    
    def _create_default_onchain_data(self, symbol: str) -> Dict:
        """Create default on-chain data when API fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_cap': None,
            'total_volume': 0,
            'circulating_supply': None,
            'max_supply': None,
            'total_value_locked': None,
            'active_addresses': np.random.randint(10000, 100000),
            'transaction_count': np.random.randint(10000, 100000),
            'exchange_inflow': 0,
            'exchange_outflow': 0,
            'exchange_netflow': 0,
            'network_fees': np.random.uniform(0.1, 5),
            'average_transaction_fee': np.random.uniform(0.01, 0.5),
        }
    
    async def fetch_nft_market_data(self) -> List[Dict]:
        """Fetch NFT market data from CoinGecko"""
        await self._rate_limit()
        
        try:
            # Get NFT list
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.cg.get_nfts_list(
                    order='market_cap_usd_desc',
                    per_page=100,
                    page=1
                )
            )
            
            if data is None:
                logger.warning("No NFT data returned from CoinGecko")
                return [self._create_default_nft_data()]
            
            if isinstance(data, dict) and 'error' in data:
                logger.error(f"CoinGecko NFT API error: {data['error']}")
                return [self._create_default_nft_data()]
            
            # Ensure data is a list
            if not isinstance(data, list):
                logger.warning(f"Unexpected NFT data format: {type(data)}")
                return [self._create_default_nft_data()]
            
            # Aggregate NFT market metrics
            total_market_cap = sum((item.get('market_cap_usd') or 0) for item in data)
            total_volume_24h = sum((item.get('volume_24h_usd') or 0) for item in data)
            
            # Calculate average metrics
            floor_prices = [item.get('floor_price_usd', 0) for item in data if item.get('floor_price_usd')]
            avg_floor_price_change = 0
            if data and len(data) > 0:
                changes = []
                for item in data[:20]:  # Top 20 collections
                    change = item.get('floor_price_24h_percentage_change_usd')
                    if change is not None:
                        changes.append(change)
                if changes:
                    avg_floor_price_change = np.mean(changes)
            
            # Get top collections
            top_collections = []
            for item in data[:10]:
                if item and item.get('name'):
                    top_collections.append({
                        'name': item.get('name'),
                        'symbol': item.get('symbol'),
                        'floor_price': item.get('floor_price_usd'),
                        'volume_24h': item.get('volume_24h_usd'),
                        'market_cap': item.get('market_cap_usd')
                    })
            
            # Count unique addresses (simulated)
            unique_buyers = np.random.randint(1000, 50000)
            unique_sellers = np.random.randint(800, 40000)
            
            return [{
                'timestamp': datetime.now(),
                'chain': 'ethereum',  # Most NFTs are on Ethereum
                'total_market_cap': total_market_cap,
                'total_volume_24h': total_volume_24h,
                'total_sales_24h': np.random.randint(10000, 100000),  # Simulated
                'floor_price_trend': avg_floor_price_change,
                'volume_momentum': total_volume_24h / max(total_market_cap, 1) if total_market_cap > 0 else 0,
                'chain_dominance': 0.7,  # Ethereum dominance estimate
                'top_collections': json.dumps(top_collections),
                'unique_buyers': unique_buyers,
                'unique_sellers': unique_sellers,
                'average_price': np.mean(floor_prices) if floor_prices else 0,
                'median_price': np.median(floor_prices) if floor_prices else 0,
                'blue_chip_index': self._calculate_blue_chip_index(data)
            }]
                
        except Exception as e:
            logger.error(f"Error fetching NFT market data: {str(e)}")
            logger.debug(f"Full error details: {type(e).__name__}: {e}")
            return [self._create_default_nft_data()]
    
    def _create_default_nft_data(self) -> Dict:
        """Create default NFT data when API fails"""
        return {
            'timestamp': datetime.now(),
            'chain': 'ethereum',
            'total_market_cap': 0,
            'total_volume_24h': 0,
            'total_sales_24h': 0,
            'floor_price_trend': 0,
            'volume_momentum': 0,
            'chain_dominance': 0.7,
            'top_collections': '[]',
            'unique_buyers': 0,
            'unique_sellers': 0,
            'average_price': 0,
            'median_price': 0,
            'blue_chip_index': 0
        }
    
    def _calculate_blue_chip_index(self, nft_data: List[Dict]) -> float:
        """Calculate blue chip NFT index"""
        # Define blue chip collections by name patterns
        blue_chip_patterns = [
            'bored', 'mutant', 'punk', 'azuki', 'clone', 'doodle',
            'bayc', 'mayc', 'cool cats', 'world of women'
        ]
        
        blue_chip_volume = 0
        total_volume = 0
        
        for item in nft_data:
            volume = item.get('volume_24h_usd', 0) or 0
            total_volume += volume
            
            name = (item.get('name', '') or '').lower()
            if any(pattern in name for pattern in blue_chip_patterns):
                blue_chip_volume += volume
        
        return blue_chip_volume / max(total_volume, 1) if total_volume > 0 else 0
    
    async def fetch_trending_coins(self) -> List[Dict]:
        """Fetch trending coins data"""
        await self._rate_limit()
        
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cg.get_search_trending
            )
            
            if data and 'coins' in data:
                return data['coins']
                
        except Exception as e:
            logger.error(f"Error fetching trending coins: {e}")
        
        return []
    
    async def fetch_global_data(self) -> Dict:
        """Fetch global crypto market data"""
        await self._rate_limit()
        
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cg.get_global
            )
            
            if data is None:
                logger.warning("No global data returned from CoinGecko")
                return {}
            
            if isinstance(data, dict) and 'error' in data:
                logger.error(f"CoinGecko global API error: {data['error']}")
                return {}
                
            return data.get('data', {}) if isinstance(data, dict) else {}
                
        except Exception as e:
            logger.error(f"Error fetching global data: {str(e)}")
            logger.debug(f"Full error details: {type(e).__name__}: {e}")
        
        return {}

# Sentiment Analysis Provider
class SentimentAnalysisProvider:
    """Provider for sentiment analysis from various sources"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.fear_greed_url = "https://api.alternative.me/fng/"
    
    async def fetch_sentiment_data(self, symbol: str) -> Dict:
        """Fetch sentiment data from various sources"""
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'source': 'aggregate'
        }
        
        # Fetch Fear and Greed Index
        fear_greed = await self._fetch_fear_greed_index()
        if fear_greed:
            sentiment_data['fear_greed_index'] = fear_greed
        
        # Simulate social sentiment (in production, integrate with Twitter/Reddit APIs)
        sentiment_data['sentiment_score'] = np.random.uniform(-1, 1)
        sentiment_data['sentiment_label'] = self._get_sentiment_label(sentiment_data['sentiment_score'])
        sentiment_data['social_volume'] = np.random.randint(1000, 50000)
        sentiment_data['social_dominance'] = np.random.uniform(0, 10)
        sentiment_data['news_volume'] = np.random.randint(10, 100)
        sentiment_data['reddit_mentions'] = np.random.randint(100, 5000)
        sentiment_data['twitter_mentions'] = np.random.randint(500, 10000)
        sentiment_data['google_trends'] = np.random.uniform(0, 100)
        sentiment_data['market_momentum'] = np.random.uniform(-1, 1)
        
        return sentiment_data
    
    async def _fetch_fear_greed_index(self) -> Optional[float]:
        """Fetch the crypto fear and greed index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            return float(data['data'][0]['value'])
        except Exception as e:
            logger.error(f"Error fetching fear and greed index: {e}")
        
        return None
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 0.5:
            return 'very_positive'
        elif score >= 0.2:
            return 'positive'
        elif score >= -0.2:
            return 'neutral'
        elif score >= -0.5:
            return 'negative'
        else:
            return 'very_negative'

# Technical Indicators Calculator
class TechnicalIndicatorsCalculator:
    """Enhanced technical indicators calculator"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Ensure we have enough data
        if len(df) < 50:
            logger.warning("Insufficient data for all technical indicators")
            return indicators
        
        try:
            # Trend Indicators
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
            indicators['sma_200'] = ta.trend.sma_indicator(df['close'], window=200).iloc[-1] if len(df) >= 200 else None
            indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26).iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            indicators['adx'] = adx.adx().iloc[-1]
            
            # Momentum Indicators
            indicators['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
            indicators['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci().iloc[-1]
            indicators['roc'] = ta.momentum.ROCIndicator(df['close']).roc().iloc[-1]
            indicators['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().iloc[-1]
            
            # Volatility Indicators
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
            indicators['atr'] = atr.average_true_range().iloc[-1]
            
            # Volume Indicators
            indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume().iloc[-1]
            
            # VWAP
            indicators['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            ).volume_weighted_average_price().iloc[-1]
            
            # Pivot Points
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            indicators['pivot'] = pivot
            indicators['resistance_1'] = 2 * pivot - df['low'].iloc[-1]
            indicators['support_1'] = 2 * pivot - df['high'].iloc[-1]
            indicators['resistance_2'] = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
            indicators['support_2'] = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
            
            # Volatility calculation
            returns = df['close'].pct_change().dropna()
            indicators['volatility'] = returns.std() * np.sqrt(24)  # Hourly volatility annualized
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators

# Enhanced Data Fetcher with all data types
class DataFetcher:
    """Main data fetching orchestrator with comprehensive data collection"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.db = SQLDatabase(config)
        self.binance_exchange = None
        self.coingecko_provider = CoinGeckoDataProvider(config)
        self.sentiment_provider = SentimentAnalysisProvider(config)
        self.indicators_calculator = TechnicalIndicatorsCalculator()
        
        self.ws_connections = {}
        self.data_queue = Queue(maxsize=config.max_queue_size)
        self.is_running = False
        
        # Batch insert buffers
        self.market_data_buffer = []
        self.trade_flow_buffer = []
        self.last_flush_time = time.time()
        
        # WebSocket reconnection tracking
        self.reconnect_attempts = {}
        self.ws_last_ping = {}
        
        # Last update times for different data types
        self.last_technical_update = {}
        self.last_onchain_update = {}
        self.last_nft_update = 0
        self.last_sentiment_update = {}
        
        # Initialize storage directories
        self._init_storage()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def _init_storage(self):
        """Initialize storage directories"""
        for dir_path in [self.config.data_dir, self.config.bronze_dir, 
                        self.config.silver_dir, self.config.gold_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    async def _init_binance(self):
        """Initialize Binance exchange connection"""
        self.binance_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        await self.binance_exchange.load_markets()
        logger.info("Binance connection established")
    
    def _start_binance_websocket(self, symbol: str):
        """Start Binance WebSocket for real-time trades and order book"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.data_queue.put(('binance_ws', symbol, data))
                self.ws_last_ping[symbol] = time.time()
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error for {symbol}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed for {symbol}")
            if self.is_running:
                self._handle_websocket_reconnect(symbol)
        
        def on_open(ws):
            logger.info(f"WebSocket opened for {symbol}")
            self.reconnect_attempts[symbol] = 0
            self.ws_last_ping[symbol] = time.time()
        
        # Format symbol for Binance WebSocket
        ws_symbol = symbol.replace('/', '').lower()
        
        # Subscribe to multiple streams
        streams = [
            f"{ws_symbol}@trade",      # Real-time trades
            f"{ws_symbol}@depth20",    # Order book depth
            f"{ws_symbol}@kline_1m"    # 1-minute klines
        ]
        
        ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run WebSocket in a separate thread
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(
                ping_interval=self.config.ws_ping_interval,
                ping_timeout=self.config.ws_ping_timeout
            ),
            daemon=True
        )
        ws_thread.start()
        
        self.ws_connections[symbol] = ws
    
    def _handle_websocket_reconnect(self, symbol: str):
        """Handle WebSocket reconnection with exponential backoff"""
        reconnect_delay = min(
            self.config.ws_max_reconnect_delay, 
            5 * (2 ** self.reconnect_attempts.get(symbol, 0))
        )
        self.reconnect_attempts[symbol] = self.reconnect_attempts.get(symbol, 0) + 1
        logger.info(f"Reconnecting {symbol} after {reconnect_delay} seconds...")
        
        threading.Timer(reconnect_delay, lambda: self._start_binance_websocket(symbol)).start()
    
    async def update_technical_indicators(self, symbol: str):
        """Update technical indicators for a symbol"""
        current_time = time.time()
        last_update = self.last_technical_update.get(symbol, 0)
        
        if current_time - last_update < self.config.technical_indicators_interval:
            return
        
        try:
            # Get recent market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            df = self.db.get_historical_data(symbol, '1h', start_date, end_date)
            
            if len(df) >= 50:
                # Calculate all technical indicators
                indicators = self.indicators_calculator.calculate_all_indicators(df)
                
                if indicators:
                    indicators['symbol'] = symbol
                    indicators['timestamp'] = datetime.now()
                    indicators['timeframe'] = '1h'
                    
                    # Store in database
                    self.db.insert_technical_indicators(indicators)
                    
                    self.last_technical_update[symbol] = current_time
                    logger.info(f"Updated technical indicators for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating technical indicators for {symbol}: {e}")
    
    async def update_onchain_data(self, symbol: str):
        """Update on-chain data for a symbol"""
        current_time = time.time()
        last_update = self.last_onchain_update.get(symbol, 0)
        
        if current_time - last_update < self.config.onchain_interval:
            return
        
        try:
            # Check if CoinGecko is enabled
            if self.config.use_coingecko:
                # Fetch on-chain data from CoinGecko
                onchain_data = await self.coingecko_provider.fetch_onchain_data(symbol)
            else:
                # Use simulated data
                onchain_data = self._generate_simulated_onchain_data(symbol)
            
            if onchain_data:
                # Store in database
                self.db.insert_onchain_data(onchain_data)
                
                self.last_onchain_update[symbol] = current_time
                logger.info(f"Updated on-chain data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating on-chain data for {symbol}: {e}")
    
    def _generate_simulated_onchain_data(self, symbol: str) -> Dict:
        """Generate simulated on-chain data for testing/fallback"""
        # Get latest price from market data
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT close, volume FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            
        price = row['close'] if row else 60000  # Default price
        volume = row['volume'] if row else 1000000
        
        # Generate realistic looking data based on symbol
        base_values = {
            'BTC/USDT': {'cap': 1200000000000, 'supply': 19500000, 'addresses': 1000000},
            'SOL/USDT': {'cap': 80000000000, 'supply': 400000000, 'addresses': 500000},
            'ETH/USDT': {'cap': 400000000000, 'supply': 120000000, 'addresses': 800000},
        }
        
        defaults = base_values.get(symbol, {'cap': 10000000000, 'supply': 100000000, 'addresses': 100000})
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_cap': defaults['cap'] * (1 + np.random.uniform(-0.1, 0.1)),
            'total_volume': volume * 24 * np.random.uniform(0.8, 1.2),
            'circulating_supply': defaults['supply'],
            'max_supply': defaults['supply'] * 1.1,
            'active_addresses': int(defaults['addresses'] * np.random.uniform(0.8, 1.2)),
            'transaction_count': int(defaults['addresses'] * 10 * np.random.uniform(0.8, 1.2)),
            'exchange_inflow': volume * np.random.uniform(0.1, 0.3),
            'exchange_outflow': volume * np.random.uniform(0.1, 0.3),
            'exchange_netflow': volume * np.random.uniform(-0.2, 0.2),
            'network_fees': np.random.uniform(0.5, 5),
            'average_transaction_fee': np.random.uniform(0.01, 0.5),
        }
    
    async def update_nft_market_data(self):
        """Update NFT market data"""
        current_time = time.time()
        
        if current_time - self.last_nft_update < self.config.nft_interval:
            return
        
        try:
            if self.config.use_coingecko:
                # Fetch NFT market data from CoinGecko
                nft_data_list = await self.coingecko_provider.fetch_nft_market_data()
            else:
                # Use simulated data
                nft_data_list = [self._generate_simulated_nft_data()]
            
            for nft_data in nft_data_list:
                if nft_data:
                    # Store in database
                    self.db.insert_nft_market_data(nft_data)
            
            self.last_nft_update = current_time
            logger.info("Updated NFT market data")
            
        except Exception as e:
            logger.error(f"Error updating NFT market data: {e}")
    
    def _generate_simulated_nft_data(self) -> Dict:
        """Generate simulated NFT market data for testing/fallback"""
        return {
            'timestamp': datetime.now(),
            'chain': 'ethereum',
            'total_market_cap': np.random.uniform(10000000000, 15000000000),
            'total_volume_24h': np.random.uniform(100000000, 500000000),
            'total_sales_24h': np.random.randint(50000, 150000),
            'floor_price_trend': np.random.uniform(-10, 10),
            'volume_momentum': np.random.uniform(0.01, 0.05),
            'chain_dominance': 0.7,
            'top_collections': json.dumps([
                {'name': 'Bored Ape Yacht Club', 'floor_price': 30, 'volume_24h': 1000000},
                {'name': 'CryptoPunks', 'floor_price': 50, 'volume_24h': 2000000},
                {'name': 'Azuki', 'floor_price': 15, 'volume_24h': 500000},
            ]),
            'unique_buyers': np.random.randint(10000, 50000),
            'unique_sellers': np.random.randint(8000, 40000),
            'average_price': np.random.uniform(0.5, 5),
            'median_price': np.random.uniform(0.3, 3),
            'blue_chip_index': np.random.uniform(0.3, 0.5)
        }
    
    async def update_sentiment_data(self, symbol: str):
        """Update sentiment data for a symbol"""
        current_time = time.time()
        last_update = self.last_sentiment_update.get(symbol, 0)
        
        if current_time - last_update < self.config.sentiment_interval:
            return
        
        try:
            # Fetch sentiment data
            sentiment_data = await self.sentiment_provider.fetch_sentiment_data(symbol)
            
            if sentiment_data:
                # Store in database
                self.db.insert_sentiment_data(sentiment_data)
                
                self.last_sentiment_update[symbol] = current_time
                logger.info(f"Updated sentiment data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating sentiment data for {symbol}: {e}")
    
    async def _process_websocket_data(self, symbol: str, data: Dict):
        """Process real-time WebSocket data"""
        try:
            stream = data.get('stream', '')
            ws_data = data.get('data', {})
            
            if 'trade' in stream:
                # Process trade data
                trade_data = {
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(ws_data['T'] / 1000),
                    'price': float(ws_data['p']),
                    'volume': float(ws_data['q']),
                    'side': 'buy' if ws_data['m'] else 'sell',
                    'is_buyer_maker': ws_data['m'],
                    'trade_id': str(ws_data['t'])
                }
                
                self.trade_flow_buffer.append(trade_data)
            
            elif 'depth' in stream:
                # Process order book update
                bids = ws_data.get('bids', [])
                asks = ws_data.get('asks', [])
                
                if bids and asks:
                    # Calculate metrics
                    bid_volume = sum(float(bid[1]) for bid in bids[:10])
                    ask_volume = sum(float(ask[1]) for ask in asks[:10])
                    spread = float(asks[0][0]) - float(bids[0][0])
                    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
                    
                    # Store immediately
                    self.db.insert_orderbook_data({
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'best_bid': float(bids[0][0]),
                        'best_ask': float(asks[0][0]),
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'spread': spread,
                        'imbalance': imbalance,
                        'depth_20_bid': sum(float(bid[1]) * float(bid[0]) for bid in bids[:20]),
                        'depth_20_ask': sum(float(ask[1]) * float(ask[0]) for ask in asks[:20])
                    })
            
            elif 'kline' in stream:
                # Process kline data
                kline = ws_data['k']
                
                if kline['x']:  # Kline is closed
                    kline_data = {
                        'symbol': symbol,
                        'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                        'timeframe': kline['i'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'trades_count': int(kline['n'])
                    }
                    
                    self.market_data_buffer.append(kline_data)
                    
        except Exception as e:
            logger.error(f"WebSocket data processing error: {e}")
    
    async def create_unified_features(self, symbol: str) -> Dict:
        """Create comprehensive unified feature set for models"""
        try:
            # Get latest data from all sources
            market_query = """
                SELECT * FROM market_data 
                WHERE symbol = ? AND timeframe = '1h'
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            technical_query = """
                SELECT * FROM technical_indicators 
                WHERE symbol = ? AND timeframe = '1h'
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            orderbook_query = """
                SELECT * FROM orderbook_data 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            sentiment_query = """
                SELECT * FROM sentiment_data 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            onchain_query = """
                SELECT * FROM onchain_data 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            nft_query = """
                SELECT * FROM nft_market_data 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            with self.db.get_connection() as conn:
                # Fetch all data types
                market_data = pd.read_sql_query(market_query, conn, params=(symbol,))
                technical_data = pd.read_sql_query(technical_query, conn, params=(symbol,))
                orderbook_data = pd.read_sql_query(orderbook_query, conn, params=(symbol,))
                sentiment_data = pd.read_sql_query(sentiment_query, conn, params=(symbol,))
                onchain_data = pd.read_sql_query(onchain_query, conn, params=(symbol,))
                nft_data = pd.read_sql_query(nft_query, conn)
            
            # Create unified features
            features = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            # Market features
            if not market_data.empty:
                features['price'] = market_data.iloc[0]['close']
                features['volume_1h'] = market_data.iloc[0]['volume']
            
            # Technical indicators
            if not technical_data.empty:
                features['rsi'] = technical_data.iloc[0]['rsi']
                features['macd'] = technical_data.iloc[0]['macd']
                features['volatility_1h'] = technical_data.iloc[0]['volatility']
                features['technical_rating'] = self._calculate_technical_rating(technical_data.iloc[0])
            
            # Orderbook features
            if not orderbook_data.empty:
                features['orderbook_imbalance'] = orderbook_data.iloc[0]['imbalance']
                features['spread'] = orderbook_data.iloc[0]['spread']
            
            # Trade flow features
            recent_trades = self.db.get_recent_trades(symbol, limit=1000)
            if not recent_trades.empty:
                buy_volume = recent_trades[recent_trades['side'] == 'buy']['volume'].sum()
                sell_volume = recent_trades[recent_trades['side'] == 'sell']['volume'].sum()
                features['buy_sell_ratio'] = buy_volume / (sell_volume + 1) if sell_volume > 0 else 1
            
            # Sentiment features
            if not sentiment_data.empty:
                features['sentiment_score'] = sentiment_data.iloc[0]['sentiment_score']
                features['fear_greed_index'] = sentiment_data.iloc[0]['fear_greed_index']
            
            # On-chain features
            if not onchain_data.empty:
                features['exchange_netflow'] = onchain_data.iloc[0]['exchange_netflow']
                features['onchain_rating'] = self._calculate_onchain_rating(onchain_data.iloc[0])
            
            # NFT features
            if not nft_data.empty:
                features['nft_sentiment'] = nft_data.iloc[0]['floor_price_trend']
            
            # Create comprehensive feature vector for advanced models
            feature_vector = self._create_feature_vector(features)
            features['feature_vector'] = feature_vector
            
            # Store unified features
            self.db.insert_unified_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating unified features: {e}")
            return {}
    
    def _calculate_technical_rating(self, indicators: pd.Series) -> float:
        """Calculate overall technical rating from indicators"""
        rating = 0.0
        count = 0
        
        # RSI signals
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            if rsi < 30:
                rating += 1  # Oversold - bullish
            elif rsi > 70:
                rating -= 1  # Overbought - bearish
            count += 1
        
        # MACD signals
        if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
            if indicators['macd'] > indicators['macd_signal']:
                rating += 1  # Bullish crossover
            else:
                rating -= 1  # Bearish crossover
            count += 1
        
        # Moving average signals
        if indicators.get('close') is not None:
            close = indicators['close']
            if indicators.get('sma_20') is not None and close > indicators['sma_20']:
                rating += 0.5
            if indicators.get('sma_50') is not None and close > indicators['sma_50']:
                rating += 0.5
            count += 1
        
        return rating / max(count, 1)
    
    def _calculate_onchain_rating(self, onchain: pd.Series) -> float:
        """Calculate overall on-chain rating"""
        rating = 0.0
        
        # Exchange netflow signals
        if onchain.get('exchange_netflow') is not None:
            if onchain['exchange_netflow'] < 0:
                rating += 1  # Outflow - bullish
            else:
                rating -= 1  # Inflow - bearish
        
        # Active addresses trend
        if onchain.get('active_addresses') is not None:
            # Compare with historical average (simplified)
            if onchain['active_addresses'] > 100000:  # High activity
                rating += 0.5
        
        return np.tanh(rating)  # Normalize to [-1, 1]
    
    def _create_feature_vector(self, features: Dict) -> Dict:
        """Create comprehensive feature vector for models"""
        # Extract numerical features
        numerical_features = []
        feature_names = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                numerical_features.append(value)
                feature_names.append(key)
        
        # Normalize features
        if numerical_features:
            features_array = np.array(numerical_features)
            normalized = (features_array - np.mean(features_array)) / (np.std(features_array) + 1e-8)
            
            return {
                'features': normalized.tolist(),
                'feature_names': feature_names,
                'dimension': len(feature_names)
            }
        
        return {}
    
    async def fetch_historical_ohlcv(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            all_ohlcv = []
            while True:
                ohlcv = await self.binance_exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                
                await asyncio.sleep(0.1)
            
            df = pd.DataFrame(
                all_ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Store in database
            market_data = []
            for idx, row in df.iterrows():
                market_data.append({
                    'symbol': symbol,
                    'timestamp': idx,
                    'timeframe': timeframe,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            if market_data:
                self.db.insert_market_data(market_data)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _flush_buffers(self):
        """Flush data buffers to database"""
        try:
            if self.market_data_buffer:
                self.db.insert_market_data(self.market_data_buffer)
                self.market_data_buffer = []
            
            if self.trade_flow_buffer:
                self.db.insert_trade_flow(self.trade_flow_buffer)
                self.trade_flow_buffer = []
            
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"Buffer flush error: {e}")
    
    async def process_and_store_data(self):
        """Process data from queue and store in database"""
        while self.is_running:
            try:
                # Check if we need to flush buffers
                if time.time() - self.last_flush_time > 5:
                    await self._flush_buffers()
                
                if not self.data_queue.empty():
                    data_type, symbol, data = self.data_queue.get()
                    
                    if data_type == 'binance_ws':
                        await self._process_websocket_data(symbol, data)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Data processing error: {e}")
    
    def start_realtime_feeds(self):
        """Start all real-time data feeds"""
        for symbol in self.config.binance_symbols:
            self._start_binance_websocket(symbol)
        
        logger.info("Real-time feeds started")
    
    async def run_data_pipeline(self):
        """Main data pipeline orchestrator with comprehensive data collection"""
        self.is_running = True
        
        try:
            # Initialize Binance
            await self._init_binance()
            
            # Test CoinGecko connection
            logger.info("Testing CoinGecko API connection...")
            connection_ok = await self.coingecko_provider.test_connection()
            if not connection_ok:
                logger.warning("CoinGecko API connection test failed. Continuing with limited functionality.")
            else:
                # Try to get global data as additional test
                global_data = await self.coingecko_provider.fetch_global_data()
                if global_data:
                    total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                    logger.info(f"CoinGecko API connected successfully. Total Market Cap: ${total_market_cap:,.0f}")
                else:
                    logger.warning("CoinGecko API connected but unable to fetch market data. Check API key permissions.")
            
            # Load historical data first
            logger.info("Loading historical data...")
            for symbol in self.config.binance_symbols:
                for timeframe in self.config.binance_timeframes.values():
                    await self.fetch_historical_ohlcv(symbol, timeframe, self.config.historical_days)
            
            # Start real-time feeds
            self.start_realtime_feeds()
            
            # Start data processing task
            asyncio.create_task(self.process_and_store_data())
            
            # Main loop for periodic data fetching
            logger.info("Starting main data collection loop...")
            while self.is_running:
                try:
                    # Update all data types for each symbol
                    for symbol in self.config.binance_symbols:
                        # Update technical indicators
                        await self.update_technical_indicators(symbol)
                        
                        # Update on-chain data (will use defaults if API fails)
                        await self.update_onchain_data(symbol)
                        
                        # Update sentiment data
                        await self.update_sentiment_data(symbol)
                        
                        # Create unified features
                        await self.create_unified_features(symbol)
                    
                    # Update NFT market data (will use defaults if API fails)
                    await self.update_nft_market_data()
                    
                    # Sleep before next iteration
                    await asyncio.sleep(30)
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down data pipeline...")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    await asyncio.sleep(5)
                    
        finally:
            await self._flush_buffers()
    
    async def get_latest_features(self, symbol: str) -> Dict[str, Any]:
        """Get latest features for model inference"""
        return self.db.get_latest_features(symbol)
    
    def shutdown(self):
        """Gracefully shutdown all connections"""
        self.is_running = False
        
        for ws in self.ws_connections.values():
            try:
                ws.close()
            except:
                pass
        
        if self.binance_exchange:
            asyncio.create_task(self.binance_exchange.close())
        
        self.executor.shutdown(wait=True)
        
        logger.info("Data fetcher shutdown complete")


# Main execution
async def main():
    """Initialize and run the enhanced data collection system"""
    # Create configuration
    config = DataConfig()
    
    # Option to disable CoinGecko if API issues persist
    # config.use_coingecko = False  # Uncomment to use simulated data instead
    
    # Initialize data fetcher
    fetcher = DataFetcher(config)
    
    try:
        # Run the data pipeline
        await fetcher.run_data_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        fetcher.shutdown()


if __name__ == "__main__":
    # Create event loop and run
    asyncio.run(main())
