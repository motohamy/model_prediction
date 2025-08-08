"""
Complete Trading System with Per-Pair Webhook Support
Fully compatible with main.py - includes all required parameters
"""

import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import threading
import requests
from flask import Flask, request, jsonify
import ccxt
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingSystem")


# Replace the TradingConfig class in your trading_system.py with this flexible version:

@dataclass
class TradingConfig:
    """Configuration for the trading system - accepts any parameters"""
    # Core trading mode
    mode: str = 'paper'  # 'paper' or 'live'
    
    # Capital management
    initial_balance: float = 1000.0
    position_size_pct: float = 0.95
    max_positions: int = 2
    max_concurrent_positions: int = 2
    
    # Risk management
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_pct: float = 0.20
    risk_per_trade: float = 0.02
    max_daily_loss: float = 0.10  # Added this parameter
    max_daily_trades: int = 10  # Might be needed
    
    # Timing
    check_interval: int = 60
    
    # Database
    db_path: str = 'crypto_trading.db'
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'SOL/USDT'])
    
    # Webhook configuration per pair
    btc_webhook_url: str = 'https://api.primeautomation.ai/webhook/ChartPrime/bafbdc00-a670-48ad-9624-c7c059f2c385'
    btc_webhook_enabled: bool = True
    sol_webhook_url: str = 'https://api.primeautomation.ai/webhook/ChartPrime/ca60dbfd-46b9-4a44-bdec-43c0a024a379'
    sol_webhook_enabled: bool = True
    
    # Webhook server settings
    webhook_port: int = 5000
    webhook_host: str = '0.0.0.0'
    enable_webhooks: bool = True
    
    # Legacy webhook support
    webhook_urls: List[str] = field(default_factory=lambda: [])
    
    # Exchange API
    api_key: str = ''
    api_secret: str = ''
    exchange: str = 'binance'
    
    # Trading features
    enable_stop_loss_check: bool = True
    enable_take_profit_check: bool = True
    enable_trailing_stop: bool = False
    trailing_stop_pct: float = 0.01
    
    # Model integration
    use_ensemble_predictions: bool = True
    min_confidence_threshold: float = 0.6
    
    # Performance tracking
    save_trade_history: bool = True
    performance_report_interval: int = 3600
    
    # Advanced settings
    slippage_tolerance: float = 0.001
    fee_percentage: float = 0.001
    
    # Catch-all for any other parameters main.py might send
    def __post_init__(self):
        """Handle any additional parameters passed from main.py"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary including all attributes"""
        # Get all attributes including any dynamically added ones
        result = {}
        for field_name in dir(self):
            if not field_name.startswith('_') and not callable(getattr(self, field_name)):
                result[field_name] = getattr(self, field_name)
        return result


# Alternative solution - A more flexible config class that accepts ANY parameters:
class FlexibleTradingConfig:
    """Flexible configuration that accepts any parameters from main.py"""
    
    def __init__(self, **kwargs):
        self.symbol_allocation = {
    'BTC/USDT': 0.5,  # 50% of capital
    'SOL/USDT': 0.5   # 50% of capital
    }
        # Set default values for core parameters
        self.mode = kwargs.get('mode', 'paper')
        self.initial_balance = kwargs.get('initial_balance', 1000.0)
        self.position_size_pct = kwargs.get('position_size_pct', 0.45)
        self.max_positions = kwargs.get('max_positions', 2)
        self.max_concurrent_positions = kwargs.get('max_concurrent_positions', 2)
        
        # Risk management with defaults
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.03)
        self.take_profit_pct = kwargs.get('take_profit_pct', 0.04)
        self.max_drawdown_pct = kwargs.get('max_drawdown_pct', 0.20)
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.02)
        self.max_daily_loss = kwargs.get('max_daily_loss', 0.10)
        self.max_daily_trades = kwargs.get('max_daily_trades', 10)
        
        # Timing
        self.check_interval = kwargs.get('check_interval', 60)
        
        # Database
        self.db_path = kwargs.get('db_path', 'crypto_trading.db')
        
        # Trading symbols
        self.symbols = kwargs.get('symbols', ['BTC/USDT', 'SOL/USDT'])
        
        # Webhook configuration
        self.btc_webhook_url = kwargs.get('btc_webhook_url', 'http://localhost:5001/webhook/btc')
        self.btc_webhook_enabled = kwargs.get('btc_webhook_enabled', True)
        self.sol_webhook_url = kwargs.get('sol_webhook_url', 'http://localhost:5001/webhook/sol')
        self.sol_webhook_enabled = kwargs.get('sol_webhook_enabled', True)
        self.webhook_port = kwargs.get('webhook_port', 5000)
        self.webhook_host = kwargs.get('webhook_host', '0.0.0.0')
        self.enable_webhooks = kwargs.get('enable_webhooks', True)
        self.webhook_urls = kwargs.get('webhook_urls', [])
        
        # Exchange
        self.api_key = kwargs.get('api_key', '')
        self.api_secret = kwargs.get('api_secret', '')
        self.exchange = kwargs.get('exchange', 'binance')
        
        # Trading features
        self.enable_stop_loss_check = kwargs.get('enable_stop_loss_check', True)
        self.enable_take_profit_check = kwargs.get('enable_take_profit_check', True)
        self.enable_trailing_stop = kwargs.get('enable_trailing_stop', False)
        self.trailing_stop_pct = kwargs.get('trailing_stop_pct', 0.01)
        
        # Model integration
        self.use_ensemble_predictions = kwargs.get('use_ensemble_predictions', True)
        self.min_confidence_threshold = kwargs.get('min_confidence_threshold', 0.7)
        
        # Performance
        self.save_trade_history = kwargs.get('save_trade_history', True)
        self.performance_report_interval = kwargs.get('performance_report_interval', 3600)
        
        # Advanced
        self.slippage_tolerance = kwargs.get('slippage_tolerance', 0.001)
        self.fee_percentage = kwargs.get('fee_percentage', 0.001)
        
        # Store any additional parameters that main.py might send
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return getattr(self, key, default)



class TradingDatabase:
    """Database helper for trading system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_active_positions(self):
        """Get all active positions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM active_positions 
                WHERE status = 'active'
            """)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            positions = []
            for row in rows:
                position = dict(zip(columns, row))
                positions.append(position)
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []


class TradingSystem:
    """Paper trading system with live market data and per-pair webhooks"""
    
    def __init__(self, config):
        """Initialize the trading system"""
        # Handle different config types
        if hasattr(config, '__dict__'):
            # It's an object (TradingConfig or similar)
            self.config = config.__dict__.copy()
        elif isinstance(config, dict):
            # It's already a dictionary
            self.config = config.copy()
        else:
            # Try to convert to dict
            try:
                self.config = dict(config)
            except:
                self.config = {'mode': 'paper', 'initial_balance': 1000}
        
        # Basic settings with safe access
        self.mode = self.config.get('mode', 'paper')
        self.initial_balance = self.config.get('initial_balance', 1000)
        self.balance = self.initial_balance
        self.db_path = self.config.get('db_path', 'crypto_trading.db')
        
        # Per-pair webhook configuration
        self.webhook_config = {
            'BTC/USDT': {
                'url': self.config.get('btc_webhook_url', 'http://localhost:5000/webhook/btc'),
                'enabled': self.config.get('btc_webhook_enabled', True)
            },
            'SOL/USDT': {
                'url': self.config.get('sol_webhook_url', 'http://localhost:5000/webhook/sol'),
                'enabled': self.config.get('sol_webhook_enabled', True)
            }
        }
        
        # Handle external_webhooks parameter from main.py
        if 'external_webhooks' in self.config and self.config['external_webhooks']:
            # Update webhook URLs from external_webhooks if provided
            for webhook_url in self.config['external_webhooks']:
                if 'btc' in webhook_url.lower():
                    self.webhook_config['BTC/USDT']['url'] = webhook_url
                elif 'sol' in webhook_url.lower():
                    self.webhook_config['SOL/USDT']['url'] = webhook_url
        
      
        
        # Trading parameters
        self.position_size_pct = self.config.get('position_size_pct', 0.95)
        self.max_positions = self.config.get('max_positions', 2)
        self.max_concurrent_positions = self.config.get('max_concurrent_positions', self.max_positions)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        
        # Risk management
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', 0.20)
        
        # Trading features
        self.enable_stop_loss_check = self.config.get('enable_stop_loss_check', True)
        self.enable_take_profit_check = self.config.get('enable_take_profit_check', True)
        self.enable_trailing_stop = self.config.get('enable_trailing_stop', False)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 0.01)
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Position tracking
        self.active_positions = {}
        self.completed_trades = {}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0
        
        # Flask app for receiving webhooks
        self.app = Flask(__name__)
        self.setup_webhook_endpoints()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load positions from database
        self.load_positions_from_db()
        
        logger.info("Trading system initialized successfully")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        exchange_name = self.config.get('exchange', 'binance')
        
        exchange_config = {
            'apiKey': self.config.get('api_key'),
            'secret': self.config.get('api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        if exchange_name.lower() == 'binance':
            exchange = ccxt.binance(exchange_config)
        else:
            # Default to binance
            exchange = ccxt.binance(exchange_config)
        
        if self.mode == 'paper':
            exchange.set_sandbox_mode(True)
        
        return exchange
    
    def setup_webhook_endpoints(self):
        """Setup Flask webhook endpoints"""
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            try:
                data = request.json
                logger.info(f"Received webhook: {data}")
                # Process webhook data in a separate thread
                threading.Thread(target=self.process_webhook, args=(data,)).start()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def status():
            try:
                portfolio = self.get_portfolio_status()
                return jsonify(portfolio), 200
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200
    
    def send_webhook(self, event_type: str, position: Dict[str, Any]) -> bool:
        """Send webhook notification in the correct flat format for specific pair"""
        try:
            symbol = position['symbol']
            
            # Check if webhooks are enabled
            if not self.config.get('enable_webhooks', True):
                return False
            
            # Check if webhook is configured for this symbol
            if symbol not in self.webhook_config:
                logger.warning(f"No webhook configured for {symbol}")
                return False
            
            webhook_info = self.webhook_config[symbol]
            if not webhook_info['enabled']:
                return False
            
            webhook_url = webhook_info['url']
            
            # Format based on event type
            if event_type == 'POSITION_OPENED':
                # Format for opening position
                ticker = symbol.replace('/', '')  # BTC/USDT -> BTCUSDT
                
                webhook_data = {
                    'ticker': ticker,
                    'action': position['side'],  # 'buy' or 'sell'
                    'price': str(position['entry_price']),
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': '24.0h',
                    'confidence': f"{position.get('confidence', 0.75):.2f}",
                    'regime': self._get_regime_string(position.get('market_regime', 'ranging'))
                }
                
            elif event_type in ['POSITION_CLOSED', 'STOP_LOSS_HIT', 'TAKE_PROFIT_HIT']:
                # Format for closing position
                ticker = symbol.replace('/', '')
                
                # Determine action type
                if position['side'] == 'buy':
                    action = 'exit_buy'
                else:
                    action = 'exit_sell'
                
                # Determine close reason and percentage
                if event_type == 'TAKE_PROFIT_HIT':
                    reason = 'tp_hit'
                    per_str = '100%'
                elif event_type == 'STOP_LOSS_HIT':
                    reason = 'sl_hit'
                    per_str = '0%'
                else:
                    # Manual close or signal change
                    reason = 'manual'
                    pnl_pct = position.get('pnl_percentage', 0)
                    
                    # Check if it was actually SL or TP
                    if position.get('exit_price'):
                        if position['side'] == 'buy':
                            if position.get('take_profit') and position['exit_price'] >= position['take_profit'] * 0.99:
                                reason = 'tp_hit'
                                per_str = '100%'
                            elif position.get('stop_loss') and position['exit_price'] <= position['stop_loss'] * 1.01:
                                reason = 'sl_hit'
                                per_str = '0%'
                            else:
                                per_str = f"{abs(pnl_pct):.0f}%"
                        else:  # sell
                            if position.get('take_profit') and position['exit_price'] <= position['take_profit'] * 1.01:
                                reason = 'tp_hit'
                                per_str = '100%'
                            elif position.get('stop_loss') and position['exit_price'] >= position['stop_loss'] * 0.99:
                                reason = 'sl_hit'
                                per_str = '0%'
                            else:
                                per_str = f"{abs(pnl_pct):.0f}%"
                
                webhook_data = {
                    'ticker': ticker,
                    'action': action,
                    'price': str(position.get('exit_price', position.get('current_price', position['entry_price']))),
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'size': '3.0',
                    'per': per_str,
                    'sl': f"{position.get('stop_loss', 0):.2f}",
                    'tp': f"{position.get('take_profit', 0):.2f}",
                    'reason': reason
                }
            
            else:
                logger.warning(f"Unknown webhook event type: {event_type}")
                return False
            
            # Log the webhook data
            logger.info(f"Sending {symbol} webhook to {webhook_url}: {webhook_data}")
            
            # Send webhook
            try:
                response = requests.post(
                    webhook_url,
                    json=webhook_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"Webhook sent successfully to {webhook_url}")
                    return True
                else:
                    logger.warning(f"Webhook failed with status {response.status_code}: {response.text}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to send webhook to {webhook_url}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error formatting webhook: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_regime_string(self, regime: str) -> str:
        """Convert market regime to webhook format"""
        regime_map = {
            'trending_up': 'Trending',
            'trending_down': 'Trending',
            'sideways': 'Ranging',
            'volatile': 'Volatile',
            'breakout': 'Breakout'
        }
        return regime_map.get(regime, 'Ranging')
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on available balance and risk management"""
        with self.lock:
            # Get symbol-specific allocation (50% each by default)
            allocation = self.symbol_allocation.get(symbol, 0.5)
            
            # Calculate position value based on allocation
            position_value = self.balance * allocation
            
            # Check if we already have too many positions
            active_count = len([p for p in self.active_positions.values() if p['status'] == 'active'])
            if active_count >= self.max_concurrent_positions:
                logger.warning(f"Max positions ({self.max_concurrent_positions}) reached")
                return 0
            
            # Check if we already have this symbol
            if self.get_position(symbol):
                logger.info(f"Already have position for {symbol}")
                return 0
            
            # Apply risk limits
            max_risk = self.balance * self.risk_per_trade / self.stop_loss_pct
            position_value = min(position_value, max_risk)
            
            # Calculate size
            position_size = position_value / price
            
            # Check minimum order size
            try:
                market = self.exchange.market(symbol)
                min_amount = market['limits']['amount']['min']
                if position_size < min_amount:
                    logger.warning(f"Size {position_size} below minimum {min_amount}")
                    return 0
            except:
                pass
            
            logger.info(f"{symbol} position: size={position_size:.6f}, value=${position_value:.2f}")
            return position_size
    
    def open_position(self, decision: Dict[str, Any]) -> bool:
        """Open a new position based on decision"""
        try:
            symbol = decision['symbol']
            side = decision['action']
            
            if side not in ['buy', 'sell']:
                return False
            
            # Check confidence threshold
            confidence = decision.get('confidence', 0)
            if confidence < self.min_confidence_threshold:
                logger.info(f"Confidence {confidence:.2f} below threshold {self.min_confidence_threshold}")
                return False
            
            with self.lock:
                # Check if we already have a position
                if self.get_position(symbol):
                    logger.info(f"Already have position for {symbol}")
                    return False
                
                # Check position limit
                active_positions_count = len([p for p in self.active_positions.values() if p['status'] == 'active'])
                if active_positions_count >= self.max_concurrent_positions:
                    logger.warning(f"Maximum concurrent positions ({self.max_concurrent_positions}) reached")
                    return False
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, current_price)
            if position_size <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return False
            
            # Calculate stop loss and take profit
            if side == 'buy':
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
                trailing_stop = current_price * (1 - self.trailing_stop_pct) if self.enable_trailing_stop else None
            else:
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)
                trailing_stop = current_price * (1 + self.trailing_stop_pct) if self.enable_trailing_stop else None
            
            # Override with decision values if provided
            if decision.get('stop_loss'):
                stop_loss = decision['stop_loss']
            if decision.get('take_profit'):
                take_profit = decision['take_profit']
            
            # Create position
            position_id = str(uuid.uuid4())
            position = {
                'id': position_id,
                'symbol': symbol,
                'side': side,
                'size': position_size,
                'entry_price': current_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'highest_price': current_price if side == 'buy' else None,
                'lowest_price': current_price if side == 'sell' else None,
                'status': 'active',
                'entry_time': datetime.now(),
                'exit_time': None,
                'exit_price': None,
                'pnl': 0,
                'pnl_percentage': 0,
                'fees_paid': 0,
                'model_source': decision.get('model_name', 'unknown'),
                'confidence': confidence,
                'market_regime': decision.get('market_regime', 'unknown'),
                'notes': ''
            }
            
            with self.lock:
                # Deduct from balance (paper trading)
                position_value = position_size * current_price
                self.balance -= position_value
                
                # Add to active positions
                self.active_positions[position_id] = position
            
            # Save to database
            self.save_position_to_db(position)
            
            # Send webhook
            self.send_webhook('POSITION_OPENED', position)
            
            logger.info(f"Opened {side} position for {symbol}: "
                       f"{position_size:.6f} @ ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def close_position(self, position_id: str, reason: str = "signal_change") -> bool:
        """Close a specific position"""
        try:
            with self.lock:
                if position_id not in self.active_positions:
                    logger.warning(f"Position {position_id} not found")
                    return False
                
                position = self.active_positions[position_id]
            
            # Get current price
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                logger.error(f"Could not get current price for {position['symbol']}")
                return False
            
            # Update position with exit details
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            position['status'] = 'closed'
            position['exit_reason'] = reason
            position['current_price'] = current_price
            
            # Calculate P&L
            self.calculate_pnl(position)
            
            with self.lock:
                # Update balance (paper trading)
                exit_value = position['size'] * current_price
                self.balance += exit_value
                
                # Update statistics
                self.update_statistics(position)
                
                # Move to completed trades
                self.completed_trades[position_id] = position
                del self.active_positions[position_id]
            
            # Send webhook
            self.send_webhook('POSITION_CLOSED', position)
            
            # Save to database
            self.save_completed_trade(position)
            self.update_position_in_db(position)
            
            logger.info(f"Closed {position['side']} position for {position['symbol']} "
                       f"at ${current_price:.2f} with P&L: ${position['pnl']:.2f} "
                       f"({position['pnl_percentage']:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def calculate_pnl(self, position: Dict[str, Any]):
        """Calculate P&L for a position"""
        if position['side'] == 'buy':
            pnl = (position['exit_price'] - position['entry_price']) * position['size']
            pnl_percentage = ((position['exit_price'] - position['entry_price']) / position['entry_price']) * 100
        else:
            pnl = (position['entry_price'] - position['exit_price']) * position['size']
            pnl_percentage = ((position['entry_price'] - position['exit_price']) / position['entry_price']) * 100
        
        # Calculate fees
        fee_rate = self.config.get('fee_percentage', 0.001)
        fees = position['size'] * position['entry_price'] * fee_rate  # Entry fee
        fees += position['size'] * position['exit_price'] * fee_rate  # Exit fee
        
        position['fees_paid'] = fees
        position['pnl'] = pnl - fees
        position['pnl_percentage'] = pnl_percentage

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of all active positions"""
        with self.lock:
            return [pos for pos in self.active_positions.values() if pos.get('status') == 'active']

    def check_exit_conditions(self, position: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a position should be closed based on exit conditions"""
        try:
            symbol = position['symbol']
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                return False, ""
            
            # Update current price in position
            position['current_price'] = current_price
            
            # Check stop loss
            if position['side'] == 'buy':
                # For buy positions
                if self.enable_stop_loss_check and current_price <= position['stop_loss']:
                    return True, "stop_loss"
                elif self.enable_take_profit_check and current_price >= position['take_profit']:
                    return True, "take_profit"
            else:  # sell position
                # For sell positions
                if self.enable_stop_loss_check and current_price >= position['stop_loss']:
                    return True, "stop_loss"
                elif self.enable_take_profit_check and current_price <= position['take_profit']:
                    return True, "take_profit"
            
            # Check maximum hold time (optional)
            if hasattr(self, 'max_hold_hours'):
                hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
                if hold_time > self.max_hold_hours:
                    return True, "max_hold_time"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False, ""
    
    def update_statistics(self, position: Dict[str, Any]):
        """Update trading statistics"""
        self.total_trades += 1
        
        if position['pnl'] > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_pnl += position['pnl']
        
        # Update peak and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def check_stop_loss_take_profit(self):
        """Check all active positions for stop loss or take profit hits"""
        if not self.enable_stop_loss_check and not self.enable_take_profit_check:
            return
        
        try:
            with self.lock:
                positions_to_check = list(self.active_positions.items())
            
            for position_id, position in positions_to_check:
                if position['status'] != 'active':
                    continue
                
                symbol = position['symbol']
                
                # Get current price
                current_price = self.get_current_price(symbol)
                if not current_price:
                    continue
                
                position['current_price'] = current_price
                hit = False
                event_type = None
                
                # Update trailing stop if enabled
                if self.enable_trailing_stop and position.get('trailing_stop'):
                    if position['side'] == 'buy':
                        if current_price > position.get('highest_price', current_price):
                            position['highest_price'] = current_price
                            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                            if new_trailing_stop > position['trailing_stop']:
                                position['trailing_stop'] = new_trailing_stop
                                position['stop_loss'] = max(position['stop_loss'], new_trailing_stop)
                    else:  # sell
                        if current_price < position.get('lowest_price', current_price):
                            position['lowest_price'] = current_price
                            new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
                            if new_trailing_stop < position['trailing_stop']:
                                position['trailing_stop'] = new_trailing_stop
                                position['stop_loss'] = min(position['stop_loss'], new_trailing_stop)
                
                # Check based on position side
                if position['side'] == 'buy':
                    # For buy positions
                    if self.enable_stop_loss_check and current_price <= position['stop_loss']:
                        logger.info(f"Stop loss hit for {symbol} buy position at ${current_price:.2f}")
                        hit = True
                        event_type = 'STOP_LOSS_HIT'
                        
                    elif self.enable_take_profit_check and current_price >= position['take_profit']:
                        logger.info(f"Take profit hit for {symbol} buy position at ${current_price:.2f}")
                        hit = True
                        event_type = 'TAKE_PROFIT_HIT'
                        
                else:  # sell position
                    # For sell positions
                    if self.enable_stop_loss_check and current_price >= position['stop_loss']:
                        logger.info(f"Stop loss hit for {symbol} sell position at ${current_price:.2f}")
                        hit = True
                        event_type = 'STOP_LOSS_HIT'
                        
                    elif self.enable_take_profit_check and current_price <= position['take_profit']:
                        logger.info(f"Take profit hit for {symbol} sell position at ${current_price:.2f}")
                        hit = True
                        event_type = 'TAKE_PROFIT_HIT'
                
                if hit:
                    # Update position
                    position['exit_price'] = current_price
                    position['exit_time'] = datetime.now()
                    position['status'] = 'closed'
                    position['exit_reason'] = 'stop_loss' if event_type == 'STOP_LOSS_HIT' else 'take_profit'
                    
                    # Calculate final P&L
                    self.calculate_pnl(position)
                    
                    with self.lock:
                        # Update balance
                        exit_value = position['size'] * current_price
                        self.balance += exit_value
                        
                        # Update statistics
                        self.update_statistics(position)
                        
                        # Move to completed trades
                        self.completed_trades[position_id] = position
                        del self.active_positions[position_id]
                    
                    # Send webhook with correct event type
                    self.send_webhook(event_type, position)
                    
                    # Save completed trade
                    self.save_completed_trade(position)
                    self.update_position_in_db(position)
                    
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {e}")
    
    def execute_trade(self, decision: Dict[str, Any]) -> bool:
        """Execute a trading decision"""
        try:
            symbol = decision['symbol']
            action = decision['action']
            
            # Skip if no action needed
            if action == 'hold':
                return False
            
            # Check if we already have a position
            existing_position = self.get_position(symbol)
            
            if existing_position:
                # Check if we should close or keep the position
                if existing_position['side'] != action:
                    # Close existing position
                    self.close_position(existing_position['id'], "signal_reversal")
                    # Wait a moment
                    time.sleep(1)
                    # Open new position
                    return self.open_position(decision)
                else:
                    # Same direction, keep the position
                    logger.info(f"Already have {action} position for {symbol}")
                    return False
            else:
                # Open new position
                return self.open_position(decision)
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get active position for a symbol"""
        with self.lock:
            for position in self.active_positions.values():
                if position['symbol'] == symbol and position['status'] == 'active':
                    return position
        return None
    
    def update_positions(self):
        """Update current prices and P&L for all positions"""
        with self.lock:
            positions_to_update = list(self.active_positions.values())
        
        for position in positions_to_update:
            if position['status'] != 'active':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if current_price:
                position['current_price'] = current_price
                
                # Calculate unrealized P&L
                if position['side'] == 'buy':
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                
                position['unrealized_pnl'] = unrealized_pnl
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        self.update_positions()
        
        with self.lock:
            total_value = self.balance
            positions_value = 0
            
            for position in self.active_positions.values():
                if position['status'] == 'active':
                    positions_value += position['size'] * position['current_price']
            
            total_value += positions_value
            
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            return {
                'balance': self.balance,
                'positions_value': positions_value,
                'total_value': total_value,
                'total_pnl': self.total_pnl,
                'total_pnl_percentage': ((total_value - self.initial_balance) / self.initial_balance) * 100,
                'active_positions': len([p for p in self.active_positions.values() if p['status'] == 'active']),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'max_drawdown': self.max_drawdown,
                'positions': list(self.active_positions.values())
            }
    
    # Database methods
    def save_position_to_db(self, position: Dict[str, Any]):
        """Save position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop REAL,
                    status TEXT DEFAULT 'active',
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    pnl REAL DEFAULT 0,
                    pnl_percentage REAL DEFAULT 0,
                    model_source TEXT,
                    confidence REAL DEFAULT 0,
                    market_regime TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO active_positions 
                (id, symbol, side, size, entry_price, stop_loss, take_profit, 
                 trailing_stop, status, entry_time, model_source, confidence, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['id'], position['symbol'], position['side'],
                position['size'], position['entry_price'],
                position['stop_loss'], position['take_profit'],
                position.get('trailing_stop'), position['status'], 
                position['entry_time'], position['model_source'], 
                position['confidence'], position.get('market_regime', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")
    
    def update_position_in_db(self, position: Dict[str, Any]):
        """Update position in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE active_positions 
                SET status = ?, exit_time = ?, exit_price = ?, 
                    pnl = ?, pnl_percentage = ?, trailing_stop = ?
                WHERE id = ?
            """, (
                position['status'], position.get('exit_time'),
                position.get('exit_price'), position.get('pnl'),
                position.get('pnl_percentage'), position.get('trailing_stop'),
                position['id']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating position in database: {e}")
    
    def save_completed_trade(self, position: Dict[str, Any]):
        """Save completed trade to database"""
        if not self.config.get('save_trade_history', True):
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completed_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    fees_paid REAL DEFAULT 0,
                    exit_reason TEXT,
                    model_source TEXT,
                    confidence REAL,
                    market_regime TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO completed_trades 
                (id, symbol, side, size, entry_price, exit_price, stop_loss, 
                 take_profit, entry_time, exit_time, pnl, pnl_percentage, 
                 fees_paid, exit_reason, model_source, confidence, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['id'], position['symbol'], position['side'],
                position['size'], position['entry_price'], position['exit_price'],
                position['stop_loss'], position['take_profit'],
                position['entry_time'], position['exit_time'],
                position['pnl'], position['pnl_percentage'],
                position.get('fees_paid', 0), position.get('exit_reason', ''),
                position.get('model_source', ''), position.get('confidence', 0),
                position.get('market_regime', '')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving completed trade: {e}")
    
    def load_positions_from_db(self):
        """Load active positions from database on startup"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='active_positions'
            """)
            
            if not cursor.fetchone():
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE active_positions (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        trailing_stop REAL,
                        status TEXT DEFAULT 'active',
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        exit_price REAL,
                        pnl REAL DEFAULT 0,
                        pnl_percentage REAL DEFAULT 0,
                        model_source TEXT,
                        confidence REAL DEFAULT 0,
                        market_regime TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            
            # Load active positions
            cursor.execute("""
                SELECT id, symbol, side, size, entry_price, stop_loss, take_profit,
                       trailing_stop, status, entry_time, exit_time, exit_price, 
                       pnl, pnl_percentage, model_source, confidence, market_regime
                FROM active_positions 
                WHERE status = 'active'
            """)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            for row in rows:
                position_dict = dict(zip(columns, row))
                
                # Create position with all required fields
                position = {
                    'id': position_dict['id'],
                    'symbol': position_dict['symbol'],
                    'side': position_dict['side'],
                    'size': position_dict['size'],
                    'entry_price': position_dict['entry_price'],
                    'stop_loss': position_dict['stop_loss'],
                    'take_profit': position_dict['take_profit'],
                    'trailing_stop': position_dict.get('trailing_stop'),
                    'status': position_dict['status'],
                    'entry_time': position_dict['entry_time'],
                    'exit_time': position_dict['exit_time'],
                    'exit_price': position_dict['exit_price'],
                    'pnl': position_dict['pnl'],
                    'pnl_percentage': position_dict['pnl_percentage'],
                    'model_source': position_dict.get('model_source', 'unknown'),
                    'confidence': position_dict.get('confidence', 0),
                    'market_regime': position_dict.get('market_regime', 'unknown'),
                    'notes': '',
                    'fees_paid': 0
                }
                
                # Get current price
                current_price = self.get_current_price(position['symbol'])
                position['current_price'] = current_price or position['entry_price']
                
                # Set highest/lowest price for trailing stops
                if position['side'] == 'buy':
                    position['highest_price'] = position['current_price']
                else:
                    position['lowest_price'] = position['current_price']
                
                self.active_positions[position['id']] = position
            
            conn.close()
            logger.info(f"Loaded {len(self.active_positions)} active positions from database")
            
        except Exception as e:
            logger.error(f"Error loading positions from database: {e}")
    
    def process_webhook(self, data: Dict[str, Any]):
        """Process incoming webhook data"""
        try:
            # Extract data from webhook
            ticker = data.get('ticker', '')
            action = data.get('action', '')
            
            # Convert ticker to symbol (BTCUSDT -> BTC/USDT)
            if ticker.endswith('USDT'):
                symbol = ticker[:-4] + '/USDT'
            else:
                symbol = ticker
            
            # Handle different action types
            if action in ['buy', 'sell']:
                # Open position signal
                decision = {
                    'symbol': symbol,
                    'action': action,
                    'confidence': float(data.get('confidence', 0.75)),
                    'model_name': 'webhook',
                    'market_regime': data.get('regime', 'unknown').lower()
                }
                
                # Execute trade
                self.execute_trade(decision)
                
            elif action.startswith('exit_'):
                # Close position signal
                position = self.get_position(symbol)
                if position:
                    reason = data.get('reason', 'webhook_signal')
                    self.close_position(position['id'], reason)
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
    
    def start(self):
        """Start the trading system"""
        logger.info("Starting trading system...")
        
        # Get webhook configuration from config
        webhook_host = self.config.get('webhook_host', '0.0.0.0')
        webhook_port = self.config.get('webhook_port', 5000)
        
        # Start Flask webhook server in a separate thread
        webhook_thread = threading.Thread(
            target=lambda: self.app.run(
                host=webhook_host, 
                port=webhook_port, 
                debug=False,
                use_reloader=False
            )
        )
        webhook_thread.daemon = True
        webhook_thread.start()
        
        logger.info("Trading system started successfully")
        logger.info(f"Webhook server running on {webhook_host}:{webhook_port}")
        logger.info(f"BTC webhooks: {self.webhook_config['BTC/USDT']['url']}")
        logger.info(f"SOL webhooks: {self.webhook_config['SOL/USDT']['url']}")
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        
        # Close all positions
        with self.lock:
            positions_to_close = list(self.active_positions.keys())
        
        for position_id in positions_to_close:
            self.close_position(position_id, "system_shutdown")
        
        logger.info("Trading system stopped")




# Helper function to create default config
def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'mode': 'paper',
        'initial_balance': 1000,
        'position_size_pct': 0.95,
        'max_positions': 2,
        'max_concurrent_positions': 2,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'db_path': 'crypto_trading.db',
        
        # Webhook configuration per pair
        'btc_webhook_url': 'http://localhost:5000/webhook/btc',
        'btc_webhook_enabled': True,
        'sol_webhook_url': 'http://localhost:5000/webhook/sol',
        'sol_webhook_enabled': True,
        'webhook_port': 5000,
        'webhook_host': '0.0.0.0',
        
        # Exchange credentials (optional for paper trading)
        'api_key': '',
        'api_secret': ''
    }

def shutdown(self):
    """Shutdown the trading system gracefully"""
    self.stop()
    
# Test functions
def test_webhook_formats():
    """Test webhook format generation"""
    config = create_default_config()
    trading_system = TradingSystem(config)
    
    print("Testing Webhook Formats")
    print("=" * 50)
    
    # Test BTC open position
    test_btc_open = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'entry_price': 104322,
        'confidence': 0.75,
        'market_regime': 'sideways'
    }
    
    print("\n1. BTC Buy Position Open:")
    trading_system.send_webhook('POSITION_OPENED', test_btc_open)
    
    # Test BTC TP hit
    test_btc_tp = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'entry_price': 104322,
        'exit_price': 104633,
        'current_price': 104633,
        'stop_loss': 104122,
        'take_profit': 104522,
        'size': 0.01
    }
    
    print("\n2. BTC Take Profit Hit:")
    trading_system.send_webhook('TAKE_PROFIT_HIT', test_btc_tp)
    
    # Test SOL SL hit
    test_sol_sl = {
        'symbol': 'SOL/USDT',
        'side': 'sell',
        'entry_price': 171.18,
        'exit_price': 171.14,
        'current_price': 171.14,
        'stop_loss': 171.14,
        'take_profit': 143.98,
        'size': 3.0
    }
    
    print("\n3. SOL Stop Loss Hit:")
    trading_system.send_webhook('STOP_LOSS_HIT', test_sol_sl)
    
    print("\nWebhook format testing completed!")


if __name__ == "__main__":
    # Run tests when executed directly
    test_webhook_formats()
