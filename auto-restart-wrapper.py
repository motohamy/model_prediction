"""
Auto-Restart Wrapper for Crypto Trading System
Alternative method that doesn't require Windows service installation
This script monitors and automatically restarts the main trading system
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timedelta
import psutil
import json
import threading
import signal
from pathlib import Path

# CONFIGURATION
PROJECT_DIR = r"C:\crypto_trading"  # Change to your project directory
MAIN_SCRIPT = "main.py"
PYTHON_EXECUTABLE = sys.executable
LOG_DIR = os.path.join(PROJECT_DIR, "wrapper_logs")
CONFIG_FILE = os.path.join(PROJECT_DIR, "wrapper_config.json")

# Default configuration
DEFAULT_CONFIG = {
    "max_restarts_per_hour": 5,
    "restart_delay_seconds": 30,
    "memory_limit_mb": 4096,
    "check_interval_seconds": 60,
    "auto_restart_on_crash": True,
    "auto_restart_on_high_memory": True,
    "log_retention_days": 30,
    "startup_delay_seconds": 10
}


class TradingSystemWrapper:
    """Wrapper to keep the trading system running continuously"""
    
    def __init__(self):
        self.setup_directories()
        self.load_config()
        self.setup_logging()
        
        self.process = None
        self.is_running = False
        self.restart_count = 0
        self.restart_timestamps = []
        self.start_time = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("=" * 60)
        self.logger.info("Trading System Wrapper Started")
        self.logger.info(f"Project Directory: {PROJECT_DIR}")
        self.logger.info(f"Main Script: {MAIN_SCRIPT}")
        self.logger.info("=" * 60)
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(PROJECT_DIR, exist_ok=True)
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        self.config = DEFAULT_CONFIG.copy()
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        else:
            # Save default config
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_logging(self):
        """Setup logging with rotation"""
        log_file = os.path.join(LOG_DIR, f"wrapper_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Clean old logs
        self.clean_old_logs()
    
    def clean_old_logs(self):
        """Remove log files older than retention period"""
        try:
            retention_days = self.config.get("log_retention_days", 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for log_file in Path(LOG_DIR).glob("wrapper_*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.logger.info(f"Deleted old log file: {log_file.name}")
        except Exception as e:
            self.logger.error(f"Error cleaning old logs: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def check_rate_limit(self):
        """Check if we've restarted too many times recently"""
        current_time = datetime.now()
        
        # Remove timestamps older than 1 hour
        self.restart_timestamps = [
            ts for ts in self.restart_timestamps 
            if (current_time - ts).total_seconds() < 3600
        ]
        
        # Check if we've exceeded the limit
        max_restarts = self.config.get("max_restarts_per_hour", 5)
        if len(self.restart_timestamps) >= max_restarts:
            self.logger.error(f"Restart limit reached ({max_restarts} per hour)")
            return False
        
        return True
    
    def start_trading_system(self):
        """Start the main trading system"""
        try:
            # Check rate limit
            if not self.check_rate_limit():
                self.logger.error("Cannot restart - rate limit exceeded")
                return False
            
            # Change to project directory
            os.chdir(PROJECT_DIR)
            
            # Prepare environment variables
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
            
            # Start the process
            self.logger.info(f"Starting trading system: {PYTHON_EXECUTABLE} {MAIN_SCRIPT}")
            
            self.process = subprocess.Popen(
                [PYTHON_EXECUTABLE, MAIN_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            self.restart_count += 1
            self.restart_timestamps.append(datetime.now())
            
            self.logger.info(f"Trading system started with PID: {self.process.pid}")
            self.logger.info(f"Total restarts: {self.restart_count}")
            
            # Start output monitoring thread
            output_thread = threading.Thread(target=self.monitor_output)
            output_thread.daemon = True
            output_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            return False
    
    def monitor_output(self):
        """Monitor and log the process output"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    # Log the output from the trading system
                    line = line.strip()
                    if 'ERROR' in line or 'CRITICAL' in line:
                        self.logger.error(f"[TRADING] {line}")
                    elif 'WARNING' in line:
                        self.logger.warning(f"[TRADING] {line}")
                    else:
                        self.logger.info(f"[TRADING] {line}")
        except Exception as e:
            self.logger.error(f"Error monitoring output: {e}")
    
    def check_process_health(self):
        """Check if the process is healthy"""
        if self.process is None:
            return False
        
        # Check if process is still running
        poll = self.process.poll()
        if poll is not None:
            self.logger.warning(f"Trading system exited with code: {poll}")
            return False
        
        try:
            # Check memory usage
            proc = psutil.Process(self.process.pid)
            memory_mb = proc.memory_info().rss / 1024 / 1024
            
            self.logger.debug(f"Trading system memory usage: {memory_mb:.0f}MB")
            
            # Check if memory limit exceeded
            memory_limit = self.config.get("memory_limit_mb", 4096)
            if memory_mb > memory_limit:
                self.logger.warning(f"Memory limit exceeded: {memory_mb:.0f}MB > {memory_limit}MB")
                if self.config.get("auto_restart_on_high_memory", True):
                    return False
            
            # Check CPU usage
            cpu_percent = proc.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            
            return True
            
        except psutil.NoSuchProcess:
            self.logger.error("Process no longer exists")
            return False
        except Exception as e:
            self.logger.error(f"Error checking process health: {e}")
            return True  # Assume healthy if we can't check
    
    def stop_trading_system(self):
        """Stop the trading system gracefully"""
        if self.process:
            try:
                self.logger.info(f"Stopping trading system (PID: {self.process.pid})")
                
                # Try graceful termination
                self.process.terminate()
                
                # Wait for process to exit
                try:
                    self.process.wait(timeout=30)
                    self.logger.info("Trading system stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.logger.warning("Graceful shutdown failed, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                
            except Exception as e:
                self.logger.error(f"Error stopping trading system: {e}")
    
    def run(self):
        """Main execution loop"""
        self.is_running = True
        
        # Initial startup delay
        startup_delay = self.config.get("startup_delay_seconds", 10)
        self.logger.info(f"Waiting {startup_delay} seconds before starting...")
        time.sleep(startup_delay)
        
        # Start the trading system
        if not self.start_trading_system():
            self.logger.error("Failed to start trading system initially")
            return
        
        # Main monitoring loop
        check_interval = self.config.get("check_interval_seconds", 60)
        
        while self.is_running:
            try:
                # Wait for the specified interval
                time.sleep(check_interval)
                
                # Check system health
                if not self.check_process_health():
                    if self.config.get("auto_restart_on_crash", True):
                        self.logger.warning("Trading system unhealthy, restarting...")
                        
                        # Stop the current process
                        self.stop_trading_system()
                        
                        # Wait before restarting
                        restart_delay = self.config.get("restart_delay_seconds", 30)
                        self.logger.info(f"Waiting {restart_delay} seconds before restart...")
                        time.sleep(restart_delay)
                        
                        # Restart the trading system
                        if not self.start_trading_system():
                            self.logger.error("Failed to restart trading system")
                            time.sleep(300)  # Wait 5 minutes before trying again
                    else:
                        self.logger.error("Trading system crashed and auto-restart is disabled")
                        break
                
                # Log status periodically (every 10 checks)
                if self.restart_count % 10 == 0:
                    uptime = datetime.now() - self.start_time
                    self.logger.info(f"Status: Uptime={uptime}, Restarts={self.restart_count}")
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the wrapper and trading system"""
        self.logger.info("Stopping wrapper...")
        self.is_running = False
        self.stop_trading_system()
        self.logger.info("Wrapper stopped")
    
    def get_status(self):
        """Get current status information"""
        status = {
            "running": self.is_running,
            "process_pid": self.process.pid if self.process else None,
            "restart_count": self.restart_count,
            "start_time": self.start_time.isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "recent_restarts": len(self.restart_timestamps)
        }
        
        if self.process:
            try:
                proc = psutil.Process(self.process.pid)
                status["memory_mb"] = proc.memory_info().rss / 1024 / 1024
                status["cpu_percent"] = proc.cpu_percent()
            except:
                pass
        
        return status


def create_startup_batch_file():
    """Create a batch file for easy startup"""
    batch_content = f"""@echo off
title Crypto Trading System Wrapper
cd /d {PROJECT_DIR}
{PYTHON_EXECUTABLE} {os.path.abspath(__file__)}
pause
"""
    
    batch_file = os.path.join(PROJECT_DIR, "start_wrapper.bat")
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    print(f"Created startup batch file: {batch_file}")
    return batch_file


def main():
    """Main entry point"""
    print("=" * 60)
    print("Crypto Trading System Auto-Restart Wrapper")
    print("=" * 60)
    
    # Create startup batch file
    batch_file = create_startup_batch_file()
    print(f"You can use {batch_file} to start the wrapper")
    print("")
    
    # Create and run wrapper
    wrapper = TradingSystemWrapper()
    
    try:
        wrapper.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        wrapper.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        wrapper.stop()
        raise


if __name__ == "__main__":
    main()
