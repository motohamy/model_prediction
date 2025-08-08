"""
Windows Service Wrapper for Crypto Trading System
This script creates a Windows service that runs your trading system continuously
even when you're not connected to the server.
"""

import os
import sys
import time
import win32serviceutil
import win32service
import win32event
import win32api
import servicemanager
import socket
import traceback
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import threading
from datetime import datetime
import psutil
import json

# CONFIGURATION - CHANGE THESE PATHS TO MATCH YOUR SETUP
PROJECT_DIR = r"C:\crypto_trading"  # Change to your actual project directory
PYTHON_PATH = r"C:\Python310\python.exe"  # Change to your Python installation path
MAIN_SCRIPT = "main.py"  # Your main trading script
SERVICE_NAME = "CryptoTradingBot"
SERVICE_DISPLAY_NAME = "Crypto Trading Bot Service"
SERVICE_DESCRIPTION = "Automated crypto trading system for BTC and SOL"


class CryptoTradingService(win32serviceutil.ServiceFramework):
    """Windows Service wrapper for the crypto trading system"""
    
    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = SERVICE_DISPLAY_NAME
    _svc_description_ = SERVICE_DESCRIPTION
    
    def __init__(self, args):
        """Initialize the service"""
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        
        # Setup logging
        self.setup_logging()
        
        # Process handle
        self.process = None
        self.monitor_thread = None
        self.is_running = False
        
        self.logger.info(f"Service initialized at {datetime.now()}")
    
    def setup_logging(self):
        """Setup rotating log files"""
        log_dir = os.path.join(PROJECT_DIR, "service_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "crypto_service.log")
        
        # Create logger
        self.logger = logging.getLogger('CryptoService')
        self.logger.setLevel(logging.INFO)
        
        # Create rotating file handler (10MB per file, keep 10 files)
        handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=10
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def SvcStop(self):
        """Stop the service"""
        self.logger.info("Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
        # Signal to stop
        self.is_running = False
        win32event.SetEvent(self.hWaitStop)
        
        # Stop the trading process
        self.stop_trading_process()
    
    def SvcDoRun(self):
        """Main service loop"""
        try:
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            
            self.logger.info("Service started successfully")
            self.is_running = True
            
            # Start the main loop
            self.main()
            
        except Exception as e:
            self.logger.error(f"Service error: {e}\n{traceback.format_exc()}")
            servicemanager.LogErrorMsg(f"Service error: {e}")
    
    def main(self):
        """Main service execution loop"""
        self.logger.info("Starting main service loop")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Main loop
        while self.is_running:
            try:
                # Start or restart the trading process if needed
                if not self.is_process_running():
                    self.logger.info("Trading process not running, starting it...")
                    self.start_trading_process()
                
                # Wait for stop signal or timeout (check every 30 seconds)
                rc = win32event.WaitForSingleObject(self.hWaitStop, 30000)
                
                if rc == win32event.WAIT_OBJECT_0:
                    # Stop signal received
                    self.logger.info("Stop signal received")
                    break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
                time.sleep(10)  # Wait before retry
    
    def start_trading_process(self):
        """Start the trading system process"""
        try:
            # Change to project directory
            os.chdir(PROJECT_DIR)
            
            # Prepare the command
            cmd = [PYTHON_PATH, MAIN_SCRIPT]
            
            self.logger.info(f"Starting trading process: {' '.join(cmd)}")
            
            # Create startup info to hide the console window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            
            self.logger.info(f"Trading process started with PID: {self.process.pid}")
            
            # Start a thread to monitor the process output
            output_thread = threading.Thread(target=self.monitor_process_output)
            output_thread.daemon = True
            output_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start trading process: {e}\n{traceback.format_exc()}")
    
    def stop_trading_process(self):
        """Stop the trading process gracefully"""
        try:
            if self.process:
                self.logger.info(f"Stopping trading process (PID: {self.process.pid})")
                
                # Try graceful termination first
                self.process.terminate()
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                    self.logger.info("Trading process stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.logger.warning("Graceful shutdown failed, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                
        except Exception as e:
            self.logger.error(f"Error stopping trading process: {e}")
    
    def is_process_running(self):
        """Check if the trading process is running"""
        if self.process is None:
            return False
        
        poll = self.process.poll()
        if poll is None:
            # Process is still running
            return True
        else:
            # Process has terminated
            self.logger.warning(f"Trading process terminated with code: {poll}")
            self.process = None
            return False
    
    def monitor_process_output(self):
        """Monitor and log the process output"""
        try:
            # Read stdout in a separate thread
            for line in iter(self.process.stdout.readline, b''):
                if line:
                    self.logger.info(f"[TRADING] {line.decode('utf-8').strip()}")
            
            # Read stderr
            for line in iter(self.process.stderr.readline, b''):
                if line:
                    self.logger.error(f"[TRADING ERROR] {line.decode('utf-8').strip()}")
                    
        except Exception as e:
            self.logger.error(f"Error monitoring process output: {e}")
    
    def monitor_system(self):
        """Monitor system resources and health"""
        while self.is_running:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage(PROJECT_DIR)
                
                # Log system status every 5 minutes
                self.logger.info(f"System Status - CPU: {cpu_percent}%, "
                               f"Memory: {memory.percent}%, "
                               f"Disk: {disk.percent}%")
                
                # Alert if resources are critical
                if cpu_percent > 90:
                    self.logger.warning(f"High CPU usage: {cpu_percent}%")
                
                if memory.percent > 90:
                    self.logger.warning(f"High memory usage: {memory.percent}%")
                
                if disk.percent > 95:
                    self.logger.error(f"Critical disk space: {disk.percent}%")
                
                # Check if process is using too much memory
                if self.process:
                    try:
                        proc = psutil.Process(self.process.pid)
                        proc_memory = proc.memory_info().rss / 1024 / 1024  # MB
                        
                        if proc_memory > 2048:  # If using more than 2GB
                            self.logger.warning(f"Trading process using {proc_memory:.0f}MB RAM")
                            
                            if proc_memory > 4096:  # Restart if using more than 4GB
                                self.logger.error("Trading process memory usage too high, restarting...")
                                self.stop_trading_process()
                                time.sleep(5)
                                self.start_trading_process()
                                
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                time.sleep(60)


class ServiceManager:
    """Helper class to manage the Windows service"""
    
    @staticmethod
    def install():
        """Install the Windows service"""
        print(f"Installing {SERVICE_NAME} service...")
        
        # Get the Python executable and script paths
        python_path = sys.executable
        script_path = os.path.abspath(__file__)
        
        # Install command
        win32serviceutil.InstallService(
            None,
            SERVICE_NAME,
            SERVICE_DISPLAY_NAME,
            startType=win32service.SERVICE_AUTO_START,
            description=SERVICE_DESCRIPTION
        )
        
        print(f"Service {SERVICE_NAME} installed successfully!")
        print(f"Service will run from: {PROJECT_DIR}")
        print(f"Using Python: {PYTHON_PATH}")
    
    @staticmethod
    def uninstall():
        """Uninstall the Windows service"""
        print(f"Uninstalling {SERVICE_NAME} service...")
        win32serviceutil.RemoveService(SERVICE_NAME)
        print(f"Service {SERVICE_NAME} uninstalled successfully!")
    
    @staticmethod
    def start():
        """Start the Windows service"""
        print(f"Starting {SERVICE_NAME} service...")
        win32serviceutil.StartService(SERVICE_NAME)
        print(f"Service {SERVICE_NAME} started successfully!")
    
    @staticmethod
    def stop():
        """Stop the Windows service"""
        print(f"Stopping {SERVICE_NAME} service...")
        win32serviceutil.StopService(SERVICE_NAME)
        print(f"Service {SERVICE_NAME} stopped successfully!")
    
    @staticmethod
    def restart():
        """Restart the Windows service"""
        print(f"Restarting {SERVICE_NAME} service...")
        win32serviceutil.RestartService(SERVICE_NAME)
        print(f"Service {SERVICE_NAME} restarted successfully!")
    
    @staticmethod
    def status():
        """Check the status of the Windows service"""
        try:
            status = win32serviceutil.QueryServiceStatus(SERVICE_NAME)
            status_string = {
                win32service.SERVICE_STOPPED: "STOPPED",
                win32service.SERVICE_START_PENDING: "STARTING",
                win32service.SERVICE_STOP_PENDING: "STOPPING",
                win32service.SERVICE_RUNNING: "RUNNING",
                win32service.SERVICE_CONTINUE_PENDING: "CONTINUING",
                win32service.SERVICE_PAUSE_PENDING: "PAUSING",
                win32service.SERVICE_PAUSED: "PAUSED"
            }
            print(f"Service {SERVICE_NAME} is {status_string.get(status[1], 'UNKNOWN')}")
            return status[1]
        except Exception as e:
            print(f"Error checking service status: {e}")
            return None


def main():
    """Main entry point for service management"""
    if len(sys.argv) == 1:
        # If no arguments, run as service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(CryptoTradingService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Handle command line arguments
        if sys.argv[1].lower() == 'install':
            ServiceManager.install()
        elif sys.argv[1].lower() == 'uninstall':
            ServiceManager.uninstall()
        elif sys.argv[1].lower() == 'start':
            ServiceManager.start()
        elif sys.argv[1].lower() == 'stop':
            ServiceManager.stop()
        elif sys.argv[1].lower() == 'restart':
            ServiceManager.restart()
        elif sys.argv[1].lower() == 'status':
            ServiceManager.status()
        else:
            print("Usage: python crypto_service.py [install|uninstall|start|stop|restart|status]")
            print("\nCommands:")
            print("  install   - Install the Windows service")
            print("  uninstall - Uninstall the Windows service")
            print("  start     - Start the service")
            print("  stop      - Stop the service")
            print("  restart   - Restart the service")
            print("  status    - Check service status")


if __name__ == '__main__':
    main()
