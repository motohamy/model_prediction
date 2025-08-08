"""
Quick Installer for Crypto Trading System
Run this script to automatically set up everything
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path

# Configuration
PROJECT_DIR = r"C:\crypto_trading"
PYTHON_PATH = sys.executable

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    if description:
        print(f"📦 {description}")
    print(f"Running: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ Failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def install_packages():
    """Install required Python packages"""
    packages = [
        "pywin32",
        "pywin32-ctypes", 
        "psutil",
        "ccxt",
        "aiohttp",
        "pandas",
        "numpy",
        "ta",
        "flask",
        "requests",
        "websocket-client",
        "scikit-learn",
        "xgboost",
        "joblib",
        "pycoingecko",
        "textblob",
        "pyarrow",
        "boruta",
        "torch"  # This might take a while
    ]
    
    print("\n" + "="*60)
    print("📦 Installing Required Packages")
    print("="*60)
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if not run_command(f"pip install {package}"):
            print(f"⚠️ Warning: Failed to install {package}")
    
    print("\n✅ Package installation completed")

def create_directories():
    """Create necessary directories"""
    directories = [
        PROJECT_DIR,
        os.path.join(PROJECT_DIR, "models"),
        os.path.join(PROJECT_DIR, "models", "ensemble"),
        os.path.join(PROJECT_DIR, "models", "boruta_cnn_lstm"),
        os.path.join(PROJECT_DIR, "models", "helformer"),
        os.path.join(PROJECT_DIR, "models", "temporal_fusion"),
        os.path.join(PROJECT_DIR, "models", "sentiment_analyzer"),
        os.path.join(PROJECT_DIR, "data"),
        os.path.join(PROJECT_DIR, "logs"),
        os.path.join(PROJECT_DIR, "service_logs"),
        os.path.join(PROJECT_DIR, "wrapper_logs"),
        os.path.join(PROJECT_DIR, "analytics"),
        os.path.join(PROJECT_DIR, "analytics", "feedback"),
    ]
    
    print("\n" + "="*60)
    print("📁 Creating Directory Structure")
    print("="*60)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_startup_scripts():
    """Create helpful startup scripts"""
    
    # Create start_service.bat
    service_bat = f"""@echo off
title Crypto Trading Service Manager
cd /d {PROJECT_DIR}
echo ========================================
echo    Crypto Trading Service Manager
echo ========================================
echo.
echo 1. Install Service
echo 2. Start Service
echo 3. Stop Service
echo 4. Restart Service
echo 5. Check Status
echo 6. Uninstall Service
echo.
set /p choice="Enter your choice (1-6): "

if %choice%==1 {PYTHON_PATH} crypto_service.py install
if %choice%==2 {PYTHON_PATH} crypto_service.py start
if %choice%==3 {PYTHON_PATH} crypto_service.py stop
if %choice%==4 {PYTHON_PATH} crypto_service.py restart
if %choice%==5 {PYTHON_PATH} crypto_service.py status
if %choice%==6 {PYTHON_PATH} crypto_service.py uninstall

pause
"""
    
    # Create test_run.bat
    test_bat = f"""@echo off
title Test Trading System
cd /d {PROJECT_DIR}
echo Testing Trading System...
{PYTHON_PATH} main.py
pause
"""
    
    # Create view_logs.bat
    logs_bat = f"""@echo off
title View Trading Logs
cd /d {PROJECT_DIR}
echo ========================================
echo         Trading System Logs
echo ========================================
echo.
if exist service_logs\\crypto_service.log (
    echo === Last 50 lines of service log ===
    powershell Get-Content service_logs\\crypto_service.log -Tail 50
) else (
    echo No service logs found yet.
)
echo.
pause
"""
    
    # Create emergency_stop.bat
    stop_bat = f"""@echo off
title Emergency Stop
echo EMERGENCY STOP - Killing all Python processes...
taskkill /F /IM python.exe
echo.
echo All Python processes terminated.
echo To restart the service, run start_service.bat
pause
"""
    
    print("\n" + "="*60)
    print("📝 Creating Startup Scripts")
    print("="*60)
    
    scripts = {
        "start_service.bat": service_bat,
        "test_run.bat": test_bat,
        "view_logs.bat": logs_bat,
        "emergency_stop.bat": stop_bat
    }
    
    for filename, content in scripts.items():
        filepath = os.path.join(PROJECT_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Created: {filename}")

def create_config_file():
    """Create default configuration file"""
    config = {
        "mode": "paper",
        "initial_balance": 1000,
        "symbols": ["BTC/USDT", "SOL/USDT"],
        "confidence_threshold": 0.6,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.04,
        "max_concurrent_positions": 2,
        "enable_webhooks": True,
        "btc_webhook_url": "https://api.primeautomation.ai/webhook/ChartPrime/bafbdc00-a670-48ad-9624-c7c059f2c385",
        "sol_webhook_url": "https://api.primeautomation.ai/webhook/ChartPrime/ca60dbfd-46b9-4a44-bdec-43c0a024a379",
        "db_path": os.path.join(PROJECT_DIR, "crypto_trading.db")
    }
    
    config_file = os.path.join(PROJECT_DIR, "trading_config.json")
    
    print("\n" + "="*60)
    print("⚙️ Creating Configuration File")
    print("="*60)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Created: trading_config.json")
    print("\n📌 Important: Edit this file to add your API keys and adjust settings")

def copy_trading_files():
    """Copy trading system files to project directory"""
    print("\n" + "="*60)
    print("📋 Copying Trading System Files")
    print("="*60)
    
    current_dir = os.getcwd()
    
    files_to_copy = [
        "main.py",
        "complete_prediction_models.py",
        "crypto_data.py",
        "ensemble_manager.py",
        "feedback_learner.py",
        "trading_system.py",
        "crypto_service.py",  # The service wrapper
        "wrapper.py"  # The auto-restart wrapper
    ]
    
    for filename in files_to_copy:
        source = os.path.join(current_dir, filename)
        destination = os.path.join(PROJECT_DIR, filename)
        
        if os.path.exists(source):
            shutil.copy2(source, destination)
            print(f"✅ Copied: {filename}")
        else:
            print(f"⚠️ Not found: {filename} (you'll need to copy it manually)")

def test_installation():
    """Test if everything is installed correctly"""
    print("\n" + "="*60)
    print("🔍 Testing Installation")
    print("="*60)
    
    # Test Python
    print("\n1. Python Version:")
    run_command("python --version")
    
    # Test key packages
    print("\n2. Testing Key Packages:")
    test_imports = [
        "import ccxt",
        "import torch",
        "import xgboost",
        "import win32service",
        "import psutil"
    ]
    
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
        except ImportError as e:
            print(f"❌ {test_import} - {e}")
    
    # Check if main files exist
    print("\n3. Checking Project Files:")
    required_files = ["main.py", "crypto_service.py"]
    for file in required_files:
        filepath = os.path.join(PROJECT_DIR, file)
        if os.path.exists(filepath):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")

def main():
    """Main installation process"""
    print("="*60)
    print("🚀 CRYPTO TRADING SYSTEM - QUICK INSTALLER")
    print("="*60)
    print(f"\nProject Directory: {PROJECT_DIR}")
    print(f"Python Path: {PYTHON_PATH}")
    
    # Check if running as administrator
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False
    
    if not is_admin:
        print("\n⚠️ WARNING: Not running as administrator!")
        print("Some features (like Windows Service) require admin privileges.")
        print("Right-click this script and select 'Run as administrator'")
        input("\nPress Enter to continue anyway...")
    
    steps = [
        ("Install Python Packages", install_packages),
        ("Create Directory Structure", create_directories),
        ("Copy Trading Files", copy_trading_files),
        ("Create Configuration File", create_config_file),
        ("Create Startup Scripts", create_startup_scripts),
        ("Test Installation", test_installation)
    ]
    
    for i, (description, func) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {description}")
        try:
            func()
        except Exception as e:
            print(f"❌ Error in {description}: {e}")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                break
    
    print("\n" + "="*60)
    print("✅ INSTALLATION COMPLETED!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Edit trading_config.json with your API keys")
    print("2. Test the system: Run test_run.bat")
    print("3. Install as service: Run start_service.bat and choose option 1")
    print("4. Start the service: Run start_service.bat and choose option 2")
    print("\n📁 All files are in: " + PROJECT_DIR)
    print("\n🔍 To monitor:")
    print("   - View logs: Run view_logs.bat")
    print("   - Check status: Run start_service.bat and choose option 5")
    print("\n⚠️ Emergency stop: Run emergency_stop.bat")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
