"""
Live Trading with Multi-Timeframe PPO Agent

This script runs live trading using the trained multi-timeframe PPO agent
with Interactive Brokers.
"""

import os
import argparse
import logging
import datetime
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import our components
from src.ibkr_interface import IBKRInterface
from src.ibkr_processor import IBKRProcessor
from src.live_trading_system import IBKRTradingSystem
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_default_features():
    """Create default feature lists."""
    # 1-hour features
    features_1h = [
        # Price-based
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma10', 'ma20', 'ma50',
        'ema5', 'ema10', 'ema20', 'ema50',
        'return_5', 'return_10', 'return_20',
        'high_low_ratio_5', 'high_low_ratio_10', 'high_low_ratio_20',
        'price_position_5', 'price_position_10', 'price_position_20',
        
        # Volume-based
        'volume_normalized_5', 'volume_normalized_10', 'volume_normalized_20',
        'volume_zscore', 'obv', 'obv_normalized',
        
        # Technical indicators
        'rsi_14', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_lower', 'bb_mid', 'bb_position',
        'adx', 'atr', 'atr_percent',
        
        # Volatility and regime
        'volatility_5', 'volatility_10', 'volatility_20',
        'realized_vol_5', 'realized_vol_10', 'realized_vol_20',
        'trend_strength', 'trend_direction',
        
        # Momentum
        'momentum_5', 'momentum_10', 'momentum_20'
    ]
    
    # 4-hour features (subset of 1h for efficiency)
    features_4h = [
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma20', 'ma50',
        'ema5', 'ema20', 'ema50',
        'return_5', 'return_20',
        'rsi_14', 'macd',
        'adx', 'atr_percent',
        'volatility_10', 'realized_vol_10',
        'trend_direction',
        'momentum_5', 'momentum_20'
    ]
    
    return features_1h, features_4h

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dict with configuration
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def validate_market_hours():
    """
    Check if the current time is within market hours.
    
    Returns:
        bool: True if within market hours, False otherwise
    """
    now = datetime.datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Convert to US Eastern time (approximate check)
    # This is a simplified conversion - in production you should use pytz
    eastern_hour = (now.hour - 3) % 24  # Assuming GMT-3 for Eastern Time
    
    # Regular trading hours: 9:30 AM - 4:00 PM Eastern
    if eastern_hour < 9 or eastern_hour >= 16:
        return False
    if eastern_hour == 9 and now.minute < 30:
        return False
    
    return True

def setup_trading_system(args):
    """
    Set up and return the trading system.
    
    Args:
        args: Command line arguments
        
    Returns:
        Trading system instance
    """
    # Get default features
    features_1h, features_4h = create_default_features()
    
    # Create trading system
    try:
        trading_system = IBKRTradingSystem(
            model_path=args.model_path,
            ticker_list=args.ticker_list,
            features_1h=features_1h,
            features_4h=features_4h,
            lookback_window_1h=args.lookback_window_1h,
            lookback_window_4h=args.lookback_window_4h,
            max_position=args.max_position,
            commission=args.commission,
            ibkr_host=args.ibkr_host,
            ibkr_port=args.ibkr_port,
            ibkr_client_id=args.ibkr_client_id,
            is_paper=args.paper,
            check_interval=args.check_interval,
            hidden_dim=args.hidden_dim,
            use_attention=not args.no_attention,
            regime_detection=not args.no_regime,
            max_trades_per_day=args.max_trades_per_day,
            trade_records_path=args.trade_records,
            device=args.device
        )
        return trading_system
    except Exception as e:
        logger.error(f"Error setting up trading system: {e}")
        return None

def start_trading(trading_system, args):
    """
    Start trading with the given system.
    
    Args:
        trading_system: Trading system instance
        args: Command line arguments
    """
    # Check if we're in market hours
    if not validate_market_hours() and not args.force:
        logger.warning("Outside market hours. Use --force to override.")
        print("Outside market hours. Use --force to override.")
        return
    
    if args.dry_run:
        logger.info("Dry run mode - testing system without actual trading")
        print("Dry run mode - testing system without actual trading")
        
        # Run a single step without actual trading
        try:
            # Connect to IBKR
            if not trading_system.ibkr.connected:
                trading_system.ibkr.connect()
            
            # Update positions
            trading_system.update_positions()
            
            # Get account balance
            balance = trading_system.processor.get_account_balance()
            print(f"Account balance: ${balance:.2f}")
            
            # Check day trades
            day_trades_used = trading_system.ibkr.day_trades_count
            day_trades_limit = trading_system.ibkr.max_day_trades
            print(f"Day trades used: {day_trades_used}/{day_trades_limit}")
            
            # Execute one step without actual trading
            print("Running test trading step...")
            original_place_market_order = trading_system.ibkr.place_market_order
            
            # Mock trading function for testing
            def mock_place_market_order(symbol, action, quantity):
                logger.info(f"[DRY RUN] Would execute: {action} {quantity} {symbol}")
                print(f"[DRY RUN] Would execute: {action} {quantity} {symbol}")
                return {"status": "submitted", "order_id": 0}
            
            # Replace with mock function
            trading_system.ibkr.place_market_order = mock_place_market_order
            
            # Execute trading step
            trading_system.trading_step()
            
            # Restore original function
            trading_system.ibkr.place_market_order = original_place_market_order
            
            print("Dry run completed successfully")
            
        except Exception as e:
            logger.error(f"Error during dry run: {e}")
            print(f"Error during dry run: {e}")
        finally:
            # Disconnect from IBKR
            if trading_system.ibkr.connected:
                trading_system.ibkr.disconnect()
    else:
        # Start actual trading
        logger.info("Starting live trading")
        print("Starting live trading...")
        
        try:
            trading_system.start_trading()
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            print("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error during trading: {e}")
            print(f"Error during trading: {e}")
        finally:
            # Ensure disconnection
            if trading_system.ibkr and trading_system.ibkr.connected:
                trading_system.ibkr.disconnect()
                logger.info("Disconnected from IBKR")

def main():
    """Main function for live trading."""
    parser = argparse.ArgumentParser(description='Live Trading with Multi-Timeframe PPO Agent')
    
    # Input parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--ticker_list', type=str, nargs='+', required=True,
                        help='List of tickers to trade')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides command line args)')
    
    # Environment parameters
    parser.add_argument('--lookback_window_1h', type=int, default=20,
                        help='Lookback window for 1h data')
    parser.add_argument('--lookback_window_4h', type=int, default=10,
                        help='Lookback window for 4h data')
    parser.add_argument('--max_position', type=float, default=0.5,
                        help='Maximum position size (0.5 = 50% of account)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate')
    parser.add_argument('--max_trades_per_day', type=int, default=3,
                        help='Maximum trades per day (PDT rule)')
    
    # IBKR parameters
    parser.add_argument('--ibkr_host', type=str, default='127.0.0.1',
                        help='IBKR host address')
    parser.add_argument('--ibkr_port', type=int, default=4002,
                        help='IBKR port (4001 for Gateway, 4002 for paper)')
    parser.add_argument('--ibkr_client_id', type=int, default=1,
                        help='IBKR client ID')
    parser.add_argument('--paper', action='store_true',
                        help='Use paper trading account')
    parser.add_argument('--check_interval', type=int, default=60,
                        help='Interval between trading checks (seconds)')
    
    # Agent parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for neural networks')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--no_regime', action='store_true',
                        help='Disable regime detection')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (cpu or cuda)')
    
    # Run parameters
    parser.add_argument('--dry_run', action='store_true',
                        help='Test system without actual trading')
    parser.add_argument('--force', action='store_true',
                        help='Force trading even outside market hours')
    parser.add_argument('--trade_records', type=str, default='trade_records.json',
                        help='Path to save trade records')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load config if specified
    if args.config:
        config = load_config(args.config)
        if config:
            # Override args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            logger.info(f"Loaded configuration from {args.config}")
    
    # Display configuration
    print("\nTrading Configuration:")
    print(f"Model: {args.model_path}")
    print(f"Tickers: {args.ticker_list}")
    print(f"Max Position: {args.max_position * 100}% of account")
    print(f"Check Interval: {args.check_interval} seconds")
    print(f"IBKR Connection: {args.ibkr_host}:{args.ibkr_port} (Paper: {args.paper})")
    print(f"Mode: {'Dry Run (no actual trading)' if args.dry_run else 'Live Trading'}")
    print(f"Trade Records: {args.trade_records}\n")
    
    # Confirm before proceeding with live trading
    if not args.dry_run:
        confirmation = input("Are you sure you want to start live trading? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            print("Live trading aborted by user")
            return
    
    # Set up trading system
    trading_system = setup_trading_system(args)
    
    if trading_system:
        # Start trading
        start_trading(trading_system, args)
    else:
        logger.error("Failed to set up trading system")
        print("Failed to set up trading system. Check logs for details.")

if __name__ == "__main__":
    main()