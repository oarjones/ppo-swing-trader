"""
Live Trading System for Interactive Brokers

This module provides a complete system for live trading using the trained
multi-timeframe PPO agent with Interactive Brokers.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import datetime
import time
import json
import argparse
from typing import List, Dict, Tuple, Any, Optional, Union

# Import our components
from src.ibkr_interface import IBKRInterface
from src.ibkr_processor import IBKRProcessor
from finrl_multi_timeframe_env import FinRLMultiTimeframeEnv

# Import ElegantRL components if available

from elegantrl_multi_timeframe_model import MultiTimeframePPOAgent
ELEGANTRL_AVAILABLE = True


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBKRTradingSystem:
    """
    Live trading system for Interactive Brokers using our PPO agent.
    """
    
    def __init__(self,
                model_path: str,
                ticker_list: List[str],
                features_1h: List[str],
                features_4h: List[str],
                lookback_window_1h: int = 20,
                lookback_window_4h: int = 10,
                max_position: float = 1.0,
                commission: float = 0.001,
                ibkr_host: str = '127.0.0.1',
                ibkr_port: int = 4002,
                ibkr_client_id: int = 1,
                is_paper: bool = True,
                check_interval: int = 60,  # Seconds between checks
                hidden_dim: int = 128,
                use_attention: bool = True,
                regime_detection: bool = True,
                max_trades_per_day: int = 3,
                trade_records_path: str = 'trade_records.json',
                device: str = 'cpu'):
        """
        Initialize the trading system.
        
        Args:
            model_path: Path to the trained model
            ticker_list: List of tickers to trade
            features_1h: List of feature names for 1h data
            features_4h: List of feature names for 4h data
            lookback_window_1h: Lookback window for 1h data
            lookback_window_4h: Lookback window for 4h data
            max_position: Maximum position size (1.0 = 100% of account)
            commission: Commission rate
            ibkr_host: IBKR host address
            ibkr_port: IBKR port
            ibkr_client_id: IBKR client ID
            is_paper: Whether to use paper trading
            check_interval: Interval between trading checks (seconds)
            hidden_dim: Hidden dimension for neural networks
            use_attention: Whether to use attention mechanism
            regime_detection: Whether to use regime detection
            max_trades_per_day: Maximum trades per day (PDT rule)
            trade_records_path: Path to save trade records
            device: Device to run model (cpu or cuda)
        """
        # Store parameters
        self.model_path = model_path
        self.ticker_list = ticker_list
        self.features_1h = features_1h
        self.features_4h = features_4h
        self.lookback_window_1h = lookback_window_1h
        self.lookback_window_4h = lookback_window_4h
        self.max_position = max_position
        self.commission = commission
        self.check_interval = check_interval
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.regime_detection = regime_detection
        self.max_trades_per_day = max_trades_per_day
        self.trade_records_path = trade_records_path
        self.device = device
        
        # Initialize components
        self.ibkr = IBKRInterface(
            host=ibkr_host,
            port=ibkr_port,
            client_id=ibkr_client_id,
            is_paper=is_paper
        )
        
        # Connect to IBKR
        if not self.ibkr.connect():
            raise ConnectionError("Failed to connect to Interactive Brokers")
        
        # Initialize processor
        self.processor = IBKRProcessor(
            ibkr_interface=self.ibkr,
            timeframes=['1h', '4h']
        )
        
        # Load trade records if exist
        self.trade_records = self._load_trade_records()
        
        # Initialize environment and agent
        self.env = None
        self.agent = None
        self._initialize_env_and_agent()
        
        # Track the state of the system
        self.is_running = False
        self.current_positions = {}
        self.last_check_time = None
        self.update_positions()
    
    def _initialize_env_and_agent(self):
        """Initialize environment and agent."""
        logger.info("Initializing environment and agent")
        
        try:
            # First, get some data to initialize the environment
            historical_data = self.processor.download_multi_timeframe_data(
                ticker_list=self.ticker_list,
                start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                timeframes=['1h', '4h']
            )
            
            # Create environment
            self.env = FinRLMultiTimeframeEnv(
                ibkr_interface=self.ibkr,
                ibkr_processor=self.processor,
                ticker_list=self.ticker_list,
                lookback_window_1h=self.lookback_window_1h,
                lookback_window_4h=self.lookback_window_4h,
                initial_amount=self.processor.get_account_balance(),
                commission_percentage=self.commission,
                max_position=self.max_position,
                max_trades_per_day=self.max_trades_per_day,
                features_1h=self.features_1h,
                features_4h=self.features_4h,
                live_trading=True
            )
            
            # Reset environment to initialize
            initial_state = self.env.reset()
            
            # Check if ElegantRL is available
            if not ELEGANTRL_AVAILABLE:
                raise ImportError("ElegantRL is required for the trading system")
            
            # Create agent
            self.agent = MultiTimeframePPOAgent(
                env=self.env,
                features_1h=len(self.features_1h),
                seq_len_1h=self.lookback_window_1h,
                features_4h=len(self.features_4h),
                seq_len_4h=self.lookback_window_4h,
                hidden_dim=self.hidden_dim,
                use_attention=self.use_attention,
                regime_detection=self.regime_detection
            )
            
            # Load trained model
            self.agent.load_model(self.model_path)
            
            logger.info("Environment and agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing environment and agent: {e}")
            raise
    
    def update_positions(self):
        """Update current positions from IBKR."""
        try:
            positions = self.ibkr.get_positions()
            self.current_positions = {}
            
            for position in positions:
                symbol = position['symbol']
                if symbol in self.ticker_list:
                    self.current_positions[symbol] = {
                        'position': position['position'],
                        'marketPrice': position['marketPrice'],
                        'marketValue': position['marketValue'],
                        'averageCost': position['averageCost'],
                        'unrealizedPNL': position['unrealizedPNL']
                    }
            
            logger.info(f"Current positions: {self.current_positions}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _load_trade_records(self):
        """Load trade records from file."""
        if os.path.exists(self.trade_records_path):
            try:
                with open(self.trade_records_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading trade records: {e}")
                return {'trades': []}
        else:
            return {'trades': []}
    
    def _save_trade_records(self):
        """Save trade records to file."""
        try:
            with open(self.trade_records_path, 'w') as f:
                json.dump(self.trade_records, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade records: {e}")
    
    def _record_trade(self, trade_info):
        """Record a trade in the trade records."""
        trade_info['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.trade_records['trades'].append(trade_info)
        self._save_trade_records()
    
    def check_market_hours(self):
        """Check if we're in market hours."""
        now = datetime.datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Convert to US Eastern time (approximate check)
        eastern_hour = (now.hour - 3) % 24  # Assuming GMT-3 for Eastern Time
        
        # Regular trading hours: 9:30 AM - 4:00 PM Eastern
        return 9 <= eastern_hour < 16 or (eastern_hour == 9 and now.minute >= 30)
    
    def trading_step(self):
        """Execute one step of the trading loop."""
        try:
            # Check if we're in market hours
            if not self.check_market_hours():
                logger.info("Outside market hours. Skipping trading step.")
                return
            
            # Check if day trading is allowed
            if not self.ibkr.can_day_trade():
                logger.warning("Day trading limit reached. Limited to position management.")
            
            # Get environment state
            state = self.env._get_observation()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from agent
            with torch.no_grad():
                # Select action
                if isinstance(self.agent, MultiTimeframePPOAgent):
                    # ElegantRL agent
                    action = self.agent.select_action(state_tensor)
                else:
                    # Fallback for other agent types
                    action = self.agent.act(state_tensor)
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Log action and results
            logger.info(f"Action: {action[0]}, Reward: {reward:.4f}")
            logger.info(f"Info: {info}")
            
            # Record trade if executed
            if info.get('trade_executed', False):
                trade_info = {
                    'ticker': self.ticker_list[0],  # For simplicity, using first ticker
                    'action': 'BUY' if info['position'] > 0 else 'SELL' if info['position'] < 0 else 'FLAT',
                    'position': info['position'],
                    'price': info.get('position_price', 0),
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance']
                }
                self._record_trade(trade_info)
                logger.info(f"Trade executed: {trade_info}")
            
            # Update positions after action
            self.update_positions()
            
        except Exception as e:
            logger.error(f"Error in trading step: {e}")
    
    def start_trading(self):
        """Start the trading loop."""
        self.is_running = True
        logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                # Record current time
                current_time = datetime.datetime.now()
                
                # Execute trading step
                self.trading_step()
                
                # Update last check time
                self.last_check_time = current_time
                
                # Wait until next check
                next_check_time = current_time + datetime.timedelta(seconds=self.check_interval)
                sleep_time = (next_check_time - datetime.datetime.now()).total_seconds()
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time:.1f} seconds until next check")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop the trading loop and clean up."""
        logger.info("Stopping trading")
        self.is_running = False
        
        # Save final trade records
        self._save_trade_records()
        
        # Disconnect from IBKR
        if self.ibkr and self.ibkr.connected:
            self.ibkr.disconnect()
            logger.info("Disconnected from IBKR")


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


def main():
    """Main function to run the trading system."""
    parser = argparse.ArgumentParser(description='Interactive Brokers Live Trading System')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--ticker_list', type=str, nargs='+', required=True,
                        help='List of tickers to trade')
    
    # System parameters
    parser.add_argument('--max_position', type=float, default=0.5,
                        help='Maximum position size (0.5 = 50% of account)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate')
    parser.add_argument('--check_interval', type=int, default=60,
                        help='Interval between trading checks (seconds)')
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
                        help='Use paper trading')
    
    # Neural network parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for neural networks')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--no_regime', action='store_true',
                        help='Disable regime detection')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model (cpu or cuda)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get default features
    features_1h, features_4h = create_default_features()
    
    # Create and start trading system
    try:
        trading_system = IBKRTradingSystem(
            model_path=args.model_path,
            ticker_list=args.ticker_list,
            features_1h=features_1h,
            features_4h=features_4h,
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
            device=args.device
        )
        
        trading_system.start_trading()
        
    except Exception as e:
        logger.error(f"Error starting trading system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()