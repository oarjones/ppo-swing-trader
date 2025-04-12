"""
Training script for Multi-Timeframe PPO Agent using ElegantRL

This script trains the multi-timeframe PPO agent using ElegantRL's training framework
and FinRL's environment. It supports both historical data and IBKR data.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt

# Import our components
from external.elegantrl.train.config import Arguments
from src.ibkr_processor import IBKRProcessor
from src.finrl_multi_timeframe_env import FinRLMultiTimeframeEnv
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent

# Import ElegantRL components if available

from external.elegantrl.train import config
from external.elegantrl.train.run import train_and_evaluate
ELEGANTRL_AVAILABLE = True


# Import FinRL components if available
from external.finrl.meta.preprocessor.preprocessors import FeatureEngineer
FINRL_AVAILABLE = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
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

def load_data(data_dir, ticker_list, start_date, end_date, include_indicators=True):
    """
    Load and process data from CSV files.
    
    Args:
        data_dir: Directory containing CSV files
        ticker_list: List of tickers to process
        start_date: Start date for filtering
        end_date: End date for filtering
        include_indicators: Whether to include technical indicators
        
    Returns:
        Dict mapping tickers to DataFrames for 1h and 4h data
    """
    data = {}
    
    for ticker in ticker_list:
        data[ticker] = {}
        
        # Try to load 1h data
        path_1h = os.path.join(data_dir, f"{ticker}_1h.csv")
        if os.path.exists(path_1h):
            df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
            
            # Filter by date range
            if start_date:
                df_1h = df_1h[df_1h.index >= pd.to_datetime(start_date, utc=True)]
            if end_date:
                df_1h = df_1h[df_1h.index <= pd.to_datetime(end_date, utc=True)]
            
            # Add technical indicators if needed
            if include_indicators and FINRL_AVAILABLE:
                fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list=[
                        'macd', 'rsi_14', 'cci_20', 'dx_20', 'adx',
                        'volatility_10', 'atr', 'obv'
                    ]
                )
                df_1h = fe.preprocess_data(df_1h)
            
            data[ticker]['1h'] = df_1h
            logger.info(f"Loaded {len(df_1h)} rows for {ticker}_1h")
        
        # Try to load 4h data
        path_4h = os.path.join(data_dir, f"{ticker}_4h.csv")
        if os.path.exists(path_4h):
            df_4h = pd.read_csv(path_4h, index_col=0, parse_dates=True)
            
            # Filter by date range
            if start_date:
                df_4h = df_4h[df_4h.index >= pd.to_datetime(start_date, utc=True)]
            if end_date:
                df_4h = df_4h[df_4h.index <= pd.to_datetime(end_date, utc=True)]
            
            # Add technical indicators if needed
            if include_indicators and FINRL_AVAILABLE:
                fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list=[
                        'macd', 'rsi_14', 'adx', 'volatility_10', 'atr'
                    ]
                )
                df_4h = fe.preprocess_data(df_4h)
            
            data[ticker]['4h'] = df_4h
            logger.info(f"Loaded {len(df_4h)} rows for {ticker}_4h")
    
    return data

def download_ibkr_data(ticker_list, ibkr_params, start_date, end_date):
    """
    Download data from IBKR.
    
    Args:
        ticker_list: List of tickers to download
        ibkr_params: IBKR connection parameters
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dict mapping tickers to DataFrames for 1h and 4h data
    """
    from src.ibkr_interface import IBKRInterface
    
    # Initialize IBKR
    ibkr = IBKRInterface(
        host=ibkr_params.get('host', '127.0.0.1'),
        port=ibkr_params.get('port', 4002),
        client_id=ibkr_params.get('client_id', 1),
        is_paper=ibkr_params.get('is_paper', True)
    )
    
    # Connect to IBKR
    if not ibkr.connect():
        raise ConnectionError("Failed to connect to Interactive Brokers")
    
    # Initialize processor
    processor = IBKRProcessor(ibkr_interface=ibkr)
    
    try:
        # Download data for each timeframe
        data = processor.download_multi_timeframe_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            timeframes=['1h', '4h']
        )
        
        return data
    finally:
        # Disconnect from IBKR
        ibkr.disconnect()

def process_data(data, features_1h, features_4h):
    """
    Process data for training.
    
    Args:
        data: Dict mapping tickers to DataFrames
        features_1h: List of features for 1h data
        features_4h: List of features for 4h data
        
    Returns:
        DataFrame ready for training
    """
    # Process each ticker
    processed_dfs = []
    
    for ticker, timeframes in data.items():
        if '1h' not in timeframes or '4h' not in timeframes:
            logger.warning(f"Skipping {ticker} due to missing timeframe data")
            continue
        
        df_1h = timeframes['1h']
        df_4h = timeframes['4h']
        
        # Ensure all required features exist in 1h
        for feature in features_1h:
            if feature not in df_1h.columns and feature not in ['open', 'high', 'low', 'close', 'volume']:
                # Calculate some basic features
                if feature.startswith('ma') and feature[2:].isdigit():
                    window = int(feature[2:])
                    df_1h[feature] = df_1h['close'].rolling(window=window).mean()
                elif feature.startswith('ema') and feature[3:].isdigit():
                    window = int(feature[3:])
                    df_1h[feature] = df_1h['close'].ewm(span=window, adjust=False).mean()
                elif feature == 'rsi_14':
                    delta = df_1h['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    df_1h[feature] = 100 - (100 / (1 + rs))
                elif feature.startswith('volatility_') and feature.split('_')[1].isdigit():
                    window = int(feature.split('_')[1])
                    df_1h[feature] = df_1h['close'].pct_change().rolling(window=window).std()
                else:
                    # Default to zeros for missing features
                    logger.warning(f"Feature {feature} not found in 1h data, setting to zeros")
                    df_1h[feature] = 0
        
        # Ensure all required features exist in 4h
        for feature in features_4h:
            if feature not in df_4h.columns and feature not in ['open', 'high', 'low', 'close', 'volume']:
                # Calculate some basic features (similar to 1h)
                if feature.startswith('ma') and feature[2:].isdigit():
                    window = int(feature[2:])
                    df_4h[feature] = df_4h['close'].rolling(window=window).mean()
                elif feature.startswith('ema') and feature[3:].isdigit():
                    window = int(feature[3:])
                    df_4h[feature] = df_4h['close'].ewm(span=window, adjust=False).mean()
                else:
                    # Default to zeros for missing features
                    logger.warning(f"Feature {feature} not found in 4h data, setting to zeros")
                    df_4h[feature] = 0
        
        # Add ticker column
        df_1h['tic'] = ticker
        df_4h['tic'] = ticker
        
        # Handle missing values
        df_1h = df_1h.fillna(method='ffill').fillna(method='bfill').dropna()
        df_4h = df_4h.fillna(method='ffill').fillna(method='bfill').dropna()
        
        # Store processed dataframes
        processed_dfs.append({
            '1h': df_1h,
            '4h': df_4h
        })
    
    return processed_dfs

def train_elegantrl(env, agent_params, training_params):
    """
    Train agent using ElegantRL.
    
    Args:
        env: Training environment
        agent_params: Agent parameters
        training_params: Training parameters
        
    Returns:
        Trained agent
    """
    if not ELEGANTRL_AVAILABLE:
        raise ImportError("ElegantRL is required for training")
    
    # Create ElegantRL Arguments object
    args = Arguments()
    
    # Set environment parameters
    args.env = env
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_step = int(training_params.get('max_step', 1000))
    args.if_discrete = False
    
    # Agent parameters
    args.net_dim = agent_params.get('hidden_dim', 128)
    args.learning_rate = agent_params.get('learning_rate', 3e-4)
    args.gamma = agent_params.get('gamma', 0.99)
    args.lambda_gae = agent_params.get('lambda_gae', 0.95)
    args.repeat_times = agent_params.get('ppo_epochs', 10)
    args.batch_size = agent_params.get('batch_size', 64)
    args.target_step = agent_params.get('target_step', 2048)
    args.clip_ratio = agent_params.get('clip_ratio', 0.2)
    args.entropy_coef = agent_params.get('entropy_coef', 0.01)
    
    # Training parameters
    args.eval_gap = training_params.get('eval_gap', 500)
    args.save_gap = training_params.get('save_gap', 10)
    args.break_step = int(training_params.get('break_step', 2e5))
    args.if_allow_break = training_params.get('if_allow_break', True)
    
    # Set device
    args.device = training_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set cwd and save_title
    args.cwd = training_params.get('cwd', 'models/MultiTimeframePPO')
    args.env_name = training_params.get('env_name', 'MultiTimeframeTradingEnv')
    
    # Create agent
    agent = MultiTimeframePPOAgent(
        env=env,
        features_1h=len(agent_params.get('features_1h', [])),
        seq_len_1h=agent_params.get('lookback_window_1h', 20),
        features_4h=len(agent_params.get('features_4h', [])),
        seq_len_4h=agent_params.get('lookback_window_4h', 10),
        hidden_dim=agent_params.get('hidden_dim', 128),
        use_attention=agent_params.get('use_attention', True),
        regime_detection=agent_params.get('regime_detection', True),
        learning_rate=agent_params.get('learning_rate', 3e-4)
    )
    
    # Train agent
    logger.info("Starting training with ElegantRL")
    # if training_params.get('use_mp', False):
    #     # Multi-processing training
    #     train_and_evaluate_mp(args, [agent])
    # else:
    train_and_evaluate(args, agent)
    
    return agent

def train_directly(env, agent_params, training_params):
    """
    Train agent directly without using ElegantRL's training framework.
    
    Args:
        env: Training environment
        agent_params: Agent parameters
        training_params: Training parameters
        
    Returns:
        Trained agent
    """
    # Create agent
    agent = MultiTimeframePPOAgent(
        env=env,
        features_1h=len(agent_params.get('features_1h', [])),
        seq_len_1h=agent_params.get('lookback_window_1h', 20),
        features_4h=len(agent_params.get('features_4h', [])),
        seq_len_4h=agent_params.get('lookback_window_4h', 10),
        hidden_dim=agent_params.get('hidden_dim', 128),
        use_attention=agent_params.get('use_attention', True),
        regime_detection=agent_params.get('regime_detection', True)
    )
    
    # Training parameters
    max_epochs = training_params.get('epochs', 100)
    update_freq = training_params.get('update_freq', 2048)
    batch_size = training_params.get('batch_size', 64)
    ppo_epochs = training_params.get('ppo_epochs', 10)
    
    # Training loop
    logger.info("Starting direct training")
    total_steps = 0
    returns = []
    
    for epoch in range(max_epochs):
        state = env.reset()
        episode_return = 0
        done = False
        
        # Collect experience
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_log_probs = []
        
        for step in range(update_freq):
            # Get action
            action, log_prob = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
            
            # Execute action
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
            
            # Record experience
            batch_states.append(state)
            batch_actions.append(action.detach().cpu().numpy())
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_log_probs.append(log_prob.detach().cpu().numpy())
            
            episode_return += reward
            total_steps += 1
            
            # Move to next state
            state = next_state
            
            if done:
                state = env.reset()
                returns.append(episode_return)
                episode_return = 0
        
        # Update agent
        agent.update(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            dones=batch_dones,
            log_probs=batch_log_probs,
            epochs=ppo_epochs,
            batch_size=batch_size
        )
        
        # Log progress
        if epoch % 10 == 0:
            avg_return = sum(returns[-10:]) / max(len(returns[-10:]), 1)
            logger.info(f"Epoch {epoch}, Avg Return: {avg_return:.2f}, Total Steps: {total_steps}")
    
    return agent

def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description='Train Multi-Timeframe PPO Agent')
    
    # Data parameters
    parser.add_argument('--data_source', type=str, choices=['csv', 'ibkr'], default='ibkr', help='Source of data (csv or ibkr)')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directory containing CSV files (for csv source)')
    parser.add_argument('--ticker_list', type=str, nargs='+', default=['BAC', 'INTC', 'PFE'], required=False, help='List of tickers to train on')
    parser.add_argument('--start_date', type=str, default='2021-01-01', help='Start date for training data')
    parser.add_argument('--end_date', type=str, default='2025-01-01',help='End date for training data')
    
    # IBKR parameters (if using IBKR data source)
    parser.add_argument('--ibkr_host', type=str, default='127.0.0.1', help='IBKR host address')
    parser.add_argument('--ibkr_port', type=int, default=4002, help='IBKR port (4001 for Gateway, 4002 for paper)')
    parser.add_argument('--ibkr_client_id', type=int, default=1, help='IBKR client ID')
    parser.add_argument('--ibkr_paper', action='store_true', default=True, help='Use paper trading account for data')
    
    # Environment parameters
    parser.add_argument('--lookback_window_1h', type=int, default=20, help='Lookback window for 1h data')
    parser.add_argument('--lookback_window_4h', type=int, default=10, help='Lookback window for 4h data')
    parser.add_argument('--initial_amount', type=float, default=10000.0, help='Initial amount for simulation')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--max_position', type=float, default=1.0, help='Maximum position size')
    parser.add_argument('--max_trades_per_day', type=int, default=3, help='Maximum trades per day (PDT rule)')
    
    # Agent parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for neural networks')
    parser.add_argument('--no_attention', action='store_true', help='Disable attention mechanism')
    parser.add_argument('--no_regime', action='store_true', help='Disable regime detection')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--update_freq', type=int, default=2048, help='Frequency of updates (steps)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--use_elegantrl', action='store_true', default=True, help='Use ElegantRL training framework')
    parser.add_argument('--use_mp', action='store_true', help='Use multiprocessing for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu)')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--model_name', type=str, default='ppo_multi_timeframe', help='Name of the model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Get default features
    features_1h, features_4h = create_default_features()
    
    # Load or download data
    if args.data_source == 'csv':
        data = load_data(
            data_dir=args.data_dir,
            ticker_list=args.ticker_list,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:  # ibkr
        ibkr_params = {
            'host': args.ibkr_host,
            'port': args.ibkr_port,
            'client_id': args.ibkr_client_id,
            'is_paper': args.ibkr_paper
        }
        data = download_ibkr_data(
            ticker_list=args.ticker_list,
            ibkr_params=ibkr_params,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    # Process data
    processed_data = process_data(data, features_1h, features_4h)
    
    if not processed_data:
        logger.error("No valid data found. Exiting.")
        return
    
    # Create environment for training
    env = FinRLMultiTimeframeEnv(
        df_1h=processed_data[0]['1h'],  # First ticker's 1h data
        df_4h=processed_data[0]['4h'],  # First ticker's 4h data
        ticker_list=args.ticker_list,
        lookback_window_1h=args.lookback_window_1h,
        lookback_window_4h=args.lookback_window_4h,
        initial_amount=args.initial_amount,
        commission_percentage=args.commission,
        max_position=args.max_position,
        max_trades_per_day=args.max_trades_per_day,
        features_1h=features_1h,
        features_4h=features_4h,
        live_trading=False
    )
    
    # Set up agent parameters
    agent_params = {
        'features_1h': features_1h,
        'seq_len_1h': args.lookback_window_1h,
        'features_4h': features_4h,
        'seq_len_4h': args.lookback_window_4h,
        'hidden_dim': args.hidden_dim,
        'use_attention': not args.no_attention,
        'regime_detection': not args.no_regime,
        'learning_rate': args.learning_rate
    }
    
    # Set up training parameters
    training_params = {
        'epochs': args.epochs,
        'update_freq': args.update_freq,
        'batch_size': args.batch_size,
        'ppo_epochs': args.ppo_epochs,
        'device': args.device,
        'cwd': args.model_dir,
        'use_mp': args.use_mp
    }
    
    # Train agent
    start_time = time.time()
    
    if args.use_elegantrl and ELEGANTRL_AVAILABLE:
        # Train with ElegantRL
        agent = train_elegantrl(env, agent_params, training_params)
    else:
        # Train directly
        agent = train_directly(env, agent_params, training_params)
    
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = os.path.join(args.model_dir, f"{args.model_name}.pt")
    agent.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()