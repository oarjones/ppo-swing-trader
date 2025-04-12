"""
Backtesting script for Multi-Timeframe PPO Agent

This script runs backtesting for the trained multi-timeframe PPO agent
and generates performance reports and visualizations.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from datetime import datetime

# Import our components
from src.ibkr_processor import IBKRProcessor
from src.finrl_multi_timeframe_env import FinRLMultiTimeframeEnv
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent

# Import FinRL components if available
try:
    from external.finrl.meta.preprocessor.preprocessors import FeatureEngineer
    #from external.finrl.plot import backtest_stats, backtest_plot
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    print("Warning: FinRL not found. Some visualization features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
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
            '4h': df_4h,
            'ticker': ticker
        })
    
    return processed_dfs

def calculate_performance_metrics(returns):
    """
    Calculate key performance metrics from returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dict with performance metrics
    """
    # Convert to numpy array for calculations
    returns_np = returns.dropna().values
    
    # Basic metrics
    total_return = (1 + returns_np).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns_np)) - 1 if len(returns_np) > 0 else 0
    
    # Risk metrics
    daily_volatility = returns_np.std()
    annual_volatility = daily_volatility * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Drawdown metrics
    cumulative_returns = (1 + returns_np).cumprod() - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / (1 + running_max)
    max_drawdown = drawdowns.min()
    
    # Additional metrics
    positive_days = (returns_np > 0).sum()
    negative_days = (returns_np < 0).sum()
    win_rate = positive_days / (positive_days + negative_days) if positive_days + negative_days > 0 else 0
    
    # Sortino ratio (downside risk only)
    negative_returns = returns_np[returns_np < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    calmar_ratio = -annual_return / max_drawdown if max_drawdown < 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'num_trades': positive_days + negative_days
    }

def run_backtest(env, agent, lookback_window_1h, initial_amount):
    """
    Run backtest on environment with agent.
    
    Args:
        env: Trading environment
        agent: Trained agent
        lookback_window_1h: Lookback window for 1h data
        initial_amount: Initial investment amount
        
    Returns:
        DataFrame with backtest results and actions
    """
    # Reset environment
    state = env.reset()
    done = False
    
    # Tracking variables
    actions_list = []
    timestamps = []
    portfolio_values = []
    positions = []
    prices = []
    
    # Start with lookback_window_1h to ensure enough data
    current_step = lookback_window_1h
    
    # Run until episode is done
    while not done:
        # Get action from agent
        action = agent.select_action(torch.FloatTensor(state).unsqueeze(0), deterministic=True)
        
        # Execute action
        next_state, reward, done, info = env.step(action.cpu().numpy())
        
        # Record data
        timestamps.append(info['date'])
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        actions_list.append(action.cpu().numpy()[0])
        
        # Record price if available
        if hasattr(env, 'df_1h') and env.current_step < len(env.df_1h):
            prices.append(env.df_1h.iloc[env.current_step]['close'])
        else:
            prices.append(0)
        
        # Update state
        state = next_state
        current_step += 1
    
    # Create results DataFrame
    results = pd.DataFrame({
        'timestamp': timestamps,
        'portfolio_value': portfolio_values,
        'position': positions,
        'action': actions_list,
        'price': prices
    })
    
    # Calculate returns
    results['return'] = results['portfolio_value'].pct_change()
    results['cumulative_return'] = (results['portfolio_value'] / initial_amount) - 1
    
    # Add benchmark - buy and hold strategy
    benchmark_returns = results['price'].pct_change()
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    results['benchmark_return'] = benchmark_returns
    results['benchmark_cumulative'] = benchmark_cumulative
    
    return results

def plot_backtest_results(results, ticker, output_dir):
    """
    Plot backtest results.
    
    Args:
        results: DataFrame with backtest results
        ticker: Ticker symbol
        output_dir: Output directory for plots
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot portfolio value vs benchmark
    plt.figure(figsize=(12, 6))
    plt.plot(results['timestamp'], results['cumulative_return'] * 100, label='Strategy', linewidth=2)
    
    if 'benchmark_cumulative' in results.columns:
        plt.plot(results['timestamp'], results['benchmark_cumulative'] * 100, label='Buy & Hold', linewidth=2, alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title(f'Backtest Results for {ticker}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{ticker}_cumulative_returns.png'))
    plt.close()
    
    # Plot positions over time
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Portfolio Value
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(results['timestamp'], results['portfolio_value'], color='blue', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'Portfolio Value and Positions - {ticker}')
    ax1.grid(True)
    
    # Plot 2: Positions and Price
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(results['timestamp'], results['price'], color='gray', alpha=0.5, label='Price')
    
    # Twin axis for positions
    ax3 = ax2.twinx()
    ax3.plot(results['timestamp'], results['position'], color='green', drawstyle='steps-post', linewidth=2, label='Position')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.2, 1.2)
    
    # Fill position areas
    for i in range(1, len(results)):
        if results['position'].iloc[i] > 0:
            ax3.fill_between([results['timestamp'].iloc[i-1], results['timestamp'].iloc[i]], 
                           [0, 0], [results['position'].iloc[i], results['position'].iloc[i]], 
                           color='green', alpha=0.3)
        elif results['position'].iloc[i] < 0:
            ax3.fill_between([results['timestamp'].iloc[i-1], results['timestamp'].iloc[i]], 
                           [0, 0], [results['position'].iloc[i], results['position'].iloc[i]], 
                           color='red', alpha=0.3)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True)
    
    # Add legend for both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{ticker}_positions.png'))
    plt.close()
    
    # Plot monthly returns heatmap
    if len(results) > 30:  # Only if enough data
        # Calculate monthly returns
        results['month'] = results['timestamp'].dt.strftime('%Y-%m')
        monthly_returns = results.groupby('month')['return'].sum().unstack().fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, cmap='RdYlGn', center=0, fmt='.1%')
        plt.title(f'Monthly Returns Heatmap - {ticker}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{ticker}_monthly_returns.png'))
        plt.close()

def generate_performance_report(ticker_results, output_dir):
    """
    Generate performance report across all tickers.
    
    Args:
        ticker_results: Dict with results for each ticker
        output_dir: Output directory for report
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    
    for ticker, results in ticker_results.items():
        # Calculate metrics
        metrics = calculate_performance_metrics(results['return'])
        
        # Add to summary
        summary_data.append({
            'ticker': ticker,
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'win_rate': metrics['win_rate'],
            'num_trades': metrics['num_trades']
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    # Generate summary plots
    if len(summary_df) > 1:  # Only if multiple tickers
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot returns comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='ticker', y='annual_return', data=summary_df)
        plt.title('Annual Return by Ticker')
        plt.ylabel('Annual Return')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'annual_returns.png'))
        plt.close()
        
        # Plot Sharpe ratio comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='ticker', y='sharpe_ratio', data=summary_df)
        plt.title('Sharpe Ratio by Ticker')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sharpe_ratios.png'))
        plt.close()
        
        # Plot maximum drawdown comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='ticker', y='max_drawdown', data=summary_df)
        plt.title('Maximum Drawdown by Ticker')
        plt.ylabel('Maximum Drawdown')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'max_drawdowns.png'))
        plt.close()
        
        # Plot risk vs return
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='annual_volatility', y='annual_return', size='sharpe_ratio', 
                      hue='ticker', data=summary_df, sizes=(50, 500))
        
        # Add ticker labels
        for i, row in summary_df.iterrows():
            plt.annotate(row['ticker'], 
                      (row['annual_volatility'], row['annual_return']),
                      xytext=(5, 5), textcoords='offset points')
        
        plt.title('Risk vs Return by Ticker')
        plt.xlabel('Annual Volatility (Risk)')
        plt.ylabel('Annual Return')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_vs_return.png'))
        plt.close()
    
    # Generate combined equity curve
    plt.figure(figsize=(12, 6))
    
    for ticker, results in ticker_results.items():
        # Plot equity curve
        plt.plot(results['timestamp'], results['cumulative_return'] * 100, label=ticker, linewidth=1.5)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Returns Across All Tickers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_equity_curve.png'))
    plt.close()
    
    # Return summary DataFrame
    return summary_df

def main():
    """Main function for backtesting."""
    parser = argparse.ArgumentParser(description='Backtest Multi-Timeframe PPO Agent')
    
    # Input parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing CSV files')
    parser.add_argument('--ticker_list', type=str, nargs='+', required=True,
                        help='List of tickers to backtest on')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for backtest data')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for backtest data')
    
    # Environment parameters
    parser.add_argument('--lookback_window_1h', type=int, default=20,
                        help='Lookback window for 1h data')
    parser.add_argument('--lookback_window_4h', type=int, default=10,
                        help='Lookback window for 4h data')
    parser.add_argument('--initial_amount', type=float, default=10000.0,
                        help='Initial amount for simulation')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate')
    parser.add_argument('--max_position', type=float, default=1.0,
                        help='Maximum position size')
    parser.add_argument('--max_trades_per_day', type=int, default=3,
                        help='Maximum trades per day (PDT rule)')
    
    # Agent parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for neural networks')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--no_regime', action='store_true',
                        help='Disable regime detection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/backtest',
                        help='Directory to save backtest results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get default features
    features_1h, features_4h = create_default_features()
    
    # Load data
    data = load_data(
        data_dir=args.data_dir,
        ticker_list=args.ticker_list,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Process data
    processed_data = process_data(data, features_1h, features_4h)
    
    if not processed_data:
        logger.error("No valid data found. Exiting.")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return
    
    # Run backtest for each ticker
    ticker_results = {}
    
    for data_item in processed_data:
        ticker = data_item['ticker']
        df_1h = data_item['1h']
        df_4h = data_item['4h']
        
        logger.info(f"Backtesting on {ticker}")
        
        # Create environment
        env = FinRLMultiTimeframeEnv(
            df_1h=df_1h,
            df_4h=df_4h,
            ticker_list=[ticker],
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
        
        # Create agent
        agent = MultiTimeframePPOAgent(
            env=env,
            features_1h=len(features_1h),
            seq_len_1h=args.lookback_window_1h,
            features_4h=len(features_4h),
            seq_len_4h=args.lookback_window_4h,
            hidden_dim=args.hidden_dim,
            use_attention=not args.no_attention,
            regime_detection=not args.no_regime
        )
        
        # Load model
        agent.load_model(args.model_path)
        
        # Move to device
        agent.to(args.device)
        
        # Run backtest
        results = run_backtest(env, agent, args.lookback_window_1h, args.initial_amount)
        
        # Plot results
        ticker_output_dir = os.path.join(args.output_dir, ticker)
        plot_backtest_results(results, ticker, ticker_output_dir)
        
        # Save results to CSV
        results.to_csv(os.path.join(ticker_output_dir, f'{ticker}_backtest_results.csv'), index=False)
        
        # Store results for summary
        ticker_results[ticker] = results
    
    # Generate performance report
    summary_df = generate_performance_report(ticker_results, args.output_dir)
    
    # Print summary
    print("\nBacktest Performance Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"))
    
    logger.info(f"Backtest completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()