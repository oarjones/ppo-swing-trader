"""
Multi-Timeframe Trading Environment for FinRL

This module provides a trading environment that integrates multiple timeframes
(1h and 4h) and handles both training and live trading with Interactive Brokers.
It extends the FinRL BaseEnv to be compatible with the FinRL ecosystem.
"""

import numpy as np
import pandas as pd
import gym
import matplotlib
import matplotlib.pyplot as plt
from gym import spaces
import datetime
import logging
from typing import List, Dict, Tuple, Any, Optional, Union

# Import IBKR Interface
from src.ibkr_interface import IBKRInterface

# Import FinRL components

from external.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from external.finrl.meta.preprocessor.preprocessors import data_split
parent_class = StockTradingEnv

# Import processor
from src.ibkr_processor import IBKRProcessor

logger = logging.getLogger(__name__)

class FinRLMultiTimeframeEnv(parent_class):
    """
    A trading environment that integrates multiple timeframes.
    Extends FinRL StockTradingEnv to support swing trading strategies.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                df_1h: pd.DataFrame = None,
                df_4h: pd.DataFrame = None,
                ibkr_interface: IBKRInterface = None,
                ibkr_processor: IBKRProcessor = None,
                ticker_list: List[str] = None,
                lookback_window_1h: int = 20,
                lookback_window_4h: int = 10,
                initial_amount: float = 10000.0,
                commission_percentage: float = 0.001,
                reward_scaling: float = 1.0,
                max_position: int = 1,
                max_trades_per_day: int = 3,
                features_1h: List[str] = None,
                features_4h: List[str] = None,
                tech_indicator_list: List[str] = None,
                turbulence_threshold: Optional[float] = None,
                risk_indicator_col: str = 'turbulence',
                day_trade_multiplier: float = 0.5,
                live_trading: bool = False,
                **kwargs):
        """
        Initialize the multi-timeframe trading environment.
        
        Args:
            df_1h: DataFrame with 1h OHLCV and features data
            df_4h: DataFrame with 4h OHLCV and features data
            ibkr_interface: IBKRInterface instance for live trading
            ibkr_processor: IBKRProcessor instance for data processing
            ticker_list: List of ticker symbols
            lookback_window_1h: Lookback window for 1h data
            lookback_window_4h: Lookback window for 4h data
            initial_amount: Initial cash amount
            commission_percentage: Trading commission percentage
            reward_scaling: Scaling factor for reward
            max_position: Maximum position size (1 = 100% of portfolio)
            max_trades_per_day: Maximum trades per day (PDT rule)
            features_1h: List of feature columns for 1h data
            features_4h: List of feature columns for 4h data
            tech_indicator_list: List of technical indicators (for FinRL compatibility)
            turbulence_threshold: Threshold for market turbulence
            risk_indicator_col: Column to use as risk indicator
            day_trade_multiplier: Multiplier for day trade rewards
            live_trading: Whether to run in live trading mode
            **kwargs: Additional arguments for parent class
        """
        # Setup for live trading mode
        self.live_trading = live_trading
        self.ibkr = ibkr_interface
        self.ibkr_processor = ibkr_processor
        self.ticker_list = ticker_list or []
        
        # Initialize parent class if not in live trading mode
        if not live_trading and df_1h is not None:
            if tech_indicator_list is None and features_1h is not None:
                # Use features_1h as tech_indicator_list for compatibility
                tech_indicator_list = features_1h
                
            super().__init__(
                df=df_1h,  # Use 1h as base dataframe
                stock_dim=len(ticker_list) if ticker_list else len(df_1h['tic'].unique()),
                hmax=max_position,
                initial_amount=initial_amount,
                transaction_cost_pct=commission_percentage,
                reward_scaling=reward_scaling,
                state_space=len(tech_indicator_list) * lookback_window_1h + len(features_4h) * lookback_window_4h + 3,
                action_space=3,  # buy, hold, sell
                tech_indicator_list=tech_indicator_list,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **kwargs
            )
        else:
            # For live trading, we'll initialize differently
            # Just set basic attributes, will initialize spaces in setup_live_trading
            self.stock_dim = len(ticker_list) if ticker_list else 1
            self.hmax = max_position
            self.initial_amount = initial_amount
            self.transaction_cost_pct = commission_percentage
            self.reward_scaling = reward_scaling
            self.tech_indicator_list = tech_indicator_list or []
            self.turbulence_threshold = turbulence_threshold
            self.risk_indicator_col = risk_indicator_col
        
        # Store dataframes
        self.df_1h = df_1h
        self.df_4h = df_4h
        
        # Store multi-timeframe parameters
        self.lookback_window_1h = lookback_window_1h
        self.lookback_window_4h = lookback_window_4h
        self.features_1h = features_1h or []
        self.features_4h = features_4h or []
        
        # Trading constraints
        self.max_position = max_position
        self.max_trades_per_day = max_trades_per_day
        self.day_trade_multiplier = day_trade_multiplier
        
        # For tracking day trades
        self.trades_today = 0
        self.last_trade_day = None
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Additional state information
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.actions_memory = []
        
        # Track portfolio state
        self.portfolio_value_memory = [self.initial_amount]
        self.date_memory = [self._get_date(0)]
        self.asset_memory = [self.initial_amount]
        
        # For live trading
        if live_trading:
            self.setup_live_trading()
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        # Represented as a continuous space from -1 to 1
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # Observation space includes:
        # 1. 1h features (lookback_window_1h timesteps)
        # 2. 4h features (lookback_window_4h timesteps)
        # 3. Portfolio state (balance, position, position_price)
        
        # Calculate total observation dimension
        n_1h_features = len(self.features_1h) * self.lookback_window_1h
        n_4h_features = len(self.features_4h) * self.lookback_window_4h
        n_portfolio_features = 3  # balance, position, position_price
        
        total_dim = n_1h_features + n_4h_features + n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
    
    def setup_live_trading(self):
        """Setup environment for live trading."""
        if not self.ibkr or not self.ibkr_processor:
            raise ValueError("IBKRInterface and IBKRProcessor required for live trading")
            
        if not self.ibkr.connected:
            logger.info("Connecting to IBKR...")
            self.ibkr.connect()
            
        # Initialize portfolio state
        self.balance = self.ibkr_processor.get_account_balance()
        self.initial_amount = self.balance
        self.position = 0
        self.position_price = 0
        
        # Reset day trades count
        self.trades_today = 0
        self.last_trade_day = datetime.datetime.now().date()
        
        # Initialize stocks_owned
        self.stocks_owned = {}
        positions = self.ibkr_processor.get_positions()
        
        for ticker in self.ticker_list:
            self.stocks_owned[ticker] = 0
            
        if not positions.empty:
            for _, row in positions.iterrows():
                if row['tic'] in self.ticker_list:
                    self.stocks_owned[row['tic']] = row['amount']
        
        logger.info(f"Live trading setup complete. Balance: ${self.balance:.2f}")
    
    def _get_date(self, idx):
        """Get date for the current step."""
        if self.live_trading:
            return datetime.datetime.now()
        else:
            try:
                return self.df_1h.index[idx]
            except IndexError:
                return self.df_1h.index[-1]
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        if self.live_trading:
            # For live trading, we don't reset everything
            self.setup_live_trading()
            # But we do reset trading session variables
            self.cost = 0
            self.trades = 0
            self.actions_memory = []
            self.portfolio_value_memory = [self.balance]
            self.date_memory = [datetime.datetime.now()]
            self.asset_memory = [self.balance]
        else:
            # For backtesting, reset to initial state
            self.balance = self.initial_amount
            self.position = 0
            self.position_price = 0
            self.cost = 0
            self.trades = 0
            self.trades_today = 0
            self.last_trade_day = None
            
            # Reset step to start after lookback window
            self.current_step = self.lookback_window_1h
            
            # Reset episode variables
            self.actions_memory = []
            self.portfolio_value_memory = [self.initial_amount]
            self.date_memory = [self._get_date(0)]
            self.asset_memory = [self.initial_amount]
            
            # Increment episode counter
            self.episode += 1
        
        # Get initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Observation array
        """
        if self.live_trading:
            return self._get_live_observation()
        else:
            return self._get_backtest_observation()
    
    def _get_backtest_observation(self):
        """
        Get observation for backtesting.
        
        Returns:
            Observation array
        """
        # Ensure the step is valid
        if self.current_step >= len(self.df_1h.index):
            self.current_step = len(self.df_1h.index) - 1
            
        # Get current timestamp
        current_time = self.df_1h.index[self.current_step]
        
        # 1. Get 1h features
        start_idx_1h = max(0, self.current_step - self.lookback_window_1h)
        end_idx_1h = self.current_step
        
        # Extract 1h feature values
        obs_1h = []
        for i in range(start_idx_1h, end_idx_1h + 1):
            features = []
            for feat in self.features_1h:
                if feat in self.df_1h.columns:
                    features.append(self.df_1h.iloc[i][feat])
                else:
                    features.append(0)  # Default value if feature not found
            obs_1h.append(features)
            
        # If we don't have enough history, pad with the first value
        if len(obs_1h) < self.lookback_window_1h:
            padding = [obs_1h[0]] * (self.lookback_window_1h - len(obs_1h))
            obs_1h = padding + obs_1h
        
        # 2. Get 4h features
        # Find the 4h bar that contains or is before the current time
        idx_4h = self.df_4h.index.get_indexer([current_time], method='pad')[0]
        
        # Ensure we have a valid index
        if idx_4h < 0:
            idx_4h = 0
            
        start_idx_4h = max(0, idx_4h - self.lookback_window_4h + 1)
        end_idx_4h = idx_4h
        
        # Extract 4h feature values
        obs_4h = []
        for i in range(start_idx_4h, end_idx_4h + 1):
            features = []
            for feat in self.features_4h:
                if feat in self.df_4h.columns:
                    features.append(self.df_4h.iloc[i][feat])
                else:
                    features.append(0)  # Default value if feature not found
            obs_4h.append(features)
            
        # If we don't have enough history, pad with the first value
        if len(obs_4h) < self.lookback_window_4h:
            padding = [obs_4h[0]] * (self.lookback_window_4h - len(obs_4h)) if obs_4h else [[0] * len(self.features_4h)] * self.lookback_window_4h
            obs_4h = padding + obs_4h
        
        # 3. Portfolio state
        # Normalize portfolio values
        current_price = self.df_1h.iloc[self.current_step]['close']
        portfolio_state = [
            self.balance / self.initial_amount,  # Normalized balance
            self.position,  # Current position (-1, 0, 1)
            self.position_price / current_price if self.position != 0 else 0  # Normalized entry price
        ]
        
        # Combine all features
        obs_1h_flat = np.array(obs_1h).flatten()
        obs_4h_flat = np.array(obs_4h).flatten()
        portfolio_state = np.array(portfolio_state)
        
        # Create complete observation
        observation = np.concatenate([obs_1h_flat, obs_4h_flat, portfolio_state])
        
        return observation.astype(np.float32)
    
    def _get_live_observation(self):
        """
        Get observation for live trading.
        
        Returns:
            Observation array
        """
        # Get current data
        ticker_1h_data = {}
        ticker_4h_data = {}
        
        # Fetch latest data for all tickers
        for ticker in self.ticker_list:
            try:
                # Get 1h data
                df_1h = self.ibkr.get_historical_data(
                    symbol=ticker,
                    duration="2 W",  # Get 2 weeks of data to ensure we have enough history
                    bar_size="1 hour",
                    what_to_show="TRADES"
                )
                
                # Get 4h data
                df_4h = self.ibkr.get_historical_data(
                    symbol=ticker,
                    duration="2 M",  # Get 2 months of data to ensure we have enough history
                    bar_size="4 hours",
                    what_to_show="TRADES"
                )
                
                if not df_1h.empty:
                    ticker_1h_data[ticker] = df_1h
                if not df_4h.empty:
                    ticker_4h_data[ticker] = df_4h
                    
            except Exception as e:
                logger.error(f"Error fetching live data for {ticker}: {e}")
        
        # Process the data to generate features
        if self.ibkr_processor and ticker_1h_data and ticker_4h_data:
            # For simplicity, we'll just use the first ticker's data in this example
            # In a real multi-asset environment, you'd process all tickers
            ticker = self.ticker_list[0]
            
            if ticker in ticker_1h_data and ticker in ticker_4h_data:
                df_1h = ticker_1h_data[ticker]
                df_4h = ticker_4h_data[ticker]
                
                # Generate features - this would normally be done by your feature generator
                # Here we're just using basic OHLCV
                for df in [df_1h, df_4h]:
                    # Add some basic technical indicators
                    # Moving averages
                    df['ma5'] = df['close'].rolling(5).mean()
                    df['ma10'] = df['close'].rolling(10).mean()
                    df['ma20'] = df['close'].rolling(20).mean()
                    
                    # Volatility
                    df['volatility'] = df['close'].rolling(10).std()
                    
                    # Momentum
                    df['momentum'] = df['close'].pct_change(5)
                
                # Get latest values
                latest_1h = df_1h.iloc[-self.lookback_window_1h:]
                latest_4h = df_4h.iloc[-self.lookback_window_4h:]
                
                # Extract features
                obs_1h = []
                for i in range(len(latest_1h)):
                    features = []
                    for feat in self.features_1h:
                        if feat in latest_1h.columns:
                            features.append(latest_1h.iloc[i][feat])
                        else:
                            features.append(0)  # Default if feature not found
                    obs_1h.append(features)
                
                obs_4h = []
                for i in range(len(latest_4h)):
                    features = []
                    for feat in self.features_4h:
                        if feat in latest_4h.columns:
                            features.append(latest_4h.iloc[i][feat])
                        else:
                            features.append(0)  # Default if feature not found
                    obs_4h.append(features)
                
                # Pad if needed
                if len(obs_1h) < self.lookback_window_1h:
                    padding = [obs_1h[0]] * (self.lookback_window_1h - len(obs_1h)) if obs_1h else [[0] * len(self.features_1h)] * self.lookback_window_1h
                    obs_1h = padding + obs_1h
                
                if len(obs_4h) < self.lookback_window_4h:
                    padding = [obs_4h[0]] * (self.lookback_window_4h - len(obs_4h)) if obs_4h else [[0] * len(self.features_4h)] * self.lookback_window_4h
                    obs_4h = padding + obs_4h
                
                # Portfolio state
                current_price = df_1h.iloc[-1]['close']
                portfolio_state = [
                    self.balance / self.initial_amount,  # Normalized balance
                    self.position,  # Current position (-1, 0, 1)
                    self.position_price / current_price if self.position != 0 else 0  # Normalized entry price
                ]
                
                # Combine all features
                obs_1h_flat = np.array(obs_1h).flatten()
                obs_4h_flat = np.array(obs_4h).flatten()
                portfolio_state = np.array(portfolio_state)
                
                # Create complete observation
                observation = np.concatenate([obs_1h_flat, obs_4h_flat, portfolio_state])
                
                return observation.astype(np.float32)
        
        # If we couldn't get proper data, return zeros
        logger.warning("Returning zero observation due to data retrieval issues")
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _calculate_reward(self, action):
        """
        Calculate reward for the current step.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get current price and previous price
        if self.live_trading:
            ticker = self.ticker_list[0]  # For simplicity, just use first ticker
            market_data = self.ibkr.get_market_data(ticker)
            current_price = market_data.get('last', 0)
            # We don't have previous price in live trading, so use position price or estimate
            prev_price = self.position_price if self.position != 0 else current_price
        else:
            current_price = self.df_1h.iloc[self.current_step]['close']
            prev_price = self.df_1h.iloc[self.current_step - 1]['close'] if self.current_step > 0 else current_price
        
        # Calculate price change
        price_change = current_price / prev_price - 1
        
        # Basic reward is price change aligned with position
        reward = self.position * price_change * 100  # Scale price change for better gradients
        
        # Add trading cost penalty
        reward -= self.cost
        
        # Penalize excessive trading
        if self.trades > 0:
            reward -= 0.01 * self.trades
        
        # Bonus for holding aligned with trend
        # For simplicity, we'll define trend direction based on 1h and 4h MAs
        # In a real implementation, you'd use more sophisticated trend detection
        try:
            if not self.live_trading:
                ma5_1h = self.df_1h.iloc[self.current_step]['ma5'] if 'ma5' in self.df_1h.columns else None
                ma20_1h = self.df_1h.iloc[self.current_step]['ma20'] if 'ma20' in self.df_1h.columns else None
                
                if ma5_1h is not None and ma20_1h is not None:
                    trend_1h = 1 if ma5_1h > ma20_1h else -1 if ma5_1h < ma20_1h else 0
                    
                    # Bonus if position aligns with trend
                    if self.position * trend_1h > 0:
                        reward += 0.05
        except Exception as e:
            logger.warning(f"Error calculating trend reward: {e}")
        
        # Day trade multiplier
        if self.trades_today > 0:
            reward *= self.day_trade_multiplier
        
        # Apply reward scaling
        return reward * self.reward_scaling
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        if self.live_trading:
            return self._live_step(action)
        else:
            return self._backtest_step(action)
    
    def _backtest_step(self, action):
        """
        Take a step in backtest mode.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        # Save current values before step
        self.actions_memory.append(action)
        
        # Convert continuous action to discrete
        action_value = action[0]  # For single-asset case
        
        # Discretize action: -1 (sell), 0 (hold), 1 (buy)
        # Using thresholds: action < -0.33 -> sell, action > 0.33 -> buy, else hold
        target_position = 0
        if action_value < -0.33:
            target_position = -1
        elif action_value > 0.33:
            target_position = 1
        
        # Check for day trading rule
        current_date = self._get_date(self.current_step).date()
        if self.last_trade_day is None or current_date != self.last_trade_day:
            self.trades_today = 0
            self.last_trade_day = current_date
        
        # Store old values
        old_position = self.position
        old_portfolio_value = self._calculate_portfolio_value()
        
        # Get current price
        current_price = self.df_1h.iloc[self.current_step]['close']
        
        # Determine if trade is allowed (based on PDT rule)
        allow_trade = True
        if target_position != old_position:
            if self.trades_today >= self.max_trades_per_day:
                allow_trade = False
                logger.debug(f"Trade rejected: exceeds daily limit ({self.trades_today} trades already)")
        
        # Execute trade if allowed
        self.cost = 0
        if allow_trade and target_position != old_position:
            # Close existing position if any
            if old_position != 0:
                # Calculate P&L
                position_value = old_position * self.max_position * self.initial_amount
                pnl = position_value * (current_price / self.position_price - 1)
                self.balance += pnl
                
                # Apply commission
                commission = abs(position_value) * self.transaction_cost_pct
                self.balance -= commission
                self.cost += commission
                
                # Record trade
                self.trades += 1
                self.trades_today += 1
            
            # Open new position if not flat
            if target_position != 0:
                # Set new position
                self.position = target_position
                self.position_price = current_price
                
                # Calculate position value
                position_value = target_position * self.max_position * self.initial_amount
                
                # Apply commission
                commission = abs(position_value) * self.transaction_cost_pct
                self.balance -= commission
                self.cost += commission
                
                # Record trade
                if old_position == 0:  # Only count as new trade if coming from flat
                    self.trades += 1
                    self.trades_today += 1
            else:
                # Flat position
                self.position = 0
                self.position_price = 0
        
        # Calculate reward
        reward = self._calculate_reward(action_value)
        
        # Update portfolio value
        new_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value_memory.append(new_portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df_1h.index) - 1
        
        # Update date memory
        self.date_memory.append(self._get_date(self.current_step))
        
        # Additional info
        info = {
            'date': self._get_date(self.current_step),
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'cost': self.cost,
            'balance': self.balance,
            'trades_today': self.trades_today,
            'position_price': self.position_price
        }
        
        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _live_step(self, action):
        """
        Take a step in live trading mode.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        # Save action
        self.actions_memory.append(action)
        
        # Convert continuous action to discrete
        action_value = action[0]  # For single-asset case
        
        # Discretize action
        target_position = 0
        if action_value < -0.33:
            target_position = -1
        elif action_value > 0.33:
            target_position = 1
        
        # Check for day trading rule
        current_date = datetime.datetime.now().date()
        if self.last_trade_day is None or current_date != self.last_trade_day:
            self.trades_today = 0
            self.last_trade_day = current_date
        
        # Store old values
        old_position = self.position
        ticker = self.ticker_list[0]  # For simplicity, just use first ticker
        
        # Get current price
        market_data = self.ibkr.get_market_data(ticker)
        current_price = market_data.get('last', 0)
        
        # Determine if trade is allowed (based on PDT rule)
        allow_trade = self.ibkr.can_day_trade()
        
        # Execute trade if allowed
        trade_executed = False
        self.cost = 0
        
        if allow_trade and target_position != old_position:
            # Determine action
            if old_position == 0 and target_position == 1:
                # Buy
                order_size = self.max_position * self.balance / current_price
                result = self.ibkr.place_market_order(ticker, 'BUY', order_size)
                
                if result.get('status') == 'submitted':
                    self.position = 1
                    self.position_price = current_price
                    self.trades += 1
                    self.trades_today += 1
                    trade_executed = True
                    
                    # Apply commission
                    commission = order_size * current_price * self.transaction_cost_pct
                    self.balance -= commission
                    self.cost += commission
                
            elif old_position == 0 and target_position == -1:
                # Short sell
                order_size = self.max_position * self.balance / current_price
                result = self.ibkr.place_market_order(ticker, 'SELL', order_size)
                
                if result.get('status') == 'submitted':
                    self.position = -1
                    self.position_price = current_price
                    self.trades += 1
                    self.trades_today += 1
                    trade_executed = True
                    
                    # Apply commission
                    commission = order_size * current_price * self.transaction_cost_pct
                    self.balance -= commission
                    self.cost += commission
                
            elif old_position == 1 and target_position <= 0:
                # Sell
                order_size = self.stocks_owned.get(ticker, 0)
                if order_size > 0:
                    result = self.ibkr.place_market_order(ticker, 'SELL', order_size)
                    
                    if result.get('status') == 'submitted':
                        # Calculate P&L
                        pnl = order_size * (current_price - self.position_price)
                        self.balance += pnl
                        
                        self.position = 0
                        self.position_price = 0
                        self.trades += 1
                        self.trades_today += 1
                        trade_executed = True
                        
                        # Apply commission
                        commission = order_size * current_price * self.transaction_cost_pct
                        self.balance -= commission
                        self.cost += commission
                        
                        # If going short
                        if target_position == -1:
                            order_size = self.max_position * self.balance / current_price
                            result = self.ibkr.place_market_order(ticker, 'SELL', order_size)
                            
                            if result.get('status') == 'submitted':
                                self.position = -1
                                self.position_price = current_price
                                
                                # Apply commission
                                commission = order_size * current_price * self.transaction_cost_pct
                                self.balance -= commission
                                self.cost += commission
                
            elif old_position == -1 and target_position >= 0:
                # Cover short
                order_size = abs(self.stocks_owned.get(ticker, 0))
                if order_size > 0:
                    result = self.ibkr.place_market_order(ticker, 'BUY', order_size)
                    
                    if result.get('status') == 'submitted':
                        # Calculate P&L
                        pnl = order_size * (self.position_price - current_price)
                        self.balance += pnl
                        
                        self.position = 0
                        self.position_price = 0
                        self.trades += 1
                        self.trades_today += 1
                        trade_executed = True
                        
                        # Apply commission
                        commission = order_size * current_price * self.transaction_cost_pct
                        self.balance -= commission
                        self.cost += commission
                        
                        # If going long
                        if target_position == 1:
                            order_size = self.max_position * self.balance / current_price
                            result = self.ibkr.place_market_order(ticker, 'BUY', order_size)
                            
                            if result.get('status') == 'submitted':
                                self.position = 1
                                self.position_price = current_price
                                
                                # Apply commission
                                commission = order_size * current_price * self.transaction_cost_pct
                                self.balance -= commission
                                self.cost += commission
        
        # Calculate reward
        reward = self._calculate_reward(action_value)
        
        # Update portfolio value
        new_portfolio_value = self.balance
        if self.position != 0:
            # Add current position value
            position_size = abs(self.stocks_owned.get(ticker, 0))
            position_value = position_size * current_price
            if self.position > 0:
                new_portfolio_value += position_value
            else:
                new_portfolio_value -= position_value
        
        self.portfolio_value_memory.append(new_portfolio_value)
        
        # Update date memory
        self.date_memory.append(datetime.datetime.now())
        
        # In live trading, episode is never "done" automatically
        done = False
        
        # Additional info
        info = {
            'date': datetime.datetime.now(),
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'cost': self.cost,
            'balance': self.balance,
            'trades_today': self.trades_today,
            'trade_executed': trade_executed,
            'position_price': self.position_price
        }
        
        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _calculate_portfolio_value(self):
        """
        Calculate current portfolio value.
        
        Returns:
            Portfolio value
        """
        if self.live_trading:
            # For live trading, get real-time portfolio value
            return self.ibkr_processor.get_account_balance()
        else:
            # For backtesting
            value = self.balance
            
            if self.position != 0:
                current_price = self.df_1h.iloc[self.current_step]['close']
                position_value = self.position * self.max_position * self.initial_amount
                pnl = position_value * (current_price / self.position_price - 1)
                value += pnl
            
            return value
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            # Print current state
            if self.live_trading:
                current_time = datetime.datetime.now()
                print(f"Date: {current_time}")
                print(f"Portfolio Value: ${self.portfolio_value_memory[-1]:.2f}")
                print(f"Position: {self.position}")
                if self.position != 0:
                    print(f"Position Price: ${self.position_price:.2f}")
                print(f"Balance: ${self.balance:.2f}")
                print(f"Trades Today: {self.trades_today}/{self.max_trades_per_day}")
            else:
                current_time = self._get_date(self.current_step)
                current_price = self.df_1h.iloc[self.current_step]['close']
                print(f"Date: {current_time}")
                print(f"Price: ${current_price:.2f}")
                print(f"Portfolio Value: ${self.portfolio_value_memory[-1]:.2f}")
                print(f"Position: {self.position}")
                if self.position != 0:
                    print(f"Position Price: ${self.position_price:.2f}")
                print(f"Balance: ${self.balance:.2f}")
                print(f"Trades Today: {self.trades_today}/{self.max_trades_per_day}")
            
            print("-" * 50)
    
    def get_portfolio_history(self):
        """
        Get portfolio history as DataFrame.
        
        Returns:
            DataFrame with portfolio history
        """
        history = pd.DataFrame({
            'date': self.date_memory,
            'portfolio_value': self.portfolio_value_memory
        })
        
        # Calculate returns
        history['return'] = history['portfolio_value'].pct_change()
        history['cumulative_return'] = (history['portfolio_value'] / history['portfolio_value'].iloc[0]) - 1
        
        return history
    
    def get_trade_history(self):
        """
        Get trade history (simplified).
        
        Returns:
            DataFrame with trade history
        """
        # For a proper trade history, you'd need to track each trade individually
        # This is a simplified version
        history = pd.DataFrame({
            'date': self.date_memory[1:],  # Skip initial state
            'action': [a[0] for a in self.actions_memory]
        })
        
        # Identify actual trades (when position changes)
        trades = []
        positions = [0]  # Start with no position
        
        for i, action in enumerate(history['action']):
            # Discretize action
            target_position = 0
            if action < -0.33:
                target_position = -1
            elif action > 0.33:
                target_position = 1
                
            # If position changed, it's a trade
            if target_position != positions[-1]:
                trades.append(1)
            else:
                trades.append(0)
                
            positions.append(target_position)
        
        history['trade'] = trades
        history['position'] = positions[:-1]  # Remove last position which is beyond our data
        
        return history[history['trade'] == 1]  # Return only actual trades