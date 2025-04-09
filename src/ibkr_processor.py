"""
IBKR Data Processor for FinRL

This module provides a data processor that converts Interactive Brokers data
into the format expected by FinRL. It handles both historical and real-time data
for multiple timeframes.
"""

import pandas as pd
import numpy as np
import datetime
import logging
import os
from typing import List, Dict, Union, Optional, Tuple

# Import the IBKRInterface
from external.finrl.meta.data_processors.data_processor import DataProcessor
from ibkr_interface import IBKRInterface




logger = logging.getLogger(__name__)

class IBKRProcessor(DataProcessor):
    """
    Data processor for Interactive Brokers data.
    Extends the FinRL DataProcessor to handle IBKR data.
    """
    
    def __init__(self, 
                 data_source: str = "ibkr", 
                 ibkr_interface: Optional[IBKRInterface] = None,
                 timeframes: List[str] = None,
                 **kwargs):
        """
        Initialize the IBKR data processor.
        
        Args:
            data_source: Name of the data source (should be "ibkr")
            ibkr_interface: An initialized IBKRInterface instance
            timeframes: List of timeframes to process (e.g. ["1h", "4h"])
            **kwargs: Additional arguments for the base DataProcessor
        """
        super().__init__(data_source=data_source, **kwargs)
        
        # Initialize or use provided IBKRInterface
        self.ibkr = ibkr_interface if ibkr_interface is not None else IBKRInterface()
        
        # Set default timeframes if not provided
        self.timeframes = timeframes or ["1h", "4h"]
        
        # Map timeframes to IBKR bar size format
        self.timeframe_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "2h": "2 hours", 
            "4h": "4 hours",
            "1d": "1 day"
        }
        
        # Ensure we're connected to IBKR
        if not self.ibkr.connected:
            logger.info("IBKRProcessor: Not connected to IBKR, attempting to connect...")
            self.ibkr.connect()
    
    def download_data(self, 
                     ticker_list: List[str], 
                     start_date: str,
                     end_date: str,
                     timeframe: str = "1h",
                     save_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple tickers.
        
        Args:
            ticker_list: List of ticker symbols
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            timeframe: Timeframe for the data (e.g. "1h", "4h")
            save_path: Optional path to save data to
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        # Ensure the timeframe is supported
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.timeframe_map.keys())}")
        
        # Calculate duration based on start and end dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        delta = end_dt - start_dt
        
        # Convert to IBKR duration format based on the delta
        if delta.days > 365:
            duration = f"{delta.days // 365 + 1} Y"
        elif delta.days > 30:
            duration = f"{delta.days // 30 + 1} M"
        else:
            duration = f"{delta.days + 1} D"
        
        # Convert timeframe to IBKR bar size
        bar_size = self.timeframe_map[timeframe]
        
        # Download data for each ticker
        data_dict = {}
        for ticker in ticker_list:
            logger.info(f"Downloading {timeframe} data for {ticker} from {start_date} to {end_date}")
            
            try:
                # Get data from IBKR
                df = self.ibkr.get_historical_data(
                    symbol=ticker,
                    duration=duration,
                    bar_size=bar_size,
                    end_datetime=end_date
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue
                
                # Filter to the specified date range
                df = self._filter_date_range(df, start_date, end_date)
                
                # Process dataframe to match FinRL format
                df = self._process_dataframe(df, ticker)
                
                # Save data if path provided
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    filename = os.path.join(save_path, f"{ticker}_{timeframe}.csv")
                    df.to_csv(filename)
                    logger.info(f"Saved data for {ticker} to {filename}")
                
                data_dict[ticker] = df
                logger.info(f"Successfully downloaded {len(df)} records for {ticker}")
                
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {e}")
        
        return data_dict
    
    def download_multi_timeframe_data(self, 
                                     ticker_list: List[str], 
                                     start_date: str,
                                     end_date: str,
                                     timeframes: List[str] = None,
                                     save_path: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download data for multiple tickers across multiple timeframes.
        
        Args:
            ticker_list: List of ticker symbols
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            timeframes: List of timeframes (e.g. ["1h", "4h"])
            save_path: Optional path to save data to
            
        Returns:
            Nested dictionary mapping symbols to timeframes to DataFrames
        """
        # Use default timeframes if not specified
        timeframes = timeframes or self.timeframes
        
        # Download data for each ticker and timeframe
        result = {}
        for ticker in ticker_list:
            result[ticker] = {}
            for tf in timeframes:
                try:
                    data = self.download_data(
                        ticker_list=[ticker],
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=tf,
                        save_path=save_path
                    )
                    if ticker in data and not data[ticker].empty:
                        result[ticker][tf] = data[ticker]
                except Exception as e:
                    logger.error(f"Error downloading {tf} data for {ticker}: {e}")
        
        return result
    
    def get_real_time_data(self, ticker_list: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get real-time market data for multiple tickers.
        
        Args:
            ticker_list: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to DataFrames with real-time data
        """
        result = {}
        for ticker in ticker_list:
            try:
                # Get market data
                market_data = self.ibkr.get_market_data(ticker)
                
                if not market_data:
                    logger.warning(f"No real-time data returned for {ticker}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([market_data])
                
                # Process dataframe to match FinRL format
                if 'time' in df.columns:
                    df['date'] = pd.to_datetime(df['time'])
                    df.set_index('date', inplace=True)
                
                # Rename columns to match FinRL format
                rename_map = {
                    'last': 'close',
                    'volume': 'volume',
                    'bid': 'bid',
                    'ask': 'ask'
                }
                df.rename(columns=rename_map, inplace=True)
                
                result[ticker] = df
                
            except Exception as e:
                logger.error(f"Error getting real-time data for {ticker}: {e}")
        
        return result
    
    def _filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter DataFrame to specified date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            
        Returns:
            Filtered DataFrame
        """
        # Convert to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
                df.set_index('date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        # Ensure start and end dates are datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter the DataFrame
        return df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    def _process_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Process DataFrame to match FinRL format.
        
        Args:
            df: DataFrame to process
            ticker: Ticker symbol
            
        Returns:
            Processed DataFrame
        """
        # Make a copy of the DataFrame
        result = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(result.index, pd.DatetimeIndex):
            if 'datetime' in result.columns:
                result['date'] = pd.to_datetime(result['datetime'])
                result.set_index('date', inplace=True)
            elif 'date' in result.columns:
                result['date'] = pd.to_datetime(result['date'])
                result.set_index('date', inplace=True)
        
        # Ensure we have the necessary columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col.capitalize() in result.columns and col not in result.columns:
                # Convert capitalized column names
                result[col] = result[col.capitalize()]
        
        # Rename columns to OHLCV if needed
        rename_map = {}
        for old, new in [('Open', 'open'), ('High', 'high'), ('Low', 'low'), 
                         ('Close', 'close'), ('Volume', 'volume')]:
            if old in result.columns and new not in result.columns:
                rename_map[old] = new
        
        if rename_map:
            result.rename(columns=rename_map, inplace=True)
        
        # Add ticker symbol column
        result['tic'] = ticker
        
        # Handle missing values
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        # Sort by date
        result = result.sort_index()
        
        return result
    
    def prepare_finrl_format(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data in the format expected by FinRL.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            
        Returns:
            DataFrame in FinRL format (combined data for all symbols)
        """
        # Combine all DataFrames
        df_list = []
        for tic, df in data_dict.items():
            df_with_tic = df.copy()
            if 'tic' not in df_with_tic.columns:
                df_with_tic['tic'] = tic
            df_list.append(df_with_tic)
        
        if not df_list:
            return pd.DataFrame()
        
        # Concatenate all DataFrames
        combined_df = pd.concat(df_list)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'tic']
        for col in required_cols:
            if col not in combined_df.columns:
                logger.warning(f"Required column {col} missing from data")
        
        # Sort by ticker and date
        combined_df = combined_df.sort_values(['tic', combined_df.index])
        
        return combined_df
    
    def get_account_balance(self) -> float:
        """
        Get the current account balance.
        
        Returns:
            Account balance
        """
        account_summary = self.ibkr.get_account_summary()
        return account_summary.get('NetLiquidation', 0.0)
    
    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions.
        
        Returns:
            DataFrame with current positions
        """
        positions = self.ibkr.get_positions()
        
        if not positions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Rename columns for consistency
        rename_map = {
            'symbol': 'tic',
            'position': 'amount',
            'marketPrice': 'price',
            'marketValue': 'value',
            'averageCost': 'cost_basis'
        }
        df.rename(columns=rename_map, inplace=True)
        
        return df
    
    def get_trading_bars_per_day(self, timeframe: str) -> int:
        """
        Get the number of trading bars per day for a given timeframe.
        
        Args:
            timeframe: Timeframe string (e.g. "1h", "4h")
            
        Returns:
            Number of bars per trading day
        """
        # Standard trading day is 6.5 hours (9:30 AM - 4:00 PM ET)
        trading_minutes = 6.5 * 60
        
        # Calculate bars based on timeframe
        if timeframe == "1m":
            return int(trading_minutes)
        elif timeframe == "5m":
            return int(trading_minutes / 5)
        elif timeframe == "15m":
            return int(trading_minutes / 15)
        elif timeframe == "30m":
            return int(trading_minutes / 30)
        elif timeframe == "1h":
            return int(trading_minutes / 60)
        elif timeframe == "2h":
            return int(trading_minutes / 120)
        elif timeframe == "4h":
            return int(trading_minutes / 240)
        elif timeframe == "1d":
            return 1
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def can_day_trade(self) -> bool:
        """
        Check if day trading is allowed based on PDT rule.
        
        Returns:
            Boolean indicating if day trading is allowed
        """
        return self.ibkr.can_day_trade()
    
    def get_day_trades_remaining(self) -> int:
        """
        Get number of day trades remaining before hitting PDT rule limit.
        
        Returns:
            Number of day trades remaining
        """
        # Max day trades allowed is typically 3 in a 5-day rolling period for accounts under $25k
        return max(0, self.ibkr.max_day_trades - self.ibkr.day_trades_count)