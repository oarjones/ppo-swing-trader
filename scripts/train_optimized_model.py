#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento de modelo PPO Multi-Timeframe con parámetros optimizados.

Este script carga los hiperparámetros optimizados y entrena el modelo
completo con la configuración óptima.
"""

import os
import json
import argparse
import logging
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Importar componentes de nuestro sistema
from src.finrl_multi_timeframe_env import FinRLMultiTimeframeEnv
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent
from external.elegantrl.train.config import Arguments
from external.elegantrl.train.run import train_and_evaluate

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_optimized_params(params_file: str) -> Dict:
    """
    Cargar parámetros optimizados desde un archivo JSON.
    
    Args:
        params_file: Ruta al archivo JSON con los parámetros
        
    Returns:
        Dict con los parámetros optimizados
    """
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
        
        # Si el archivo contiene un objeto con 'params', extraer solo los parámetros
        if isinstance(data, dict) and 'params' in data:
            return data['params']
        return data
    except Exception as e:
        logger.error(f"Error cargando parámetros optimizados: {e}")
        return {}

def load_data(data_dir: str, ticker_list: List[str], start_date: str, end_date: str) -> Dict:
    """
    Cargar datos para entrenamiento.
    
    Args:
        data_dir: Directorio con archivos CSV
        ticker_list: Lista de tickers a utilizar
        start_date: Fecha de inicio
        end_date: Fecha de fin
        
    Returns:
        Dict con DataFrames para datos 1h y 4h
    """
    data = {}
    
    for ticker in ticker_list:
        data[ticker] = {}
        
        # Cargar datos 1h
        path_1h = os.path.join(data_dir, f"{ticker}_1h.csv")
        if os.path.exists(path_1h):
            df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
            
            # Filtrar por rango de fechas
            if start_date:
                df_1h = df_1h[df_1h.index >= pd.to_datetime(start_date, utc=True)]
            if end_date:
                df_1h = df_1h[df_1h.index <= pd.to_datetime(end_date, utc=True)]
                
            # Asegurar que tenemos la columna 'tic'
            if 'tic' not in df_1h.columns:
                df_1h['tic'] = ticker
                
            # Manejar valores faltantes
            df_1h = df_1h.fillna(method='ffill').fillna(method='bfill').dropna()
            
            data[ticker]['1h'] = df_1h
            logger.info(f"Cargados {len(df_1h)} registros para {ticker}_1h")
        else:
            logger.warning(f"No se encontró el archivo {path_1h}")
        
        # Cargar datos 4h
        path_4h = os.path.join(data_dir, f"{ticker}_4h.csv")
        if os.path.exists(path_4h):
            df_4h = pd.read_csv(path_4h, index_col=0, parse_dates=True)
            
            # Filtrar por rango de fechas
            if start_date:
                df_4h = df_4h[df_4h.index >= pd.to_datetime(start_date, utc=True)]
            if end_date:
                df_4h = df_4h[df_4h.index <= pd.to_datetime(end_date, utc=True)]
                
            # Asegurar que tenemos la columna 'tic'
            if 'tic' not in df_4h.columns:
                df_4h['tic'] = ticker
                
            # Manejar valores faltantes
            df_4h = df_4h.fillna(method='ffill').fillna(method='bfill').dropna()
            
            data[ticker]['4h'] = df_4h
            logger.info(f"Cargados {len(df_4h)} registros para {ticker}_4h")
        else:
            logger.warning(f"No se encontró el archivo {path_4h}")
    
    return data

def create_features_from_params(params: Dict) -> Tuple[List[str], List[str]]:
    """
    Crear listas de features basadas en parámetros optimizados.
    
    Args:
        params: Diccionario de parámetros optimizados
        
    Returns:
        Tupla de (features_1h, features_4h)
    """
    # Si los parámetros contienen listas de features, usarlas
    if 'features_1h' in params and 'features_4h' in params:
        return params['features_1h'], params['features_4h']
    
    # Si no, crear listas predeterminadas
    features_1h = [
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma10', 'ma20', 'ma50',
        'ema5', 'ema10', 'ema20', 'ema50',
        'return_5', 'return_10', 'return_20',
        'rsi_14', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_lower', 'bb_mid', 'bb_position',
        'adx', 'atr', 'atr_percent',
        'volatility_10', 'realized_vol_10',
        'trend_direction',
        'momentum_5', 'momentum_10', 'momentum_20'
    ]
    
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

def prepare_environment(data: Dict, ticker: str, params: Dict, features_1h: List[str], features_4h: List[str]) -> FinRLMultiTimeframeEnv:
    """
    Preparar entorno de entrenamiento.
    
    Args:
        data: Datos cargados
        ticker: Ticker a usar
        params: Parámetros optimizados
        features_1h: Lista de features 1h
        features_4h: Lista de features 4h
        
    Returns:
        Entorno FinRLMultiTimeframeEnv configurado
    """
    if ticker not in data or '1h' not in data[ticker] or '4h' not in data[ticker]:
        raise ValueError(f"Datos incompletos para {ticker}")
    
    # Extraer parámetros relevantes
    lookback_window_1h = params.get('lookback_window_1h', 20)
    lookback_window_4h = params.get('lookback_window_4h', 10)
    initial_amount = params.get('initial_amount', 10000.0)
    commission = params.get('commission', 0.001)
    max_position = params.get('max_position', 1.0)
    max_trades_per_day = params.get('max_trades_per_day', 5)
    
    # Crear entorno
    env = FinRLMultiTimeframeEnv(
        df_1h=data[ticker]['1h'],
        df_4h=data[ticker]['4h'],
        ticker_list=[ticker],
        lookback_window_1h=lookback_window_1h,
        lookback_window_4h=lookback_window_4h,
        initial_amount=initial_amount,
        commission_percentage=commission,
        max_position=max_position,
        max_trades_per_day=max_trades_per_day,
        features_1h=features_1h,
        features_4h=features_4h,
        live_trading=False
    )
    
    return env

def train_model(env, params: Dict, model_dir: str) -> str:
    """
    Entrenar modelo con parámetros optimizados.
    
    Args:
        env: Entorno de entrenamiento
        params: Parámetros optimizados
        model_dir: Directorio para guardar el modelo
        
    Returns:
        Ruta al modelo entrenado
    """
    # Crear directorio para modelos si no existe
    os.makedirs(model_dir, exist_ok=True)
    
    # Extraer parámetros relevantes para el agente
    hidden_dim = params.get('hidden_dim', 128)
    learning_rate = params.get('learning_rate', 3e-4)
    use_attention = params.get('use_attention', True)
    regime_detection = params.get('regime_detection', True)
    
    # Crear agente
    agent = MultiTimeframePPOAgent(
        env=env,
        features_1h=len(env.features_1h),
        seq_len_1h=env.lookback_window_1h,
        features_4h=len(env.features_4h),
        seq_len_4h=env.lookback_window_4h,
        hidden_dim=hidden_dim,
        use_attention=use_attention,
        regime_detection=regime_detection,
        learning_rate=learning_rate
    )
    
    # Configurar argumentos para ElegantRL
    args = Arguments()
    args.env = env
    args.agent = type(agent)  # Usar la clase del agente
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_step = env.max_step if hasattr(env, 'max_step') else 1000
    args.if_discrete = False
    
    # Parámetros del agente
    args.net_dim = hidden_dim
    args.learning_rate = learning_rate
    args.gamma = params.get('gamma', 0.99)
    args.lambda_gae = params.get('lambda_gae', 0.95)
    args.repeat_times = params.get('ppo_epochs', 10)
    args.batch_size = params.get('batch_size', 64)
    args.target_step = params.get('target_step', 2048)
    args.clip_ratio = params.get('clip_ratio', 0.2)
    args.entropy_coef = params.get('entropy_coef', 0.01)
    
    # Parámetros de entrenamiento
    args.eval_gap = 1000
    args.eval_times = 20
    args.break_step = int(1e6)  # Número máximo de pasos
    args.if_allow_break = True
    
    # Usar CPU o GPU según disponibilidad
    args.device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar directorio para resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.cwd = os.path.join(model_dir, f"ppo_mt_{timestamp}")
    args.init_before_training()
    
    # Entrenar el modelo
    try:
        logger.info(f"Iniciando entrenamiento con parámetros: {params}")
        train_and_evaluate(args)
        
        # Guardar el modelo final
        model_path = os.path.join(args.cwd, "model_final.pt")
        agent.save_model(model_path)
        
        logger.info(f"Entrenamiento completo. Modelo guardado en {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        return None

def plot_training_results(cwd: str):
    """
    Graficar resultados del entrenamiento.
    
    Args:
        cwd: Directorio de trabajo con los resultados
    """
    # Verificar si existe el archivo recorder.npy
    recorder_path = os.path.join(cwd, "recorder.npy")
    if not os.path.exists(recorder_path):
        logger.warning(f"No se encontró el archivo de registro en {recorder_path}")
        return
    
    try:
        # Cargar datos del recorder
        recorder = np.load(recorder_path)
        
        # Extraer datos
        steps = recorder[:, 0]  # Pasos de entrenamiento
        returns = recorder[:, 1]  # Retornos
        if recorder.shape[1] > 2:
            critic_losses = recorder[:, 4]  # Pérdidas del crítico
            actor_losses = recorder[:, 5]  # Pérdidas del actor
        
        # Graficar retornos
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(steps, returns, label='Episode Return')
        plt.title('Training Returns')
        plt.xlabel('Steps')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        
        # Graficar pérdidas si están disponibles
        if recorder.shape[1] > 5:
            plt.subplot(2, 1, 2)
            plt.plot(steps, critic_losses, label='Critic Loss', color='blue')
            plt.plot(steps, actor_losses, label='Actor Loss', color='red')
            plt.title('Training Losses')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cwd, 'training_plot.png'))
        plt.close()
        
        logger.info(f"Gráficos de entrenamiento guardados en {cwd}")
    
    except Exception as e:
        logger.error(f"Error al graficar resultados: {e}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenamiento de PPO Multi-Timeframe con parámetros optimizados')
    
    # Parámetros de entrada
    parser.add_argument('--params_file', type=str, default='./optimization_results/best_params.json',
                       help='Ruta al archivo JSON con parámetros optimizados')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Directorio con datos de entrenamiento')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Ticker para entrenamiento')
    parser.add_argument('--start_date', type=str, default='2019-01-01',
                       help='Fecha de inicio para datos de entrenamiento')
    parser.add_argument('--end_date', type=str, default='2022-01-01',
                       help='Fecha de fin para datos de entrenamiento')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directorio para guardar modelos')
    
    args = parser.parse_args()
    
    # Cargar parámetros optimizados
    params = load_optimized_params(args.params_file)
    if not params:
        logger.error(f"No se pudieron cargar parámetros desde {args.params_file}")
        return
    
    logger.info(f"Parámetros cargados: {params}")
    
    # Cargar datos
    data = load_data(
        data_dir=args.data_dir,
        ticker_list=[args.ticker],
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not data or args.ticker not in data:
        logger.error(f"No se pudieron cargar datos para {args.ticker}")
        return
    
    # Crear listas de features
    features_1h, features_4h = create_features_from_params(params)
    
    # Preparar entorno
    try:
        env = prepare_environment(data, args.ticker, params, features_1h, features_4h)
        logger.info("Entorno preparado correctamente")
    except Exception as e:
        logger.error(f"Error preparando entorno: {e}")
        return
    
    # Entrenar modelo
    model_path = train_model(env, params, args.model_dir)
    
    if model_path:
        logger.info(f"Entrenamiento completado exitosamente. Modelo guardado en: {model_path}")
        
        # Graficar resultados
        plot_training_results(os.path.dirname(model_path))
    else:
        logger.error("Fallo en el entrenamiento del modelo")

if __name__ == "__main__":
    main()