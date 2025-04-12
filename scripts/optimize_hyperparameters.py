#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimización de hiperparámetros para PPO Swing Trader
usando Optuna y ElegantRL.

Este script automatiza la búsqueda de hiperparámetros óptimos
para el modelo MultiTimeframePPO.
"""

import os
import json
import optuna
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
from typing import Dict, List, Any

# Importamos los componentes necesarios de nuestro proyecto
from src.finrl_multi_timeframe_env import FinRLMultiTimeframeEnv
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent
from external.elegantrl.train.config import Arguments

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_default_features():
    """Crear listas de features predeterminadas."""
    # 1-hour features (subset para optimización más rápida)
    features_1h = [
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma20', 'ma50',
        'rsi_14', 'macd',
        'volatility_10',
        'trend_direction',
        'momentum_5'
    ]
    
    # 4-hour features
    features_4h = [
        'open', 'high', 'low', 'close', 'volume',
        'ma5', 'ma20', 'ma50',
        'rsi_14', 'macd',
        'volatility_10',
        'trend_direction',
        'momentum_5'
    ]
    
    return features_1h, features_4h

def load_data(data_dir: str, ticker: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Cargar datos para optimización.
    
    Args:
        data_dir: Directorio con archivos CSV
        ticker: Ticker a utilizar
        start_date: Fecha de inicio
        end_date: Fecha de fin
        
    Returns:
        Dict con DataFrames para datos 1h y 4h
    """
    data = {}
    
    # Cargar datos 1h
    path_1h = os.path.join(data_dir, f"{ticker}_1h.csv")
    if os.path.exists(path_1h):
        df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
        
        # Filtrar por rango de fechas
        if start_date:
            df_1h = df_1h[df_1h.index >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df_1h = df_1h[df_1h.index <= pd.to_datetime(end_date, utc=True)]
            
        # Asegurar que las columnas requeridas existen
        for col in ['open', 'high', 'low', 'close', 'volume', 'tic']:
            if col not in df_1h.columns:
                if col == 'tic':
                    df_1h['tic'] = ticker
                else:
                    raise ValueError(f"Columna {col} no encontrada en datos 1h")
        
        # Calcular indicadores técnicos básicos si no existen
        if 'ma5' not in df_1h.columns:
            df_1h['ma5'] = df_1h['close'].rolling(5).mean()
        if 'ma20' not in df_1h.columns:
            df_1h['ma20'] = df_1h['close'].rolling(20).mean()
        if 'ma50' not in df_1h.columns:
            df_1h['ma50'] = df_1h['close'].rolling(50).mean()
        if 'rsi_14' not in df_1h.columns:
            delta = df_1h['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df_1h['rsi_14'] = 100 - (100 / (1 + rs))
        if 'volatility_10' not in df_1h.columns:
            df_1h['volatility_10'] = df_1h['close'].pct_change().rolling(10).std()
        if 'trend_direction' not in df_1h.columns:
            df_1h['trend_direction'] = np.where(df_1h['ma5'] > df_1h['ma20'], 1, -1)
        if 'momentum_5' not in df_1h.columns:
            df_1h['momentum_5'] = df_1h['close'].pct_change(5)
        if 'macd' not in df_1h.columns:
            ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()
            df_1h['macd'] = ema12 - ema26
            
        # Manejar valores faltantes
        df_1h = df_1h.fillna(method='ffill').fillna(method='bfill').dropna()
        
        data['1h'] = df_1h
        logger.info(f"Cargados {len(df_1h)} registros para {ticker}_1h")
    
    # Cargar datos 4h (proceso similar)
    path_4h = os.path.join(data_dir, f"{ticker}_4h.csv")
    if os.path.exists(path_4h):
        df_4h = pd.read_csv(path_4h, index_col=0, parse_dates=True)
        
        # Filtrar por rango de fechas
        if start_date:
            df_4h = df_4h[df_4h.index >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df_4h = df_4h[df_4h.index <= pd.to_datetime(end_date, utc=True)]
            
        # Asegurar que las columnas requeridas existen 
        for col in ['open', 'high', 'low', 'close', 'volume', 'tic']:
            if col not in df_4h.columns:
                if col == 'tic':
                    df_4h['tic'] = ticker
                else:
                    raise ValueError(f"Columna {col} no encontrada en datos 4h")
        
        # Calcular indicadores técnicos básicos si no existen (similar a 1h)
        if 'ma5' not in df_4h.columns:
            df_4h['ma5'] = df_4h['close'].rolling(5).mean()
        if 'ma20' not in df_4h.columns:
            df_4h['ma20'] = df_4h['close'].rolling(20).mean()
        if 'ma50' not in df_4h.columns:
            df_4h['ma50'] = df_4h['close'].rolling(50).mean()
        if 'rsi_14' not in df_4h.columns:
            delta = df_4h['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df_4h['rsi_14'] = 100 - (100 / (1 + rs))
        if 'volatility_10' not in df_4h.columns:
            df_4h['volatility_10'] = df_4h['close'].pct_change().rolling(10).std()
        if 'trend_direction' not in df_4h.columns:
            df_4h['trend_direction'] = np.where(df_4h['ma5'] > df_4h['ma20'], 1, -1)
        if 'momentum_5' not in df_4h.columns:
            df_4h['momentum_5'] = df_4h['close'].pct_change(5)
        if 'macd' not in df_4h.columns:
            ema12 = df_4h['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_4h['close'].ewm(span=26, adjust=False).mean()
            df_4h['macd'] = ema12 - ema26
            
        # Manejar valores faltantes
        df_4h = df_4h.fillna(method='ffill').fillna(method='bfill').dropna()
        
        data['4h'] = df_4h
        logger.info(f"Cargados {len(df_4h)} registros para {ticker}_4h")
    
    return data

def train_and_evaluate(env, agent_config: Dict[str, Any], train_config: Dict[str, Any]) -> float:
    """
    Entrenar y evaluar el agente con una configuración específica.
    
    Args:
        env: Entorno de trading
        agent_config: Configuración del agente
        train_config: Configuración del entrenamiento
        
    Returns:
        float: Métrica de rendimiento (Sharpe ratio)
    """
    # Crear agente con la configuración proporcionada
    agent = MultiTimeframePPOAgent(
        env=env,
        features_1h=len(agent_config['features_1h']),
        seq_len_1h=agent_config['lookback_window_1h'],
        features_4h=len(agent_config['features_4h']),
        seq_len_4h=agent_config['lookback_window_4h'],
        hidden_dim=agent_config['hidden_dim'],
        use_attention=agent_config['use_attention'],
        regime_detection=agent_config['regime_detection'],
        learning_rate=agent_config['learning_rate']
    )
    
    # Configurar argumentos para ElegantRL
    args = Arguments()
    args.env = env
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_step = train_config.get('max_step', 1000)
    args.if_discrete = False
    
    # Parámetros del agente
    args.net_dim = agent_config['hidden_dim']
    args.learning_rate = agent_config['learning_rate']
    args.gamma = agent_config.get('gamma', 0.99)
    args.lambda_gae = agent_config.get('lambda_gae', 0.95)
    args.repeat_times = agent_config.get('ppo_epochs', 10)
    args.batch_size = agent_config.get('batch_size', 64)
    args.target_step = agent_config.get('target_step', 2048)
    args.clip_ratio = agent_config.get('clip_ratio', 0.2)
    args.entropy_coef = agent_config.get('entropy_coef', 0.01)
    
    # Parámetros de entrenamiento
    args.eval_gap = train_config.get('eval_gap', 500)
    args.if_allow_break = False
    
    # Usar CPU o GPU según disponibilidad
    args.device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.cwd = f"./optimization_results/trial_{timestamp}"
    os.makedirs(args.cwd, exist_ok=True)
    
    # Entrenar agente
    try:
        if hasattr(agent, 'prepare_for_training'):
            agent.prepare_for_training(args)
        
        # Entrenamiento manual simplificado para optimización
        total_steps = 0
        returns = []
        
        for epoch in range(train_config.get('epochs', 100)):
            state = env.reset()
            episode_return = 0
            done = False
            
            # Recopilar experiencia
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            batch_log_probs = []
            
            for step in range(args.target_step):
                # Obtener acción
                action, log_prob = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
                
                # Registrar experiencia
                batch_states.append(state)
                batch_actions.append(action.detach().cpu().numpy())
                batch_rewards.append(reward)
                batch_dones.append(done)
                batch_log_probs.append(log_prob.detach().cpu().numpy())
                
                episode_return += reward
                total_steps += 1
                
                # Pasar al siguiente estado
                state = next_state
                
                if done:
                    state, _ = env.reset()
                    returns.append(episode_return)
                    episode_return = 0
                    break
            
            # Actualizar agente
            loss = agent.update(
                states=batch_states,
                actions=batch_actions,
                rewards=batch_rewards,
                dones=batch_dones,
                log_probs=batch_log_probs,
                epochs=args.repeat_times,
                batch_size=args.batch_size
            )
            
            if epoch % 10 == 0:
                avg_return = sum(returns[-10:]) / max(len(returns[-10:]), 1)
                logger.info(f"Epoch {epoch}, Avg Return: {avg_return:.2f}, Total Steps: {total_steps}")
        
        # Evaluar rendimiento
        returns_array = np.array(returns)
        if len(returns_array) > 0:
            mean_return = returns_array.mean()
            std_return = returns_array.std() if len(returns_array) > 1 else 1.0
            sharpe = mean_return / std_return if std_return != 0 else 0.0
            
            # Guardar resultados
            with open(f"{args.cwd}/results.json", 'w') as f:
                json.dump({
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'sharpe': float(sharpe),
                    'config': agent_config
                }, f, indent=2)
            
            logger.info(f"Entrenamiento completo - Retorno Medio: {mean_return:.2f}, Sharpe: {sharpe:.2f}")
            return sharpe
        else:
            logger.warning("No se obtuvieron retornos durante el entrenamiento")
            return -1.0
        
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        return -1.0

def objective(trial):
    """
    Función objetivo para optimización con Optuna.
    
    Args:
        trial: Objeto de prueba de Optuna
        
    Returns:
        float: Métrica a optimizar (Sharpe ratio)
    """
    # Cargar datos
    data_dir = './data/raw'
    ticker = 'AAPL'  # Cambiar según disponibilidad de datos
    data = load_data(
        data_dir=data_dir,
        ticker=ticker,
        start_date='2020-01-01',
        end_date='2022-01-01'
    )
    
    if '1h' not in data or '4h' not in data:
        logger.error("No se pudieron cargar los datos necesarios")
        return -1.0
    
    # Obtener features predeterminadas
    features_1h, features_4h = create_default_features()
    
    # Parámetros a optimizar
    agent_config = {
        'features_1h': features_1h,
        'features_4h': features_4h,
        'lookback_window_1h': trial.suggest_int('lookback_window_1h', 10, 30),
        'lookback_window_4h': trial.suggest_int('lookback_window_4h', 5, 20),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'lambda_gae': trial.suggest_float('lambda_gae', 0.9, 0.99),
        'ppo_epochs': trial.suggest_int('ppo_epochs', 5, 20),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),
        'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.05, log=True),
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'regime_detection': trial.suggest_categorical('regime_detection', [True, False])
    }
    
    # Crear entorno para entrenamiento
    train_env = FinRLMultiTimeframeEnv(
        df_1h=data['1h'],
        df_4h=data['4h'],
        ticker_list=[ticker],
        lookback_window_1h=agent_config['lookback_window_1h'],
        lookback_window_4h=agent_config['lookback_window_4h'],
        initial_amount=10000.0,
        commission_percentage=0.001,
        max_position=1.0,
        max_trades_per_day=5,
        features_1h=features_1h,
        features_4h=features_4h,
        live_trading=False
    )
    
    # Configuración de entrenamiento
    train_config = {
        'epochs': 50,  # Número reducido para optimización
        'target_step': 2048,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_gap': 10
    }
    
    # Entrenar y evaluar
    sharpe = train_and_evaluate(train_env, agent_config, train_config)
    
    return sharpe

def main():
    """Función principal para ejecutar la optimización."""
    # Verificar que exista el directorio de datos
    data_dir = './data/raw'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.warning(f"Directorio de datos {data_dir} no existía y ha sido creado")
    
    # Crear directorio para resultados
    os.makedirs('./optimization_results', exist_ok=True)
    
    # Configurar estudio de Optuna
    study_name = f"ppo_swing_trader_{datetime.now().strftime('%Y%m%d')}"
    storage_name = f"sqlite:///optimization_results/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=5, n_jobs=1)
    
    # Guardar resultados
    logger.info("Mejor configuración encontrada:")
    logger.info(f"  Valor: {study.best_value:.4f}")
    logger.info(f"  Parámetros: {study.best_params}")
    
    # Guardar la mejor configuración
    with open('./optimization_results/best_params.json', 'w') as f:
        json.dump({
            'value': study.best_value,
            'params': study.best_params
        }, f, indent=2)
    
    # Crear visualizaciones
    try:
        # Importar aquí para no requerir dependencias si no se usa
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Historial de optimización
        fig = plot_optimization_history(study)
        fig.write_image('./optimization_results/optimization_history.png')
        
        # Importancia de parámetros
        fig = plot_param_importances(study)
        fig.write_image('./optimization_results/param_importances.png')
        
        logger.info("Visualizaciones guardadas en './optimization_results/'")
    except ImportError:
        logger.warning("No se pudieron crear visualizaciones. Instala plotly para verlas.")

if __name__ == "__main__":
    main()