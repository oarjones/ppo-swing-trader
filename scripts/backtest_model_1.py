#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtesting para PPO Multi-Timeframe.

Este script ejecuta un backtest del modelo entrenado en datos históricos
y genera informes de rendimiento.
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
from typing import Dict, List, Tuple, Any, Optional

# Importar componentes de nuestro sistema
from src.finrl_multi_timeframe_env import FinRLMultiTimeframeEnv
from src.elegantrl_multi_timeframe_model import MultiTimeframePPOAgent

# Métricas de evaluación
from external.finrl.plot import plot_return, backtest_stats

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path: str, env: FinRLMultiTimeframeEnv) -> MultiTimeframePPOAgent:
    """
    Cargar modelo entrenado.
    
    Args:
        model_path: Ruta al modelo
        env: Entorno de trading para configurar el agente
        
    Returns:
        Agente MultiTimeframePPO cargado
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
        # Crear el agente
        agent = MultiTimeframePPOAgent(
            env=env,
            features_1h=len(env.features_1h),
            seq_len_1h=env.lookback_window_1h,
            features_4h=len(env.features_4h),
            seq_len_4h=env.lookback_window_4h,
            hidden_dim=128  # Este valor será sobrescrito por el modelo cargado
        )
        
        # Cargar pesos del modelo
        agent.load_model(model_path)
        logger.info(f"Modelo cargado correctamente desde {model_path}")
        
        return agent
    
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

def load_config(config_path: str) -> Dict:
    """
    Cargar configuración desde archivo JSON.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Dict con configuración
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        else:
            logger.warning(f"No se encontró archivo de configuración en {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return {}

def load_data(data_dir: str, ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Cargar datos para backtesting.
    
    Args:
        data_dir: Directorio con archivos CSV
        ticker: Ticker a cargar
        start_date: Fecha de inicio
        end_date: Fecha de fin
        
    Returns:
        Tupla de (df_1h, df_4h) con los datos cargados
    """
    # Cargar datos 1h
    path_1h = os.path.join(data_dir, f"{ticker}_1h.csv")
    df_1h = None
    
    if os.path.exists(path_1h):
        try:
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
            
            logger.info(f"Cargados {len(df_1h)} registros para {ticker}_1h")
        except Exception as e:
            logger.error(f"Error cargando datos 1h: {e}")
            df_1h = None
    else:
        logger.warning(f"No se encontró el archivo {path_1h}")
    
    # Cargar datos 4h
    path_4h = os.path.join(data_dir, f"{ticker}_4h.csv")
    df_4h = None
    
    if os.path.exists(path_4h):
        try:
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
            
            logger.info(f"Cargados {len(df_4h)} registros para {ticker}_4h")
        except Exception as e:
            logger.error(f"Error cargando datos 4h: {e}")
            df_4h = None
    else:
        logger.warning(f"No se encontró el archivo {path_4h}")
    
    return df_1h, df_4h

def create_features():
    """
    Crear listas de features predeterminadas.
    
    Returns:
        Tupla de (features_1h, features_4h)
    """
    # 1-hour features
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
    
    # 4-hour features
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

def run_backtest(env: FinRLMultiTimeframeEnv, agent: MultiTimeframePPOAgent) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecutar backtest con el modelo cargado.
    
    Args:
        env: Entorno de trading configurado para backtest
        agent: Agente entrenado
        
    Returns:
        Tupla de (account_value_df, trade_history_df)
    """
    # Resetear entorno
    state, _ = env.reset()
    done = False
    
    # Inicializar registros
    account_values = []
    trade_history = []
    current_date = None
    
    # Configurar device
    device = next(agent.act.parameters()).device if agent.act else torch.device('cpu')
    
    # Ejecutar backtest
    logger.info("Iniciando backtesting...")
    
    while not done:
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Obtener acción del agente
        with torch.no_grad():
            action = agent.act(state_tensor).detach().cpu().numpy()[0]
        
        # Ejecutar paso en el entorno
        next_state, reward, done, truncated, info = env.step(action)
        
        # Registrar información
        current_date = info.get('date', current_date)
        account_value = info.get('portfolio_value', 0)
        
        account_values.append({
            'date': current_date,
            'account_value': account_value,
            'reward': reward,
            'position': info.get('position', 0)
        })
        
        # Registrar operaciones si hubo cambio en la posición
        # (este criterio puede variar según cómo esté implementado tu entorno)
        if info.get('trade_executed', False):
            trade_history.append({
                'date': current_date,
                'action': 'BUY' if info.get('position', 0) > 0 else 'SELL' if info.get('position', 0) < 0 else 'FLAT',
                'price': info.get('position_price', 0),
                'quantity': abs(info.get('position', 0)),
                'cost': info.get('cost', 0)
            })
        
        # Actualizar estado
        state = next_state
    
    # Convertir a DataFrames
    account_value_df = pd.DataFrame(account_values)
    trade_history_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
    
    logger.info(f"Backtest completado. Pasos: {len(account_value_df)}, Operaciones: {len(trade_history_df)}")
    
    return account_value_df, trade_history_df

def calculate_performance_metrics(account_value_df: pd.DataFrame) -> Dict:
    """
    Calcular métricas de rendimiento.
    
    Args:
        account_value_df: DataFrame con valores de cuenta
        
    Returns:
        Dict con métricas de rendimiento
    """
    # Convertir 'date' a datetime si no lo es
    if 'date' in account_value_df.columns and not pd.api.types.is_datetime64_any_dtype(account_value_df['date']):
        account_value_df['date'] = pd.to_datetime(account_value_df['date'], utc=True)
    
    # Calcular retorno total
    initial_value = account_value_df['account_value'].iloc[0]
    final_value = account_value_df['account_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Calcular retornos diarios
    account_value_df['daily_return'] = account_value_df['account_value'].pct_change()
    
    # Calcular Drawdown
    account_value_df['peak'] = account_value_df['account_value'].cummax()
    account_value_df['drawdown'] = (account_value_df['account_value'] / account_value_df['peak'] - 1) * 100
    max_drawdown = account_value_df['drawdown'].min()
    
    # Calcular Sharpe y Sortino (asumiendo datos diarios)
    # Anualizar retornos (aproximadamente 252 días de trading por año)
    daily_returns = account_value_df['daily_return'].dropna()
    annual_return = daily_returns.mean() * 252 * 100
    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (asumiendo tasa libre de riesgo 0 para simplificar)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    
    # Sortino Ratio (considerando solo retornos negativos para el riesgo)
    negative_returns = daily_returns[daily_returns < 0]
    downside_risk = negative_returns.std() * np.sqrt(252) * 100
    sortino_ratio = annual_return / downside_risk if downside_risk != 0 else 0
    
    # Calcular número de operaciones rentables/no rentables si hay datos de operaciones
    win_rate = None
    if 'trade_return' in account_value_df.columns:
        total_trades = len(account_value_df[account_value_df['trade_return'].notnull()])
        winning_trades = len(account_value_df[account_value_df['trade_return'] > 0])
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
    
    # Retornar métricas
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
    }
    
    if win_rate is not None:
        metrics['win_rate'] = win_rate
    
    return metrics

def plot_performance(account_value_df: pd.DataFrame, trade_history_df: pd.DataFrame, output_dir: str):
    """
    Generar gráficos de rendimiento.
    
    Args:
        account_value_df: DataFrame con valores de cuenta
        trade_history_df: DataFrame con historial de operaciones
        output_dir: Directorio para guardar gráficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Asegurar que la fecha es datetime
    if 'date' in account_value_df.columns and not pd.api.types.is_datetime64_any_dtype(account_value_df['date']):
        account_value_df['date'] = pd.to_datetime(account_value_df['date'], utc=True)
    
    # 1. Gráfico de valor de la cuenta
    plt.figure(figsize=(12, 6))
    plt.plot(account_value_df['date'], account_value_df['account_value'], label='Valor de la Cuenta', color='blue')
    plt.title('Evolución del Valor de la Cuenta')
    plt.xlabel('Fecha')
    plt.ylabel('Valor ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'account_value.png'))
    plt.close()
    
    # 2. Gráfico de drawdown
    if 'drawdown' not in account_value_df.columns:
        account_value_df['peak'] = account_value_df['account_value'].cummax()
        account_value_df['drawdown'] = (account_value_df['account_value'] / account_value_df['peak'] - 1) * 100
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(account_value_df['date'], account_value_df['drawdown'], 0, color='red', alpha=0.3)
    plt.plot(account_value_df['date'], account_value_df['drawdown'], color='red', label='Drawdown')
    plt.title('Drawdown (%)')
    plt.xlabel('Fecha')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'))
    plt.close()
    
    # 3. Gráfico de posiciones
    if 'position' in account_value_df.columns:
        plt.figure(figsize=(12, 8))
        
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(account_value_df['date'], account_value_df['account_value'], label='Valor de la Cuenta', color='blue')
        ax1.set_title('Valor de la Cuenta y Posiciones')
        ax1.set_ylabel('Valor ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(account_value_df['date'], account_value_df['position'], label='Posición', color='green')
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Posición')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'positions.png'))
        plt.close()
    
    # 4. Gráfico de operaciones si hay datos
    if not trade_history_df.empty and 'date' in trade_history_df.columns and 'price' in trade_history_df.columns:
        buy_trades = trade_history_df[trade_history_df['action'] == 'BUY']
        sell_trades = trade_history_df[trade_history_df['action'] == 'SELL']
        
        plt.figure(figsize=(12, 6))
        plt.plot(account_value_df['date'], account_value_df['account_value'], label='Valor de la Cuenta', color='blue')
        
        # Marcar operaciones de compra y venta
        if not buy_trades.empty:
            # Asegurar que la fecha es datetime
            if not pd.api.types.is_datetime64_any_dtype(buy_trades['date']):
                buy_trades['date'] = pd.to_datetime(buy_trades['date'], utc=True)
            plt.scatter(buy_trades['date'], buy_trades['price'], marker='^', color='green', s=100, label='Compra')
        
        if not sell_trades.empty:
            # Asegurar que la fecha es datetime
            if not pd.api.types.is_datetime64_any_dtype(sell_trades['date']):
                sell_trades['date'] = pd.to_datetime(sell_trades['date'], utc=True)
            plt.scatter(sell_trades['date'], sell_trades['price'], marker='v', color='red', s=100, label='Venta')
        
        plt.title('Operaciones')
        plt.xlabel('Fecha')
        plt.ylabel('Valor ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trades.png'))
        plt.close()
    
    logger.info(f"Gráficos guardados en {output_dir}")

def generate_performance_report(metrics: Dict, account_value_df: pd.DataFrame, trade_history_df: pd.DataFrame, output_dir: str):
    """
    Generar informe de rendimiento en HTML.
    
    Args:
        metrics: Diccionario con métricas de rendimiento
        account_value_df: DataFrame con valores de cuenta
        trade_history_df: DataFrame con historial de operaciones
        output_dir: Directorio para guardar el informe
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear informe HTML
    html_content = f"""
    <html>
    <head>
        <title>Informe de Backtest</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric {{ font-weight: bold; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ width: 48%; margin: 1%; }}
            .full-width {{ width: 98%; margin: 1%; }}
        </style>
    </head>
    <body>
        <h1>Informe de Backtest</h1>
        <p>Fecha del informe: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Métricas de Rendimiento</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
            </tr>
            <tr>
                <td>Retorno Total</td>
                <td class="{'positive' if metrics['total_return'] >= 0 else 'negative'}">{metrics['total_return']:.2f}%</td>
            </tr>
            <tr>
                <td>Retorno Anualizado</td>
                <td class="{'positive' if metrics['annual_return'] >= 0 else 'negative'}">{metrics['annual_return']:.2f}%</td>
            </tr>
            <tr>
                <td>Volatilidad Anualizada</td>
                <td>{metrics['annual_volatility']:.2f}%</td>
            </tr>
            <tr>
                <td>Ratio de Sharpe</td>
                <td class="{'positive' if metrics['sharpe_ratio'] >= 1 else ''}">{metrics['sharpe_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Ratio de Sortino</td>
                <td class="{'positive' if metrics['sortino_ratio'] >= 1 else ''}">{metrics['sortino_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Drawdown Máximo</td>
                <td class="negative">{metrics['max_drawdown']:.2f}%</td>
            </tr>
    """
    
    # Agregar Win Rate si está disponible
    if 'win_rate' in metrics:
        html_content += f"""
            <tr>
                <td>Tasa de Acierto</td>
                <td>{metrics['win_rate']:.2f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Gráficos</h2>
        <div class="container">
            <div class="chart">
                <img src="account_value.png" alt="Valor de la Cuenta" style="width: 100%;">
            </div>
            <div class="chart">
                <img src="drawdown.png" alt="Drawdown" style="width: 100%;">
            </div>
        </div>
    """
    
    # Agregar gráfico de posiciones si existe
    if os.path.exists(os.path.join(output_dir, 'positions.png')):
        html_content += """
        <div class="container">
            <div class="full-width">
                <img src="positions.png" alt="Posiciones" style="width: 100%;">
            </div>
        </div>
        """
    
    # Agregar gráfico de operaciones si existe
    if os.path.exists(os.path.join(output_dir, 'trades.png')):
        html_content += """
        <div class="container">
            <div class="full-width">
                <img src="trades.png" alt="Operaciones" style="width: 100%;">
            </div>
        </div>
        """
    
    # Agregar historial de operaciones si hay datos
    if not trade_history_df.empty:
        html_content += """
        <h2>Historial de Operaciones</h2>
        <table>
            <tr>
                <th>Fecha</th>
                <th>Acción</th>
                <th>Precio</th>
                <th>Cantidad</th>
                <th>Coste</th>
            </tr>
        """
        
        for _, row in trade_history_df.iterrows():
            action_class = "positive" if row['action'] == 'BUY' else "negative" if row['action'] == 'SELL' else ""
            html_content += f"""
            <tr>
                <td>{row['date']}</td>
                <td class="{action_class}">{row['action']}</td>
                <td>${row['price']:.2f}</td>
                <td>{row['quantity']}</td>
                <td>${row['cost']:.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Guardar informe HTML
    with open(os.path.join(output_dir, 'backtest_report.html'), 'w') as f:
        f.write(html_content)
    
    # Guardar datos en CSV para análisis adicionales
    account_value_df.to_csv(os.path.join(output_dir, 'account_value.csv'), index=False)
    if not trade_history_df.empty:
        trade_history_df.to_csv(os.path.join(output_dir, 'trade_history.csv'), index=False)
    
    logger.info(f"Informe de rendimiento guardado en {os.path.join(output_dir, 'backtest_report.html')}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Backtest para PPO Multi-Timeframe')
    
    # Parámetros de entrada
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Directorio con datos para backtesting')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Ticker para backtest')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Fecha de inicio para datos de backtest')
    parser.add_argument('--end_date', type=str, default=None,
                       help='Fecha de fin para datos de backtest')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Ruta a archivo de configuración (opcional)')
    parser.add_argument('--output_dir', type=str, default='./backtest_results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--initial_amount', type=float, default=10000.0,
                       help='Cantidad inicial para backtest')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Comisión por operación (porcentaje)')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar configuración si se proporciona
    config = load_config(args.config_path) if args.config_path else {}
    
    # Cargar datos
    df_1h, df_4h = load_data(
        data_dir=args.data_dir,
        ticker=args.ticker,
        start_date=args.start_date or config.get('start_date'),
        end_date=args.end_date or config.get('end_date')
    )
    
    if df_1h is None or df_4h is None:
        logger.error("No se pudieron cargar los datos necesarios")
        return
    
    # Crear listas de features
    features_1h, features_4h = create_features()
    
    # Configurar entorno para backtest
    env = FinRLMultiTimeframeEnv(
        df_1h=df_1h,
        df_4h=df_4h,
        ticker_list=[args.ticker],
        lookback_window_1h=config.get('lookback_window_1h', 20),
        lookback_window_4h=config.get('lookback_window_4h', 10),
        initial_amount=args.initial_amount,
        commission_percentage=args.commission,
        max_position=config.get('max_position', 1.0),
        max_trades_per_day=config.get('max_trades_per_day', 5),
        features_1h=features_1h,
        features_4h=features_4h,
        live_trading=False
    )
    
    # Cargar modelo
    agent = load_model(args.model_path, env)
    
    # Ejecutar backtest
    account_value_df, trade_history_df = run_backtest(env, agent)
    
    # Calcular métricas de rendimiento
    metrics = calculate_performance_metrics(account_value_df)
    logger.info(f"Métricas de rendimiento: {metrics}")
    
    # Guardar resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_dir = os.path.join(args.output_dir, f"backtest_{args.ticker}_{timestamp}")
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Generar gráficos
    plot_performance(account_value_df, trade_history_df, backtest_dir)
    
    # Generar informe
    generate_performance_report(metrics, account_value_df, trade_history_df, backtest_dir)
    
    logger.info(f"Backtest completado. Resultados guardados en {backtest_dir}")
    
    # Mostrar ruta a informe en HTML
    print(f"\nInforme de backtest generado en: {os.path.join(backtest_dir, 'backtest_report.html')}")

if __name__ == "__main__":
    main()