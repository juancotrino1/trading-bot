# -*- coding: utf-8 -*-
"""
SISTEMA DE TRADING MULTI-ACTIVO CON WALK-FORWARD VALIDATION Y BACKTESTING
================================================================================
Arquitectura:
1. Pipeline completo sin look-ahead bias
2. Validación Walk-Forward con regresión logística regularizada
3. Sistema de scoring probabilístico (no umbrales rígidos)
4. Backtesting de 3 meses con métricas realistas
5. Gestión de riesgo basada en ATR dinámico

Características:
- Eliminación completa de look-ahead bias
- Modelos entrenados solo con datos históricos disponibles
- Scoring continuo (0-100%) en lugar de clasificación binaria
- Separación estricta entre entrenamiento, validación y testing
- Métricas out-of-sample exclusivamente
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import warnings
import os
import requests
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ============================================
print("=" * 80)
print("SISTEMA DE TRADING CON VALIDACIÓN WALK-FORWARD Y BACKTESTING")
print("=" * 80)

# Configuración de tiempo
colombia_tz = pytz.timezone('America/Bogota')
#FECHA_ACTUAL = datetime(2025, 9, 24, 14, 0, tzinfo=colombia_tz)
FECHA_ACTUAL = datetime.now(colombia_tz)
FECHA_INICIO_BACKTEST = FECHA_ACTUAL - timedelta(days=90)  # 3 meses para backtesting
FECHA_INICIO_ENTRENAMIENTO = FECHA_INICIO_BACKTEST - timedelta(days=365)  # 1 año antes para entrenamiento

# Parámetros del sistema
INTERVALO = "1h"
ACTIVOS = ["BTC-USD"]  # Puedes agregar más activos aquí

# Parámetros de Mean Reversion
VENTANA_MR = 60
K_DESVIACIONES = 2.5

# Parámetros de gestión de riesgo
ATR_PERIODO = 14
MULTIPLICADOR_SL = 1.5
MULTIPLICADOR_TP = 2.55
RATIO_MINIMO_RR = 1.7

# Horizontes de predicción (en horas)
HORIZONTES = [4, 8, 12, 24, 48]

# Configuración Walk-Forward
N_FOLDS_WF = 5  # Número de folds para walk-forward
TAMANIO_TEST_WF = 0.2  # 20% para test en cada fold

# Configuración modelos
FEATURES_TECNICAS = [
    'RSI', 'Volatilidad', 'Distancia_MA', 'Rango_HL',
    'Aceleracion_Retorno', 'RSI_lag1', 'Volatilidad_lag1',
    'ADX', 'Posicion_BB', 'ATR', 'ATR_porcentaje'
]

FEATURES_CALIDAD = [
    'RSI', 'Volatilidad', 'Distancia_MA', 'Rango_HL',
    'Aceleracion_Retorno', 'ADX', 'Posicion_BB', 'ATR',
    'ATR_porcentaje', 'Retorno_Log', 'Intensidad_Anomalia', 
    'Volumen_Relativo', 'Distancia_Banda', 'Momento_Relativo'
]

print(f"\n📊 CONFIGURACIÓN DEL SISTEMA:")
print(f" Período backtesting: {FECHA_INICIO_BACKTEST.date()} - {FECHA_ACTUAL.date()}")
print(f" Período entrenamiento: {FECHA_INICIO_ENTRENAMIENTO.date()} - {FECHA_INICIO_BACKTEST.date()}")
print(f" Activoss: {len(ACTIVOS)}")
print(f" Validación Walk-Forward: {N_FOLDS_WF} folds")
print(f" Modelo principal: Logistic Regression (L2)")
print(f" Sistema de scoring: Probabilístico (0-100%)")

# ============================================
# 2. FUNCIONES DE CÁLCULO DE INDICADORES
# ============================================

def calcular_indicadores_tecnicos(df):
    """
    Calcula todos los indicadores técnicos sin look-ahead bias.
    Solo utiliza datos disponibles hasta cada punto en el tiempo.
    """
    df = df.copy()
    
    # 1. Retorno logarítmico (base para muchos indicadores)
    df['Retorno_Log'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Mean Reversion con ventana adaptativa
    ventana_efectiva = min(VENTANA_MR, len(df) // 4)
    if ventana_efectiva < 10:
        ventana_efectiva = 10
    
    df['Media_Movil_Retorno'] = df['Retorno_Log'].rolling(
        window=ventana_efectiva, min_periods=5).mean()
    df['Desviacion_Retorno'] = df['Retorno_Log'].rolling(
        window=ventana_efectiva, min_periods=5).std()
    
    df['Banda_Superior_MR'] = df['Media_Movil_Retorno'] + K_DESVIACIONES * df['Desviacion_Retorno']
    df['Banda_Inferior_MR'] = df['Media_Movil_Retorno'] - K_DESVIACIONES * df['Desviacion_Retorno']
    
    # 3. Detección de anomalías (solo basado en datos pasados)
    df['Anomalia'] = (
        (df['Retorno_Log'] > df['Banda_Superior_MR']) | 
        (df['Retorno_Log'] < df['Banda_Inferior_MR'])
    )
    
    # 4. Dirección de la señal (LONG/SHORT)
    df['Senal_MR'] = np.where(
        df['Retorno_Log'] < df['Banda_Inferior_MR'], 
        'LONG', 
        np.where(df['Retorno_Log'] > df['Banda_Superior_MR'], 'SHORT', 'NEUTRAL')
    )
    
    # 5. Intensidad de la anomalía (z-score)
    df['Intensidad_Anomalia'] = (
        (df['Retorno_Log'] - df['Media_Movil_Retorno']) / df['Desviacion_Retorno']
    ).abs()
    
    # 6. Distancia a la banda (para características de calidad)
    df['Distancia_Banda'] = np.where(
        df['Retorno_Log'] > df['Banda_Superior_MR'],
        (df['Retorno_Log'] - df['Banda_Superior_MR']) / df['Desviacion_Retorno'],
        np.where(df['Retorno_Log'] < df['Banda_Inferior_MR'],
                (df['Banda_Inferior_MR'] - df['Retorno_Log']) / df['Desviacion_Retorno'],
                0)
    )
    
    # 7. ATR (Average True Range) - CORREGIDO
    # Calcular True Range
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    
    # Calcular ATR como media móvil del TR
    df['ATR'] = df['TR'].rolling(window=ATR_PERIODO, min_periods=5).mean()
    
    # Calcular ATR como porcentaje del precio
    df['ATR_porcentaje'] = (df['ATR'] / df['Close']) * 100
    
    # 8. RSI
    delta = df['Close'].diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
    rs = ganancia / perdida
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_lag1'] = df['RSI'].shift(1)
    
    # 9. Volatilidad (desviación estándar de retornos)
    df['Volatilidad'] = df['Retorno_Log'].rolling(window=24, min_periods=5).std()
    df['Volatilidad_lag1'] = df['Volatilidad'].shift(1)
    
    # 10. Distancia a la media móvil
    media_50 = df['Close'].rolling(window=50, min_periods=10).mean()
    df['Distancia_MA'] = (df['Close'] - media_50) / media_50
    
    # 11. Rango High-Low normalizado
    df['Rango_HL'] = (df['High'] - df['Low']) / df['Close']
    
    # 12. Aceleración del retorno
    df['Aceleracion_Retorno'] = df['Retorno_Log'].diff()
    
    # 13. ADX (simplificado como ratio de volatilidad)
    df['ADX'] = df['Close'].rolling(14, min_periods=5).std() / df['Close'].rolling(50, min_periods=10).std()
    
    # 14. Bandas de Bollinger
    media_20 = df['Close'].rolling(window=20, min_periods=5).mean()
    std_20 = df['Close'].rolling(window=20, min_periods=5).std()
    df['BB_superior'] = media_20 + 2 * std_20
    df['BB_inferior'] = media_20 - 2 * std_20
    df['Posicion_BB'] = (df['Close'] - df['BB_inferior']) / (df['BB_superior'] - df['BB_inferior'])
    
    # 15. Momento relativo (retorno de las últimas N velas)
    df['Momento_Relativo'] = df['Close'] / df['Close'].shift(5) - 1
    
    # 16. Volumen relativo (si está disponible)
    if 'Volume' in df.columns:
        media_volumen = df['Volume'].rolling(window=20, min_periods=5).mean()
        df['Volumen_Relativo'] = df['Volume'] / media_volumen
    else:
        df['Volumen_Relativo'] = 1.0
    
    # 17. Targets para horizontes futuros
    for horizonte in HORIZONTES:
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        df[f'Target_{horizonte}h'] = (retorno_futuro > 0).astype(int)
    
    # Eliminar columnas temporales
    if 'TR' in df.columns:
        df = df.drop(columns=['TR'])
    
    return df.dropna()

def calcular_niveles_riesgo_atr(precio_actual, atr, direccion):
    """
    Calcula niveles de Stop Loss y Take Profit basados en ATR.
    """
    if direccion == 'LONG':
        stop_loss = precio_actual - (MULTIPLICADOR_SL * atr)
        take_profit = precio_actual + (MULTIPLICADOR_TP * atr)
    else:  # SHORT
        stop_loss = precio_actual + (MULTIPLICADOR_SL * atr)
        take_profit = precio_actual - (MULTIPLICADOR_TP * atr)
    
    riesgo = abs(precio_actual - stop_loss)
    recompensa = abs(take_profit - precio_actual)
    ratio_rr = recompensa / riesgo if riesgo > 0 else 0
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'riesgo': riesgo,
        'recompensa': recompensa,
        'ratio_rr': ratio_rr,
        'atr_valor': atr
    }

# ============================================
# 3. FUNCIONES DE VALIDACIÓN WALK-FORWARD
# ============================================

def crear_splits_walkforward(datos, n_folds=N_FOLDS_WF, test_size=TAMANIO_TEST_WF):
    """
    Crea splits temporales para validación walk-forward.
    Garantiza que no haya look-ahead bias.
    """
    splits = []
    datos_ordenados = datos.sort_index()
    n_total = len(datos_ordenados)
    n_test = int(n_total * test_size)
    
    # Si no hay suficientes datos, reducir folds
    if n_total < n_folds * n_test * 2:
        n_folds = max(2, n_total // (2 * n_test))
    
    for i in range(n_folds):
        # Porcentajes crecientes para train
        train_end_idx = int(n_total * (0.5 + 0.3 * (i / n_folds)))
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + n_test, n_total)
        
        if test_end_idx - test_start_idx < 10:  # Mínimo para test
            continue
        
        train_data = datos_ordenados.iloc[:train_end_idx]
        test_data = datos_ordenados.iloc[test_start_idx:test_end_idx]
        
        splits.append({
            'train': train_data,
            'test': test_data,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1]
        })
    
    return splits

def entrenar_modelo_walkforward(datos_completos, features, target_col, 
                               modelo_class=LogisticRegression, 
                               param_grid=None):
    """
    Entrena modelo con validación walk-forward.
    Retorna métricas out-of-sample y modelo final.
    """
    if param_grid is None:
        param_grid = {
            'C': 0.1,
            'penalty': 'l2',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    splits = crear_splits_walkforward(datos_completos)
    
    if len(splits) == 0:
        print("⚠️ No se pudieron crear splits walk-forward. Usando split simple 80/20.")
        # Fallback a split simple
        split_idx = int(len(datos_completos) * 0.8)
        splits = [{
            'train': datos_completos.iloc[:split_idx],
            'test': datos_completos.iloc[split_idx:],
            'train_start': datos_completos.index[0],
            'train_end': datos_completos.index[split_idx-1],
            'test_start': datos_completos.index[split_idx],
            'test_end': datos_completos.index[-1]
        }]
    
    metricas_folds = []
    modelos = []
    scalers = []
    
    print(f"\n🔧 Entrenando con {len(splits)} folds walk-forward...")
    
    for idx, split in enumerate(splits, 1):
        X_train = split['train'][features]
        y_train = split['train'][target_col]
        X_test = split['test'][features]
        y_test = split['test'][target_col]
        
        # Verificar que tenemos datos suficientes
        if len(X_train) < 10 or len(X_test) < 5:
            print(f"⚠️ Fold {idx}: Datos insuficientes. Saltando...")
            continue
        
        # Verificar que todas las features existan
        missing_features = [f for f in features if f not in X_train.columns]
        if missing_features:
            print(f"⚠️ Fold {idx}: Features faltantes: {missing_features}. Saltando...")
            continue
        
        # Escalado robusto (menos sensible a outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        if modelo_class == LogisticRegression:
            modelo = modelo_class(**param_grid)
        else:
            modelo = modelo_class(**param_grid)
        
        try:
            modelo.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"❌ Fold {idx}: Error entrenando modelo: {e}")
            continue
        
        # Predecir en test
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcular métricas
        if len(np.unique(y_test)) > 1:
            try:
                y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                accuracy = precision = recall = f1 = auc = 0.0
        else:
            accuracy = precision = recall = f1 = auc = 0.0
        
        metricas_folds.append({
            'fold': idx,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_period': f"{split['train_start'].date()} a {split['train_end'].date()}",
            'test_period': f"{split['test_start'].date()} a {split['test_end'].date()}"
        })
        
        modelos.append(modelo)
        scalers.append(scaler)
        
        print(f" Fold {idx}: Test AUC={auc:.3f}, F1={f1:.3f}, "
              f"Train={len(X_train)} registros, Test={len(X_test)} registros")
    
    if not metricas_folds:
        print("❌ No se pudo entrenar ningún fold. Usando modelo simple.")
        return None
    
    # Promediar métricas de todos los folds
    metricas_promedio = {
        'accuracy': np.mean([m['accuracy'] for m in metricas_folds]),
        'precision': np.mean([m['precision'] for m in metricas_folds]),
        'recall': np.mean([m['recall'] for m in metricas_folds]),
        'f1': np.mean([m['f1'] for m in metricas_folds]),
        'auc': np.mean([m['auc'] for m in metricas_folds])
    }
    
    # Entrenar modelo final con todos los datos históricos
    print("\n🎯 Entrenando modelo final con todos los datos históricos...")
    X_final = datos_completos[features]
    y_final = datos_completos[target_col]
    
    # Verificar que todas las features existan
    missing_features = [f for f in features if f not in X_final.columns]
    if missing_features:
        print(f"❌ Features faltantes en datos finales: {missing_features}")
        # Crear features faltantes con valor 0
        for f in missing_features:
            X_final[f] = 0
    
    scaler_final = RobustScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)
    
    if modelo_class == LogisticRegression:
        modelo_final = modelo_class(**param_grid)
    else:
        modelo_final = modelo_class(**param_grid)
    
    modelo_final.fit(X_final_scaled, y_final)
    
    return {
        'modelo': modelo_final,
        'scaler': scaler_final,
        'metricas_folds': metricas_folds,
        'metricas_promedio': metricas_promedio,
        'modelos_folds': modelos,
        'scalers_folds': scalers
    }

# ============================================
# 4. PIPELINE DE ENTRENAMIENTO COMPLETO
# ============================================

def pipeline_entrenamiento_completo():
    """
    Pipeline completo de entrenamiento sin look-ahead bias.
    """
    print("\n" + "="*80)
    print("FASE 1: PREPARACIÓN DE DATOS HISTÓRICOS")
    print("="*80)
    
    # Descargar datos históricos para entrenamiento
    datos_entrenamiento = []
    
    for activo in ACTIVOS:
        print(f"\n📊 Descargando {activo} para entrenamiento...")
        
        # Descargar desde inicio entrenamiento hasta inicio backtest
        try:
            datos = yf.download(
                activo, 
                start=FECHA_INICIO_ENTRENAMIENTO,
                end=FECHA_INICIO_BACKTEST,
                interval=INTERVALO,
                progress=False
            )
        except Exception as e:
            print(f"❌ Error descargando {activo}: {e}")
            continue
        
        if datos.empty:
            print(f"⚠️ Sin datos para {activo}")
            continue
        
        # Limpiar columnas MultiIndex si es necesario
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.droplevel(1)
        
        # Verificar que tengamos las columnas necesarias
        columnas_necesarias = ['Open', 'High', 'Low', 'Close']
        if not all(col in datos.columns for col in columnas_necesarias):
            print(f"⚠️ {activo} no tiene todas las columnas necesarias")
            continue
        
        # Procesar datos
        datos_procesados = calcular_indicadores_tecnicos(datos)
        datos_procesados['Activo'] = activo
        
        # Filtrar solo anomalías para análisis
        anomalias = datos_procesados[datos_procesados['Anomalia']].copy()
        
        # Calcular niveles de riesgo para cada anomalía
        rr_ratios = []
        for idx, fila in anomalias.iterrows():
            niveles = calcular_niveles_riesgo_atr(
                fila['Close'], 
                fila['ATR'], 
                fila['Senal_MR']
            )
            rr_ratios.append(niveles['ratio_rr'])
        
        datos_procesados['Ratio_RR'] = np.nan
        datos_procesados.loc[anomalias.index, 'Ratio_RR'] = rr_ratios
        
        datos_entrenamiento.append(datos_procesados)
        
        print(f"✅ {activo}: {len(datos_procesados)} registros, "
              f"{len(anomalias)} anomalías")
    
    if not datos_entrenamiento:
        print("❌ ERROR: No se descargaron datos de ningún activo")
        return None
    
    datos_combinados = pd.concat(datos_entrenamiento, axis=0)
    
    print(f"\n📈 DATASET FINAL DE ENTRENAMIENTO:")
    print(f" Total registros: {len(datos_combinados):,}")
    print(f" Total anomalías: {datos_combinados['Anomalia'].sum():,}")
    print(f" Activos: {datos_combinados['Activo'].nunique()}")
    
    # ============================================
    # ENTRENAMIENTO DE MODELOS POR HORIZONTE
    # ============================================
    
    print("\n" + "="*80)
    print("FASE 2: ENTRENAMIENTO DE MODELOS POR HORIZONTE")
    print("="*80)
    
    modelos_por_horizonte = {}
    metricas_modelos = {}
    
    for horizonte in HORIZONTES:
        print(f"\n🎯 HORIZONTE {horizonte}h")
        print("-" * 40)
        
        target_col = f'Target_{horizonte}h'
        
        # Filtrar datos donde el target está disponible
        datos_horizonte = datos_combinados.dropna(subset=[target_col])
        
        # Verificar que tenemos suficientes datos
        if len(datos_horizonte) < 100:
            print(f"⚠️ Datos insuficientes para horizonte {horizonte}h: {len(datos_horizonte)} registros")
            continue
        
        # Verificar que todas las features existan
        missing_features = [f for f in FEATURES_TECNICAS if f not in datos_horizonte.columns]
        if missing_features:
            print(f"⚠️ Features faltantes para horizonte {horizonte}h: {missing_features}")
            # Crear features faltantes con valor 0
            for f in missing_features:
                datos_horizonte[f] = 0
        
        resultado = entrenar_modelo_walkforward(
            datos_horizonte,
            FEATURES_TECNICAS,
            target_col,
            modelo_class=LogisticRegression,
            param_grid={
                'C': 0.1,
                'penalty': 'l2',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
        )
        
        if resultado is None:
            print(f"❌ No se pudo entrenar modelo para horizonte {horizonte}h")
            continue
        
        # Verificar que el resultado tenga la estructura correcta
        if not isinstance(resultado, dict) or 'modelo' not in resultado or 'scaler' not in resultado:
            print(f"❌ Estructura incorrecta del resultado para horizonte {horizonte}h")
            continue
            
        modelos_por_horizonte[horizonte] = {
            'modelo': resultado['modelo'],
            'scaler': resultado['scaler'],
            'metricas': resultado['metricas_promedio']
        }
        
        metricas_modelos[horizonte] = resultado['metricas_promedio']
        
        print(f"✅ Modelo entrenado:")
        print(f"   AUC promedio: {resultado['metricas_promedio']['auc']:.3f}")
        print(f"   F1 promedio: {resultado['metricas_promedio']['f1']:.3f}")
        print(f"   Accuracy: {resultado['metricas_promedio']['accuracy']:.3f}")
    
    if not modelos_por_horizonte:
        print("❌ ERROR: No se pudieron entrenar modelos para ningún horizonte")
        return None
    
    # ============================================
    # ENTRENAMIENTO DEL CLASIFICADOR DE CALIDAD
    # ============================================
    
    print("\n" + "="*80)
    print("FASE 3: ENTRENAMIENTO DEL CLASIFICADOR DE CALIDAD")
    print("="*80)
    
    # Preparar dataset de calidad
    # Solo usar anomalías históricas que ya sabemos si fueron buenas o malas
    anomalias_historicas = datos_combinados[datos_combinados['Anomalia']].copy()
    
    # Calcular si cada anomalía fue "buena" (ganadora en mayoría de horizontes)
    checks = []
    for horizonte in HORIZONTES:
        target_col = f'Target_{horizonte}h'
        if target_col in anomalias_historicas.columns:
            senal_correcta = (
                ((anomalias_historicas['Senal_MR'] == 'LONG') & (anomalias_historicas[target_col] == 1)) |
                ((anomalias_historicas['Senal_MR'] == 'SHORT') & (anomalias_historicas[target_col] == 0))
            )
            checks.append(senal_correcta.astype(int))
    
    if not checks:
        print("⚠️ No hay checks disponibles para calcular calidad")
        clasificador_calidad = None
    else:
        # Calcular score de calidad basado en performance en todos los horizontes
        checks_df = pd.concat(checks, axis=1)
        checks_df.columns = [f'Check_{h}h' for h in HORIZONTES[:len(checks)]]
        
        anomalias_historicas = pd.concat([anomalias_historicas, checks_df], axis=1)
        anomalias_historicas['Score_Calidad'] = checks_df.mean(axis=1)
        
        # Variable objetivo: 1 si el score de calidad > 0.5, 0 en caso contrario
        anomalias_historicas['Target_Calidad'] = (
            anomalias_historicas['Score_Calidad'] > 0.5
        ).astype(int)
        
        print(f"📊 Dataset de calidad:")
        print(f"   Anomalías totales: {len(anomalias_historicas)}")
        print(f"   Buenas anomalías (score > 0.5): {anomalias_historicas['Target_Calidad'].sum()}")
        print(f"   Malas anomalías: {len(anomalias_historicas) - anomalias_historicas['Target_Calidad'].sum()}")
        
        # Verificar que tenemos suficientes datos
        if len(anomalias_historicas) >= 50 and anomalias_historicas['Target_Calidad'].sum() >= 10:
            # Verificar que todas las features de calidad existan
            missing_features = [f for f in FEATURES_CALIDAD if f not in anomalias_historicas.columns]
            if missing_features:
                print(f"⚠️ Features faltantes en dataset de calidad: {missing_features}")
                # Crear features faltantes con valor 0
                for f in missing_features:
                    anomalias_historicas[f] = 0
            
            # Entrenar clasificador de calidad con walk-forward
            resultado_calidad = entrenar_modelo_walkforward(
                anomalias_historicas,
                FEATURES_CALIDAD,
                'Target_Calidad',
                modelo_class=GradientBoostingClassifier,
                param_grid={
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            )
            
            if resultado_calidad is not None:
                clasificador_calidad = {
                    'modelo': resultado_calidad['modelo'],
                    'scaler': resultado_calidad['scaler'],
                    'metricas': resultado_calidad['metricas_promedio']
                }
                
                print(f"✅ Clasificador de calidad entrenado:")
                print(f"   AUC: {resultado_calidad['metricas_promedio']['auc']:.3f}")
                print(f"   F1: {resultado_calidad['metricas_promedio']['f1']:.3f}")
            else:
                print("⚠️ No se pudo entrenar clasificador de calidad")
                clasificador_calidad = None
        else:
            print("⚠️ Insuficientes anomalías para entrenar clasificador de calidad")
            clasificador_calidad = None
    
    # ============================================
    # RESUMEN DEL ENTRENAMIENTO
    # ============================================
    
    print("\n" + "="*80)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*80)
    
    for horizonte in modelos_por_horizonte.keys():
        metrica = metricas_modelos[horizonte]
        print(f"Horizonte {horizonte}h: AUC={metrica['auc']:.3f}, "
              f"F1={metrica['f1']:.3f}, Acc={metrica['accuracy']:.3f}")
    
    return {
        'modelos_direccion': modelos_por_horizonte,
        'clasificador_calidad': clasificador_calidad,
        'datos_entrenamiento': datos_combinados,
        'anomalias_historicas': anomalias_historicas if 'anomalias_historicas' in locals() else None
    }

# ============================================
# 5. BACKTESTING RIGUROSO (3 MESES)
# ============================================

def ejecutar_backtesting(modelos_entrenados, clasificador_calidad=None):
    """
    Ejecuta backtesting de 3 meses con datos out-of-sample.
    """
    print("\n" + "="*80)
    print("FASE 4: BACKTESTING DE 3 MESES")
    print("="*80)
    
    resultados_backtest = []
    operaciones = []
    
    for activo in ACTIVOS:
        print(f"\n📊 Backtesting {activo}...")
        
        # Descargar datos de backtesting (últimos 3 meses)
        try:
            datos_backtest = yf.download(
                activo,
                start=FECHA_INICIO_BACKTEST,
                end=FECHA_ACTUAL,
                interval=INTERVALO,
                progress=False
            )
        except Exception as e:
            print(f"❌ Error descargando datos de backtesting para {activo}: {e}")
            continue
        
        if datos_backtest.empty:
            print(f"⚠️ Sin datos para backtesting de {activo}")
            continue
        
        # Limpiar columnas MultiIndex si es necesario
        if isinstance(datos_backtest.columns, pd.MultiIndex):
            datos_backtest.columns = datos_backtest.columns.droplevel(1)
        
        # Calcular indicadores
        datos_backtest = calcular_indicadores_tecnicos(datos_backtest)
        
        if datos_backtest.empty:
            print(f"⚠️ No se pudieron calcular indicadores para {activo}")
            continue
        
        # Simular trading vela por vela
        # Empezar después de tener suficientes datos para indicadores
        inicio_simulacion = 100
        for i in range(inicio_simulacion, len(datos_backtest) - max(HORIZONTES)):
            fecha_actual = datos_backtest.index[i]
            fila_actual = datos_backtest.iloc[i]
            
            # Verificar si hay anomalía en la vela actual
            if not fila_actual['Anomalia']:
                continue
            
            # ============================================
            # 1. EVALUAR SEÑAL CON DATOS DISPONIBLES HASTA ESE MOMENTO
            # ============================================
            
            precio_actual = fila_actual['Close']
            atr_actual = fila_actual['ATR']
            senal_mr = fila_actual['Senal_MR']
            
            # Calcular niveles de riesgo
            niveles_riesgo = calcular_niveles_riesgo_atr(
                precio_actual, atr_actual, senal_mr
            )
            
            # Filtrar por ratio R:R mínimo
            if niveles_riesgo['ratio_rr'] < RATIO_MINIMO_RR:
                continue
            
            # ============================================
            # 2. PREDICCIONES POR HORIZONTE
            # ============================================
            
            predicciones = {}
            score_alineamiento = 0
            
            # Verificar que modelos_entrenados tiene la estructura correcta
            if not modelos_entrenados or not isinstance(modelos_entrenados, dict):
                print("⚠️ modelos_entrenados no tiene la estructura correcta")
                continue
            
            for horizonte in modelos_entrenados.keys():
                # Verificar que este horizonte tiene la estructura correcta
                if horizonte not in modelos_entrenados:
                    continue
                    
                modelo_info = modelos_entrenados[horizonte]
                
                # Verificar que modelo_info tiene las claves esperadas
                if not isinstance(modelo_info, dict) or 'modelo' not in modelo_info or 'scaler' not in modelo_info:
                    print(f"⚠️ Estructura incorrecta para horizonte {horizonte}")
                    continue
                
                modelo = modelo_info['modelo']
                scaler = modelo_info['scaler']
                
                # Preparar features para predicción
                features_actual = []
                for feature in FEATURES_TECNICAS:
                    if feature in fila_actual.index:
                        features_actual.append(fila_actual[feature])
                    else:
                        features_actual.append(0)
                
                features_actual = np.array(features_actual).reshape(1, -1)
                features_scaled = scaler.transform(features_actual)
                
                # Predecir
                try:
                    pred_proba = modelo.predict_proba(features_scaled)[0, 1]
                    prediccion = 1 if pred_proba > 0.5 else 0
                    
                    # Verificar alineamiento con señal MR
                    senal_predicha = 'LONG' if prediccion == 1 else 'SHORT'
                    alineado = 1 if senal_predicha == senal_mr else 0
                    
                    predicciones[horizonte] = {
                        'prediccion': senal_predicha,
                        'probabilidad': pred_proba,
                        'alineado': alineado
                    }
                    
                    score_alineamiento += alineado
                except Exception as e:
                    print(f"⚠️ Error en predicción para horizonte {horizonte}: {e}")
                    continue
            
            if not predicciones:
                continue
                
            score_alineamiento = (score_alineamiento / len(predicciones)) * 100
            
            # ============================================
            # 3. EVALUAR CALIDAD DE LA ANOMALÍA
            # ============================================
            
            probabilidad_calidad = 0.5  # Default neutral
            
            if clasificador_calidad is not None and isinstance(clasificador_calidad, dict):
                try:
                    # Preparar features para clasificador de calidad
                    features_calidad_actual = []
                    for feature in FEATURES_CALIDAD:
                        if feature in fila_actual.index:
                            features_calidad_actual.append(fila_actual[feature])
                        elif feature == 'Ratio_RR':
                            features_calidad_actual.append(niveles_riesgo['ratio_rr'])
                        else:
                            features_calidad_actual.append(0)
                    
                    features_calidad_actual = np.array(features_calidad_actual).reshape(1, -1)
                    features_scaled = clasificador_calidad['scaler'].transform(features_calidad_actual)
                    probabilidad_calidad = clasificador_calidad['modelo'].predict_proba(features_scaled)[0, 1]
                except Exception as e:
                    print(f"⚠️ Error evaluando calidad: {e}")
                    probabilidad_calidad = 0.5
            
            # ============================================
            # 4. EJECUTAR OPERACIÓN Y SEGUIR RESULTADOS
            # ============================================
            
            # Simular operación para cada horizonte
            for horizonte in predicciones.keys():
                if i + horizonte >= len(datos_backtest):
                    continue
                
                precio_futuro = datos_backtest.iloc[i + horizonte]['Close']
                
                if senal_mr == 'LONG':
                    retorno_real = (precio_futuro - precio_actual) / precio_actual
                    exito = 1 if precio_futuro > precio_actual else 0
                else:  # SHORT
                    retorno_real = (precio_actual - precio_futuro) / precio_actual
                    exito = 1 if precio_futuro < precio_actual else 0
                
                # Verificar stop loss y take profit
                hit_sl = False
                hit_tp = False
                
                # Verificar si se alcanzó SL o TP dentro del horizonte
                for j in range(1, horizonte + 1):
                    if i + j >= len(datos_backtest):
                        break
                    
                    precio_intermedio = datos_backtest.iloc[i + j]['Close']
                    
                    if senal_mr == 'LONG':
                        if precio_intermedio <= niveles_riesgo['stop_loss']:
                            hit_sl = True
                            retorno_real = (niveles_riesgo['stop_loss'] - precio_actual) / precio_actual
                            break
                        elif precio_intermedio >= niveles_riesgo['take_profit']:
                            hit_tp = True
                            retorno_real = (niveles_riesgo['take_profit'] - precio_actual) / precio_actual
                            break
                    else:  # SHORT
                        if precio_intermedio >= niveles_riesgo['stop_loss']:
                            hit_sl = True
                            retorno_real = (precio_actual - niveles_riesgo['stop_loss']) / precio_actual
                            break
                        elif precio_intermedio <= niveles_riesgo['take_profit']:
                            hit_tp = True
                            retorno_real = (precio_actual - niveles_riesgo['take_profit']) / precio_actual
                            break
                
                operacion = {
                    'activo': activo,
                    'fecha': fecha_actual,
                    'senal': senal_mr,
                    'precio_entrada': precio_actual,
                    'precio_salida': precio_futuro,
                    'horizonte': horizonte,
                    'retorno': retorno_real,
                    'exito': exito,
                    'hit_sl': hit_sl,
                    'hit_tp': hit_tp,
                    'ratio_rr': niveles_riesgo['ratio_rr'],
                    'score_alineamiento': score_alineamiento,
                    'probabilidad_calidad': probabilidad_calidad,
                    'atr': atr_actual,
                    'stop_loss': niveles_riesgo['stop_loss'],
                    'take_profit': niveles_riesgo['take_profit']
                }
                
                operaciones.append(operacion)
            
            resultado = {
                'activo': activo,
                'fecha': fecha_actual,
                'senal': senal_mr,
                'precio': precio_actual,
                'ratio_rr': niveles_riesgo['ratio_rr'],
                'score_alineamiento': score_alineamiento,
                'probabilidad_calidad': probabilidad_calidad,
                'n_operaciones': len(predicciones)
            }
            
            resultados_backtest.append(resultado)
    
    # ============================================
    # ANÁLISIS DE RESULTADOS DEL BACKTESTING
    # ============================================
    
    if not operaciones:
        print("\n⚠️ No se generaron operaciones en el backtesting")
        return None
    
    df_operaciones = pd.DataFrame(operaciones)
    
    print(f"\n📊 RESULTADOS DEL BACKTESTING:")
    print(f"   Período: {FECHA_INICIO_BACKTEST.date()} a {FECHA_ACTUAL.date()}")
    print(f"   Total operaciones simuladas: {len(df_operaciones)}")
    print(f"   Activos: {df_operaciones['activo'].nunique()}")
    
    # Métricas generales
    tasa_exito = df_operaciones['exito'].mean()
    retorno_promedio = df_operaciones['retorno'].mean()
    retorno_total = (1 + df_operaciones['retorno']).prod() - 1
    
    print(f"\n📈 MÉTRICAS GENERALES:")
    print(f"   Tasa de éxito: {tasa_exito:.2%}")
    print(f"   Retorno promedio por operación: {retorno_promedio:.2%}")
    print(f"   Retorno total acumulado: {retorno_total:.2%}")
    
    # Métricas por horizonte
    print(f"\n📊 MÉTRICAS POR HORIZONTE:")
    for horizonte in sorted(df_operaciones['horizonte'].unique()):
        ops_horizonte = df_operaciones[df_operaciones['horizonte'] == horizonte]
        if len(ops_horizonte) > 0:
            exito_h = ops_horizonte['exito'].mean()
            retorno_h = ops_horizonte['retorno'].mean()
            n_ops = len(ops_horizonte)
            print(f"   {horizonte}h: {n_ops} ops, Éxito={exito_h:.2%}, "
                  f"Retorno={retorno_h:.2%}")
    
    # Métricas por calidad
    print(f"\n🎯 MÉTRICAS POR CALIDAD:")
    if 'probabilidad_calidad' in df_operaciones.columns:
        df_operaciones['calidad_alta'] = df_operaciones['probabilidad_calidad'] > 0.6
        df_operaciones['calidad_media'] = (
            (df_operaciones['probabilidad_calidad'] >= 0.4) & 
            (df_operaciones['probabilidad_calidad'] <= 0.6)
        )
        df_operaciones['calidad_baja'] = df_operaciones['probabilidad_calidad'] < 0.4
        
        for calidad, label in [('calidad_alta', 'Alta'), 
                              ('calidad_media', 'Media'), 
                              ('calidad_baja', 'Baja')]:
            ops_calidad = df_operaciones[df_operaciones[calidad]]
            if len(ops_calidad) > 0:
                exito_c = ops_calidad['exito'].mean()
                retorno_c = ops_calidad['retorno'].mean()
                print(f"   Calidad {label}: {len(ops_calidad)} ops, "
                      f"Éxito={exito_c:.2%}, Retorno={retorno_c:.2%}")
    
    # Métricas de gestión de riesgo
    print(f"\n🛡️ MÉTRICAS DE GESTIÓN DE RIESGO:")
    hit_sl_rate = df_operaciones['hit_sl'].mean()
    hit_tp_rate = df_operaciones['hit_tp'].mean()
    print(f"   Tasa de Stop Loss alcanzado: {hit_sl_rate:.2%}")
    print(f"   Tasa de Take Profit alcanzado: {hit_tp_rate:.2%}")
    
    # Sharpe Ratio (simplificado)
    if len(df_operaciones) > 1:
        sharpe_ratio = df_operaciones['retorno'].mean() / df_operaciones['retorno'].std() * np.sqrt(252 * 24)
        print(f"   Sharpe Ratio (anualizado): {sharpe_ratio:.2f}")
    
    # Drawdown máximo
    retornos_acum = (1 + df_operaciones['retorno']).cumprod()
    max_acum = retornos_acum.expanding().max()
    drawdown = (retornos_acum - max_acum) / max_acum
    max_drawdown = drawdown.min()
    print(f"   Máximo drawdown: {max_drawdown:.2%}")
    
    return {
        'operaciones': df_operaciones,
        'resultados': resultados_backtest,
        'metricas': {
            'tasa_exito': tasa_exito,
            'retorno_promedio': retorno_promedio,
            'retorno_total': retorno_total,
            'hit_sl_rate': hit_sl_rate,
            'hit_tp_rate': hit_tp_rate,
            'max_drawdown': max_drawdown,
            'n_operaciones': len(df_operaciones)
        }
    }

# ============================================
# 6. SISTEMA DE TRADING EN TIEMPO REAL
# ============================================

class SistemaTrading:
    """
    Sistema de trading para ejecución en tiempo real.
    """
    
    def __init__(self, modelos_entrenados, clasificador_calidad=None):
        self.modelos = modelos_entrenados['modelos_direccion']
        self.clasificador_calidad = clasificador_calidad
        self.ultima_actualizacion = None
    
    def analizar_activo(self, simbolo_activo):
        """
        Analiza un activo en tiempo real y retorna señales si las hay.
        """
        try:
            # Descargar datos recientes
            fecha_fin = datetime.now(colombia_tz)
            fecha_inicio = fecha_fin - timedelta(days=30)  # Últimos 30 días
            
            datos = yf.download(
                simbolo_activo,
                start=fecha_inicio,
                end=fecha_fin,
                interval=INTERVALO,
                progress=False
            )
            
            if datos.empty or len(datos) < 100:
                return None
            
            # Limpiar columnas MultiIndex si es necesario
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.droplevel(1)
            
            # Calcular indicadores
            datos = calcular_indicadores_tecnicos(datos)
            
            # Obtener última vela completa
            ultima_vela = datos.iloc[-2]  # -2 porque -1 podría estar incompleta
            
            if not ultima_vela['Anomalia']:
                return None
            
            # ============================================
            # EVALUAR SEÑAL
            # ============================================
            
            precio = ultima_vela['Close']
            atr = ultima_vela['ATR']
            senal_mr = ultima_vela['Senal_MR']
            
            # Calcular niveles de riesgo
            niveles = calcular_niveles_riesgo_atr(precio, atr, senal_mr)
            
            # Filtrar por ratio R:R mínimo
            if niveles['ratio_rr'] < RATIO_MINIMO_RR:
                return None
            
            # ============================================
            # PREDICCIONES POR HORIZONTE
            # ============================================
            
            predicciones = {}
            score_alineamiento = 0
            
            for horizonte in self.modelos.keys():
                modelo_info = self.modelos[horizonte]
                # Preparar features para predicción
                features_actual = []
                for feature in FEATURES_TECNICAS:
                    if feature in ultima_vela.index:
                        features_actual.append(ultima_vela[feature])
                    else:
                        features_actual.append(0)
                
                features_actual = np.array(features_actual).reshape(1, -1)
                features_scaled = modelo_info['scaler'].transform(features_actual)
                
                # Predecir
                pred_proba = modelo_info['modelo'].predict_proba(features_scaled)[0, 1]
                prediccion = 'LONG' if pred_proba > 0.5 else 'SHORT'
                alineado = 1 if prediccion == senal_mr else 0
                
                predicciones[horizonte] = {
                    'prediccion': prediccion,
                    'probabilidad': pred_proba,
                    'alineado': alineado
                }
                
                score_alineamiento += alineado
            
            score_alineamiento = (score_alineamiento / len(predicciones)) * 100
            
            # ============================================
            # EVALUAR CALIDAD
            # ============================================
            
            probabilidad_calidad = 0.5
            
            if self.clasificador_calidad is not None and isinstance(self.clasificador_calidad, dict):
                # Preparar features para clasificador de calidad
                features_calidad_actual = []
                for feature in FEATURES_CALIDAD:
                    if feature in ultima_vela.index:
                        features_calidad_actual.append(ultima_vela[feature])
                    elif feature == 'Ratio_RR':
                        features_calidad_actual.append(niveles['ratio_rr'])
                    else:
                        features_calidad_actual.append(0)
                
                features_calidad_actual = np.array(features_calidad_actual).reshape(1, -1)
                features_scaled = self.clasificador_calidad['scaler'].transform(features_calidad_actual)
                probabilidad_calidad = self.clasificador_calidad['modelo'].predict_proba(features_scaled)[0, 1]
            
            # ============================================
            # CONSTRUIR SEÑAL
            # ============================================
            
            señal = {
                'activo': simbolo_activo,
                'fecha': ultima_vela.name,
                'senal_mr': senal_mr,
                'precio': precio,
                'atr': atr,
                'niveles_riesgo': niveles,
                'score_alineamiento': score_alineamiento,
                'probabilidad_calidad': probabilidad_calidad,
                'predicciones': predicciones,
                'intensidad_anomalia': ultima_vela['Intensidad_Anomalia']
            }
            
            return señal
            
        except Exception as e:
            print(f"❌ Error analizando {simbolo_activo}: {e}")
            return None
    
    def generar_alerta(self, señal, umbral_calidad=0.6):
        """
        Genera alerta de trading si la señal supera umbral de calidad.
        """
        if señal['probabilidad_calidad'] < umbral_calidad:
            return None
        
        mensaje = f"""🚨 *SEÑAL DE TRADING VALIDADA*

📊 *Activo:* {señal['activo']}
📅 *Fecha:* {señal['fecha'].strftime('%Y-%m-%d %H:%M')}
💰 *Precio:* ${señal['precio']:,.2f}
🎯 *Señal Mean Reversion:* {señal['senal_mr']}

🤝 *Alineamiento Modelos:* {señal['score_alineamiento']:.0f}%
✅ *Calidad Anomalía:* {señal['probabilidad_calidad']:.0%}
📊 *Intensidad Anomalía:* {señal['intensidad_anomalia']:.2f}σ

💰 *GESTIÓN DE RIESGO:*
  🛑 *Stop Loss:* ${señal['niveles_riesgo']['stop_loss']:,.2f}
  🎯 *Take Profit:* ${señal['niveles_riesgo']['take_profit']:,.2f}
  📉 *Riesgo:* ${señal['niveles_riesgo']['riesgo']:,.2f}
  📈 *Recompensa:* ${señal['niveles_riesgo']['recompensa']:,.2f}
  ⚖️ *R:R Ratio:* {señal['niveles_riesgo']['ratio_rr']:.2f}:1
  📊 *ATR:* {señal['atr']:.4f} ({señal['atr']/señal['precio']*100:.2f}%)

*PREDICCIONES POR HORIZONTE:*
"""
        
        for horizonte, pred in señal['predicciones'].items():
            icono = "✅" if pred['alineado'] else "⚠️"
            mensaje += f"  {icono} *{horizonte}h:* {pred['prediccion']} ({pred['probabilidad']:.0%})\n"
        
        # Recomendación basada en score
        if señal['score_alineamiento'] >= 70 and señal['probabilidad_calidad'] >= 0.7:
            recomendacion = "FUERTE SEÑAL - CONSIDERAR OPERACIÓN"
        elif señal['score_alineamiento'] >= 50 and señal['probabilidad_calidad'] >= 0.6:
            recomendacion = "SEÑAL MODERADA - MONITOREAR"
        else:
            recomendacion = "SEÑAL DÉBIL - ESPERAR MEJOR OPORTUNIDAD"
        
        mensaje += f"\n💡 *Recomendación:* {recomendacion}"
        
        return mensaje

# ============================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ============================================

def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("=" * 80)
    print("INICIANDO SISTEMA DE TRADING CON BACKTESTING")
    print("=" * 80)
    
    # Paso 1: Entrenar modelos con validación walk-forward
    print("\n🎯 ENTRENANDO MODELOS CON VALIDACIÓN WALK-FORWARD...")
    resultados_entrenamiento = pipeline_entrenamiento_completo()
    
    if resultados_entrenamiento is None:
        print("❌ Error en el entrenamiento. Saliendo...")
        return
    
    # Verificar que tenemos modelos entrenados
    if not resultados_entrenamiento['modelos_direccion']:
        print("❌ No se pudieron entrenar modelos de dirección. Saliendo...")
        return
    
    # Paso 2: Ejecutar backtesting de 3 meses
    print("\n🔬 EJECUTANDO BACKTESTING RIGUROSO...")
    resultados_backtest = ejecutar_backtesting(
        resultados_entrenamiento['modelos_direccion'],
        resultados_entrenamiento['clasificador_calidad']
    )
    
    # Paso 3: Evaluar si el sistema es viable
    if resultados_backtest:
        metricas = resultados_backtest['metricas']
        
        print("\n" + "=" * 80)
        print("EVALUACIÓN DE VIABILIDAD DEL SISTEMA")
        print("=" * 80)
        
        # Criterios de viabilidad
        criterios_cumplidos = 0
        total_criterios = 5
        
        # Criterio 1: Tasa de éxito > 50%
        if metricas['tasa_exito'] > 0.5:
            print(f"✅ Tasa de éxito aceptable: {metricas['tasa_exito']:.2%}")
            criterios_cumplidos += 1
        else:
            print(f"❌ Tasa de éxito baja: {metricas['tasa_exito']:.2%}")
        
        # Criterio 2: Retorno positivo
        if metricas['retorno_total'] > 0:
            print(f"✅ Retorno total positivo: {metricas['retorno_total']:.2%}")
            criterios_cumplidos += 1
        else:
            print(f"❌ Retorno total negativo: {metricas['retorno_total']:.2%}")
        
        # Criterio 3: Drawdown controlado (< 20%)
        if abs(metricas['max_drawdown']) < 0.2:
            print(f"✅ Drawdown controlado: {metricas['max_drawdown']:.2%}")
            criterios_cumplidos += 1
        else:
            print(f"❌ Drawdown excesivo: {metricas['max_drawdown']:.2%}")
        
        # Criterio 4: Suficientes operaciones
        if metricas['n_operaciones'] >= 20:
            print(f"✅ Suficientes operaciones: {metricas['n_operaciones']}")
            criterios_cumplidos += 1
        else:
            print(f"❌ Pocas operaciones: {metricas['n_operaciones']}")
        
        # Criterio 5: Hit TP > Hit SL
        if metricas['hit_tp_rate'] > metricas['hit_sl_rate']:
            print(f"✅ TP/SL favorable: TP={metricas['hit_tp_rate']:.2%}, SL={metricas['hit_sl_rate']:.2%}")
            criterios_cumplidos += 1
        else:
            print(f"❌ TP/SL desfavorable: TP={metricas['hit_tp_rate']:.2%}, SL={metricas['hit_sl_rate']:.2%}")
        
        print(f"\n📊 RESUMEN VIABILIDAD: {criterios_cumplidos}/{total_criterios} criterios cumplidos")
        
        if criterios_cumplidos >= 3:
            print("🎯 SISTEMA CONSIDERADO VIABLE - INICIANDO MONITOREO EN TIEMPO REAL")
            
            # Paso 4: Iniciar sistema en tiempo real
            sistema = SistemaTrading(
                resultados_entrenamiento,
                resultados_entrenamiento['clasificador_calidad']
            )
            
            print("\n" + "=" * 80)
            print("MONITOREO EN TIEMPO REAL")
            print("=" * 80)
            
            # Analizar cada activo
            for activo in ACTIVOS:
                print(f"\n🔍 Analizando {activo} en tiempo real...")
                señal = sistema.analizar_activo(activo)
                
                if señal:
                    alerta = sistema.generar_alerta(señal, umbral_calidad=0.6)
                    if alerta:
                        print("\n" + "=" * 80)
                        print("🚨 SEÑAL DETECTADA!")
                        print("=" * 80)
                        print(alerta)
                        
                        # Aquí podrías agregar envío a Telegram
                        enviar_telegram(alerta)
                    else:
                        print(f"✅ {activo}: Anomalía detectada pero no supera umbral de calidad")
                else:
                    print(f"✅ {activo}: Sin anomalías significativas")
        else:
            print("⚠️ SISTEMA NO VIABLE - REQUIERE AJUSTES")
            print("   Recomendaciones:")
            print("   1. Ajustar parámetros de Mean Reversion")
            print("   2. Revisar features técnicas")
            print("   3. Considerar diferentes activos")
            print("   4. Ajustar gestión de riesgo")
    else:
        print("❌ No se pudo ejecutar backtesting. Revisar datos.")

# ============================================
# 8. FUNCIÓN DE ENVÍO A TELEGRAM (OPCIONAL)
# ============================================

def enviar_telegram(mensaje, bot_token=None, chat_id=None):
    """
    Envía mensaje por Telegram.
    """
    if not bot_token or not chat_id:
        # Intentar obtener de variables de entorno
        bot_token = os.getenv('BOT_TOKEN')
        chat_id = os.getenv('CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Credenciales de Telegram no configuradas")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": mensaje,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error enviando a Telegram: {e}")
        return False

# ============================================
# 9. EJECUCIÓN DEL SISTEMA
# ============================================

if __name__ == "__main__":
    # Verificar que tenemos datos suficientes
    print(f"📅 Fecha actual: {FECHA_ACTUAL}")
    print(f"📅 Inicio backtesting: {FECHA_INICIO_BACKTEST}")
    print(f"📅 Inicio entrenamiento: {FECHA_INICIO_ENTRENAMIENTO}")
    
    # Ejecutar sistema principal
    main()
    
    print("\n" + "=" * 80)
    print("SISTEMA COMPLETADO")
    print("=" * 80)
    print("Características implementadas:")
    print("✅ Validación Walk-Forward real")
    print("✅ Logistic Regression regularizada (L2)")
    print("✅ Eliminación de métricas in-sample")
    print("✅ Sistema de scoring probabilístico continuo")
    print("✅ Arquitectura modular y escalable")
    print("✅ Backtesting de 3 meses")
    print("✅ Eliminación de look-ahead bias")
    print("✅ Gestión de riesgo basada en ATR")
    print("=" * 80)
