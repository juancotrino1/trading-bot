# -*- coding: utf-8 -*-
"""
Sistema Integrado de Trading Multi-Activo con Gestión de Riesgo ATR
- Mean Reversion para detección de anomalías
- KNN para predicción de dirección por horizonte
- Gradient Boosting para clasificación de calidad de anomalías
- Stop Loss y Take Profit basados en ATR
- Filtro de R:R ≥ 1.7
- Alertas por Telegram solo cuando hay señales válidas
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='yfinance')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.validation')

# ============================================
# 1. CONFIGURACIÓN GLOBAL DEL SISTEMA
# ============================================

print("="*80)
print("SISTEMA INTEGRADO DE TRADING MULTI-ACTIVO CON GESTIÓN DE RIESGO ATR")
print("="*80)

# Parámetros de datos
colombia_tz = pytz.timezone('America/Bogota')
end = datetime.now()
#end = datetime(2026, 1, 21, 16, 0, tzinfo=colombia_tz)
start = end - timedelta(days=365)
interval = "1h"

# Parámetros de Mean Reversion
window = 60
k = 2.5

# Parámetros de gestión de riesgo
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5  # Stop Loss = 1.5 * ATR
ATR_TP_MULTIPLIER = 2.55  # Take Profit = 2.55 * ATR (para R:R ≥ 1.7)
MIN_RR_RATIO = 1.7  # Mínimo ratio riesgo-recompensa

# Parámetros de descarga
max_retries = 3
retry_delay_seconds = 60

# Lista de activos a monitorear
tickers_monitorear = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "XRP-USD"
]

# Horizontes de predicción (en horas)
horizontes = [4, 8, 12, 24, 48]

# Features para los modelos
features_list = [
    'RSI',
    'Volatility',
    'Dist_from_MA',
    'HL_Range',
    'Log_Return_Accel',
    'RSI_lag1',
    'Volatility_lag1',
    'ADX',
    'BB_position',
    'ATR',
    'ATR_pct'
]

# Features para clasificador de calidad
features_calidad = [
    'RSI',
    'Volatility',
    'Dist_from_MA',
    'HL_Range',
    'Log_Return_Accel',
    'ADX',
    'BB_position',
    'ATR',
    'ATR_pct',
    'MR_Alignment_Score',
    'log_return',
    'RR_Ratio'
]

# Umbral de calidad para anomalías
umbral_calidad = 0.6

print(f"\n📋 Configuración:")
print(f"   Activos: {len(tickers_monitorear)}")
print(f"   Período: {start.date()} a {end.date()}")
print(f"   Intervalo: {interval}")
print(f"   Horizontes: {horizontes}")
print(f"   Umbral calidad: {umbral_calidad*100:.0f}%")
print(f"\n💰 Gestión de Riesgo:")
print(f"   ATR Period: {ATR_PERIOD}")
print(f"   Stop Loss: {ATR_SL_MULTIPLIER} x ATR")
print(f"   Take Profit: {ATR_TP_MULTIPLIER} x ATR")
print(f"   R:R Mínimo: {MIN_RR_RATIO}:1")

# ============================================
# 2. FUNCIÓN PARA CALCULAR ATR
# ============================================

def calcular_atr(df, period=ATR_PERIOD):
    """Calcula Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

# ============================================
# 3. FUNCIÓN PARA CALCULAR NIVELES SL/TP
# ============================================

def calcular_niveles_riesgo(precio_entrada, atr, signal_type):
    """
    Calcula niveles de Stop Loss y Take Profit basados en ATR

    Parámetros:
    - precio_entrada: precio de entrada
    - atr: valor del ATR
    - signal_type: 'LONG' o 'SHORT'

    Retorna:
    - dict con stop_loss, take_profit, riesgo, recompensa, rr_ratio
    """

    if signal_type == 'LONG':
        stop_loss = precio_entrada - (ATR_SL_MULTIPLIER * atr)
        take_profit = precio_entrada + (ATR_TP_MULTIPLIER * atr)
    else:  # SHORT
        stop_loss = precio_entrada + (ATR_SL_MULTIPLIER * atr)
        take_profit = precio_entrada - (ATR_TP_MULTIPLIER * atr)

    riesgo = abs(precio_entrada - stop_loss)
    recompensa = abs(take_profit - precio_entrada)
    rr_ratio = recompensa / riesgo if riesgo > 0 else 0

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'riesgo': riesgo,
        'recompensa': recompensa,
        'rr_ratio': rr_ratio,
        'atr_value': atr
    }

# ============================================
# 4. FUNCIÓN PARA CALCULAR FEATURES COMPLETAS
# ============================================

def calcular_features_completas(data_raw):
    """Calcula todas las features necesarias para un dataset"""
    df = data_raw.copy()

    # Mean Reversion
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    effective_window_calc = min(window, len(df) // 2) if len(df) > 0 else 1

    df['mean'] = df['log_return'].rolling(effective_window_calc).mean()
    df['std'] = df['log_return'].rolling(effective_window_calc).std()
    df['upper_band'] = df['mean'] + k * df['std']
    df['lower_band'] = df['mean'] - k * df['std']

    df['anomaly'] = (df['log_return'] > df['upper_band']) | (df['log_return'] < df['lower_band'])
    df['signal'] = np.where(df['log_return'] < df['lower_band'], 'Long', 'Short')

    # ATR
    df['ATR'] = calcular_atr(df, ATR_PERIOD)
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100  # ATR como % del precio

    # Features técnicas
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Volatility'] = df['log_return'].rolling(window=24).std()
    df['Dist_from_MA'] = (df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=50).mean()
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Log_Return_Accel'] = df['log_return'].diff()
    df['RSI_lag1'] = df['RSI'].shift(1)
    df['Volatility_lag1'] = df['Volatility'].shift(1)
    df['ADX'] = df['Close'].rolling(14).std() / df['Close'].rolling(50).std()

    df['BB_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    return df.dropna()

# ============================================
# 5. DESCARGA DE DATOS DE TODOS LOS ACTIVOS
# ============================================

print("\n" + "="*80)
print("FASE 1: DESCARGA DE DATOS DE TODOS LOS ACTIVOS")
print("="*80)

datos_todos_activos = []

for ticker in tickers_monitorear:
    print(f"\n📊 Descargando {ticker}...")

    data_ticker = pd.DataFrame()
    for attempt in range(max_retries):
        try:
            data_ticker = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not data_ticker.empty:
                print(f"   ✅ {len(data_ticker)} registros descargados")
                break
            else:
                print(f"   ⚠️ Intento {attempt + 1}: Sin datos")
        except Exception as e:
            print(f"   ❌ Intento {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay_seconds)

    if data_ticker.empty:
        print(f"   ⚠️ Saltando {ticker} - no hay datos")
        continue

    # Procesar
    if isinstance(data_ticker.columns, pd.MultiIndex):
        data_ticker.columns = data_ticker.columns.droplevel(1)
        data_ticker.columns.name = None

    data_ticker = data_ticker[['Open', 'High', 'Low', 'Close']].dropna()
    data_ticker.index = data_ticker.index.tz_convert(colombia_tz)

    # Agregar columna de ticker
    data_ticker['ticker'] = ticker

    datos_todos_activos.append(data_ticker)
    time.sleep(1)

if len(datos_todos_activos) == 0:
    print("❌ ERROR CRÍTICO: No se descargaron datos de ningún activo")
    exit(1)

# Combinar todos los datos
data_combinada = pd.concat(datos_todos_activos, axis=0)
print(f"\n✅ Total de registros combinados: {len(data_combinada)}")
print(f"   Activos con datos: {len(datos_todos_activos)}")

# ============================================
# 6. CALCULAR FEATURES PARA CADA ACTIVO
# ============================================

print("\n" + "="*80)
print("FASE 2: CÁLCULO DE FEATURES POR ACTIVO")
print("="*80)

df_features_list = []

for ticker in tickers_monitorear:
    data_ticker = data_combinada[data_combinada['ticker'] == ticker].copy()

    if len(data_ticker) == 0:
        continue

    print(f"\n🔧 Procesando features de {ticker}...")

    # Calcular features
    df_ticker_features = calcular_features_completas(data_ticker)

    # Calcular R:R ratio para cada anomalía
    rr_ratios = []
    for idx, row in df_ticker_features.iterrows():
        if row['anomaly']:
            signal_type = 'LONG' if row['signal'] == 'Long' else 'SHORT'
            niveles = calcular_niveles_riesgo(row['Close'], row['ATR'], signal_type)
            rr_ratios.append(niveles['rr_ratio'])
        else:
            rr_ratios.append(np.nan)

    df_ticker_features['RR_Ratio'] = rr_ratios

    # Agregar targets
    for h in horizontes:
        future_return = df_ticker_features['Close'].shift(-h) / df_ticker_features['Close'] - 1
        df_ticker_features[f'target_{h}h'] = (future_return > 0).astype(int)

    df_ticker_features = df_ticker_features.dropna()
    df_ticker_features['ticker'] = ticker

    print(f"   ✅ {len(df_ticker_features)} registros válidos")
    print(f"   📊 Anomalías: {df_ticker_features['anomaly'].sum()}")

    # Estadísticas de R:R
    anomalias_con_rr = df_ticker_features[df_ticker_features['anomaly']]
    if len(anomalias_con_rr) > 0:
        rr_promedio = anomalias_con_rr['RR_Ratio'].mean()
        rr_validas = (anomalias_con_rr['RR_Ratio'] >= MIN_RR_RATIO).sum()
        print(f"   💰 R:R Promedio: {rr_promedio:.2f}")
        print(f"   ✅ Anomalías con R:R ≥ {MIN_RR_RATIO}: {rr_validas}/{len(anomalias_con_rr)}")

    df_features_list.append(df_ticker_features)

# Combinar todos los features
df_features_todos = pd.concat(df_features_list, axis=0)

print(f"\n✅ Dataset completo de entrenamiento:")
print(f"   Total registros: {len(df_features_todos)}")
print(f"   Total anomalías: {df_features_todos['anomaly'].sum()}")
print(f"   Anomalías con R:R ≥ {MIN_RR_RATIO}: {(df_features_todos['RR_Ratio'] >= MIN_RR_RATIO).sum()}")
print(f"   Activos incluidos: {df_features_todos['ticker'].nunique()}")

# ============================================
# 7. ENTRENAMIENTO DE MODELOS KNN
# ============================================

print("\n" + "="*80)
print("FASE 3: ENTRENAMIENTO DE MODELOS KNN")
print("="*80)

mejores_modelos = {}
scalers = {}
metricas_detalladas = {}

# Parámetros para Grid Search
param_grid_knn = {
    'n_neighbors': [5, 7, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 20]
}

X = df_features_todos[features_list]

for h in horizontes:
    print(f"\n{'='*80}")
    print(f"HORIZONTE: {h} HORAS")
    print(f"{'='*80}")

    y = df_features_todos[f'target_{h}h']

    # Split temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Probar KNN
    print("🔍 Grid Search KNN...")
    grid_knn = GridSearchCV(
        KNeighborsClassifier(),
        param_grid_knn,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_knn.fit(X_train_scaled, y_train)

    # Probar Random Forest
    print("🌲 Grid Search Random Forest...")
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_rf.fit(X_train_scaled, y_train)

    # Comparar modelos
    knn_test_score = grid_knn.score(X_test_scaled, y_test)
    rf_test_score = grid_rf.score(X_test_scaled, y_test)

    # Seleccionar mejor modelo
    if knn_test_score >= rf_test_score:
        mejor_modelo = grid_knn.best_estimator_
        modelo_nombre = "KNN"
        mejores_params = grid_knn.best_params_
        cv_score = grid_knn.best_score_
    else:
        mejor_modelo = grid_rf.best_estimator_
        modelo_nombre = "Random Forest"
        mejores_params = grid_rf.best_params_
        cv_score = grid_rf.best_score_

    print(f"🏆 Ganador: {modelo_nombre}")
    print(f"   Parámetros: {mejores_params}")

    # Métricas detalladas
    y_pred = mejor_modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    baseline = y_test.value_counts().max() / len(y_test)

    print(f"   Accuracy: {accuracy:.2%} | Baseline: {baseline:.2%} | Mejora: {(accuracy-baseline)*100:+.2f}%")

    # Re-entrenar con todos los datos
    scaler_final = StandardScaler()
    X_scaled_full = scaler_final.fit_transform(X)

    if modelo_nombre == "KNN":
        modelo_final = KNeighborsClassifier(**mejores_params)
    else:
        modelo_final = RandomForestClassifier(**mejores_params, random_state=42)

    modelo_final.fit(X_scaled_full, y)

    # Guardar
    mejores_modelos[h] = modelo_final
    scalers[h] = scaler_final
    metricas_detalladas[h] = {
        'modelo': modelo_nombre,
        'params': mejores_params,
        'cv_score': cv_score,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'baseline': baseline,
        'mejora_vs_baseline': accuracy - baseline
    }

print("\n✅ Modelos KNN entrenados para todos los horizontes con datos de TODOS los activos")

# ============================================
# 8. ANÁLISIS HISTÓRICO DE ANOMALÍAS
# ============================================

print("\n" + "="*80)
print("FASE 4: ANÁLISIS HISTÓRICO DE ANOMALÍAS")
print("="*80)

anomalias_historicas = df_features_todos[df_features_todos['anomaly']].copy()
print(f"📊 Anomalías históricas: {len(anomalias_historicas)}")

# Generar predicciones para todas las anomalías
for h in horizontes:
    predicciones_h = []
    for idx, row in anomalias_historicas.iterrows():
        X_anomalia = row[features_list].values.reshape(1, -1)
        X_scaled = scalers[h].transform(X_anomalia)
        pred = mejores_modelos[h].predict(X_scaled)[0]
        label = "ALCISTA" if pred == 1 else "BAJISTA"
        predicciones_h.append(label)
    anomalias_historicas[f'KNN_{h}h'] = predicciones_h

# Normalizar señal Mean Reversion
anomalias_historicas['signal'] = np.where(
    anomalias_historicas['signal'] == "Long", "LONG", "SHORT"
)
anomalias_historicas['signal_dir'] = np.where(
    anomalias_historicas['signal'] == "LONG", "ALCISTA", "BAJISTA"
)

# Calcular MR_Alignment_Score
alignment_checks = pd.DataFrame()
for h in horizontes:
    alignment_checks[f'Align_{h}h'] = (
        anomalias_historicas[f'KNN_{h}h'] == anomalias_historicas['signal_dir']
    ).astype(int)

anomalias_historicas['MR_Alignment_Score'] = alignment_checks.mean(axis=1) * 100

# Verificar predicciones vs realidad
for h in horizontes:
    pred = anomalias_historicas[f'KNN_{h}h']
    target = anomalias_historicas[f'target_{h}h']

    anomalias_historicas[f'Check_{h}h'] = np.where(
        ((pred == "ALCISTA") & (target == 1)) |
        ((pred == "BAJISTA") & (target == 0)),
        "✅", "❌"
    )

# Filtrar por R:R ratio
anomalias_validas_rr = anomalias_historicas[anomalias_historicas['RR_Ratio'] >= MIN_RR_RATIO].copy()
print(f"✅ Anomalías con R:R ≥ {MIN_RR_RATIO}: {len(anomalias_validas_rr)}/{len(anomalias_historicas)}")

print("✅ Análisis histórico completado")

# ============================================
# 9. ENTRENAMIENTO DE CLASIFICADOR DE CALIDAD
# ============================================

print("\n" + "="*80)
print("FASE 5: ENTRENAMIENTO DE CLASIFICADOR DE CALIDAD")
print("="*80)

# Preparar dataset de calidad (solo con anomalías que pasan filtro R:R)
reporte_final = anomalias_validas_rr.copy()
checks_columns = [f'Check_{h}h' for h in horizontes]
reporte_final['aciertos_count'] = (reporte_final[checks_columns] == "✅").sum(axis=1)
reporte_final['tasa_acierto'] = reporte_final['aciertos_count'] / len(horizontes)
reporte_final['es_buena_anomalia'] = (reporte_final['tasa_acierto'] >= umbral_calidad).astype(int)

print(f"📊 Distribución de calidad:")
print(f"   Buenas (≥{umbral_calidad*100:.0f}%): {reporte_final['es_buena_anomalia'].sum()}")
print(f"   Malas (<{umbral_calidad*100:.0f}%): {len(reporte_final) - reporte_final['es_buena_anomalia'].sum()}")

# Verificar si hay suficientes datos
X_calidad = reporte_final[features_calidad]
y_calidad = reporte_final['es_buena_anomalia']

usar_clasificador_calidad = False

if y_calidad.sum() >= 5 and (len(y_calidad) - y_calidad.sum()) >= 5:
    print("\n🎯 Entrenando Gradient Boosting Classifier...")

    modelo_calidad = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    cv_scores_calidad = cross_val_score(
        modelo_calidad,
        X_calidad,
        y_calidad,
        cv=min(5, len(X_calidad)//2),
        scoring='accuracy'
    )

    print(f"✅ CV Accuracy: {cv_scores_calidad.mean():.2%} (+/- {cv_scores_calidad.std():.2%})")

    modelo_calidad.fit(X_calidad, y_calidad)

    baseline_calidad = y_calidad.value_counts().max() / len(y_calidad)
    mejora_calidad = cv_scores_calidad.mean() - baseline_calidad

    print(f"📈 Baseline: {baseline_calidad:.2%} | Mejora: {mejora_calidad:+.2%}")

    usar_clasificador_calidad = True
else:
    print("⚠️ Datos insuficientes para entrenar clasificador de calidad")
    print("   El sistema operará solo con filtro R:R")

# ============================================
# 10. FUNCIÓN DE ANÁLISIS PARA UN ACTIVO
# ============================================

def analizar_activo(ticker_symbol):
    """Analiza un activo y retorna señal si detecta anomalía válida"""
    print(f"\n{'='*80}")
    print(f"Analizando: {ticker_symbol}")
    print(f"{'='*80}")

    try:
        # Descargar datos
        data_activo = yf.download(ticker_symbol, start=start, end=end, interval=interval, progress=False)

        if data_activo.empty:
            print(f"⚠️ No hay datos disponibles")
            return None

        # Procesar
        if isinstance(data_activo.columns, pd.MultiIndex):
            data_activo.columns = data_activo.columns.droplevel(1)
            data_activo.columns.name = None

        data_activo = data_activo[['Open', 'High', 'Low', 'Close']].dropna()
        data_activo.index = data_activo.index.tz_convert(colombia_tz)

        # Calcular features
        data_activo = calcular_features_completas(data_activo)

        # Verificar última vela
        ultima_vela = data_activo.tail(1)

        if not ultima_vela['anomaly'].values[0]:
            print(f"✅ Mercado normal - Sin anomalías")
            return None

        # HAY ANOMALÍA
        fecha = ultima_vela.index[0]
        precio = ultima_vela['Close'].values[0]
        signal_mr = ultima_vela['signal'].values[0]
        log_ret = ultima_vela['log_return'].values[0]
        atr = ultima_vela['ATR'].values[0]
        atr_pct = ultima_vela['ATR_pct'].values[0]

        print(f"🚨 ANOMALÍA DETECTADA")
        print(f"   Precio: ${precio:,.2f}")
        print(f"   Señal MR: {signal_mr}")
        print(f"   ATR: {atr:.4f} ({atr_pct:.2f}%)")

        # Calcular niveles de riesgo
        signal_type = 'LONG' if signal_mr == 'Long' else 'SHORT'
        niveles_riesgo = calcular_niveles_riesgo(precio, atr, signal_type)

        print(f"\n💰 NIVELES DE GESTIÓN DE RIESGO:")
        print(f"   Stop Loss: ${niveles_riesgo['stop_loss']:,.2f}")
        print(f"   Take Profit: ${niveles_riesgo['take_profit']:,.2f}")
        print(f"   Riesgo: ${niveles_riesgo['riesgo']:,.2f}")
        print(f"   Recompensa: ${niveles_riesgo['recompensa']:,.2f}")
        print(f"   R:R Ratio: {niveles_riesgo['rr_ratio']:.2f}:1")

        # FILTRO R:R
        if niveles_riesgo['rr_ratio'] < MIN_RR_RATIO:
            print(f"   ⛔ R:R insuficiente (< {MIN_RR_RATIO}:1) - Anomalía descartada")
            return None

        print(f"   ✅ R:R válido (≥ {MIN_RR_RATIO}:1)")

        # Predicciones KNN
        X_actual = ultima_vela[features_list]
        predicciones_activo = {}

        for h in horizontes:
            X_scaled = scalers[h].transform(X_actual)
            pred = mejores_modelos[h].predict(X_scaled)[0]
            prob = mejores_modelos[h].predict_proba(X_scaled)[0]

            tendencia = "ALCISTA" if pred == 1 else "BAJISTA"
            confianza = prob[pred]

            normalized_signal = "ALCISTA" if signal_mr == "Long" else "BAJISTA"
            compatible = (tendencia == normalized_signal)

            predicciones_activo[h] = {
                "prediccion": tendencia,
                "confianza": confianza,
                "compatible": compatible
            }

        # Calcular MR_Alignment_Score
        alignment_count = sum(1 for p in predicciones_activo.values() if p["compatible"])
        mr_alignment_score = (alignment_count / len(horizontes)) * 100

        print(f"   Alineamiento KNN-MR: {mr_alignment_score:.0f}%")

        # Clasificador de calidad
        if usar_clasificador_calidad:
            features_disponibles = [f for f in features_calidad if f in ultima_vela.columns]
            X_calidad_activo = ultima_vela[features_disponibles].copy()
            # Agregar las features calculadas manualmente
            X_calidad_activo['MR_Alignment_Score'] = mr_alignment_score
            X_calidad_activo['RR_Ratio'] = niveles_riesgo['rr_ratio']
    
            # Asegurarse de que las columnas estén en el orden correcto
            X_calidad_activo = X_calidad_activo[features_calidad]
    

            es_buena = modelo_calidad.predict(X_calidad_activo)[0]
            prob_buena = modelo_calidad.predict_proba(X_calidad_activo)[0][1]

            print(f"   Calidad: {'✅ BUENA' if es_buena else '❌ MALA'} ({prob_buena:.0%})")

            if not es_buena:
                print(f"   ⛔ Anomalía descartada por clasificador de calidad")
                return None
        else:
            prob_buena = None

        # Señal válida
        return {
            'ticker': ticker_symbol,
            'fecha': fecha,
            'precio': precio,
            'signal_mr': signal_mr,
            'log_return': log_ret,
            'mr_alignment_score': mr_alignment_score,
            'predicciones': predicciones_activo,
            'prob_calidad': prob_buena,
            'niveles_riesgo': niveles_riesgo
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================
# 11. FUNCIÓN DE ENVÍO A TELEGRAM
# ============================================

def enviar_telegram(mensaje):
    """Envía mensaje por Telegram"""
    # Descomentar y configurar tus credenciales
    #bot_token = os.getenv('BOT_TOKEN')
    #chat_id = os.getenv('CHAT_ID')

    # Para testing, comentar estas líneas:
    bot_token = os.getenv('BOT_TOKEN')
    chat_id = os.getenv('CHAT_ID')

    if not bot_token or not chat_id:
        print("⚠️ BOT_TOKEN o CHAT_ID no configurados - Mensaje no enviado")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "Markdown"
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        print("✅ Mensaje enviado a Telegram")
    else:
        print(f"❌ Error enviando mensaje: {response.text}")

# ============================================
# 12. ESCANEO MULTI-ACTIVO Y ALERTAS
# ============================================

print("\n" + "="*80)
print("FASE 6: ESCANEO MULTI-ACTIVO EN TIEMPO REAL")
print("="*80)

señales_validas = []

for ticker_symbol in tickers_monitorear:
    resultado = analizar_activo(ticker_symbol)
    if resultado is not None:
        señales_validas.append(resultado)
    time.sleep(1)

# ============================================
# 13. ENVÍO DE ALERTAS (SOLO SI HAY SEÑALES)
# ============================================

if len(señales_validas) > 0:
    print(f"\n{'='*80}")
    print(f"🎯 {len(señales_validas)} SEÑALES VÁLIDAS ENCONTRADAS")
    print(f"{'='*80}")

    for señal in señales_validas:
        niveles = señal['niveles_riesgo']

        mensaje = f"""
🚨 *SEÑAL DE TRADING VALIDADA*

📊 *Activo:* {señal['ticker']}
📅 *Fecha:* {señal['fecha'].strftime('%Y-%m-%d %H:%M')}
💰 *Precio:* ${señal['precio']:,.2f}
🎯 *Mean Reversion:* {señal['signal_mr']}
🤝 *Alineamiento:* {señal['mr_alignment_score']:.0f}%
"""

        if señal['prob_calidad'] is not None:
            mensaje += f"✅ *Calidad:* {señal['prob_calidad']:.0%}\n"

        mensaje += f"""
💰 *GESTIÓN DE RIESGO:*
  🛑 *Stop Loss:* ${niveles['stop_loss']:,.2f}
  🎯 *Take Profit:* ${niveles['take_profit']:,.2f}
  📉 *Riesgo:* ${niveles['riesgo']:,.2f}
  📈 *Recompensa:* ${niveles['recompensa']:,.2f}
  ⚖️ *R:R Ratio:* {niveles['rr_ratio']:.2f}:1
  📊 *ATR:* {niveles['atr_value']:.4f}

*PREDICCIONES POR HORIZONTE:*
"""

        for h, pred_data in señal['predicciones'].items():
            icono = "✅" if pred_data['compatible'] else "⚠️"
            mensaje += f"  {icono} *{h}h:* {pred_data['prediccion']} ({pred_data['confianza']:.0%})\n"

        recomendacion = "OPERAR" if señal['mr_alignment_score'] >= 60 else "PRECAUCIÓN"
        mensaje += f"\n💡 *Recomendación:* {recomendacion}"

        # Calcular tamaño de posición sugerido (ejemplo con $1000)
        capital_ejemplo = 1000
        riesgo_pct = 0.02  # 2% de riesgo por operación
        riesgo_dolares = capital_ejemplo * riesgo_pct
        tamaño_posicion = riesgo_dolares / niveles['riesgo']

        # Enviar mensaje
        enviar_telegram(mensaje)
        print(f"\n{'='*80}")
        print(mensaje)
        print(f"{'='*80}")
        print(f"✅ Alerta generada para {señal['ticker']}")
        time.sleep(2)

else:
    print("\n✅ No se detectaron anomalías válidas en ningún activo")
    print("   Sistema en silencio - No hay alertas que enviar")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print(f"📊 Resumen:")
print(f"   Activos escaneados: {len(tickers_monitorear)}")
print(f"   Señales válidas: {len(señales_validas)}")
print(f"   Modelos entrenados con {df_features_todos['ticker'].nunique()} activos")
print(f"   Total de anomalías históricas: {len(anomalias_historicas)}")
print(f"   Anomalías con R:R ≥ {MIN_RR_RATIO}: {len(anomalias_validas_rr)}")
print(f"   Filtro R:R aplicado: ✅")
print("="*80)
