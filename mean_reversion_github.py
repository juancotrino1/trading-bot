# -*- coding: utf-8 -*-
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import warnings
import requests
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='yfinance')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.validation')

# ====== CONFIGURACIÓN DE TELEGRAM ======
BOT_TOKEN = "8219822992:AAFTSaawrgdPcvyZ5QJ8bxSIPJ-eEFge6cQ"
CHAT_ID = "805512543"

def enviar_telegram(mensaje):
    """Envía mensaje por Telegram"""
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Variables de Telegram no configuradas")
        return
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Mensaje enviado a Telegram")
        else:
            print(f"❌ Error enviando mensaje: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en Telegram: {e}")

def main():
    try:
        print("="*60)
        print(f"INICIO ANÁLISIS: {datetime.now()}")
        print("="*60)
        
        # 1. Configuración de parámetros
        colombia_tz = pytz.timezone('America/Bogota')
        ticker = "BTC-USD"
        end = datetime.now()
        start = end - timedelta(days=365)
        interval = "1h"
        window = 50
        k = 2.5
        
        # 2. Descarga de datos con reintentos
        max_retries = 3
        retry_delay_seconds = 60
        
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
                if not data.empty:
                    break
                else:
                    print(f"Intento {attempt + 1}: No hay datos")
            except Exception as e:
                print(f"Intento {attempt + 1}: Error: {e}")
            
            if attempt < max_retries - 1:
                print(f"Esperando {retry_delay_seconds}s...")
                time.sleep(retry_delay_seconds)
            else:
                print("Max reintentos alcanzado")
                data = pd.DataFrame()
        
        if data.empty:
            enviar_telegram("❌ No se pudo descargar datos de BTC-USD")
            return
        
        # Procesar datos
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            data.columns.name = None
        
        data = data[['Open', 'High', 'Low', 'Close']].dropna()
        data.index = data.index.tz_convert(colombia_tz)
        
        # 3. Cálculo de métricas
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        effective_window = min(window, len(data) // 2) if len(data) > 0 else 1
        
        data['mean'] = data['log_return'].rolling(effective_window).mean()
        data['std'] = data['log_return'].rolling(effective_window).std()
        data['upper_band'] = data['mean'] + k * data['std']
        data['lower_band'] = data['mean'] - k * data['std']
        
        # Identificación de anomalías
        data['anomaly'] = (data['log_return'] > data['upper_band']) | (data['log_return'] < data['lower_band'])
        data['signal'] = np.where(data['log_return'] < data['lower_band'], 'Long',
                                np.where(data['log_return'] > data['upper_band'], 'Short', None))
        
        # Features para ML
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        
        df_features = data.copy()
        
        # Crear features
        delta = df_features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_features['RSI'] = 100 - (100 / (1 + rs))
        df_features['Volatility'] = df_features['log_return'].rolling(window=24).std()
        df_features['Dist_from_MA'] = (df_features['Close'] - df_features['Close'].rolling(window=50).mean()) / df_features['Close'].rolling(window=50).mean()
        df_features['HL_Range'] = (df_features['High'] - df_features['Low']) / df_features['Close']
        df_features['Log_Return_Accel'] = df_features['log_return'].diff()
        df_features['RSI_lag1'] = df_features['RSI'].shift(1)
        df_features['Volatility_lag1'] = df_features['Volatility'].shift(1)
        df_features['ADX'] = df_features['Close'].rolling(14).std() / df_features['Close'].rolling(50).std()
        df_features['BB_upper'] = df_features['Close'].rolling(20).mean() + 2 * df_features['Close'].rolling(20).std()
        df_features['BB_lower'] = df_features['Close'].rolling(20).mean() - 2 * df_features['Close'].rolling(20).std()
        df_features['BB_position'] = (df_features['Close'] - df_features['BB_lower']) / (df_features['BB_upper'] - df_features['BB_lower'])
        
        df_features = df_features.dropna()
        
        # Crear targets
        horizontes = [4, 8, 12, 24, 48]
        for h in horizontes:
            future_return = df_features['Close'].shift(-h) / df_features['Close'] - 1
            df_features[f'target_{h}h'] = (future_return > 0).astype(int)
        
        df_features = df_features.dropna()
        
        # Entrenar modelos (versión simplificada para velocidad)
        features_list = ['RSI', 'Volatility', 'Dist_from_MA', 'HL_Range', 'Log_Return_Accel',
                        'RSI_lag1', 'Volatility_lag1', 'ADX', 'BB_position']
        
        X = df_features[features_list]
        mejores_modelos = {}
        scalers = {}
        metricas_detalladas = {}
        
        print("\n🔄 Entrenando modelos...")
        for h in horizontes:
            y = df_features[f'target_{h}h']
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Modelo simple KNN para velocidad
            modelo = KNeighborsClassifier(n_neighbors=7, weights='distance')
            modelo.fit(X_train_scaled, y_train)
            
            # Re-entrenar con todos los datos
            scaler_final = StandardScaler()
            X_scaled_full = scaler_final.fit_transform(X)
            modelo_final = KNeighborsClassifier(n_neighbors=7, weights='distance')
            modelo_final.fit(X_scaled_full, y)
            
            mejores_modelos[h] = modelo_final
            scalers[h] = scaler_final
            
            # Calcular métricas básicas
            baseline = y.value_counts().max() / len(y)
            metricas_detalladas[h] = {
                'baseline': baseline,
                'mejora_vs_baseline': 0.02,  # Placeholder
                'test_accuracy': 0.52  # Placeholder
            }
        
        print("✅ Modelos entrenados")
        
        # Analizar última vela
        ultima_fila = df_features.tail(1)
        es_anomalia_actual = ultima_fila['anomaly'].values[0]
        fecha_actual = ultima_fila.index[0]
        precio_actual = ultima_fila['Close'].values[0]
        
        # Generar mensaje para Telegram
        mensaje_telegram_final = f"""
📊 *Análisis de Mercado Actual*

📅 Fecha/Hora: {fecha_actual.strftime('%Y-%m-%d %H:%M')} COT
💰 Precio: ${precio_actual:,.2f}
🎯 Estado: {'🚨 ANOMALÍA DETECTADA' if es_anomalia_actual else '✅ Mercado Normal'}
"""
        
        if es_anomalia_actual:
            signal_mr = ultima_fila['signal'].values[0]
            log_return = ultima_fila['log_return'].values[0]
            
            X_actual = ultima_fila[features_list]
            predicciones = {}
            
            for h in horizontes:
                X_actual_scaled = scalers[h].transform(X_actual)
                pred = mejores_modelos[h].predict(X_actual_scaled)[0]
                prob = mejores_modelos[h].predict_proba(X_actual_scaled)[0]
                
                tendencia = "ALCISTA" if pred == 1 else "BAJISTA"
                confianza = prob[pred]
                
                nivel_confianza = "🟢" if confianza >= 0.65 else "🟡" if confianza >= 0.55 else "🔴"
                
                normalized_signal_mr = "ALCISTA" if signal_mr == "Long" else "BAJISTA"
                compatible = (tendencia == normalized_signal_mr)
                
                predicciones[h] = {
                    "Predicción": tendencia,
                    "Confianza": confianza,
                    "Nivel": nivel_confianza,
                    "vs MeanRev": "✅" if compatible else "⚠️"
                }
            
            # Calcular MR_Alignment_Score
            alignment_scores = [1 if p["vs MeanRev"] == "✅" else 0 for p in predicciones.values()]
            current_mr_alignment_score = (sum(alignment_scores) / len(alignment_scores)) * 100
            
            # Categorizar
            if current_mr_alignment_score >= 81:
                categoria = "🟢Excelente"
            elif current_mr_alignment_score >= 61:
                categoria = "🟡Bueno"
            elif current_mr_alignment_score >= 41:
                categoria = "🟠Medio"
            elif current_mr_alignment_score >= 21:
                categoria = "🔴Bajo"
            else:
                categoria = "⚪Basura"
            
            mensaje_telegram_final += f"""
🎯 Señal Mean Reversion: {signal_mr}
🤝 Alineamiento KNN-MR: {current_mr_alignment_score:.0f}% ({categoria})

*PREDICCIONES POR HORIZONTE:*
"""
            
            for h, p_data in predicciones.items():
                mensaje_telegram_final += f"  *{h}h:* {p_data['Predicción']} {p_data['Nivel']} ({p_data['Confianza']:.0%}) {p_data['vs MeanRev']}\n"
            
            # Consenso
            predicciones_alcistas = sum(1 for p in predicciones.values() if p["Predicción"] == "ALCISTA")
            consenso_pct = (predicciones_alcistas / len(horizontes)) * 100
            
            if consenso_pct >= 60:
                recomendacion = "✅ CONSIDERAR OPERAR"
            elif consenso_pct >= 40:
                recomendacion = "⚠️ NEUTRAL - ESPERAR"
            else:
                recomendacion = "⚠️ PRECAUCIÓN"
            
            mensaje_telegram_final += f"\n💡 *Recomendación:* {recomendacion}\n"
            
        else:
            mensaje_telegram_final += "\n💡 Mercado en condiciones normales. Sin señales de trading.\n"
        
        # Enviar mensaje
        enviar_telegram(mensaje_telegram_final)
        print("✅ Análisis completado exitosamente")
        
    except Exception as e:
        error_msg = f"❌ *ERROR EN SCRIPT*\n\n```\n{str(e)}\n```"
        enviar_telegram(error_msg)
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
