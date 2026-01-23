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
warnings.filterwarnings("ignore")

# ====== CONFIGURACIÓN DE TELEGRAM ======
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

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
            return True
        else:
            print(f"❌ Error enviando mensaje: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en Telegram: {e}")
        return False

def descargar_datos(ticker, start, end, interval, max_retries=5):
    """Descarga datos con múltiples reintentos y estrategias alternativas"""
    
    # Estrategia 1: Ticker normal con reintentos
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Intento {attempt + 1}/{max_retries} descargando {ticker}...")
            
            # Crear objeto Ticker explícitamente
            btc = yf.Ticker(ticker)
            data = btc.history(start=start, end=end, interval=interval)
            
            if not data.empty:
                print(f"✅ Datos descargados: {len(data)} registros")
                return data
            else:
                print(f"⚠️ Datos vacíos en intento {attempt + 1}")
        except Exception as e:
            print(f"❌ Error en intento {attempt + 1}: {str(e)[:100]}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 15  # Espera progresiva: 15s, 30s, 45s...
            print(f"⏳ Esperando {wait_time}s antes de reintentar...")
            time.sleep(wait_time)
    
    return pd.DataFrame()

def main():
    try:
        print("="*60)
        print(f"🤖 INICIO ANÁLISIS: {datetime.now()}")
        print("="*60)
        
        # 1. Configuración de parámetros
        colombia_tz = pytz.timezone('America/Bogota')
        ticker = "BTC-USD"
        end = datetime.now()
        start = end - timedelta(days=365)
        interval = "1h"
        window = 50
        k = 2.5
        
        # 2. Descarga de datos con estrategia robusta
        data = descargar_datos(ticker, start, end, interval)
        
        if data.empty:
            error_msg = f"❌ *Error de descarga*\n\nNo se pudieron obtener datos de {ticker} después de múltiples intentos.\n\nPosibles causas:\n• Yahoo Finance temporalmente inaccesible\n• Problemas de red\n• Ticker delisted\n\nSe reintentará en la próxima ejecución."
            enviar_telegram(error_msg)
            print("❌ No se pudo descargar datos después de todos los intentos")
            return
        
        # Procesar datos
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            data.columns.name = None
        
        # Verificar que tenemos las columnas necesarias
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            error_msg = f"❌ *Error de datos*\n\nColumnas faltantes en los datos descargados.\nColumnas disponibles: {list(data.columns)}"
            enviar_telegram(error_msg)
            return
        
        data = data[required_cols].dropna()
        
        if len(data) < 100:
            error_msg = f"❌ *Datos insuficientes*\n\nSolo se obtuvieron {len(data)} registros. Se necesitan al menos 100."
            enviar_telegram(error_msg)
            return
        
        data.index = data.index.tz_convert(colombia_tz)
        
        print(f"📊 Datos procesados: {len(data)} registros")
        print(f"📅 Rango: {data.index[0]} a {data.index[-1]}")
        
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
        
        if len(df_features) < 50:
            error_msg = f"❌ *Datos insuficientes para ML*\n\nDespués del procesamiento solo quedan {len(df_features)} registros."
            enviar_telegram(error_msg)
            return
        
        # Crear targets
        horizontes = [4, 8, 12, 24, 48]
        for h in horizontes:
            future_return = df_features['Close'].shift(-h) / df_features['Close'] - 1
            df_features[f'target_{h}h'] = (future_return > 0).astype(int)
        
        df_features = df_features.dropna()
        
        # Entrenar modelos
        features_list = ['RSI', 'Volatility', 'Dist_from_MA', 'HL_Range', 'Log_Return_Accel',
                        'RSI_lag1', 'Volatility_lag1', 'ADX', 'BB_position']
        
        X = df_features[features_list]
        mejores_modelos = {}
        scalers = {}
        
        print("\n🔄 Entrenando modelos...")
        for h in horizontes:
            y = df_features[f'target_{h}h']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            modelo = KNeighborsClassifier(n_neighbors=7, weights='distance')
            modelo.fit(X_scaled, y)
            
            mejores_modelos[h] = modelo
            scalers[h] = scaler
        
        print("✅ Modelos entrenados")
        
        # Analizar última vela
        ultima_fila = df_features.tail(1)
        es_anomalia_actual = ultima_fila['anomaly'].values[0]
        fecha_actual = ultima_fila.index[0]
        precio_actual = ultima_fila['Close'].values[0]
        
        # Generar mensaje para Telegram
        mensaje_telegram_final = f"""
📊 *Análisis BTC-USD*

📅 {fecha_actual.strftime('%Y-%m-%d %H:%M')} COT
💰 Precio: ${precio_actual:,.2f}
🎯 Estado: {'🚨 ANOMALÍA' if es_anomalia_actual else '✅ Normal'}
"""
        
        if es_anomalia_actual:
            signal_mr = ultima_fila['signal'].values[0]
            
            X_actual = ultima_fila[features_list]
            predicciones = {}
            
            for h in horizontes:
                X_actual_scaled = scalers[h].transform(X_actual)
                pred = mejores_modelos[h].predict(X_actual_scaled)[0]
                prob = mejores_modelos[h].predict_proba(X_actual_scaled)[0]
                
                tendencia = "ALCISTA" if pred == 1 else "BAJISTA"
                confianza = prob[pred]
                
                nivel = "🟢" if confianza >= 0.65 else "🟡" if confianza >= 0.55 else "🔴"
                
                normalized_signal = "ALCISTA" if signal_mr == "Long" else "BAJISTA"
                compatible = "✅" if tendencia == normalized_signal else "⚠️"
                
                predicciones[h] = {
                    "Predicción": tendencia,
                    "Confianza": confianza,
                    "Nivel": nivel,
                    "vs MeanRev": compatible
                }
            
            # Alineamiento
            alignment = sum(1 for p in predicciones.values() if p["vs MeanRev"] == "✅")
            alignment_pct = (alignment / len(horizontes)) * 100
            
            if alignment_pct >= 81:
                cat = "🟢Excelente"
            elif alignment_pct >= 61:
                cat = "🟡Bueno"
            elif alignment_pct >= 41:
                cat = "🟠Medio"
            else:
                cat = "🔴Bajo"
            
            mensaje_telegram_final += f"""
🎯 Señal MR: *{signal_mr}*
🤝 Alineamiento: {alignment_pct:.0f}% {cat}

*PREDICCIONES:*
"""
            
            for h, p in predicciones.items():
                mensaje_telegram_final += f"  *{h}h:* {p['Predicción']} {p['Nivel']} ({p['Confianza']:.0%}) {p['vs MeanRev']}\n"
            
            # Consenso
            alcistas = sum(1 for p in predicciones.values() if p["Predicción"] == "ALCISTA")
            consenso_pct = (alcistas / len(horizontes)) * 100
            
            if consenso_pct >= 60:
                rec = "✅ CONSIDERAR OPERAR"
            elif consenso_pct >= 40:
                rec = "⚠️ NEUTRAL"
            else:
                rec = "⚠️ PRECAUCIÓN"
            
            mensaje_telegram_final += f"\n💡 *Recomendación:* {rec}"
            
        else:
            mensaje_telegram_final += "\n💡 Sin señales de trading."
        
        # Enviar mensaje
        if enviar_telegram(mensaje_telegram_final):
            print("✅ Análisis completado exitosamente")
        else:
            print("⚠️ Análisis completado pero falló envío a Telegram")
        
    except Exception as e:
        error_msg = f"❌ *ERROR CRÍTICO*\n\n```\n{str(e)[:500]}\n```"
        enviar_telegram(error_msg)
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
