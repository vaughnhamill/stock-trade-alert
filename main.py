import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import requests
import numpy as np
from finvizfinance.screener.ownership import Ownership
from finvizfinance.screener.performance import Performance
from datetime import datetime, timedelta
import pytz
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global Config ---
thresholds = [0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001] # in percentage
windows = [10, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 720]  # in minutes
best_score = -np.inf
best_ticker = None
best_params = None
best_model = None
best_feature_cols = None
best_close_col = None
final_ticker_interval = None
est = pytz.timezone('America/New_York')
 
# --- Telegram Config and Function---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message):
    """Send notification via Telegram."""
    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={'chat_id': CHAT_ID, 'text': message},
            timeout=2
        )
        print("📬 Notification sent")
    except Exception as e:
        print(f"⚠️ Telegram error: {str(e)}")

# --- Main Functions ---

def scan_day_trade_candidates():
    """Scan Finviz for potential day trading candidates."""
    print("🔍 Scanning Finviz for high-momentum small caps...")
    
    performance_filters = {
        "Price": "Under $20",
        "Relative Volume": "Over 2",
        "Average Volume": "Over 500K",
        "Change": "Up 10%"
    }

    try:
        perf_screen = Performance()
        perf_screen.set_filter(filters_dict=performance_filters)
        perf_df = perf_screen.screener_view(limit=200, verbose=0)

        if perf_df.empty:
            print("⚠️ No tickers found with these filters.")
            return []

        tickers = perf_df['Ticker'].tolist()
        
        # Filter by float
        own_screen = Ownership()
        own_screen.set_filter(ticker=",".join(tickers))
        own_df = own_screen.screener_view(limit=len(tickers), verbose=0)

        if own_df.empty:
            print("⚠️ No ownership data found.")
            return []

        def float_to_num(s):
            if isinstance(s, str):
                print(f"Converting float string: {s}")
                s = s.strip()
                if s.endswith('M'):
                    return float(s[:-1]) * 1e6
                elif s.endswith('K'):
                    return float(s[:-1]) * 1e3
                elif s.endswith('B'):
                    return float(s[:-1]) * 1e9
                try:
                    return float(s.replace(',', ''))
                except:
                    return 0
            elif pd.isna(s):
                return 0
            return float(s)

        own_df['FloatNum'] = own_df['Float'].apply(float_to_num)
        filtered_df = own_df[own_df['FloatNum'] < 50_000_000]

        print(f"✅ Found {len(filtered_df)} qualifying tickers")
        return filtered_df['Ticker'].tolist()

    except Exception as e:
        print(f"⚠️ Finviz error: {str(e)}")
        return []

def fetch_stock_data(ticker, interval={"5m": "60d", "10m": "60d", "15m": "60d"}, max_attempts=3):
    global final_ticker_interval
    """
    Stock data fetcher with multiple fallback options.
    
    Args:
        ticker (str): Stock symbol
        interval (str): Preferred data interval
        fallback_intervals (list): Fallback intervals to try
        max_attempts (int): Maximum fetch attempts
        
    Returns:
        pd.DataFrame: Stock data or None if all attempts fail
    """
    attempts = 0
    intervals_to_try = list(interval.items())
    
    while attempts < max_attempts and intervals_to_try:
        current_interval, period = intervals_to_try.pop(0)
        print(f"📊 Attempt {attempts + 1}: Fetching {ticker} ({current_interval}, period={period})")

        try:            
            df = yf.download(
                ticker,
                period=period,
                interval=current_interval,
                auto_adjust=True,
                progress=False,
                threads=True
            )

            final_ticker_interval = current_interval
            
            if not df.empty:
                print(f"✅ Successfully fetched {len(df)} rows for {ticker}")
                return df
            
            print(f"⚠️ Empty data for {ticker} ({current_interval})")
            
        except Exception as e:
            print(f"⚠️ Error fetching {ticker} ({current_interval}): {str(e)}")
        
        # Move to next fallback interval
        attempts += 1
    
    print(f"❌ All attempts failed for {ticker}")
    return None

def add_indicators(df):
    """Add technical indicators to DataFrame."""
    close_col = next((col for col in df.columns if "Close" in col), None)
    if not close_col:
        print("⚠️ No Close price column found")
        return None

    high_col = next((col for col in df.columns if "High" in col), None)
    low_col = next((col for col in df.columns if "Low" in col), None)
    volume_col = next((col for col in df.columns if "Volume" in col), None)

    indicators = {
        'sma_20': (ta.sma, [df[close_col]], {'length': 20}),
        'ema_20': (ta.ema, [df[close_col]], {'length': 20}),
        'rsi': (ta.rsi, [df[close_col]], {'length': 14}),
        'adx': (ta.adx, [df[high_col], df[low_col], df[close_col]], {'length': 14}),
        'atr': (ta.atr, [df[high_col], df[low_col], df[close_col]], {'length': 14}),
        'macd': (ta.macd, [df[close_col]], {'fast': 8, 'slow': 21, 'signal': 9}),
        'bollinger': (ta.bbands, [df[close_col]], {'length': 20, 'std': 2}),
        'obv': (ta.obv, [df[close_col], df[volume_col]], {}),
        'stoch': (ta.stoch, [df[high_col], df[low_col], df[close_col]], {'k': 14, 'd': 3, 'smooth_k': 3})
    }

    for name, (func, args, kwargs) in indicators.items():
        try:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    df[f"{name}_{col}"] = result[col]
            else:
                df[name] = result
        except Exception as e:
            print(f"⚠️ Error calculating {name}: {str(e)}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def label_intraday(df, threshold, window):
    """Label data for intraday trading strategy."""
    close_col = next((col for col in df.columns if "Close" in col), None)
    if not close_col:
        print("❌ No Close column found")
        return None

    df = df.copy()
    # print(f"📊 Using close column: {close_col}")
    
    df['FutureMax'] = df[close_col].shift(-window).rolling(window=window, min_periods=1).max()

    if df['FutureMax'].isna().all():
        print("❌ FutureMax is all NaN")
        return None
    if df[close_col].isna().all():
        print("❌ Close column is all NaN")
        return None

    try:
        df['Target'] = ((df['FutureMax'] / df[close_col] - 1) > threshold).astype(int)
        # print("✅ Target column created")
    except Exception as e:
        print(f"⚠️ Error creating Target: {str(e)}")
        return None

    # print(f"🧪 Target column exists: {'Target' in df.columns}")
    # print(f"🧪 Target NaNs: {df['Target'].isna().sum()} out of {len(df)}")

    try:
        if 'Target' in df.columns and not df['Target'].isna().all() or df.empty or df['Target'].nunique() < 2:
            df = df.dropna(subset=['Target'])
            # print("✅ Dropped rows with NaN in Target (if any)")
        else:
            print("❌ Target column missing, is empty, is all NaN, or has only one unique value")
            return None
    except KeyError as e:
        print(f"KeyError during dropna: {e}")
        print(f"df.columns: {df.columns.tolist()}")
        return None

    return df

def evaluate_model_profitability(df, model, feature_cols, close_col, threshold):
    """Evaluate model profitability metrics."""
    if df.empty or model is None:
        print("❌ Empty DataFrame or model is None")
        return None

    try:
        X = df[feature_cols]
        preds = model.predict(X)
        valid_range = len(df) - 1
        
        profits = []
        durations = []
        
        for i in range(valid_range):
            if preds[i] == 1:
                buy_price = df[close_col].iloc[i]
                target_price = buy_price * (1 + threshold)
                future_prices = df[close_col].iloc[i+1:]
                sell_idx = np.argmax(future_prices >= target_price)
                
                if sell_idx > 0:
                    profit = future_prices.iloc[sell_idx] - buy_price
                    duration = (future_prices.index[sell_idx] - df.index[i]).total_seconds() / 60
                    profits.append(profit)
                    durations.append(duration)
        
        if not profits:
            return None
            
        return {
            'avg_profit': np.mean(profits),
            'avg_duration': np.mean(durations),
            'win_rate': len(profits) / (preds[:valid_range] == 1).sum(),
            'score': np.mean(profits) / np.mean(durations) if np.mean(durations) > 0 else 0
        }
    except Exception as e:
        print(f"⚠️ Evaluation error: {str(e)}")
        return None

def run_and_select_best(ticker, df):
    """Find best parameters for given ticker."""
    global best_score, best_ticker, best_params, best_model, best_feature_cols, best_close_col
    
    close_col = next((col for col in df.columns if "Close" in col), None)
    if not close_col:
        return

    local_best = {'score': -np.inf}
    
    for th in thresholds:
        for w in windows:
            print(f"🧪 Testing {ticker}: threshold={th:.4f}, window={w}")
            
            df_labeled = label_intraday(df, th, w)
            if df_labeled is None:
                continue
                
            feature_cols = [col for col in df_labeled.columns 
                          if col not in ['Target', 'FutureMax'] 
                          and pd.api.types.is_numeric_dtype(df_labeled[col])]

            if not feature_cols:
                continue
                
            X = df_labeled[feature_cols]
            y = df_labeled['Target']
                
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            try:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    early_stopping_rounds=10,
                    random_state=42,
                    base_score=0.5
                )
                
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                
                metrics = evaluate_model_profitability(
                    df_labeled.iloc[split_idx:],
                    model,
                    feature_cols,
                    close_col,
                    th
                )
                
                if metrics and metrics['score'] > local_best['score']:
                    local_best.update({
                        'score': metrics['score'],
                        'params': (th, w),
                        'model': model,
                        'features': feature_cols,
                        'close_col': close_col,
                        'metrics': metrics,
                    })
                    
            except Exception as e:
                print(f"⚠️ Training error: {str(e)}")
                continue
    
    if local_best['score'] > best_score:
        best_score = local_best['score']
        best_ticker = ticker
        best_params = local_best['params']
        best_model = local_best['model']
        best_feature_cols = local_best['features']
        best_close_col = local_best['close_col']
        
        print(f"🏆 New best: {ticker} (score: {best_score:.4f})")

def simulate_sell_signal(df, model, feature_cols, close_col, steps, increase_rate):
    """Simulate future price movements to predict sell signals."""
    if df.empty or model is None:
        print("❌ Empty DataFrame or model is None")
        return None
    
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
    }

    last_row = df.iloc[-1]
    interval = interval_map.get(final_ticker_interval, timedelta(minutes=1))
    
    simulated = []
    for i in range(1, steps + 1):
        future_row = last_row.copy()
        future_time = df.index[-1] + i * interval
        
        # Simulate price and indicators
        future_row[close_col] *= (1 + increase_rate)  # simulate close price increase

        for col in feature_cols:
            if col == close_col:
                # Close price already simulated above
                # print(f"Skipping {col} as it's the close price")
                continue

            # RSI: bounded between 0 and 100, usually oscillates between ~30-70
            if 'rsi' in col.lower():
                future_row[col] = np.clip(future_row[col] + np.random.uniform(-1, 1), 0, 100)
                # print(f"Simulated {col}: {future_row[col]}")
            
            # SMA/EMA: moving averages — generally smooth, so small proportional changes
            elif any(x in col.lower() for x in ['sma', 'ema']):
                future_row[col] *= (1 + increase_rate * 0.9)
                # print(f"Simulated {col}: {future_row[col]}")

            # ADX: bounded roughly 0-100, but often below 60, measure trend strength
            elif 'adx' in col.lower():
                future_row[col] = np.clip(future_row[col] + np.random.uniform(-0.5, 0.5), 0, 100)
                # print(f"Simulated {col}: {future_row[col]}")

            # ATR: average true range, positive and fluctuates around volatility scale, allow small noise
            elif 'atr' in col.lower():
                future_row[col] = max(0, future_row[col] * (1 + np.random.uniform(-0.05, 0.05)))
                # print(f"Simulated {col}: {future_row[col]}")

            # MACD components: can be positive or negative, small random walk
            elif 'macd' in col.lower():
                future_row[col] += np.random.uniform(-0.01, 0.01)
                # print(f"Simulated {col}: {future_row[col]}")

            # Bollinger Bands: upper, middle, lower bands, reflect price volatility, small proportional noise
            elif 'bollinger' in col.lower() or 'bb_' in col.lower():
                future_row[col] *= (1 + np.random.uniform(-0.01, 0.01))
                # print(f"Simulated {col}: {future_row[col]}")

            # OBV: cumulative volume, can be very large; simulate small random walk
            elif 'obv' in col.lower():
                future_row[col] += np.random.randint(-1000, 1000)
                # print(f"Simulated {col}: {future_row[col]}")

            # Stochastic Oscillator: bounded 0-100, oscillates fast; add small noise bounded 0-100
            elif 'stoch' in col.lower():
                future_row[col] = np.clip(future_row[col] + np.random.uniform(-2, 2), 0, 100)
                # print(f"Simulated {col}: {future_row[col]}")

            # For any other numeric columns, apply slight random noise
            elif pd.api.types.is_numeric_dtype(future_row[col]):
                future_row[col] *= (1 + np.random.uniform(-0.005, 0.005))
                # print(f"Simulated {col}: {future_row[col]}")

        
        future_row.name = future_time
        simulated.append(future_row)
    
    simulated_df = pd.DataFrame(simulated)[feature_cols]
    predictions = model.predict(simulated_df)
    
    sell_points = np.where(predictions == 0)[0]
    if len(sell_points) > 0:
        sell_time = simulated_df.index[sell_points[0]]
        sell_price = simulated[sell_points[0]][close_col]
        return sell_time, sell_price
    
    return None

def run_algorithm():
    """Main trading algorithm execution."""
    global best_model, best_params
    
    print("🚀 Starting trading algorithm")
    tickers = scan_day_trade_candidates()
    if not tickers:
        current_time_est = datetime.now(est)
        print(f"⚠️ No tickers found at {current_time_est.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
        return
    
    top_tickers = tickers[:10]
    print(f"📊 Analyzing: {', '.join(top_tickers)}")
    for ticker in top_tickers:
        print(f"\n🔍 Processing {ticker}")
        
        df = fetch_stock_data(ticker)
        if df is None:
            continue
            
        df = add_indicators(df)
        if df is None:
            continue

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns[0], tuple):
            df.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in df.columns]
 
        run_and_select_best(ticker, df)
    
    if best_model is None:
        current_time_est = datetime.now(est)
        print(f"⚠️ No valid models found {current_time_est.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
        return
    
    # # Final analysis on best ticker
    print(f"\n🏆 Running final analysis on {best_ticker}")
    df = fetch_stock_data(best_ticker)
    df = add_indicators(df)

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in df.columns]
    
    close_col = best_close_col
    current_price = df[close_col].iloc[-1]
    current_time = df.index[-1].tz_convert('US/Eastern')
    
    # Get prediction
    latest_features = df[best_feature_cols].iloc[-1:].values
    prediction = best_model.predict(latest_features)[0]
    proba = best_model.predict_proba(latest_features)[0][1]
    
    message = [
        f"📈 {best_ticker} Analysis Results (Threshold: {best_params[0]*100}%, Window: {best_params[1]} min)",
        f"🕒 Time: {current_time.strftime('%m-%d-%Y %I:%M %p %Z')}",
        f"💰 Price: ${current_price:.2f}",
        f"🔮 Signal: {'BUY' if prediction == 1 else 'HOLD'} ({proba:.1%} confidence)"
    ]
    
    if prediction == 1:
        sell_info = simulate_sell_signal(
            df,
            best_model,
            best_feature_cols,
            best_close_col,
            steps=best_params[1],
            increase_rate = (1 + best_params[0]/100) ** (1 / best_params[1]) - 1
        )

        if sell_info:
            sell_time, sell_price = sell_info
            profit_pct = (sell_price/current_price - 1) * 100
            message.extend([
                "",
                "🎯 Predicted Sell Signal:",
                f"⏰ Time: {sell_time.tz_convert('US/Eastern').strftime('%m-%d-%Y %I:%M %p %Z')}",
                f"💰 Price: ${sell_price:.2f}",
                f"📊 Potential Profit: {profit_pct:.2f}%"
            ])
        
        send_telegram_message("\n".join(message))

    print("\n".join(message))

if __name__ == "__main__":
    run_algorithm()
