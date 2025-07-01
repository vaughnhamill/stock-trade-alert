import numpy as np
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pytz
import warnings
import os
import requests
import time
import json
from pycoingecko import CoinGeckoAPI
from sklearn.metrics import accuracy_score
from joblib import dump, load
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.025]  # Percentage targets
WINDOWS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 420, 480, 540, 600]  # Minutes
MIN_DATA_POINTS = 1000  # Minimum candles required
EST = pytz.timezone('America/New_York')
MODEL_DIR = 'models'
TRADE_HISTORY_FILE = 'trade_history.json'
MODEL_METADATA_FILE = 'model_metadata.json'
FEEDBACK_INTERVAL_HOURS = 3  # How often to retrain models
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
best_score = -np.inf
best_coin = None
best_params = None
best_model = None
best_features = None
best_close = None
final_coin_interval = None

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# --- Telegram Config and Function---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(self, message):
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

# --- File System Setup ---
os.makedirs(MODEL_DIR, exist_ok=True)

class CryptoTrader:
    def __init__(self):
        self.models = {}
        self.trade_history = []
        self.model_metadata = {}
        self.load_state()

    def load_state(self):
        """Load saved models and trade history"""
        # Load trade history
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                self.trade_history = json.load(f)

        # Load model metadata
        if os.path.exists(MODEL_METADATA_FILE):
            with open(MODEL_METADATA_FILE, 'r') as f:
                self.model_metadata = json.load(f)

        # Load models
        for threshold in THRESHOLDS:
            model_file = f"{MODEL_DIR}/model_{threshold:.4f}.joblib"
            if os.path.exists(model_file):
                self.models[threshold] = {
                    'model': load(model_file),
                    'last_retrain': datetime.fromisoformat(
                        self.model_metadata.get(str(threshold), {}).get('last_retrain', '2000-01-01')
                    )
                }

    def scan_candidates(self):
            """Find potential trading candidates"""
            print("🔍 Scanning for candidates...")
            try:
                coins = cg.get_coins_markets(
                    vs_currency='usd',
                    order='market_cap_desc',
                    per_page=1000,
                    price_change_percentage='1h'
                )

                return [{
                    'id': c['id'],
                    'symbol': c['symbol'].upper(),
                    'price_change_percentage_1h': c['price_change_percentage_1h_in_currency'],
                    'price_change_percentage_24h': c['price_change_percentage_24h']
                } for c in coins if c['price_change_percentage_1h_in_currency'] > 1.5 and c['price_change_percentage_24h'] > 3]  # Min % hourly change

            except Exception as e:
                print(f"⚠️ Error scanning candidates: {str(e)}")
                return []

    def fetch_crypto_data(self, coin_id, interval={"5m": "90d", "10m": "90d", "15m": "90d"}, max_attempts=3):
            """Fetch data from CoinGecko with retries"""
            attempts = 0
            intervals_to_try = list(interval.items())

            while attempts < max_attempts and intervals_to_try:
                current_interval, period = intervals_to_try.pop(0)
                print(f"📊 Attempt {attempts + 1}: Fetching {coin_id} ({current_interval}, period={period})")
                try:
                    data = cg.get_coin_market_chart_by_id(
                        id=coin_id,
                        vs_currency='usd',
                        days=period
                    )

                    if not data or 'prices' not in data:
                        print(f"❌ No data found for {coin_id} with interval {current_interval}")
                        return None

                    # Process into DataFrame
                    price_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

                    # Merge price and volume data
                    df = price_df.merge(volume_df, on='timestamp')

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')

                    # Resample to desired interval
                    if current_interval == '5m':
                       # Already 5m, no resampling needed
                       df['open'] = df['price']
                       df['high'] = df['price']
                       df['low'] = df['price']
                       df['close'] = df['price']
                       # You can keep volume as-is
                    else:
                       df = df.resample(current_interval).agg(
                          open=('price', 'first'),
                          high=('price', 'max'),
                          low=('price', 'min'),
                          close=('price', 'last'),
                          volume=('volume', 'sum')  # total volume over resample period
                        )
                    print("📈 Data fetched successfully")

                    if len(df) >= MIN_DATA_POINTS:
                        print(f"✅ Enough data points ({len(df)}) for {coin_id}")
                        return df
                    print(f"❌ Not enough data points ({len(df)}) for {coin_id}")
                    return None

                except Exception as e:
                    print(f"⚠️ Error fetching data: {str(e)}")
                    time.sleep(5)

                attempts += 1

            print(f"❌ Failed to fetch data for {coin_id} after {max_attempts} attempts")
            return None

    def calculate_features(self, df):
        """Add technical indicators"""
        if df.empty:
            return None
        try:
          # Basic features
          df['returns'] = df['close'].pct_change()
          df['volatility'] = df['returns'].rolling(20).std()

          # Technical indicators
          df['rsi'] = ta.rsi(df['close'], length=14)
          df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
          df['bollinger'] = ta.bbands(df['close'], length=20, std=2)['BBM_20_2.0']
          df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
          df['sma_20'] = ta.sma(df['close'], length=20)
          df['ema_20'] = ta.ema(df['close'], length=20)
          df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
          df['obv'] = ta.obv(df['close'], df['volume'])
          stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
          df = pd.concat([df, stoch], axis=1)

        except Exception as e:
          print(f"⚠️ Error: {str(e)}")
          return None

        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        print(f"✅ Technical indicators added")

        return df

    def train_model(self, coin, df):
        """Train or update a model"""
        global best_score, best_coin, best_params, best_model, best_features, best_close

        print(f"📊 Attempting to train model")

        close = next((col for col in df.columns if "close" in col), None)
        if not close:
          return

        local_best = {'score': -np.inf}

        for th in THRESHOLDS:
          for w in WINDOWS:
              print(f"🧪 Testing {coin['symbol']}: threshold={th:.4f}, window={w}")

              labeled_data = self.prepare_training_data(df, th, w)
              if labeled_data is None:
                  continue

              features = [col for col in labeled_data.columns
                          if col not in ['target', 'future_max']
                          and pd.api.types.is_numeric_dtype(labeled_data[col])]

              if not features:
                  continue

              X = labeled_data[features]
              y = labeled_data['target']

              split = int(0.8 * len(X))
              X_train, X_test = X.iloc[:split], X.iloc[split:]
              y_train, y_test = y.iloc[:split], y.iloc[split:]

              try:
                if th not in self.models:
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
                    self.models[th] = {
                        'model': model,
                        'last_retrain': datetime.now()
                    }
                else:
                    model = self.models[th]['model']

                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                metrics = self.evaluate_model_profitability(
                                  labeled_data.iloc[split:],
                                  model,
                                  features,
                                  close,
                                  th
                              )

                # Update metadata
                self.models[th]['last_retrain'] = datetime.now()
                self.models[th]['score'] = accuracy_score(y_test, model.predict(X_test))

                if metrics and metrics['score'] > local_best['score']:
                    local_best.update({
                        'score': metrics['score'],
                        'params': (th, w),
                        'model': model,
                        'features': features,
                        'close': close,
                        'metrics': metrics,
                    })

              except Exception as e:
                  print(f"⚠️ Training error: {str(e)}")
                  continue

          if local_best['score'] > best_score:
              best_score = local_best['score']
              best_coin = coin
              best_params = local_best['params']
              best_model = local_best['model']
              best_features = local_best['features']
              best_close = local_best['close']

              print(f"🏆 New best: {coin['symbol']} (score: {best_score:.4f})")

    def prepare_training_data(self, df, threshold, window):
        """Create labeled dataset"""
        try:
          df = df.copy()
          df['future_max'] = df['close'].shift(-window).rolling(window, min_periods=1).max()
          df['target'] = ((df['future_max'] / df['close'] - 1) >= threshold).astype(int)
          df.dropna(subset=['target'], inplace=True)

          if df['target'].nunique() < 2:
              return None

          print(f"✅ Data labeled")

        except Exception as e:
          print(f"⚠️ Error: {str(e)}")
          return None

        return df

    def evaluate_model_profitability(self, df, model, features, close, threshold):
        """Evaluate model profitability metrics."""
        if df.empty or model is None:
            print("❌ Empty DataFrame or model is None")
            return None

        try:
            X = df[features]
            preds = model.predict(X)
            valid_range = len(df) - 1

            profits = []
            durations = []

            for i in range(valid_range):
                if preds[i] == 1:
                    buy_price = df[close].iloc[i]
                    target_price = buy_price * (1 + threshold)
                    future_prices = df[close].iloc[i+1:]
                    sell = np.argmax(future_prices >= target_price)

                    if sell > 0:
                        profit = future_prices.iloc[sell] - buy_price
                        duration = (future_prices.index[sell] - df.index[i]).total_seconds() / 60
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

    def record_trade(self, symbol, threshold, features, prediction, entry_price, entry_time):
        """Store trade details for later outcome evaluation"""
        trade_id = hashlib.md5(
            f"{symbol}{threshold}{entry_time.isoformat()}".encode()
        ).hexdigest()

        # Convert NumPy types in features dictionary to standard Python types
        processed_features = {}
        for key, value in features.items():
            if isinstance(value, (np.int64, np.int32)):
                processed_features[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                 processed_features[key] = float(value)
            else:
                processed_features[key] = value

        self.trade_history.append({
              'id': trade_id,
              'symbol': symbol,
              'threshold': threshold,
              'timestamp': entry_time.isoformat(),
              'features': processed_features,  # Use the processed features
              'prediction': int(prediction),  # Ensure prediction is a standard int
              'entry_price': float(entry_price), # Ensure entry_price is a standard float
              'status': 'pending'  # Mark as unresolved
          })

    def evaluate_pending_trades(self, delay_minutes):
        """Evaluate outcome of pending trades"""
        now = datetime.now()
        for trade in self.trade_history:
            if trade['status'] != 'pending':
                continue

            entry_time = datetime.fromisoformat(trade['timestamp'])
            if (now - entry_time).total_seconds() < delay_minutes * 60:
                continue  # Too early to evaluate

            # Fetch real price at current time
            future_price = cg.get_price(ids=best_coin['id'], vs_currencies='usd')

            target_price = trade['entry_price'] * (1 + trade['threshold'])
            trade['exit_price'] = future_price
            trade['evaluated_at'] = now.isoformat()

            if future_price >= target_price:
                trade['outcome'] = 1
            else:
                trade['outcome'] = 0

            trade['status'] = 'evaluated'
            print(f"📊 Trade evaluated for {trade['symbol']}: {'✅ Success' if trade['outcome'] else '❌ Fail'}")

        # Trigger retrain if enough new trades
        self.retrain_models()

    def simulate_sell_signal(self, df, model, features, close, steps, increase_rate):
        """Simulate future price movements to predict sell signals."""
        if df.empty or model is None:
            print("❌ Empty DataFrame or model is None")
            return None

        interval_map = {
            "5m": timedelta(minutes=5),
            "10m":timedelta(minutes=10),
            "15m": timedelta(minutes=15),
        }

        last_row = df.iloc[-1]
        interval = interval_map.get(final_coin_interval, timedelta(minutes=5))

        simulated = []
        for i in range(1, steps + 1):
            future_row = last_row.copy()
            future_time = df.index[-1] + i * interval

            # Simulate price and indicators
            future_row[close] *= (1 + increase_rate)  # simulate close price increase

            for col in features:
                if col == close:
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

        simulated_df = pd.DataFrame(simulated)[features]
        predictions = model.predict(simulated_df)

        sell_points = np.where(predictions == 0)[0]
        if len(sell_points) > 0:
            sell_time = simulated_df.index[sell_points[0]]
            sell_price = simulated[sell_points[0]][close]
            return sell_time, sell_price

        return None

    def retrain_models(self):
        """Retrain models with new trade data"""
        recent_trades = [
            t for t in self.trade_history
            if (datetime.now() - datetime.fromisoformat(t['timestamp'])).days < 7
        ]

        if len(recent_trades) < 50:  # Minimum trades for retraining
            return

        # Group trades by threshold
        for threshold in THRESHOLDS:
            threshold_trades = [
                t for t in recent_trades
                if abs(t['threshold'] - threshold) < 0.0001
            ]

            if len(threshold_trades) < 10:  # Minimum per threshold
                continue

            # Prepare data
            X = np.array([list(t['features'].values()) for t in threshold_trades])
            y = np.array([t['outcome'] for t in threshold_trades])

            # Split for validation
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # Update model
            if threshold in self.models:
                self.train_model(threshold, X_train, y_train, X_val, y_val)

    def save_state(self):
        """Persist models and trade history"""
        # Save models
        for threshold, data in self.models.items():
            model_file = f"{MODEL_DIR}/model_{threshold:.4f}.joblib"
            dump(data['model'], model_file)

            # Update metadata
            self.model_metadata[str(threshold)] = {
                'last_retrain': data['last_retrain'].isoformat(),
                'score': data.get('score', 0)
            }

        # Save trade history
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

        # Save model metadata
        with open(MODEL_METADATA_FILE, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        if GITHUB_ACTIONS:
            print("Persisting files for GitHub Actions...")
            os.system("git config --global user.email 'actions@github.com'")
            os.system("git config --global user.name 'GitHub Actions'")
            os.system("git add .")
            os.system(f"git commit -m 'Update models and trade history {datetime.now().isoformat()}'")
            os.system("git push")

    def run_analysis(self):
        """Main analysis pipeline"""
        print("🚀 Starting trading algorithm")
        # 1. Scan for candidates
        candidates = self.scan_candidates()

        if not candidates:
            print("❌ No candidates found")
            return

        # 2. Analyze each candidate
        for coin in candidates[:10]:  # Limit to top 10
            print(f"\n📊 Analyzing {coin['symbol']}")
            df = self.fetch_crypto_data(coin['id'])
            if df is None:
                continue

            df = self.calculate_features(df)
            if df is None:
                continue

            # 3. Train/update models for each threshold
            self.train_model(coin, df)

        if best_model is None:
          current_time_est = datetime.now(EST)
          print(f"⚠️ No valid models found {current_time_est.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
          return

        print(f"\n🏆 Running final analysis on {best_coin['symbol']}")
        df = self.fetch_crypto_data(best_coin['id'])
        df = self.calculate_features(df)

        close = best_close
        current_price = df[close].iloc[-1]
        current_time = df.index[-1].tz_convert('US/Eastern')

        # 4. Make predictions and simulate trades
        current_features = df[best_features].iloc[-1:].values
        prediction = best_model.predict(current_features)[0]
        proba = best_model.predict_proba(current_features)[0][1]

        if prediction == 1 and proba > 0.5:  # Confidence threshold
          message = None
          current_time = datetime.now()
          entry_price = df['close'].iloc[-1]
          print(f"🚀 Trade signal: {coin['symbol']} at {best_params[0]:.2%} threshold")

          self.record_trade(
              symbol=coin['symbol'],
              threshold=best_params[0],
              features=df[best_features].iloc[-1].to_dict(),
              prediction=prediction,
              entry_price=entry_price,
              entry_time=current_time
          )

          message = [
              f"📈 {best_coin['symbol']} Analysis Results (Threshold: {best_params[0]*100}%, Window: {best_params[1]} min)",
              f"🕒 Time: {current_time.strftime('%m-%d-%Y %I:%M %p %Z')}",
              f"💰 Price: ${current_price:.2f}",
              f"🔮 Signal: {'BUY' if prediction == 1 else 'HOLD'} ({proba:.1%} confidence)"
          ]

          # Predict sell signal
          sell_info = self.simulate_sell_signal(
              df,
              best_model,
              best_features,
              best_close,
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
              
              self.send_telegram_message("\n".join(message))
          
          elif sell_info is None and message is not None:
              print("✅ Buy signal predicted \n❌ No sell signal predicted")
          
          # Record price based on threshold and window
          self.evaluate_pending_trades(best_params[1])

        else:
          print("❌ No buy signal predicted")
    
        self.save_state()

if __name__ == "__main__":
    trader = CryptoTrader()
    trader.run_analysis()
