# ------------- API DOCUMENTATION ------------- #
# Binance US - https://docs.binance.us/
# Coinbase - https://api.exchange.coinbase.com/

# ------------- IMPORTS ------------- #
import numpy as np
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime, timedelta
import pytz
import warnings
import os
import requests
import time
import json
import pickle
from pycoingecko import CoinGeckoAPI
from sklearn.metrics import f1_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import hashlib
from textblob import TextBlob
from scipy.stats import t
import arch
import ccxt
from imblearn.over_sampling import SMOTE, ADASYN
import subprocess

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
THRESHOLDS = [0.02, 0.99]  # Min, Max
MAX_WINDOW_MINUTES = 500
EST = pytz.timezone('America/New_York')
MODEL_DIR = 'crypto/spot/models'
TRADE_HISTORY_FILE = 'crypto/spot/trade_history.json'
MODEL_METADATA_FILE = 'crypto/spot/model_metadata.json'
FEEDBACK_INTERVAL_HOURS = 3  # How often to retrain models
SENTIMENT_CACHE_FILE = 'crypto/spot/sentiment_cache.pkl'
SENTIMENT_CACHE_TTL = 14400  # Cache sentiment for 4 hours
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
RATE_LIMIT_HIT = False
W_1M = 0.75  # Weight for 1 min df in buy score
W_1H = 0.25  # Weight for 1 hour df in buy score
W_SENTIMENT = 0.1  # Weight for sentiment in buy score
PORTFOLIO_SIZE = 1000  # Portfolio size for position sizing
best_score = -np.inf
best_coin = None
best_analysis = None
best_1m_df = None
best_1h_df = None

# Load environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize APIs
cg = CoinGeckoAPI()
binanceus = ccxt.binanceus()

# ------------- File System Setup ---------------- #
os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------- CryptoTrader Class ---------------- #
class CryptoTrader:
    def __init__(self):
        self.models = {}  # Format: {(coin, timeframe, threshold): {'clf': model, 'reg': model, 'features': list, 'score': float}}
        self.trade_history = []
        self.model_metadata = {}
        self.coinbase_pairs = None
        self.binanceus_pairs = None
        self.sentiment_cache = {}
        self.last_buy_signal = None
        self.load_sentiment_cache()
        self.load_state()

    # ------------- Telegram Message Setup ------------- #
    def send_telegram_message(self, message):
        """Send notification via Telegram."""
        if not TELEGRAM_TOKEN or not CHAT_ID:
            print("⚠️ Telegram not configured: TELEGRAM_TOKEN or CHAT_ID missing")
            return
        try:
            requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                params={'chat_id': CHAT_ID, 'text': message},
                timeout=2
            )
            print("📬 Telegram notification sent")
        except Exception as e:
            print(f"⚠️ Telegram error: {str(e)}")

    # ------------- State Management ------------- #
    def load_state(self):
        """Load saved models, trade history, and metadata."""
        # Ensure directories exist
        os.makedirs(os.path.dirname(TRADE_HISTORY_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_METADATA_FILE), exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Load trade history
        if os.path.exists(TRADE_HISTORY_FILE):
            try:
                with open(TRADE_HISTORY_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        self.trade_history = json.load(f)
                        print(f"✅ Loaded trade history with {len(self.trade_history)} entries")
                    else:
                        print(f"⚠️ Trade history file is empty, initializing empty trade history")
                        self.trade_history = []
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in trade history: {str(e)}, initializing empty trade history")
                self.trade_history = []
            except Exception as e:
                print(f"⚠️ Error loading trade history: {str(e)}, initializing empty trade history")
                self.trade_history = []
        else:
            print(f"⚠️ Trade history file not found, initializing empty trade history")
            self.trade_history = []

        # Load model metadata
        if os.path.exists(MODEL_METADATA_FILE):
            try:
                with open(MODEL_METADATA_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.model_metadata = json.load(f)
                        print(f"✅ Loaded model metadata with {len(self.model_metadata)} entries")
                        for key, meta in self.model_metadata.items():
                            coin, timeframe, threshold = key.split('_')
                            threshold = float(threshold)
                            model_key = (coin, timeframe, threshold)
                            clf_path = meta.get('clf_path')
                            reg_path = meta.get('reg_path')
                            self.models[model_key] = {
                                'clf': load(clf_path) if clf_path and os.path.exists(clf_path) else None,
                                'reg': load(reg_path) if reg_path and os.path.exists(reg_path) else None,
                                'features': meta.get('features', []),
                                'score': meta.get('score', 0.0)
                            }
                        print(f"✅ Loaded {len(self.models)} models from disk")
                    else:
                        print(f"⚠️ Model metadata file is empty, initializing empty metadata")
                        self.model_metadata = {}
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in model metadata: {str(e)}, initializing empty metadata")
                self.model_metadata = {}
            except Exception as e:
                print(f"⚠️ Error loading models: {str(e)}, initializing empty metadata")
                self.model_metadata = {}
                self.models = {}
    
    def load_sentiment_cache(self):
        """Load sentiment cache from disk."""
        os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
        try:
            if os.path.exists(SENTIMENT_CACHE_FILE):
                with open(SENTIMENT_CACHE_FILE, 'rb') as f:
                    self.sentiment_cache = pickle.load(f)
                print(f"✅ Loaded sentiment cache with {len(self.sentiment_cache)} entries")
        except (PermissionError, pickle.PickleError) as e:
            print(f"⚠️ Error loading sentiment cache: {str(e)}, initializing empty cache")
            self.sentiment_cache = {}

    def save_sentiment_cache(self):
        """Save sentiment cache to disk and commit to Git if in GitHub Actions."""
        os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
        try:
            with open(SENTIMENT_CACHE_FILE, 'wb') as f:
                pickle.dump(self.sentiment_cache, f)
            print("✅ Saved sentiment cache")
            self.send_telegram_message("Saved sentiment cache")

            # Commit to Git in GitHub Actions
            if GITHUB_ACTIONS:
                print("Persisting sentiment cache for GitHub Actions...")
                try:
                    subprocess.run(['git', 'config', '--global', 'user.email', 'actions@github.com'], check=True)
                    subprocess.run(['git', 'config', '--global', 'user.name', 'GitHub Actions'], check=True)
                    subprocess.run(['git', 'add', SENTIMENT_CACHE_FILE], check=True)
                    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
                    if result.stdout.strip():
                        commit_message = f"Update sentiment cache {datetime.now(EST).isoformat()}"
                        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
                        repo_url = os.getenv('GITHUB_REPOSITORY')
                        if repo_url:
                            token = os.getenv('GITHUB_TOKEN', '')
                            if not token:
                                raise ValueError("GITHUB_TOKEN not set for Git push")
                            auth_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
                            subprocess.run(['git', 'push', auth_url, 'HEAD'], check=True)
                            print("✅ Pushed sentiment cache to GitHub repository")
                            self.send_telegram_message("Pushed sentiment cache to GitHub repository")
                        else:
                            raise ValueError("GITHUB_REPOSITORY not set")
                    else:
                        print("ℹ️ No changes to commit for sentiment cache")
                        self.send_telegram_message("No changes to commit for sentiment cache")
                except (subprocess.CalledProcessError, ValueError) as e:
                    print(f"⚠️ Git operation failed for sentiment cache: {str(e)}")
                    self.send_telegram_message(f"Git operation failed for sentiment cache: {str(e)}")
        except (PermissionError, pickle.PickleError) as e:
            print(f"⚠️ Error saving sentiment cache: {str(e)}, skipping cache save")
            self.send_telegram_message(f"Error saving sentiment cache: {str(e)}")

    # ------------- Record Trades and Retrain Models ------------- #
    def record_trade(self, coin, entry_price, expected_return, entry_time, sell_time, features_1m, features_1h, buy_score, position_size_pct, exchange):
        """Store trade details for later outcome evaluation"""
        trade_id = hashlib.md5(
            f"{coin['symbol']}{expected_return}{entry_time.isoformat()}".encode()
        ).hexdigest()

        # Convert NumPy types in features dictionary to standard Python types
        processed_features_1m = {}
        for key, value in features_1m.items():
            if isinstance(value, (np.int64, np.int32)):
                processed_features_1m[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                processed_features_1m[key] = float(value)
            else:
                processed_features_1m[key] = value

        processed_features_1h = {}
        for key, value in features_1h.items():
            if isinstance(value, (np.int64, np.int32)):
                processed_features_1h[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                processed_features_1h[key] = float(value)
            else:
                processed_features_1h[key] = value

        self.trade_history.append({
            'trade_id': trade_id,
            'id': coin['id'],
            'symbol': coin['symbol'],
            'exchange': exchange,
            'trade_info': {
                'entry_price': float(entry_price),
                'expected_return': float(expected_return),
                'entry_time': entry_time.isoformat(),
                'sell_time': sell_time.isoformat(),
                'buy_score': float(buy_score),
                'position_size_pct': float(position_size_pct),
                'selected_symbol': coin['selected_symbol']
            },
            'features': {
                '1m': processed_features_1m,
                '1h': processed_features_1h
            },
            'status': 'pending'
        })
        self.save_state()

    def evaluate_pending_trades(self):
        """Evaluate outcome of pending trades"""
        print("🔍 Evaluating pending trades...")
        now = datetime.now(EST)
        for trade in self.trade_history:
            if trade['status'] != 'pending':
                continue

            sell_time = datetime.fromisoformat(trade['trade_info']['sell_time']).astimezone(EST)
            if now < sell_time:
                continue  # Too early to evaluate

            # Fetch prices at sell_time with a 10-minute window
            symbol = trade['trade_info']['selected_symbol']
            exchange = trade['exchange']
            print(f"📊 Evaluating trade for {symbol} at sell time {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
            price_data = self.fetch_data(symbol, '1m', exchange, limit=10, start_time=sell_time - timedelta(minutes=5), end_time=sell_time + timedelta(minutes=5))

            if price_data is None or price_data.empty:
                print(f"⚠️ No price data available for {symbol} on {exchange} around {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}, skipping evaluation")
                continue
            
            # Ensure price_data index is in EST
            price_data.index = price_data.index.tz_convert(EST)

            # Search for exact sell_time in price_data index
            if sell_time in price_data.index:
                sell_price = price_data.loc[sell_time]['close']
                print(f"✅ Found exact sell time {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')} with close price: ${sell_price:.6f}")
            else:
                # Find closest timestamps before and after
                before = price_data[price_data.index <= sell_time]
                after = price_data[price_data.index >= sell_time]
                if before.empty or after.empty:
                    print(f"⚠️ Insufficient data around sell time for {symbol}, skipping")
                    continue
                closest_before = before.index.max()
                closest_after = after.index.min()
                time_diff = (closest_after - closest_before).total_seconds() / 60
                if time_diff > 5:
                    print(f"⚠️ Closest candles {closest_before.strftime('%m-%d-%Y %H:%M:%S %Z%z')} and {closest_after.strftime('%m-%d-%Y %H:%M:%S %Z%z')} are {time_diff:.2f} minutes apart, skipping")
                    continue
                # Linear interpolation
                price_before = price_data.loc[closest_before]['close']
                price_after = price_data.loc[closest_after]['close']
                time_to_before = (sell_time - closest_before).total_seconds() / 60
                time_between = (closest_after - closest_before).total_seconds() / 60
                weight = time_to_before / time_between
                sell_price = price_before + (price_after - price_before) * weight
                print(f"⚠️ Interpolated price for {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}: ${sell_price:.6f} (between {price_before:.6f} and {price_after:.6f})")

            # Calculate actual return
            entry_price = float(trade['trade_info']['entry_price'])
            actual_return = (sell_price - entry_price) / entry_price
            expected_return = float(trade['trade_info']['expected_return'])
            trade['trade_info']['sell_price'] = float(sell_price)
            trade['trade_info']['actual_return'] = float(actual_return)
            trade['status'] = 'completed'
            trade['trade_info']['evaluation_time'] = now.isoformat()

            # Determine outcome
            outcome = 'profitable' if actual_return >= 0 else 'loss'
            trade['trade_info']['outcome'] = outcome
            print(f"📈 Trade outcome: {outcome}, Actual return: {actual_return * 100:.2f}% (Expected: {expected_return * 100:.2f}%)")
            self.send_telegram_message(
                f"Trade evaluated: {trade['symbol']} {outcome}, Actual return: {actual_return * 100:.2f}% (Expected: {expected_return * 100:.2f}%)"
            )

            self.save_state()

        # Trigger retrain if enough new trades
        self.retrain_models()

    def retrain_models(self):
        """Retrain models with recent trade data, retaining them on disk."""
        print("🔄 Retraining models with recent trade data")
        now = datetime.now(EST)
        recent_trades = [
            t for t in self.trade_history
            if t['status'] == 'completed' and
            (now - datetime.fromisoformat(t['trade_info']['entry_time']).astimezone(EST)).days < 7
        ]
        
        if len(recent_trades) < 50:
            print(f"⚠️ Insufficient trades ({len(recent_trades)} < 50) for retraining")
            return

        # Group trades by coin and timeframe
        for coin in set(t['symbol'] for t in recent_trades):
            for timeframe in ['1m', '1h']:
                coin_trades = [t for t in recent_trades if t['symbol'] == coin]
                if not coin_trades:
                    continue
                
                # Derive dynamic thresholds from expected returns
                expected_returns = [t['trade_info']['expected_return'] for t in coin_trades]
                thresholds = [np.percentile(expected_returns, 25), np.percentile(expected_returns, 75)]
                thresholds = [max(min(t, THRESHOLDS[1]), THRESHOLDS[0]) for t in thresholds]
                print(f"📈 Thresholds for {coin} ({timeframe}): {thresholds}")

                for threshold in thresholds:
                    threshold_trades = [
                        t for t in coin_trades
                        if abs(t['trade_info']['expected_return'] - threshold) < 0.01
                    ]
                    if len(threshold_trades) < 10:
                        print(f"⚠️ Insufficient trades ({len(threshold_trades)} < 10) for {coin} ({timeframe}, threshold: {threshold})")
                        continue

                    # Prepare data
                    X = np.array([list(t['features'][timeframe].values()) for t in threshold_trades])
                    le = LabelEncoder()
                    y = le.fit_transform([t['trade_info']['outcome'] for t in threshold_trades])  # 'profitable' -> 1, 'loss' -> 0
                    y_reg = np.array([t['trade_info']['actual_return'] for t in threshold_trades])

                    # Split for validation
                    split = int(0.8 * len(X))
                    if split < 2 or len(X) - split < 2:
                        print(f"⚠️ Insufficient data after split for {coin} ({timeframe}, threshold: {threshold})")
                        continue
                    X_train, X_val = X[:split], X[split:]
                    y_train, y_val = y[:split], y[split:]
                    y_reg_train, y_reg_val = y_reg[:split], y[split:]

                    # Train classifier
                    clf = None
                    if len(np.unique(y_train)) >= 2 and len(np.unique(y_val)) >= 2:
                        class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if 1 in y_train else 1.0
                        clf = XGBClassifier(
                            n_estimators=100, max_depth=3, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                            early_stopping_rounds=10, random_state=42,
                            scale_pos_weight=class_ratio
                        )
                        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                        y_pred = clf.predict(X_val)
                        f1 = f1_score(y_val, y_pred)
                        print(f"✅ Classifier for {coin} ({timeframe}, threshold: {threshold}) - F1 Score: {f1:.4f}")
                    else:
                        print(f"⚠️ Skipping classifier for {coin} ({timeframe}, threshold: {threshold}) - only one class")

                    # Train regressor
                    reg = XGBRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        early_stopping_rounds=20, verbosity=0
                    )
                    reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)], verbose=False)
                    y_reg_pred = reg.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
                    print(f"✅ Regressor for {coin} ({timeframe}, threshold: {threshold}) - RMSE: {rmse:.5f}")

                    # Save models
                    features = list(threshold_trades[0]['features'][timeframe].keys())
                    self.save_state(coin, timeframe, threshold, clf, reg, features)

    def save_state(self, coin=None, timeframe=None, threshold=None, clf=None, reg=None, features=None, score=None):
        """Persist models, trade history, and metadata to disk."""
        try:
            # Ensure model directory exists
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Save specific model if provided
            if coin and timeframe and threshold is not None:
                model_key = (coin, timeframe, threshold)
                model_id = f"{coin}_{timeframe}_{threshold:.4f}"
                clf_path = os.path.join(MODEL_DIR, f"{model_id}_clf.joblib") if clf else None
                reg_path = os.path.join(MODEL_DIR, f"{model_id}_reg.joblib") if reg else None
                
                if clf:
                    dump(clf, clf_path)
                if reg:
                    dump(reg, reg_path)
                
                self.model_metadata[model_id] = {
                    'coin': coin,
                    'timeframe': timeframe,
                    'threshold': threshold,
                    'clf_path': clf_path,
                    'reg_path': reg_path,
                    'features': features or [],
                    'last_retrain': datetime.now(EST).isoformat(),
                    'score': score or 0.0
                }
                self.models[model_key] = {
                    'clf': clf,
                    'reg': reg,
                    'features': features or [],
                    'score': score or 0.0
                }
                print(f"✅ Saved model for {coin} ({timeframe}, threshold: {threshold:.4f})")
            
            # Save all models if no specific model provided
            else:
                for model_key, data in self.models.items():
                    coin, timeframe, threshold = model_key
                    model_id = f"{coin}_{timeframe}_{threshold:.4f}"
                    clf_path = os.path.join(MODEL_DIR, f"{model_id}_clf.joblib")
                    reg_path = os.path.join(MODEL_DIR, f"{model_id}_reg.joblib")
                    if data['clf']:
                        dump(data['clf'], clf_path)
                    if data['reg']:
                        dump(data['reg'], reg_path)
                    self.model_metadata[model_id] = {
                        'coin': coin,
                        'timeframe': timeframe,
                        'threshold': threshold,
                        'clf_path': clf_path if data['clf'] else None,
                        'reg_path': reg_path if data['reg'] else None,
                        'features': data['features'],
                        'last_retrain': datetime.now(EST).isoformat(),
                        'score': data.get('score', 0.0)
                    }
                print(f"✅ Saved {len(self.models)} models to disk")

            # Save trade history
            try:
                with open(TRADE_HISTORY_FILE, 'w') as f:
                    json.dump(self.trade_history, f, indent=2)
                print(f"✅ Saved trade history with {len(self.trade_history)} entries")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving trade history: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving trade history: {str(e)}")

            # Save model metadata
            try:
                with open(MODEL_METADATA_FILE, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)
                print("✅ Saved model metadata")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving model metadata: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving model metadata: {str(e)}")

            # GitHub Actions integration
            if GITHUB_ACTIONS:
                print("Persisting files for GitHub Actions...")
                try:
                    subprocess.run(['git', 'config', '--global', 'user.email', 'actions@github.com'], check=True)
                    subprocess.run(['git', 'config', '--global', 'user.name', 'GitHub Actions'], check=True)
                    subprocess.run(['git', 'add', 'crypto/spot/models/*.joblib', 'crypto/spot/model_metadata.json', 'crypto/spot/trade_history.json', 'crypto/spot/sentiment_cache.pkl'], check=True)
                    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
                    if result.stdout.strip():
                        commit_message = f"Update models and trade history {datetime.now(EST).isoformat()}"
                        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
                        repo_url = os.getenv('GITHUB_REPOSITORY')
                        if repo_url:
                            token = os.getenv('GITHUB_TOKEN', '')
                            if not token:
                                raise ValueError("GITHUB_TOKEN not set for Git push")
                            auth_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
                            subprocess.run(['git', 'push', auth_url, 'HEAD'], check=True)
                            print("✅ Pushed changes to GitHub repository")
                        else:
                            raise ValueError("GITHUB_REPOSITORY not set")
                    else:
                        print("ℹ️ No changes to commit")
                except (subprocess.CalledProcessError, ValueError) as e:
                    print(f"⚠️ Git operation failed: {str(e)}")
                    self.send_telegram_message(f"Git operation failed: {str(e)}")

        except Exception as e:
            print(f"⚠️ Error saving state: {str(e)}")
    
    # ------------- Candidate Scanning and Mapping ------------- #
    def scan_candidates(self):
        """Find potential trading candidates using CoinGecko API and include sentiment scores."""
        print('🔍 Scanning for candidates...')
        try:
            coins = cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=1000,
                price_change_percentage='1h'
            )

            sorted_coins = []
            for c in coins:
                if c['price_change_percentage_1h_in_currency'] > 1.5 and c['price_change_percentage_24h'] > 3: # Min % hourly change
                    coin = {
                        'id': c['id'],
                        'symbol': c['symbol'].upper(),
                        'price_change_percentage_1h': c['price_change_percentage_1h_in_currency'],
                        'price_change_percentage_24h': c['price_change_percentage_24h'],
                    }
                    sorted_coins.append(coin)
            sorted_coins = self.map_coingecko_to_exchange(sorted_coins)

            sorted_coins = sorted_coins[:10]  # Limit to top 10 candidates
            for c in sorted_coins:
                c['sentiment'] = self.fetch_news_sentiment(c['symbol'], c['id'])
                time.sleep(0.5)
            sorted_coins = sorted(
                sorted_coins,
                key=lambda x: (x['price_change_percentage_1h'] + x['sentiment'], x['price_change_percentage_24h']),
                reverse=True
            )
            
            return sorted_coins

        except Exception as e:
            print(f'⚠️ Error scanning candidates: {str(e)}')
            return []

    def get_coinbase_pairs(self):
        """Fetch available trading pairs from Coinbase."""
        if self.coinbase_pairs is None:
            try:
                url = "https://api.exchange.coinbase.com/products"
                response = requests.get(url)
                response.raise_for_status()
                products = response.json()
                self.coinbase_pairs = {p['id']: p['base_currency'] for p in products if p['quote_currency'] in ['USD', 'USDT']}
                print(f"✅ Loaded {len(self.coinbase_pairs)} Coinbase USD/USDT pairs")
            except Exception as e:
                print(f"⚠️ Error fetching Coinbase pairs: {str(e)}")
                self.coinbase_pairs = {}
        return self.coinbase_pairs

    def get_binanceus_pairs(self):
        """Fetch available trading pairs from Binance US."""
        if self.binanceus_pairs is None:
            try:
                markets = binanceus.load_markets()
                self.binanceus_pairs = {symbol: market['base'] for symbol, market in markets.items() if market['quote'] in ['USD', 'USDT'] and market['active']}
                print(f"✅ Loaded {len(self.binanceus_pairs)} Binance US USD/USDT pairs")
            except Exception as e:
                print(f"⚠️ Error fetching Binance US pairs: {str(e)}")
                self.binanceus_pairs = {}
        return self.binanceus_pairs

    def fetch_news_sentiment(self, coin_symbol, coin_id=None, max_retries=3):
        """Fetch up to 100 recent news articles for the given coin and compute sentiment score."""
        global RATE_LIMIT_HIT
        cache_key = f"{coin_symbol}:{coin_id or ''}"
        current_time = datetime.now(pytz.UTC).timestamp()

        # Check cache
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            if current_time - cached['timestamp'] < SENTIMENT_CACHE_TTL:
                print(f"✅ Using cached sentiment for {coin_symbol}: {cached['sentiment']:.3f}")
                return cached['sentiment']
        if RATE_LIMIT_HIT:
            print(f"⚠️ Skipping sentiment for {coin_symbol} due to previous rate limit")
            return 0.0

        try:
            if not NEWS_API_KEY:
                print("⚠️ News API key not found, skipping sentiment analysis")
                return 0.0

            query = f"{coin_symbol} OR {coin_id or coin_symbol} crypto OR cryptocurrency"
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=100&apiKey={NEWS_API_KEY}"
            from_date = (datetime.now(pytz.UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
            params = {'from': from_date}

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    articles = data.get('articles', [])
                    print(f"✅ Found {len(articles)} articles for {coin_symbol}")
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:
                        print(f"⚠️ NewsAPI rate limit hit for {coin_symbol} (attempt {attempt + 1}/{max_retries})")
                        if attempt == max_retries - 1:
                            RATE_LIMIT_HIT = True
                            print(f"⚠️ Global rate limit set, skipping further sentiment fetches")
                            return 0.0
                        time.sleep(2 ** (attempt + 1))
                        continue
                    print(f"⚠️ HTTP error for {coin_symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2 ** attempt)
                    continue
                except Exception as e:
                    print(f"⚠️ Error fetching news for {coin_symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
            else:
                print(f"⚠️ Failed to fetch news for {coin_symbol} after {max_retries} attempts")
                return 0.0

            if not articles:
                print(f"⚠️ No recent news found for {coin_symbol}")
                return 0.0

            relevant_articles = []
            for article in articles:
                title = article.get('title', '') or ''
                description = article.get('description', '') or ''
                text = f"{title} {description}".lower()
                if coin_symbol.lower() in text or (coin_id and coin_id.lower() in text) or 'crypto' in text:
                    relevant_articles.append(article)
            
            if not relevant_articles:
                print(f"⚠️ No relevant news articles found for {coin_symbol}")
                return 0.0

            sentiment_scores = []
            weights = []
            now = datetime.now(pytz.UTC)
            for article in relevant_articles[:100]:
                title = article.get('title', '') or ''
                description = article.get('description', '') or ''
                text = f"{title} {description}"
                if text.strip():
                    try:
                        analysis = TextBlob(text)
                        sentiment_scores.append(analysis.sentiment.polarity)
                        published_at = datetime.strptime(article.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.UTC)
                        hours_old = (now - published_at).total_seconds() / 3600
                        weight = max(1.0 - hours_old / 24.0, 0.1)
                        weights.append(weight)
                    except Exception as e:
                        print(f"⚠️ Error processing article for {coin_symbol}: {str(e)}")
                        continue
            
            if not sentiment_scores:
                print(f"⚠️ No valid sentiment scores for {coin_symbol}")
                return 0.0

            weighted_sentiment = np.average(sentiment_scores, weights=weights)
            print(f"📰 Sentiment score for {coin_symbol}: {weighted_sentiment:.3f} (based on {len(sentiment_scores)} articles)")
            
            self.sentiment_cache[cache_key] = {'sentiment': weighted_sentiment, 'timestamp': current_time}
            self.save_sentiment_cache()
            return weighted_sentiment

        except Exception as e:
            print(f"⚠️ Error fetching news sentiment for {coin_symbol}: {str(e)}")
            return 0.0

    def map_coingecko_to_exchange(self, coins):
        """Map CoinGecko coins to valid Coinbase and Binance trading pairs."""
        cg_cb_sorted_coins = []
        coinbase_pairs = self.get_coinbase_pairs()
        binanceus_pairs = self.get_binanceus_pairs()
        
        try:
            for coin in coins:
                symbol = coin['symbol'].upper()
                coin_id = coin['id'].lower().replace('-', '')
                cb_found_pair = None
                bin_found_pair = None
                coin_copy = coin.copy()

                # Try Coinbase
                for pair, base_currency in coinbase_pairs.items():
                    base_currency_normalized = base_currency.lower().replace('-', '')
                    if symbol.lower() == base_currency_normalized or coin_id == base_currency_normalized:
                        cb_found_pair = pair
                        coin_copy['coinbase_symbol'] = cb_found_pair
                        print(f"✅ Found Coinbase pair for {symbol} ({coin['id']}): {cb_found_pair}")
                        break

                # Try Binance US
                for pair, base_currency in binanceus_pairs.items():
                    base_currency_normalized = base_currency.lower().replace('-', '')
                    if symbol.lower() == base_currency_normalized or coin_id == base_currency_normalized:
                        bin_found_pair = pair
                        coin_copy['binance_symbol'] = bin_found_pair
                        print(f"✅ Found Binance US pair for {symbol} ({coin['id']}): {bin_found_pair}")
                        break

                # Fetch data to check freshness
                cb_data_1m = None
                bin_data_1m = None
                cb_time_diff = float('inf')
                bin_time_diff = float('inf')

                if cb_found_pair:
                    cb_data_1m = self.fetch_data(cb_found_pair, '1m', 'coinbase', limit=1)
                    if cb_data_1m is not None and not cb_data_1m.empty:
                        cb_latest_time = cb_data_1m.index[-1]
                        cb_time_diff = (datetime.now(EST) - cb_latest_time).total_seconds() / 60
                        print(f"📅 Coinbase data for {cb_found_pair}: {cb_time_diff:.2f} minutes old")

                if bin_found_pair:
                    bin_data_1m = self.fetch_data(bin_found_pair, '1m', 'binanceus', limit=1)
                    if bin_data_1m is not None and not bin_data_1m.empty:
                        bin_latest_time = bin_data_1m.index[-1]
                        bin_time_diff = (datetime.now(EST) - bin_latest_time).total_seconds() / 60
                        print(f"📅 Binance US data for {bin_found_pair}: {bin_time_diff:.2f} minutes old")

                # Select the exchange with fresher data
                if cb_data_1m is not None or bin_data_1m is not None:
                    if cb_time_diff <= bin_time_diff and cb_time_diff <= 60:
                        coin_copy['exchange'] = 'coinbase'
                        coin_copy['selected_symbol'] = cb_found_pair
                        print(f"✅ Selected Coinbase for {symbol} ({coin['id']}) as fresher data ({cb_time_diff:.2f} min vs {bin_time_diff:.2f} min)")
                    elif bin_time_diff <= 60:
                        coin_copy['exchange'] = 'binanceus'
                        coin_copy['selected_symbol'] = bin_found_pair
                        print(f"✅ Selected Binance US for {symbol} ({coin['id']}) as fresher data ({bin_time_diff:.2f} min vs {cb_time_diff:.2f} min)")
                    else:
                        print(f"⚠️ No fresh data for {symbol} ({coin['id']}) on either exchange, skipping")
                        continue
                    cg_cb_sorted_coins.append(coin_copy)
                else:
                    print(f"⚠️ No valid data for {symbol} ({coin['id']}) on either exchange, skipping")

            return cg_cb_sorted_coins
        
        except Exception as e:
            print(f"⚠️ Error mapping CoinGecko to exchanges: {str(e)}")
            return []

    # ------------- Data Fetching and Processing ------------- # 
    def fetch_data(self, symbol, interval, exchange, limit=MAX_WINDOW_MINUTES, max_retries=3, start_time=None, end_time=None):
        """Fetch OHLCV data from specified exchange (Coinbase or Binance US) for a given time range."""
        if exchange == 'coinbase':
            return self.fetch_coinbase_data(symbol, interval, limit, max_retries, start_time, end_time)
        elif exchange == 'binanceus':
            return self.fetch_binance_data(symbol, interval, limit, max_retries, start_time, end_time)
        else:
            print(f"⚠️ Unknown exchange: {exchange}")
            return None

    def fetch_binance_data(self, symbol, interval, limit=MAX_WINDOW_MINUTES, max_retries=3, start_time=None, end_time=None):
        """Fetch OHLCV data from Binance via ccxt for a specific time range."""
        timeframe = {'1m': '1m', '1h': '1h'}.get(interval, '1m')
        min_candles = 20 if interval == '1h' else 100  # Require more candles for 1h to ensure sufficient data
        try:
            all_ohlcv = []
            current_limit = min(limit, 500)  # Binance API limit per request
            since = None
            if start_time:
                # Convert start_time to milliseconds since epoch (UTC)
                since = int(start_time.astimezone(pytz.UTC).timestamp() * 1000)
            
            while len(all_ohlcv) < limit:
                for attempt in range(max_retries):
                    try:
                        ohlcv = binanceus.fetch_ohlcv(symbol, timeframe, since=since, limit=current_limit)
                        if not ohlcv:
                            print(f"⚠️ No Binance US data for {symbol} at attempt {attempt + 1}/{max_retries}")
                            return None
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(EST)
                        df.set_index('timestamp', inplace=True)

                        # Filter by end_time if provided
                        if end_time:
                            df = df[df.index <= end_time.astimezone(EST)]

                        all_ohlcv.extend(ohlcv)
                        latest_time = df.index[-1]
                        current_time = datetime.now(EST)
                        time_diff = (current_time - latest_time).total_seconds() / 60
                        print(f"📅 Raw Binance US timestamp for {symbol}: {ohlcv[-1][0]} (converted: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")

                        if time_diff > 60 and not start_time:
                            print(f"⚠️ Binance US data for {symbol} is stale (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}, {time_diff:.2f} minutes old)")
                            return None

                        print(f"✅ Fetched {len(ohlcv)} Binance US candles for {symbol} (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
                        all_ohlcv = all_ohlcv[-limit:]  # Keep only the most recent candles up to limit
                        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(EST)
                        df.set_index('timestamp', inplace=True)
                        if len(df) < min_candles:
                            print(f"⚠️ Insufficient candles ({len(df)} < {min_candles}) for {symbol} ({interval})")
                            return None
                        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    except Exception as e:
                        print(f"⚠️ Binance US fetch error for {symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                        continue
                else:
                    print(f"⚠️ Failed to fetch Binance US data for {symbol} after {max_retries} attempts")
                    return None
                # Update 'since' for pagination
                if all_ohlcv:
                    since = int(all_ohlcv[-1][0]) + 1  # Next timestamp in milliseconds
                else:
                    break
            return None
        except Exception as e:
            print(f"⚠️ Error fetching Binance US data for {symbol}: {str(e)}")
            return None

    def fetch_coinbase_data(self, symbol, interval, limit=MAX_WINDOW_MINUTES, max_retries=3, start_time=None, end_time=None):
        """Fetches recent OHLCV data from Coinbase Advanced Trade API for a specific time range."""
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        granularity_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '6h': 21600, '1d': 86400}
        granularity = granularity_map.get(interval, 60)
        min_candles = 20 if interval == '1h' else 100
        params = {'granularity': granularity}
        if start_time and end_time:
            params['start'] = start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            params['end'] = end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            params['limit'] = min(limit, 300)  # Coinbase API limit is 300 candles

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, list):
                    print(f"⚠️ Invalid Coinbase data for {symbol}: {data}")
                    return None

                df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(EST)
                df.set_index('timestamp', inplace=True)

                if len(df) < min_candles:
                    print(f"⚠️ Insufficient candles ({len(df)} < {min_candles}) for {symbol} ({interval})")
                    return None

                latest_time = df.index[-1]
                current_time = datetime.now(EST)
                time_diff = (current_time - latest_time).total_seconds() / 60
                print(f"📅 Raw Coinbase timestamp for {symbol}: {data[-1][0]} (converted: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
                if time_diff > 60 and not start_time:
                    print(f"⚠️ Coinbase data for {symbol} is stale (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}, {time_diff:.2f} minutes old)")
                    return None
                
                print(f"✅ Fetched {len(df)} Coinbase candles for {symbol} (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
                return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            except Exception as e:
                print(f"⚠️ Coinbase fetch error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        return None

    def calculate_features(self, df, df_type):
        """Add technical indicators and sentiment score to the DataFrame."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {df_type}, cannot calculate features")
            return None

        lags = [1, 3, 5]
        rolling_windows = [3, 6]

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

            if 'sentiment' not in df.columns:
                df['sentiment'] = 0.0

            # Add time series features
            base_features = ['open', 'high', 'low', 'volume', 'returns', 'volatility', 
                             'rsi', 'macd', 'bollinger', 'atr', 'sma_20', 'ema_20', 
                             'adx', 'obv', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'sentiment']

            for col in base_features:
                # Lag features
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

                # Rolling window features
                for window in rolling_windows:
                    df[f'{col}_roll_mean_{window}'] = df[col].shift(1).rolling(window=window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].shift(1).rolling(window=window).std()

            # Clean data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df = df.loc[~df[['open', 'high', 'low', 'close', 'volume']].duplicated()]

            # Drop zero-volume candles (optional for hourly, recommended for 1m)
            df = df[df['volume'] > 0]

            # Remove rows where OHLC values are identical for multiple rows
            df = df[~((df['open'] == df['close']) & (df['high'] == df['low']) & (df['volume'] == 0))]
            
            if df.empty:
                print(f"⚠️ DataFrame for {df_type} is empty after cleaning")
                return None

            print(f'✅ Technical indicators added for {df_type} data')
            return df

        except Exception as e:
            print(f'⚠️ Error in calculate_features for {df_type}: {str(e)}')
            return None
    
    def label_1m_model(self, coin, df, window=360, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-minute model with dynamic threshold."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {coin['symbol']} (1m), cannot label")
            return None
        try:
            print(f"📊 Attempting to label 1m model for {coin['symbol']}")

            df = df.copy()
            df['future_max'] = df['close'].shift(-1).rolling(window=window, min_periods=1).max()
            df['target_return'] = df['future_max'] / df['close'] - 1
            df['target_return'] = df['target_return'].clip(lower=0, upper=max_return)
            
            # Dynamic threshold: Use 25th percentile, fallback to mean, then min_return
            valid_returns = df['target_return'].dropna()
            if len(valid_returns) > 0:
                dynamic_threshold = np.percentile(valid_returns, 10)  # Lower percentile for more balance
                if dynamic_threshold < min_return:
                    dynamic_threshold = valid_returns.mean() + valid_returns.std() if valid_returns.mean() > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)
                print(f"📈 Dynamic threshold for {coin['symbol']}: {dynamic_threshold:.4f}")
            else:
                dynamic_threshold = min_return
                print(f"⚠️ No valid returns for {coin['symbol']}, using default threshold: {min_return}")
            
            df['label'] = (df['target_return'] >= dynamic_threshold).astype(int)
            labeled_df = df.dropna()
            
            # Check class distribution
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1m labels for {coin['symbol']}, will rely on regressor")
            
            return labeled_df

        except Exception as e:
            print(f'⚠️ Error preparing 1m data: {str(e)}')
            return None

    def label_1h_model(self, coin, df, forward_hours=6, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-hour model with dynamic threshold and target_return."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {coin['symbol']} (1h), cannot label")
            return None
        try:
            print(f"📊 Attempting to label 1h model for {coin['symbol']}")
            df = df.copy()
            df['fwd_return'] = df['close'].shift(-forward_hours) / df['close'] - 1
            df['future_max'] = df['close'].shift(-1).rolling(window=forward_hours, min_periods=1).max()
            df['target_return'] = df['future_max'] / df['close'] - 1
            df['target_return'] = df['target_return'].clip(lower=0, upper=max_return)
            df['fwd_return'] = df['fwd_return'].clip(lower=-max_return, upper=max_return)
 
            # Dynamic threshold: Use 25th percentile, fallback to mean, then min_return
            valid_returns = df['fwd_return'].dropna()
            if len(valid_returns) > 0:
                dynamic_threshold = np.percentile(valid_returns, 10)  # Lower percentile for more balance
                if dynamic_threshold < min_return:
                    dynamic_threshold = valid_returns.mean() + valid_returns.std() if valid_returns.mean() > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)
                print(f"📈 Dynamic threshold for {coin['symbol']}: {dynamic_threshold:.4f}")
            else:
                dynamic_threshold = min_return
                print(f"⚠️ No valid returns for {coin['symbol']}, using default threshold: {min_return}")
            
            df['label'] = (df['fwd_return'] >= dynamic_threshold).astype(int)
            labeled_df = df.dropna()
            
            # Check class distribution
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1h labels for {coin['symbol']}, will rely on regressor")
            
            return labeled_df
        
        except Exception as e:
            print(f'⚠️ Error preparing 1h data: {str(e)}')
            return None

    def train_hybrid_model(self, df, coin, df_type):
        """Train a hybrid model using XGBoost, with fallback to regressor if classification fails."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {coin['symbol']} ({df_type}), cannot train model")
            return None, None, None
        drop_cols = ['close', 'future_max', 'fwd_return']

        try:
            print(f"📊 Training {coin['symbol']} hybrid model for {df_type} data")

            df = df.copy()
            features = [col for col in df.columns if col not in ['label', 'target_return'] + drop_cols and pd.api.types.is_numeric_dtype(df[col])]
            X = df[features]
            y_class = df['label']
            y_reg = df['target_return'] if 'target_return' in df.columns else None

            split = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_class_train, y_class_test = y_class.iloc[:split], y_class.iloc[split:]

            # Classifier training with SMOTE
            best_class_result = None
            if y_class_train.nunique() >= 2 and y_class_test.nunique() >= 2:
                minority_count = len(y_class_train[y_class_train == 1])
                k_neighbors = min(5, max(1, minority_count - 1))
                try:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_res, y_class_train_res = smote.fit_resample(X_train, y_class_train)
                except ValueError as e:
                    print(f"⚠️ SMOTE failed for {df_type}: {str(e)}, trying ADASYN")
                    try:
                        adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                        X_train_res, y_class_train_res = adasyn.fit_resample(X_train, y_class_train)
                    except ValueError as e:
                        print(f"⚠️ ADASYN also failed for {df_type}: {str(e)}, proceeding without oversampling")
                        X_train_res, y_class_train_res = X_train, y_class_train
                class_ratio = len(y_class_train_res[y_class_train_res == 0]) / len(y_class_train_res[y_class_train_res == 1]) if 1 in y_class_train_res.value_counts() else 1.0
                for reg_alpha in [0, 0.1]:
                    for reg_lambda in [0.5, 1.0]:
                        for gamma in [0, 1]:
                            clf = XGBClassifier(
                                n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric='auc', early_stopping_rounds=10,
                                random_state=42, base_score=0.5,
                                reg_alpha=reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
                                scale_pos_weight=class_ratio
                            )

                            clf.fit(X_train_res, y_class_train_res, eval_set=[(X_test, y_class_test)], verbose=False)
                            y_pred = clf.predict(X_test)
                            score = f1_score(y_class_test, y_pred)

                            if not best_class_result or score > best_class_result['score']:
                                best_class_result = {
                                    'score': score,
                                    'model': clf,
                                    'y_class_test': y_class_test,
                                    'y_pred': y_pred,
                                    'params': {
                                        'reg_alpha': reg_alpha,
                                        'reg_lambda': reg_lambda,
                                        'gamma': gamma
                                    }
                                }
                print('\nBest Classifier Report:')
                print(classification_report(best_class_result['y_class_test'], best_class_result['y_pred']))
            else:
                print(f'⚠️ Skipping classifier training for {df_type} — only one class in training/test set')

            # Regressor training (always attempt if target_return exists)
            best_reg_result = None
            if y_reg is not None and hasattr(y_reg, 'isnull') and not y_reg.isnull().all():
                y_reg_train, y_reg_test = y_reg.iloc[:split], y_reg.iloc[split:]
                
                for max_depth in [4, 6, 8]:
                    for reg_alpha in [0, 0.1, 0.5]:
                        for reg_lambda in [0.5, 1.0, 2.0]:
                            regressor = XGBRegressor(
                                n_estimators=200,
                                max_depth=max_depth,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=reg_alpha,
                                reg_lambda=reg_lambda,
                                random_state=42,
                                early_stopping_rounds=20,
                                verbosity=0
                            )
                            regressor.fit(
                                X_train, y_reg_train,
                                eval_set=[(X_test, y_reg_test)],
                                verbose=False
                            )
                            preds = regressor.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_reg_test, preds))

                            if not best_reg_result or rmse < best_reg_result['rmse']:
                                best_reg_result = {
                                    'rmse': rmse,
                                    'model': regressor,
                                    'params': {
                                        'max_depth': max_depth,
                                        'reg_alpha': reg_alpha,
                                        'reg_lambda': reg_lambda
                                    }
                                }
                print(f"\nBest Regressor RMSE: {best_reg_result['rmse']:.5f}")
            else:
                print(f'❕ Skipping regressor for {df_type} — no valid "target_return" values')

            return best_class_result, best_reg_result, features
        
        except Exception as e:
            print(f'⚠️ Error training hybrid model: {str(e)}')
            return None, None, None

    def predict_future_movement(self, expected_return, sentiment):
        """Predict the future movement of the best coin by simulating realistic data points."""
        print(f'\n🔮 Predicting future movement for {best_coin['symbol']}')

        # Ensure latest timestamp is in EST
        latest_timestamp = best_1m_df.index[-1]
        if not latest_timestamp.tzinfo:
            latest_timestamp = latest_timestamp.tz_localize('UTC').tz_convert(EST)
        else:
            latest_timestamp = latest_timestamp.tz_convert(EST)
        
        # Validate data freshness
        current_time = datetime.now(EST)
        time_diff = (current_time - latest_timestamp).total_seconds() / 60
        if time_diff > 60:  # More than 1 hour old
            print(f"⚠️ Warning: Latest data timestamp {latest_timestamp.strftime('%m-%d-%Y %H:%M:%S %Z%z')} is {time_diff:.2f} minutes old, may be stale")
            return None, None
        print(f"📅 Latest data timestamp: {latest_timestamp.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
        
        # Parameters
        max_minutes = 360  # 6 hours
        threshold = expected_return  # Use the regressor's expected return as the threshold
        
        # Get recent trends for simulation
        recent_data = best_1m_df.tail(20)
        avg_return = recent_data['returns'].mean()
        avg_volume = recent_data['volume'].mean()
        vol_volatility = recent_data['volume'].std() / avg_volume if avg_volume > 0 else 0.1

        # Fit GARCH(1,1) model for volatility simulation
        try:
            garch_model = arch.arch_model(recent_data['returns'].dropna() * 100, vol='GARCH', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            vol_forecast = garch_fit.forecast(horizon=1)
            init_vol = np.sqrt(vol_forecast.variance.values[-1, 0]) / 100
        except Exception as e:
            print(f'⚠️ GARCH fitting failed: {str(e)}, using historical volatility')
            init_vol = recent_data['volatility'].mean()
        
        # Initialize DataFrame for simulation
        sim_data = best_1m_df[['open', 'high', 'low', 'close', 'volume']].copy()
        last_row = sim_data.iloc[-1]
        current_vol = init_vol

        # Simulate future candles
        for minute in range(1, max_minutes + 1):
            next_time = latest_timestamp + timedelta(minutes=minute)
            
            # Update volatility with GARCH
            try:
                garch_forecast = garch_model.fit(disp='off', first_obs=sim_data['returns'].dropna()[-20:]).forecast(horizon=1)
                current_vol = np.sqrt(garch_forecast.variance.values[-1, 0]) / 100
            except:
                pass  # Keep current_vol if GARCH fails

            # Simulate price movement with t-distribution (fat-tailed)
            df = 5  # Degrees of freedom for t-distribution (lower = fatter tails)
            price_change = t.rvs(df, loc=avg_return + sentiment * 0.001, scale=current_vol)
            
            # Introduce extreme events (e.g., 5-10% price jump/drop)
            if np.random.random() < 0.02:  # 2% chance per minute
                extreme_move = np.random.choice([-0.10, -0.05, 0.05, 0.10])  # Random ±5% or ±10% move
                price_change += extreme_move

            new_close = last_row['close'] * (1 + price_change)
            new_open = last_row['close']
            new_high = max(new_open, new_close) * (1 + np.random.uniform(0, current_vol / 2))
            new_low = min(new_open, new_close) * (1 - np.random.uniform(0, current_vol / 2))
            
            # Volume spike during extreme events
            volume_factor = 1.5 if abs(price_change) > 0.05 else 1.0
            new_volume = max(avg_volume * volume_factor * (1 + np.random.normal(0, vol_volatility)), 0)
            
            # Append new candle with sentiment
            new_row = pd.DataFrame({
                'open': [new_open],
                'high': [new_high],
                'low': [new_low],
                'close': [new_close],
                'volume': [new_volume],
                'sentiment': [sentiment]
            }, index=[next_time])
            
            sim_data = pd.concat([sim_data, new_row])
            last_row = new_row.iloc[0]
        
        # Recalculate features for the entire simulated dataset
        sim_data = self.calculate_features(sim_data, '1m')
        if sim_data is None:
            print('⚠️ Error recalculating features for simulated data')
            return None, None
        
        # Extract future data
        future_data = sim_data.loc[sim_data.index > latest_timestamp][best_analysis['feat_1m']]
        
        # Predict returns for each future minute
        found_signal = False
        signal_time = None
        predicted_return = None

        for i, (timestamp, features) in enumerate(future_data.iterrows()):
            if i >= max_minutes:
                break
            features_df = pd.DataFrame([features], columns=best_analysis['feat_1m'], index=[timestamp])
            predicted_return = best_analysis['reg_1m']['model'].predict(features_df)[0] if best_analysis['reg_1m'] is not None else 0.0
            
            if predicted_return >= threshold:
                signal_time = timestamp
                found_signal = True
                break
        
        if not found_signal:
            print(f'❌ No sell signal found for {best_coin['symbol']} within 6 hours')

        return signal_time, predicted_return

# ------------- MAIN FUNCTION ------------- #
    def run_analysis(self):
        """Main analysis pipeline with sentiment-integrated buy score."""
        global best_score, best_coin, best_analysis, best_1m_df, best_1h_df
        print('🚀 Starting trading algorithm')
        current_time = datetime.now(EST)
        print(f"📅 Analysis started at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
        
        # Scan for candidates
        candidates = self.scan_candidates()

        if not candidates:
            print(f"❌ No candidates found at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
            return

        # Analyze each candidate
        for coin in candidates:
            print(f"\n📊 Analyzing {coin['symbol']} ({coin['selected_symbol']} on {coin['exchange']})")
            df_1m = self.fetch_data(coin['selected_symbol'], '1m', coin['exchange'])
            df_1h = self.fetch_data(coin['selected_symbol'], '1h', coin['exchange'])
            
            if df_1m is None or df_1h is None:
                print(f"⚠️ Skipping {coin['symbol']} due to data fetch error")
                continue

            # Add sentiment to dataframes
            df_1m['sentiment'] = coin['sentiment']
            df_1h['sentiment'] = coin['sentiment']

            # Calculate features
            df_1m, df_1h = self.calculate_features(df_1m, '1m'), self.calculate_features(df_1h, '1h')

            if df_1m is None or df_1h is None:
                print(f"⚠️ Skipping {coin['symbol']} due to calculation error")
                continue

            # Label data
            labeled_1m = self.label_1m_model(coin, df_1m)
            labeled_1h = self.label_1h_model(coin, df_1h)

            if labeled_1m is None or labeled_1h is None:
                print(f"⚠️ Skipping {coin['symbol']} due to labeling error")
                continue

            # Train models
            clf_1m, reg_1m, feat_1m = self.train_hybrid_model(labeled_1m, coin, '1m')
            clf_1h, reg_1h, feat_1h = self.train_hybrid_model(labeled_1h, coin, '1h')

            # Use regressor score if classifier fails
            score = clf_1m['score'] if clf_1m is not None else (reg_1m['rmse'] if reg_1m is not None else -np.inf)
            if score > best_score:
                best_score = score
                best_coin = coin
                best_analysis = {
                    'clf_1m': clf_1m,
                    'reg_1m': reg_1m,
                    'feat_1m': feat_1m,
                    'clf_1h': clf_1h,
                    'reg_1h': reg_1h,
                    'feat_1h': feat_1h,
                }
                best_1m_df = labeled_1m.copy()
                best_1h_df = labeled_1h.copy()

                print(f"🏆 New best: {coin['symbol']} ({coin['selected_symbol']} on {coin['exchange']}) (Score: {best_score:.4f})")
        
        if best_analysis is None:
            print(f"❌ No valid models found at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
            return
        
        # Run final analysis on the best coin
        print(f"\n🏆 Running final analysis on {best_coin['symbol']} ({best_coin['selected_symbol']} on {best_coin['exchange']})")
        
        # Get the latest row from each labeled dataframe
        latest_1m = best_1m_df.iloc[[-1]][best_analysis['feat_1m']]
        latest_1h = best_1h_df.iloc[[-1]][best_analysis['feat_1h']]

        # Make predictions - Probability of class 1
        prob_1m = best_analysis['clf_1m']['model'].predict_proba(latest_1m)[0][1] if best_analysis['clf_1m'] is not None else 0.5
        prob_1h = best_analysis['clf_1h']['model'].predict_proba(latest_1h)[0][1] if best_analysis['clf_1h'] is not None else 0.5

        # Combine predictions with sentiment
        sentiment_score = best_coin['sentiment']
        expected_return = (best_analysis['reg_1m']['model'].predict(latest_1m)[0] if best_analysis['reg_1m'] is not None else 0.0) or \
                          (best_analysis['reg_1h']['model'].predict(latest_1h)[0] if best_analysis['reg_1h'] is not None else 0.0)
        
        if best_analysis['clf_1m'] is None and best_analysis['reg_1m'] is not None:
            prob_1m = min(expected_return / THRESHOLDS[0], 1.0)  # Normalize regressor output
        if best_analysis['clf_1h'] is None and best_analysis['reg_1h'] is not None:
            prob_1h = min(expected_return / THRESHOLDS[0], 1.0)  # Normalize regressor output
        buy_score = W_1M * prob_1m + W_1H * prob_1h + W_SENTIMENT * (sentiment_score + 1) / 2
        buy = buy_score >= 0.45 or (expected_return >= THRESHOLDS[0])  # Relaxed thresholds

        # Position sizing (scale by confidence)
        position_size_pct = min(max(buy_score, 0.01), 1.0)
        position_size_dollars = PORTFOLIO_SIZE * position_size_pct

        if buy:
            close_price = best_1m_df['close'].iloc[-1]
            # Final buy report
            message = [
                f"✅ Action: BUY {best_coin['symbol']} ({best_coin['selected_symbol']} at {close_price:.4f} on {best_coin['exchange']})",
                f"🤖 Buy score: {buy_score:.2f} (1m: {prob_1m:.2f}, 1h: {prob_1h:.2f}, sentiment: {sentiment_score:.2f})",
                f"📈 Expected return: {expected_return * 100:.2f}%",
                f"💰 Position size: {position_size_pct * 100:.1f}% (${position_size_dollars:.2f})"
            ]

            signal_time, predicted_return = self.predict_future_movement(expected_return, sentiment_score)

            if signal_time is None or predicted_return is None:
                print(f"❌ No valid future movement prediction for {best_coin['symbol']}")
            else:
                self.record_trade(
                    coin=best_coin,
                    entry_price=best_1m_df['close'].iloc[-1],
                    expected_return=expected_return,
                    entry_time=current_time,
                    sell_time=signal_time,
                    features_1m=best_1m_df[best_analysis['feat_1m']].iloc[-1].to_dict(),
                    features_1h=best_1h_df[best_analysis['feat_1h']].iloc[-1].to_dict(),
                    buy_score=buy_score,
                    position_size_pct=position_size_pct,
                    exchange=best_coin['exchange']                   
                )

                message.extend([
                    "",
                    f"🎯 Predicted sell signal for {best_coin['symbol']} at {signal_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}",
                    f"📈 Predicted sell price: {(1 + predicted_return) * close_price::.4f} (predicted return: {predicted_return * 100:.2f}%)"
                ])

                self.send_telegram_message("\n".join(message))
        
        else:
            print(f"❌ No buy signal for best coin: {best_coin['symbol']} ({best_coin['selected_symbol']} on {best_coin['exchange']})")

if __name__ == '__main__':
    trader = CryptoTrader()
    trader.run_analysis()
    trader.evaluate_pending_trades()
