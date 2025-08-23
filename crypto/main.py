# ------------- API DOCUMENTATION ------------- #
# Binance US - https://docs.binance.us/
# Coinbase - https://api.exchange.coinbase.com/
# Kraken - https://docs.kraken.com/rest/
# OKX - https://www.okx.com/help-center/
# Crypto.com - https://exchange-docs.crypto.com/
# CoinGecko - https://docs.coingecko.com/v3.0.1/reference/introduction

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
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import hashlib
from textblob import TextBlob
from scipy.stats import t
import arch
import ccxt
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import subprocess
import glob
from xgboost.callback import EarlyStopping

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
THRESHOLDS = [0.02, 0.99]  # Min, Max
EST = pytz.timezone('America/New_York')
MODEL_DIR = 'crypto/spot/models'
TRADE_HISTORY_FILE = 'crypto/spot/trade_history.json'
MODEL_METADATA_FILE = 'crypto/spot/model_metadata.json'
SENTIMENT_CACHE_FILE = 'crypto/spot/sentiment_cache.pkl'
PORTFOLIO_FILE = 'crypto/spot/portfolio.json'
PORTFOLIO_SIZE = 10000.00
RATE_LIMIT_HIT = False
FEEDBACK_INTERVAL_HOURS = 1  # How often to retrain models
SENTIMENT_CACHE_TTL = 14400  # Cache sentiment for 4 hours
W_1M = 0.75  # Weight for 1 min df in buy score
W_1H = 0.25  # Weight for 1 hour df in buy score
W_SENTIMENT = 0.1  # Weight for sentiment in buy score
MIN_CANDLES_1M = 300
MIN_CANDLES_1H = 100
FRESHNESS_THRESHOLD_1M = 5  # Minutes  
FRESHNESS_THRESHOLD_1H = 60  # Minutes
FRESHNESS_THRESHOLD_1D = 1440  # Minutes
RATE_LIMIT_DELAY = 1  # Seconds between retries
best_score = -np.inf
best_coin = None
best_analysis = None
best_1m_df = None
best_1h_df = None
MIN_PROFITABLE_RETURN = 0.005  # 0.5% min for profitable label
FEE_RATE = 0.002  # 0.2% trading fees
MIN_TRADES_FOR_RETRAIN = 100  # Increased for better data
MIN_LIQUIDITY = 1000000  # 24h volume threshold
MAX_VOLATILITY = 0.10  # Max 10% volatility filter
NUM_MONTE_CARLO_PATHS = 100  # For simulations

# Load environment variables
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize APIs
cg = CoinGeckoAPI()

# ------------- File System Setup ---------------- #
os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------- CryptoTrader Class ---------------- #
class CryptoTrader:
    def __init__(self):
        self.models = {}  # Format: {(coin, timeframe, threshold): {'clf': model, 'reg': model, 'features': list, 'score': float}}
        self.trade_history = []
        self.portfolio = []
        self.model_metadata = {}
        self.coinbase_pairs = None
        self.binanceus_pairs = None
        self.kraken_pairs = None
        self.okx_pairs = None
        self.cryptocom_pairs = None
        self.binanceus = ccxt.binanceus({
            'enableRateLimit': True,
            'rateLimit': 100,
        })
        self.coinbase = ccxt.coinbase({
            'enableRateLimit': True,
            'rateLimit': 360,
        })
        self.kraken = ccxt.kraken({
            'enableRateLimit': True,
            'rateLimit': 3100,
        })
        self.okx = ccxt.okx({
            'enableRateLimit': True,
            'rateLimit': 50,
        })
        self.cryptocom = ccxt.cryptocom({
            'enableRateLimit': True,
            'rateLimit': 500,
        })
        self.sentiment_cache = {}
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
                        self.trade_history = json.loads(content)
                        print(f"✅ Loaded trade history with {len(self.trade_history)} entries")
                    else:
                        print("⚠️ Trade history file is empty, initializing empty trade history")
                        self.trade_history = []
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in trade history: {str(e)}, initializing empty trade history")
                self.trade_history = []
                with open(TRADE_HISTORY_FILE, 'w') as f:
                    json.dump(self.trade_history, f, indent=2)
            except Exception as e:
                print(f"⚠️ Error loading trade history: {str(e)}, initializing empty trade history")
                self.trade_history = []
                with open(TRADE_HISTORY_FILE, 'w') as f:
                    json.dump(self.trade_history, f, indent=2)
        else:
            print("⚠️ Trade history file not found, initializing empty trade history")
            self.trade_history = []
            with open(TRADE_HISTORY_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2)

        # Load portfolio
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.portfolio = json.loads(content)
                        if not isinstance(self.portfolio, list) or not self.portfolio:
                            print("⚠️ Invalid portfolio format, initializing default portfolio")
                            self.portfolio = [{'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()}]
                        print(f"✅ Loaded portfolio with {len(self.portfolio)} portfolios, latest size ${self.portfolio[-1]['portfolio_size']:.2f}")
                    else:
                        print("⚠️ Portfolio file is empty, initializing default portfolio")
                        self.portfolio = [{'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()}]
                        with open(PORTFOLIO_FILE, 'w') as f:
                            json.dump(self.portfolio, f, indent=2)
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in portfolio: {str(e)}, initializing default portfolio")
                self.portfolio = [{'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()}]
                with open(PORTFOLIO_FILE, 'w') as f:
                    json.dump(self.portfolio, f, indent=2)
            except Exception as e:
                print(f"⚠️ Error loading portfolio: {str(e)}, initializing default portfolio")
                self.portfolio = [{'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()}]
                with open(PORTFOLIO_FILE, 'w') as f:
                    json.dump(self.portfolio, f, indent=2)
        else:
            print("⚠️ Portfolio file not found, initializing default portfolio")
            self.portfolio = [{'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()}]
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(self.portfolio, f, indent=2)

        # Load model metadata
        if os.path.exists(MODEL_METADATA_FILE):
            try:
                with open(MODEL_METADATA_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.model_metadata = json.loads(content)
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
                        print("⚠️ Model metadata file is empty, initializing empty metadata")
                        self.model_metadata = {}
                        with open(MODEL_METADATA_FILE, 'w') as f:
                            json.dump(self.model_metadata, f, indent=2)
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in model metadata: {str(e)}, initializing empty metadata")
                self.model_metadata = {}
                with open(MODEL_METADATA_FILE, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)
            except Exception as e:
                print(f"⚠️ Error loading models: {str(e)}, initializing empty metadata")
                self.model_metadata = {}
                self.models = {}
                with open(MODEL_METADATA_FILE, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)

    def load_sentiment_cache(self):
        """Load sentiment cache from disk."""
        os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
        try:
            if os.path.exists(SENTIMENT_CACHE_FILE):
                with open(SENTIMENT_CACHE_FILE, 'rb') as f:
                    self.sentiment_cache = pickle.load(f)
                print(f"✅ Loaded sentiment cache with {len(self.sentiment_cache)} entries")
            else:
                print("⚠️ Sentiment cache file not found, initializing empty cache")
                self.sentiment_cache = {}
                with open(SENTIMENT_CACHE_FILE, 'wb') as f:
                    pickle.dump(self.sentiment_cache, f)
        except (PermissionError, pickle.PickleError) as e:
            print(f"⚠️ Error loading sentiment cache: {str(e)}, initializing empty cache")
            self.sentiment_cache = {}
            with open(SENTIMENT_CACHE_FILE, 'wb') as f:
                pickle.dump(self.sentiment_cache, f)

    def save_sentiment_cache(self):
        """Save sentiment cache to disk and commit to Git if in GitHub Actions."""
        os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
        try:
            with open(SENTIMENT_CACHE_FILE, 'wb') as f:
                pickle.dump(self.sentiment_cache, f)
            print("✅ Saved sentiment cache")

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
                            token = os.getenv('GH_PAT', '')
                            if not token:
                                raise ValueError("GITHUB_TOKEN not set for Git push")
                            auth_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
                            subprocess.run(['git', 'push', auth_url, 'HEAD'], check=True)
                            print("✅ Pushed sentiment cache to GitHub repository")
                        else:
                            raise ValueError("GITHUB_REPOSITORY not set")
                    else:
                        print("ℹ️ No changes to commit for sentiment cache")
                except (subprocess.CalledProcessError, ValueError) as e:
                    print(f"⚠️ Git operation failed for sentiment cache: {str(e)}")
                    self.send_telegram_message(f"Git operation failed for crypto sentiment cache: {str(e)}")
        except (PermissionError, pickle.PickleError) as e:
            print(f"⚠️ Error saving sentiment cache: {str(e)}, skipping cache save")
            self.send_telegram_message(f"Error saving crypto sentiment cache: {str(e)}")

    # ------------- Record Trades and Retrain Models ------------- #
    def record_trade(self, coin, entry_price, expected_return, entry_time, sell_time, features_1m, features_1h, buy_score, position_size_pct, exchange):
        """Store trade details for later outcome evaluation."""
        trade_id = hashlib.md5(
            f"{coin['symbol']}{expected_return}{entry_time.isoformat()}".encode()
        ).hexdigest()

        # Convert NumPy types in features dictionary to standard Python types
        processed_features_1m = {key: float(value) if isinstance(value, (np.float64, np.float32)) else int(value) if isinstance(value, (np.int64, np.int32)) else value for key, value in features_1m.items()}
        processed_features_1h = {key: float(value) if isinstance(value, (np.float64, np.float32)) else int(value) if isinstance(value, (np.int64, np.int32)) else value for key, value in features_1h.items()}

        # Ensure sell_time > entry_time + min hold
        min_hold = timedelta(minutes=5)
        if sell_time <= entry_time + min_hold:
            print(f"⚠️ Adjusting sell_time for {coin['symbol']} to minimum hold period.")
            sell_time = entry_time + min_hold

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
        """Evaluate outcome of pending trades."""
        print("🔍 Evaluating pending trades...")
        now = datetime.now(EST)
        break_even_count = 0
        total_recent = 0
        for trade in self.trade_history:
            if trade['status'] == 'completed':
                self.paper_trade(trade)
                continue  # Skip re-evaluation

            if trade['status'] == 'pending':
                sell_time = datetime.fromisoformat(trade['trade_info']['sell_time']).astimezone(EST)
                if now < sell_time:
                    continue  # Too early to evaluate

                # Fetch prices at sell_time with a 10-minute window
                symbol = trade['trade_info']['selected_symbol']
                exchange = trade['exchange']
                print(f"📊 Evaluating trade for {symbol} at sell time {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
                price_data = self.fetch_trade_data(symbol, '1m', exchange, start_time=sell_time - timedelta(minutes=5), end_time=sell_time + timedelta(minutes=5))

                if price_data is None or price_data.empty:
                    print(f"⚠️ No price data available for {symbol} on {exchange} around {sell_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}, skipping evaluation")
                    continue

                price_data.index = price_data.index.tz_convert(EST)

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
                outcome = 'profitable' if actual_return >= MIN_PROFITABLE_RETURN else 'break_even' if actual_return >= 0 else 'loss'
                trade['trade_info']['outcome'] = outcome
                print(f"📈 Trade outcome: {outcome}, Actual return: {actual_return * 100:.2f}% (Expected: {expected_return * 100:.2f}%)")
                self.send_telegram_message(
                    f"Crypto trade evaluated: {trade['symbol']} {outcome}, Actual return: {actual_return * 100:.2f}% (Expected: {expected_return * 100:.2f}%)"
                )

                # Track break-even for alerting
                entry_time = datetime.fromisoformat(trade['trade_info']['entry_time']).astimezone(EST)
                if (now - entry_time).days < 7:  # Recent trades
                    total_recent += 1
                    if outcome == 'break_even':
                        break_even_count += 1

        self.save_state()

        # Trigger retrain if enough new trades
        self.retrain_models()

    def paper_trade(self, trade):
        """Simulate a trade without real money and record in portfolio."""
        print(f"💰 Paper trading trade id: {trade['trade_id']}")
        try:
            # Use the most recent portfolio dictionary
            current_portfolio = self.portfolio[-1]
            reset_time = datetime.fromisoformat(current_portfolio['reset_timestamp']).astimezone(EST)
            trade_entry_time = datetime.fromisoformat(trade['trade_info']['entry_time']).astimezone(EST)

            # Skip trades before the portfolio reset time
            if trade_entry_time < reset_time:
                return
            
            # Check if trade_id already exists in current portfolio's trades
            if any(t['trade_id'] == trade['trade_id'] for t in current_portfolio['trades']):
                return

            if current_portfolio['portfolio_size'] <= 0:
                print("⚠️ Portfolio size is zero or negative, creating new portfolio")
                self.portfolio.append({'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()})
                current_portfolio = self.portfolio[-1]
                self.send_telegram_message(f"Crypto portfolio reset: New portfolio created with ${PORTFOLIO_SIZE:.2f}")
            
            position_size_pct = float(trade['trade_info']['position_size_pct'])
            trade_size = current_portfolio['portfolio_size'] * position_size_pct
            actual_return = float(trade['trade_info']['actual_return'])
            profit_loss = trade_size * actual_return
            new_trade_value = trade_size + profit_loss
            new_portfolio_size = current_portfolio['portfolio_size'] + profit_loss

            # Append trade to the current portfolio's trade history
            trade_record = {
                'trade_id': trade['trade_id'],
                'entry_time': trade['trade_info']['entry_time'],
                'p&l': profit_loss
            }
            current_portfolio['trades'].append(trade_record)
            current_portfolio['portfolio_size'] = new_portfolio_size

            print(f"✅ Paper traded {trade['symbol']}: P&L ${profit_loss:.2f}, New portfolio size ${new_portfolio_size:.2f}")
            self.send_telegram_message(
                f"Crypto paper trade completed: {trade['symbol']} ({trade['trade_info']['outcome']}), "
                f"P&L: ${profit_loss:.2f}, New portfolio: ${new_portfolio_size:.2f}"
            )
            self.save_state()

        except Exception as e:
            print(f"⚠️ Error in paper trading {trade['symbol']}: {str(e)}")
            self.send_telegram_message(f"Error in crypto paper trading {trade['symbol']}: {str(e)}")  

    def retrain_models(self):
        """Retrain models with recent trade data, grouped by timeframe only, retaining them on disk."""
        print("🔄 Retraining models with recent trade data")
        now = datetime.now(EST)
        recent_trades = [
            t for t in self.trade_history
            if t['status'] == 'completed' and
            (now - datetime.fromisoformat(t['trade_info']['entry_time']).astimezone(EST)).days < 28
        ]
        
        if len(recent_trades) < MIN_TRADES_FOR_RETRAIN:
            print(f"⚠️ Insufficient trades ({len(recent_trades)} < {MIN_TRADES_FOR_RETRAIN}) for retraining")
            return

        # Group trades by timeframe only
        for timeframe in ['1m', '1h']:
            timeframe_trades = [t for t in recent_trades if timeframe in t['features']]
            if not timeframe_trades:
                print(f"⚠️ No trades for {timeframe}, skipping")
                continue
            
            print(f"📊 Retraining models for timeframe: {timeframe}")

            # Prepare data
            X = np.array([list(t['features'][timeframe].values()) for t in timeframe_trades])
            le = LabelEncoder()
            y = le.fit_transform([t['trade_info']['outcome'] for t in timeframe_trades])  # Multi-class: profitable/break_even/loss
            y_reg = np.array([t['trade_info']['actual_return'] for t in timeframe_trades])

            # Split for validation
            split = int(0.8 * len(X))
            if split < 2 or len(X) - split < 2:
                print(f"⚠️ Insufficient data after split for {timeframe}")
                continue
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            y_reg_train, y_reg_val = y_reg[:split], y[split:]

            # Train classifier with improved imbalance handling
            clf = None
            if len(np.unique(y_train)) >= 2 and len(np.unique(y_val)) >= 2:
                over = SMOTE(random_state=42)
                under = RandomUnderSampler(random_state=42)
                pipeline = Pipeline([('over', over), ('under', under)])
                X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
                base_clf = self.models.get((timeframe,), {}).get('clf')
                clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                    early_stopping_rounds=10, random_state=42
                )
                clf.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False, xgb_model=base_clf)
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted')  # Weighted for multi-class
                cv_f1 = cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring='f1_weighted').mean()
                print(f"✅ Classifier for {timeframe} - F1 Score: {f1:.4f}, CV F1: {cv_f1:.4f}")
                base_score = self.models.get((timeframe,), {}).get('score', 0.0)
                if cv_f1 > base_score + 0.05:
                    self.models[(timeframe,)] = self.models.get((timeframe,), {})
                    self.models[(timeframe,)]['clf'] = clf
                    self.models[(timeframe,)]['score'] = cv_f1
                else:
                    print(f"ℹ️ No improvement in classifier for {timeframe}, keeping base model")
            else:
                print(f"⚠️ Skipping classifier for {timeframe} - insufficient classes")

            # Train regressor similarly
            reg = None
            if not np.isnan(y_reg_train).all():
                base_reg = self.models.get((timeframe,), {}).get('reg')
                reg = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    early_stopping_rounds=20
                )
                reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)], verbose=False, xgb_model=base_reg)
                y_reg_pred = reg.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))  # Fixed: y_reg_test -> y_reg_val
                cv_rmse = -cross_val_score(reg, X_train, y_reg_train, cv=5, scoring='neg_root_mean_squared_error').mean()  # Negative for consistency
                print(f"✅ Regressor for {timeframe} - RMSE: {rmse:.5f}, CV RMSE: {cv_rmse:.5f}")
                base_score = self.models.get((timeframe,), {}).get('score', 0.0)
                if cv_rmse < base_score * 0.95:  # Better if lower RMSE
                    self.models[(timeframe,)] = self.models.get((timeframe,), {})
                    self.models[(timeframe,)]['reg'] = reg
                    self.models[(timeframe,)]['score'] = cv_rmse
                else:
                    print(f"ℹ️ No improvement in regressor for {timeframe}, keeping base model")
            else:
                print(f"⚠️ Skipping regressor for {timeframe} - invalid targets")

    def save_state(self, coin=None, timeframe=None, threshold=None, clf=None, reg=None, features=None, score=None):
        """Persist models, trade history, and metadata to disk."""
        try:
            # Ensure model directory exists
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Save specific model if provided
            if timeframe and score is not None:
                model_key = (timeframe, score)
                model_id = f"{timeframe}_{score:.4f}"
                clf_path = os.path.join(MODEL_DIR, f"{model_id}_clf.joblib") if clf else None
                reg_path = os.path.join(MODEL_DIR, f"{model_id}_reg.joblib") if reg else None
                
                if clf:
                    dump(clf, clf_path)
                if reg:
                    dump(reg, reg_path)
                
                self.model_metadata[model_id] = {
                    'timeframe': timeframe,
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
                print(f"✅ Saved model for {model_id})")
            
            # Save all models if no specific model provided
            else:
                for model_key, data in self.models.items():
                    timeframe, score = model_key
                    model_id = f"{timeframe}_{score:.4f}"
                    clf_path = os.path.join(MODEL_DIR, f"{model_id}_clf.joblib")
                    reg_path = os.path.join(MODEL_DIR, f"{model_id}_reg.joblib")
                    if data['clf']:
                        dump(data['clf'], clf_path)
                    if data['reg']:
                        dump(data['reg'], reg_path)
                    self.model_metadata[model_id] = {
                        'timeframe': timeframe,
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
                self.send_telegram_message(f"Permission denied when saving crypto trade history: {str(e)}")

            # Save paper trading
            try:
                with open(PORTFOLIO_FILE, 'w') as f:
                    json.dump(self.portfolio, f, indent=2)
                print(f"✅ Saved paper trading history")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving paper trading history: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving crypto paper trading history: {str(e)}")

            # Save model metadata
            try:
                with open(MODEL_METADATA_FILE, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)
                print("✅ Saved model metadata")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving model metadata: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving crypto model metadata: {str(e)}")

            # GitHub Actions integration
            if GITHUB_ACTIONS:
                print("Persisting files for GitHub Actions...")
                try:
                    subprocess.run(['git', 'config', '--global', 'user.email', 'actions@github.com'], check=True)
                    subprocess.run(['git', 'config', '--global', 'user.name', 'GitHub Actions'], check=True)
                    
                    # Prepare files to add
                    files_to_add = [MODEL_METADATA_FILE, TRADE_HISTORY_FILE, PORTFOLIO_FILE, SENTIMENT_CACHE_FILE]
                    
                    # Check for .joblib files and add them if they exist
                    joblib_files = glob.glob(os.path.join(MODEL_DIR, '*.joblib'))
                    if joblib_files:
                        files_to_add.extend(joblib_files)
                        print(f"✅ Found {len(joblib_files)} .joblib files to add")
                    else:
                        print("ℹ️ No .joblib files found in crypto/spot/models/")

                    # Add files to Git
                    subprocess.run(['git', 'add'] + files_to_add, check=True)
                    
                    # Check if there are changes to commit
                    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
                    if result.stdout.strip():
                        commit_message = f"Update models and trade history {datetime.now(EST).isoformat()}"
                        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
                        repo_url = os.getenv('GITHUB_REPOSITORY')
                        if repo_url:
                            token = os.getenv('GH_PAT', '')
                            if not token:
                                print("⚠️ GITHUB_TOKEN not set, skipping Git push")
                                return
                            auth_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
                            subprocess.run(['git', 'push', auth_url, 'HEAD'], check=True)
                            print("✅ Pushed changes to GitHub repository")
                        else:
                            raise ValueError("GITHUB_REPOSITORY not set")
                    else:
                        print("ℹ️ No changes to commit")
                except (subprocess.CalledProcessError, ValueError) as e:
                    print(f"⚠️ Git operation failed: {str(e)}")
                    self.send_telegram_message(f"Git operation failed (crypto): {str(e)}")

        except Exception as e:
            print(f"⚠️ Error saving state: {str(e)}")
            self.send_telegram_message(f"Error saving crypto state: {str(e)}")

    # ------------- Candidate Scanning and Mapping ------------- #
    def scan_candidates(self):
        """Scan top 20 coins and filter by liquidity/volatility."""
        print('🔍 Scanning for candidates...')
        top_coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=1, price_change_percentage='1h')
        candidates = []
        for coin in top_coins:
            # Filter by liquidity (24h volume) and volatility
            if coin['total_volume'] < MIN_LIQUIDITY or abs(coin.get('price_change_percentage_24h', 0) / 100) > MAX_VOLATILITY:
                print(f"⚠️ Skipping {coin['symbol']} (Volume: ${coin['total_volume']:,.2f}, Volatility: {coin.get('price_change_percentage_24h', 0):.2f}%)")
                continue
            sentiment = self.fetch_news_sentiment(coin['id'])  # Assume get_sentiment is defined
            coin['sentiment'] = sentiment
            # Select exchange with most recent data
            exchanges = ['binanceus', 'coinbase', 'kraken', 'okx', 'cryptocom']
            selected_exchange = None
            selected_symbol = None
            latest_timestamp = None
            for exchange_name in exchanges:
                try:
                    exchange = getattr(self, exchange_name)
                    pairs = getattr(self, f"{exchange_name}_pairs", [])
                    symbol = next((s for s in pairs if s.endswith('/USD') and coin['symbol'].upper() in s), None)
                    if not symbol:
                        continue
                    # Fetch 1m data to check recency
                    df = self.fetch_data(symbol, '1m', exchange_name)
                    if df is not None and not df.empty:
                        last_time = df.index[-1].tz_convert(EST)
                        if latest_timestamp is None or last_time > latest_timestamp:
                            latest_timestamp = last_time
                            selected_exchange = exchange_name
                            selected_symbol = symbol
                except Exception as e:
                    print(f"⚠️ Error checking {exchange_name} for {coin['symbol']}: {str(e)}")
                    continue
            if selected_exchange and selected_symbol:
                coin['exchange'] = selected_exchange
                coin['selected_symbol'] = selected_symbol
                candidates.append(coin)
                print(f"✅ Added {coin['symbol']} ({selected_symbol} on {selected_exchange}, Sentiment: {sentiment:.2f})")
            else:
                print(f"⚠️ No valid exchange data for {coin['symbol']}")
        # Sort by combined score: market cap * (1 - volatility) + sentiment
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c['market_cap'] * (1 - abs(c.get('price_change_percentage_24h', 0)/100)) + c['sentiment'],
            reverse=True
        )[:10]
        print(f"🏆 Top {len(sorted_candidates)} candidates selected")
        return sorted_candidates
        
    def map_coingecko_to_exchange(self, coins):
        """Map CoinGecko coins to valid trading pairs across multiple exchanges."""
        cg_sorted_coins = []
        coinbase_pairs = self.coinbase_pairs
        binanceus_pairs = self.binanceus_pairs
        kraken_pairs = self.kraken_pairs
        okx_pairs = self.okx_pairs
        cryptocom_pairs = self.cryptocom_pairs
        
        try:
            for coin in coins:
                symbol = coin['symbol'].upper()
                coin_id = coin['id'].lower().replace('-', '')
                coin_copy = coin.copy()
                exchange_pairs = {
                    'coinbase': None,
                    'binanceus': None,
                    'kraken': None,
                    'okx': None,
                    'cryptocom': None
                }

                # Map pairs for each exchange
                for exchange, pairs in [
                    ('coinbase', coinbase_pairs),
                    ('binanceus', binanceus_pairs),
                    ('kraken', kraken_pairs),
                    ('okx', okx_pairs),
                    ('cryptocom', cryptocom_pairs)
                ]:
                    for pair, base_currency in pairs.items():
                        base_currency_normalized = base_currency.lower().replace('-', '')
                        if symbol.lower() == base_currency_normalized or coin_id == base_currency_normalized:
                            exchange_pairs[exchange] = pair
                            coin_copy[f'{exchange}_symbol'] = pair
                            print(f"✅ Found {exchange.capitalize()} pair for {symbol} ({coin['id']}): {pair}") # Debugging line
                            break

                # Fetch data to check freshness
                exchange_data = {
                    'coinbase': None,
                    'binanceus': None,
                    'kraken': None,
                    'okx': None,
                    'cryptocom': None
                }
                time_diffs = {
                    'coinbase': float('inf'),
                    'binanceus': float('inf'),
                    'kraken': float('inf'),
                    'okx': float('inf'),
                    'cryptocom': float('inf')
                }

                selected_exchange = None
                selected_symbol = None

                for exchange in exchange_pairs:
                    if exchange_pairs[exchange]:
                        data_1m = self.fetch_data(exchange_pairs[exchange], '1m', exchange)
                        if data_1m is not None and not data_1m.empty:
                            exchange_data[exchange] = data_1m
                            latest_time = data_1m.index[-1]
                            time_diff = (datetime.now(EST) - latest_time).total_seconds() / 60
                            time_diffs[exchange] = time_diff
                            
                            # If data is less than 1 minute old, select this exchange and stop checking others
                            if time_diff < 1.0:
                                selected_exchange = exchange
                                selected_symbol = exchange_pairs[exchange]
                                print(f"✅ Selected {exchange.capitalize()} for {symbol} ({coin['id']}) as data is less than 1 minute old ({time_diff:.2f} min)") # Debugging line
                                break

                # If an exchange was selected due to fresh data (< 1 min), use it
                if selected_exchange:
                    coin_copy['exchange'] = selected_exchange
                    coin_copy['selected_symbol'] = selected_symbol
                    cg_sorted_coins.append(coin_copy)
                else:
                    # Fallback to selecting the freshest data if no exchange has data < 1 min
                    valid_exchanges = [ex for ex in exchange_data if exchange_data[ex] is not None]
                    if valid_exchanges:
                        freshest_exchange = min(valid_exchanges, key=lambda ex: time_diffs[ex])
                        if time_diffs[freshest_exchange] <= FRESHNESS_THRESHOLD_1M:
                            coin_copy['exchange'] = freshest_exchange
                            coin_copy['selected_symbol'] = exchange_pairs[freshest_exchange]
                            print(f"✅ Selected {freshest_exchange.capitalize()} for {symbol} ({coin['id']}) as fresher data ({time_diffs[freshest_exchange]:.2f} min)") # Debugging line
                            cg_sorted_coins.append(coin_copy)
                        else:
                            print(f"⚠️ No fresh data for {symbol} ({coin['id']}) on any exchange, skipping")
                    else:
                        print(f"⚠️ No valid data for {symbol} ({coin['id']}) on any exchange, skipping")

            return cg_sorted_coins
        
        except Exception as e:
            print(f"⚠️ Error mapping CoinGecko to exchanges: {str(e)}")
            return []
        
    def get_exchange_pairs(self):
        """Fetch available trading pairs for each exchange."""
        exchanges = {
            'binanceus': self.binanceus,
            'coinbase': self.coinbase,
            'kraken': self.kraken,
            'okx': self.okx,
            'cryptocom': self.cryptocom
        }
        for exchange_name, exchange in exchanges.items():
            try:
                markets = exchange.load_markets()
                pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USD')]
                setattr(self, f"{exchange_name}_pairs", pairs)
                print(f"✅ Loaded {len(pairs)} USD pairs for {exchange_name}")
            except Exception as e:
                print(f"⚠️ Error loading pairs for {exchange_name}: {str(e)}")
                setattr(self, f"{exchange_name}_pairs", [])

    def fetch_news_sentiment(self, coin_symbol, coin_id=None, max_retries=3):
        """Fetch up to 100 recent news articles for the given coin and compute sentiment score."""
        global RATE_LIMIT_HIT
        cache_key = f"{coin_symbol}:{coin_id or ''}"
        current_time = datetime.now(pytz.UTC).timestamp()

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
                            print("⚠️ Global rate limit set, skipping further sentiment fetches")
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
        
    # ------------- Data Fetching and Processing ------------- # 
    def fetch_data(self, symbol, interval, exchange):
        """Fetch OHLCV data from specified exchange for analysis."""
        timeframe = {'1m': '1m', '1h': '1h', '1d': '1d'}.get(interval, '1m')
        min_candles = None if interval == '1m' or interval == '1h' else 20
        
        if exchange == 'coinbase':
            ex = self.coinbase
        elif exchange == 'binanceus':
            ex = self.binanceus
        elif exchange == 'kraken':
            ex = self.kraken
        elif exchange == 'okx':
            ex = self.okx
        elif exchange == 'cryptocom':
            ex = self.cryptocom
        else:
            print(f"⚠️ Unknown exchange: {exchange}")
            return None
        
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=min_candles)
            if not ohlcv:
                print(f"⚠️ No {exchange} data for {symbol}")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(EST)
            df.set_index('timestamp', inplace=True)
            latest_time = df.index[-1]
            current_time = datetime.now(EST)
            time_diff = (current_time - latest_time).total_seconds() / 60
            print(f"Raw {exchange} timestamp for {symbol}: {ohlcv[-1][0]} (converted: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
            print(f"{exchange} data for {symbol}: {time_diff:.2f} minutes old")
            if time_diff > FRESHNESS_THRESHOLD_1M and interval == '1m':
                print(f"⚠️ {exchange} data for {symbol} is stale ({time_diff:.2f} minutes old)")
                return None
            elif time_diff > FRESHNESS_THRESHOLD_1H and interval == '1h':
                print(f"⚠️ {exchange} data for {symbol} is stale ({time_diff:.2f} minutes old)")
                return None
            elif time_diff > FRESHNESS_THRESHOLD_1D and interval == '1d':
                print(f"⚠️ {exchange} data for {symbol} is stale ({time_diff:.2f} minutes old)")
                return None
            print(f"✅ Fetched {len(ohlcv)} {exchange} candles for {symbol} (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching {exchange} data for {symbol}: {str(e)}")
            return None

    def fetch_trade_data(self, symbol, interval, exchange, start_time=None, end_time=None):
        """Fetch OHLCV data within a specific time range for trade evaluation."""
        start_time = start_time.astimezone(EST)
        end_time = end_time.astimezone(EST)
        timeframe = {'1m': '1m', '1h': '1h', '1d': '1d'}.get(interval, '1m')
        exchange_obj = {
            'coinbase': self.coinbase,
            'binanceus': self.binanceus,
            'kraken': self.kraken,
            'okx': self.okx,
            'cryptocom': self.cryptocom
        }.get(exchange)
        
        if not exchange_obj:
            print(f"⚠️ Unknown exchange: {exchange}")
            return None

        since = int(start_time.timestamp() * 1000) if start_time else None
        limit = int((end_time - start_time).total_seconds() / 60) + 1 if start_time and end_time else 500
        
        try:
            ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                print(f"⚠️ No data for {symbol} on {exchange}")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(EST)
            df.set_index('timestamp', inplace=True)
            if start_time and end_time:
                df = df[(df.index >= start_time) & (df.index <= end_time)]
            if df.empty:
                print(f"⚠️ Empty dataframe after filtering for {symbol} on {exchange}")
                return None
            print(f"✅ Fetched {len(df)} candles for {symbol} on {exchange}")
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching trade data for {symbol} on {exchange}: {str(e)}")
            return None

    def calculate_features(self, df, df_type, lags=[1,3,5], rolling_windows=[3,6]):
        """Calculate technical features with additions."""
        if df is None or df.empty:
            print(f"⚠️ Empty DataFrame for {df_type}, skipping features")
            return None
        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['bollinger'] = ta.bbands(df['close'], length=20, std=2)['BBM_20_2.0']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            df['obv'] = ta.obv(df['close'], df['volume'])
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df = pd.concat([df, stoch], axis=1)
            df['mom'] = ta.mom(df['close'], length=10)

            if 'sentiment' not in df.columns:
                df['sentiment'] = 0.0

            # Add time series features
            base_features = ['open', 'high', 'low', 'volume', 'returns', 'volatility',
                             'rsi', 'macd', 'bollinger', 'atr', 'sma_20', 'ema_20',
                             'adx', 'obv', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'sentiment', 'mom']

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

            print(f"✅ Technical indicators added for {df_type} data")
            return df

        except Exception as e:
            print(f"⚠️ Error in calculate_features for {df_type}: {str(e)}")
            return None
    
    def label_1m_model(self, coin, df, window=360, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-minute model with dynamic threshold from history."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {coin['symbol']} (1m), cannot label")
            return None
        try:
            print(f"📊 Attempting to label 1m model for {coin['symbol']}")
            df = df.copy()
            df['future_max'] = df['close'].shift(-1).rolling(window=window, min_periods=1).max()
            df['target_return'] = df['future_max'] / df['close'] - 1
            df['target_return'] = df['target_return'].clip(lower=0, upper=max_return)

            # Log return statistics
            print(f"📊 Target return stats for {coin['symbol']}: min={df['target_return'].min():.4f}, max={df['target_return'].max():.4f}, mean={df['target_return'].mean():.4f}")

            # Dynamic threshold with history
            coin_trades = [t for t in self.trade_history if t['id'] == coin['id'] and 'actual_return' in t['trade_info']]
            if coin_trades:
                hist_returns = [t['trade_info']['actual_return'] for t in coin_trades]
                dynamic_threshold = np.mean(hist_returns) + np.std(hist_returns) if np.mean(hist_returns) > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)
                print(f"📈 History-adjusted threshold for {coin['symbol']}: {dynamic_threshold:.4f}")
            else:
                valid_returns = df['target_return'].dropna()
                dynamic_threshold = np.percentile(valid_returns, 10) if len(valid_returns) > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)

            # Create multi-class labels
            df['label'] = 0  # Default: break-even
            df.loc[df['target_return'] >= dynamic_threshold, 'label'] = 1  # Profitable
            df.loc[df['target_return'] < 0, 'label'] = 0  # Loss (map to 0 for now, will adjust later)
            labeled_df = df.dropna()

            # Check class distribution
            class_counts = labeled_df['label'].value_counts()
            print(f"📊 Label distribution for {coin['symbol']} (1m): {class_counts.to_dict()}")
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1m labels for {coin['symbol']} ({class_counts.to_dict()}), adjusting threshold")
                if len(valid_returns) > 0:
                    dynamic_threshold = np.percentile(valid_returns, 5)  # Lower percentile
                    df['label'] = 0
                    df.loc[df['target_return'] >= dynamic_threshold, 'label'] = 1
                    df.loc[df['target_return'] < 0, 'label'] = 0
                    labeled_df = df.dropna()
                    print(f"📊 Retried with lower threshold {dynamic_threshold:.4f}, new label distribution: {labeled_df['label'].value_counts().to_dict()}")
                    if labeled_df['label'].nunique() < 2:
                        print(f"⚠️ Still only one class, skipping classifier for {coin['symbol']} (1m)")
                        return labeled_df  # Return for regressor use

            return labeled_df

        except Exception as e:
            print(f"⚠️ Error preparing 1m data: {str(e)}")
            return None

    def label_1h_model(self, coin, df, forward_hours=6, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-hour model with dynamic threshold and target_return from history."""
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

            # Log return statistics
            print(f"📊 Forward return stats for {coin['symbol']}: min={df['fwd_return'].min():.4f}, max={df['fwd_return'].max():.4f}, mean={df['fwd_return'].mean():.4f}")

            # Dynamic threshold with history
            coin_trades = [t for t in self.trade_history if t['id'] == coin['id'] and 'actual_return' in t['trade_info']]
            if coin_trades:
                hist_returns = [t['trade_info']['actual_return'] for t in coin_trades]
                dynamic_threshold = np.mean(hist_returns) + np.std(hist_returns) if np.mean(hist_returns) > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)
                print(f"📈 History-adjusted threshold for {coin['symbol']}: {dynamic_threshold:.4f}")
            else:
                valid_returns = df['fwd_return'].dropna()
                dynamic_threshold = np.percentile(valid_returns, 10) if len(valid_returns) > 0 else min_return
                dynamic_threshold = max(dynamic_threshold, min_return)

            # Create multi-class labels and map to [0, 1, 2]
            df['label'] = 1  # Default: break-even
            df.loc[df['fwd_return'] >= dynamic_threshold, 'label'] = 2  # Profitable
            df.loc[df['fwd_return'] < 0, 'label'] = 0  # Loss
            labeled_df = df.dropna()

            # Check class distribution
            class_counts = labeled_df['label'].value_counts()
            print(f"📊 Label distribution for {coin['symbol']} (1h): {class_counts.to_dict()}")
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1h labels for {coin['symbol']} ({class_counts.to_dict()}), adjusting threshold")
                if len(valid_returns) > 0:
                    dynamic_threshold = np.percentile(valid_returns, 5)  # Lower percentile
                    df['label'] = 1
                    df.loc[df['fwd_return'] >= dynamic_threshold, 'label'] = 2
                    df.loc[df['fwd_return'] < 0, 'label'] = 0
                    labeled_df = df.dropna()
                    print(f"📊 Retried with lower threshold {dynamic_threshold:.4f}, new label distribution: {labeled_df['label'].value_counts().to_dict()}")
                    if labeled_df['label'].nunique() < 2:
                        print(f"⚠️ Still only one class, skipping classifier for {coin['symbol']} (1h)")
                        return labeled_df  # Return for regressor use

            return labeled_df

        except Exception as e:
            print(f"⚠️ Error preparing 1h data: {str(e)}")
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
            if split < 2 or len(X) - split < 2:
                print(f"⚠️ Insufficient data for {coin['symbol']} ({df_type}) after split")
                return None, None, None
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_class_train, y_class_test = y_class.iloc[:split], y_class.iloc[split:]

            # Load base models if available
            model_key = (df_type,)
            base_clf = self.models.get(model_key, {}).get('clf')
            base_reg = self.models.get(model_key, {}).get('reg')
            base_score = self.models.get(model_key, {}).get('score', 0.0)

            # Train classifier
            best_class_result = None
            if y_class_train.nunique() >= 2 and y_class_test.nunique() >= 2:
                minority_count = min([sum(y_class_train == c) for c in np.unique(y_class_train)])
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
                # Add undersampling if imbalance > 3:1
                class_counts = np.bincount(y_class_train_res - y_class_train_res.min()) if y_class_train_res.min() < 0 else np.bincount(y_class_train_res)
                if len(class_counts) > 1 and max(class_counts) / min(class_counts) > 3:
                    under = RandomUnderSampler(random_state=42)
                    X_train_res, y_class_train_res = under.fit_resample(X_train_res, y_class_train_res)

                # Train classifier with early stopping
                clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                    random_state=42, early_stopping_rounds=10
                )
                clf.fit(
                    X_train_res, y_class_train_res,
                    eval_set=[(X_test, y_class_test)],
                    verbose=False,
                    xgb_model=base_clf if base_clf else None
                )
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_class_test, y_pred, average='weighted')

                # Cross-validation without early stopping
                cv_clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                    random_state=42
                )
                cv_f1 = cross_val_score(cv_clf, X_train_res, y_class_train_res, cv=5, scoring='f1_weighted').mean()
                print(f"✅ Classifier for {coin['symbol']} ({df_type}) - F1 Score: {f1:.4f}, CV: {cv_f1:.4f}")

                # Lightweight tuning if performance is poor
                if f1 < 0.5 or (base_clf and f1 < base_score * 0.9):
                    print(f"⚠️ Classifier F1 score {f1:.4f} is low or worse than base ({base_score:.4f}), attempting lightweight tuning")
                    best_f1 = f1
                    best_clf = clf
                    for lr in [0.05, 0.2]:
                        for depth in [2, 4]:
                            temp_clf = XGBClassifier(
                                n_estimators=100, max_depth=depth, learning_rate=lr,
                                subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                                random_state=42, early_stopping_rounds=10
                            )
                            temp_clf.fit(
                                X_train_res, y_class_train_res,
                                eval_set=[(X_test, y_class_test)],
                                verbose=False,
                                xgb_model=base_clf if base_clf else None
                            )
                            temp_pred = temp_clf.predict(X_test)
                            temp_f1 = f1_score(y_class_test, temp_pred, average='weighted')
                            if temp_f1 > best_f1:
                                best_f1 = temp_f1
                                best_clf = temp_clf
                            print(f"🔍 Tuning: lr={lr}, depth={depth}, F1={temp_f1:.4f}")
                    f1 = best_f1
                    clf = best_clf
                    print(f"✅ Best tuned classifier F1 Score: {f1:.4f}")

                best_class_result = {
                    'score': f1,
                    'model': clf,
                    'y_class_test': y_class_test,
                    'y_pred': y_pred
                }
                print('\nClassifier Report:')
                print(classification_report(best_class_result['y_class_test'], best_class_result['y_pred']))
            else:
                print(f"⚠️ Skipping classifier training for {df_type} — only one class in training/test set")

            # Train regressor
            best_reg_result = None
            if y_reg is not None and not y_reg.isnull().all():
                y_reg_train, y_reg_test = y_reg.iloc[:split], y_reg.iloc[split:]
                reg = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    early_stopping_rounds=20, verbosity=0
                )
                reg.fit(X_train, y_reg_train, eval_set=[(X_test, y_reg_test)], verbose=False, xgb_model=base_reg if base_reg else None)
                y_reg_pred = reg.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

                # Cross-validation without early stopping
                cv_reg = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                cv_rmse = -cross_val_score(cv_reg, X_train, y_reg_train, cv=5, scoring='neg_root_mean_squared_error').mean()
                print(f"✅ Regressor for {coin['symbol']} ({df_type}) - RMSE: {rmse:.5f}, CV RMSE: {cv_rmse:.5f}")

                # Lightweight tuning if performance is poor
                if rmse > 0.1 or (base_reg and base_score and rmse > base_score * 1.1):
                    print(f"⚠️ Regressor RMSE {rmse:.5f} is high or worse than base, attempting lightweight tuning")
                    best_rmse = rmse
                    best_reg = reg
                    for lr in [0.03, 0.1]:
                        for depth in [3, 5]:
                            temp_reg = XGBRegressor(
                                n_estimators=200, max_depth=depth, learning_rate=lr,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                early_stopping_rounds=20, verbosity=0
                            )
                            temp_reg.fit(X_train, y_reg_train, eval_set=[(X_test, y_reg_test)], verbose=False, xgb_model=base_reg if base_reg else None)
                            temp_pred = temp_reg.predict(X_test)
                            temp_rmse = np.sqrt(mean_squared_error(y_reg_test, temp_pred))
                            if temp_rmse < best_rmse:
                                best_rmse = temp_rmse
                                best_reg = temp_reg
                            print(f"🔍 Tuning: lr={lr}, depth={depth}, RMSE={temp_rmse:.5f}")
                    rmse = best_rmse
                    reg = best_reg
                    print(f"✅ Best tuned regressor RMSE: {rmse:.5f}")

                best_reg_result = {
                    'rmse': rmse,
                    'model': reg
                }
            else:
                print(f"❕ Skipping regressor for {df_type} — no valid 'target_return' values")

            return best_class_result, best_reg_result, features

        except Exception as e:
            print(f"⚠️ Error training hybrid model: {str(e)}")
            return None, None, None

    def predict_future_movement(self, expected_return, sentiment):
        """Predict the future movement of the best coin by simulating realistic data points with Monte Carlo."""
        print(f"\n🔮 Predicting future movement for {best_coin['symbol']}")

        # Ensure latest timestamp is in EST
        latest_timestamp = best_1m_df.index[-1]
        if not latest_timestamp.tzinfo:
            latest_timestamp = latest_timestamp.tz_localize('UTC').tz_convert(EST)
        else:
            latest_timestamp = latest_timestamp.tz_convert(EST)
        
        # Parameters
        min_minutes = 5 # No transactions less than 5 min
        max_minutes = 360  # 6 hours
        threshold = expected_return  # Use the regressor's expected return as the threshold
        
        # Get recent trends for simulation
        recent_data = best_1m_df.tail(20)
        avg_return = recent_data['returns'].mean()
        avg_volume = recent_data['volume'].mean()
        vol_volatility = recent_data['volume'].std() / avg_volume if avg_volume > 0 else 0.1

        # Adjust based on history
        recent_trades = [t for t in self.trade_history if t['status'] == 'completed' and t['actual_return'] < t['trade_info']['expected_return']]
        if recent_trades:
            adjustment_factor = 0.8  # 20% reduction if over-optimistic
            avg_return *= adjustment_factor
            print(f"📉 Adjusted avg_return to {avg_return:.4f} based on history")

        # Fit GARCH(1,1) model for volatility simulation
        try:
            garch_model = arch.arch_model(recent_data['returns'].dropna() * 100, vol='GARCH', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            vol_forecast = garch_fit.forecast(horizon=1)
            init_vol = np.sqrt(vol_forecast.variance.values[-1, 0]) / 100
        except Exception as e:
            print(f"⚠️ GARCH fitting failed: {str(e)}, using historical volatility")
            init_vol = recent_data['volatility'].mean()

        # Monte Carlo simulations
        signal_times = []
        predicted_returns = []
        for path in range(NUM_MONTE_CARLO_PATHS):
            sim_data = best_1m_df[['open', 'high', 'low', 'close', 'volume']].copy()
            last_row = sim_data.iloc[-1]
            current_vol = init_vol

            for minute in range(1, max_minutes + 1):
                next_time = latest_timestamp + timedelta(minutes=minute)
                
                # Update volatility with GARCH
                try:
                    garch_forecast = garch_model.fit(disp='off', first_obs=sim_data['returns'].dropna()[-20:]).forecast(horizon=1)
                    current_vol = np.sqrt(garch_forecast.variance.values[-1, 0]) / 100
                except:
                    pass  # Keep current_vol if GARCH fails

                # Simulate price movement with t-distribution (fat-tailed)
                df_freedom = 5  # Degrees of freedom for t-distribution (lower = fatter tails)
                price_change = t.rvs(df_freedom, loc=avg_return + sentiment * 0.001, scale=current_vol)
                
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
                continue
            
            # Extract future data
            future_data = sim_data.loc[sim_data.index > latest_timestamp][best_analysis['feat_1m']]
            
            # Predict returns for each future minute
            found_signal = False
            signal_time = None
            predicted_return = None

            for i, (timestamp, features) in enumerate(future_data.iloc[min_minutes:].iterrows()):
                if i >= max_minutes:
                    break
                features_df = pd.DataFrame([features], columns=best_analysis['feat_1m'], index=[timestamp])
                predicted_return = best_analysis['reg_1m']['model'].predict(features_df)[0] if best_analysis['reg_1m'] is not None else 0.0
                slippage = current_vol * 0.5
                adjusted_return = predicted_return - FEE_RATE - slippage
                if adjusted_return >= threshold:
                    signal_time = timestamp
                    found_signal = True
                    break
            
            if found_signal:
                signal_times.append(signal_time)
                predicted_returns.append(adjusted_return)
            else:
                signal_times.append(latest_timestamp + timedelta(minutes=max_minutes))
                predicted_returns.append(0.0)
        
        if not signal_times:
            print(f"❌ No sell signal found for {best_coin['symbol']} within 6 hours")
            return None, None
        
        signal_time = pd.Series(signal_times).median()
        predicted_return = np.mean(predicted_returns)
        return_var = np.std(predicted_returns)
        if return_var > 0.05:
            print(f"⚠️ High variance {return_var:.4f} in predictions, skipping trade")
            return None, None

        return signal_time, predicted_return

# ------------- MAIN FUNCTION ------------- #
    def run_analysis(self):
        """Main analysis pipeline with sentiment-integrated buy score."""
        global best_score, best_coin, best_analysis, best_1m_df, best_1h_df
        print('🚀 Starting trading algorithm')
        current_time = datetime.now(EST)
        print(f"📅 Analysis started at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")

        # Scan for candidates
        self.get_exchange_pairs()
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

        # Predict probabilities and returns
        prob_1m = 0.5
        prob_1h = 0.5
        expected_return_1m = 0.0
        expected_return_1h = 0.0

        # 1m predictions
        if best_analysis['clf_1m'] is not None:
            prob_1m = best_analysis['clf_1m']['model'].predict_proba(latest_1m)[0][1]  # Probability of positive class
        elif best_analysis['reg_1m'] is not None:
            expected_return_1m = best_analysis['reg_1m']['model'].predict(latest_1m)[0]
            prob_1m = min(expected_return_1m / THRESHOLDS[0], 1.0) if expected_return_1m > 0 else 0.5

        # 1h predictions
        if best_analysis['clf_1h'] is not None:
            prob_1h = best_analysis['clf_1h']['model'].predict_proba(latest_1h)[0][2]  # Probability of profitable class (label 2)
        elif best_analysis['reg_1h'] is not None:
            expected_return_1h = best_analysis['reg_1h']['model'].predict(latest_1h)[0]
            prob_1h = min(expected_return_1h / THRESHOLDS[0], 1.0) if expected_return_1h > 0 else 0.5

        # Combine predictions with sentiment
        sentiment_score = best_coin['sentiment']
        expected_return = expected_return_1m * 0.6 + expected_return_1h * 0.4
        
        buy_score = W_1M * prob_1m + W_1H * prob_1h + W_SENTIMENT * (sentiment_score + 1) / 2
        buy = buy_score >= 0.45 or (expected_return >= THRESHOLDS[0])  # Relaxed thresholds

       # Dynamic position sizing with risk
        recent_vol = best_1m_df['volatility'].iloc[-1]
        position_size_pct = min(buy_score * (1 / recent_vol) if recent_vol > 0 else buy_score, 0.05)  # Max 5%

        if buy:
            close_price = best_1m_df['close'].iloc[-1]
            # Final buy report
            message = [
                f"✅ Action: BUY {best_coin['symbol']} ({best_coin['selected_symbol']} at ${close_price:.4f} on {best_coin['exchange']})",
                f"🤖 Buy score: {buy_score:.2f} (1m: {prob_1m:.2f}, 1h: {prob_1h:.2f}, sentiment: {sentiment_score:.2f})",
                f"📈 Expected return: {expected_return * 100:.2f}%",
                f"💰 Position size: {position_size_pct * 100:.1f}%"
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
                    f"📈 Predicted sell price: ${(1 + predicted_return) * close_price:.4f} (predicted return: {predicted_return * 100:.2f}%)"
                ])
                print(message)
                self.send_telegram_message("\n".join(message))
        
        else:
            print(f"❌ No buy signal for best coin: {best_coin['symbol']} ({best_coin['selected_symbol']} on {best_coin['exchange']})")

if __name__ == '__main__':
    trader = CryptoTrader()
    trader.run_analysis()
    trader.evaluate_pending_trades()
