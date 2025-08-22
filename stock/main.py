# ------------- API DOCUMENTATION ------------- #
# yfinance - https://pypi.org/project/yfinance/ (for stock data)

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
from sklearn.metrics import f1_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import hashlib
from textblob import TextBlob
from scipy.stats import t
import arch
from imblearn.over_sampling import SMOTE, ADASYN
import subprocess
import glob
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
THRESHOLDS = [0.02, 0.99]  # Min, Max
EST = pytz.timezone('America/New_York')
MODEL_DIR = 'stock/ind/models'
TRADE_HISTORY_FILE = 'stock/ind/trade_history.json'
MODEL_METADATA_FILE = 'stock/ind/model_metadata.json'
SENTIMENT_CACHE_FILE = 'stock/ind/sentiment_cache.pkl'
PORTFOLIO_FILE = 'stock/ind/portfolio.json'
PORTFOLIO_SIZE = 10000.00
RATE_LIMIT_HIT = False
FEEDBACK_INTERVAL_HOURS = 3  # How often to retrain models
SENTIMENT_CACHE_TTL = 14400  # Cache sentiment for 4 hours
W_1M = 0.75  # Weight for 1 min df in buy score
W_1H = 0.25  # Weight for 1 hour df in buy score
W_SENTIMENT = 0.1  # Weight for sentiment in buy score
MIN_CANDLES_1M = 300
MIN_CANDLES_1H = 100
FRESHNESS_THRESHOLD_1M = 10  # Minutes  
FRESHNESS_THRESHOLD_1H = 60  # Minutes
FRESHNESS_THRESHOLD_1D = 1440  # Minutes
RATE_LIMIT_DELAY = 1  # Seconds between retries
best_score = -np.inf
best_stock = None
best_analysis = None
best_1m_df = None
best_1h_df = None

# Load environment variables
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# ------------- File System Setup ---------------- #
os.makedirs(os.path.dirname(SENTIMENT_CACHE_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------- StockTrader Class ---------------- #
class StockTrader:
    def __init__(self):
        self.models = {}  # Format: {(stock, timeframe, threshold): {'clf': model, 'reg': model, 'features': list, 'score': float}}
        self.trade_history = []
        self.portfolio = []
        self.model_metadata = {}
        self.sentiment_cache = {}
        self.top_stocks = []
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
                            stock, timeframe, threshold = key.split('_')
                            threshold = float(threshold)
                            model_key = (stock, timeframe, threshold)
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
                    self.send_telegram_message(f"Git operation failed for stock sentiment cache: {str(e)}")
        except (PermissionError, pickle.PickleError) as e:
            print(f"⚠️ Error saving sentiment cache: {str(e)}, skipping cache save")
            self.send_telegram_message(f"Error saving stock sentiment cache: {str(e)}")

    # ------------- Record Trades and Retrain Models ------------- #
    def record_trade(self, stock, entry_price, expected_return, entry_time, sell_time, features_1m, features_1h, buy_score, position_size_pct, exchange):
        """Store trade details for later outcome evaluation."""
        trade_id = hashlib.md5(
            f"{stock['symbol']}{expected_return}{entry_time.isoformat()}".encode()
        ).hexdigest()

        # Convert NumPy types in features dictionary to standard Python types
        processed_features_1m = {key: float(value) if isinstance(value, (np.float64, np.float32)) else int(value) if isinstance(value, (np.int64, np.int32)) else value for key, value in features_1m.items()}
        processed_features_1h = {key: float(value) if isinstance(value, (np.float64, np.float32)) else int(value) if isinstance(value, (np.int64, np.int32)) else value for key, value in features_1h.items()}

        self.trade_history.append({
            'trade_id': trade_id,
            'id': stock['id'],
            'symbol': stock['symbol'],
            'exchange': exchange,
            'trade_info': {
                'entry_price': float(entry_price),
                'expected_return': float(expected_return),
                'entry_time': entry_time.isoformat(),
                'sell_time': sell_time.isoformat(),
                'buy_score': float(buy_score),
                'position_size_pct': float(position_size_pct),
                'selected_symbol': stock['selected_symbol']
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
        for trade in self.trade_history:
            if trade['status'] == 'completed':
                self.paper_trade(trade)

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

                # Ensure index is timezone-aware and unique
                price_data.index = pd.to_datetime(price_data.index, utc=True).tz_convert(EST)
                if price_data.index.duplicated().any():
                    print(f"⚠️ Duplicate timestamps found in price_data for {symbol}, keeping first occurrence")
                    price_data = price_data[~price_data.index.duplicated(keep='first')]

                # Check if close column exists and is numeric
                if 'close' not in price_data.columns or not pd.api.types.is_numeric_dtype(price_data['close']):
                    print(f"⚠️ Invalid 'close' column in price_data for {symbol}, skipping")
                    continue

                price_data.index = price_data.index.tz_convert(EST)

                if sell_time in price_data.index:
                    sell_price = price_data.loc[sell_time]['close']
                    # Ensure sell_price is a scalar
                    sell_price = sell_price.iloc[0] if isinstance(sell_price, pd.Series) else sell_price
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
                    price_before = price_before.iloc[0] if isinstance(price_before, pd.Series) else price_before
                    price_after = price_data.loc[closest_after]['close']
                    price_after = price_after.iloc[0] if isinstance(price_after, pd.Series) else price_after
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
                    f"Stock trade evaluated: {trade['symbol']} {outcome}, Actual return: {actual_return * 100:.2f}% (Expected: {expected_return * 100:.2f}%)"
                )

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

            # # Skip trades before the portfolio reset time
            # if trade_entry_time < reset_time:
            #     return
            
            # Check if trade_id already exists in current portfolio's trades
            if any(t['trade_id'] == trade['trade_id'] for t in current_portfolio['trades']):
                return

            # Use the most recent portfolio dictionary
            current_portfolio = self.portfolio[-1]
            if current_portfolio['portfolio_size'] <= 0:
                print("⚠️ Portfolio size is zero or negative, creating new portfolio")
                self.portfolio.append({'portfolio_size': PORTFOLIO_SIZE, 'trades': [], 'reset_timestamp': datetime.now(EST).isoformat()})
                current_portfolio = self.portfolio[-1]
                self.send_telegram_message(f"Stock portfolio reset: New portfolio created with ${PORTFOLIO_SIZE:.2f}")
            
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
                f"Stock paper trade completed: {trade['symbol']} ({trade['trade_info']['outcome']}), "
                f"P&L: ${profit_loss:.2f}, New portfolio: ${new_portfolio_size:.2f}"
            )
            self.save_state()

        except Exception as e:
            print(f"⚠️ Error in paper trading {trade['symbol']}: {str(e)}")
            self.send_telegram_message(f"Error in stock paper trading {trade['symbol']}: {str(e)}")  

    def retrain_models(self):
        """Retrain models with recent trade data, grouped by timeframe only, retaining them on disk."""
        print("🔄 Retraining models with recent trade data")
        now = datetime.now(EST)
        recent_trades = [
            t for t in self.trade_history
            if t['status'] == 'completed' and
            (now - datetime.fromisoformat(t['trade_info']['entry_time']).astimezone(EST)).days < 28
        ]
        
        if len(recent_trades) < 50:
            print(f"⚠️ Insufficient trades ({len(recent_trades)} < 50) for retraining")
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
            y = le.fit_transform([t['trade_info']['outcome'] for t in timeframe_trades])  # 'profitable' -> 1, 'loss' -> 0
            y_reg = np.array([t['trade_info']['actual_return'] for t in timeframe_trades])

            # Split for validation
            split = int(0.8 * len(X))
            if split < 2 or len(X) - split < 2:
                print(f"⚠️ Insufficient data after split for {timeframe}")
                continue
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            y_reg_train, y_reg_val = y_reg[:split], y[split:]

            # Train classifier
            clf = None
            if len(np.unique(y_train)) >= 2 and len(np.unique(y_val)) >= 2:
                class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if 1 in y_train else 1.0
                minority_count = len(y_train[y_train == 1])
                k_neighbors = min(5, max(1, minority_count - 1))
                try:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                except ValueError as e:
                    print(f"⚠️ SMOTE failed for {timeframe}: {str(e)}, trying ADASYN")
                    try:
                        adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
                    except ValueError as e:
                        print(f"⚠️ ADASYN also failed for {timeframe}: {str(e)}, proceeding without oversampling")
                        X_train_res, y_train_res = X_train, y_train
                clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                    early_stopping_rounds=10, random_state=42,
                    scale_pos_weight=class_ratio
                )
                clf.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred)
                print(f"✅ Classifier for {timeframe} - F1 Score: {f1:.4f}")
            else:
                print(f"⚠️ Skipping classifier for {timeframe} - only one class")

            # Train regressor
            reg = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                early_stopping_rounds=20, verbosity=0
            )
            reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)], verbose=False)
            y_reg_pred = reg.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
            print(f"✅ Regressor for {timeframe} - RMSE: {rmse:.5f}")

            # Save models
            features = list(timeframe_trades[0]['features'][timeframe].keys())
            self.save_state(timeframe=timeframe, clf=clf, reg=reg, features=features, score=f1 if clf else rmse)

    def save_state(self, stock=None, timeframe=None, threshold=None, clf=None, reg=None, features=None, score=None):
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
                self.send_telegram_message(f"Permission denied when saving stock trade history: {str(e)}")

            # Save paper trading
            try:
                with open(PORTFOLIO_FILE, 'w') as f:
                    json.dump(self.portfolio, f, indent=2)
                print(f"✅ Saved paper trading history")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving paper trading history: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving stock paper trading history: {str(e)}")

            # Save model metadata
            try:
                with open(MODEL_METADATA_FILE, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)
                print("✅ Saved model metadata")
            except PermissionError as e:
                print(f"⚠️ Permission denied when saving model metadata: {str(e)}, skipping save")
                self.send_telegram_message(f"Permission denied when saving stock model metadata: {str(e)}")

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
                        print("ℹ️ No .joblib files found in stock/ind/models/")

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
                    self.send_telegram_message(f"Git operation failed (stock): {str(e)}")

        except Exception as e:
            print(f"⚠️ Error saving state: {str(e)}")
            self.send_telegram_message(f"Error saving stock state: {str(e)}")

    # ------------- Candidate Scanning and Mapping ------------- #
    def scan_candidates(self):
        """Find potential trading candidates using yfinance and include sentiment scores."""
        print('🔍 Scanning for candidates...')
        try:
            # Scrape S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]  # First table is the S&P 500 constituents
            self.top_stocks = sp500_table['Symbol'].tolist()

            stocks = []
            for symbol in self.top_stocks:
                ticker = yf.Ticker(symbol)
                # Fetch 1-hour data
                hist_1h = ticker.history(period='1d', interval='1h')
                if hist_1h.empty or len(hist_1h) < 2:
                    print(f"⚠️ No 1h data for {symbol}, skipping")
                    continue
                price_change_1h = (hist_1h['Close'][-1] - hist_1h['Close'][-2]) / hist_1h['Close'][-2] * 100

                # Fetch 1-day data with extended period to ensure data availability
                hist_1d = ticker.history(period='5d', interval='1d')  # Extended to 5 days
                if hist_1d.empty or len(hist_1d) < 2:
                    print(f"⚠️ No 1d data for {symbol}, skipping")
                    continue
                # Use the latest two available candles
                price_change_24h = (hist_1d['Close'][-1] - hist_1d['Close'][-2]) / hist_1d['Close'][-2] * 100

                stock = {
                    'id': symbol.lower(),
                    'symbol': symbol,
                    'price_change_percentage_1h': price_change_1h,
                    'price_change_percentage_24h': price_change_24h,
                }
                stocks.append(stock)
                time.sleep(0.1)  # Rate limit for yfinance

            if not stocks:
                print("⚠️ No valid stocks retrieved, returning empty list")
                return []

            top_stocks = sorted(stocks, key=lambda x: x['price_change_percentage_1h'], reverse=True)[:20]
            thresh_1h, thresh_24h = self.calculate_dynamic_thresholds(top_stocks)

            sorted_stocks = []
            for s in stocks:
                if s['price_change_percentage_1h'] > thresh_1h and s['price_change_percentage_24h'] > thresh_24h:
                    sorted_stocks.append(s)
            sorted_stocks = sorted(
                sorted_stocks,
                key=lambda x: (x['price_change_percentage_1h'], x['price_change_percentage_24h']),
                reverse=True
            )
            sorted_stocks = sorted_stocks[:10]  # Limit to top 10 candidates
            if not sorted_stocks:
                print(f"⚠️ No stocks met threshold criteria (1h: {thresh_1h:.4f}, 24h: {thresh_24h:.4f})")
                # Fallback: Include top stocks even if they don't meet thresholds
                sorted_stocks = top_stocks[:10]
                print(f"ℹ️ Falling back to top {len(sorted_stocks)} stocks without threshold filtering")

            sorted_stocks = self.map_to_exchange(sorted_stocks)
            if not sorted_stocks:
                print("⚠️ No stocks successfully mapped to exchange")
                return []

            for s in sorted_stocks:
                s['sentiment'] = self.fetch_news_sentiment(s['symbol'], s['id'])
                time.sleep(0.5)
            sorted_stocks = sorted(
                sorted_stocks,
                key=lambda x: (x['price_change_percentage_1h'] + x['sentiment'], x['price_change_percentage_24h']),
                reverse=True
            )
            
            print(f"✅ Found {len(sorted_stocks)} candidates")
            return sorted_stocks

        except Exception as e:
            print(f"⚠️ Error scanning candidates: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            return []
        
    def calculate_dynamic_thresholds(self, top_stocks):
        """Calculate dynamic thresholds based on 7-day market-wide volatility."""
        try:
            if not top_stocks:
                print("⚠️ No top stocks retrieved, using default thresholds")
                return 0.5, 1.0  # Lowered defaults
            
            mapped_stocks = self.map_to_exchange(top_stocks)
            if not mapped_stocks:
                print("⚠️ No mapped stocks available, using default thresholds")
                return 0.5, 1.0
                        
            vol_1h_list = []
            vol_1d_list = []
            min_valid_stocks = 5

            for stock in mapped_stocks:
                symbol = stock.get('selected_symbol')
                exchange = stock.get('exchange')
                if not symbol or not exchange:
                    print(f"⚠️ Skipping {stock.get('symbol', 'unknown')} due to missing symbol or exchange")
                    continue

                # Fetch 1-hour and 1-day data
                df_1h = self.fetch_data(symbol, '1h', exchange)
                df_1d = self.fetch_data(symbol, '1d', exchange)

                if df_1h is not None and not df_1h.empty and df_1d is not None and not df_1d.empty:
                    returns_1h = df_1h['close'].pct_change().dropna()
                    returns_1d = df_1d['close'].pct_change().dropna()
                    if not returns_1h.empty and not returns_1d.empty:
                        vol_1h = float(returns_1h.std()) * 1.0  # Reduced multiplier
                        vol_1d = float(returns_1d.std()) * 1.0  # Reduced multiplier
                        vol_1h_list.append(vol_1h)
                        vol_1d_list.append(vol_1d)
                        print(f"✅ Calculated volatility for {stock['symbol']} on {exchange}: 1h={vol_1h:.4f}, 1d={vol_1d:.4f}")
                    else:
                        print(f"⚠️ Empty returns for {stock['symbol']} (1h: {len(returns_1h)}, 1d: {len(returns_1d)})")
                else:
                    print(f"⚠️ Invalid data for {stock['symbol']} (1h: {df_1h is None or df_1h.empty}, 1d: {df_1d is None or df_1d.empty})")

            if len(vol_1h_list) < min_valid_stocks or len(vol_1d_list) < min_valid_stocks:
                print(f"⚠️ Insufficient valid stocks ({len(vol_1h_list)} < {min_valid_stocks}), using default thresholds")
                return 0.5, 1.0
            
            # Compute market-wide thresholds as the median of individual stock volatilities
            market_vol_1h = np.median(vol_1h_list)
            market_vol_1d = np.median(vol_1d_list)
            
            # Ensure minimum thresholds
            final_vol_1h = max(market_vol_1h, 0.5)  # Lowered minimum
            final_vol_1d = max(market_vol_1d, 1.0)  # Lowered minimum
            
            print(f"✅ Market-wide thresholds: 1h={final_vol_1h:.4f}, 1d={final_vol_1d:.4f} (based on {len(vol_1h_list)} stocks)")
            return final_vol_1h, final_vol_1d

        except Exception as e:
            print(f"⚠️ Error calculating market-wide thresholds: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.5, 1.0
        
    def map_to_exchange(self, stocks):
        """Map stocks to yfinance (no real mapping needed)."""
        sorted_stocks = []
        try:
            for stock in stocks:
                symbol = stock['symbol'].upper()
                stock_id = stock['id'].lower()
                stock_copy = stock.copy()
                stock_copy['yahoo_symbol'] = symbol
                print(f"✅ Found Yahoo pair for {symbol} ({stock['id']}): {symbol}")

                # Fetch data to check freshness
                data_1m = self.fetch_data(symbol, '1m', 'yahoo')
                if data_1m is not None and not data_1m.empty:
                    latest_time = data_1m.index[-1]
                    time_diff = (datetime.now(EST) - latest_time).total_seconds() / 60
                    
                    # If data is less than 1 minute old, select and stop
                    if time_diff < 1.0:
                        stock_copy['exchange'] = 'yahoo'
                        stock_copy['selected_symbol'] = symbol
                        print(f"✅ Selected Yahoo for {symbol} ({stock['id']}) as data is less than 1 minute old ({time_diff:.2f} min)")
                        sorted_stocks.append(stock_copy)
                        continue

                # Fallback to freshest data
                if data_1m is not None and not data_1m.empty:
                    if time_diff <= FRESHNESS_THRESHOLD_1M:
                        stock_copy['exchange'] = 'yahoo'
                        stock_copy['selected_symbol'] = symbol
                        print(f"✅ Selected Yahoo for {symbol} ({stock['id']}) as fresher data ({time_diff:.2f} min)")
                        sorted_stocks.append(stock_copy)
                    else:
                        print(f"⚠️ No fresh data for {symbol} ({stock['id']}) on Yahoo, skipping")
                else:
                    print(f"⚠️ No valid data for {symbol} ({stock['id']}) on Yahoo, skipping")

            return sorted_stocks
        
        except Exception as e:
            print(f"⚠️ Error mapping stocks to Yahoo: {str(e)}")
            return []
        
    def fetch_news_sentiment(self, stock_symbol, stock_id=None, max_retries=3):
        """Fetch up to 100 recent news articles for the given stock and compute sentiment score."""
        global RATE_LIMIT_HIT
        cache_key = f"{stock_symbol}:{stock_id or ''}"
        current_time = datetime.now(pytz.UTC).timestamp()

        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            if current_time - cached['timestamp'] < SENTIMENT_CACHE_TTL:
                print(f"✅ Using cached sentiment for {stock_symbol}: {cached['sentiment']:.3f}")
                return cached['sentiment']
        if RATE_LIMIT_HIT:
            print(f"⚠️ Skipping sentiment for {stock_symbol} due to previous rate limit")
            return 0.0

        try:
            if not NEWS_API_KEY:
                print("⚠️ News API key not found, skipping sentiment analysis")
                return 0.0

            query = f"{stock_symbol} stock OR {stock_id or stock_symbol} stock"
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=100&apiKey={NEWS_API_KEY}"
            from_date = (datetime.now(pytz.UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
            params = {'from': from_date}

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    articles = data.get('articles', [])
                    print(f"✅ Found {len(articles)} articles for {stock_symbol}")
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:
                        print(f"⚠️ NewsAPI rate limit hit for {stock_symbol} (attempt {attempt + 1}/{max_retries})")
                        if attempt == max_retries - 1:
                            RATE_LIMIT_HIT = True
                            print("⚠️ Global rate limit set, skipping further sentiment fetches")
                            return 0.0
                        time.sleep(2 ** (attempt + 1))
                        continue
                    print(f"⚠️ HTTP error for {stock_symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2 ** attempt)
                    continue
                except Exception as e:
                    print(f"⚠️ Error fetching news for {stock_symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
            else:
                print(f"⚠️ Failed to fetch news for {stock_symbol} after {max_retries} attempts")
                return 0.0

            if not articles:
                print(f"⚠️ No recent news found for {stock_symbol}")
                return 0.0

            relevant_articles = []
            for article in articles:
                title = article.get('title', '') or ''
                description = article.get('description', '') or ''
                text = f"{title} {description}".lower()
                if stock_symbol.lower() in text or (stock_id and stock_id.lower() in text) or 'stock' in text:
                    relevant_articles.append(article)
            
            if not relevant_articles:
                print(f"⚠️ No relevant news articles found for {stock_symbol}")
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
                        print(f"⚠️ Error processing article for {stock_symbol}: {str(e)}")
                        continue
            
            if not sentiment_scores:
                print(f"⚠️ No valid sentiment scores for {stock_symbol}")
                return 0.0

            weighted_sentiment = np.average(sentiment_scores, weights=weights)
            print(f"📰 Sentiment score for {stock_symbol}: {weighted_sentiment:.3f} (based on {len(sentiment_scores)} articles)")
            
            self.sentiment_cache[cache_key] = {'sentiment': weighted_sentiment, 'timestamp': current_time}
            self.save_sentiment_cache()
            return weighted_sentiment

        except Exception as e:
            print(f"⚠️ Error fetching news sentiment for {stock_symbol}: {str(e)}")
            return 0.0
        
    # ------------- Data Fetching and Processing ------------- # 
    def fetch_data(self, symbol, interval, exchange):
        """Fetch OHLCV data from yfinance for analysis."""
        timeframe = {'1m': '1m', '1h': '1h', '1d': '1d'}.get(interval, '1m')
        if timeframe == '1m':
            period = '7d'  # Max for 1m
        elif timeframe == '1h':
            period = '1mo'
        else:
            period = '5d'  # Extended for 1d
        
        try:
            df = yf.download(symbol, period=period, interval=timeframe)
            if df.empty:
                print(f"⚠️ No data for {symbol}")
                return None
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df.index = df.index.tz_convert(EST) if df.index.tz else df.index.tz_localize('UTC').tz_convert(EST)
            latest_time = df.index[-1]
            current_time = datetime.now(EST)
            time_diff = (current_time - latest_time).total_seconds() / 60
            print(f"yfinance data for {symbol}: {time_diff:.2f} minutes old")
            
            # Validate data
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                print(f"⚠️ Invalid prices (zero or negative) for {symbol}")
                return None
            if df.duplicated().any():
                print(f"⚠️ Duplicate rows found for {symbol}, removing duplicates")
                df = df[~df.duplicated()]
            
            # Relax freshness check for 1-day data
            if interval == '1d' and time_diff > FRESHNESS_THRESHOLD_1D:
                print(f"⚠️ yfinance 1d data for {symbol} is stale ({time_diff:.2f} minutes old), but proceeding")
            elif interval == '1m' and time_diff > FRESHNESS_THRESHOLD_1M:
                print(f"⚠️ yfinance data for {symbol} is stale ({time_diff:.2f} minutes old)")
                return None
            elif interval == '1h' and time_diff > FRESHNESS_THRESHOLD_1H:
                print(f"⚠️ yfinance data for {symbol} is stale ({time_diff:.2f} minutes old)")
                return None
            
            print(f"✅ Fetched {len(df)} candles for {symbol} (latest: {latest_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')})")
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_trade_data(self, symbol, interval, exchange, start_time=None, end_time=None):
        """Fetch OHLCV data within a specific time range for trade evaluation."""
        timeframe = {'1m': '1m', '1h': '1h', '1d': '1d'}.get(interval, '1m')
        period = '1mo' if start_time is None else None
        
        try:
            if start_time and end_time:
                df = yf.download(symbol, start=start_time, end=end_time, interval=timeframe)
            else:
                df = yf.download(symbol, period=period, interval=timeframe)
            if df.empty:
                print(f"⚠️ No data for {symbol}")
                return None
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            # Reset MultiIndex and ensure clean DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_convert(EST) if df.index.tz else df.index.tz_localize('UTC').tz_convert(EST)
            
            if start_time and end_time:
                df = df[(df.index >= start_time) & (df.index <= end_time)]
            if df.empty:
                print(f"⚠️ Empty dataframe after filtering for {symbol}")
                return None
            print(f"✅ Fetched {len(df)} candles for {symbol}")
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching trade data for {symbol}: {str(e)}")
            return None

    def calculate_features(self, df, df_type):
        """Add technical indicators and sentiment score to the DataFrame."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {df_type}, cannot calculate features")
            return None

        lags = [1, 3, 5]
        rolling_windows = [3, 6]

        try:
            print(f"📊 Adding technical indicators for {df_type} data")
            df = df.copy()
            # Reset MultiIndex and ensure clean DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)  # Use first level (open, high, low, close, volume, sentiment)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['rsi'] = ta.rsi(df['close'], length=14, talib=False)
            macd = ta.macd(df['close'], talib=False)
            df['macd'] = macd['MACD_12_26_9']
            bbands = ta.bbands(df['close'], length=20, std=2, talib=False)
            df['bollinger'] = bbands['BBM_20_2.0']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14, talib=False)
            df['sma_20'] = ta.sma(df['close'], length=20, talib=False)
            df['ema_20'] = ta.ema(df['close'], length=20, talib=False)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14, talib=False)
            df['adx'] = adx['ADX_14']
            df['obv'] = ta.obv(df['close'], df['volume'], talib=False)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3, talib=False)
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

            print(f"✅ Technical indicators added for {df_type} data")
            return df

        except Exception as e:
            print(f"⚠️ Error in calculate_features for {df_type}: {str(e)}")
            return None
        
    def label_1m_model(self, stock, df, window=360, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-minute model with dynamic threshold."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {stock['symbol']} (1m), cannot label")
            return None
        try:
            print(f"📊 Attempting to label 1m model for {stock['symbol']}")

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
                print(f"📈 Dynamic threshold for {stock['symbol']}: {dynamic_threshold:.4f}")
            else:
                dynamic_threshold = min_return
                print(f"⚠️ No valid returns for {stock['symbol']}, using default threshold: {min_return}")
            
            df['label'] = (df['target_return'] >= dynamic_threshold).astype(int)
            labeled_df = df.dropna()
            
            # Check class distribution
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1m labels for {stock['symbol']}, will rely on regressor")
            
            return labeled_df

        except Exception as e:
            print(f"⚠️ Error preparing 1m data: {str(e)}")
            return None

    def label_1h_model(self, stock, df, forward_hours=6, min_return=THRESHOLDS[0], max_return=THRESHOLDS[1]):
        """Prepare labeled data for 1-hour model with dynamic threshold and target_return."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {stock['symbol']} (1h), cannot label")
            return None
        try:
            print(f"📊 Attempting to label 1h model for {stock['symbol']}")
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
                print(f"📈 Dynamic threshold for {stock['symbol']}: {dynamic_threshold:.4f}")
            else:
                dynamic_threshold = min_return
                print(f"⚠️ No valid returns for {stock['symbol']}, using default threshold: {min_return}")
            
            df['label'] = (df['fwd_return'] >= dynamic_threshold).astype(int)
            labeled_df = df.dropna()
            
            # Check class distribution
            if labeled_df['label'].nunique() < 2:
                print(f"⚠️ Only one class in 1h labels for {stock['symbol']}, will rely on regressor")
            
            return labeled_df
        
        except Exception as e:
            print(f" Error preparing 1h data: {str(e)}")
            return None

    def train_hybrid_model(self, df, stock, df_type):
        """Train a hybrid model using XGBoost, with fallback to regressor if classification fails."""
        if df is None or df.empty:
            print(f"⚠️ Invalid or empty DataFrame for {stock['symbol']} ({df_type}), cannot train model")
            return None, None, None
        drop_cols = ['close', 'future_max', 'fwd_return']

        try:
            print(f"📊 Training {stock['symbol']} hybrid model for {df_type} data")

            df = df.copy()
            features = [col for col in df.columns if col not in ['label', 'target_return'] + drop_cols and pd.api.types.is_numeric_dtype(df[col])]
            X = df[features]
            y_class = df['label']
            y_reg = df['target_return'] if 'target_return' in df.columns else None

            split = int(0.8 * len(X))
            if split < 2 or len(X) - split < 2:
                print(f"⚠️ Insufficient data for {stock['symbol']} ({df_type}) after split")
                return None, None, None
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_class_train, y_class_test = y_class.iloc[:split], y_class.iloc[split:]

            # Load base models if available
            base_clf = self.models.get(df_type, {}).get('clf')
            base_reg = self.models.get(df_type, {}).get('reg')
            base_score = self.models.get(df_type, {}).get('score', 0.0)

            # Train classifier
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
                class_ratio = len(y_class_train_res[y_class_train_res == 0]) / len(y_class_train_res[y_class_train_res == 1]) if 1 in y_class_train_res else 1.0
                clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric='auc', early_stopping_rounds=10,
                    random_state=42, scale_pos_weight=class_ratio
                )
                clf.fit(X_train_res, y_class_train_res, eval_set=[(X_test, y_class_test)], verbose=False, xgb_model=base_clf if base_clf else None)
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_class_test, y_pred)
                print(f"✅ Classifier for {stock['symbol']} ({df_type}) - F1 Score: {f1:.4f}")

                # Lightweight tuning if performance is poor
                if f1 < 0.5 or (base_clf and f1 < base_score * 0.9):
                    print(f"⚠️ Classifier F1 score {f1:.4f} is low or worse than base ({base_score:.4f}), attempting lightweight tuning")
                    best_f1 = f1
                    best_clf = clf
                    for lr in [0.05, 0.2]:
                        for depth in [2, 4]:
                            temp_clf = XGBClassifier(
                                n_estimators=100, max_depth=depth, learning_rate=lr,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric='auc', early_stopping_rounds=10,
                                random_state=42, scale_pos_weight=class_ratio
                            )
                            temp_clf.fit(X_train_res, y_class_train_res, eval_set=[(X_test, y_class_test)], verbose=False, xgb_model=base_clf if base_clf else None)
                            temp_pred = temp_clf.predict(X_test)
                            temp_f1 = f1_score(y_class_test, temp_pred)
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
                print(f"✅ Regressor for {stock['symbol']} ({df_type}) - RMSE: {rmse:.5f}")

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
        """Predict the future movement of the best stock by simulating realistic data points."""
        print(f"\n🔮 Predicting future movement for {best_stock['symbol']}")

        # Ensure latest timestamp is in EST
        latest_timestamp = best_1m_df.index[-1]
        if not latest_timestamp.tzinfo:
            latest_timestamp = latest_timestamp.tz_localize('UTC').tz_convert(EST)
        else:
            latest_timestamp = latest_timestamp.tz_convert(EST)
        
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
            print(f"⚠️ GARCH fitting failed: {str(e)}, using historical volatility")
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
            print(f"❌ No sell signal found for {best_stock['symbol']} within 6 hours")

        return signal_time, predicted_return

# ------------- MAIN FUNCTION ------------- #
    def run_analysis(self):
        """Main analysis pipeline with sentiment-integrated buy score."""
        global best_score, best_stock, best_analysis, best_1m_df, best_1h_df
        print('🚀 Starting trading algorithm')
        current_time = datetime.now(EST)
        print(f"📅 Analysis started at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")

        # Scan for candidates
        candidates = self.scan_candidates()

        if not candidates:
            print(f"❌ No candidates found at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
            return
        
        # Analyze each candidate
        for stock in candidates:
            print(f"\n📊 Analyzing {stock['symbol']} ({stock['selected_symbol']} on {stock['exchange']})")
            df_1m = self.fetch_data(stock['selected_symbol'], '1m', stock['exchange'])
            df_1h = self.fetch_data(stock['selected_symbol'], '1h', stock['exchange'])
            
            if df_1m is None or df_1h is None:
                print(f"⚠️ Skipping {stock['symbol']} due to data fetch error")
                continue

            # Add sentiment to dataframes
            df_1m['sentiment'] = stock['sentiment']
            df_1h['sentiment'] = stock['sentiment']

            # Calculate features
            df_1m, df_1h = self.calculate_features(df_1m, '1m'), self.calculate_features(df_1h, '1h')

            if df_1m is None or df_1h is None:
                print(f"⚠️ Skipping {stock['symbol']} due to calculation error")
                continue

            # Label data
            labeled_1m = self.label_1m_model(stock, df_1m)
            labeled_1h = self.label_1h_model(stock, df_1h)

            if labeled_1m is None or labeled_1h is None:
                print(f"⚠️ Skipping {stock['symbol']} due to labeling error")
                continue

            # Train models
            clf_1m, reg_1m, feat_1m = self.train_hybrid_model(labeled_1m, stock, '1m')
            clf_1h, reg_1h, feat_1h = self.train_hybrid_model(labeled_1h, stock, '1h')

            # Use regressor score if classifier fails
            score = clf_1m['score'] if clf_1m is not None else (reg_1m['rmse'] if reg_1m is not None else -np.inf)
            if score > best_score:
                best_score = score
                best_stock = stock
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

                print(f"🏆 New best: {stock['symbol']} ({stock['selected_symbol']} on {stock['exchange']}) (Score: {best_score:.4f})")
        
        if best_analysis is None:
            print(f"❌ No valid models found at {current_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}")
            return
        
        # Run final analysis on the best stock
        print(f"\n🏆 Running final analysis on {best_stock['symbol']} ({best_stock['selected_symbol']} on {best_stock['exchange']})")
        
        # Get the latest row from each labeled dataframe
        latest_1m = best_1m_df.iloc[[-1]][best_analysis['feat_1m']]
        latest_1h = best_1h_df.iloc[[-1]][best_analysis['feat_1h']]

        # Make predictions - Probability of class 1
        prob_1m = best_analysis['clf_1m']['model'].predict_proba(latest_1m)[0][1] if best_analysis['clf_1m'] is not None else 0.5
        prob_1h = best_analysis['clf_1h']['model'].predict_proba(latest_1h)[0][1] if best_analysis['clf_1h'] is not None else 0.5

        # Combine predictions with sentiment
        sentiment_score = best_stock['sentiment']
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

        if buy:
            close_price = best_1m_df['close'].iloc[-1]
            # Final buy report
            message = [
                f"Stock Trading:",
                f"✅ Action: BUY {best_stock['symbol']} at ${close_price:.4f}",
                f"🤖 Buy score: {buy_score:.2f} (1m: {prob_1m:.2f}, 1h: {prob_1h:.2f}, sentiment: {sentiment_score:.2f})",
                f"📈 Expected return: {expected_return * 100:.2f}%",
                f"💰 Position size: {position_size_pct * 100:.1f}%"
            ]
            
            signal_time, predicted_return = self.predict_future_movement(expected_return, sentiment_score)

            if signal_time is None or predicted_return is None:
                print(f"❌ No valid future movement prediction for {best_stock['symbol']}")
            else:
                self.record_trade(
                    stock=best_stock,
                    entry_price=best_1m_df['close'].iloc[-1],
                    expected_return=expected_return,
                    entry_time=current_time,
                    sell_time=signal_time,
                    features_1m=best_1m_df[best_analysis['feat_1m']].iloc[-1].to_dict(),
                    features_1h=best_1h_df[best_analysis['feat_1h']].iloc[-1].to_dict(),
                    buy_score=buy_score,
                    position_size_pct=position_size_pct,
                    exchange=best_stock['exchange']                   
                )

                message.extend([
                    "",
                    f"🎯 Predicted sell signal for {best_stock['symbol']} at {signal_time.strftime('%m-%d-%Y %H:%M:%S %Z%z')}",
                    f"📈 Predicted sell price: ${(1 + predicted_return) * close_price:.4f} (predicted return: {predicted_return * 100:.2f}%)"
                ])
                print(message)
                self.send_telegram_message("\n".join(message))
        
        else:
            print(f"❌ No buy signal for best stock: {best_stock['symbol']} ({best_stock['selected_symbol']} on {best_stock['exchange']})")

if __name__ == '__main__':
    trader = StockTrader()
    trader.run_analysis()
    trader.evaluate_pending_trades()
