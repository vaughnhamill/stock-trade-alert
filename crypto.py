import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import numpy as np
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
THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005]  # Percentage targets
WINDOWS = [15, 30, 60, 120, 240, 360]  # Minutes
MIN_DATA_POINTS = 1000  # Minimum candles required
EST = pytz.timezone('America/New_York')
MODEL_DIR = 'models'
TRADE_HISTORY_FILE = 'trade_history.json'
MODEL_METADATA_FILE = 'model_metadata.json'
FEEDBACK_INTERVAL_HOURS = 6  # How often to retrain models
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'

# Initialize CoinGecko API
cg = CoinGeckoAPI()

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

    def calculate_features(self, df):
        """Add technical indicators"""
        if df.empty:
            return None
            
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Technical indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['bollinger'] = ta.bbands(df['close'])['BBL_20_2.0']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df

    def prepare_training_data(self, df, threshold, window):
        """Create labeled dataset"""
        df = df.copy()
        df['future_max'] = df['close'].shift(-window).rolling(window, min_periods=1).max()
        df['target'] = ((df['future_max'] / df['close'] - 1) >= threshold).astype(int)
        df.dropna(subset=['target'], inplace=True)
        
        if df['target'].nunique() < 2:
            return None
            
        return df

    def train_model(self, threshold, X_train, y_train, X_val, y_val):
        """Train or update a model"""
        if threshold not in self.models:
            # Initialize new model
            model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                eval_metric='logloss',
                early_stopping_rounds=10,
                random_state=42
            )
            self.models[threshold] = {
                'model': model,
                'last_retrain': datetime.now()
            }
        else:
            model = self.models[threshold]['model']
            
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Update metadata
        self.models[threshold]['last_retrain'] = datetime.now()
        self.models[threshold]['score'] = accuracy_score(y_val, model.predict(X_val))
        
        return model

    def record_trade(self, symbol, threshold, features, prediction, outcome):
        """Store trade details for feedback"""
        trade_id = hashlib.md5(
            f"{symbol}{threshold}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        self.trade_history.append({
            'id': trade_id,
            'symbol': symbol,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'outcome': outcome
        })
        
        # Trigger retrain if enough new trades
        self.retrain_models()

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
                
        self.save_state()
# --- Code Checked for Method and Errors ---
    def scan_candidates(self):
            """Find potential trading candidates"""
            print("🔍 Scanning for candidates...")
            try:
                coins = cg.get_coins_markets(
                    vs_currency='usd',
                    order='market_cap_desc',
                    per_page=50,
                    price_change_percentage='24h'
                )
                
                return [{
                    'id': c['id'],
                    'symbol': c['symbol'].upper(),
                    'price_change_percentage_24h': c['price_change_percentage_24h']
                } for c in coins if c['price_change_percentage_24h'] > 5]  # Min 5% daily change
                
            except Exception as e:
                print(f"⚠️ Error scanning candidates: {str(e)}")
                return []

    def fetch_crypto_data(self, coin_id, interval={"5m": "60d", "10m": "60d", "15m": "60d"}, max_attempts=3):
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
                    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Resample to desired interval
                    ohlc = {
                        'price': {
                            'open': 'first',
                            'high': 'max', 
                            'low': 'min',
                            'close': 'last'
                        }
                    }
                    df = df.resample(current_interval).agg(ohlc)
                    df.columns = ['open', 'high', 'low', 'close']
                    print("📈 Data fetched successfully")
                    
                    if len(df) >= MIN_DATA_POINTS:
                        print(f"✅ Enough data points ({len(df)}) for {coin_id}")
                        return df
                    return None
                    
                except Exception as e:
                    print(f"⚠️ Error fetching data: {str(e)}")
                    time.sleep(5)
                
                attempts += 1

            print(f"❌ Failed to fetch data for {coin_id} after {max_attempts} attempts")    
            return None

    def run_analysis(self):
        """Main analysis pipeline"""
        print("🚀 Starting trading algorithm")
        # 1. Scan for candidates
        # FOR TESTING PURPOSES
        candidates = [{'id': 'avalanche-2', 'symbol': 'AVAX', 'price_change_percentage_24h': 6.5}]
        # candidates = self.scan_candidates()

        # if not candidates:
        #     print("❌ No candidates found")
        #     return
        
        # 2. Analyze each candidate
        for coin in candidates[:10]:  # Limit to top 10
            print(f"\n📊 Analyzing {coin['symbol']}")
            df = self.fetch_crypto_data(coin['id'])
            # if df is None:
            #     continue
                
        #     df = self.calculate_features(df)
        #     if df is None:
        #         continue
                
        #     # 3. Train/update models for each threshold
        #     for threshold in THRESHOLDS:
        #         for window in WINDOWS:
        #             labeled_data = self.prepare_training_data(df, threshold, window)
        #             if labeled_data is None:
        #                 continue
                        
        #             features = [col for col in labeled_data.columns 
        #                       if col not in ['target', 'future_max']]
                    
        #             X = labeled_data[features]
        #             y = labeled_data['target']
                    
        #             split = int(0.8 * len(X))
        #             X_train, X_val = X.iloc[:split], X.iloc[split:]
        #             y_train, y_val = y.iloc[:split], y.iloc[split:]
                    
        #             model = self.train_model(threshold, X_train.values, y_train.values, X_val.values, y_val.values)
                    
        #             # 4. Make predictions and simulate trades
        #             current_features = X.iloc[-1:].values
        #             prediction = model.predict(current_features)[0]
        #             proba = model.predict_proba(current_features)[0][1]
                    
        #             if prediction == 1 and proba > 0.7:  # Confidence threshold
        #                 print(f"🚀 Trade signal: {coin['symbol']} at {threshold:.2%} threshold")
        #                 # Simulate trade outcome (in reality, track actual trades)
        #                 outcome = 1 if np.random.random() > 0.4 else 0  # Simulated outcome
        #                 self.record_trade(
        #                     coin['symbol'],
        #                     threshold,
        #                     X.iloc[-1].to_dict(),
        #                     prediction,
        #                     outcome
        #                 )
        
        # self.save_state()

if __name__ == "__main__":
    trader = CryptoTrader()
    trader.run_analysis()