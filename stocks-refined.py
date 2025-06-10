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
import json
import joblib
from joblib import dump, load
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
WINDOWS = [10, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 720]
MIN_DATA_POINTS = 1000
MODEL_DIR = 'models'
TRADE_HISTORY_FILE = 'trade_history.json'
MODEL_METADATA_FILE = 'model_metadata.json'
FEEDBACK_INTERVAL_HOURS = 6
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
EST = pytz.timezone('America/New_York')

# Initialize directories
os.makedirs(MODEL_DIR, exist_ok=True)

class StockTrader:
    def __init__(self):
        self.models = {}
        self.trade_history = []
        self.model_metadata = {}
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
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
                    ),
                    'score': self.model_metadata.get(str(threshold), {}).get('score', 0)
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
            self._commit_to_github()

    def _commit_to_github(self):
        """Commit changes to GitHub repository"""
        print("Persisting files for GitHub Actions...")
        os.system("git config --global user.email 'actions@github.com'")
        os.system("git config --global user.name 'GitHub Actions'")
        os.system("git add .")
        os.system(f"git commit -m 'Update models and trade history {datetime.now().isoformat()}'")
        os.system("git push")

    def scan_day_trade_candidates(self):
        """Scan Finviz for potential day trading candidates"""
        print("🔍 Scanning Finviz for high-momentum small caps...")
        
        try:
            perf_screen = Performance()
            perf_screen.set_filter({
                "Price": "Under $20",
                "Relative Volume": "Over 2",
                "Average Volume": "Over 500K",
                "Change": "Up 10%"
            })
            perf_df = perf_screen.screener_view(limit=200, verbose=0)

            if perf_df.empty:
                return []

            tickers = perf_df['Ticker'].tolist()
            
            # Filter by float
            own_screen = Ownership()
            own_screen.set_filter(ticker=",".join(tickers))
            own_df = own_screen.screener_view(limit=len(tickers), verbose=0)

            if own_df.empty:
                return []

            def float_to_num(s):
                if isinstance(s, str):
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

    def fetch_stock_data(self, ticker, interval="5m", days=60):
        """Fetch stock data with multiple fallback options"""
        try:
            df = yf.download(
                ticker,
                period=f"{days}d",
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
            if len(df) < MIN_DATA_POINTS:
                return None
                
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            return None

    def calculate_features(self, df):
        """Add technical indicators to DataFrame"""
        if df.empty:
            return None
            
        close_col = 'Close'
        
        # Basic features
        df['returns'] = df[close_col].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Technical indicators
        df['rsi'] = ta.rsi(df[close_col], length=14)
        df['macd'] = ta.macd(df[close_col])['MACD_12_26_9']
        df['bollinger'] = ta.bbands(df[close_col])['BBL_20_2.0']
        df['atr'] = ta.atr(df['High'], df['Low'], df[close_col], length=14)
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df

    def prepare_training_data(self, df, threshold, window):
        """Create labeled dataset for training"""
        df = df.copy()
        df['future_max'] = df['Close'].shift(-window).rolling(window, min_periods=1).max()
        df['target'] = ((df['future_max'] / df['Close'] - 1) >= threshold).astype(int)
        df.dropna(subset=['target'], inplace=True)
        
        if df['target'].nunique() < 2:
            return None
            
        return df

    def train_model(self, threshold, X_train, y_train, X_val, y_val):
        """Train or update a model with new data"""
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
                'last_retrain': datetime.now(),
                'score': 0
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

    def record_trade(self, ticker, threshold, features, prediction, outcome):
        """Store trade details for feedback"""
        trade_id = hashlib.md5(
            f"{ticker}{threshold}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        self.trade_history.append({
            'id': trade_id,
            'ticker': ticker,
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

    def run_analysis(self):
        """Main analysis pipeline with continual learning"""
        # 1. Evaluate recent trades and update models
        self.retrain_models()
        
        # 2. Scan for candidates
        tickers = self.scan_day_trade_candidates()
        if not tickers:
            print("No tickers found for analysis")
            return
            
        # 3. Analyze each candidate
        for ticker in tickers[:5]:  # Limit to top 5 for efficiency
            print(f"\nAnalyzing {ticker}")
            df = self.fetch_stock_data(ticker)
            if df is None:
                continue
                
            df = self.calculate_features(df)
            if df is None:
                continue
                
            # 4. Train/update models for each threshold
            for threshold in THRESHOLDS:
                for window in WINDOWS:
                    labeled_data = self.prepare_training_data(df, threshold, window)
                    if labeled_data is None:
                        continue
                        
                    features = [col for col in labeled_data.columns 
                              if col not in ['target', 'future_max']]
                    
                    X = labeled_data[features]
                    y = labeled_data['target']
                    
                    split = int(0.8 * len(X))
                    X_train, X_val = X.iloc[:split], X.iloc[split:]
                    y_train, y_val = y.iloc[:split], y.iloc[split:]
                    
                    model = self.train_model(threshold, X_train.values, y_train.values, X_val.values, y_val.values)
                    
                    # 5. Make predictions and simulate trades
                    current_features = X.iloc[-1:].values
                    prediction = model.predict(current_features)[0]
                    proba = model.predict_proba(current_features)[0][1]
                    
                    if prediction == 1 and proba > 0.7:  # Confidence threshold
                        print(f"🚀 Trade signal: {ticker} at {threshold:.2%} threshold")
                        # Simulate trade outcome (in reality, track actual trades)
                        outcome = 1 if np.random.random() > 0.4 else 0  # Simulated outcome
                        self.record_trade(
                            ticker,
                            threshold,
                            X.iloc[-1].to_dict(),
                            prediction,
                            outcome
                        )
        
        self.save_state()

if __name__ == "__main__":
    trader = StockTrader()
    trader.run_analysis()