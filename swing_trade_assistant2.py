import os
import math
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import asyncio
import aiohttp
import time
import json
from flask import Flask, render_template_string, request, jsonify
import plotly.graph_objects as go
import plotly.express as px
from functools import lru_cache
import concurrent.futures

# Para análise de sentimento (NLTK)
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Suprimir mensagens do yfinance
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ---------------- CONFIGURAÇÕES GERAIS ----------------
PORTFOLIO_FILE = "portfolioatual.csv"  # Arquivo CSV com suas posições
LOG_FILE = "trade_log.txt"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Valores iniciais (atualizáveis via formulário)
ACCOUNT_BALANCE = 3000.0
INVESTED_AMOUNT = 0.0
RISK_TOLERANCE = "medium"  # low, medium, high

# Chave da NewsAPI
NEWSAPI_KEY = "1c381d2eb1454239a527ebe33bcaff15"

# Mapeamento de setores -> ETFs representativos
sector_etfs = {
    "Technology": ["QQQ", "XLK", "VGT", "SMH", "SOXX"],
    "Healthcare": ["XLV", "IYH", "VHT", "IBB", "XBI"],
    "Financial Services": ["XLF", "VFH", "KRE", "KBE", "KBWB"],
    "Consumer Cyclical": ["XLY", "IYC", "VCR", "PEJ", "RTH"],
    "Consumer Defensive": ["XLP", "VDC", "KXI", "IYK", "FSTA"],
    "Energy": ["XLE", "VDE", "OIH", "XOP", "AMLP"],
    "Industrials": ["XLI", "VIS", "IYJ", "ITA", "XTN"],
    "Materials": ["XLB", "VAW", "IYM", "GDX", "SLX"],
    "Utilities": ["XLU", "VPU", "IDU", "URA", "TAN"],
    "Real Estate": ["VNQ", "IYR", "XLRE", "REZ", "REM"],
    "Communication Services": ["XLC", "VOX", "IYZ", "FCOM", "PBS"],
    "Crypto": ["BITO", "BTCZ", "ETHE", "GBTC", "FBTC"]
}

# Top tickers by market cap per sector (for opportunity scanning)
top_tickers_by_sector = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "IBM"],
    "Healthcare": ["LLY", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN"],
    "Financial Services": ["V", "JPM", "MA", "BAC", "WFC", "MS", "GS", "BLK", "SPGI", "AXP"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "BKNG", "MAR", "HLT"],
    "Consumer Defensive": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "EL", "CL", "GIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "VLO", "PSX", "PXD", "OXY", "MPC"],
    "Industrials": ["RTX", "HON", "UPS", "CAT", "LMT", "GE", "DE", "BA", "MMM", "ITW"],
    "Materials": ["LIN", "APD", "ECL", "SHW", "CRH", "FCX", "NEM", "CTVA", "NUE", "DOW"],
    "Utilities": ["NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "PCG", "ED", "EXC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "DLR", "VICI", "SPG"],
    "Communication Services": ["GOOG", "META", "NFLX", "TMUS", "CMCSA", "VZ", "T", "DIS", "CHTR", "EA"]
}

# Configurations for signal weights based on risk profiles
signal_weights = {
    "low": {
        "RSI": 0.15,
        "MA_crossover": 0.25,
        "bollinger": 0.15,
        "sentiment": 0.10,
        "sector_momentum": 0.15,
        "volume": 0.10,
        "macd": 0.10
    },
    "medium": {
        "RSI": 0.20,
        "MA_crossover": 0.20,
        "bollinger": 0.15,
        "sentiment": 0.15,
        "sector_momentum": 0.10,
        "volume": 0.10,
        "macd": 0.10
    },
    "high": {
        "RSI": 0.25,
        "MA_crossover": 0.15,
        "bollinger": 0.10,
        "sentiment": 0.20,
        "sector_momentum": 0.10,
        "volume": 0.10,
        "macd": 0.10
    }
}

# ---------------- FUNÇÃO DE LOG ----------------
def setup_logger():
    """Configure advanced logging with proper formatting"""
    logger = logging.getLogger('swing_trade')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def log_message(message):
    """Legacy log function for backward compatibility"""
    logger.info(message)

# ---------------- CACHE MANAGEMENT ----------------
def get_cache_path(cache_type, key):
    """Get path for a cache file"""
    import hashlib
    filename = hashlib.md5(str(key).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{cache_type}_{filename}.json")

def save_to_cache(cache_type, key, data, expire_seconds=3600):
    """Save data to cache with expiration"""
    cache_file = get_cache_path(cache_type, key)
    cache_data = {
        'data': data,
        'expires': time.time() + expire_seconds
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def load_from_cache(cache_type, key):
    """Load data from cache if not expired"""
    cache_file = get_cache_path(cache_type, key)
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        if time.time() < cache_data.get('expires', 0):
            return cache_data.get('data')
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
    
    return None

# ---------------- LÓGICA DE LIMIARES DINÂMICOS ----------------
def get_dynamic_thresholds(total_investido, risk_profile="medium"):
    """
    Exemplo:
      - Meta global: 5% do total investido (mínimo US$100)
      - Meta parcial: 0.5% do total investido (mínimo US$10)
    Ajustado por perfil de risco
    """
    risk_multipliers = {
        "low": {"global": 0.04, "partial": 0.004, "min_global": 80, "min_partial": 8},
        "medium": {"global": 0.05, "partial": 0.005, "min_global": 100, "min_partial": 10},
        "high": {"global": 0.06, "partial": 0.006, "min_global": 120, "min_partial": 12}
    }
    
    multipliers = risk_multipliers.get(risk_profile, risk_multipliers["medium"])
    
    target_profit = max(multipliers["global"] * total_investido, multipliers["min_global"])
    daily_target = max(multipliers["partial"] * total_investido, multipliers["min_partial"])
    
    return target_profit, daily_target

# ---------------- TAXA DE CORRETAGEM XP ----------------
def xp_equities_fee(volume):
    """
    Corretagem de renda variável (Equities) na XP, conforme:
      - Até US$100 -> US$1.00
      - De US$100 a US$1.000 -> US$1.50
      - De US$1.000 a US$2.000 -> US$4.30
      - Acima de US$2.000 -> US$8.60
    """
    if volume <= 100:
        return 1.0
    elif volume <= 1000:
        return 1.5
    elif volume <= 2000:
        return 4.3
    else:
        return 8.6

# ---------------- FUNÇÃO PARA VENDER (META DE LUCRO) ----------------
def find_shares_to_sell_for_target(target_lucro_liquido, profit_per_share, updated_price, max_shares):
    """
    Tenta encontrar a menor quantidade s de ações para atingir 'target_lucro_liquido'
    após a taxa XP. Ex.: (s * profit_per_share - fee) >= target_lucro_liquido.
    Se não encontrar, retorna max_shares.
    """
    if profit_per_share <= 0:
        return 0

    # Garante que max_shares é inteiro e não NaN
    if pd.isna(max_shares):
        max_shares = 0
    else:
        max_shares = int(round(max_shares))

    best_s = 0
    for s in range(1, max_shares + 1):
        volume = s * updated_price
        fee = xp_equities_fee(volume)
        net_after_fee = s * profit_per_share - fee
        if net_after_fee >= target_lucro_liquido:
            best_s = s
            break
    if best_s == 0:
        best_s = max_shares
    return best_s

# ---------------- FUNÇÃO PARA COMPRAR (CONSIDERANDO TAXA) ----------------
def find_shares_to_buy_for_amount(amount_to_spend, updated_price):
    """
    Tenta encontrar a maior quantidade s de ações que pode ser comprada
    com 'amount_to_spend', levando em conta a taxa XP.
    Loop simples: se s * updated_price + fee <= amount_to_spend,
    então s é viável. Retorna o maior s.
    """
    if pd.isna(updated_price) or updated_price <= 0:
        return 0
        
    best_s = 0
    for s in range(1, 100000):  # limite arbitrário, pode ser ajustado
        volume = s * updated_price
        fee = xp_equities_fee(volume)
        total_cost = volume + fee
        if total_cost <= amount_to_spend:
            best_s = s
        else:
            break
    return best_s

# ---------------- ANÁLISE DE SENTIMENTO ----------------
async def get_sentiment_async(session, ticker):
    """Asynchronously get sentiment from news API"""
    url = f"https://newsapi.org/v2/top-headlines?q={ticker}&language=en&apiKey={NEWSAPI_KEY}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                articles = data.get("articles", [])
                if not articles:
                    return ticker, None, ["No news found."]
                    
                scores = []
                headlines = []
                for art in articles:
                    headline = art.get("title", "")
                    headlines.append(headline)
                    score = analyzer.polarity_scores(headline)
                    scores.append(score["compound"])
                    
                if scores:
                    avg_score = sum(scores) / len(scores)
                    return ticker, avg_score, headlines[:5]
                else:
                    return ticker, None, ["No scores."]
            else:
                return ticker, None, [f"Error: {response.status}"]
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}")
        return ticker, None, [f"Error: {str(e)}"]

def get_sentiment(ticker):
    """Legacy synchronous sentiment function for backward compatibility"""
    # Check cache first
    cached_data = load_from_cache("sentiment", ticker)
    if cached_data:
        return cached_data.get('score'), cached_data.get('headlines', [])
    
    url = f"https://newsapi.org/v2/top-headlines?q={ticker}&language=en&apiKey={NEWSAPI_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return None, ["Nenhuma notícia encontrada."]
            scores = []
            headlines = []
            for art in articles:
                headline = art.get("title", "")
                headlines.append(headline)
                score = analyzer.polarity_scores(headline)
                scores.append(score["compound"])
            if scores:
                avg_score = sum(scores) / len(scores)
                
                # Save to cache for 30 minutes
                save_to_cache("sentiment", ticker, {
                    'score': avg_score,
                    'headlines': headlines[:5]
                }, 1800)
                
                return avg_score, headlines[:5]
            else:
                return None, ["Sem pontuações."]
        else:
            return None, [f"Erro: {resp.status_code}"]
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}")
        return None, [f"Erro: {str(e)}"]

async def fetch_sentiments_batch(symbols):
    """Fetch sentiment for multiple symbols concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            # Check cache first
            cached_data = load_from_cache("sentiment", symbol)
            if cached_data:
                tasks.append(asyncio.Future())
                tasks[-1].set_result((symbol, cached_data.get('score'), cached_data.get('headlines', [])))
            else:
                tasks.append(get_sentiment_async(session, symbol))
        
        results = await asyncio.gather(*tasks)
        
        # Save results to cache
        for symbol, score, headlines in results:
            if score is not None:
                save_to_cache("sentiment", symbol, {
                    'score': score,
                    'headlines': headlines[:5]
                }, 1800)  # 30 minutes cache
                
        return results

# ---------------- ASYNC PRICE FETCHING ----------------
async def fetch_price_async(session, symbol):
    """Asynchronously fetch current price for a symbol"""
    try:
        # First check cache
        cached_data = load_from_cache("price", symbol)
        if cached_data:
            return symbol, cached_data
            
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get("chart", {}).get("result", [])
                if result and result[0]:
                    meta = result[0].get("meta", {})
                    price = meta.get("regularMarketPrice")
                    if price:
                        # Cache the price for 5 minutes
                        save_to_cache("price", symbol, price, 300)
                        return symbol, price
            return symbol, None
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return symbol, None

async def fetch_prices_batch(symbols):
    """Fetch prices for multiple symbols concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            # Check cache first
            cached_data = load_from_cache("price", symbol)
            if cached_data:
                tasks.append(asyncio.Future())
                tasks[-1].set_result((symbol, cached_data))
            else:
                tasks.append(fetch_price_async(session, symbol))
        return await asyncio.gather(*tasks)

# ---------------- LER CSV ----------------
def carregar_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        log_message(f"Arquivo {PORTFOLIO_FILE} não encontrado!")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PORTFOLIO_FILE, sep=",")
    except Exception as e:
        log_message(f"Erro ao carregar CSV: {e}")
        return pd.DataFrame()

    required_cols = ['Symbol', 'Current Price', 'Purchase Price', 'Quantity']
    for col in required_cols:
        if col not in df.columns:
            log_message(f"Coluna '{col}' não encontrada no CSV.")
            return pd.DataFrame()

    # Converte Quantity para int, tratando valores NaN
    df['Quantity'] = df['Quantity'].apply(lambda x: int(round(x)) if pd.notna(x) else 0)

    log_message(f"Portfolio carregado com {df.shape[0]} entradas.")
    return df

# ---------------- CONSOLIDAR PORTFOLIO ----------------
def consolidar_portfolio(df):
    df = df.copy()
    df['TotalCost'] = df['Purchase Price'] * df['Quantity']
    consolidated = df.groupby('Symbol').agg({
        'Quantity': 'sum',
        'TotalCost': 'sum',
        'Current Price': 'last'
    }).reset_index()
    consolidated['Weighted Purchase Price'] = consolidated['TotalCost'] / consolidated['Quantity']
    consolidated = consolidated.rename(columns={
        'Quantity': 'Total Quantity',
        'Current Price': 'CSV Current Price'
    })
    return consolidated[['Symbol', 'Total Quantity', 'Weighted Purchase Price', 'CSV Current Price']]

# ---------------- ATUALIZAR PREÇOS ----------------
def atualizar_precos(df):
    """Legacy synchronous price updater for backward compatibility"""
    precos = []
    for _, row in df.iterrows():
        simbolo = row['Symbol']
        try:
            # Check cache first
            cached_price = load_from_cache("price", simbolo)
            if cached_price:
                precos.append(cached_price)
                continue
                
            ticker = yf.Ticker(simbolo)
            info = ticker.info
            preco = info.get('regularMarketPrice') or info.get('currentPrice') or row['CSV Current Price']
            precos.append(preco)
            
            # Save to cache for 5 minutes
            save_to_cache("price", simbolo, preco, 300)
            
            log_message(f"{simbolo}: Atualizado: {preco} (pre: {info.get('preMarketPrice')}, pos: {info.get('postMarketPrice')}, target: {info.get('targetMeanPrice')})")
        except Exception as e:
            log_message(f"Erro ao atualizar {simbolo}: {e}")
            precos.append(row['CSV Current Price'])
    df['Atualizado Current Price'] = precos
    return df

async def atualizar_precos_async(df):
    """Asynchronous price updater for batch processing"""
    symbols = df['Symbol'].tolist()
    price_results = await fetch_prices_batch(symbols)
    
    # Create price dictionary
    price_dict = {symbol: price for symbol, price in price_results if price is not None}
    
    # Update dataframe with new prices
    updated_prices = []
    for _, row in df.iterrows():
        symbol = row['Symbol']
        updated_price = price_dict.get(symbol, row['CSV Current Price'])
        updated_prices.append(updated_price)
        
    df['Atualizado Current Price'] = updated_prices
    return df

# ---------------- CALCULAR INDICADORES ----------------
def calcular_indicadores(simbolo):
    """Calculate technical indicators for a symbol"""
    # Check cache first
    cached_data = load_from_cache("indicators", simbolo)
    if cached_data:
        return cached_data
    
    try:
        data = yf.download(simbolo, period="3mo", interval="1d", auto_adjust=True, progress=False)
    except Exception as e:
        return f"Erro ao baixar dados para {simbolo}: {e}"
    if data.empty:
        return f"Não foi possível obter dados para {simbolo}."
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        data.columns = [str(col) for col in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]

    close_col = None
    for col in data.columns:
        if col.lower() == 'close':
            close_col = col
            break
    if close_col is None:
        for col in data.columns:
            if col.lower() == 'adj close':
                close_col = col
                break
    if close_col is None:
        return f"Erro: Nenhuma coluna de fechamento encontrada para {simbolo}."

    # Basic indicators
    data['MA9'] = data[close_col].rolling(window=9, min_periods=9).mean()
    data['MA21'] = data[close_col].rolling(window=21, min_periods=21).mean()
    
    # RSI calculation
    delta = data[close_col].diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['MA20'] = data[close_col].rolling(window=20, min_periods=20).mean()
    data['std'] = data[close_col].rolling(window=20, min_periods=20).std()
    data['UpperBand'] = data['MA20'] + 2 * data['std']
    data['LowerBand'] = data['MA20'] - 2 * data['std']
    
    # MACD
    ema12 = data[close_col].ewm(span=12, adjust=False).mean()
    ema26 = data[close_col].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    latest = data.iloc[-1]
    try:
        current_price = float(latest[close_col].item()) if pd.notna(latest[close_col]) else None
        ma9 = float(latest['MA9'].item()) if pd.notna(latest['MA9']) else None
        ma21 = float(latest['MA21'].item()) if pd.notna(latest['MA21']) else None
        rsi = float(latest['RSI'].item()) if pd.notna(latest['RSI']) else None
        ma20 = float(latest['MA20'].item()) if pd.notna(latest['MA20']) else None
        upper_band = float(latest['UpperBand'].item()) if pd.notna(latest['UpperBand']) else None
        lower_band = float(latest['LowerBand'].item()) if pd.notna(latest['LowerBand']) else None
        macd = float(latest['MACD'].item()) if pd.notna(latest['MACD']) else None
        macd_signal = float(latest['MACD_Signal'].item()) if pd.notna(latest['MACD_Signal']) else None
        macd_hist = float(latest['MACD_Hist'].item()) if pd.notna(latest['MACD_Hist']) else None
        
        # Calculate MA crossover signal
        ma_crossover = 0
        if len(data) >= 2:
            if (data['MA9'].iloc[-2] <= data['MA21'].iloc[-2] and ma9 > ma21):
                ma_crossover = 1  # Bullish crossover
            elif (data['MA9'].iloc[-2] >= data['MA21'].iloc[-2] and ma9 < ma21):
                ma_crossover = -1  # Bearish crossover
                
        # Calculate MACD crossover signal
        macd_crossover = 0
        if len(data) >= 2:
            if (data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2] and macd > macd_signal):
                macd_crossover = 1  # Bullish crossover
            elif (data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2] and macd < macd_signal):
                macd_crossover = -1  # Bearish crossover
    except Exception as e:
        return f"Erro ao converter valores para {simbolo}: {e}"

    result = {
        "Preço Atual Yahoo": current_price,
        "MA9": ma9,
        "MA21": ma21,
        "RSI": rsi,
        "MA20": ma20,
        "UpperBand": upper_band,
        "LowerBand": lower_band,
        "MACD": macd,
        "MACD_Signal": macd_signal,
        "MACD_Hist": macd_hist,
        "Signals": {
            "ma_trend": "bullish" if ma9 > ma21 else "bearish",
            "ma_crossover": ma_crossover,
            "rsi_zone": "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral"),
            "bollinger_position": "lower" if current_price <= lower_band else 
                                ("upper" if current_price >= upper_band else "middle"),
            "macd_signal": "bullish" if macd > macd_signal else "bearish",
            "macd_crossover": macd_crossover
        }
    }
    
    # Add volume analysis if available
    if 'Volume' in data.columns:
        try:
            volume = float(latest['Volume'].item()) if pd.notna(latest['Volume']) else None
            volume_ma20 = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = volume / volume_ma20 if volume_ma20 > 0 else None
            
            result["Volume"] = volume
            result["Volume_MA20"] = volume_ma20
            result["Volume_Ratio"] = volume_ratio
            result["Signals"]["volume_trend"] = "high" if volume_ratio > 1.5 else 
                                             ("low" if volume_ratio < 0.5 else "normal")
        except Exception as e:
            logger.error(f"Error calculating volume indicators for {simbolo}: {e}")
    
    # Cache the result for 30 minutes
    save_to_cache("indicators", simbolo, result, 1800)
    
    return result

def generate_weighted_signal(indicadores, sentimento, risk_profile="medium"):
    """
    Generate a weighted buy/sell signal based on technical indicators and sentiment
    Returns a score between -100 (strong sell) and 100 (strong buy)
    """
    if isinstance(indicadores, str) or indicadores is None:
        return 0, "insufficient data"
        
    weights = signal_weights.get(risk_profile, signal_weights["medium"])
    
    # Initialize score components
    score_components = {}
    
    # RSI component (-100 to 100)
    rsi = indicadores.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30:
            # Oversold - bullish signal
            rsi_score = 100 - (rsi * 2)  # 30 -> 40, 20 -> 60, 10 -> 80
        elif rsi > 70:
            # Overbought - bearish signal
            rsi_score = (100 - rsi) * 2 - 100  # 70 -> -40, 80 -> -60, 90 -> -80
        else:
            # Neutral zone
            rsi_score = (50 - rsi) * (100/20)  # 50 -> 0, 40 -> 50, 60 -> -50
        score_components["RSI"] = rsi_score
    else:
        score_components["RSI"] = 0
    
    # MA crossover component (-100 to 100)
    signals = indicadores.get("Signals", {})
    ma_crossover = signals.get("ma_crossover", 0)
    ma_trend = signals.get("ma_trend")
    
    if ma_crossover != 0:
        # Recent crossover - strong signal
        ma_score = ma_crossover * 100
    elif ma_trend == "bullish":
        # Bullish trend
        ma_score = 50
    elif ma_trend == "bearish":
        # Bearish trend
        ma_score = -50
    else:
        ma_score = 0
        
    score_components["MA_crossover"] = ma_score
    
    # Bollinger Bands component (-100 to 100)
    bb_position = signals.get("bollinger_position")
    if bb_position == "lower":
        bb_score = 75  # Near lower band - bullish
    elif bb_position == "upper":
        bb_score = -75  # Near upper band - bearish
    else:
        bb_score = 0  # Middle - neutral
    score_components["bollinger"] = bb_score
    
    # MACD component (-100 to 100)
    macd_crossover = signals.get("macd_crossover", 0)
    macd_signal = signals.get("macd_signal")
    
    if macd_crossover != 0:
        # Recent crossover - strong signal
        macd_score = macd_crossover * 100
    elif macd_signal == "bullish":
        macd_score = 50
    elif macd_signal == "bearish":
        macd_score = -50
    else:
        macd_score = 0
        
    score_components["macd"] = macd_score
    
    # Volume component (-100 to 100)
    volume_trend = signals.get("volume_trend")
    if volume_trend == "high":
        # High volume confirms trend
        if ma_trend == "bullish":
            volume_score = 75
        elif ma_trend == "bearish":
            volume_score = -75
        else:
            volume_score = 0
    elif volume_trend == "low":
        # Low volume might indicate weak trend
        volume_score = 0
    else:
        volume_score = 25 if ma_trend == "bullish" else (-25 if ma_trend == "bearish" else 0)
        
    score_components["volume"] = volume_score
    
    # Sentiment component (-100 to 100)
    if sentimento is not None:
        sentiment_score = sentimento * 100  # -1.0 to 1.0 -> -100 to 100
    else:
        sentiment_score = 0
        
    score_components["sentiment"] = sentiment_score
    
    # Sector momentum would be added here
    score_components["sector_momentum"] = 0
    
    # Calculate weighted score
    final_score = 0
    for component, score in score_components.items():
        weight = weights.get(component, 0)
        final_score += score * weight
        
    # Determine signal strength and direction
    if final_score >= 70:
        signal_type = "strong buy"
    elif final_score >= 30:
        signal_type = "buy"
    elif final_score > -30:
        signal_type = "neutral"
    elif final_score > -70:
        signal_type = "sell"
    else:
        signal_type = "strong sell"
        
    return final_score, signal_type

# ---------------- ANÁLISE DO PORTFOLIO (COM TAXAS NA COMPRA E VENDA) ----------------
def analisar_portfolio(df, risk_profile="medium"):
    resultados = []
    for _, row in df.iterrows():
        simbolo = row['Symbol']
        preco_medio = row['Weighted Purchase Price']
        quantidade = row['Total Quantity']
        atualizado_price = row['Atualizado Current Price']

        pct_variacao = ((atualizado_price - preco_medio) / preco_medio) * 100
        lucro_em_dolares = (atualizado_price - preco_medio) * quantidade

        indicadores = calcular_indicadores(simbolo)
        indicadores_validos = not isinstance(indicadores, str) and indicadores is not None

        sentimento, headlines = get_sentiment(simbolo)
        sentimento_str = f"{sentimento:.2f}" if sentimento is not None else "N/A"

        # Generate weighted signal
        signal_score, signal_type = generate_weighted_signal(indicadores, sentimento, risk_profile)

        # Limiar dinâmico
        total_investido_ativo = preco_medio * quantidade
        target_profit, daily_target = get_dynamic_thresholds(total_investido_ativo, risk_profile)

        # Decision logic based on signal score, price movement, and profit targets
        if signal_score <= -70 or pct_variacao <= -7:
            acao = "VENDER"
            recomendacao = f"Vender todas as {quantidade} ações (sinal forte de venda ou limite de perda atingido)."
        elif lucro_em_dolares >= target_profit and signal_score < 30:
            # Met profit target and signal isn't bullish
            acao = "REALIZAR LUCRO"
            profit_per_share = atualizado_price - preco_medio
            shares_to_sell = find_shares_to_sell_for_target(target_profit, profit_per_share, atualizado_price, quantidade)
            volume = shares_to_sell * atualizado_price
            fee = xp_equities_fee(volume)
            net_after_fee = shares_to_sell * profit_per_share - fee
            recomendacao = (f"Vender {shares_to_sell} ações para lucro líquido ~US$ {net_after_fee:.2f} "
                           f"(meta ~{target_profit:.2f}).")
        elif lucro_em_dolares >= daily_target and signal_score < 0:
            # Met partial profit target and signal is bearish
            acao = "REALIZAR LUCRO PARCIAL"
            profit_per_share = atualizado_price - preco_medio
            shares_to_sell = find_shares_to_sell_for_target(daily_target, profit_per_share, atualizado_price, quantidade)
            volume = shares_to_sell * atualizado_price
            fee = xp_equities_fee(volume)
            net_after_fee = shares_to_sell * profit_per_share - fee
            recomendacao = (f"Vender {shares_to_sell} ações para lucro líquido ~US$ {net_after_fee:.2f} "
                           f"(meta parcial ~{daily_target:.2f}).")
        elif signal_score >= 70 and ACCOUNT_BALANCE > 0:
            # Strong buy signal and we have available funds
            acao = "COMPRAR MAIS"
            # Calculate position size based on volatility if available
            if indicadores_validos and "ATR" in indicadores:
                atr = indicadores["ATR"]
                risk_per_trade = 0.01 * ACCOUNT_BALANCE  # 1% risk
                shares_per_stop = risk_per_trade / (2 * atr)  # 2 ATR stop
                suggested_shares = max(1, int(shares_per_stop))
                valor_a_investir = suggested_shares * atualizado_price
            else:
                valor_a_investir = min(0.1 * ACCOUNT_BALANCE, 1000)  # 10% of balance or $1000 max
                suggested_shares = find_shares_to_buy_for_amount(valor_a_investir, atualizado_price)
                
            recomendacao = f"Comprar {suggested_shares} ações (sinal forte de compra)."
        elif signal_score >= 30 and signal_score < 70 and ACCOUNT_BALANCE > 0:
            # Moderate buy signal
            acao = "CONSIDERAR COMPRA"
            valor_a_investir = min(0.05 * ACCOUNT_BALANCE, 500)  # 5% of balance or $500 max
            suggested_shares = find_shares_to_buy_for_amount(valor_a_investir, atualizado_price)
            recomendacao = f"Considerar comprar {suggested_shares} ações (sinal moderado de compra)."
        else:
            acao = "MANTER"
            recomendacao = "Manter a posição."

        # Add sentiment context to the recommendation
        if sentimento is not None:
            if sentimento < -0.3:
                recomendacao += " [Notícias negativas predominantes]"
            elif sentimento > 0.3:
                recomendacao += " [Notícias positivas predominantes]"

        resultados.append({
            "Symbol": simbolo,
            "Weighted Purchase Price": preco_medio,
            "Total Quantity": quantidade,
            "Atualizado Current Price": atualizado_price,
            "P&L Variation": f"{pct_variacao:.2f}%",
            "Profit": f"${lucro_em_dolares:.2f}",
            "Signal Score": f"{signal_score:.1f}",
            "Signal Type": signal_type,
            "Sentimento": sentimento_str,
            "Ação Recomendada": acao,
            "Recomendação": recomendacao
        })

    return resultados

# ---------------- ENHANCED OPPORTUNITY ENGINE ----------------
def get_sector_tickers(sector):
    """Get top tickers for a sector from our predefined lists"""
    return top_tickers_by_sector.get(sector, [])

def get_etfs_for_sector(sector):
    """Get ETFs that represent a sector"""
    return sector_etfs.get(sector, [])

async def scan_for_opportunities(criteria=None, max_results=20):
    """
    Scan for trading opportunities based on criteria
    
    Parameters:
    criteria (dict): filtering criteria like sector, price range, etc.
    max_results (int): maximum number of opportunities to return
    
    Returns:
    list: opportunities that match criteria
    """
    if criteria is None:
        criteria = {}
    
    # Generate cache key from criteria
    cache_key = f"scan_{json.dumps(criteria, sort_keys=True)}_{max_results}"
    cached_results = load_from_cache("opportunities", cache_key)
    if cached_results:
        return cached_results
    
    # Get tickers to scan
    tickers_to_scan = []
    
    # If sector specified, get tickers for that sector
    if 'sector' in criteria and criteria['sector'] in top_tickers_by_sector:
        tickers_to_scan.extend(get_sector_tickers(criteria['sector']))
        tickers_to_scan.extend(get_etfs_for_sector(criteria['sector']))
    else:
        # Otherwise scan top tickers from multiple sectors
        for sector, tickers in top_tickers_by_sector.items():
            tickers_to_scan.extend(tickers[:5])  # Top 5 per sector
            
        # Add some popular ETFs
        for sector, etfs in sector_etfs.items():
            tickers_to_scan.extend(etfs[:2])  # Top 2 ETFs per sector
    
    # Add custom tickers if provided
    if 'tickers' in criteria and isinstance(criteria['tickers'], list):
        tickers_to_scan.extend(criteria['tickers'])
    
    # Remove duplicates
    tickers_to_scan = list(set(tickers_to_scan))
    
    # Fetch technical indicators and sentiments in parallel
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Process in batches to avoid overwhelming APIs
    batch_size = 10
    all_opportunities = []
    
    for i in range(0, len(tickers_to_scan), batch_size):
        batch = tickers_to_scan[i:i+batch_size]
        
        # Fetch sentiment for batch
        sentiment_results = await fetch_sentiments_batch(batch)
        sentiment_dict = {symbol: (score, headlines) for symbol, score, headlines in sentiment_results}
        
        # Analyze technical indicators for each ticker
        opportunities = []
        for symbol in batch:
            try:
                # Get indicators
                indicators = calcular_indicadores(symbol)
                if isinstance(indicators, str):  # Error message
                    continue
                
                # Get sentiment
                sentiment, headlines = sentiment_dict.get(symbol, (None, []))
                
                # Generate signal
                signal_score, signal_type = generate_weighted_signal(indicators, sentiment, criteria.get('risk_profile', 'medium'))
                
                # Skip if doesn't meet strength threshold
                min_score = criteria.get('min_signal_strength', 30)
                max_score = criteria.get('max_signal_strength', 100)
                if signal_type == "neutral" or not (min_score <= abs(signal_score) <= max_score):
                    continue
                
                # Determine opportunity type
                if signal_score >= 70:
                    opp_type = "Forte sinal de compra"
                    sugestao = "Considerar compra (momentum)"
                elif signal_score >= 30:
                    opp_type = "Sinal moderado de compra"
                    sugestao = "Monitorar para entrada"
                elif signal_score <= -70:
                    opp_type = "Forte sinal de venda"
                    sugestao = "Considerar venda (fraqueza)"
                elif signal_score <= -30:
                    opp_type = "Sinal moderado de venda"
                    sugestao = "Monitorar para saída"
                else:
                    continue  # Skip neutral signals
                
                # Get current price
                price = indicators.get("Preço Atual Yahoo")
                if price is None:
                    continue
                
                # Apply price filter criteria
                min_price = criteria.get('min_price', 0)
                max_price = criteria.get('max_price', float('inf'))
                if not (min_price <= price <= max_price):
                    continue
                
                # Check if RSI indicates oversold/overbought
                rsi = indicators.get("RSI")
                rsi_note = ""
                if rsi is not None:
                    if rsi <= 30:
                        rsi_note = "RSI indica sobrevenda"
                    elif rsi >= 70:
                        rsi_note = "RSI indica sobrecompra"
                
                # Check Bollinger position
                bb_position = indicators.get("Signals", {}).get("bollinger_position")
                bb_note = ""
                if bb_position == "lower":
                    bb_note = "Próximo à banda inferior de Bollinger"
                elif bb_position == "upper":
                    bb_note = "Próximo à banda superior de Bollinger"
                
                # Adjust suggestion based on technical factors
                if signal_score > 0 and bb_position == "lower" and rsi is not None and rsi < 40:
                    sugestao = "Potencial de reversão para cima (pullback)"
                elif signal_score < 0 and bb_position == "upper" and rsi is not None and rsi > 60:
                    sugestao = "Potencial de reversão para baixo (topo)"
                
                # Add sentiment context
                sentiment_note = ""
                if sentiment is not None:
                    if sentiment < -0.3:
                        sentiment_note = "Notícias negativas predominantes"
                        if signal_score > 0:
                            sugestao += " (atenção: sentimento negativo nas notícias)"
                    elif sentiment > 0.3:
                        sentiment_note = "Notícias positivas predominantes"
                        if signal_score < 0:
                            sugestao += " (atenção: sentimento positivo nas notícias)"
                
                # Create opportunity object
                opportunity = {
                    "Symbol": symbol,
                    "Price": price,
                    "Signal_Score": signal_score,
                    "Signal_Type": signal_type,
                    "Opportunity": opp_type,
                    "RSI": rsi,
                    "RSI_Note": rsi_note,
                    "Bollinger_Note": bb_note,
                    "MA9": indicators.get("MA9"),
                    "MA21": indicators.get("MA21"),
                    "Sentiment": sentiment,
                    "Sentiment_Note": sentiment_note,
                    "Headlines": headlines[:3] if headlines else [],
                    "Suggestion": sugestao
                }
                
                # Add volume info if available
                if "Volume" in indicators and "Volume_Ratio" in indicators:
                    opportunity["Volume"] = indicators["Volume"]
                    opportunity["Volume_Ratio"] = indicators["Volume_Ratio"]
                    if indicators["Volume_Ratio"] > 1.5:
                        opportunity["Volume_Note"] = "Volume acima da média (confirma tendência)"
                        
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.error(f"Error analyzing opportunity for {symbol}: {e}")
        
        all_opportunities.extend(opportunities)
        
        # If we already have enough opportunities, stop processing
        if len(all_opportunities) >= max_results:
            break
    
    # Sort opportunities by signal strength (absolute value)
    all_opportunities.sort(key=lambda x: abs(x["Signal_Score"]), reverse=True)
    
    # Limit to max_results
    results = all_opportunities[:max_results]
    
    # Cache results for 1 hour
    save_to_cache("opportunities", cache_key, results, 3600)
    
    return results

def analyze_specific_ticker(ticker, risk_profile="medium"):
    """
    Perform detailed analysis on a specific ticker
    
    Parameters:
    ticker (str): the ticker symbol to analyze
    risk_profile (str): risk profile to use for analysis
    
    Returns:
    dict: detailed analysis of the ticker
    """
    # Check cache first
    cache_key = f"analyze_{ticker}_{risk_profile}"
    cached_analysis = load_from_cache("ticker_analysis", cache_key)
    if cached_analysis:
        return cached_analysis
    
    try:
        # Get technical indicators
        indicators = calcular_indicadores(ticker)
        if isinstance(indicators, str):  # Error message
            return {"error": indicators}
            
        # Get sentiment
        sentiment, headlines = get_sentiment(ticker)
        
        # Get additional info from yfinance
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            # Extract key info
            company_name = info.get('shortName') or info.get('longName') or ticker
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None
            target_price = info.get('targetMeanPrice')
            
            company_info = {
                "Name": company_name,
                "Sector": sector,
                "Industry": industry,
                "Market_Cap": market_cap,
                "PE_Ratio": pe_ratio,
                "Dividend_Yield": dividend_yield,
                "Target_Price": target_price
            }
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {e}")
            company_info = {"Name": ticker}
        
        # Generate signal
        signal_score, signal_type = generate_weighted_signal(indicators, sentiment, risk_profile)
        
        # Create analysis object
        analysis = {
            "Symbol": ticker,
            "Company_Info": company_info,
            "Current_Price": indicators.get("Preço Atual Yahoo"),
            "Technical_Indicators": {
                "RSI": indicators.get("RSI"),
                "MA9": indicators.get("MA9"),
                "MA21": indicators.get("MA21"),
                "MA20": indicators.get("MA20"),
                "Upper_Bollinger": indicators.get("UpperBand"),
                "Lower_Bollinger": indicators.get("LowerBand"),
                "MACD": indicators.get("MACD"),
                "MACD_Signal": indicators.get("MACD_Signal"),
                "MACD_Hist": indicators.get("MACD_Hist")
            },
            "Sentiment": {
                "Score": sentiment,
                "Headlines": headlines
            },
            "Signal": {
                "Score": signal_score,
                "Type": signal_type
            }
        }
        
        # Add volume info if available
        if "Volume" in indicators and "Volume_Ratio" in indicators:
            analysis["Volume"] = {
                "Current": indicators["Volume"],
                "MA20": indicators["Volume_MA20"],
                "Ratio": indicators["Volume_Ratio"]
            }
        
        # Generate position recommendation
        price = indicators.get("Preço Atual Yahoo")
        if price and ACCOUNT_BALANCE > 0:
            # Consider position sizing based on account size and risk
            risk_percentages = {"low": 0.02, "medium": 0.05, "high": 0.1}
            max_position = ACCOUNT_BALANCE * risk_percentages.get(risk_profile, 0.05)
            
            if signal_score >= 70:  # Strong buy
                suggested_shares = find_shares_to_buy_for_amount(max_position, price)
                position_rec = {
                    "Recommendation": "Comprar",
                    "Strength": "Forte",
                    "Suggested_Shares": suggested_shares,
                    "Approximate_Value": suggested_shares * price,
                    "Note": "Sinal técnico forte de compra"
                }
            elif signal_score >= 30:  # Moderate buy
                half_position = max_position / 2
                suggested_shares = find_shares_to_buy_for_amount(half_position, price)
                position_rec = {
                    "Recommendation": "Comprar",
                    "Strength": "Moderada",
                    "Suggested_Shares": suggested_shares,
                    "Approximate_Value": suggested_shares * price,
                    "Note": "Sinal técnico moderado de compra"
                }
            elif signal_score <= -70:  # Strong sell
                position_rec = {
                    "Recommendation": "Vender/Evitar",
                    "Strength": "Forte",
                    "Note": "Sinal técnico forte de venda"
                }
            elif signal_score <= -30:  # Moderate sell
                position_rec = {
                    "Recommendation": "Vender/Evitar",
                    "Strength": "Moderada",
                    "Note": "Sinal técnico moderado de venda"
                }
            else:  # Neutral
                position_rec = {
                    "Recommendation": "Neutro",
                    "Note": "Sem sinal claro. Sugerimos aguardar melhores condições."
                }
                
            # Add sentiment context
            if sentiment is not None:
                if sentiment < -0.3 and signal_score > 0:
                    position_rec["Note"] += " (atenção: sentimento negativo nas notícias)"
                elif sentiment > 0.3 and signal_score < 0:
                    position_rec["Note"] += " (atenção: sentimento positivo nas notícias)"
                    
            analysis["Position_Recommendation"] = position_rec
        
        # Cache result for 30 minutes
        save_to_cache("ticker_analysis", cache_key, analysis, 1800)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing ticker {ticker}: {e}")
        return {"error": f"Error analyzing ticker: {str(e)}"}

# ---------------- SIMULAÇÃO DE TRADES ----------------
def simulate_trades(analise, total_invested_initial):
    """Simple trade simulation without considering fees"""
    simulated = total_invested_initial
    for item in analise:
        rec = item['Recomendação']
        if rec.startswith("Vender"):
            try:
                parts = rec.split(" ")
                shares = float(parts[1])
                cost = item['Weighted Purchase Price'] * shares
                simulated -= cost
            except Exception:
                pass
        elif rec.startswith("Comprar"):
            try:
                parts = rec.split(" ")
                shares = float(parts[1])
                cost = item['Atualizado Current Price'] * shares
                simulated += cost
            except Exception:
                pass
    return simulated

# ---------------- ROTA PRINCIPAL ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    global ACCOUNT_BALANCE, INVESTED_AMOUNT, RISK_TOLERANCE
    if request.method == "POST":
        try:
            ACCOUNT_BALANCE = float(request.form.get("account_balance", ACCOUNT_BALANCE))
            INVESTED_AMOUNT = float(request.form.get("invested_amount", INVESTED_AMOUNT))
            RISK_TOLERANCE = request.form.get("risk_tolerance", RISK_TOLERANCE)
            log_message(f"Valores atualizados: ACCOUNT_BALANCE = {ACCOUNT_BALANCE}, INVESTED_AMOUNT = {INVESTED_AMOUNT}, RISK_TOLERANCE = {RISK_TOLERANCE}")
        except Exception as e:
            log_message(f"Erro ao atualizar valores do formulário: {e}")

    df_raw = carregar_portfolio()
    if df_raw.empty:
        tabela_final = "<p>Planilha vazia ou não encontrada.</p>"
        total_invested_initial = 0
        total_current_value = 0
        profit_total = 0
        simulated_invested = 0
    else:
        df_consolidado = consolidar_portfolio(df_raw)
        
        # Use async price updates if running in Python 3.7+
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df_consolidado = loop.run_until_complete(atualizar_precos_async(df_consolidado))
        except Exception as e:
            logger.error(f"Error using async price updates: {e}. Falling back to sync method.")
            df_consolidado = atualizar_precos(df_consolidado)
            
        analise = analisar_portfolio(df_consolidado, RISK_TOLERANCE)

        tabela_final = """
        <table class="table table-bordered">
            <tr>
                <th>Symbol</th>
                <th>Weighted Purchase Price</th>
                <th>Total Quantity</th>
                <th>Atualizado Current Price</th>
                <th>P&L Variation</th>
                <th>Profit</th>
                <th>Signal</th>
                <th>Sentimento</th>
                <th>Ação Recomendada</th>
                <th>Recomendação</th>
            </tr>
        """
        for item in analise:
            signal_color = "text-success" if float(item['Signal Score']) > 30 else ("text-danger" if float(item['Signal Score']) < -30 else "")
            tabela_final += f"""
            <tr>
                <td>{item['Symbol']}</td>
                <td>{item['Weighted Purchase Price']:.2f}</td>
                <td>{item['Total Quantity']}</td>
                <td>{item['Atualizado Current Price']:.2f}</td>
                <td>{item['P&L Variation']}</td>
                <td>{item['Profit']}</td>
                <td class="{signal_color}">{item['Signal Score']} ({item['Signal Type']})</td>
                <td>{item['Sentimento']}</td>
                <td>{item['Ação Recomendada']}</td>
                <td>{item['Recomendação']}</td>
            </tr>
            """
        tabela_final += "</table>"

        total_invested_initial = (df_consolidado["Weighted Purchase Price"] * df_consolidado["Total Quantity"]).sum()
        total_current_value = (df_consolidado["Atualizado Current Price"] * df_consolidado["Total Quantity"]).sum()
        profit_total = total_current_value - total_invested_initial
        simulated_invested = simulate_trades(analise, total_invested_initial)

    refresh_meta = '<meta http-equiv="refresh" content="60">'  # Less frequent updates to reduce API calls

    return render_template_string(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise Dinâmica de Portfolio v4</title>
            {refresh_meta}
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{ margin-top: 20px; }}
                table {{ margin-bottom: 30px; }}
                .card {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Análise Dinâmica de Portfolio v4</h1>
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                <h4>Configurações da Conta</h4>
                            </div>
                            <div class="card-body">
                                <form method="post" action="/">
                                    <div class="form-group">
                                        <label for="account_balance">Saldo Disponível (Account Balance):</label>
                                        <input type="number" step="0.01" name="account_balance" id="account_balance" class="form-control" value="{ACCOUNT_BALANCE}">
                                    </div>
                                    <div class="form-group">
                                        <label for="invested_amount">Valor Investido (Invested Amount):</label>
                                        <input type="number" step="0.01" name="invested_amount" id="invested_amount" class="form-control" value="{INVESTED_AMOUNT}">
                                    </div>
                                    <div class="form-group">
                                        <label for="risk_tolerance">Perfil de Risco:</label>
                                        <select name="risk_tolerance" id="risk_tolerance" class="form-control">
                                            <option value="low" {"selected" if RISK_TOLERANCE == "low" else ""}>Conservador (Low)</option>
                                            <option value="medium" {"selected" if RISK_TOLERANCE == "medium" else ""}>Moderado (Medium)</option>
                                            <option value="high" {"selected" if RISK_TOLERANCE == "high" else ""}>Agressivo (High)</option>
                                        </select>
                                    </div>
                                    <button type="submit" class="btn btn-secondary">Atualizar Valores</button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                <h4>Resumo do Portfolio</h4>
                            </div>
                            <div class="card-body">
                                <p><strong>Total Investido Inicial:</strong> US$ {total_invested_initial:.2f}</p>
                                <p><strong>Valor Atual do Portfolio:</strong> US$ {total_current_value:.2f}</p>
                                <p><strong>Lucro/Prejuízo Total:</strong> <span class="{'text-success' if profit_total >= 0 else 'text-danger'}">US$ {profit_total:.2f}</span></p>
                                <p><strong>Valor Investido Ideal (simulado):</strong> US$ {simulated_invested:.2f}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h4>Análise do Portfolio</h4>
                    </div>
                    <div class="card-body">
                        {tabela_final}
                    </div>
                </div>
                
                <div class="row mb-4">
                    <div class="col">
                        <a href="/" class="btn btn-primary">Atualizar Página</a>
                        <a href="/oportunidades" class="btn btn-success ml-2">Analisar Oportunidades Setoriais</a>
                        <a href="/pesquisar" class="btn btn-info ml-2">Pesquisar Ativo Específico</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    )

# ---------------- ROTA OPORTUNIDADES ----------------
@app.route("/oportunidades", methods=["GET"])
def oportunidades():
    tickers_input = request.args.get("tickers", "").strip()
    sector = request.args.get("sector", "").strip()
    min_price = float(request.args.get("min_price", 5))
    max_price = float(request.args.get("max_price", 500))
    signal_type = request.args.get("signal_type", "all")  # all, buy, sell
    
    # Build criteria dict
    criteria = {
        "min_price": min_price,
        "max_price": max_price,
        "risk_profile": RISK_TOLERANCE
    }
    
    if sector:
        criteria["sector"] = sector
    
    if signal_type == "buy":
        criteria["min_signal_strength"] = 30
        criteria["max_signal_strength"] = 100
    elif signal_type == "sell":
        criteria["min_signal_strength"] = -100
        criteria["max_signal_strength"] = -30
    
    # Add custom tickers if provided
    if tickers_input:
        criteria["tickers"] = [t.strip().strip('"').strip("'") for t in tickers_input.split(",") if t.strip()]
    
    # Get opportunities
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    oportunidades_list = loop.run_until_complete(scan_for_opportunities(criteria))
    
    if not oportunidades_list:
        ops_html = "<p>Nenhuma oportunidade encontrada.</p>"
    else:
        ops_html = """
        <table class="table table-bordered">
            <tr>
                <th>Symbol</th>
                <th>Preço Atual</th>
                <th>Sinal (Score)</th>
                <th>RSI</th>
                <th>Padrão Técnico</th>
                <th>Sentimento</th>
                <th>Sugestão</th>
            </tr>
        """
        for op in oportunidades_list:
            signal_color = "text-success" if op['Signal_Score'] > 30 else ("text-danger" if op['Signal_Score'] < -30 else "")
            technical_notes = []
            if op.get('RSI_Note'):
                technical_notes.append(op['RSI_Note'])
            if op.get('Bollinger_Note'):
                technical_notes.append(op['Bollinger_Note'])
            if op.get('Volume_Note'):
                technical_notes.append(op['Volume_Note'])
                
            technical_pattern = ", ".join(technical_notes) or "Nenhum padrão relevante"
            
            sentiment_str = f"{op['Sentiment']:.2f}" if op.get('Sentiment') is not None else "N/A"
            sentiment_color = ""
            if op.get('Sentiment') is not None:
                sentiment_color = "text-success" if op['Sentiment'] > 0.3 else ("text-danger" if op['Sentiment'] < -0.3 else "")
                
            headlines_html = ""
            if op.get('Headlines'):
                headlines_html = "<ul class='small mt-1 mb-0'>"
                for headline in op['Headlines']:
                    headlines_html += f"<li>{headline}</li>"
                headlines_html += "</ul>"
                
            ops_html += f"""
            <tr>
                <td><a href="/pesquisar?ticker={op['Symbol']}" target="_blank">{op['Symbol']}</a></td>
                <td>${op['Price']:.2f}</td>
                <td class="{signal_color}">{op['Signal_Type']} ({op['Signal_Score']:.1f})</td>
                <td>{op['RSI']:.1f if op['RSI'] is not None else 'N/A'}</td>
                <td>{technical_pattern}</td>
                <td class="{sentiment_color}">{sentiment_str}{headlines_html}</td>
                <td>{op['Suggestion']}</td>
            </tr>
            """
        ops_html += "</table>"

    # Get available sectors for dropdown
    sectors_html = ""
    for s in sorted(top_tickers_by_sector.keys()):
        selected = "selected" if s == sector else ""
        sectors_html += f'<option value="{s}" {selected}>{s}</option>'
        
    return render_template_string(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Oportunidades de Trading - Swing Trade v4</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{ margin-top: 20px; }}
                table {{ margin-bottom: 30px; }}
                .card {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Oportunidades de Trading (Swing Trade v4)</h1>
                
                <div class="card">
                    <div class="card-header">
                        <h4>Critérios de Busca</h4>
                    </div>
                    <div class="card-body">
                        <form method="get" action="/oportunidades" class="mb-4">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label for="tickers">Tickers Específicos (separados por vírgula):</label>
                                        <input type="text" name="tickers" id="tickers" class="form-control" value="{tickers_input}">
                                        <small class="form-text text-muted">Opcional. Se vazio, serão usados ativos populares.</small>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label for="sector">Setor:</label>
                                        <select name="sector" id="sector" class="form-control">
                                            <option value="">Todos os Setores</option>
                                            {sectors_html}
                                        </select>
                                    </div>
                                </div>
                                
                                <div class="col-md-6">
                                    <div class="form-row">
                                        <div class="form-group col-md-6">
                                            <label for="min_price">Preço Mínimo ($):</label>
                                            <input type="number" step="0.01" name="min_price" id="min_price" class="form-control" value="{min_price}">
                                        </div>
                                        <div class="form-group col-md-6">
                                            <label for="max_price">Preço Máximo ($):</label>
                                            <input type="number" step="0.01" name="max_price" id="max_price" class="form-control" value="{max_price}">
                                        </div>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label for="signal_type">Tipo de Sinal:</label>
                                        <select name="signal_type" id="signal_type" class="form-control">
                                            <option value="all" {"selected" if signal_type == "all" else ""}>Todos os Sinais</option>
                                            <option value="buy" {"selected" if signal_type == "buy" else ""}>Apenas Compra</option>
                                            <option value="sell" {"selected" if signal_type == "sell" else ""}>Apenas Venda</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <button type="submit" class="btn btn-primary">Buscar Oportunidades</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h4>Resultados</h4>
                    </div>
                    <div class="card-body">
                        {ops_html}
                    </div>
                </div>
                
                <a href="/" class="btn btn-secondary mt-3">Voltar</a>
            </div>
        </body>
        </html>
        """
    )

# ---------------- ROTA PESQUISA DE ATIVO ----------------
@app.route("/pesquisar", methods=["GET"])
def pesquisar_ativo():
    ticker = request.args.get("ticker", "").strip().upper()
    analysis_html = ""
    
    if ticker:
        analysis = analyze_specific_ticker(ticker, RISK_TOLERANCE)
        
        if "error" in analysis:
            analysis_html = f'<div class="alert alert-danger">{analysis["error"]}</div>'
        else:
            # Get company info
            company_info = analysis["Company_Info"]
            company_name = company_info.get("Name", ticker)
            sector = company_info.get("Sector", "N/A")
            industry = company_info.get("Industry", "N/A")
            
            # Format market cap
            market_cap = company_info.get("Market_Cap")
            if market_cap:
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f} T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f} B"
                elif market_cap >= 1e6:
                    market_cap_str = f"${market_cap/1e6:.2f} M"
                else:
                    market_cap_str = f"${market_cap:,.2f}"
            else:
                market_cap_str = "N/A"
            
            # Format other metrics
            price = analysis.get("Current_Price", "N/A")
            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else price
            
            target_price = company_info.get("Target_Price")
            target_price_str = f"${target_price:.2f}" if isinstance(target_price, (int, float)) else "N/A"
            
            pe_ratio = company_info.get("PE_Ratio")
            pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
            
            dividend_yield = company_info.get("Dividend_Yield")
            dividend_yield_str = f"{dividend_yield:.2f}%" if isinstance(dividend_yield, (int, float)) else "N/A"
            
            # Technical indicators
            indicators = analysis["Technical_Indicators"]
            rsi = indicators.get("RSI")
            rsi_str = f"{rsi:.2f}" if isinstance(rsi, (int, float)) else "N/A"
            rsi_class = ""
            if isinstance(rsi, (int, float)):
                if rsi <= 30:
                    rsi_class = "text-success"  # Oversold - bullish
                elif rsi >= 70:
                    rsi_class = "text-danger"   # Overbought - bearish
            
            # Signal info
            signal = analysis["Signal"]
            signal_score = signal.get("Score", 0)
            signal_type = signal.get("Type", "N/A")
            signal_class = ""
            if signal_score >= 70:
                signal_class = "text-success font-weight-bold"
            elif signal_score >= 30:
                signal_class = "text-success"
            elif signal_score <= -70:
                signal_class = "text-danger font-weight-bold"
            elif signal_score <= -30:
                signal_class = "text-danger"
            
            # Sentiment info
            sentiment_data = analysis["Sentiment"]
            sentiment_score = sentiment_data.get("Score")
            sentiment_str = f"{sentiment_score:.2f}" if isinstance(sentiment_score, (int, float)) else "N/A"
            sentiment_class = ""
            if isinstance(sentiment_score, (int, float)):
                if sentiment_score >= 0.3:
                    sentiment_class = "text-success"
                elif sentiment_score <= -0.3:
                    sentiment_class = "text-danger"
            
            # Headlines
            headlines = sentiment_data.get("Headlines", [])
            headlines_html = ""
            if headlines:
                headlines_html = "<ul class='list-group mt-3'>"
                for headline in headlines:
                    headlines_html += f"<li class='list-group-item'>{headline}</li>"
                headlines_html += "</ul>"
            
            # Position recommendation
            position_rec = analysis.get("Position_Recommendation", {})
            rec_type = position_rec.get("Recommendation", "Neutro")
            rec_strength = position_rec.get("Strength", "")
            rec_note = position_rec.get("Note", "")
            
            rec_class = ""
            if rec_type == "Comprar":
                rec_class = "text-success"
                if rec_strength == "Forte":
                    rec_class += " font-weight-bold"
            elif rec_type == "Vender/Evitar":
                rec_class = "text-danger"
                if rec_strength == "Forte":
                    rec_class += " font-weight-bold"
            
            # Position sizing
            position_size_html = ""
            if "Suggested_Shares" in position_rec:
                shares = position_rec["Suggested_Shares"]
                approx_value = position_rec["Approximate_Value"]
                position_size_html = f"""
                <div class="row mt-3">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                Sugestão de Posição
                            </div>
                            <div class="card-body">
                                <p><strong>Quantidade sugerida:</strong> {shares} ações</p>
                                <p><strong>Valor aproximado:</strong> ${approx_value:.2f}</p>
                                <p><small>Baseado no seu perfil de risco {RISK_TOLERANCE} e saldo disponível.</small></p>
                            </div>
                        </div>
                    </div>
                </div>
                """
            
            analysis_html = f"""
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h4>{company_name} ({ticker})</h4>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h5>Informações da Empresa</h5>
                            <table class="table table-sm">
                                <tr>
                                    <th>Setor:</th>
                                    <td>{sector}</td>
                                </tr>
                                <tr>
                                    <th>Indústria:</th>
                                    <td>{industry}</td>
                                </tr>
                                <tr>
                                    <th>Market Cap:</th>
                                    <td>{market_cap_str}</td>
                                </tr>
                                <tr>
                                    <th>P/E Ratio:</th>
                                    <td>{pe_ratio_str}</td>
                                </tr>
                                <tr>
                                    <th>Dividend Yield:</th>
                                    <td>{dividend_yield_str}</td>
                                </tr>
                                <tr>
                                    <th>Preço Atual:</th>
                                    <td>{price_str}</td>
                                </tr>
                                <tr>
                                    <th>Preço Alvo (Médio):</th>
                                    <td>{target_price_str}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div class="col-md-6">
                            <h5>Indicadores Técnicos</h5>
                            <table class="table table-sm">
                                <tr>
                                    <th>RSI (14):</th>
                                    <td class="{rsi_class}">{rsi_str}</td>
                                </tr>
                                <tr>
                                    <th>MA9:</th>
                                    <td>${indicators.get('MA9', 'N/A'):.2f if isinstance(indicators.get('MA9'), (int, float)) else 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>MA21:</th>
                                    <td>${indicators.get('MA21', 'N/A'):.2f if isinstance(indicators.get('MA21'), (int, float)) else 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>Bollinger Superior:</th>
                                    <td>${indicators.get('Upper_Bollinger', 'N/A'):.2f if isinstance(indicators.get('Upper_Bollinger'), (int, float)) else 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>Bollinger Inferior:</th>
                                    <td>${indicators.get('Lower_Bollinger', 'N/A'):.2f if isinstance(indicators.get('Lower_Bollinger'), (int, float)) else 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>MACD:</th>
                                    <td>{indicators.get('MACD', 'N/A'):.4f if isinstance(indicators.get('MACD'), (int, float)) else 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>Sinal MACD:</th>
                                    <td>{indicators.get('MACD_Signal', 'N/A'):.4f if isinstance(indicators.get('MACD_Signal'), (int, float)) else 'N/A'}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Análise de Sinal</h5>
                                </div>
                                <div class="card-body">
                                    <p><strong>Score:</strong> <span class="{signal_class}">{signal_score:.1f}</span></p>
                                    <p><strong>Tipo de Sinal:</strong> <span class="{signal_class}">{signal_type}</span></p>
                                    <div class="progress">
                                        <div class="progress-bar {'bg-success' if signal_score > 0 else 'bg-danger'}" role="progressbar" style="width: {min(100, abs(signal_score))}%" aria-valuenow="{signal_score}" aria-valuemin="-100" aria-valuemax="100"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Análise de Sentimento</h5>
                                </div>
                                <div class="card-body">
                                    <p><strong>Score de Sentimento:</strong> <span class="{sentiment_class}">{sentiment_str}</span></p>
                                    <p><strong>Manchetes Recentes:</strong></p>
                                    {headlines_html}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Recomendação</h5>
                                </div>
                                <div class="card-body">
                                    <h3 class="{rec_class}">{rec_type} {rec_strength}</h3>
                                    <p>{rec_note}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {position_size_html}
                </div>
            </div>
            """
    
    return render_template_string(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise de Ativo - Swing Trade v4</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{ margin-top: 20px; }}
                .card {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Análise de Ativo Específico</h1>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h4>Pesquisar Ativo</h4>
                    </div>
                    <div class="card-body">
                        <form method="get" action="/pesquisar">
                            <div class="input-group">
                                <input type="text" name="ticker" class="form-control" placeholder="Digite o ticker (ex: AAPL, MSFT, NTLA)" value="{ticker}">
                                <div class="input-group-append">
                                    <button type="submit" class="btn btn-primary">Analisar</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
                
                {analysis_html}
                
                <a href="/" class="btn btn-secondary mt-3">Voltar</a>
            </div>
        </body>
        </html>
        """
    )

if __name__ == "__main__":
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    app.run(debug=True, port=5002)
