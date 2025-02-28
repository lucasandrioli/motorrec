import os
import math
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from flask import Flask, render_template_string, request

# Para análise de sentimento (NLTK)
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Suprimir mensagens do yfinance
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ---------------- CONFIGURAÇÕES GERAIS ----------------
PORTFOLIO_FILE = "portfolioatual.csv"  # Arquivo CSV com suas posições
LOG_FILE = "trade_log.txt"

# Valores iniciais (atualizáveis via formulário)
ACCOUNT_BALANCE = 3000.0
INVESTED_AMOUNT = 0.0

# Chave da NewsAPI
NEWSAPI_KEY = "1c381d2eb1454239a527ebe33bcaff15"

# Mapeamento de setores -> ETFs representativos
sector_etfs = {
    "Technology": ["QQQ", "XLK"],
    "Healthcare": ["XLV", "IYH"],
    "Financial Services": ["XLF", "VFH"],
    "Consumer Cyclical": ["XLY", "IYC"],
    "Consumer Defensive": ["XLP", "VDC"],
    "Energy": ["XLE", "VDE"],
    "Industrials": ["XLI", "VIS"],
    "Materials": ["XLB", "VAW"],
    "Utilities": ["XLU", "VPU"],
    "Real Estate": ["VNQ", "IYR"],
    "Communication Services": ["IYZ", "VOX"]
}

# ---------------- FUNÇÃO DE LOG ----------------
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

# ---------------- LÓGICA DE LIMIARES DINÂMICOS ----------------
def get_dynamic_thresholds(total_investido):
    """
    Exemplo:
      - Meta global: 5% do total investido (mínimo US$100)
      - Meta parcial: 0.5% do total investido (mínimo US$10)
    """
    target_profit = max(0.05 * total_investido, 100)
    daily_target = max(0.005 * total_investido, 10)
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

    max_shares = int(round(max_shares))  # garante inteiro

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
def get_sentiment(ticker):
    url = f"https://newsapi.org/v2/top-headlines?q={ticker}&language=en&apiKey={NEWSAPI_KEY}"
    try:
        resp = requests.get(url)
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
                return avg_score, headlines[:5]
            else:
                return None, ["Sem pontuações."]
        else:
            return None, [f"Erro: {resp.status_code}"]
    except Exception as e:
        return None, [f"Erro: {str(e)}"]

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

    # Converte Quantity para int, caso seja float
    df['Quantity'] = df['Quantity'].apply(lambda x: int(round(x)))

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
    precos = []
    for _, row in df.iterrows():
        simbolo = row['Symbol']
        try:
            ticker = yf.Ticker(simbolo)
            info = ticker.info
            preco = info.get('regularMarketPrice') or info.get('currentPrice') or row['CSV Current Price']
            precos.append(preco)
            log_message(f"{simbolo}: Atualizado: {preco} (pre: {info.get('preMarketPrice')}, pos: {info.get('postMarketPrice')}, target: {info.get('targetMeanPrice')})")
        except Exception as e:
            log_message(f"Erro ao atualizar {simbolo}: {e}")
            precos.append(row['CSV Current Price'])
    df['Atualizado Current Price'] = precos
    return df

# ---------------- CALCULAR INDICADORES ----------------
def calcular_indicadores(simbolo):
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

    data['MA9'] = data[close_col].rolling(window=9, min_periods=9).mean()
    data['MA21'] = data[close_col].rolling(window=21, min_periods=21).mean()
    delta = data[close_col].diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MA20'] = data[close_col].rolling(window=20, min_periods=20).mean()
    data['std'] = data[close_col].rolling(window=20, min_periods=20).std()
    data['UpperBand'] = data['MA20'] + 2 * data['std']
    data['LowerBand'] = data['MA20'] - 2 * data['std']

    latest = data.iloc[-1]
    try:
        current_price = float(latest[close_col].item())
        ma9 = float(latest['MA9'].item()) if pd.notna(latest['MA9']) else None
        ma21 = float(latest['MA21'].item()) if pd.notna(latest['MA21']) else None
        rsi = float(latest['RSI'].item()) if pd.notna(latest['RSI']) else None
        ma20 = float(latest['MA20'].item()) if pd.notna(latest['MA20']) else None
        upper_band = float(latest['UpperBand'].item()) if pd.notna(latest['UpperBand']) else None
        lower_band = float(latest['LowerBand'].item()) if pd.notna(latest['LowerBand']) else None
    except Exception as e:
        return f"Erro ao converter valores para {simbolo}: {e}"

    return {
        "Preço Atual Yahoo": current_price,
        "MA9": ma9,
        "MA21": ma21,
        "RSI": rsi,
        "MA20": ma20,
        "UpperBand": upper_band,
        "LowerBand": lower_band
    }

# ---------------- ANÁLISE DO PORTFOLIO (COM TAXAS NA COMPRA E VENDA) ----------------
def analisar_portfolio(df):
    resultados = []
    for _, row in df.iterrows():
        simbolo = row['Symbol']
        preco_medio = row['Weighted Purchase Price']
        quantidade = row['Total Quantity']
        atualizado_price = row['Atualizado Current Price']

        pct_variacao = ((atualizado_price - preco_medio) / preco_medio) * 100
        lucro_em_dolares = (atualizado_price - preco_medio) * quantidade

        indicadores = calcular_indicadores(simbolo)
        indicadores_validos = not isinstance(indicadores, str)

        sentimento, _ = get_sentiment(simbolo)
        sentimento_str = f"{sentimento:.2f}" if sentimento is not None else "N/A"

        # Limiar dinâmico
        total_investido_ativo = preco_medio * quantidade
        target_profit, daily_target = get_dynamic_thresholds(total_investido_ativo)

        # Decisão
        if (pct_variacao <= -7) or (indicadores_validos and atualizado_price < preco_medio
                                   and indicadores['MA9'] < indicadores['MA21']
                                   and indicadores['RSI'] is not None and indicadores['RSI'] < 40):
            acao = "VENDER"
            recomendacao = f"Vender todas as {quantidade} ações para evitar prejuízos."
        else:
            profit_per_share = atualizado_price - preco_medio

            # Meta global
            if lucro_em_dolares >= target_profit and profit_per_share > 0:
                acao = "REALIZAR LUCRO"
                shares_to_sell = find_shares_to_sell_for_target(target_profit, profit_per_share, atualizado_price, quantidade)
                volume = shares_to_sell * atualizado_price
                fee = xp_equities_fee(volume)
                net_after_fee = shares_to_sell * profit_per_share - fee
                recomendacao = (f"Vender {shares_to_sell} ações para lucro líquido ~US$ {net_after_fee:.2f} "
                                f"(meta ~{target_profit:.2f}).")
            # Meta parcial
            elif lucro_em_dolares >= daily_target and profit_per_share > 0:
                acao = "REALIZAR LUCRO PARCIAL"
                shares_to_sell = find_shares_to_sell_for_target(daily_target, profit_per_share, atualizado_price, quantidade)
                volume = shares_to_sell * atualizado_price
                fee = xp_equities_fee(volume)
                net_after_fee = shares_to_sell * profit_per_share - fee
                recomendacao = (f"Vender {shares_to_sell} ações para lucro líquido ~US$ {net_after_fee:.2f} "
                                f"(meta parcial ~{daily_target:.2f}).")
            else:
                # Verificar pullback para comprar
                if (indicadores_validos
                    and atualizado_price <= indicadores['LowerBand'] * 1.03
                    and indicadores['RSI'] is not None and indicadores['RSI'] < 50
                    and indicadores['MA9'] is not None and indicadores['MA21'] is not None
                    and indicadores['MA9'] > indicadores['MA21']):

                    acao = "COMPRAR MAIS"
                    valor_a_investir = min(1000, ACCOUNT_BALANCE)
                    # Usa find_shares_to_buy_for_amount para considerar a taxa
                    additional_shares = find_shares_to_buy_for_amount(valor_a_investir, atualizado_price)
                    recomendacao = f"Comprar {additional_shares} ações (pullback identificado)."
                else:
                    acao = "MANUTER"
                    recomendacao = "Manter a posição."

        # Ajuste de sentimento
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
            "Sentimento": sentimento_str,
            "Ação Recomendada": acao,
            "Recomendação": recomendacao
        })

    return resultados

# ---------------- SIMULAR (NÃO INCORPORA TAXA AQUI) ----------------
def simulate_trades(analise, total_invested_initial):
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
    global ACCOUNT_BALANCE, INVESTED_AMOUNT
    if request.method == "POST":
        try:
            ACCOUNT_BALANCE = float(request.form.get("account_balance", ACCOUNT_BALANCE))
            INVESTED_AMOUNT = float(request.form.get("invested_amount", INVESTED_AMOUNT))
            log_message(f"Valores atualizados: ACCOUNT_BALANCE = {ACCOUNT_BALANCE}, INVESTED_AMOUNT = {INVESTED_AMOUNT}")
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
        df_consolidado = atualizar_precos(df_consolidado)
        analise = analisar_portfolio(df_consolidado)

        tabela_final = """
        <table class="table table-bordered">
            <tr>
                <th>Symbol</th>
                <th>Weighted Purchase Price</th>
                <th>Total Quantity</th>
                <th>Atualizado Current Price</th>
                <th>P&L Variation</th>
                <th>Profit</th>
                <th>Sentimento</th>
                <th>Ação Recomendada</th>
                <th>Recomendação</th>
            </tr>
        """
        for item in analise:
            tabela_final += f"""
            <tr>
                <td>{item['Symbol']}</td>
                <td>{item['Weighted Purchase Price']:.2f}</td>
                <td>{item['Total Quantity']}</td>
                <td>{item['Atualizado Current Price']:.2f}</td>
                <td>{item['P&L Variation']}</td>
                <td>{item['Profit']}</td>
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

    refresh_meta = '<meta http-equiv="refresh" content="5">'

    return render_template_string(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise Dinâmica de Portfolio</title>
            {refresh_meta}
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style> body {{ margin-top: 20px; }} table {{ margin-bottom: 30px; }} </style>
        </head>
        <body>
            <div class="container">
                <h1>Análise Dinâmica de Portfolio</h1>
                <form method="post" action="/">
                    <div class="form-group">
                        <label for="account_balance">Saldo Disponível (Account Balance):</label>
                        <input type="number" step="0.01" name="account_balance" id="account_balance" class="form-control" value="{ACCOUNT_BALANCE}">
                    </div>
                    <div class="form-group">
                        <label for="invested_amount">Valor Investido (Invested Amount):</label>
                        <input type="number" step="0.01" name="invested_amount" id="invested_amount" class="form-control" value="{INVESTED_AMOUNT}">
                    </div>
                    <button type="submit" class="btn btn-secondary">Atualizar Valores</button>
                </form>
                <br>
                <p><strong>Total Investido Inicial:</strong> US$ {total_invested_initial:.2f}</p>
                <p><strong>Valor Atual do Portfolio:</strong> US$ {total_current_value:.2f}</p>
                <p><strong>Lucro/Prejuízo Total:</strong> US$ {profit_total:.2f}</p>
                <p><strong>Valor Investido Ideal (simulado):</strong> US$ {simulated_invested:.2f}</p>
                {tabela_final}
                <a href="/" class="btn btn-primary mt-3">Atualizar Página</a>
                <br><br>
                <a href="/oportunidades" class="btn btn-success mt-3">Analisar Oportunidades Setoriais</a>
            </div>
        </body>
        </html>
        """
    )

# ---------------- ROTA OPORTUNIDADES ----------------
@app.route("/oportunidades", methods=["GET"])
def oportunidades():
    tickers_input = request.args.get("tickers", "").strip()
    candidate_tickers = []
    if tickers_input:
        candidate_tickers = [t.strip().strip('"').strip("'") for t in tickers_input.split(",") if t.strip()]
    else:
        df_raw = carregar_portfolio()
        if not df_raw.empty:
            df_consolidado = consolidar_portfolio(df_raw)
            portfolio_tickers = list(df_consolidado['Symbol'].unique())
            candidate_tickers.extend(portfolio_tickers)
            # Para cada ticker do portfolio, tenta obter setor e adicionar tickers do dicionário
            for symbol in portfolio_tickers:
                try:
                    ticker_obj = yf.Ticker(symbol)
                    info = ticker_obj.info
                    sector = info.get("sector", None)
                    if sector and sector in sector_etfs:
                        for sug in sector_etfs[sector]:
                            if sug not in candidate_tickers:
                                candidate_tickers.append(sug)
                except Exception as e:
                    log_message(f"Erro ao obter setor para {symbol}: {e}")

    if not candidate_tickers:
        return render_template_string("<p>Não foram encontrados tickers para análise. Insira manualmente.</p>")

    oportunidades_list = []
    for simbolo in candidate_tickers:
        try:
            data = yf.download(simbolo, period="3mo", interval="1d", auto_adjust=True, progress=False)
        except Exception as e:
            log_message(f"Erro ao baixar dados para {simbolo}: {e}")
            continue
        if data.empty or data.shape[0] < 20:
            continue
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
            continue

        data['MA9'] = data[close_col].rolling(window=9, min_periods=9).mean()
        data['MA21'] = data[close_col].rolling(window=21, min_periods=21).mean()
        delta = data[close_col].diff().fillna(0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['MA20'] = data[close_col].rolling(window=20, min_periods=20).mean()
        data['std'] = data[close_col].rolling(window=20, min_periods=20).std()
        data['UpperBand'] = data['MA20'] + 2 * data['std']
        data['LowerBand'] = data['MA20'] - 2 * data['std']

        # Volume médio
        if 'Volume' in data.columns:
            volume_avg = data['Volume'].rolling(window=20, min_periods=20).mean().iloc[-1]
            if volume_avg < 1e6:
                continue

        latest = data.iloc[-1]
        try:
            preco_atual = float(latest[close_col].item())
            rsi = float(latest['RSI'].item()) if pd.notna(latest['RSI']) else None
            ma9 = float(latest['MA9'].item()) if pd.notna(latest['MA9']) else None
            ma21 = float(latest['MA21'].item()) if pd.notna(latest['MA21']) else None
            upper_band = float(latest['UpperBand'].item()) if pd.notna(latest['UpperBand']) else None
            lower_band = float(latest['LowerBand'].item()) if pd.notna(latest['LowerBand']) else None
        except Exception:
            continue

        sentimento, _ = get_sentiment(simbolo)
        sentimento_str = f"{sentimento:.2f}" if sentimento is not None else "N/A"

        # Critérios de pullback ou breakout
        if (ma9 is not None and ma21 is not None and ma9 > ma21
            and rsi is not None and 40 <= rsi < 50
            and preco_atual <= lower_band * 1.03):
            recomend = "Pullback moderado (RSI 40-50, Bollinger inferior)"
            if sentimento is not None and sentimento < -0.3:
                recomend += " [Notícias negativas]"
            oportunidades_list.append({
                "Symbol": simbolo,
                "Motivo": recomend,
                "Preço Atual": preco_atual,
                "RSI": rsi,
                "Sentimento": sentimento_str,
                "Sugestão": "Comprar? Verificar se o pullback é legítimo."
            })
        elif (rsi is not None and rsi > 70
              and upper_band is not None and preco_atual >= upper_band * 0.98):
            recomend = "Breakout (RSI > 70, Bollinger superior)"
            if sentimento is not None and sentimento > 0.3:
                recomend += " [Notícias positivas]"
            oportunidades_list.append({
                "Symbol": simbolo,
                "Motivo": recomend,
                "Preço Atual": preco_atual,
                "RSI": rsi,
                "Sentimento": sentimento_str,
                "Sugestão": "Comprar? Forte momentum detectado."
            })

    if not oportunidades_list:
        ops_html = "<p>Nenhuma oportunidade encontrada.</p>"
    else:
        ops_html = """
        <table class="table table-bordered">
            <tr>
                <th>Symbol</th>
                <th>Motivo</th>
                <th>Preço Atual</th>
                <th>RSI</th>
                <th>Sentimento</th>
                <th>Sugestão</th>
            </tr>
        """
        for op in oportunidades_list:
            ops_html += f"""
            <tr>
                <td>{op['Symbol']}</td>
                <td>{op['Motivo']}</td>
                <td>{op['Preço Atual']:.2f}</td>
                <td>{op['RSI']:.2f}</td>
                <td>{op['Sentimento']}</td>
                <td>{op['Sugestão']}</td>
            </tr>
            """
        ops_html += "</table>"

    return render_template_string(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Oportunidades de Compra - Swing Trade</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style> body {{ margin-top: 20px; }} table {{ margin-bottom: 30px; }} </style>
        </head>
        <body>
            <div class="container">
                <h1>Oportunidades de Compra (Swing Trade)</h1>
                <form method="get" action="/oportunidades">
                    <div class="form-group">
                        <label for="tickers">Digite os tickers (separados por vírgula) ou deixe vazio para usar os ativos do portfolio e sugestões setoriais:</label>
                        <input type="text" name="tickers" id="tickers" class="form-control" value="{tickers_input}">
                    </div>
                    <button type="submit" class="btn btn-secondary">Buscar Oportunidades</button>
                </form>
                <br>
                {ops_html}
                <a href="/" class="btn btn-primary mt-3">Voltar</a>
            </div>
        </body>
        </html>
        """
    )

if __name__ == "__main__":
    app.run(debug=True, port=5002)
