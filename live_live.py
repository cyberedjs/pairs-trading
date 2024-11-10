from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance.exceptions import BinanceAPIException
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from binance.client import Client
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import threading
import requests
import logging
import time
import json
import os
import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

# 글로벌 변수 및 락 정의
data_updated = False  # 데이터 업데이트 플래그
data_lock = threading.Lock()  # 쓰레드 안전성을 위한 락

pd.options.mode.chained_assignment = None

with open('./binance/api2.json', 'r') as f:
    api = json.load(f)

### API
binance_api_key = api['key']       #Enter your own API-key here
binance_api_secret = api['secret'] #Enter your own API-secret here

client = Client(binance_api_key, binance_api_secret, testnet=False)

# 로깅 설정
logging.basicConfig(filename='trading.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# 데이터 디렉토리 설정
DATA_DIR = './data'
data_dir = DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

def initialize():
    try:
        # Step 1: 모든 선물 종목 가져오기
        exchange_info = client.futures_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if (s['status'] == 'TRADING') and (s['symbol'][-4:] == 'USDT')]
        logging.info(f"총 선물 종목 수: {len(symbols)}")

        current_time = int(time.time() * 1000)  # 현재 시간을 밀리초로 변환
        start = current_time - (6 * 30 * 24 * 60 * 60 * 1000)  # 대략 6개월 전 타임스탬프

        # 다중 스레드를 이용한 데이터 수집
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(collect_data, symbol, start): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                data = future.result()
                if not data.empty:
                    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
                    data.to_csv(file_path)
                    logging.info(f"{symbol}의 데이터를 저장했습니다.")
                else:
                    logging.info(f"{symbol}의 데이터를 수집하지 못했습니다.")
        
    except Exception as e:
        logging.error(f"초기화 중 오류 발생: {e}")

# Slack에 메시지를 보내는 함수
def send_message_to_slack(message):
    # 메시지를 전송하기 위한 데이터 구성
    payload = {
        'text': message
    }

    slack_webhook_url = "https://hooks.slack.com/services/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"

    # POST 요청을 통해 슬랙으로 메시지 전송
    response = requests.post(slack_webhook_url, data=json.dumps(payload),
                             headers={'Content-Type': 'application/json'})

    # 요청 결과 확인
    if response.status_code == 200:
        logging.info("메시지가 슬랙으로 전송되었습니다.")
    else:
        logging.error(f"메시지 전송 실패: {response.status_code}")

def get_binance_futures_5m(symbol, start_time, end_time=None, limit=1000):
    """ 바이낸스 선물 5분봉 데이터를 수집하는 함수 """
    params = {
        'symbol': symbol,
        'interval': '5m',  # 5분봉
        'startTime': start_time,
        'limit': limit
    }
    if end_time:
        params['endTime'] = end_time

    url = "https://fapi.binance.com/fapi/v1/klines"
    
    # API 요청
    response = requests.get(url, params=params)
    data = response.json()
    
    # 데이터프레임으로 변환
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    
    # timestamp를 사람이 읽을 수 있는 시간으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def collect_data(symbol, start):
    """ 과거 6개월 데이터를 수집하는 함수 """
    # 6개월 전의 타임스탬프 계산
    current_time = int(time.time() * 1000)  # 현재 시간을 밀리초로 변환
    # six_months_ago = current_time - (6 * 30 * 24 * 60 * 60 * 1000)  # 대략 6개월 전 타임스탬프
    all_data = []

    while current_time > start:

        # 데이터를 요청하고 수집
        df = get_binance_futures_5m(symbol=symbol, start_time=start)
        
        if df.empty:
            break
        
        # 최신 데이터를 수집했으므로, 마지막 데이터의 타임스탬프를 사용해 이전 데이터를 가져옴
        start = int(df['timestamp'].max().timestamp() * 1000) + 1  # 밀리초로 변환
        all_data.append(df)
        
        # API 제한을 피하기 위해 잠시 대기
        time.sleep(1)

    # 전체 데이터를 하나의 데이터프레임으로 병합
    all_data = pd.concat(all_data, ignore_index=True)
    
    return all_data

# 상위 100개 종목 선정 함수
def get_top_100_symbols(data_dir=DATA_DIR):
    """
    과거 6개월간의 평균 거래대금을 기준으로 상위 100개 종목을 선정합니다.
    6개월 데이터가 없는 종목은 제외됩니다.
    """
    symbols = []
    avg_volumes = []

    six_months_ago = datetime.utcnow() - timedelta(days=180)

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            symbol = file.replace('.csv', '')
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
            # 6개월 이상의 데이터가 있는지 확인
            if df.index.min() < six_months_ago:
                # 6개월간의 데이터를 사용하여 평균 거래대금 계산
                recent_df = df.loc[df.index >= six_months_ago]
                required_length = int(180 * 24 * 12 * 0.5)  # 6개월 * 24시간 * 12 (5분봉) * 0.5 = 절반 이상
                if len(recent_df) < required_length:
                    continue  # 데이터가 충분하지 않으면 제외
                recent_df['pv'] = recent_df['close'] * recent_df['volume']
                avg_volume = recent_df['pv'].mean()
                symbols.append(symbol)
                avg_volumes.append(avg_volume)

    # 평균 거래대금 기준으로 정렬하여 상위 100개 종목 선정
    volume_df = pd.DataFrame({'symbol': symbols, 'avg_volume': avg_volumes})
    top_100 = volume_df.sort_values(by='avg_volume', ascending=False).head(100)['symbol'].tolist()

    return top_100

# 데이터 업데이트 함수
def update_data(data_dir=DATA_DIR):

    exchange_info = client.futures_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols'] if (s['status'] == 'TRADING') and (s['symbol'][-4:] == 'USDT')]

    logging.info(f"Symbol의 개수는 : {len(symbols)}")
    
    for symbol in symbols:

        file_path = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(file_path):

            df_existing = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            last_time = df_existing.index[-1]

            if int(time.time() * 1000) - int(last_time.timestamp() * 1000) < 300000:
                logging.info(f"Symbol : {symbol}; UPDATED TO THE LATEST 5M")
                continue

            new_start_time = last_time + timedelta(minutes=5)
            new_data = collect_data(symbol, int(new_start_time.timestamp() * 1000))
            new_data = new_data.set_index('timestamp', drop=True)

            if new_data.empty:
                continue  # 데이터가 없으면 건너뜀

            df_updated = pd.concat([df_existing, new_data]).drop_duplicates().sort_index()
        else:
            logging.error(f"{symbol}의 데이터가 존재하지 않아 제외됩니다.")
            continue
        
        df_updated.to_csv(file_path)
        logging.info(f"{symbol}의 데이터를 업데이트했습니다.")

# PairSelector 클래스 (수정 필요 없음)
class PairSelector:
    def __init__(self, top100, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.top_100 = top100
        self.prices = self.load_prices()
        self.pairs = self.generate_pairs()
        self.filtered_pairs_1 = []
        self.filtered_pairs_2 = []
    
    def load_prices(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        df_list = []
        for file in files:
            sym = file[:-4]

            if sym in self.top_100:
                symbol = file.replace('.csv', '')
                df = pd.read_csv(os.path.join(data_dir, file), index_col='timestamp', parse_dates=True)
                df.rename(columns={'close': symbol}, inplace=True)
                df_list.append(df[[symbol]])
            else:
                continue

        prices = pd.concat(df_list, axis=1).dropna()
        prices = np.log(prices)  # 로그 가격 변환
        return prices
    
    def kss_test(self, spread):
        # Step 1: ΔS_t 계산
        delta_spread = spread.diff().dropna()

        # Step 2: 비선형 항 생성: (S_t-1)^3
        lag_spread = spread.shift(1).dropna() ** 3

        # ΔS_t와 (S_t-1)^3을 정렬하여 공통된 인덱스를 맞춤
        delta_spread, lag_spread = delta_spread.align(lag_spread, join='inner')

        # Step 3: OLS 회귀 실행: ΔS_t = δ(S_t-1)^3 + ε_t
        model = sm.OLS(delta_spread, lag_spread).fit()

        # Step 4: KSS 검정 통계량 및 p-value 출력
        p_value = model.pvalues.values[0]

        return p_value

    def generate_pairs(self):
        tickers = self.prices.columns.tolist()
        pairs = []
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                if tickers[i] != tickers[j]:
                    pairs.append((tickers[i], tickers[j]))
        return pairs

    def filter_pairs(self, pair):
        try:
            price1 = self.prices[pair[0]]
            price2 = self.prices[pair[1]]

            corr = price1.corr(price2)
            if (abs(corr) > 0.975):  # 상관계수가 높은 페어만 통과
                return pair
            return None
        except Exception as e:
            return None

    def filter_pairs2(self, pair):
        try:
            price1 = self.prices[pair[0]]
            price2 = self.prices[pair[1]]

            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            alpha, beta = model.params
            spread = price1 - beta * price2 - alpha
            p_value = self.kss_test(spread)
            if p_value < 0.01:
                return pair
            else:
                adf_result = adfuller(spread)
                if adf_result[1] < 0.01:
                    return pair
            return None
        except Exception as e:
            return None    

    def run(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            # 각 페어를 비동기로 제출
            futures = {executor.submit(self.filter_pairs, pair): pair for pair in self.pairs}
            cnt = 0
            for future in as_completed(futures):
                try:
                    if cnt % 1000 == 0:
                        logging.info(f"{cnt}번째 끝.....")
                    result = future.result()
                    if result is not None:
                        self.filtered_pairs_1.append(result)
                except Exception as e:
                    logging.error(f"Error while runningg filter_pairs : {e}")
                cnt += 1

        logging.info(f'첫 번째 필터에서 살아남은 페어의 개수는 {len(self.filtered_pairs_1)}개입니다---------')

        with ThreadPoolExecutor(max_workers=1) as executor:
            # 각 페어를 비동기로 제출
            futures = {executor.submit(self.filter_pairs2, pair): pair for pair in self.filtered_pairs_1}
            cnt = 0
            for future in as_completed(futures):
                try:
                    if cnt % 5 == 0:
                        logging.info(f"{cnt}번째 끝.....")
                    result = future.result()
                    if result is not None:
                        self.filtered_pairs_2.append(result)
                except Exception as e:
                    logging.error(f"Error while runningg filter_pairs : {e}")
                cnt += 1

        logging.info(f'두 번째 필터에서 살아남은 페어의 개수는 {len(self.filtered_pairs_2)}개입니다---------')

        return self.filtered_pairs_2

# HalfLifeEstimator 클래스 (변경 없음)
class HalfLifeEstimator:
    def __init__(self, prices, pairs):
        self.prices = prices
        self.pairs = pairs
        self.half_lives = {}
        self.alphas = {}
        self.betas = {}
        
    def estimate_half_life(self, spread):
        delta_spread = spread.diff().dropna()  # ΔS_t = S_t - S_t-1 계산
        spread_lag = spread.shift(1).dropna()  # S_t-1 생성 (이전 시점의 스프레드)
        spread_lag, delta_spread = spread_lag.align(delta_spread, join='inner')  # ΔS_t와 S_t-1 맞춤

        # 회귀 분석 수행하여 θ 추정
        theta_model = sm.OLS(delta_spread, sm.add_constant(spread_lag)).fit()
        theta = -theta_model.params[0]  # θ 추정값 (음수 부호 주의)

        if theta > 0:
            halflife = np.log(2) / (theta * 288)
        else:
            halflife = -999

        return halflife
    
    def select_pair(self):
        for pair in self.pairs:
            price1 = self.prices[pair[0]]
            price2 = self.prices[pair[1]]
            
            X = sm.add_constant(price2)  # 자산 2를 독립 변수로 설정
            model = sm.OLS(price1, X).fit()  # OLS 회귀 분석 수행
            alpha, beta = model.params  # 상수항(α)과 기울기(β) 추정
            spread = price1 - beta * price2 - alpha  # 스프레드 계산
            # 반감기 추정
            halflife = self.estimate_half_life(spread)
            if halflife > 0:
                self.half_lives[pair] = halflife
                self.alphas[pair] = alpha
                self.betas[pair] = beta

        # 반감기가 가장 낮은 페어 선택
        sorted_values = sorted(self.half_lives.values())
        selected_pair = next(k for k, v in self.half_lives.items() if v == sorted_values[0])

        return selected_pair, self.half_lives[selected_pair], self.alphas[selected_pair], self.betas[selected_pair]

# LiveTradingStrategy 클래스
class LiveTradingStrategy:
    def __init__(self, client, pair, alpha, beta, lookback, leverage=1, transaction_cost=0.0005, pre_spread_history=None):

        self.client = client
        self.pair = pair
        self.alpha = alpha
        self.beta = beta
        self.lookback = lookback
        self.leverage = leverage
        self.transaction_cost = transaction_cost
        self.positions = {'A': 0, 'B': 0}
        self.entry_prices = {'A': 0.0, 'B': 0.0}
        self.order_history = []
        self.pnl_history = []
        self.spread_history = pre_spread_history.copy() if pre_spread_history else []
        self.zscore_history = []  # z-score 히스토리 저장 리스트
        self.set_leverage()
        self.initialize_capital()

        # 시각화를 위한 변수 초기화
        self.upper_entry_history = []
        self.lower_entry_history = []
        self.upper_exit_history = []
        self.lower_exit_history = []
        self.timestamps = []

        # 초기 스프레드 히스토리를 기반으로 zscore_history 초기화
        if self.spread_history:
            self.initialize_zscore_history()

    def initialize_capital(self):

        try:
            balance_info = self.client.futures_account_balance()
            usdt_balance = next((item for item in balance_info if item['asset'] == 'USDT'), None)
            if usdt_balance:
                total_balance = float(usdt_balance['balance'])
                self.capital = total_balance
                logging.info(f"초기 자본 설정: {self.capital} USDT (총 잔고)")
            else:
                logging.error("USDT 잔고 정보를 찾을 수 없습니다.")
                self.capital = 200
        except BinanceAPIException as e:
            logging.error(f"잔고 조회 오류: {e}")
            self.capital = 200  # 기본값 설정
        except Exception as e:
            logging.error(f"잔고 초기화 중 예상치 못한 오류: {e}")
            self.capital = 200

    def set_leverage(self):
        """
        선택된 페어에 대해 레버리지를 설정합니다.
        """
        symbol = self.pair[0]
        symbol2 = self.pair[1]
        try:
            response = self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            response2 = self.client.futures_change_leverage(symbol=symbol2, leverage=self.leverage)
            logging.info(f"레버리지 설정: {symbol}, 레버리지: {self.leverage}")
            logging.info(f"레버리지 설정: {symbol2}, 레버리지: {self.leverage}")
        except BinanceAPIException as e:
            logging.error(f"레버리지 설정 오류 ({symbol}): {e}")
        except Exception as e:
            logging.error(f"레버리지 설정 중 예상치 못한 오류 ({symbol}): {e}")

    def initialize_zscore_history(self):

        spread_series = pd.Series(self.spread_history)
        spread_mean = spread_series.ewm(span=self.lookback, adjust=False).mean()
        spread_std = spread_series.ewm(span=self.lookback, adjust=False).std()
        zscore = (spread_series - spread_mean) / spread_std

        self.zscore_history = zscore.tolist()
    
    def calculate_spread(self, price1, price2):
        return price1 - self.beta * price2 - self.alpha
    
    def update_spread_history(self, spread):

        self.spread_history.append(spread)
        self.spread_history = self.spread_history[-self.lookback*2:]

    def calculate_zscore(self):

        spread_series = pd.Series(self.spread_history)
        spread_mean = spread_series.ewm(span=self.lookback, adjust=False).mean()
        spread_std = spread_series.ewm(span=self.lookback, adjust=False).std()
        zscore = (spread_series - spread_mean) / spread_std

        current_zscore = zscore.iloc[-1]
        # z-score 히스토리에 추가
        self.zscore_history.append(current_zscore)

        # zscore_st: z-score의 지수 이동 평균의 표준편차
        zscore_series = pd.Series(self.zscore_history)
        zscore_st = zscore_series.ewm(span=self.lookback, adjust=False).std().iloc[-1]
        
        self.timestamps.append(datetime.utcnow())

        return zscore_st

    def generate_signals(self, zscore_st):

         # 신호 임계값 설정
        upper_entry = zscore_st * 2.5
        lower_entry = -upper_entry
        upper_exit = zscore_st
        lower_exit = -upper_exit

        # 임계값들을 히스토리에 추가
        self.upper_entry_history.append(upper_entry)
        self.lower_entry_history.append(lower_entry)
        self.upper_exit_history.append(upper_exit)
        self.lower_exit_history.append(lower_exit)

        # zscore_history 리스트의 마지막 두 값 사용 (z_t1 = list[-2], z_t2 = list[-3])
        if len(self.zscore_history) < 2:
            return None  # 신호 생성에 충분한 데이터가 없음

        z_t1 = self.zscore_history[-1]
        z_t2 = self.zscore_history[-2]

        signal = None

        # 현재 포지션 상태 확인
        has_long_position = self.positions['A'] > 0 and self.positions['B'] < 0
        has_short_position = self.positions['A'] < 0 and self.positions['B'] > 0

        if z_t1 < lower_entry and z_t2 >= lower_entry and not (has_long_position or has_short_position):
            signal = 'LONG'
        elif z_t1 > lower_exit and z_t2 <= lower_exit and has_long_position:
            signal = 'EXIT_LONG'
        elif z_t1 > upper_entry and z_t2 <= upper_entry and not (has_long_position or has_short_position):
            signal = 'SHORT'
        elif z_t1 < upper_exit and z_t2 >= upper_exit and has_short_position:
            signal = 'EXIT_SHORT'   

        return signal

    def execute_order(self, signal, current_prices):
        """
        거래 신호에 따라 주문 실행
        
        Parameters:
        - signal: 생성된 거래 신호
        - current_prices: 현재 자산들의 가격 딕셔너리
        """
        symbol_A, symbol_B = self.pair
        price_A = current_prices[symbol_A]
        price_B = current_prices[symbol_B]
        timestamp = datetime.utcnow()

        try:
            half_capital = round(self.capital / 2)
            total_weight = round(1 + abs(self.beta), 1)
            allocation_A = round(half_capital * (1 / total_weight))
            allocation_B = round(half_capital * (abs(self.beta) / total_weight))

            logging.info(f"half_captial : {half_capital}, total_weight = {total_weight}, allocation_A = {allocation_A}, allocation_B = {allocation_B}")

            if signal == 'LONG':
                # 매수 A, 매도 B
                qty_A = round(allocation_A / price_A)
                qty_B = round(allocation_B / price_B)

                logging.info(f"Qty_A : {qty_A}, Qty_B : {qty_B}")
                message_content = f"Qty_A : {qty_A}, Qty_B : {qty_B}" 
                send_message_to_slack(message_content)

                # 시장가 주문 실행
                order_A = self.client.futures_create_order(symbol=symbol_A, side='BUY', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='SELL', type='MARKET', quantity=qty_B)
                # 포지션 업데이트
                self.positions['A'] += qty_A
                self.positions['B'] -= qty_B
                self.entry_prices['A'] = price_A
                self.entry_prices['B'] = price_B
                # 자본 업데이트 (거래 비용 반영)
                self.capital -= (allocation_A + allocation_B) * self.transaction_cost
                # 주문 기록
                self.order_history.append((timestamp, 'LONG', qty_A, qty_B, price_A, price_B))
                logging.info(f"LONG 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                message_content = f"LONG 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}"
                send_message_to_slack(message_content)

            elif signal == 'EXIT_LONG':

                if self.positions['A'] > 0 and self.positions['B'] < 0:
                    # 매도 A, 매수 B
                    qty_A = self.positions['A']
                    qty_B = -self.positions['B']
                    order_A = self.client.futures_create_order(symbol=symbol_A, side='SELL', type='MARKET', quantity=qty_A)
                    order_B = self.client.futures_create_order(symbol=symbol_B, side='BUY', type='MARKET', quantity=qty_B)
                    # 포지션 초기화
                    self.positions['A'] = 0
                    self.positions['B'] = 0
                    # 자본 업데이트 (거래 비용 반영)
                    self.capital += (allocation_A + allocation_B) * (1 - self.transaction_cost)
                    # 주문 기록
                    self.order_history.append((timestamp, 'EXIT_LONG', qty_A, qty_B, price_A, price_B))
                    logging.info(f"EXIT_LONG 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                    message_content = f"EXIT_LONG 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}"
                    send_message_to_slack(message_content)
                else:
                    logging.warning(f"EXIT_LONG 시그널이 발생했으나 포지션이 존재하지 않습니다: {self.positions}")

            elif signal == 'SHORT':
                # 매도 A, 매수 B
                qty_A = round(allocation_A / price_A)
                qty_B = round(allocation_B / price_B)

                logging.info(f"Qty_A : {qty_A}, Qty_B : {qty_B}")
                message_content = f"Qty_A : {qty_A}, Qty_B : {qty_B}"
                send_message_to_slack(message_content)

                order_A = self.client.futures_create_order(symbol=symbol_A, side='SELL', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='BUY', type='MARKET', quantity=qty_B)
                # 포지션 업데이트
                self.positions['A'] -= qty_A
                self.positions['B'] += qty_B
                self.entry_prices['A'] = price_A
                self.entry_prices['B'] = price_B
                # 자본 업데이트 (거래 비용 반영)
                self.capital -= (allocation_A + allocation_B) * self.transaction_cost
                # 주문 기록
                self.order_history.append((timestamp, 'SHORT', qty_A, qty_B, price_A, price_B))
                logging.info(f"SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                message_content = f"SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}"
                send_message_to_slack(message_content)

            elif signal == 'EXIT_SHORT':
                
                if self.positions['A'] < 0 and self.positions['B'] > 0:
                    # 매수 A, 매도 B
                    qty_A = -self.positions['A']
                    qty_B = self.positions['B']
                    order_A = self.client.futures_create_order(symbol=symbol_A, side='BUY', type='MARKET', quantity=qty_A)
                    order_B = self.client.futures_create_order(symbol=symbol_B, side='SELL', type='MARKET', quantity=qty_B)
                    # 포지션 초기화
                    self.positions['A'] = 0
                    self.positions['B'] = 0
                    # 자본 업데이트 (거래 비용 반영)
                    self.capital += (allocation_A + allocation_B) * (1 - self.transaction_cost)
                    # 주문 기록
                    self.order_history.append((timestamp, 'EXIT_SHORT', qty_A, qty_B, price_A, price_B))
                    logging.info(f"EXIT_SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                    message_content = f"EXIT_SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}"
                    send_message_to_slack(message_content)
                else:
                    logging.warning(f"EXIT_SHORT 시그널이 발생했으나 포지션이 존재하지 않습니다: {self.positions}")
        except Exception as e:
            logging.error(f"주문 실행 중 오류 발생: {e}")
            message_content = f"주문 실행 중 오류 발생: {e}"
            send_message_to_slack(message_content)

    def track_pnl(self, current_prices):
        symbol_A, symbol_B = self.pair
        price_A = current_prices[symbol_A]
        price_B = current_prices[symbol_B]
        
        if self.positions['A'] != 0 or self.positions['B'] != 0:
            pnl_A = (price_A - self.entry_prices['A']) * self.positions['A']
            pnl_B = (price_B - self.entry_prices['B']) * self.positions['B']
            total_pnl = pnl_A + pnl_B
            self.capital += total_pnl
            self.pnl_history.append((datetime.utcnow(), total_pnl))
            logging.info(f"PnL 업데이트: {datetime.utcnow()}, PnL: {total_pnl}")
            message_content = f"PnL 업데이트: {datetime.utcnow()}, PnL: {total_pnl}"
            send_message_to_slack(message_content)
        else:
            pass

    def run(self, new_price_data):
        # 이 메서드는 주기적으로 호출되어야 합니다 (예: 5분마다)
        try:

            print(f"LIVE TRADING STRATEGY run . . . .")

            price_A = new_price_data[self.pair[0]]
            price_B = new_price_data[self.pair[1]]
            log_price_A = np.log(price_A)
            log_price_B = np.log(price_B)
            spread = self.calculate_spread(log_price_A, log_price_B)
            self.update_spread_history(spread)

            # z-score와 zscore_st 계산
            zscore_st = self.calculate_zscore()

            logging.info(f"zscore_st : {zscore_st}; zscore t-2: {self.zscore_history[-2]}; zscore t-1 : {self.zscore_history[-1]}")

            # 신호 생성
            signal = self.generate_signals(zscore_st)

            # 주문 실행
            if signal:
                # 이미 포지션이 있는 경우 추가 진입 방지
                if signal in ['LONG', 'SHORT'] and (self.positions['A'] != 0 or self.positions['B'] != 0):
                    logging.info(f"이미 포지션이 잡혀 있어 {signal} 신호를 무시합니다.")
                else:
                    self.execute_order(signal, new_price_data)

            # PnL 추적
            self.track_pnl(new_price_data)

        except BinanceAPIException as e:
            logging.error(f"Binance API Exception during run: {e}")
        except Exception as e:
            logging.error(f"Error during run: {e}")

# Scheduler 설정 및 주간 작업 등록
scheduler = BackgroundScheduler()

# Flask와 Dash를 사용하여 웹 서버 및 대시보드 생성
from flask import Flask

# 전역 변수로 현재 전략을 저장
current_strategy = None

def create_dash_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        html.H1('Trading Strategy Visualization'),
        dcc.Graph(id='live-graph'),
        dcc.Interval(
            id='graph-update',
            interval=1*60*1000,  # 5초마다 업데이트
            n_intervals=0
        ),
        dcc.Store(id='data-updated', data=False)
    ])

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')],
        [State('data-updated', 'data')]
    )
    def update_graph(n, data_updated_state):
        global data_updated
        
        with data_lock:
            if current_strategy and data_updated:
                
                data_updated = False
                
                data = [
                    go.Scatter(
                        x=current_strategy.timestamps,
                        y=current_strategy.zscore_history[-len(current_strategy.timestamps):],
                        name='zscore',
                        mode='lines+markers'
                    ),
                    go.Scatter(
                        x=current_strategy.timestamps,
                        y=current_strategy.upper_entry_history,
                        name='upper_entry',
                        mode='lines'
                    ),
                    go.Scatter(
                        x=current_strategy.timestamps,
                        y=current_strategy.lower_entry_history,
                        name='lower_entry',
                        mode='lines'
                    ),
                    go.Scatter(
                        x=current_strategy.timestamps,
                        y=current_strategy.upper_exit_history,
                        name='upper_exit',
                        mode='lines'
                    ),
                    go.Scatter(
                        x=current_strategy.timestamps,
                        y=current_strategy.lower_exit_history,
                        name='lower_exit',
                        mode='lines'
                    )
                ]

                layout = go.Layout(
                    xaxis=dict(title='Time'),
                    yaxis=dict(title='Value'),
                    title='Trading Strategy Variables Over Time'
                )

                return {'data': data, 'layout': layout}
            else:
                # 데이터가 업데이트되지 않았으면 이전 그래프 유지
                raise dash.exceptions.PreventUpdate

    return app

def run_dash_app():
    app = create_dash_app()
    app.run_server(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def weekly_task():

    global current_strategy

    try:

        logging.info(f"Update_data에 입성 전. . . . .")

        # Step 1: 데이터 업데이트
        update_data(data_dir=DATA_DIR)

        # Step 2: 상위 100개 종목 선정
        top_100 = get_top_100_symbols(data_dir=DATA_DIR)
        logging.info(f"상위 100개 종목: {top_100}")

        if len(top_100) == 0:
            logging.info("상위 100개 종목을 선정할 수 없습니다.")
            return
        
        logging.info(f"PAIR SELECTOR에 입성 전. . . . .")

        # Step 3: 페어 선정
        pair_selector = PairSelector(top_100, data_dir=DATA_DIR)
        filtered_pairs = pair_selector.run()
        
        if not filtered_pairs:
            logging.info("이번 주에 적합한 페어를 찾지 못했습니다.")
            return
        
        logging.info(f"Half Life에 입성 전. . . . .")

        # Step 4: 반감기 추정 및 페어 선택
        half_life_estimator = HalfLifeEstimator(pair_selector.prices, filtered_pairs)
        selected_pair, halflife, alpha, beta = half_life_estimator.select_pair()
        lookback = int(halflife * 288)
       
        logging.info(f"alpha : {alpha}, beta : {beta}, lookback : {lookback}")
        message_content = f"alpha : {alpha}, beta : {beta}, lookback : {lookback}"
        send_message_to_slack(message_content)

        # Step 5: 스프레드 히스토리 초기화 (하루 전의 데이터 가져오기)
        
        logging.info(f"PRE SPREAD CALCULATION에 입성. . . . .")

        symbol_A, symbol_B = selected_pair
        logging.info(f"symbol_A : {symbol_A}, symbol_B : {symbol_B}")

        file_A = os.path.join(DATA_DIR, f"{symbol_A}.csv")
        file_B = os.path.join(DATA_DIR, f"{symbol_B}.csv")
        
        df_A = pd.read_csv(file_A, index_col='timestamp', parse_dates=True)
        df_B = pd.read_csv(file_B, index_col='timestamp', parse_dates=True)
        
        #이틀 전부터 현재까지의 데이터 (하루는 24시간, 288개의 5분봉)
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        df_A_recent = np.log(df_A.loc[df_A.index >= one_day_ago].tail(576))
        df_B_recent = np.log(df_B.loc[df_B.index >= one_day_ago].tail(576))
        
        # 스프레드 계산
        spread_history = (df_A_recent['close'] - beta * df_B_recent['close'] - alpha).tolist()
        desired_leverage = 1  # 필요에 따라 변경

        logging.info(f"LIVE TRADING 객체 생성. . . . .")
        
        # LiveTradingStrategy 객체 생성
        strategy = LiveTradingStrategy(
            client=client,
            pair=selected_pair,
            alpha=alpha,
            beta=beta,
            lookback=lookback,
            leverage=desired_leverage,
            transaction_cost=0.0005,
            pre_spread_history=spread_history
        )
        
        # 현재 전략을 교체
        current_strategy = strategy
        logging.info(f"새로운 거래 페어 선정: {selected_pair}, 반감기: {halflife}")
        
    except Exception as e:
        logging.error(f"weekly_task에서 오류 발생: {e}")

# 주간 작업 등록 (매주 월요일 00:00 UTC에 실행)
scheduler.add_job(weekly_task, 'cron', day_of_week='mon', hour=0, minute=0)
scheduler.start()

# # 실시간 실행 루프
def real_time_execution():
    global data_updated

    while True:

        logging.info(f"REAL TIME EXECUTION에 입성. . . . .")

        now = datetime.now()
        next_minute = (now.minute // 5 + 1) * 5

        if next_minute == 60:
            next_minute = 0
            next_time = (now + timedelta(hours=1)).replace(minute=next_minute, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)

        # 현재 시간과 다음 5분 단위까지 남은 시간 계산
        sleep_time = (next_time - now).total_seconds()

        # 남은 시간만큼 대기
        time.sleep(sleep_time)

        try:
            if current_strategy:
                # 선택된 페어의 현재 가격 가져오기
                symbol_A, symbol_B = current_strategy.pair

                ticker_A = client.futures_symbol_ticker(symbol=symbol_A)
                ticker_B = client.futures_symbol_ticker(symbol=symbol_B)
                
                new_price_data = {
                    symbol_A: float(ticker_A['price']),
                    symbol_B: float(ticker_B['price'])
                }
                message_content = f"{symbol_A} : {ticker_A['price']}, {symbol_B} : {ticker_B['price']}"
                send_message_to_slack(message_content)                                          
                logging.info(f"{symbol_A} : {ticker_A['price']}, {symbol_B} : {ticker_B['price']}")
                # 전략 실행
                current_strategy.run(new_price_data)
                # 데이터 업데이트 플래그 설정
                with data_lock:
                    data_updated = True

            else:
                logging.info("활성화된 거래 전략이 없습니다.")
        except BinanceAPIException as e:
            logging.error(f"Binance API Exception during real_time_execution: {e}")
        except Exception as e:
            logging.error(f"Error during real_time_execution: {e}")

#메인 실행 블록 수정
if __name__ == "__main__":

    # 1. weekly_task 함수를 즉시 실행
    weekly_task()

    # 2. real_time_execution 함수를 별도의 스레드에서 실행
    execution_thread = threading.Thread(target=real_time_execution, daemon=True)
    execution_thread.start()

    # 3. Dash 앱을 별도의 스레드에서 실행
    dash_thread = threading.Thread(target=run_dash_app)
    dash_thread.start()

    # 3. 메인 스레드가 종료되지 않도록 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Trading bot을 종료합니다.")
