from apscheduler.schedulers.background import BackgroundScheduler
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance.exceptions import BinanceAPIException
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from binance.client import Client
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import threading
import requests
import logging
import time
import json
import os
import numba
from threading import Lock

pd.options.mode.chained_assignment = None

with open('./binance/api2.json', 'r') as f:
    api = json.load(f)

binance_api_key = api['key']       
binance_api_secret = api['secret']

client = Client(binance_api_key, binance_api_secret, testnet=False)

# 로깅 설정
logging.basicConfig(filename='trading.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

DATA_DIR = './data'
data_dir = DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

# z-score 데이터를 저장할 전역 변수
zscore_data = {
    'timestamps': [],
    'pairs': {},
    'zscore_st': {}
}

# 전역 변수로 데이터 저장소 추가
price_data = {}
data_lock = Lock()

def update_zscore_data(pair, zscore, zscore_st):
    """z-score 데이터를 업데이트하는 함수"""
    current_time = datetime.utcnow()
    
    # 타임스탬프 추가
    if current_time not in zscore_data['timestamps']:
        zscore_data['timestamps'].append(current_time)
        # 최대 1000개의 데이터 포인트만 유지
        if len(zscore_data['timestamps']) > 1000:
            zscore_data['timestamps'].pop(0)
    
    # 페어별 z-score 데이터 추가
    if pair not in zscore_data['pairs']:
        zscore_data['pairs'][pair] = []
    zscore_data['pairs'][pair].append(zscore)
    if len(zscore_data['pairs'][pair]) > 1000:
        zscore_data['pairs'][pair].pop(0)
    
    # zscore_st 데이터 추가
    if pair not in zscore_data['zscore_st']:
        zscore_data['zscore_st'][pair] = []
    zscore_data['zscore_st'][pair].append(zscore_st)
    if len(zscore_data['zscore_st'][pair]) > 1000:
        zscore_data['zscore_st'][pair].pop(0)

def collect_data(symbol, data_dir=DATA_DIR):
    """데이터 수집 함수: 파일이 없으면 6개월치 데이터를 수집하고, 있으면 마지막 인덱스부터 이어서 수집"""
    try:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        current_time = int(time.time() * 1000)
        
        # 파일이 존재하는 경우: 마지막 인덱스부터 이어서 수집
        if os.path.exists(file_path):
            df_existing = pd.read_csv(
                file_path,
                index_col='timestamp',
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
            )
            last_time = df_existing.index[-1]
            
            # 마지막 업데이트로부터 5분이 지났는지 확인
            if int(time.time() * 1000) - int(last_time.timestamp() * 1000) < 300000:
                logging.info(f"{symbol}: 최신 데이터 유지 중")
                return True
                
            start_time = int(last_time.timestamp() * 1000)
            logging.info(f"{symbol}: 마지막 데이터 시점({last_time})부터 이어서 수집")
        else:
            # 파일이 없는 경우: 6개월치 데이터 수집
            start_time = current_time - (6 * 30 * 24 * 60 * 60 * 1000)  # 6개월 전
            logging.info(f"{symbol}: 6개월치 데이터 수집 시작 ({datetime.fromtimestamp(start_time/1000)})")
        
        # 데이터 수집
        all_klines = []
        current_start = start_time
        
        while current_start < current_time:
            try:
                klines = client.futures_klines(
                    symbol=symbol,
                    interval='5m',
                    startTime=current_start,
                    limit=1000
                )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1
                time.sleep(1)  # API 제한 고려
                
            except BinanceAPIException as e:
                if e.code == -1021:  # 타임스탬프 오류
                    logging.warning(f"{symbol} 타임스탬프 오류, 다음 요청으로 진행")
                    current_start += 1000 * 5 * 60 * 1000
                    time.sleep(1)
                    continue
                else:
                    raise e
        
        if all_klines:
            # 새로운 데이터를 데이터프레임으로 변환
            df_new = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
            
            # 필요한 컬럼만 선택
            df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df_new.set_index('timestamp', inplace=True)
            
            # 기존 데이터와 새로운 데이터 병합
            if os.path.exists(file_path):
                df_updated = pd.concat([df_existing, df_new])
                df_updated = df_updated[~df_updated.index.duplicated(keep='last')]  # 중복 제거
                df_updated = df_updated.sort_index()  # 시간순 정렬
            else:
                df_updated = df_new
            
            # CSV 파일로 저장
            df_updated.to_csv(file_path, date_format='%Y-%m-%d %H:%M:%S')
            logging.info(f"{symbol}의 데이터를 저장했습니다. (새로운 데이터: {len(df_new)} 행)")
            return True
        else:
            logging.warning(f"{symbol}의 새로운 데이터가 없습니다.")
            return False
            
    except Exception as e:
        logging.error(f"{symbol} 데이터 수집 중 오류 발생: {e}")
        return False

def get_valid_symbols(data_dir=DATA_DIR):
    """과거 6개월간의 데이터가 있는 종목들을 반환"""
    try:
        symbols = []
        six_months_ago = datetime.utcnow() - timedelta(days=180)
        required_length = int(180 * 24 * 12 * 0.5)  # 6개월 * 24시간 * 12 (5분봉) * 0.5 = 절반 이상

        # 모든 USDT 선물 심볼 가져오기
        exchange_info = client.futures_exchange_info()
        all_symbols = [s['symbol'] for s in exchange_info['symbols'] if (s['status'] == 'TRADING') and (s['symbol'][-4:] == 'USDT')]
        
        for symbol in all_symbols:
            file_path = os.path.join(data_dir, f"{symbol}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(
                        file_path,
                        index_col='timestamp',
                        parse_dates=True,
                        date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
                    )
                    
                    # 6개월 이상의 데이터가 있는지 확인
                    if df.index.min() < six_months_ago:
                        # 6개월간의 데이터를 사용하여 평균 거래대금 계산
                        recent_df = df.loc[df.index >= six_months_ago]
                        if len(recent_df) >= required_length:
                            symbols.append(symbol)
                except Exception as e:
                    logging.error(f"{symbol} 데이터 검증 중 오류 발생: {e}")
                    continue

        logging.info(f"유효한 종목 개수: {len(symbols)}")
        return symbols
        
    except Exception as e:
        logging.error(f"유효한 종목 선정 중 오류 발생: {e}")
        return []

def update_data(data_dir=DATA_DIR):
    """모든 심볼에 대해 데이터 수집/업데이트 수행"""
    try:
        exchange_info = client.futures_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if (s['status'] == 'TRADING') and (s['symbol'][-4:] == 'USDT')]
        logging.info(f"업데이트할 심볼 개수: {len(symbols)}")
        
        for symbol in symbols:
            collect_data(symbol, data_dir)
            time.sleep(1)  # API 제한 고려
                
    except Exception as e:
        logging.error(f"데이터 업데이트 중 오류 발생: {e}")
        logging.error(f"Error details: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

# PairSelector 클래스
class PairSelector:
    def __init__(self, valid_sym, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.valid_sym = valid_sym
        self.prices = self.load_prices()
        self.pairs = self.generate_pairs()
        self.filtered_pairs_1 = []
        self.filtered_pairs_2 = []
        print(f"[PairSelector] 총 후보 페어 개수: {len(self.pairs)}")
    
    def load_prices(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        df_list = []
        for file in files:
            sym = file[:-4]

            if sym in self.valid_sym:
                symbol = file.replace('.csv', '')
                df = pd.read_csv(os.path.join(data_dir, file), index_col='timestamp', parse_dates=True)
                df.rename(columns={'close': symbol}, inplace=True)
                df_list.append(df[[symbol]])
            else:
                continue

        prices = pd.concat(df_list, axis=1).dropna()
        prices = np.log(prices)  # 로그 가격 변환
        return prices
    
    def generate_pairs(self):
        tickers = self.prices.columns.tolist()
        pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                pairs.append((tickers[i], tickers[j]))
        return pairs

    @staticmethod
    @numba.njit
    def _count_mean_reversion_events(zscore_arr, threshold=1, tol=0.5):
        count = 0
        in_extreme = False
        for i in range(len(zscore_arr)):
            val = zscore_arr[i]
            if not in_extreme:
                if np.abs(val) >= threshold:
                    in_extreme = True
            else:
                if np.abs(val) <= tol:
                    count += 1
                    in_extreme = False
        return count

    def filter_pairs_mr(self, pair):
        try:
            price1 = self.prices[pair[0]]
            price2 = self.prices[pair[1]]
            
            x = price2.values
            y = price1.values
            beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
            alpha = np.mean(y) - beta * np.mean(x)
            spread = price1 - beta * price2 - alpha

            # 전체 기간에 대해 z-score 계산
            zscore = (spread - spread.mean()) / spread.std()

            # z-score가 극단치(절대값 >=2)에 도달한 후 0(또는 tol 이하)로 회귀하는 이벤트 횟수 측정
            event_count = self._count_mean_reversion_events(zscore.to_numpy(), threshold=3.5, tol=1)
            if event_count > 0:
                return (pair, event_count)
        except Exception as e:
            print(f"Error in filter_pairs_mean_reversion for {pair}: {e}")
            return None

    def run(self):
        corr_matrix = self.prices.corr()
        tickers = self.prices.columns.tolist()
        corr_pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if abs(corr_matrix.loc[tickers[i], tickers[j]]) >= 0.6 and abs(corr_matrix.loc[tickers[i], tickers[j]]) <= 0.8:
                    corr_pairs.append((tickers[i], tickers[j]))
        print(f'첫 번째 필터 통과 페어 개수: {len(corr_pairs)}개')

        with parallel_backend("threading", n_jobs=-1):
            results2 = Parallel(verbose=1)(
                delayed(self.filter_pairs_mr)(pair) for pair in corr_pairs
            )

        results2 = [r for r in results2 if r is not None]
        # 이벤트 횟수(event_count)를 기준으로 내림차순 정렬하여 상위 30개 페어 선택
        results2_sorted = sorted(results2, key=lambda x: x[1], reverse=True)
        top_30 = results2_sorted[:30]
        top_30_list = []

        for pair, count in top_30:
            print(f'{pair} - 이벤트 횟수: {count}')
            top_30_list.append(pair)

        return top_30_list

# HalfLifeEstimator 클래스
class HalfLifeEstimator:
    def __init__(self, prices, pairs):
        self.prices = prices
        self.pairs = pairs
        self.pair_stats = {}
        
    def estimate_half_life(self, spread):
        delta_spread = spread.diff().dropna()  # ΔS_t = S_t - S_t-1 계산
        spread_lag = spread.shift(1).dropna()  # S_t-1 생성 (이전 시점의 스프레드)
        spread_lag, delta_spread = spread_lag.align(delta_spread, join='inner')  # ΔS_t와 S_t-1 맞춤

        # 회귀 분석 수행하여 θ 추정
        theta_model = sm.OLS(delta_spread, sm.add_constant(spread_lag)).fit()
        theta = -theta_model.params[0]  # θ 추정값 (음수 부호 주의)

        if theta > 0:
            halflife = np.log(2) / (theta) 
        else:
            halflife = np.nan

        return halflife

    def filter_pairs_mr(self, pair):
        try:
            price1 = self.prices[pair[0]]
            price2 = self.prices[pair[1]]
            x = price2.values
            y = price1.values
            beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
            alpha = np.mean(y) - beta * np.mean(x)
            spread = price1 - beta * price2 - alpha
            halflife = self.estimate_half_life(spread)

            if np.isfinite(halflife) and halflife > 0:
                return pair, {
                    'halflife': halflife, 
                    'alpha': alpha, 
                    'beta': beta, 
                    'spread': spread
                }
            else:
                return None, None
        except Exception as e:
            print(f"Error for pair {pair}: {e}")
            return None, None

    def select_pairs_graph(self, k):
        # 각 후보 페어에 대해 halflife, alpha, beta, spread 등을 계산
        with parallel_backend("threading", n_jobs=-1):
            results = Parallel(verbose=1)(
                delayed(self.filter_pairs_mr)(pair) for pair in self.pairs
            )

        matching_list = []
        for pair, stats_dict in results:
            if pair is not None and stats_dict is not None:
                beta_val = stats_dict['beta']
                if beta_val >= 0:
                    weight = stats_dict['halflife'] 
                    matching_list.append((pair, weight, stats_dict))

        print(f'halflife 필터에서 살아남은 페어의 개수는 {len(matching_list)}개입니다---------')

        matching_list = sorted(matching_list, key=lambda x: x[1])
        selected = matching_list[:k]
        selected_pairs = [(pair, stats) for (pair, weight, stats) in selected]
        return selected_pairs

# LiveTradingStrategy 클래스
class LiveTradingStrategy:
    def __init__(self, client, pair, alpha, beta, lookback, leverage=1, initial_capital=200, transaction_cost=0.0005, pre_spread_history=None):
        logging.info(f"LiveTradingStrategy 초기화 시작: {pair}")
        self.client = client
        self.pair = pair
        self.alpha = alpha
        self.beta = beta
        self.lookback = lookback
        self.leverage = leverage
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.capital = initial_capital
        self.positions = {'A': 0, 'B': 0}
        self.entry_prices = {'A': 0.0, 'B': 0.0}
        self.order_history = []
        self.pnl_history = []
        self.spread_history = pre_spread_history.copy() if pre_spread_history else []
        self.zscore_history = []
        
        logging.info(f"레버리지 설정 시작: {pair}")
        self.set_leverage()
        logging.info(f"레버리지 설정 완료: {pair}")

        if self.spread_history:
            logging.info(f"z-score 히스토리 초기화 시작: {pair}")
            self.initialize_zscore_history()
            logging.info(f"z-score 히스토리 초기화 완료: {pair}")
        
        logging.info(f"LiveTradingStrategy 초기화 완료: {pair}")

    def set_leverage(self):
        """
        선택된 페어에 대해 레버리지를 설정합니다.
        """
        symbol = self.pair[0]
        symbol2 = self.pair[1]
        try:
            response = self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            response2 = self.client.futures_change_leverage(symbol=symbol2, leverage=self.leverage)
            logging.info(f"레버리지 설정 성공: {symbol}({response}), {symbol2}({response2})")
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
        self.zscore_history.append(current_zscore)

        zscore_series = pd.Series(self.zscore_history)
        zscore_st = zscore_series.ewm(span=self.lookback, adjust=False).std()
        
        # z-score 데이터 업데이트
        update_zscore_data(self.pair, current_zscore, zscore_st.iloc[-1])

        return zscore_st

    def generate_signals(self, zscore_st):
        # 신호 임계값 설정
        upper_entry = zscore_st * 2
        lower_entry = -upper_entry
        upper_exit = zscore_st
        lower_exit = -upper_exit

        if len(self.zscore_history) < 3:
            return None

        current_z = self.zscore_history[-1]
        z_t1 = self.zscore_history[-2]
        z_t2 = self.zscore_history[-3]

        signal = None
        if z_t1 < lower_entry.iloc[-2] and z_t2 >= lower_entry[-3] and current_z < lower_entry.iloc[-1] and current_z > -5:
            signal = 'LONG'
        elif current_z > lower_exit.iloc[-1] and z_t1 > lower_exit.iloc[-2] and z_t2 <= lower_exit.iloc[-3]:
            signal = 'EXIT_LONG'
        elif z_t1 > upper_entry.iloc[-2] and z_t2 <= upper_entry.iloc[-3] and current_z > upper_entry.iloc[-1] and current_z < 5:
            signal = 'SHORT'
        elif current_z < upper_exit.iloc[-1] and z_t1 < upper_exit.iloc[-2] and z_t2 >= upper_exit.iloc[-3]:
            signal = 'EXIT_SHORT'

        return signal

    def get_binance_trades(self):
        """바이낸스에서 해당 페어의 거래 기록을 가져옵니다."""
        try:
            trades_A = self.client.futures_account_trades(symbol=self.pair[0])
            trades_B = self.client.futures_account_trades(symbol=self.pair[1])
            
            # 거래 기록을 시간순으로 정렬
            all_trades = []
            for trade in trades_A:
                trade['symbol'] = self.pair[0]
                all_trades.append(trade)
            for trade in trades_B:
                trade['symbol'] = self.pair[1]
                all_trades.append(trade)
            
            all_trades.sort(key=lambda x: x['time'])
            return all_trades
        except Exception as e:
            logging.error(f"바이낸스 거래 기록 조회 중 오류 발생: {e}")
            return []

    def get_binance_pnl(self):
        """바이낸스에서 해당 페어의 PnL을 가져옵니다."""
        try:
            pnl_A = float(self.client.futures_position_information(symbol=self.pair[0])[0]['unRealizedProfit'])
            pnl_B = float(self.client.futures_position_information(symbol=self.pair[1])[0]['unRealizedProfit'])
            return pnl_A + pnl_B
        except Exception as e:
            logging.error(f"바이낸스 PnL 조회 중 오류 발생: {e}")
            return 0.0

    def execute_order(self, signal, current_prices):
        symbol_A, symbol_B = self.pair
        price_A = current_prices[symbol_A]
        price_B = current_prices[symbol_B]
        timestamp = datetime.utcnow()

        logging.info(f"주문 실행 시작: {self.pair} {signal}")
        logging.info(f"현재 가격 - {symbol_A}: {price_A}, {symbol_B}: {price_B}")

        try:
            allocation_A = self.capital * (1 / (self.beta + 1))
            allocation_B = self.capital * (self.beta / (self.beta + 1))
            logging.info(f"자본 배분 - {symbol_A}: {allocation_A}, {symbol_B}: {allocation_B}")

            if signal == 'LONG':
                qty_A = allocation_A / price_A
                qty_B = allocation_B / price_B
                logging.info(f"롱 포지션 수량 - {symbol_A}: {qty_A}, {symbol_B}: {qty_B}")
                
                order_A = self.client.futures_create_order(symbol=symbol_A, side='BUY', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='SELL', type='MARKET', quantity=qty_B)

                self.positions['A'] += qty_A
                self.positions['B'] -= qty_B
                self.entry_prices['A'] = price_A
                self.entry_prices['B'] = price_B
                self.capital -= (allocation_A + allocation_B) * self.transaction_cost
                
                logging.info(f"롱 포지션 진입 완료: {self.pair}")
                logging.info(f"현재 포지션 - {symbol_A}: {self.positions['A']}, {symbol_B}: {self.positions['B']}")
                
                # 바이낸스 거래 기록 가져오기
                binance_trades = self.get_binance_trades()
                
                logging.info(f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}")

            elif signal == 'EXIT_LONG':
                qty_A = self.positions['A']
                qty_B = -self.positions['B']
                logging.info(f"롱 포지션 종료 수량 - {symbol_A}: {qty_A}, {symbol_B}: {qty_B}")
                
                order_A = self.client.futures_create_order(symbol=symbol_A, side='SELL', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='BUY', type='MARKET', quantity=qty_B)
                
                self.positions['A'] = 0
                self.positions['B'] = 0
                self.capital += (allocation_A + allocation_B) * (1 - self.transaction_cost)
                
                logging.info(f"롱 포지션 종료 완료: {self.pair}")
                logging.info(f"현재 포지션 - {symbol_A}: {self.positions['A']}, {symbol_B}: {self.positions['B']}")
                
                # 바이낸스 거래 기록 가져오기
                binance_trades = self.get_binance_trades()
                logging.info(f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}")

            elif signal == 'SHORT':
                qty_A = allocation_A / price_A
                qty_B = allocation_B / price_B
                order_A = self.client.futures_create_order(symbol=symbol_A, side='SELL', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='BUY', type='MARKET', quantity=qty_B)
                self.positions['A'] -= qty_A
                self.positions['B'] += qty_B
                self.entry_prices['A'] = price_A
                self.entry_prices['B'] = price_B
                self.capital -= (allocation_A + allocation_B) * self.transaction_cost
                self.order_history.append((timestamp, 'SHORT', qty_A, qty_B, price_A, price_B))
                
                # 바이낸스 거래 기록 가져오기
                binance_trades = self.get_binance_trades()
                
                logging.info(f"SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                logging.info(f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}")
                
                message_content = f"SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}\n"
                message_content += f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}"
                send_message_to_slack(message_content)

            elif signal == 'EXIT_SHORT':
                qty_A = -self.positions['A']
                qty_B = self.positions['B']
                order_A = self.client.futures_create_order(symbol=symbol_A, side='BUY', type='MARKET', quantity=qty_A)
                order_B = self.client.futures_create_order(symbol=symbol_B, side='SELL', type='MARKET', quantity=qty_B)
                self.positions['A'] = 0
                self.positions['B'] = 0
                self.capital += (allocation_A + allocation_B) * (1 - self.transaction_cost)
                self.order_history.append((timestamp, 'EXIT_SHORT', qty_A, qty_B, price_A, price_B))
                
                # 바이낸스 거래 기록 가져오기
                binance_trades = self.get_binance_trades()
                
                logging.info(f"EXIT_SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}")
                logging.info(f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}")
                
                message_content = f"EXIT_SHORT 주문 실행: {timestamp}, {qty_A} {symbol_A}, {qty_B} {symbol_B}, 가격 A: {price_A}, 가격 B: {price_B}\n"
                message_content += f"바이낸스 거래 기록: {binance_trades[-2:] if len(binance_trades) >= 2 else binance_trades}" 
                send_message_to_slack(message_content)

        except Exception as e:
            logging.error(f"주문 실행 중 오류 발생: {e}")
            message_content = f"주문 실행 중 오류 발생: {e}"
            send_message_to_slack(message_content)

    def track_pnl(self, current_prices):
        symbol_A, symbol_B = self.pair
        price_A = current_prices[symbol_A]
        price_B = current_prices[symbol_B]

        logging.info(f"PnL 추적 시작: {self.pair}")
        logging.info(f"현재 가격 - {symbol_A}: {price_A}, {symbol_B}: {price_B}")
        logging.info(f"진입 가격 - {symbol_A}: {self.entry_prices['A']}, {symbol_B}: {self.entry_prices['B']}")
        logging.info(f"현재 포지션 - {symbol_A}: {self.positions['A']}, {symbol_B}: {self.positions['B']}")

        pnl_A = (price_A - self.entry_prices['A']) * self.positions['A']
        pnl_B = (price_B - self.entry_prices['B']) * self.positions['B']
        total_pnl = pnl_A + pnl_B
        
        logging.info(f"PnL 계산 - {symbol_A}: {pnl_A}, {symbol_B}: {pnl_B}, 총 PnL: {total_pnl}")
        
        self.capital += total_pnl
        self.pnl_history.append((datetime.utcnow(), total_pnl))
        
        # 바이낸스 PnL 가져오기
        binance_pnl = self.get_binance_pnl()
        logging.info(f"바이낸스 PnL: {binance_pnl}")
        
        logging.info(f"PnL 추적 완료: {self.pair}")

    def run(self, new_price_data):
        try:
            print(f"LIVE TRADING STRATEGY run . . . .")

            price_A = new_price_data[self.pair[0]]
            price_B = new_price_data[self.pair[1]]
            spread = self.calculate_spread(price_A, price_B)
            self.update_spread_history(spread)

            zscore_st = self.calculate_zscore()
            signal = self.generate_signals(zscore_st)

            if signal:
                self.execute_order(signal, new_price_data)

            self.track_pnl(new_price_data)

        except BinanceAPIException as e:
            logging.error(f"Binance API Exception during run: {e}")
        except Exception as e:
            logging.error(f"Error during run: {e}")

# Scheduler 설정 및 주간 작업 등록
scheduler = BackgroundScheduler()
current_strategy = None  # 활성화된 전략을 저장할 전역 변수
def recurring_task():
    global current_strategy

    try:
        logging.info("=== recurring_task 시작 ===")

        # 기존 포지션 정리
        if current_strategy:
            logging.info("기존 포지션 정리 시작...")
            strategies_to_cleanup = current_strategy if isinstance(current_strategy, list) else [current_strategy]
            for strategy in strategies_to_cleanup:
                if any(pos != 0 for pos in strategy.positions.values()):
                    logging.info(f"기존 포지션 정리 중: {strategy.pair}")
                    # 현재 가격 가져오기
                    symbol_A, symbol_B = strategy.pair
                    ticker_A = client.futures_symbol_ticker(symbol=symbol_A)
                    ticker_B = client.futures_symbol_ticker(symbol=symbol_B)
                    current_prices = {
                        symbol_A: float(ticker_A['price']),
                        symbol_B: float(ticker_B['price'])
                    }
                    logging.info(f"현재 가격 - {symbol_A}: {current_prices[symbol_A]}, {symbol_B}: {current_prices[symbol_B]}")
                    
                    # 포지션 종료
                    if strategy.positions['A'] > 0:
                        logging.info(f"{strategy.pair} 롱 포지션 종료")
                        strategy.execute_order('EXIT_LONG', current_prices)
                    elif strategy.positions['A'] < 0:
                        logging.info(f"{strategy.pair} 숏 포지션 종료")
                        strategy.execute_order('EXIT_SHORT', current_prices)
                    strategy.track_pnl(current_prices)
                    logging.info(f"{strategy.pair} 포지션 정리 완료")

        # Step 1: 데이터 업데이트
        logging.info("데이터 업데이트 시작...")
        update_data(data_dir=DATA_DIR)
        logging.info("데이터 업데이트 완료")

        # Step 2: 유효한 종목 선정
        logging.info("유효한 종목 선정 시작...")
        valid_sym = get_valid_symbols(data_dir=DATA_DIR)
        logging.info(f"유효한 종목 개수: {len(valid_sym)}")
        logging.info(f"유효한 종목 목록: {valid_sym[:5]}... (총 {len(valid_sym)}개)")

        if len(valid_sym) == 0:
            logging.warning("유효한 종목이 없어 종목을 선정할 수 없습니다.")
            return
        
        # Step 3: 페어 선정
        logging.info("페어 선정 시작...")
        pair_selector = PairSelector(valid_sym, data_dir=DATA_DIR)
        filtered_pairs = pair_selector.run()
        logging.info(f"선정된 페어 개수: {len(filtered_pairs)}")
        logging.info(f"선정된 페어 목록: {filtered_pairs[:5]}... (총 {len(filtered_pairs)}개)")
        
        if not filtered_pairs:
            logging.warning("이번 달에 적합한 페어를 찾지 못했습니다.")
            return
        
        # Step 4: 반감기 추정 및 페어 선택
        logging.info("반감기 추정 및 최종 페어 선택 시작...")
        half_life_estimator = HalfLifeEstimator(pair_selector.prices, filtered_pairs)
        selected_pairs = half_life_estimator.select_pairs_graph(k=5)
        logging.info(f"최종 선택된 페어 개수: {len(selected_pairs)}")
        
        if not selected_pairs:
            logging.warning("이번 주에 적합한 페어를 찾지 못했습니다.")
            return
        
        # LiveTradingStrategy 객체 생성
        logging.info("거래 전략 객체 생성 시작...")
        strategies = []
        for pair, stats in selected_pairs:
            logging.info(f"전략 생성 중: {pair}")
            logging.info(f"반감기: {stats['halflife']}, 알파: {stats['alpha']}, 베타: {stats['beta']}")
            
            strategy = LiveTradingStrategy(
                client=client,
                pair=pair,
                alpha=stats['alpha'],
                beta=stats['beta'],
                lookback=int(stats['halflife']),
                leverage=1,
                initial_capital=20,
                transaction_cost=0.0005,
                pre_spread_history=stats['spread'].tolist()
            )
            strategies.append(strategy)
            logging.info(f"전략 생성 완료: {pair}")
        
        # 현재 전략들을 리스트로 교체
        current_strategy = strategies
        logging.info("=== recurring_task 완료 ===")
        
    except Exception as e:
        logging.error(f"recurring_task에서 오류 발생: {e}")
        logging.error(f"Error details: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

# 주간 작업 등록 (매주 월요일 00:00 UTC에 실행)
scheduler.add_job(recurring_task, 'cron', day_of_week='mon', hour=0, minute=0)
scheduler.start()

# 실시간 실행 루프
def real_time_execution():
    global current_strategy
    active_strategies = []  # 현재 활성화된 전략들을 저장하는 리스트
    asset_positions = {}    # 각 자산별 현재 포지션을 추적하는 딕셔너리

    while True:
        logging.info(f"REAL TIME EXECUTION에 입성. . . . .")

        now = datetime.now()
        next_minute = (now.minute // 5 + 1) * 5

        if next_minute == 60:
            next_minute = 0
            next_time = (now + timedelta(hours=1)).replace(minute=next_minute, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)

        sleep_time = (next_time - now).total_seconds()
        time.sleep(sleep_time)

        try:
            if current_strategy:
                # 현재 활성화된 전략들을 active_strategies에 추가
                if isinstance(current_strategy, list):
                    active_strategies = current_strategy
                else:
                    active_strategies = [current_strategy]
                
                logging.info(f"현재 활성화된 전략 수: {len(active_strategies)}")
                for strategy in active_strategies:
                    logging.info(f"활성화된 페어: {strategy.pair}")

                # 모든 페어의 현재 가격 가져오기
                all_prices = {}
                for strategy in active_strategies:
                    symbol_A, symbol_B = strategy.pair
                    if symbol_A not in all_prices:
                        ticker_A = client.futures_symbol_ticker(symbol=symbol_A)
                        all_prices[symbol_A] = float(ticker_A['price'])
                        logging.info(f"{symbol_A} 현재 가격: {all_prices[symbol_A]}")
                    if symbol_B not in all_prices:
                        ticker_B = client.futures_symbol_ticker(symbol=symbolB)
                        all_prices[symbol_B] = float(ticker_B['price'])
                        logging.info(f"{symbol_B} 현재 가격: {all_prices[symbol_B]}")

                # 각 전략의 시그널 수집
                all_signals = {}
                for strategy in active_strategies:
                    symbol_A, symbol_B = strategy.pair
                    new_price_data = {
                        symbol_A: all_prices[symbol_A],
                        symbol_B: all_prices[symbol_B]
                    }
                    
                    # 스프레드 계산 및 시그널 생성
                    spread = strategy.calculate_spread(new_price_data[symbol_A], new_price_data[symbol_B])
                    strategy.update_spread_history(spread)
                    zscore_st = strategy.calculate_zscore()
                    signal = strategy.generate_signals(zscore_st)
                    
                    if signal:
                        all_signals[strategy.pair] = signal
                        logging.info(f"페어 {strategy.pair}에서 시그널 발생: {signal}")

                # 1. 동시에 발생한 상반된 시그널 확인 (진입 시그널만)
                entry_conflicts = set()
                for pair1, signal1 in all_signals.items():
                    if signal1 in ['LONG', 'SHORT']:  # 진입 시그널만 확인
                        for pair2, signal2 in all_signals.items():
                            if pair1 != pair2 and signal2 in ['LONG', 'SHORT']:
                                # 공통 자산 확인
                                common_assets = set(pair1) & set(pair2)
                                if common_assets:
                                    for asset in common_assets:
                                        # 각 자산의 의도된 방향 결정
                                        intended1 = signal1 if asset == pair1[0] else -signal1
                                        intended2 = signal2 if asset == pair2[0] else -signal2
                                        if intended1 != intended2:
                                            entry_conflicts.add(pair1)
                                            entry_conflicts.add(pair2)
                                            logging.info(f"동시 발생 충돌 감지: {pair1}({signal1})와 {pair2}({signal2})의 {asset} 자산")

                # 2. 기존 포지션과의 충돌 확인
                for strategy in active_strategies:
                    if strategy.pair in all_signals:
                        signal = all_signals[strategy.pair]
                        if signal in ['LONG', 'SHORT']:  # 진입 시그널만 확인
                            symbol_A, symbol_B = strategy.pair
                            logging.info(f"페어 {strategy.pair}의 시그널 {signal} 처리 중...")
                            
                            # 각 자산의 현재 포지션 확인
                            for asset in [symbol_A, symbol_B]:
                                if asset in asset_positions:
                                    existing_position = asset_positions[asset]
                                    intended_direction = signal if asset == symbol_A else -signal
                                    logging.info(f"자산 {asset}의 현재 포지션: {existing_position}, 의도된 방향: {intended_direction}")
                                    
                                    if existing_position != 0 and existing_position != intended_direction:
                                        logging.info(f"자산 {asset}에서 포지션 충돌 감지")
                                        
                                        # 기존 포지션 종료
                                        for other_strategy in active_strategies:
                                            if asset in other_strategy.pair:
                                                logging.info(f"기존 포지션 종료 중: {other_strategy.pair}")
                                                
                                                # 현재 가격 데이터 준비
                                                current_prices = {
                                                    other_strategy.pair[0]: all_prices[other_strategy.pair[0]],
                                                    other_strategy.pair[1]: all_prices[other_strategy.pair[1]]
                                                }
                                                
                                                # 포지션 종료
                                                if other_strategy.positions['A'] > 0:
                                                    logging.info(f"{other_strategy.pair} 롱 포지션 종료")
                                                    other_strategy.execute_order('EXIT_LONG', current_prices)
                                                elif other_strategy.positions['A'] < 0:
                                                    logging.info(f"{other_strategy.pair} 숏 포지션 종료")
                                                    other_strategy.execute_order('EXIT_SHORT', current_prices)
                                                other_strategy.track_pnl(current_prices)
                                                
                                                # asset_positions 업데이트
                                                asset_positions[other_strategy.pair[0]] = 0
                                                asset_positions[other_strategy.pair[1]] = 0
                                                logging.info(f"{other_strategy.pair} 포지션 정보 초기화 완료")
                                        
                                        # 새로운 포지션 진입
                                        new_prices = {
                                            symbol_A: all_prices[symbol_A],
                                            symbol_B: all_prices[symbol_B]
                                        }
                                        logging.info(f"새로운 포지션 진입: {strategy.pair} {signal}")
                                        strategy.execute_order(signal, new_prices)
                                        strategy.track_pnl(new_prices)
                                        
                                        # 새로운 포지션 정보 업데이트
                                        asset_positions[symbol_A] = 1 if signal == 'LONG' else -1
                                        asset_positions[symbol_B] = -1 if signal == 'LONG' else 1
                                        logging.info(f"새로운 포지션 정보 업데이트 완료: {strategy.pair}")
                                        
                                        # 이미 처리된 시그널이므로 루프 종료
                                        break

                # 충돌이 없는 시그널만 실행
                for strategy in active_strategies:
                    if strategy.pair in all_signals and strategy.pair not in entry_conflicts:
                        signal = all_signals[strategy.pair]
                        symbol_A, symbol_B = strategy.pair
                        new_price_data = {
                            symbol_A: all_prices[symbol_A],
                            symbol_B: all_prices[symbol_B]
                        }
                        logging.info(f"충돌 없는 시그널 실행: {strategy.pair} {signal}")
                        strategy.execute_order(signal, new_price_data)
                        strategy.track_pnl(new_price_data)
                        
                        # 포지션 업데이트
                        if signal in ['LONG', 'SHORT']:
                            asset_positions[symbol_A] = 1 if signal == 'LONG' else -1
                            asset_positions[symbol_B] = -1 if signal == 'LONG' else 1
                            logging.info(f"포지션 업데이트: {strategy.pair} {signal}")
                        elif signal in ['EXIT_LONG', 'EXIT_SHORT']:
                            asset_positions[symbol_A] = 0
                            asset_positions[symbol_B] = 0
                            logging.info(f"포지션 종료: {strategy.pair}")
                    elif strategy.pair in entry_conflicts:
                        logging.info(f"충돌로 인해 {strategy.pair} 페어의 시그널 실행이 건너뜁니다.")

            else:
                logging.info("활성화된 거래 전략이 없습니다.")
        except BinanceAPIException as e:
            logging.error(f"Binance API Exception during real_time_execution: {e}")
        except Exception as e:
            logging.error(f"Error during real_time_execution: {e}")
            logging.error(f"Error details: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")

#메인 실행 블록
if __name__ == "__main__":
    # 1. 데이터 디렉토리의 CSV 파일 개수 확인
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    logging.info("초기 데이터 수집 및 업데이트를 실행합니다.")
    update_data()  # 모든 심볼에 대해 데이터 수집
    
    # 2. recurring_task 함수를 즉시 실행
    recurring_task()

    # 3. real_time_execution 함수를 별도의 스레드에서 실행
    execution_thread = threading.Thread(target=real_time_execution, daemon=True)
    execution_thread.start()

    # 4. 스케줄러 설정 (매월 1일 00:00 UTC에 실행)
    if not scheduler.running:
        scheduler.add_job(recurring_task, 'cron', day=1, hour=0, minute=0)
        scheduler.start()
        logging.info("스케줄러가 시작되었습니다.")

    # 5. 메인 스레드가 종료되지 않도록 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Trading bot을 종료합니다.")
        scheduler.shutdown()  # 스케줄러 종료