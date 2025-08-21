from stable_baselines3 import PPO
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.utils import dropna
import itertools
import pandas as pd
import numpy as np
from typing import List
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
import statistics
import time
import logging
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from pypfopt import expected_returns, risk_models
from pypfopt import EfficientFrontier
from scipy.stats import skew, kurtosis

from typing import Optional

import json
from glob import glob
import matplotlib.dates as mdates

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2022-12-31'

TRADE_START_DATE= '2023-01-01'
TRADE_END_DATE = '2025-8-20'

DOW_30_TICKER = [
    "AAPL",
    "BA",
    "GS",
    "JPM",
]

INDICATORS_DOW_30 = [
    # --- Trend Indicators ---
    # "ema_20",        # 20-day Exponential Moving Average, shows short-term trend
    # "ema_60",        # 60-day Exponential Moving Average, shows medium-term trend
    # "macd",          # Moving Average Convergence Divergence, trend-following momentum indicator
    # "macd_signal",   # Signal line of MACD, used to identify buy/sell crossovers

    # --- Momentum Indicators ---
    "rsi_14",        # 14-day Relative Strength Index, measures overbought/oversold conditions
    # "cci_14",        # 14-day Commodity Channel Index, measures deviation from average price
    "mom_20d",       # 20-day Momentum, simple measure of price change over 20 days

    # --- Volatility Indicators ---
    # "atr_14",        # 14-day Average True Range, measures market volatility
    "boll_width",    # Bollinger Band width, indicates volatility expansion or contraction
    # "boll_lb",       # Bollinger lower band, often used for support or oversold conditions

    # --- Volume Indicators ---
    # "obv",           # On-Balance Volume, cumulative volume to confirm trends
    # "vol_sma_20",    # 20-day simple moving average of volume, smooths daily volume

    # --- Market Sentiment / External Indicators ---
    # "vix",           # Volatility Index, measures market expectations of near-term volatility
    # "vix_rolling_30",# 30-day rolling average of VIX, smooths volatility trend
    "finbert_future",# Sentiment score from FinBERT model for future stock news
    "fingpt_future", # Sentiment or prediction from GPT-based financial model for future

    # --- Price / Return Indicators ---
    "ret_1d",        # 1-day return, simple daily percentage change
    # "ret_5d",        # 5-day return, short-term performance
    "zclose_60",     # Z-score of last 60 close prices, indicates relative price deviation
    "rank_mom20",    # Ranked 20-day momentum across universe, relative strength measure
    "rank_rsi_mom",  # Ranked combination of RSI and momentum, relative trend signal
    "rank_rsi_mr",   # Ranked RSI mean reversion score, for detecting overbought/oversold
]

PPO_PARAMS = {
    "policy": "MlpPolicy",
    "n_steps": 512,             # longer rollouts → better trajectory understanding
    "batch_size": 128,
    "gamma": 0.99,
    "learning_rate": 1e-4,     # consider lr_schedule
    "ent_coef": 0.035,           # less random trades
    "clip_range": 0.2,
    "n_epochs": 8,              # more updates per rollout
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
}

DEFAULT_PARAMS = {
    "ppo": PPO_PARAMS,
}

DEFAULT_PARAMS_POLICY = {
    "ppo": {
        'net_arch': [dict(vf=[64, 64], pi=[64, 64])]
    }
}

MODELS = {"ppo": PPO}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

params = {
    "ind_train": INDICATORS_DOW_30,
    "hmax_train": 100,
    "initial_amount_train": 1_000_000,
    "reward_scaling_train": 10.0,   # scale = 1 because we normalized reward
    "hmax_test": 100,
    "initial_amount_test": 1_000_000,
    "reward_scaling_test": 10.0,
}

def _adjust_prices(data_df: pd.DataFrame) -> pd.DataFrame:
    # use adjusted close price instead of close price
    data_df["adj"] = data_df["adjcp"] / data_df["close"]
    for col in ["open", "high", "low", "close"]:
        data_df[col] *= data_df["adj"]

    # drop the adjusted close price column
    return data_df.drop(["adjcp", "adj"], axis=1)

def get_data(start_date, end_date):
    """Fetches data from Yahoo API
    Returns
    -------
    `pd.DataFrame`
        7 columns: A date, open, high, low, close, volume and tick symbol
        for the specified stock ticker
    """
    # Download and save the data in a pandas DataFrame:
    data_df = pd.DataFrame()
    num_failures = 0
    for tic in DOW_30_TICKER:
        temp_df = yf.download(
            tic,
            start=start_date,
            end=end_date,
            auto_adjust=False,
        )
        if temp_df.columns.nlevels != 1:
            temp_df.columns = temp_df.columns.droplevel(1)
        temp_df["tic"] = tic
        if len(temp_df) > 0:
            data_df = pd.concat([data_df, temp_df], axis=0)
        else:
            num_failures += 1

    if num_failures == len(DOW_30_TICKER):
        raise ValueError("no data is fetched.")

    # reset the index, we want to use numbers as index instead of dates
    data_df = data_df.reset_index()
    try:
        # convert the column names to standardized names
        data_df.rename(
            columns={
                "Date": "date",
                "Adj Close": "adjcp",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
                "Open": "open",
                "tic": "tic",
            },
            inplace=True,
        )

        data_df = _adjust_prices(data_df)
        
    except NotImplementedError:
        print("the features are not supported currently")

    # create day of the week column (monday = 0)
    data_df["day"] = data_df["date"].dt.dayofweek

    # drop missing data
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    print("Shape of DataFrame: ", data_df.shape)
    # print("Display DataFrame: ", data_df.head())

    data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    return data_df

def clean_data(data):
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    return df

def add_indicators(df):
    df = df.sort_values(['tic', 'date']).reset_index(drop=True)

    df_indicators = pd.DataFrame()

    for tic, group in df.groupby('tic'):
        group = group.copy()

        # --- Trend indicators ---
        group['ema_20'] = EMAIndicator(group['close'], window=20).ema_indicator()
        group['ema_60'] = EMAIndicator(group['close'], window=60).ema_indicator()
        macd = MACD(group['close'])
        group['macd'] = macd.macd()
        group['macd_signal'] = macd.macd_signal()

        # --- Momentum indicators ---
        group['rsi_14'] = RSIIndicator(group['close'], window=14).rsi()
        group['cci_14'] = CCIIndicator(group['high'], group['low'], group['close'], window=14).cci()

        # --- Volatility indicators ---
        boll = BollingerBands(close=group['close'], window=20, window_dev=2)
        group['boll_ub'] = boll.bollinger_hband()
        group['boll_lb'] = boll.bollinger_lband()
        group['boll_width'] = group['boll_ub'] - group['boll_lb']
        group['atr_14'] = AverageTrueRange(group['high'], group['low'], group['close'], window=14).average_true_range()

        # --- Volume indicators ---
        group['obv'] = OnBalanceVolumeIndicator(group['close'], group['volume']).on_balance_volume()
        group['vol_sma_20'] = group['volume'].rolling(window=20).mean()

        # --- Price-based features ---
        group['ret_1d']  = group['close'].pct_change(1)
        group['ret_5d']  = group['close'].pct_change(5)
        group['mom_20d'] = group['close'].pct_change(20)

        roll_mean = group['close'].rolling(60).mean()
        roll_std  = group['close'].rolling(60).std()
        group['zclose_60'] = (group['close'] - roll_mean) / (roll_std + 1e-8)

        df_indicators = pd.concat([df_indicators, group], axis=0)

    # --- Final sorting and cleanup ---
    df_indicators = df_indicators.replace([np.inf, -np.inf], np.nan).dropna()
    df_indicators = df_indicators.sort_values(['tic', 'date']).reset_index(drop=True)

    # --- Cross-sectional ranks ---
    by_date = df_indicators.groupby('date')
    df_indicators['rank_mom20']    = by_date['mom_20d'].rank(pct=True, method='first')
    df_indicators['rank_rsi_mom']  = by_date['rsi_14'].rank(pct=True, method='first')
    df_indicators['rank_rsi_mr']   = 1.0 - df_indicators['rank_rsi_mom']

    # --- Ensure all INDICATORS_DOW_30 columns exist ---
    for col in INDICATORS_DOW_30:
        if col not in df_indicators.columns:
            df_indicators[col] = np.nan

    # --- Drop columns not in original + INDICATORS_DOW_30 ---
    original_cols = [c for c in df.columns]  # keep original columns
    print(original_cols)
    cols_to_keep = original_cols + INDICATORS_DOW_30
    df_indicators = df_indicators[cols_to_keep]

    return df_indicators

def add_vix(data):
    """
    Add VIX data from Yahoo Finance using yfinance.

    :param data: (pd.DataFrame) Must have a 'date' column (datetime or string).
    :return: (pd.DataFrame) Original dataframe with 'vix' column added.
    """
    df = data.copy()

    # Ensure 'date' is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Download VIX data
    vix_df = yf.download(
        "^VIX",
        start=df["date"].min().strftime("%Y-%m-%d"),
        end=(df["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False
    )

    # Reset index to get 'Date' as a column
    vix_df = vix_df.reset_index()[["Date", "Close"]]
    vix_df.columns = ["date", "vix"]

    # Merge with original dataframe
    df = df.merge(vix_df, on="date", how="left")

    # Sort
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    return df

def calculate_turbulence(data):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    df = data.copy()
    df_price_pivot = df.pivot(index="date", columns="tic", values="close")

    # use returns to calculate turbulence
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.date.unique()
    # start after a year
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        # use one year rolling window to calcualte covariance
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - 252])
        ]
        # Drop tickers which has number missing values more than the "oldest" ticker
        filtered_hist_price = hist_price.iloc[
            hist_price.isna().sum().min() :
        ].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0
        )
        # cov_temp = hist_price.cov()
        # current_temp=(current_price - np.mean(hist_price,axis=0))

        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
            current_temp.values.T
        )
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    try:
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
    except ValueError:
        raise Exception("Turbulence information could not be added.")
    return turbulence_index

def add_turbulence(data):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = data.copy()
    turbulence_index = calculate_turbulence(df)
    df = df.merge(turbulence_index, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    return df

def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur.
                                      If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_hard_threshold=None,
        turbulence_soft_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=1,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        log_verbose: bool = True,  # NEW FLAG
    ):
        # ---------------------------
        # 📝 Logger setup
        # ---------------------------
        import logging
        import os

        log_file_path = "." + "trading_env.log"

        self.logger = logging.getLogger("StockTradingEnv")
        self.logger.setLevel(logging.DEBUG if log_verbose else logging.CRITICAL)

        # Prevent printing to console
        self.logger.propagate = False

        if not self.logger.handlers:  # avoid duplicate handlers
            # File handler (append mode to keep history)
            fh = logging.FileHandler(log_file_path, mode="a")
            fh.setLevel(logging.DEBUG if log_verbose else logging.CRITICAL)

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.logger.info("\n[INIT] ===== StockTradingEnv =====")

        # ---------------------------
        # 🔧 Core attributes (as before)
        # ---------------------------
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.init_num_stock_shares = num_stock_shares.copy()
        self.num_stock_shares = num_stock_shares.copy()
        self.initial_amount = initial_amount
        self.cash = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # ---------------------------
        # 🧪 Dataset sanity logs
        # ---------------------------
        self.logger.info(f"[INIT] df.shape = {self.df.shape}")
        self.logger.info(f"[INIT] df.index.type = {type(self.df.index)}")
        try:
            head_idx = self.df.index[:5].tolist()
        except Exception:
            head_idx = str(self.df.index)[:80]
        self.logger.info(f"[INIT] df.index[:5] = {head_idx}")
        self.logger.info(f"[INIT] df.columns = {list(self.df.columns)}")

        # Basic column checks
        if "date" not in self.df.columns:
            self.logger.warning("[WARN] 'date' column not found in df! Episode length/done logic may break.")
        if "close" not in self.df.columns:
            self.logger.warning("[WARN] 'close' column not found in df! Pricing/portfolio logic will break.")
        if "tic" not in self.df.columns:
            self.logger.warning("[WARN] 'tic' column not found in df. Multi-asset checks will be limited.")

        # Count unique dates and tickers (if available)
        if "date" in self.df.columns:
            self.n_days = len(self.df["date"].unique())
            self.logger.info(f"[INIT] unique trading days (n_days) = {self.n_days}")
        else:
            self.n_days = len(self.df.index.unique())
            self.logger.info(f"[INIT] 'date' missing; fallback n_days via index.unique() = {self.n_days} (⚠️ may be wrong)")

        if "tic" in self.df.columns:
            n_tics = int(self.df["tic"].nunique())
            self.logger.info(f"[INIT] unique tickers (n_tics) = {n_tics}, stock_dim (arg) = {self.stock_dim}")
            if n_tics != self.stock_dim:
                self.logger.warning(f"[WARN] stock_dim ({self.stock_dim}) != unique tickers ({n_tics}). "
                                    f"State/action sizes or price vector lengths may mismatch.")
        else:
            self.logger.info(f"[INFO] stock_dim (arg) = {self.stock_dim} (no 'tic' to verify).")

        # ---------------------------
        # 🎯 Spaces
        # ---------------------------
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.logger.info(f"[INIT] action_space.shape = {self.action_space.shape}")
        self.logger.info(f"[INIT] observation_space.shape = {self.observation_space.shape}")

        # ---------------------------
        # 📅 Initial day slice
        # ---------------------------
        self.data = self.df.loc[self.day, :]
        try:
            first_date = (self.data["date"].iloc[0]
                          if hasattr(self.data["date"], "iloc")
                          else self.data["date"])
        except Exception:
            first_date = "<?>"
        self.logger.info(f"[INIT] starting day = {self.day}, starting date = {first_date}")

        # ---------------------------
        # ✅ Thresholds (turbulence)
        # ---------------------------
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.risk_indicator_col = risk_indicator_col

        if turbulence_soft_threshold is None or turbulence_hard_threshold is None:
            if self.risk_indicator_col in self.df.columns:
                values = self.df[self.risk_indicator_col].values
                q95 = np.nanpercentile(values, 95)
                q995 = np.nanpercentile(values, 99.5)
                self.turbulence_soft_threshold = (turbulence_soft_threshold if turbulence_soft_threshold else q95)
                self.turbulence_hard_threshold = (turbulence_hard_threshold if turbulence_hard_threshold else q995)
                self.logger.info(f"[INIT] turbulence thresholds -> soft={self.turbulence_soft_threshold:.4f}, "
                                 f"hard={self.turbulence_hard_threshold:.4f} (from percentiles 95/99.5)")
            else:
                self.turbulence_soft_threshold = turbulence_soft_threshold
                self.turbulence_hard_threshold = turbulence_hard_threshold
                self.logger.warning(f"[WARN] risk_indicator_col '{self.risk_indicator_col}' not in df; "
                                    f"thresholds left as provided: soft={self.turbulence_soft_threshold}, "
                                    f"hard={self.turbulence_hard_threshold}")
        else:
            self.turbulence_soft_threshold = turbulence_soft_threshold
            self.turbulence_hard_threshold = turbulence_hard_threshold
            self.logger.info(f"[INIT] turbulence thresholds (user) -> soft={self.turbulence_soft_threshold}, "
                             f"hard={self.turbulence_hard_threshold}")

        # ---------------------------
        # 🔁 Previous state bookkeeping
        # ---------------------------
        self.initial = initial
        self.previous_state = previous_state
        self.previous_cash = initial_amount
        self.previous_num_stock_shares = num_stock_shares.copy()
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # ---------------------------
        # 🧮 Build initial state
        # ---------------------------
        self.state = self._initiate_state()
        state_len = len(self.state) if hasattr(self.state, "__len__") else None
        self.logger.info(f"[INIT] initial state length = {state_len}, expected state_space = {self.state_space}")

        try:
            expected_len = 1 + self.stock_dim + self.stock_dim * len(self.tech_indicator_list)
            if state_len is not None and state_len != expected_len:
                self.logger.warning(f"[WARN] state length ({state_len}) != heuristic expected ({expected_len}). "
                                    f"Check tech indicators per ticker and state assembly.")
        except Exception:
            pass

        # ---------------------------
        # 📈 Memories & counters
        # ---------------------------
        self.return_log = []
        self.vol_log = []

        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.step_cost = 0

        try:
            prices = self.data["close"].values
            if "tic" in self.df.columns:
                self.logger.info(f"[INIT] prices vector length (close) = {len(prices)} (should match stock_dim={self.stock_dim})")
                if len(prices) != self.stock_dim:
                    self.logger.warning(f"[WARN] len(close[{first_date}]) = {len(prices)} != stock_dim ({self.stock_dim}). "
                                        f"Check day slicing and multi-asset alignment.")
        except Exception as e:
            self.logger.error(f"[ERR] Could not extract initial 'close' prices: {e}")
            prices = np.zeros(self.stock_dim, dtype=float)

        self.asset_memory = [
            self.initial_amount + np.sum(np.array(self.num_stock_shares) * prices)
        ]
        self.logger.info(f"[INIT] initial cash = {self.initial_amount:.2f}, "
                         f"initial holdings value = {float(np.sum(np.array(self.num_stock_shares) * prices)):.2f}, "
                         f"initial total asset = {self.asset_memory[0]:.2f}")

        # Sanity checks for list lengths
        def _check_len(name, arr, target):
            try:
                ln = len(arr)
            except Exception:
                ln = None
            ok = (ln == target)
            self.logger.info(f"[INIT] len({name}) = {ln}  {'OK' if ok else f'≠ {target} (WARN)'}")
            return ok

        _check_len("num_stock_shares", self.num_stock_shares, self.stock_dim)
        _check_len("buy_cost_pct", self.buy_cost_pct, self.stock_dim)
        _check_len("sell_cost_pct", self.sell_cost_pct, self.stock_dim)

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.logger.info(f"[INIT] first date_memory entry = {self.date_memory[0]}")
        self.logger.info("[INIT] ===== Init complete =====\n")

        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            sell_num_shares = 0
            if self.num_stock_shares[index] > 0:
                sell_num_shares = min(abs(action), self.num_stock_shares[index])

                prices = self.data['close'].values
                sell_amount = prices[index] * sell_num_shares * (1 - self.sell_cost_pct[index])

                # update balance
                self.cash += sell_amount
                self.num_stock_shares[index] -= sell_num_shares
                self.cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.step_cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1

                # Logs
                self.logger.info(f"[SELL_NORMAL] index={index}, action={action}, sell_num_shares={sell_num_shares}")
                self.logger.info(f"[SELL_NORMAL] price={prices[index]:.2f}, sell_amount={sell_amount:.2f}")
                self.logger.info(f"[SELL_NORMAL] updated cash={self.cash:.2f}, "
                                 f"num_stock_shares={self.num_stock_shares[index]}, "
                                 f"cost={self.cost:.2f}, trades={self.trades}")

                # Warnings
                if self.cash < 0:
                    self.logger.warning(f"[WARNING] Cash is negative after selling! cash={self.cash:.2f}")
                if sell_num_shares > self.num_stock_shares[index] + sell_num_shares:
                    self.logger.error(f"[ERROR] Sold more shares than available! "
                                      f"index={index}, sell_num_shares={sell_num_shares}, "
                                      f"num_stock_shares={self.num_stock_shares[index]}")
            else:
                self.logger.info(f"[SELL_NORMAL] index={index}, action={action}, no shares to sell (num_stock_shares=0)")
            return sell_num_shares

        # turbulence logic
        if self.turbulence_hard_threshold is not None and self.turbulence >= self.turbulence_hard_threshold:
            if self.num_stock_shares[index] > 0:
                prices = self.data['close'].values
                sell_num_shares = self.num_stock_shares[index]
                sell_amount = prices[index] * sell_num_shares * (1 - self.sell_cost_pct[index])

                self.cash += sell_amount
                self.num_stock_shares[index] = 0
                self.cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.step_cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1

                self.logger.info(f"[SELL_TURBULENCE] Hard threshold triggered, index={index}, forced sell all")
                self.logger.info(f"[SELL_TURBULENCE] sell_num_shares={sell_num_shares}, "
                                 f"price={prices[index]:.2f}, sell_amount={sell_amount:.2f}")
                self.logger.info(f"[SELL_TURBULENCE] updated cash={self.cash:.2f}, "
                                 f"num_stock_shares={self.num_stock_shares[index]}, "
                                 f"cost={self.cost:.2f}, trades={self.trades}")

                if self.cash < 0:
                    self.logger.warning(f"[WARNING] Cash is negative after forced sell! cash={self.cash:.2f}")
            else:
                sell_num_shares = 0
                self.logger.info(f"[SELL_TURBULENCE] index={index}, no shares to sell")
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares



    def _buy_stock(self, index, action):
        def _do_buy():
            prices = self.data['close'].values

            # Compute max shares we can buy considering cost
            available_amount = self.cash // (prices[index] * (1 + self.buy_cost_pct[index]))
            buy_num_shares = min(available_amount, action)
            buy_amount = prices[index] * buy_num_shares * (1 + self.buy_cost_pct[index])

            # Update balance and shares
            self.cash -= buy_amount
            self.num_stock_shares[index] += buy_num_shares
            self.cost += prices[index] * buy_num_shares * self.buy_cost_pct[index]
            self.step_cost += prices[index] * buy_num_shares * self.buy_cost_pct[index]
            self.trades += 1

            # Logs
            self.logger.info(f"[BUY] index={index}, action={action}, buy_num_shares={buy_num_shares}")
            self.logger.info(f"[BUY] price={prices[index]:.2f}, buy_amount={buy_amount:.2f}")
            self.logger.info(f"[BUY] updated cash={self.cash:.2f}, "
                             f"num_stock_shares={self.num_stock_shares[index]}, "
                             f"cost={self.cost:.2f}, trades={self.trades}")

            # Warnings
            if self.cash < 0:
                self.logger.warning(f"[WARNING] Cash is negative after buying! cash={self.cash:.2f}")
            if buy_num_shares > available_amount:
                self.logger.error(f"[ERROR] Bought more shares than allowed! "
                                  f"buy_num_shares={buy_num_shares}, available_amount={available_amount}")

            return buy_num_shares

        # Apply turbulence logic
        if self.turbulence_soft_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_soft_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                self.logger.info(f"[BUY] index={index}, turbulence={self.turbulence:.2f} >= "
                                 f"soft threshold={self.turbulence_soft_threshold:.2f}, buy skipped")

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        if self.day >= self.n_days - 1:
            self.terminal = True

        if self.terminal:
            self.logger.info(f"[TERMINAL] Episode: {self.episode}, Day: {self.day}")
            if self.make_plots:
                self._make_plot()
                self.logger.info("[TERMINAL] Plot saved for this episode.")

            current_prices = self.data['close'].values
            end_total_asset = self.cash + sum(np.array(self.num_stock_shares) * current_prices)

            df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            tot_reward = end_total_asset - self.asset_memory[0]

            sharpe = None
            if df_total_value["daily_return"].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()

            df_rewards = pd.DataFrame(self.rewards_memory, columns=["account_rewards"])
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"[SUMMARY] Begin total asset: {self.asset_memory[0]:.2f}")
                print(f"[SUMMARY] End total asset: {end_total_asset:.2f}")
                print(f"[SUMMARY] Total reward: {tot_reward:.2f}")
                print(f"[SUMMARY] Total cost: {self.cost:.2f}")
                print(f"[SUMMARY] Total trades: {self.trades}")
                if sharpe is not None:
                    print(f"[SUMMARY] Sharpe ratio: {sharpe:.3f}")
                print(f"[SUMMARY] Final shares per stock: {self.num_stock_shares}")
                print("=================================")

            # Colab-friendly saving
            import os
            results_dir = os.path.join(".", "results")
            os.makedirs(results_dir, exist_ok=True)

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    os.path.join(results_dir, f"actions_{self.mode}_{self.model_name}_{self.iteration}.csv"),
                    index=False,
                )
                df_total_value.to_csv(
                    os.path.join(results_dir, f"account_value_{self.mode}_{self.model_name}_{self.iteration}.csv"),
                    index=False,
                )
                df_rewards.to_csv(
                    os.path.join(results_dir, f"account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv"),
                    index=False,
                )

                plt.plot(self.asset_memory, "r")
                plt.title(f"Account Value - Episode {self.episode}")
                plt.xlabel("Step")
                plt.ylabel("Total Asset")
                plt.savefig(
                    os.path.join(results_dir, f"account_value_{self.mode}_{self.model_name}_{self.iteration}.png")
                )
                plt.close()
                self.logger.info("[TERMINAL] All files and plots saved in /content/results.")

            # Warnings
            if end_total_asset < 0:
                self.logger.warning(f"[WARNING] End total asset negative! {end_total_asset:.2f}")
            if tot_reward < -1e6:
                self.logger.warning(f"[WARNING] Extremely negative reward: {tot_reward:.2f}")

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # scale actions to max shares
            actions = actions.astype(int)  # convert to integer shares
            self.logger.debug(f"[STEP] Day {self.day} - Raw actions scaled: {actions}")

            if self.turbulence_hard_threshold is not None and self.turbulence >= self.turbulence_hard_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)
                self.logger.debug(f"[STEP] Turbulence high ({self.turbulence:.2f}) -> forced sell all: {actions}")

            begin_prices = self.data['close'].values
            begin_total_asset = self.cash + sum(np.array(self.num_stock_shares) * begin_prices)
            self.logger.debug(f"[STEP] Begin total asset: {begin_total_asset:.2f}, Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")

            # determine buy/sell indices
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            self.logger.debug(f"[STEP] Sell indices: {sell_index}, Buy indices: {buy_index}")

            self.previous_cash = self.cash
            self.previous_state = self.state.copy()
            self.previous_num_stock_shares = self.num_stock_shares.copy()

            # Execute sell actions
            for index in sell_index:
                sell_before = self.num_stock_shares[index]
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                sell_after = self.num_stock_shares[index]
                self.logger.debug(f"[SELL] Stock {index}: sold {sell_before - sell_after} shares, Cash now: {self.cash:.2f}")

            # Execute buy actions
            for index in buy_index:
                buy_before = self.num_stock_shares[index]
                actions[index] = self._buy_stock(index, actions[index])
                buy_after = self.num_stock_shares[index]
                self.logger.debug(f"[BUY] Stock {index}: bought {buy_after - buy_before} shares, Cash now: {self.cash:.2f}")

            self.actions_memory.append(actions)

            # update day and state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.risk_indicator_col is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                else:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.logger.debug(f"[STEP] Turbulence updated: {self.turbulence:.2f}")

            self.state = self._update_state()

            # Compute end total asset
            current_prices = self.data['close'].values
            end_total_asset = self.cash + sum(np.array(self.num_stock_shares) * current_prices)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            # --- Reward hyperparameters ---
            alpha = 1.0   # weight for portfolio return
            beta = 0.1    # weight for Sharpe-like risk adjustment
            
            # --- Portfolio return ---
            simple_return = (end_total_asset - begin_total_asset) / (begin_total_asset + 1e-8)
            
            # --- Recent returns for risk measures ---
            window = len(self.asset_memory)
            recent_assets = np.array(self.asset_memory[-window:])
            recent_returns = np.diff(recent_assets) / (recent_assets[:-1] + 1e-8)
            
            # --- Sharpe-like term (stabilized) ---
            mean_r = np.mean(recent_returns)
            std_r = np.std(recent_returns)
            
            # variance floor avoids explosion
            epsilon = 1e-3   # tuned to your return scale (10^-2 typical → 10^-3 floor is good)
            sharpe_like = mean_r / max(std_r, epsilon)
            
            # squash extremes to [-1, 1]
            sharpe_like = np.tanh(sharpe_like)
            
            # --- Final risk-adjusted reward ---
            self.reward = (
                alpha * simple_return               # reward for growth
                + beta * sharpe_like                # reward for stability
            )
            
            # --- Apply reward scaling ---
            self.reward *= self.reward_scaling

            # Logging
            self.logger.info(
                f"[STEP] End asset: {end_total_asset:.2f}, "
                f"Reward: {self.reward:.6f}, "
                f"ReturnTerm: {alpha * simple_return:.6f}, "
            )
            
            # Logging
            self.return_log.append(alpha * simple_return)
            self.vol_log.append(beta * sharpe_like)
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)

            self.step_cost = 0

            return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # Reset day and data
        self.day = 0
        self.data = self.df.loc[self.day, :]

        # Reset state and shares
        self.state = self._initiate_state()
        self.num_stock_shares = self.init_num_stock_shares.copy()
        self.cash = self.initial_amount

        # Get initial prices and calculate initial total asset
        initial_prices = self.data['close'].values
        initial_total_asset = self.initial_amount + np.sum(np.array(self.num_stock_shares) * initial_prices)
        self.asset_memory = [initial_total_asset]

        # Reset accounting variables
        self.cost = 0
        self.step_cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        # Increment episode count
        self.episode += 1

        # Logging
        self.logger.info(f"Episode {self.episode} started")
        self.logger.debug(f"Day: {self.day}")
        self.logger.debug(f"Cash: {self.cash:.2f}")
        self.logger.debug(f"Initial shares: {self.num_stock_shares}")
        self.logger.debug(f"Initial total asset: {initial_total_asset:.2f}")
        self.logger.debug("=" * 40)

        return self.state, {}


    def render(self, mode="human", close=False):
        # Simple render: return current state
        self.logger.info(
            f"[RENDER] Day: {self.day}, Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}"
        )
        return self.state


    def _initiate_state(self):
        self.cash = self.initial_amount
        self.num_stock_shares = self.init_num_stock_shares.copy()

        current_prices = self.data['close'].values
        total_asset = self.cash + np.sum(np.array(self.num_stock_shares) * current_prices)
        norm_cash = self.cash / (total_asset + 1e-8)
        norm_shares = [shares / (self.hmax + 1e-8) for shares in self.num_stock_shares]

        state = (
            [norm_cash]
            + norm_shares
            + sum(
                (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                [],
            )
        )

        # Logging
        self.logger.debug(f"[INIT_STATE] Day: {self.day}")
        self.logger.debug(f"[INIT_STATE] Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")
        self.logger.debug(f"[INIT_STATE] Total asset: {total_asset:.2f}")
        self.logger.debug(f"[INIT_STATE] State length: {len(state)}")
        self.logger.debug("=" * 40)

        return state


    def _update_state(self):
        current_prices = self.data['close'].values
        total_asset = self.cash + np.sum(np.array(self.num_stock_shares) * current_prices)
        norm_cash = self.cash / (total_asset + 1e-8)
        norm_shares = [shares / (self.hmax + 1e-8) for shares in self.num_stock_shares]

        state = (
            [norm_cash]
            + norm_shares
            + sum(
                (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                [],
            )
        )

        # Logging
        self.logger.debug(f"[UPDATE_STATE] Day: {self.day}")
        self.logger.debug(f"[UPDATE_STATE] Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")
        self.logger.debug(f"[UPDATE_STATE] Total asset: {total_asset:.2f}")
        self.logger.debug(f"[UPDATE_STATE] State length: {len(state)}")
        self.logger.debug("=" * 40)

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        self.logger.debug(f"[GET_DATE] Day: {self.day}, Date: {date}")
        return date


    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory

        self.logger.debug(f"[SAVE_ASSET_MEMORY] Dates stored: {len(date_list)}, "
                    f"Assets stored: {len(asset_list)}")

        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value


    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date['date']

        self.logger.debug(f"[SAVE_ACTION_MEMORY] Saved {len(action_list)} actions for "
                    f"{len(self.data.tic.values)} tickers")

        return df_actions


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.logger.debug(f"[SEED] Environment seeded with: {seed}")
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        self.logger.info("[GET_SB_ENV] Stable-Baselines VecEnv created and reset")
        return e, obs

class TensorboardCallback(BaseCallback):
    """
    TensorBoard logging callback for SB3 models.
    Logs rewards, episode lengths, rollout stats, actions, and learning rates.
    """

    def __init__(self, verbose=0, track_actions=True):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.start_time = None
        self.track_actions = track_actions
        self.actions_buffer = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.logger.record("time/training_start", self.start_time)
        if self.verbose:
            print(f"[TB] Training started at {self.start_time}")

    def _on_step(self) -> bool:
        try:
            reward = self.locals.get("rewards", None) or self.locals.get("reward", None)
            if reward is not None:
                reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                self.current_episode_reward += reward_val
                self.current_episode_length += 1
                self.logger.record("train/step_reward", reward_val)

            if self.track_actions and "actions" in self.locals:
                act = self.locals["actions"]
                if isinstance(act, (list, np.ndarray)):
                    self.actions_buffer.append(np.array(act).flatten())
                else:
                    self.actions_buffer.append([act])
        except Exception as e:
            if self.verbose:
                print(f"[TB Warning] Step logging error: {e}")

        done = self.locals.get("dones", [False])[0] or self.locals.get("done", False)
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            self.logger.record("train/episode_reward", self.current_episode_reward)
            self.logger.record("train/episode_length", self.current_episode_length)

            if len(self.episode_rewards) >= 5:
                self.logger.record(
                    "train/episode_reward_mean_5", np.mean(self.episode_rewards[-5:])
                )

            if self.verbose:
                print(
                    f"[TB] Episode done. Reward: {self.current_episode_reward:.2f}, "
                    f"Length: {self.current_episode_length}"
                )

            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.actions_buffer.clear()

        return True

    def _on_rollout_end(self) -> None:
        try:
            rb_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record("train/rollout_reward_min", float(np.min(rb_rewards)))
            self.logger.record("train/rollout_reward_mean", float(np.mean(rb_rewards)))
            self.logger.record("train/rollout_reward_max", float(np.max(rb_rewards)))
        except Exception:
            pass

        if self.track_actions and len(self.actions_buffer) > 0:
            acts = np.concatenate(self.actions_buffer)
            self.logger.record("actions/mean", float(np.mean(acts)))
            self.logger.record("actions/std", float(np.std(acts)))
            self.logger.record("actions/min", float(np.min(acts)))
            self.logger.record("actions/max", float(np.max(acts)))

        try:
            if hasattr(self.model, "lr_schedule"):
                lr = self.model.lr_schedule(self.model._current_progress_remaining)
            elif hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
            else:
                lr = None
            if lr is not None:
                self.logger.record("train/learning_rate", float(lr))
        except Exception:
            pass

    def _on_training_end(self) -> None:
        total_time = time.time() - self.start_time
        self.logger.record("time/training_end", time.time())
        self.logger.record("time/total_training_time_sec", total_time)
        if self.verbose:
            print(f"[TB] Training finished in {total_time:.2f} seconds.")

class DRLAgent:
    """DRL agent helper for SB3 algorithms"""

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS.")

        model_kwargs = DEFAULT_PARAMS[model_name]
        policy_kwargs = DEFAULT_PARAMS_POLICY[model_name]
        
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        if verbose:
            print(f"[DRLAgent] Initializing model {model_name} with kwargs: {model_kwargs}")

        return MODELS[model_name](
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(model, tb_log_name, total_timesteps=5000, callbacks: Optional[list] = None):
        all_callbacks = [TensorboardCallback(verbose=1)]
        if callbacks:
            all_callbacks += callbacks

        print(f"[DRLAgent] Starting training for {total_timesteps} timesteps...")
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=CallbackList(all_callbacks),
        )
        print(f"[DRLAgent] Training finished.")
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """Make predictions and return account memory and actions memory"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None
        actions_memory = None
        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)

            if i == max_steps - 1:
                account_memory = test_env.env_method("save_asset_memory")
                actions_memory = test_env.env_method("save_action_memory")

            if dones[0]:
                print(f"[DRLAgent] Hit end of environment at step {i}")
                break

        print("[DRLAgent] Prediction finished.")
        return account_memory[0], actions_memory[0]

def calculate_metrics(
    df_account_value,
    initial_amount=1_000_000,
    risk_free_rate=0.0
):
    """
    Calculate period, return, risk & risk-adjusted metrics from equity curve.

    Params:
    - df_account_value: pd.DataFrame with 'account_value' and datetime index or 'date' col
    - initial_amount: starting capital
    - risk_free_rate: daily risk-free rate, default 0.0

    Returns: dict of all metrics
    """

    # Prepare DataFrame
    if 'date' in df_account_value.columns:
        df = df_account_value.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
    else:
        df = df_account_value.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    start_date = df.index[0]
    end_date = df.index[-1]
    duration_days = (end_date - start_date).days

    start_value = df['account_value'].iloc[0]
    end_value = df['account_value'].iloc[-1]

    cum_return = (end_value / start_value - 1) * 100
    years = duration_days / 365.25
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    annualized_return = cagr

    # Daily returns
    df['returns'] = df['account_value'].pct_change().fillna(0)

    # Annualized Volatility
    daily_vol = df['returns'].std()
    ann_vol = daily_vol * np.sqrt(252) * 100

    # Drawdowns
    running_max = df['account_value'].cummax()
    drawdowns = (df['account_value'] - running_max) / running_max
    max_drawdown = drawdowns.min() * 100

    # Drawdown durations
    dd_durations = []
    duration = 0
    in_dd = False
    for dd in drawdowns:
        if dd < 0:
            duration += 1
            in_dd = True
        else:
            if in_dd:
                dd_durations.append(duration)
            duration = 0
            in_dd = False
    if in_dd:
        dd_durations.append(duration)
    max_dd_duration = max(dd_durations) if dd_durations else 0
    avg_dd_duration = np.mean(dd_durations) if dd_durations else 0

    avg_drawdown = drawdowns[drawdowns < 0].mean() * 100 if any(drawdowns < 0) else 0

    # Skewness & Kurtosis
    ret_skew = skew(df['returns'])
    ret_kurt = kurtosis(df['returns'])

    # VaR & CVaR 95%
    var_95 = np.percentile(df['returns'], 5) * 100
    cvar_95 = df['returns'][df['returns'] <= var_95/100].mean() * 100 if len(df['returns'][df['returns'] <= var_95/100]) > 0 else np.nan

    # Risk-adjusted returns

    # Excess returns over risk-free
    excess_returns = df['returns'] - risk_free_rate

    # Sharpe Ratio
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan

    # Sortino Ratio (downside deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else np.nan

    # Calmar Ratio (Annualized Return / Max Drawdown absolute)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        'Start': start_date,
        'End': end_date,
        'Duration (days)': duration_days,
        'Cumulative Return [%]': cum_return,
        'Annualized Return [%]': annualized_return,
        'CAGR [%]': cagr,
        'Annualized Volatility [%]': ann_vol,
        'Max Drawdown [%]': max_drawdown,
        'Max Drawdown Duration (days)': max_dd_duration,
        'Average Drawdown [%]': avg_drawdown,
        'Average Drawdown Duration (days)': avg_dd_duration,
        'Skewness': ret_skew,
        'Kurtosis': ret_kurt,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
    }

def extract_trades_from_positions(df_actions, df_prices):
    """
    df_actions: daily positions for each ticker (index=date, columns=tickers)
    df_prices: daily close prices for tickers (index=date, columns=tickers)

    Returns DataFrame with columns:
    ['tic', 'entry_date', 'exit_date', 'entry_price', 'exit_price']
    """

    trades = []
    for tic in df_actions.columns:
        positions = df_actions[tic]
        prices = df_prices[tic]

        position = 0
        entry_date = None
        entry_price = None

        for date, pos in positions.items():
            price = prices.loc[date]

            # Entry: position changes from 0 to positive
            if position == 0 and pos > 0:
                position = pos
                entry_date = date
                entry_price = price

            # Exit: position changes from positive to 0 (or smaller)
            elif position > 0 and pos == 0:
                exit_date = date
                exit_price = price

                trades.append({
                    'tic': tic,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                })

                position = 0
                entry_date = None
                entry_price = None

            # (Optional) handle partial exits / position changes — requires more logic

    trades_df = pd.DataFrame(trades)
    return trades_df


def calculate_trade_metrics(trades_df):
    """
    trades_df must have columns:
    - 'entry_date' (datetime)
    - 'exit_date' (datetime)
    - 'entry_price'
    - 'exit_price'
    - Optionally 'quantity' or 'position_size'

    Returns dict of trade-level metrics.
    """

    if trades_df.empty:
        # No trades, return NaN for all
        nan = np.nan
        return {
            'Number of Trades': 0,
            'Win Rate [%]': nan,
            'Best Trade [%]': nan,
            'Worst Trade [%]': nan,
            'Average Trade [%]': nan,
            'Max Trade Duration (days)': nan,
            'Average Trade Duration (days)': nan,
            'Profit Factor': nan,
            'Expectancy [%]': nan,
            'SQN': nan,
            'Kelly Criterion': nan,
        }

    # Calculate trade returns (%)
    trades_df = trades_df.copy()
    trades_df['trade_return'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'] * 100

    # Trade durations in days
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['trade_duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days

    num_trades = len(trades_df)
    wins = trades_df[trades_df['trade_return'] > 0]
    win_rate = len(wins) / num_trades * 100 if num_trades > 0 else np.nan
    best_trade = trades_df['trade_return'].max()
    worst_trade = trades_df['trade_return'].min()
    avg_trade = trades_df['trade_return'].mean()
    max_trade_duration = trades_df['trade_duration'].max()
    avg_trade_duration = trades_df['trade_duration'].mean()

    gross_profit = wins['trade_return'].sum()
    gross_loss = trades_df[trades_df['trade_return'] < 0]['trade_return'].sum()
    profit_factor = (gross_profit / -gross_loss) if gross_loss != 0 else np.nan

    expectancy = avg_trade * (win_rate / 100) + (worst_trade) * (1 - win_rate / 100)

    # SQN calculation: SQN = (mean trade return / std trade return) * sqrt(num trades)
    std_trade_return = trades_df['trade_return'].std()
    sqn = (avg_trade / std_trade_return) * np.sqrt(num_trades) if std_trade_return != 0 else np.nan

    # Kelly Criterion: k = win_rate - (loss_rate / payoff_ratio)
    loss_rate = 1 - win_rate / 100
    payoff_ratio = (wins['trade_return'].mean() / abs(trades_df[trades_df['trade_return'] < 0]['trade_return'].mean())) if len(wins) > 0 else np.nan
    if payoff_ratio and payoff_ratio > 0:
        kelly = (win_rate / 100) - (loss_rate / payoff_ratio)
    else:
        kelly = np.nan

    return {
        'Number of Trades': num_trades,
        'Win Rate [%]': win_rate,
        'Best Trade [%]': best_trade,
        'Worst Trade [%]': worst_trade,
        'Average Trade [%]': avg_trade,
        'Max Trade Duration (days)': max_trade_duration,
        'Average Trade Duration (days)': avg_trade_duration,
        'Profit Factor': profit_factor,
        'Expectancy [%]': expectancy,
        'SQN': sqn,
        'Kelly Criterion': kelly,
    }

def build_mvo_and_baselines(train_df, trade_price_df, initial_capital=1_000_000, weight_bounds=(0,0.5)):
    """
    train_df: raw rows with columns ['date','tic','close'] used to compute mu/cov
    trade_price_df: pivoted index=date, columns=tic with prices (out-of-sample period)
    """
    price_train = train_df.pivot(index='date', columns='tic', values='close').sort_index()
    price_train.index = pd.to_datetime(price_train.index)
    # compute daily returns
    returns_train = price_train.pct_change().dropna()

    # try pypfopt
    try:
        mu = expected_returns.mean_historical_return(price_train)  # annualized mu
        S = risk_models.sample_cov(price_train)
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        ef.max_sharpe()
        weights = ef.clean_weights()
        print("MVO weights from PyPortfolioOpt")
    except Exception as e:
        print("pypfopt not available or failed:", e)
        # fallback to equal weight
        cols = price_train.columns.tolist()
        weights = {c: 1/len(cols) for c in cols}
        print("Using equal-weight fallback")

    # prepare weights aligned with trade_price_df columns
    w_arr = np.array([weights.get(t, 0.0) for t in trade_price_df.columns])

    # compute shares from last available price in trade_price_df
    last_prices = trade_price_df.iloc[-1].to_numpy()
    dollar_alloc = w_arr * initial_capital
    # avoid divide by zero
    shares = np.where(last_prices > 0, dollar_alloc / last_prices, 0.0)

    # portfolio value time series
    portfolio_values = (trade_price_df.values * shares).sum(axis=1)
    df_mvo = pd.DataFrame({'date': trade_price_df.index, 'account_value': portfolio_values})
    # also build equal-weight and buy-and-hold for comparison
    # equal weight: allocate equally across available assets
    n = trade_price_df.shape[1]
    ew_shares = np.where(last_prices > 0, (initial_capital / n) / last_prices, 0.0)
    port_ew = (trade_price_df.values * ew_shares).sum(axis=1)
    df_ew = pd.DataFrame({'date': trade_price_df.index, 'account_value': port_ew})
    # buy-and-hold benchmark on index: simply sum shares of each asset with equal weight buying at first trade price
    bh_shares = np.where(trade_price_df.iloc[0].to_numpy() > 0, (initial_capital / n) / trade_price_df.iloc[0].to_numpy(), 0.0)
    port_bh = (trade_price_df.values * bh_shares).sum(axis=1)
    df_bh = pd.DataFrame({'date': trade_price_df.index, 'account_value': port_bh})

    return df_mvo, df_ew, df_bh, weights


def build_benchmarks(trade, train):
    price_df = trade.pivot(index='date', columns='tic', values='close').sort_index()
    price_df.index = pd.to_datetime(price_df.index)

    df_mvo, df_ew, df_bh, mvo_weights = build_mvo_and_baselines(train, price_df)
    return df_mvo, df_ew, df_bh

def analysis_metrics_rl_benchmarks(df_rl, df_mvo, df_ew, df_bh):
    metrics_rl = calculate_metrics(df_rl)
    metrics_mvo = calculate_metrics(df_mvo)
    metrics_ew = calculate_metrics(df_ew)
    metrics_bh = calculate_metrics(df_bh)

    all_metrics = {
        'RL': metrics_rl,
        'MVO': metrics_mvo,
        'EW': metrics_ew,
        'BH': metrics_bh
    }

    df_metrics = pd.DataFrame(all_metrics)
    return df_metrics, all_metrics

def analysis_trade_metrics(trade, df_actions):
    trade['date'] = pd.to_datetime(trade['date'])
    df_prices = trade.pivot(index='date', columns='tic', values='close')
    df_prices = df_prices.sort_index()
    trades_df = extract_trades_from_positions(df_actions, df_prices)
    trade_metrics = calculate_trade_metrics(trades_df)
    return trade_metrics

def scale_features(df,
                   unscaled_cols=None,
                   global_scale_cols=None):
    df = df.copy()

    # Default groups
    if unscaled_cols is None:
        unscaled_cols = [
            "turbulence",           # absolute risk metric
            "rank_mom20", "rank_rsi_mom", "rank_rsi_mr",  # already [0,1]
            "day",
            "close",
            "date",
            "tic",
            "high",
            "low",
            "open",
            "volume",
        ]

    if global_scale_cols is None:
        global_scale_cols = [
            "finbert_future", "fingpt_future",   # sentiment probs
            # "vix", "vix_rolling_30"              # market-wide risk measures
        ]

    # Features to scale cross-sectionally (per day, across tickers)
    feature_cols = [
        c for c in df.columns if c not in ["date", "tic"] + unscaled_cols + global_scale_cols
    ]

    # --- Cross-sectional z-score ---
    def zscore(group):
        return (group - group.mean()) / (group.std() + 1e-8)

    scaled_cross = df.groupby("date")[feature_cols].transform(zscore)

    # --- Global z-score (whole dataset) ---
    scaled_global = df[global_scale_cols].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # --- Combine ---
    scaled = pd.concat([scaled_cross, scaled_global], axis=1)

    # Keep unscaled columns
    for col in unscaled_cols:
        if col in df.columns:
            scaled[col] = df[col]

    # Keep date and ticker
    scaled["date"] = df["date"]
    scaled["tic"] = df["tic"]

    # Final sort
    scaled = scaled.sort_values(["date", "tic"]).reset_index(drop=True)

    return scaled

def get_train_trade(train_start_date, train_end_date, trade_start_date, trade_end_date):
    df = get_data(start_date=train_start_date, end_date=trade_end_date)
    df = clean_data(df)
    df = add_indicators(df)
    # df = add_vix(df)
    df = df.sort_values(['tic', 'date']).reset_index(drop=True)
    print(df.columns)
    # df['vix_rolling_30'] = df['vix'].rolling(window=30, min_periods=1).mean()
    df = add_turbulence(df)

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    list_ticker = df["tic"].unique().tolist()
    list_date = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date","tic"]).merge(df, on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(df['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full["date"] = pd.to_datetime(processed_full["date"])

    tickers = ["AAPL", "BA", "GS", "JPM"]
    sentiment_dfs = []
    for tic in tickers:
        with open(f"files/{tic}_final.json") as f:
            news = json.load(f)

        temp = pd.DataFrame(news)
        temp["tic"] = tic
        temp["date"] = pd.to_datetime(temp["date"])

        def finbert_score(row):
            s = row["sentiment"].lower()
            if s == "positive":
                return row["sentiment_score"]
            elif s == "negative":
                return -row["sentiment_score"]
            else:
                return 0.0
            
        temp["finbert"] = temp.apply(finbert_score, axis=1)
        temp["fingpt"] = temp["fingpt_score"].apply(lambda x: x[1] - x[2])
        sentiment_dfs.append(temp[["date", "tic", "finbert", "fingpt"]])


    sentiment_df = pd.concat(sentiment_dfs)
    sentiment_daily = sentiment_df.groupby(["tic","date"]).mean().reset_index()
    processed_full = processed_full.merge(sentiment_daily, on=["tic","date"], how="left")
    processed_full["finbert"] = processed_full["finbert"].fillna(0.0)
    processed_full["fingpt"] = processed_full["fingpt"].fillna(0.0)

    window = 5
    decay_factor = 0.8  # decaying impact over days

    processed_full = processed_full.sort_values(["tic","date"])

    processed_full["finbert_future"] = 0.0
    processed_full["fingpt_future"] = 0.0

    for tic in processed_full["tic"].unique():
        df_tic = processed_full[processed_full["tic"] == tic].copy()
        finbert_vals = df_tic["finbert"].values
        fingpt_vals = df_tic["fingpt"].values

        finbert_future = np.zeros_like(finbert_vals)
        fingpt_future = np.zeros_like(fingpt_vals)

        for i in range(len(finbert_vals)):
            for w in range(window):
                if i + w < len(finbert_vals):
                    weight = decay_factor ** w
                    finbert_future[i + w] += finbert_vals[i] * weight
                    fingpt_future[i + w] += fingpt_vals[i] * weight

        processed_full.loc[processed_full["tic"] == tic, "finbert_future"] = finbert_future
        processed_full.loc[processed_full["tic"] == tic, "fingpt_future"] = fingpt_future

    processed_full = processed_full.drop(columns = ["finbert", "fingpt"])

    train = data_split(processed_full, train_start_date, train_end_date)
    trade = data_split(processed_full, trade_start_date, trade_end_date)

    train_scaled = scale_features(train)
    trade_scaled = scale_features(trade)

    n_repeat = 4
    n_rows = len(train_scaled)
    new_index = np.arange(n_rows // n_repeat + 1).repeat(n_repeat)[:n_rows]
    train_scaled.index = new_index

    n_rows = len(trade_scaled)
    new_index = np.arange(n_rows // n_repeat + 1).repeat(n_repeat)[:n_rows]
    trade_scaled.index = new_index

    return train_scaled, trade_scaled

def backtest(trained, train, trade, _ind, _hmax, _initial_amount, _reward_scaling):
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + stock_dimension + len(_ind) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": _hmax,
        "initial_amount": _initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": _ind,
        "action_space": stock_dimension,
        "reward_scaling": _reward_scaling,
        "model_name": "ppo",
        "mode": "rl"
    }

    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained,
        environment = e_trade_gym)

    return e_trade_gym, df_account_value, df_actions

def analysis_backtest(trade, train, df_account_value, df_actions):
    df_mvo, df_ew, df_bh = build_benchmarks(trade, train)
    df_result, results = analysis_metrics_rl_benchmarks(df_account_value, df_mvo, df_ew, df_bh)
    # trade_metrics = analysis_trade_metrics(trade, df_actions)
    trade_metrics = {}

    return df_result, results, trade_metrics