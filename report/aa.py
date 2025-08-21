import logging
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym
    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset
        initial_amount (int): Amount of cash initially available
        num_stock_shares (list[int]): Initial number of shares per stock
        buy_cost_pct (list[float]): Cost for buying shares per asset
        sell_cost_pct (list[float]): Cost for selling shares per asset
        reward_scaling (float): Scaling factor for reward
        state_space (int): Size of state space
        action_space (int): Size of action space
        tech_indicator_list (list[str]): List of technical indicators
        turbulence_hard_threshold (float): Max turbulence for forced liquidation
        turbulence_soft_threshold (float): Max turbulence for skipping buys
        risk_indicator_col (str): Column name for turbulence indicator
        make_plots (bool): Whether to generate plots
        print_verbosity (int): Frequency of printing episode summaries
        logging_enabled (bool): Whether to enable logging
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
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
        logging_enabled=True,  # New parameter
        log_level="INFO",      # New parameter
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        # Initialize core attributes (unchanged)
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

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        if self.logging_enabled:
            # Configure logger only if enabled
            self.logger.setLevel(self.log_level)
            # Remove existing handlers to avoid duplicates (if any)
            self.logger.handlers = []
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        else:
            # Disable logging by setting level to CRITICAL+1
            self.logger.setLevel(logging.CRITICAL + 1)

        # Dataset sanity logs
        self.logger.info("[INIT] ===== StockTradingEnv =====")
        self.logger.info(f"df.shape = {self.df.shape}")
        self.logger.info(f"df.index.type = {type(self.df.index)}")
        try:
            head_idx = self.df.index[:5].tolist()
        except Exception:
            head_idx = str(self.df.index)[:80]
        self.logger.info(f"df.index[:5] = {head_idx}")
        self.logger.info(f"df.columns = {list(self.df.columns)}")

        # Basic column checks
        if "date" not in self.df.columns:
            self.logger.warning("'date' column not found in df! Episode length/done logic may break.")
        if "close" not in self.df.columns:
            self.logger.warning("'close' column not found in df! Pricing/portfolio logic will break.")
        if "tic" not in self.df.columns:
            self.logger.warning("'tic' column not found in df. Multi-asset checks will be limited.")

        # Count unique dates and tickers
        if "date" in self.df.columns:
            self.n_days = len(self.df["date"].unique())
            self.logger.info(f"unique trading days (n_days) = {self.n_days}")
        else:
            self.n_days = len(self.df.index.unique())
            self.logger.info(f"'date' missing; fallback n_days via index.unique() = {self.n_days} (⚠️ may be wrong)")

        if "tic" in self.df.columns:
            n_tics = int(self.df["tic"].nunique())
            self.logger.info(f"unique tickers (n_tics) = {n_tics}, stock_dim (arg) = {self.stock_dim}")
            if n_tics != self.stock_dim:
                self.logger.warning(
                    f"stock_dim ({self.stock_dim}) != unique tickers ({n_tics}). "
                    f"State/action sizes or price vector lengths may mismatch."
                )
        else:
            self.logger.info(f"stock_dim (arg) = {self.stock_dim} (no 'tic' to verify).")

        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.logger.info(f"action_space.shape = {self.action_space.shape}")
        self.logger.info(f"observation_space.shape = {self.observation_space.shape}")

        # Initial day slice
        self.data = self.df.loc[self.day, :]
        try:
            first_date = (self.data["date"].iloc[0]
                          if hasattr(self.data["date"], "iloc")
                          else self.data["date"])
        except Exception:
            first_date = "<?>"
        self.logger.info(f"starting day = {self.day}, starting date = {first_date}")

        # Thresholds (turbulence)
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
                self.logger.info(
                    f"turbulence thresholds -> soft={self.turbulence_soft_threshold:.4f}, "
                    f"hard={self.turbulence_hard_threshold:.4f} (from percentiles 95/99.5)"
                )
            else:
                self.turbulence_soft_threshold = turbulence_soft_threshold
                self.turbulence_hard_threshold = turbulence_hard_threshold
                self.logger.warning(
                    f"risk_indicator_col '{self.risk_indicator_col}' not in df; "
                    f"thresholds left as provided: soft={self.turbulence_soft_threshold}, "
                    f"hard={self.turbulence_hard_threshold}"
                )
        else:
            self.turbulence_soft_threshold = turbulence_soft_threshold
            self.turbulence_hard_threshold = turbulence_hard_threshold
            self.logger.info(
                f"turbulence thresholds (user) -> soft={self.turbulence_soft_threshold}, "
                f"hard={self.turbulence_hard_threshold}"
            )

        # Previous state bookkeeping
        self.initial = initial
        self.previous_state = previous_state
        self.previous_cash = initial_amount
        self.previous_num_stock_shares = num_stock_shares.copy()
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Build initial state
        self.state = self._initiate_state()
        state_len = len(self.state) if hasattr(self.state, "__len__") else None
        self.logger.info(f"initial state length = {state_len}, expected state_space = {self.state_space}")

        # Heuristic state length check
        try:
            expected_len = 1 + self.stock_dim + self.stock_dim * len(self.tech_indicator_list) + 1
            if state_len is not None and state_len != expected_len:
                self.logger.warning(
                    f"state length ({state_len}) != heuristic expected ({expected_len}). "
                    f"Check tech indicators per ticker and state assembly."
                )
        except Exception:
            pass

        # Memories & counters
        self.return_log = []
        self.vol_log = []
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # Initial portfolio value
        try:
            prices = self.data["close"].values
            if "tic" in self.df.columns:
                self.logger.info(f"prices vector length (close) = {len(prices)} (should match stock_dim={self.stock_dim})")
                if len(prices) != self.stock_dim:
                    self.logger.warning(
                        f"len(close[{first_date}]) = {len(prices)} != stock_dim ({self.stock_dim}). "
                        f"Check day slicing and multi-asset alignment."
                    )
        except Exception as e:
            self.logger.error(f"Could not extract initial 'close' prices: {e}")
            prices = np.zeros(self.stock_dim, dtype=float)

        self.asset_memory = [
            self.initial_amount + np.sum(np.array(self.num_stock_shares) * prices)
        ]
        self.logger.info(
            f"initial cash = {self.initial_amount:.2f}, "
            f"initial holdings value = {float(np.sum(np.array(self.num_stock_shares) * prices)):.2f}, "
            f"initial total asset = {self.asset_memory[0]:.2f}"
        )

        # Sanity checks for list lengths
        def _check_len(name, arr, target):
            try:
                ln = len(arr)
            except Exception:
                ln = None
            ok = (ln == target)
            self.logger.info(f"len({name}) = {ln}  {'OK' if ok else f'≠ {target} (WARN)'}")
            return ok

        _check_len("num_stock_shares", self.num_stock_shares, self.stock_dim)
        _check_len("buy_cost_pct", self.buy_cost_pct, self.stock_dim)
        _check_len("sell_cost_pct", self.sell_cost_pct, self.stock_dim)

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.logger.info(f"first date_memory entry = {self.date_memory[0]}")
        self.logger.info("[INIT] ===== Init complete =====")

        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            sell_num_shares = 0
            if self.num_stock_shares[index] > 0:
                sell_num_shares = min(abs(action), self.num_stock_shares[index])
                prices = self.data['close'].values
                sell_amount = prices[index] * sell_num_shares * (1 - self.sell_cost_pct[index])
                
                self.cash += sell_amount
                self.num_stock_shares[index] -= sell_num_shares
                self.cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1

                self.logger.info(f"[SELL_NORMAL] index={index}, action={action}, sell_num_shares={sell_num_shares}")
                self.logger.info(f"[SELL_NORMAL] price={prices[index]:.2f}, sell_amount={sell_amount:.2f}")
                self.logger.info(f"[SELL_NORMAL] updated cash={self.cash:.2f}, num_stock_shares={self.num_stock_shares[index]}, cost={self.cost:.2f}, trades={self.trades}")
                
                if self.cash < 0:
                    self.logger.warning(f"Cash is negative after selling! cash={self.cash:.2f}")
                if sell_num_shares > self.num_stock_shares[index] + sell_num_shares:
                    self.logger.error(f"Sold more shares than available! index={index}, sell_num_shares={sell_num_shares}, num_stock_shares={self.num_stock_shares[index]}")
            else:
                self.logger.info(f"[SELL_NORMAL] index={index}, action={action}, no shares to sell (num_stock_shares=0)")
            return sell_num_shares

        if self.turbulence_hard_threshold is not None and self.turbulence >= self.turbulence_hard_threshold:
            if self.num_stock_shares[index] > 0:
                prices = self.data['close'].values
                sell_num_shares = self.num_stock_shares[index]
                sell_amount = prices[index] * sell_num_shares * (1 - self.sell_cost_pct[index])
                
                self.cash += sell_amount
                self.num_stock_shares[index] = 0
                self.cost += prices[index] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1

                self.logger.info(f"[SELL_TURBULENCE] Hard threshold triggered, index={index}, forced sell all")
                self.logger.info(f"[SELL_TURBULENCE] sell_num_shares={sell_num_shares}, price={prices[index]:.2f}, sell_amount={sell_amount:.2f}")
                self.logger.info(f"[SELL_TURBULENCE] updated cash={self.cash:.2f}, num_stock_shares={self.num_stock_shares[index]}, cost={self.cost:.2f}, trades={self.trades}")

                if self.cash < 0:
                    self.logger.warning(f"Cash is negative after forced sell! cash={self.cash:.2f}")
            else:
                sell_num_shares = 0
                self.logger.info(f"[SELL_TURBULENCE] index={index}, no shares to sell")
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            prices = self.data['close'].values
            available_amount = self.cash // (prices[index] * (1 + self.buy_cost_pct[index]))
            buy_num_shares = min(available_amount, action)
            buy_amount = prices[index] * buy_num_shares * (1 + self.buy_cost_pct[index])

            self.cash -= buy_amount
            self.num_stock_shares[index] += buy_num_shares
            self.cost += prices[index] * buy_num_shares * self.buy_cost_pct[index]
            self.trades += 1

            self.logger.info(f"[BUY] index={index}, action={action}, buy_num_shares={buy_num_shares}")
            self.logger.info(f"[BUY] price={prices[index]:.2f}, buy_amount={buy_amount:.2f}")
            self.logger.info(f"[BUY] updated cash={self.cash:.2f}, num_stock_shares={self.num_stock_shares[index]}, cost={self.cost:.2f}, trades={self.trades}")
            
            if self.cash < 0:
                self.logger.warning(f"Cash is negative after buying! cash={self.cash:.2f}")
            if buy_num_shares > available_amount:
                self.logger.error(f"Bought more shares than allowed! buy_num_shares={buy_num kill_shares}, available_amount={available_amount}")

            return buy_num_shares

        if self.turbulence_soft_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_soft_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                self.logger.info(f"[BUY] index={index}, turbulence={self.turbulence:.2f} >= soft threshold={self.turbulence_soft_threshold:.2f}, buy skipped")

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        if self.day >= self.n_days - 1:
            self.terminal = True

        if self.terminal:
            self.logger.info(f"\n[TERMINAL] Episode: {self.episode}, Day: {self.day}")
            if self.make_plots:
                self._make_plot()
                self.logger.info("Plot saved for this episode.")

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
                self.logger.info(f"[SUMMARY] Begin total asset: {self.asset_memory[0]:.2f}")
                self.logger.info(f"[ oscuridad total asset: {end_total_asset:.2f}")
                self.logger.info(f"[SUMMARY] Total reward: {tot_reward:.2f}")
                self.logger.info(f"[SUMMARY] Total cost: {self.cost:.2f}")
                self.logger.info(f"[SUMMARY] Total trades: {self.trades}")
                if sharpe is not None:
                    self.logger.info(f"[SUMMARY] Sharpe ratio: {sharpe:.3f}")
                self.logger.info(f"[SUMMARY] Final shares per stock: {self.num_stock_shares}")
                self.logger.info("=================================")

            # Colab-friendly saving
            import os
            results_dir = os.path.join("/content", "results")
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
                self.logger.info("All files and plots saved in /content/results.")

            if end_total_asset < 0:
                self.logger.warning(f"End total asset negative! {end_total_asset:.2f}")
            if tot_reward < -1e6:
                self.logger.warning(f"Extremely negative reward: {tot_reward:.2f}")

            return self.state, self.reward, self.terminal, False, {}
        
        else:
            actions = actions * self.hmax
            actions = actions.astype(int)
            self.logger.info(f"\n[STEP] Day {self.day} - Raw actions scaled: {actions}")

            if self.turbulence_hard_threshold is not None and self.turbulence >= self.turbulence_hard_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)
                self.logger.info(f"[STEP] Turbulence high ({self.turbulence:.2f}) -> forced sell all: {actions}")

            begin_prices = self.data['close'].values
            begin_total_asset = self.cash + sum(np.array(self.num_stock_shares) * begin_prices)
            self.logger.info(f"[STEP] Begin total asset: {begin_total_asset:.2f}, Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            self.logger.info(f"[STEP] Sell indices: {sell_index}, Buy indices: {buy_index}")

            self.previous_cash = self.cash
            self.previous_state = self.state.copy()
            self.previous_num_stock_shares = self.num_stock_shares.copy()

            for index in sell_index:
                sell_before = self.num_stock_shares[index]
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                sell_after = self.num_stock_shares[index]
                self.logger.info(f"[SELL] Stock {index}: sold {sell_before - sell_after} shares, Cash now: {self.cash:.2f}")

            for index in buy_index:
                buy_before = self.num_stock_shares[index]
                actions[index] = self._buy_stock(index, actions[index])
                buy_after = self.num_stock_shares[index]
                self.logger.info(f"[BUY] Stock {index}: bought {buy_after - buy_before} shares, Cash now: {self.cash:.2f}")

            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.risk_indicator_col is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                else:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.logger.info(f"[STEP] Turbulence updated: {self.turbulence:.2f}")

            self.state = self._update_state()

            current_prices = self.data['close'].values
            end_total_asset = self.cash + sum(np.array(self.num_stock_shares) * current_prices)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            simple_return = (end_total_asset - begin_total_asset) / self.initial_amount

            if len(self.asset_memory) > 15:
                recent_assets = np.array(self.asset_memory[-16:])
                recent_returns = np.diff(recent_assets) / (recent_assets[:-1] + 1e-8)
                vol_penalty = np.std(recent_returns)
            else:
                vol_penalty = 0.0

            step_sharpe = simple_return / (vol_penalty + 1e-8)
            risk_aversion = 0.5
            self.reward = simple_return - risk_aversion * vol_penalty + 0.1 * step_sharpe
            self.reward *= self.reward_scaling

            self.return_log.append(simple_return)
            self.vol_log.append((vol_penalty, step_sharpe))
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)

            self.logger.info(f"[STEP] End total asset: {end_total_asset:.2f}, Reward: {self.reward:.4f}, Step suosana Sharpe: {step_sharpe:.4f}, Vol penalty: {vol_penalty:.4f}")

            return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        self.num_stock_shares = self.init_num_stock_shares.copy()
        self.cash = self.initial_amount

        initial_prices = self.data['close'].values
        initial_total_asset = self.initial_amount + np.sum(np.array(self.num_stock_shares) * initial_prices)
        self.asset_memory = [initial_total_asset]

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        self.logger.info(f"\n[RESET] Episode {self.episode} started")
        self.logger.info(f"[RESET] Day: {self.day}")
        self.logger.info(f"[RESET] Cash: {self.cash:.2f}")
        self.logger.info(f"[RESET] Initial shares: {self.num_stock_shares}")
        self.logger.info(f"[RESET] Initial total asset: {initial_total_asset:.2f}")
        self.logger.info("=================================")

        return self.state, {}

    def render(self, mode="human", close=False):
        self.logger.info(f"[RENDER] Day: {self.day}, Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")
        return self.state

    def _initiate_state(self):
        self.cash = self.initial Hannah self.initial_amount
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
            + [self.data['vix'].iloc[0]]
        )

        self.logger.info(f"[INIT_STATE] Day: {self.day}")
        self.logger.info(f"[INIT_STATE] Cash: {self.cash:.2f}, Shares: {self.num_stock_shares}")
        self.logger.info مشتریان

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
            + [self.data['vix'].iloc[0]]
        )

        self.logger.info(f"[UPDATE_STATE] Day: {self.day}")
        self.logger.info(f"[UPDATE_STATE] Cash: {self.cash:. negotiation, Shares: {self.num_stock_shares}")
        self.logger.info(f"[UPDATE_STATE] Total asset: {total_asset:.2f}")
        self.logger.info(f"[UPDATE ACT, Shares: {self.num_stock_shares}")
        self.logger.info("=================================")

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        self.logger.info(f"[GET_DATE] Day: {self.day}, Date: {date}")
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date['date']

        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs