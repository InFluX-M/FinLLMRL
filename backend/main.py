from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from stable_baselines3 import PPO

from help import MODELS, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE, get_train_trade, backtest, INDICATORS, analysis_backtest, extract_trades_from_positions

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log/assistant.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# --- Create app with lifespan ---
app = FastAPI()

models = {

}

data = {

}

# --- Startup Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI is starting up...")

    # load models
    for model in MODELS:
        models[model] = MODELS[model].load(f"files/{model}.zip")

    # load data
    yesterday = datetime.now() - timedelta(days=1)
    train, trade = get_train_trade(TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, yesterday.strftime('%Y-%m-%d'))
    data["train"] = train
    data["trade"] = trade

    logger.info("Initialization complete!")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Model ---
class TradeRequest(BaseModel):
    start_trade: date
    end_trade: date 
    shares: list[int] = [0, 0, 0, 0]
    cash: int = 1_000_000
    hmax: int = 100
    model: str = "ppo"
    comission: float = 0.001
    reward_scaling: float = 10

class TradeStats(BaseModel):
    Number_of_Trades: int
    Win_Rate: float
    Best_Trade: float
    Worst_Trade: float
    Average_Trade: float
    Max_Trade_Duration_days: int
    Average_Trade_Duration_days: float
    Profit_Factor: float
    Expectancy: float
    SQN: float
    Kelly_Criterion: float


# --- Endpoint ---
@app.post("/trade/")
async def trade(request: TradeRequest):
    logger.info(f"Received trade request: {request}")
    
    df_trade = data["trade"].copy()
    df_trade['date'] = pd.to_datetime(df_trade['date']).dt.date

    mask = (df_trade['date'] >= request.start_trade) & (df_trade['date'] <= request.end_trade)
    df_trade_filtered = df_trade.loc[mask].copy()
    df_trade_filtered.reset_index(drop=True, inplace=True)
    repeat_times = 4
    new_index = np.repeat(np.arange(len(df_trade_filtered) // repeat_times + 1), repeat_times)[:len(df_trade_filtered)]
    df_trade_filtered.index = new_index
    
    e_trade_gym, df_trade_value, df_trade_actions = backtest(
        models[request.model],
        data["train"],
        df_trade_filtered,
        INDICATORS[request.model],
        request.hmax,
        request.cash,
        request.reward_scaling,
        request.shares,
        request.comission
    )
    
    df_res_trade, res_trade, trade_metrics, actions = analysis_backtest(df_trade_filtered, data["train"], df_trade_value, df_trade_actions)
    trade_metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in trade_metrics.items()}

    import math

    def safe_value(v):
        """Convert to JSON-safe values (handle numpy scalars, NaN, inf)."""
        if hasattr(v, "item"):  # convert numpy scalar to Python native
            v = v.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None  # JSON can handle None -> null
        return v

    def safe_dict(d):
        """Recursively clean dicts for JSON serialization."""
        clean = {}
        for k, v in d.items():
            if isinstance(v, dict):
                clean[k] = safe_dict(v)  # recurse for nested dicts (like res_trade)
            else:
                clean[k] = safe_value(v)
        return clean

    # Apply to both
    res_trade = safe_dict(res_trade)
    trade_metrics = safe_dict(trade_metrics)

    return {
        "res_trade": res_trade,
        "trade_metrics": trade_metrics,
        "actions": actions
    }
