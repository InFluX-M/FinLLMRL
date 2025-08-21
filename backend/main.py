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

from help import MODELS, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE, get_train_trade, backtest, INDICATORS, analysis_backtest

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
    shares: list[int]
    cash: int
    hmax: int
    model: str

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
        df_trade_filtered,  # filtered
        INDICATORS[request.model],
        request.hmax,
        request.cash,
        10,
    )
    
    df_res_trade, res_trade, trade_metrics = analysis_backtest(df_trade_filtered, data["train"], df_trade_value, df_trade_actions)
    
    return {
        "res_trade": res_trade,
        "trade_metrics": trade_metrics
    }
