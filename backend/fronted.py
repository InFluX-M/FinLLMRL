import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go


API_BASE = "http://localhost:8000"

# -----------------------
# Wake word start button
# -----------------------
st.title("🏠 InFluX Trader")

st.markdown(
    """
    <style>
    /* Make the main text larger */
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }

    /* Make titles bigger */
    h1, .stTitle {
        font-size: 2.5rem !important;
    }
    h2, .stSubheader {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.6rem !important;
    }

    /* Enlarge input boxes and dropdowns */
    input, textarea, select {
        font-size: 18px !important;
    }

    /* Enlarge buttons */
    button {
        font-size: 18px !important;
        padding: 0.6em 1.2em !important;
    }

    /* Enlarge table text */
    .stDataFrame tbody td, .stDataFrame thead th {
        font-size: 18px !important;
    }

    /* Enlarge Plotly charts axis labels & titles */
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .gtitle {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Tabs become available
# -----------------------
tab1, tab2, tab3 = st.tabs(["💬 RL suggest", "🎙️ Chart", "🎤 Analysis"])

# --- RL Suggest Tab ---
with tab1:
    st.subheader("🤖 RL Agent Trading Simulation")

    # Use a form so all inputs are submitted together
    with st.form("rl_form"):
        start_trade = st.date_input("Start Trade Date")
        end_trade = st.date_input("End Trade Date")
        shares = st.text_input("Initial Shares (comma separated)", "0, 0, 0, 0")
        cash = st.number_input("Initial Cash", value=1_000_000, step=10000)
        hmax = st.number_input("Max Shares per Trade (hmax)", value=100)
        model = st.selectbox(
            "RL Model",
            [
                "ppo-single-norm-raw",
                "rppo-single-norm-raw",
                "rppo-single-norm-risk",
                "rppo-single-rand-raw",
                "rppo-vec-norm-raw",
                "rppo-vec-norm-risk",
                "rppo-vec-rand-raw",
            ]
        )
        commission = st.number_input("Commission (%)", value=0.001, format="%.4f")
        reward_scaling = st.number_input("Reward Scaling", value=10.0)

        submitted = st.form_submit_button("Run Simulation")

    if submitted:
        # Parse shares string into list[int]
        shares_list = [int(x.strip()) for x in shares.split(",")]

        payload = {
            "start_trade": str(start_trade),
            "end_trade": str(end_trade),
            "shares": shares_list,
            "cash": cash,
            "hmax": hmax,
            "model": model,
            "commission": commission,
            "reward_scaling": reward_scaling,
        }

        st.json(payload)  # Just show what will be sent

        # Send to FastAPI backend
        try:
            res = requests.post(f"{API_BASE}/trade/", json=payload)
            if res.status_code == 200:
                st.success("Simulation complete ✅")
                data = res.json()

                # ==========================
                # 📊 Part 1: res_trade table
                # ==========================
                if "res_trade" in data:
                    st.subheader("📊 Strategy Performance Comparison")
                    df_res_trade = pd.DataFrame(data["res_trade"]).T  # strategies as rows
                    df_res_trade = df_res_trade.apply(pd.to_numeric, errors='ignore')  # keep strings as is

                    # Format only numbers
                    st.dataframe(
                        df_res_trade.style.format(
                            {col: "{:.2f}" for col in df_res_trade.select_dtypes(include="number").columns}
                        )
                    )

                # ==========================
                # 📈 Part 2: trade_metrics charts
                # ==========================
                if "trade_metrics" in data:
                    st.subheader("📊 Trade Metrics")

                    df_metrics = pd.DataFrame(list(data["trade_metrics"].items()), columns=["Metric", "Value"])

                    # Format float numbers to 4 decimal places
                    df_metrics["Value"] = df_metrics["Value"].apply(
                        lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x
                    )

                    st.dataframe(df_metrics, use_container_width=True)

                if "actions" in data:
                    st.subheader("📋 Trade Actions")
                    
                    # Convert list of dicts to DataFrame
                    df_actions = pd.DataFrame(data["actions"])
                    
                    # Make sure numeric columns are formatted properly
                    for col in df_actions.select_dtypes(include=["float64", "int64"]).columns:
                        df_actions[col] = df_actions[col].apply(lambda x: f"{x:.2f}")
                    
                    # Show the table
                    st.dataframe(df_actions, use_container_width=True)
                    

            else:
                st.error(f"Backend error: {res.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# --- Chart Tab ---
with tab2:
    st.subheader("📈 Stock Price Chart")

    stock = st.selectbox("Select Stock", ["AAPL", "BA", "GS", "JPM"])
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    chart_type = st.radio("Chart Type", ["Line (Close)", "Candlestick"], horizontal=True)

    if st.button("Get Chart"):
        try:
            # 1) Try the simplest path: single-ticker history (no MultiIndex)
            df = yf.Ticker(stock).history(start=start_date, end=end_date)
            if df.empty:
                st.warning("No data found for selected date range.")
            else:
                df = df[['Open','High','Low','Close','Volume']].copy().reset_index()

                if chart_type == "Line (Close)":
                    fig = px.line(df, x="Date", y="Close", title=f"{stock} Closing Price")
                else:
                    fig = go.Figure(data=[go.Candlestick(
                        x=df["Date"],
                        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
                    )])
                    fig.update_layout(title=f"{stock} Candlestick")

                st.plotly_chart(fig, use_container_width=True)
                with st.expander("📋 Show Data Table"):
                    st.dataframe(df, use_container_width=True)

        except Exception as e:
            # 2) Fallback: handle MultiIndex from yf.download() and flatten to ticker-first
            try:
                df = yf.download([stock], start=start_date, end=end_date, group_by="ticker")

                if isinstance(df.columns, pd.MultiIndex):
                    # e.g. (Open, AAPL) -> "AAPL_Open"
                    df.columns = [f"{t}_{fld}" for fld, t in df.columns]
                else:
                    # Single-level: add ticker prefix to be consistent
                    df.rename(columns={c: f"{stock}_{c}" for c in df.columns}, inplace=True)

                df = df.reset_index()

                # choose correct column name regardless of shape
                close_col = f"{stock}_Close"
                open_col  = f"{stock}_Open"
                high_col  = f"{stock}_High"
                low_col   = f"{stock}_Low"

                if chart_type == "Line (Close)":
                    fig = px.line(df, x="Date", y=close_col, title=f"{stock} Closing Price")
                else:
                    fig = go.Figure(data=[go.Candlestick(
                        x=df["Date"],
                        open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col]
                    )])
                    fig.update_layout(title=f"{stock} Candlestick")

                st.plotly_chart(fig, use_container_width=True)
                with st.expander("📋 Show Data Table"):
                    st.dataframe(df, use_container_width=True)

            except Exception as e2:
                st.error(f"Error fetching stock data: {e2}")

import ta  # make sure you installed: pip install ta

# --- Analysis Tab ---
with tab3:
    st.subheader("📊 Technical Analysis")

    # Inputs
    stock = st.selectbox("Select Stock", ["AAPL", "BA", "GS", "JPM"], key="analysis_stock")
    start_date = st.date_input("Start Date", key="analysis_start")
    end_date = st.date_input("End Date", key="analysis_end")

    if st.button("Run Analysis"):
        try:
            # Fetch stock data
            df = yf.Ticker(stock).history(start=start_date, end=end_date)
            if df.empty:
                st.warning("No data found for selected date range.")
            else:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().reset_index()

                # =====================
                # Add Technical Indicators
                # =====================
                df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
                df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()

                macd = ta.trend.MACD(df["Close"])
                df["MACD"] = macd.macd()
                df["MACD_Signal"] = macd.macd_signal()
                df["MACD_Diff"] = macd.macd_diff()

                df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

                bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
                df["BB_High"] = bb.bollinger_hband()
                df["BB_Low"] = bb.bollinger_lband()

                # =====================
                # Plot Price + Indicators
                # =====================
                price_fig = go.Figure()
                price_fig.add_trace(go.Candlestick(
                    x=df["Date"], open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"],
                    name="Candlestick"
                ))
                price_fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["SMA20"], mode="lines", name="SMA20"
                ))
                price_fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20"
                ))
                price_fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["BB_High"], line=dict(dash="dot"), name="BB High"
                ))
                price_fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["BB_Low"], line=dict(dash="dot"), name="BB Low"
                ))

                price_fig.update_layout(title=f"{stock} Price with Indicators", xaxis_rangeslider_visible=False)
                st.plotly_chart(price_fig, use_container_width=True)

                # =====================
                # Plot RSI
                # =====================
                rsi_fig = px.line(df, x="Date", y="RSI", title=f"{stock} RSI (14)")
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(rsi_fig, use_container_width=True)

                # =====================
                # Plot MACD
                # =====================
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
                macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], mode="lines", name="Signal"))
                macd_fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Diff"], name="MACD Diff"))
                macd_fig.update_layout(title=f"{stock} MACD")
                st.plotly_chart(macd_fig, use_container_width=True)

                # =====================
                # Show Data Table
                # =====================
                with st.expander("📋 Show Data with Indicators"):
                    st.dataframe(df.tail(50), use_container_width=True)

        except Exception as e:
            st.error(f"Error in analysis: {e}")
