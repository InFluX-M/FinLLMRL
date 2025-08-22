import streamlit as st
import struct
import requests
import os
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8000"

# -----------------------
# Wake word start button
# -----------------------
st.title("🏠 InFluX Trader")

# -----------------------
# Tabs become available
# -----------------------
tab1, tab2, tab3 = st.tabs(["💬 RL suggest", "🎙️ Chart", "🎤 Analysis"])

# --- RL Suggest Tab ---
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
        model = st.selectbox("RL Model", ["ppo", "rppo"])
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
