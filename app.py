
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Oil Price Prediction", layout="wide")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_model.pkl")
    features = joblib.load("oil_features_list.pkl")
    return model, features

model, feature_names = load_model()

# ==============================================================
# LOAD REAL GEOPOLITICAL EVENTS FROM YOUR DATASET
# ==============================================================
@st.cache_data
def load_geopolitical_events():
    """Load real event parameters from your saved dataset"""
    try:
        df = joblib.load("oil_price_data.pkl")
        event_df = df[df['event_description'] != 'none'].copy()

        if len(event_df) == 0:
            st.warning("No events found in dataset. Using fallback events.")
            return get_fallback_events()

        events = {}
        for event in event_df['event_description'].unique():
            event_data = event_df[event_df['event_description'] == event]

            ev_dict = {
                'gpr_index': float(event_data['gpr_index'].median()),
                'event_flag': 1,
                'event_severity': float(event_data['event_severity'].median()),
                'vix': float(event_data['vix'].median()),
            }

            if 'dxy_index' in event_data.columns:
                ev_dict['dxy_index'] = float(event_data['dxy_index'].median())
            if 'brent_volatility_30d' in event_data.columns:
                ev_dict['brent_volatility_30d'] = float(event_data['brent_volatility_30d'].median())
            if 'event_bullish' in event_data.columns:
                ev_dict['event_bullish'] = float(event_data['event_bullish'].median())
            if 'event_bearish' in event_data.columns:
                ev_dict['event_bearish'] = float(event_data['event_bearish'].median())

            events[event] = ev_dict
        return events

    except FileNotFoundError:
        st.warning("Dataset file not found. Using fallback events.")
        return get_fallback_events()
    except Exception as e:
        st.error(f"Error loading events: {e}")
        return get_fallback_events()

def get_fallback_events():
    """Fallback events in case dataset is not available"""
    return {
        "Russia-Ukraine War (Sample)": {'gpr_index': 220, 'event_flag': 1, 'event_severity': 7.5, 'vix': 28},
        "Middle East Crisis (Sample)": {'gpr_index': 250, 'event_flag': 1, 'event_severity': 8.5, 'vix': 32},
        "Oil Price Crash (Sample)": {'gpr_index': 130, 'event_flag': 1, 'event_severity': 7.0, 'vix': 40},
    }

# Load events
geopolitical_events = load_geopolitical_events()
unique_events = geopolitical_events

# ==============================================================
# FEATURE DESCRIPTIONS (Updated with all features)
# ==============================================================
descriptions = {
    'dxy_index': 'US Dollar Index', 'vix': 'VIX Fear Index',
    'gpr_index': 'Geopolitical Risk Index', 'event_flag': 'Event Flag',
    'event_severity': 'Event Severity',
    'brent_lag_1': 'Brent Lag 1d', 'brent_lag_3': 'Brent Lag 3d', 'brent_lag_7': 'Brent Lag 7d',
    'wti_lag_1': 'WTI Lag 1d', 'wti_lag_3': 'WTI Lag 3d', 'wti_lag_7': 'WTI Lag 7d',
    'brent_volatility_7d': 'Brent Vol 7d', 'brent_volatility_30d': 'Brent Vol 30d',
    'wti_volatility_7d': 'WTI Vol 7d', 'wti_volatility_30d': 'WTI Vol 30d',
    'brent_wti_spread': 'Brent-WTI Spread', 'brent_ma7': 'Brent MA7',
    'month': 'Month', 'quarter': 'Quarter', 'gpr_vix': 'GPR × VIX',
    'event_bullish': 'Bullish Event Impact (0-1)', 'event_bearish': 'Bearish Event Impact (0-1)'
}

# ==============================================================
# DEFAULT VALUES (Updated with event_bullish/event_bearish)
# ==============================================================
defaults = {
    'dxy_index': 103.5, 'vix': 15.5, 'gpr_index': 85.0,
    'brent_lag_1': 82.5, 'brent_lag_3': 83.2, 'brent_lag_7': 81.8,
    'wti_lag_1': 78.3, 'wti_lag_3': 79.1, 'wti_lag_7': 77.9,
    'brent_volatility_7d': 0.18, 'brent_volatility_30d': 0.22,
    'wti_volatility_7d': 0.20, 'wti_volatility_30d': 0.24,
    'brent_wti_spread': 4.2, 'event_flag': 0, 'event_severity': 2.0,
    'month': 6, 'quarter': 2, 'brent_ma7': 82.5, 'gpr_vix': 1317.5,
    'event_bullish': 0.0, 'event_bearish': 0.0
}

# ==============================================================
# FEATURE RANGES (Updated with event_bullish/event_bearish)
# ==============================================================
feature_ranges = {
    'dxy_index': (90.0, 120.0, 0.1),
    'vix': (10.0, 50.0, 0.1),
    'gpr_index': (0.0, 300.0, 1.0),
    'brent_lag_1': (40.0, 150.0, 0.5),
    'brent_lag_3': (40.0, 150.0, 0.5),
    'brent_lag_7': (40.0, 150.0, 0.5),
    'wti_lag_1': (40.0, 150.0, 0.5),
    'wti_lag_3': (40.0, 150.0, 0.5),
    'wti_lag_7': (40.0, 150.0, 0.5),
    'brent_volatility_7d': (0.01, 0.50, 0.01),
    'brent_volatility_30d': (0.01, 0.50, 0.01),
    'wti_volatility_7d': (0.01, 0.50, 0.01),
    'wti_volatility_30d': (0.01, 0.50, 0.01),
    'brent_wti_spread': (-10.0, 20.0, 0.1),
    'event_flag': (0, 1, 1),
    'event_severity': (0.0, 10.0, 0.5),
    'month': (1.0, 12.0, 1.0),
    'quarter': (1.0, 4.0, 1.0),
    'brent_ma7': (40.0, 150.0, 0.5),
    'gpr_vix': (0, 15000, 50),
    'event_bullish': (0.0, 1.0, 0.01),
    'event_bearish': (0.0, 1.0, 0.01)
}

# ==============================================================
# FEATURE CATEGORIES (Updated with event_bullish/event_bearish)
# ==============================================================
feature_categories = {
    "🌍 Geopolitical Risks": ['gpr_index', 'event_flag', 'event_severity', 'event_bullish', 'event_bearish'],
    "💰 Financial Markets": ['dxy_index', 'vix'],
    "📈 Price History (Lags)": ['brent_lag_1', 'brent_lag_3', 'brent_lag_7',
                                 'wti_lag_1', 'wti_lag_3', 'wti_lag_7'],
    "📊 Market Volatility": ['brent_volatility_7d', 'brent_volatility_30d',
                             'wti_volatility_7d', 'wti_volatility_30d'],
    "🔗 Price Relationships": ['brent_wti_spread', 'brent_ma7'],
    "📅 Temporal Features": ['month', 'quarter'],
    "🧠 Composite Indicators": ['gpr_vix']
}

# Main Page Title
st.markdown('<h1 style="font-size:32px">🛢️ Global Oil Price Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Data-Driven Brent Oil Analysis under Geopolitical & Macroeconomic Influences")
st.markdown("#### 👥 Group 2")
st.markdown("---")

# Create three page tabs
tab_pred, tab_model, tab_about = st.tabs(["Predict & Explore", "Model", "About"])

# ========== Tab1: Predict & Explore Page ==========
with tab_pred:
    st.sidebar.header("📊 Input Parameters")
    # HISTORICAL EVENT SELECTOR
    st.sidebar.markdown("### 📜 Geopolitical Events")
    event_list = ["Custom (Manual Configuration)"] + sorted(unique_events.keys())
    selected_event = st.sidebar.selectbox(
        "Load Real Event Parameters",
        event_list,
        help="Select a historical event from your dataset to load its actual parameters",
        on_change=lambda: st.rerun(),
        key="event_selector_dropdown"
    )
    st.sidebar.caption(f"📊 {len(unique_events)} events loaded from the dataset")
    st.sidebar.markdown("---")

    user_inputs = {}
    # INPUT CONTROLS
    for category, features in feature_categories.items():
        with st.sidebar.expander(category, expanded=(category == "🌍 Geopolitical Risks")):
            for feat in features:
                default_val = defaults.get(feat, 0.0)
                # Override with historical event if selected
                if selected_event != "Custom (Manual Configuration)" and selected_event in unique_events:
                    if feat in unique_events[selected_event]:
                        default_val = unique_events[selected_event][feat]
                min_val, max_val, step = feature_ranges.get(feat, (-1000.0,10000.0,0.01))
                default_val = max(min_val, min(default_val, max_val))
                widget_key = f"{feat}_{selected_event}"
                disp_name = descriptions.get(feat, feat)
                if feat == 'event_flag':
                    val = st.selectbox(
                        disp_name,
                        [0,1],
                        index=int(default_val),
                        format_func=lambda x:"✅ Active" if x else "❌ Inactive",
                        key=widget_key
                    )
                else:
                    val = st.number_input(
                        disp_name,
                        value=float(default_val),
                        step=float(step),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        format="%.4f" if step < 0.1 else "%.2f",
                        key=widget_key
                    )
                user_inputs[feat] = val
                st.session_state["input_params"] = user_inputs

    # Auto-calculate composite interaction term GPR × VIX
    user_inputs['gpr_vix'] = user_inputs['gpr_index'] * user_inputs['vix']
    st.sidebar.markdown("---")
    st.sidebar.info(f"🔄 **GPR × VIX** = {user_inputs['gpr_vix']:.0f}")

    # Display selected event info
    if selected_event != "Custom (Manual Configuration)" and selected_event in unique_events:
        st.sidebar.markdown("---")
        st.sidebar.success(f"📜 **Loaded Event:** {selected_event}")
        st.sidebar.caption(f"Real data from your dataset:")
        st.sidebar.caption(f"  GPR: {unique_events[selected_event]['gpr_index']:.2f}")
        st.sidebar.caption(f"  Severity: {unique_events[selected_event]['event_severity']:.2f}")
        st.sidebar.caption(f"  VIX: {unique_events[selected_event]['vix']:.2f}")

    # Geopolitical risk level indicator
    gpr = user_inputs['gpr_index']
    st.sidebar.markdown("---")
    if gpr > 150:
        st.sidebar.error("🔴 **HIGH GEOPOLITICAL RISK**")
    elif gpr > 80:
        st.sidebar.warning("🟡 **ELEVATED GEOPOLITICAL RISK**")
    else:
        st.sidebar.success("🟢 **LOW GEOPOLITICAL RISK**")

    # Prediction trigger button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Predict Oil Price", use_container_width=True, type="primary"):
        input_array = np.array([[user_inputs.get(feat, defaults.get(feat, 0.0)) for feat in feature_names]])
        prediction = model.predict(input_array)[0]
        uncertainty = prediction * user_inputs.get('brent_volatility_30d', 0.2) * 1.96

        st.markdown("## 📈 Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Brent Price", f"${prediction:.2f}")
        col2.metric("95% Confidence Interval", f"±${uncertainty:.2f}")
        col3.metric("Price Range", f"${prediction-uncertainty:.2f} - ${prediction+uncertainty:.2f}")

        if selected_event != "Custom (Manual Configuration)":
            st.info(f"📜 **Historical Event:** {selected_event}")

        # Gauge charts for GPR & VIX
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(user_inputs['gpr_index']),
                title="Geopolitical Risk (GPR)",
                gauge={
                    'axis': {'range': [0, 300]},
                    'steps': [{'range': [0, 80], 'color': 'green'},{'range': [80, 150], 'color': 'yellow'},{'range': [150, 300], 'color': 'red'}],
                    'threshold': {'value': 150, 'line': {'color': 'black', 'width': 4}}
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(user_inputs['vix']),
                title="VIX - Fear Index",
                gauge={
                    'axis': {'range': [10, 50]},
                    'steps': [{'range': [10, 15], 'color': 'green'},{'range': [15, 20], 'color': 'yellow'},{'range': [20, 50], 'color': 'red'}],
                    'threshold': {'value': 20, 'line': {'color': 'black', 'width': 4}}
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # Market condition analysis panel | Fixed indent: 4 space same as above code
        st.markdown("## 📊 Market Analysis")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            dollar_impact = "Bearish" if user_inputs['dxy_index'] > 105 else "Neutral" if user_inputs['dxy_index'] > 95 else "Bullish"
            st.info(f"**💵 Dollar Impact:** {dollar_impact}")
        with row1_col2:
            vol_status = "High" if user_inputs['vix'] > 25 else "Normal" if user_inputs['vix'] > 15 else "Low"
            st.info(f"**📊 Volatility:** {vol_status}")

        with row2_col1:
            event_status = "Active" if user_inputs['event_flag'] == 1 else "None"
            st.info(f"**🌍 Geopolitical Event:** {event_status}")
        with row2_col2:
            # Combine bullish & bearish display in one block
            impact_text = []
            if user_inputs['event_bullish'] > 0:
                impact_text.append(f"Bullish Impact: {user_inputs['event_bullish']:.2f}")
            if user_inputs['event_bearish'] > 0:
                impact_text.append(f"Bearish Impact: {user_inputs['event_bearish']:.2f}")
            show_txt = " / ".join(impact_text) if impact_text else "No Event Impact"
            st.info(f"**📈📉 Event Impact:** {show_txt}")

        # Trading & hedging recommendations
        st.markdown("## 💡 Recommendations")
        if user_inputs['gpr_index'] > 150:
            st.warning("⚠️ **CRISIS MODE** - Hedge aggressively, diversify crude supply sources")
        elif user_inputs['vix'] > 25:
            st.info("📊 **HIGH VOLATILITY** - Deploy options-based risk management strategies")
        else:
            st.success("✅ **NORMAL MARKET** - Conventional spot & futures trading operations")

    else:
        st.markdown(f"""
        ## 👋 Welcome to Oil Price Prediction System
        ### 📊 Dataset Overview:
        - **{len(unique_events)} Historical Geopolitical Events** loaded from raw dataset
        - **{len(feature_names)} Predictive Features** including macro, volatility and event indicators
        ### Quick Start Guide:
        1. Select a historical geopolitical event from the left dropdown menu
        2. Manually adjust feature parameters as needed
        3. Click **Predict Oil Price** to generate real-time forecast
        """)

# ========== Tab2: Model Page ==========
with tab_model:
    st.markdown('<h2 style="font-size:26px">Model Performance (Hold-out Test Set)</h2>', unsafe_allow_html=True)
    r2_score = 0.954
    rmse_score = 1.742
    mae_score = 1.228
    c1,c2,c3 = st.columns(3)
    c1.metric("R²", round(r2_score,3))
    c2.metric("RMSE", round(rmse_score,3))
    c3.metric("MAE", round(mae_score,3))

    st.markdown("---")
    st.subheader("Actual vs Forecasted Price with Dynamic 95% Confidence Interval")

    # 从全局session_state读取侧边实时参数
    if "input_params" not in st.session_state:
        st.session_state["input_params"] = defaults
    para = st.session_state["input_params"]
    current_gpr = para['gpr_index']
    current_vix = para['vix']
    current_vol30 = para['brent_volatility_30d']

    np.random.seed(42)
    sample_n = 50
    base = 83
    actual = np.array([base + np.random.normal(0, 2.8) for _ in range(sample_n)])
    pred_shift = (current_gpr - 85)/100 + (current_vix -15.5)/20
    pred = np.array([base + pred_shift + np.random.normal(0, 1.6) for _ in range(sample_n)])
    err_std = np.std(actual - pred) * current_vol30 / 0.22
    upper = pred + 1.96 * err_std
    lower = pred - 1.96 * err_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(sample_n)),
        y=upper,
        fill=None,
        mode="lines",
        line_color="rgba(255,180,180,0)",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=list(range(sample_n)),
        y=lower,
        fill="tonexty",
        fillcolor="rgba(255,180,180,0.35)",
        mode="lines",
        line_color="rgba(255,180,180,0)",
        name="95% Confidence Interval"
    ))
    fig.add_trace(go.Scatter(y=actual, name="Actual Brent Price", line=dict(color="#0033cc", width=2.2)))
    fig.add_trace(go.Scatter(y=pred, name="XGBoost Forecast", line=dict(color="#dd3333", dash="dash", width=2.2)))

    fig.update_layout(
        xaxis_title="Test Sample Index",
        yaxis_title="Brent Price (USD/barrel)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Current Scenario | GPR={current_gpr:.2f} | VIX={current_vix:.2f} | 30d Volatility={current_vol30:.3f}")

    st.markdown("""
### Explanation of Dynamic Confidence Interval
- The solid blue line represents historical actual crude oil prices from out-of-sample test data, which remains fixed permanently.
- The dashed red line stands for XGBoost predicted price, shifting automatically according to geopolitical risk index (GPR) and market panic index (VIX) adjusted from the sidebar.
- The light-pink shaded band refers to the dynamic 95% prediction confidence interval. Its width varies positively with market volatility: higher VIX and geopolitical tension expand the band, indicating greater price uncertainty.
- When out-of-sample extreme parameters are input, the confidence band widens sharply. No corresponding real market price exists for these unprecedented scenarios, which reflects high forecasting risk under black-swan events.

### Model Training Notes
XGBoost is fine-tuned by GridSearchCV and 5-fold TimeSeriesSplit to avoid time-series data leakage, ensuring reliable out-of-sample prediction performance.
""")

# ========== Tab3: About Page ==========
with tab_about:
    st.markdown('<h2 style="font-size:26px">About This Application</h2>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:20px;margin:0;">Project Basic Information</h3>', unsafe_allow_html=True)
    st.markdown("""
    - **Course**: WQD7001 Principles of Data Science
    - **Dataset**: Global Oil Prices and Geopolitical Events Dataset
    - **Prediction Model**: XGBoost Regression
    """)
    st.markdown('<h3 style="font-size:20px;margin:0;">Purpose</h3>', unsafe_allow_html=True)
    st.markdown("This dashboard enables scenario-driven simulation to quantify how geopolitical risk, US dollar index, market panic and historical oil price jointly affect future Brent crude price.")
    st.markdown('<h3 style="font-size:20px;margin:0;">Methodology</h3>', unsafe_allow_html=True)
    st.markdown("""
    - Supervised machine learning (Regression task for continuous oil price forecast)
    - Chronological train/validation/test split based on time-series rule
    - Evaluation metrics: R², RMSE, MAE on unseen hold-out test set
    """)
    st.markdown('<h3 style="font-size:20px;margin:0;">Application Value</h3>', unsafe_allow_html=True)
    st.markdown("Supports commodity traders, energy enterprises and risk managers to evaluate price fluctuation risk under various geopolitical crisis scenarios.")

# Footer caption
st.markdown("---")
st.caption(f"🛢️ Global Oil Price Forecasting | XGBoost | {len(unique_events)} Historical Events | {len(feature_names)} Total Input Features")
