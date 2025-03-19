import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Economic Data Properties Guide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .property-header {
        font-size: 1.4rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        background-color: #EFF6FF;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .definition {
        background-color: #F3F4F6;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .example-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-left: 4px solid #10B981;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-left: 4px solid #EF4444;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .domain-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #DBEAFE;
    }
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Comprehensive Guide to Economic Data Properties</h1>", unsafe_allow_html=True)
st.markdown("""
This guide serves as a reference for researchers working with different types of economic data.
It covers key properties, challenges, and considerations across various data types and economic domains.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a section:",
    ["Introduction",
     "Time Series Data",
     "Panel Data",
     "Cross-Sectional Data",
     "Domain-Specific Properties",
     "Statistical Tests",
     "Visualization Examples",
     "References"]
)


# Function to generate sample data
def generate_sample_data(data_type):
    np.random.seed(42)

    if data_type == "time_series":
        # Generate time series with trend and seasonality
        dates = pd.date_range(start="2015-01-01", periods=365, freq="D")
        trend = np.linspace(0, 10, 365)
        seasonality = 5 * np.sin(np.linspace(0, 12 * np.pi, 365))
        noise = np.random.normal(0, 1, 365)

        # Add structural break
        break_point = 180
        after_break = np.zeros(365)
        after_break[break_point:] = 5

        # Create volatility clusters
        vol_cluster = np.ones(365)
        vol_cluster[50:100] = 3  # Higher volatility period
        vol_cluster[200:250] = 4  # Even higher volatility period

        # Final series with non-stationarity, structural breaks, and volatility clustering
        y = trend + seasonality + noise * vol_cluster + after_break

        return pd.DataFrame({"Date": dates, "Value": y})

    elif data_type == "panel":
        # Generate panel data
        entities = 10
        time_periods = 20
        entity_ids = np.repeat(np.arange(1, entities + 1), time_periods)
        time_ids = np.tile(np.arange(1, time_periods + 1), entities)

        # Entity fixed effects (heterogeneity)
        entity_effects = np.repeat(np.random.normal(0, 5, entities), time_periods)

        # Time trend (non-stationarity)
        time_trend = np.tile(np.linspace(0, 5, time_periods), entities)

        # Random component
        random_effect = np.random.normal(0, 1, entities * time_periods)

        # Final value with heterogeneity
        y = 10 + entity_effects + time_trend + random_effect

        return pd.DataFrame({"Entity": entity_ids, "Time": time_ids, "Value": y})

    elif data_type == "cross_sectional":
        # Generate cross-sectional data with outliers and skewness
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # Create skewed variable (income-like)
        income = np.exp(np.random.normal(10, 0.5, n))

        # Add outliers
        income[np.random.choice(n, 5)] = income[np.random.choice(n, 5)] * 10

        # Create heteroscedastic relationship
        y = 2 * x1 + 3 * x2 + np.random.normal(0, np.abs(x1), n) + np.log(income / 1000)

        return pd.DataFrame({"X1": x1, "X2": x2, "Income": income, "Y": y})


# Generate sample data for visualizations
ts_data = generate_sample_data("time_series")
panel_data = generate_sample_data("panel")
cs_data = generate_sample_data("cross_sectional")


# Function to create downloadable plot
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">{text}</a>'
    return href


# Introduction
if page == "Introduction":
    st.markdown("<h2 class='sub-header'>Introduction to Economic Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    Economic data comes in various forms and exhibits different properties depending on its type and source. 
    Understanding these properties is essential for selecting appropriate analytical methods, ensuring valid results,
    and drawing accurate conclusions.

    This guide categorizes economic data properties across three main types:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="domain-box">
        <h3>Time Series Data</h3>
        <p>Sequential observations collected over time from the same source.</p>
        <p><b>Key properties:</b> Non-stationarity, Autocorrelation, Seasonality, Volatility clustering, Structural breaks</p>
        <p><b>Examples:</b> GDP growth, Inflation rates, Stock prices, Interest rates</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="domain-box">
        <h3>Panel Data</h3>
        <p>Data containing observations of multiple entities across several time periods.</p>
        <p><b>Key properties:</b> Heterogeneity, Cross-sectional dependence, Serial correlation, Non-stationarity</p>
        <p><b>Examples:</b> Country-level data over time, Firm financial data across years, Household surveys over multiple periods</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="domain-box">
        <h3>Cross-Sectional Data</h3>
        <p>Observations of multiple entities at a single point in time.</p>
        <p><b>Key properties:</b> Heteroskedasticity, Outliers, Skewness, Non-normality</p>
        <p><b>Examples:</b> Consumer surveys, Census data, One-time economic surveys</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    Additionally, we explore how these properties manifest differently across economic domains:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="domain-box">
        <h3>Financial Data</h3>
        <p>High-frequency data with unique properties.</p>
        <p><b>Key concerns:</b> Volatility clustering, Fat tails, Leverage effects, Market microstructure</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="domain-box">
        <h3>Macroeconomic Data</h3>
        <p>Broad economic indicators at national or regional levels.</p>
        <p><b>Key concerns:</b> Non-stationarity, Structural breaks, Co-integration, Seasonal patterns</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="domain-box">
        <h3>Microeconomic Data</h3>
        <p>Data concerning individual economic agents.</p>
        <p><b>Key concerns:</b> Heterogeneity, Selection bias, Endogeneity, Distributional properties</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### How to Use This Guide

    This application provides:

    1. **Definitions and explanations** of key properties and challenges associated with different types of economic data
    2. **Visual examples** illustrating these properties
    3. **Testing methodologies** for detecting various data characteristics
    4. **Domain-specific considerations** for financial, macro, and microeconomic data
    5. **Recommendations** for handling problematic data features

    Navigate through the different sections using the sidebar to explore specific topics in depth.
    """)

# Time Series Data
elif page == "Time Series Data":
    st.markdown("<h2 class='sub-header'>Time Series Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    Time series data consists of sequential observations collected at regular time intervals.
    In economics, time series data is prevalent in tracking economic indicators, financial markets,
    and other variables that evolve over time.
    """)

    # Non-Stationarity
    st.markdown("<h3 class='property-header'>Non-Stationarity</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> A non-stationary time series has statistical properties (mean, variance, autocorrelation) 
        that change over time. It does not revert to a constant mean or have a constant variance.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>GDP levels (typically show upward trends)</li>
            <li>Price indices (generally increase over time)</li>
            <li>Asset prices (often exhibit random walks)</li>
        </ul>

        <p><b>Why It Matters:</b> Non-stationarity can lead to spurious regressions, unreliable hypothesis tests, 
        and incorrect forecasts. Most time series models require stationarity or appropriate transformations.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Differencing (first or higher order)</li>
            <li>Detrending</li>
            <li>Taking logarithms (for variance stabilization)</li>
            <li>Using models specifically designed for non-stationary data (ARIMA, Error Correction Models)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Plot a non-stationary time series
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts_data["Date"], ts_data["Value"], linewidth=2)
        ax.set_title("Non-Stationary Time Series Example")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Plot stationary version (differenced)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts_data["Date"][1:], np.diff(ts_data["Value"]), linewidth=2, color="green")
        ax.set_title("After First-Differencing (More Stationary)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Differenced Value")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Structural Breaks
    st.markdown("<h3 class='property-header'>Structural Breaks</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Structural breaks are sudden, significant changes in the pattern or parameters of a time series. 
        They represent fundamental shifts in the underlying data-generating process.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Policy regime changes (e.g., monetary policy shifts)</li>
            <li>Economic crises (e.g., 2008 financial crisis)</li>
            <li>Major regulatory changes</li>
            <li>Technological disruptions</li>
        </ul>

        <p><b>Why It Matters:</b> Unaccounted structural breaks can lead to poor model fit, inaccurate forecasts, 
        and invalid statistical inference. They can create apparent non-stationarity even when none exists within regimes.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Testing for breaks (Chow test, CUSUM, Quandt-Andrews)</li>
            <li>Dummy variables for known break points</li>
            <li>Regime-switching models</li>
            <li>Separate models for different regimes</li>
            <li>Time-varying parameter models</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Plot showing structural break
        fig, ax = plt.subplots(figsize=(8, 4))

        # Add vertical line at break point
        break_point = ts_data["Date"][180]

        ax.plot(ts_data["Date"], ts_data["Value"], linewidth=2)
        ax.axvline(x=break_point, color='red', linestyle='--',
                   label="Structural Break")
        ax.set_title("Time Series with Structural Break")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Volatility Clustering
    st.markdown("<h3 class='property-header'>Volatility Clustering</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Volatility clustering occurs when periods of high volatility tend to be followed by high volatility, 
        and periods of low volatility by low volatility. It represents a form of heteroskedasticity in time series.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Financial asset returns (stocks, bonds, exchange rates)</li>
            <li>Commodity prices</li>
            <li>Inflation rates during unstable periods</li>
        </ul>

        <p><b>Why It Matters:</b> Standard time series models assume constant variance, which is violated with volatility clustering. 
        This affects forecasting accuracy and risk assessment.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>ARCH/GARCH family models</li>
            <li>Stochastic volatility models</li>
            <li>Regime-switching models with different volatility states</li>
            <li>Rolling window volatility estimation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create volatility clustering example
        fig, ax = plt.subplots(figsize=(8, 4))

        # Calculate returns
        returns = np.diff(ts_data["Value"]) / ts_data["Value"][:-1]

        ax.plot(ts_data["Date"][1:], returns, linewidth=1)
        ax.set_title("Returns Showing Volatility Clustering")
        ax.set_xlabel("Time")
        ax.set_ylabel("Returns")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Show squared returns to highlight volatility clusters
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts_data["Date"][1:], returns ** 2, linewidth=1, color="purple")
        ax.set_title("Squared Returns (Volatility Proxy)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Squared Returns")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Seasonality
    st.markdown("<h3 class='property-header'>Seasonality</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Seasonality refers to regular and predictable patterns or fluctuations that recur with a fixed period, 
        usually within a year (quarterly, monthly, weekly, or daily).</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Retail sales (holiday seasons)</li>
            <li>Housing market activity (stronger in spring/summer)</li>
            <li>Agricultural production (harvest cycles)</li>
            <li>Tourism and travel (vacation periods)</li>
            <li>Energy consumption (seasonal temperature variations)</li>
        </ul>

        <p><b>Why It Matters:</b> Seasonality can mask underlying trends and cycles, making it difficult to identify 
        the actual direction of an economic series or to compare different time periods.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Seasonal adjustment techniques (X-12-ARIMA, TRAMO/SEATS)</li>
            <li>Seasonal dummy variables</li>
            <li>Seasonal differencing</li>
            <li>Seasonal ARIMA (SARIMA) models</li>
            <li>Year-over-year comparisons</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create seasonal data
        dates = pd.date_range(start="2020-01-01", periods=365 * 2, freq="D")
        trend = np.linspace(0, 10, 365 * 2)
        seasonality = 5 * np.sin(np.linspace(0, 24 * np.pi, 365 * 2))
        noise = np.random.normal(0, 0.5, 365 * 2)
        seasonal_data = trend + seasonality + noise

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, seasonal_data, linewidth=1)
        ax.set_title("Time Series with Seasonal Pattern")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Seasonal component
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, seasonality, linewidth=1, color="green")
        ax.set_title("Extracted Seasonal Component")
        ax.set_xlabel("Time")
        ax.set_ylabel("Seasonal Effect")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Autocorrelation
    st.markdown("<h3 class='property-header'>Autocorrelation</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Autocorrelation is the correlation of a time series with its own past values. 
        It indicates the degree to which a value depends on previous values in the series.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>GDP growth (momentum in economic growth)</li>
            <li>Unemployment rates (labor market persistence)</li>
            <li>Interest rates (central bank smoothing)</li>
            <li>Most macroeconomic indicators</li>
        </ul>

        <p><b>Why It Matters:</b> Autocorrelation violates the independence assumption of many statistical methods. 
        It affects standard errors, hypothesis testing, and efficiency of estimates.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>ARIMA models</li>
            <li>HAC standard errors (Newey-West, etc.)</li>
            <li>Generalized Least Squares (GLS)</li>
            <li>Including lags as explanatory variables</li>
            <li>Cochrane-Orcutt procedure</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Plot autocorrelation
        from statsmodels.graphics.tsaplots import plot_acf

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(ts_data["Value"], ax=ax, lags=20)
        ax.set_title("Autocorrelation Function (ACF)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Plot partial autocorrelation
        from statsmodels.graphics.tsaplots import plot_pacf

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(ts_data["Value"], ax=ax, lags=20)
        ax.set_title("Partial Autocorrelation Function (PACF)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Fat Tails
    st.markdown("<h3 class='property-header'>Fat Tails (Leptokurtic Distributions)</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Fat tails (leptokurtosis) occur when extreme values appear more frequently than would be 
        expected under a normal distribution. The distribution has excess kurtosis, meaning it has a higher peak and 
        heavier tails than the normal distribution.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Financial returns (stock market crashes)</li>
            <li>Exchange rate movements (currency crises)</li>
            <li>Commodity price changes (supply shocks)</li>
        </ul>

        <p><b>Why It Matters:</b> Fat tails imply higher probability of extreme events than predicted by normal distribution. 
        Models assuming normality will underestimate risk and the likelihood of extreme events.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Using distributions that account for fat tails (Student's t, Generalized Error Distribution, etc.)</li>
            <li>GARCH models with non-normal error distributions</li>
            <li>Extreme Value Theory (EVT)</li>
            <li>Quantile regression for better tail estimation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate normal vs fat-tailed distribution
        x_normal = np.random.normal(0, 1, 10000)
        df = 3  # Degrees of freedom
        x_t = np.random.standard_t(df, 10000)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x_normal, ax=ax, label="Normal Distribution")
        sns.kdeplot(x_t, ax=ax, label="t-Distribution (Fat Tails)")
        ax.set_title("Normal vs. Fat-Tailed Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # QQ plot to show deviation from normality
        fig, ax = plt.subplots(figsize=(8, 4))
        stats.probplot(x_t, dist="norm", plot=ax)
        ax.set_title("QQ Plot: Fat-Tailed Distribution vs. Normal")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Leverage Effect
    st.markdown("<h3 class='property-header'>Leverage Effect</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> The leverage effect refers to the negative correlation between returns and volatility, 
        where negative returns tend to be followed by higher volatility than positive returns of the same magnitude.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Stock returns (market downturns often lead to increased volatility)</li>
            <li>Corporate bond spreads (widen during crises)</li>
            <li>Risk premiums in asset pricing</li>
        </ul>

        <p><b>Why It Matters:</b> The leverage effect creates asymmetric volatility responses to shocks, 
        which standard symmetric models fail to capture. This affects risk assessment and option pricing.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Asymmetric GARCH models (EGARCH, GJR-GARCH)</li>
            <li>Stochastic volatility models with correlation parameter</li>
            <li>Separate modeling of upside and downside risks</li>
            <li>Realized volatility measures with directional components</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create data showing leverage effect
        np.random.seed(123)
        n = 1000
        returns = np.zeros(n)
        volatility = np.zeros(n)
        volatility[0] = 0.1

        for i in range(1, n):
            # Higher volatility after negative returns (leverage effect)
            if returns[i - 1] < 0:
                volatility[i] = 0.9 * volatility[i - 1] + 0.1 * 0.2
            else:
                volatility[i] = 0.9 * volatility[i - 1] + 0.1 * 0.05

            returns[i] = np.random.normal(0, volatility[i])

        # Plot returns
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(returns, linewidth=1)
        ax.set_title("Returns with Leverage Effect")
        ax.set_xlabel("Time")
        ax.set_ylabel("Returns")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Plot volatility
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(volatility, linewidth=1, color="red")
        ax.set_title("Volatility Process")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.markdown("""
    ### Recommendations for Time Series Analysis in Economics

    1. **Always test for stationarity** before applying standard time series models
    2. **Consider structural breaks** when analyzing long economic time series
    3. **Account for seasonality** explicitly in modeling and interpretation
    4. **Test for autocorrelation** and use appropriate methods when it's present
    5. **For financial time series**, explicitly model volatility clustering and fat tails
    6. **Consider transformations** (logs, differences) to address non-stationarity and normalize distributions
    7. **Validate model assumptions** through residual diagnostics
    """)

# Panel Data
elif page == "Panel Data":
    st.markdown("<h2 class='sub-header'>Panel Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    Panel data (also known as longitudinal data) combines both cross-sectional and time series dimensions,
    tracking multiple entities (individuals, firms, countries) over multiple time periods. This data structure
    offers unique analytical possibilities but also comes with specific challenges.
    """)

    # Heterogeneity
    st.markdown("<h3 class='property-header'>Heterogeneity</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Heterogeneity in panel data refers to systematic differences in characteristics and behavior across 
        entities. Each individual, firm, or country has unique attributes that affect the dependent variable.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Country-specific institutional factors affecting economic growth</li>
            <li>Firm-specific management practices influencing productivity</li>
            <li>Individual-specific preferences or abilities affecting income</li>
        </ul>

        <p><b>Why It Matters:</b> Ignoring heterogeneity can lead to biased coefficients, incorrect inference, and omitted 
        variable bias. Proper modeling of heterogeneity is often the main reason for using panel data.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Fixed effects models (entity and/or time fixed effects)</li>
            <li>Random effects models (when appropriate)</li>
            <li>Mixed effects models</li>
            <li>Including entity-specific variables</li>
            <li>Hierarchical/multilevel models</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Plot showing entity heterogeneity
        entities_to_plot = [1, 3, 5, 7, 9]

        fig, ax = plt.subplots(figsize=(8, 5))

        for entity in entities_to_plot:
            entity_data = panel_data[panel_data["Entity"] == entity]
            ax.plot(entity_data["Time"], entity_data["Value"],
                    label=f"Entity {entity}", marker='o', markersize=4)

        ax.set_title("Heterogeneity Across Entities")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Dynamic Effects
    st.markdown("<h3 class='property-header'>Dynamic Effects</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Dynamic effects refer to the influence of past values of variables on their current values.
        In panel data, both lagged dependent variables and persistent effects of independent variables are important.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Persistence in unemployment rates</li>
            <li>Adjustment costs in investment decisions</li>
            <li>Habit formation in consumption</li>
            <li>Partial adjustment processes in economic systems</li>
        </ul>

        <p><b>Why It Matters:</b> Static panel models can give misleading results when the underlying process is dynamic. 
        Including lagged variables can help capture adjustment processes, distributed lag effects, and habit persistence.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Dynamic panel models (including lagged dependent variables)</li>
            <li>GMM estimators (Arellano-Bond, system GMM)</li>
            <li>Distributed lag specifications</li>
            <li>Error correction models for panel data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create dynamic panel example
        np.random.seed(42)
        entities = 5
        time_periods = 20

        # Initialize empty DataFrame
        dynamic_panel = pd.DataFrame()

        for e in range(1, entities + 1):
            y = np.zeros(time_periods)
            y[0] = np.random.normal(10, 2)  # Initial value

            for t in range(1, time_periods):
                # AR(1) process with entity-specific mean
                y[t] = 2 + 0.8 * y[t - 1] + np.random.normal(0, 1)

            entity_df = pd.DataFrame({
                "Entity": e,
                "Time": range(1, time_periods + 1),
                "Value": y
            })

            dynamic_panel = pd.concat([dynamic_panel, entity_df])

        # Plot dynamic panel data
        fig, ax = plt.subplots(figsize=(8, 5))

        for entity in range(1, entities + 1):
            entity_data = dynamic_panel[dynamic_panel["Entity"] == entity]
            ax.plot(entity_data["Time"], entity_data["Value"],
                    label=f"Entity {entity}", marker='o', markersize=4)

        ax.set_title("Dynamic Panel Data (AR(1) Process)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Cross-Sectional Dependence
    st.markdown("<h3 class='property-header'>Cross-Sectional Dependence</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Cross-sectional dependence occurs when the error terms across different entities 
        are correlated. This often happens due to common shocks, spatial effects, or network connections.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Global financial crises affecting multiple countries</li>
            <li>Technology spillovers between firms in the same industry</li>
            <li>Regional economic shocks affecting neighboring areas</li>
            <li>Supply chain disruptions affecting interconnected firms</li>
        </ul>

        <p><b>Why It Matters:</b> Standard panel estimators assume cross-sectional independence. Violation of this 
        assumption leads to inefficient estimates and invalid inference. Tests and confidence intervals become unreliable.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Driscoll-Kraay standard errors</li>
            <li>Common Correlated Effects (CCE) estimator</li>
            <li>Spatial panel models</li>
            <li>Time fixed effects (for common shocks)</li>
            <li>Factor models for panel data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create data with cross-sectional dependence
        np.random.seed(123)
        entities = 8
        time_periods = 30

        # Common factor affecting all entities
        common_factor = np.sin(np.linspace(0, 4 * np.pi, time_periods)) + np.random.normal(0, 0.5, time_periods)

        # Initialize empty DataFrame
        cs_dependent_panel = pd.DataFrame()

        for e in range(1, entities + 1):
            # Entity-specific sensitivity to common factor plus idiosyncratic component
            sensitivity = np.random.uniform(0.5, 1.5)
            idiosyncratic = np.random.normal(0, 1, time_periods)

            y = 5 + sensitivity * common_factor + idiosyncratic

            entity_df = pd.DataFrame({
                "Entity": e,
                "Time": range(1, time_periods + 1),
                "Value": y
            })

            cs_dependent_panel = pd.concat([cs_dependent_panel, entity_df])

        # Plot to show common factor influence
        fig, ax = plt.subplots(figsize=(8, 5))

        for entity in range(1, entities + 1):
            entity_data = cs_dependent_panel[cs_dependent_panel["Entity"] == entity]
            ax.plot(entity_data["Time"], entity_data["Value"], alpha=0.5, linewidth=1)

        # Plot the common factor (scaled)
        scaled_factor = 5 + common_factor * np.mean([0.5, 1.5])
        ax.plot(range(1, time_periods + 1), scaled_factor, 'k--', linewidth=2, label="Common Factor")

        ax.set_title("Cross-Sectional Dependence Example")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Non-Stationarity in Panel Data
    st.markdown("<h3 class='property-header'>Non-Stationarity in Panel Data</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Panel non-stationarity occurs when the time series dimension of the panel data exhibits 
        unit roots or stochastic trends. It combines the challenges of time series non-stationarity with panel heterogeneity.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>GDP levels across countries over time</li>
            <li>Stock prices for multiple firms</li>
            <li>Exchange rates for various currency pairs</li>
            <li>Price levels or monetary aggregates across regions</li>
        </ul>

        <p><b>Why It Matters:</b> Non-stationarity in panel data can lead to spurious regressions. Standard panel 
        estimators may produce misleading results when applied to non-stationary panels.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Panel unit root tests (Levin-Lin-Chu, Im-Pesaran-Shin, etc.)</li>
            <li>Panel cointegration techniques</li>
            <li>First-differencing or de-trending each time series</li>
            <li>Panel Vector Error Correction Models</li>
            <li>Mean group estimators for heterogeneous panels</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Create non-stationary panel data
        np.random.seed(42)
        entities = 5
        time_periods = 40

        # Initialize empty DataFrame
        nonstationary_panel = pd.DataFrame()

        for e in range(1, entities + 1):
            # Random walk with drift
            drift = np.random.uniform(0.05, 0.15)
            shocks = np.random.normal(0, 1, time_periods)

            y = np.zeros(time_periods)
            y[0] = 10 + np.random.normal(0, 1)

            for t in range(1, time_periods):
                y[t] = y[t - 1] + drift + shocks[t]

            entity_df = pd.DataFrame({
                "Entity": e,
                "Time": range(1, time_periods + 1),
                "Value": y
            })

            nonstationary_panel = pd.concat([nonstationary_panel, entity_df])

        # Plot non-stationary panel data
        fig, ax = plt.subplots(figsize=(8, 5))

        for entity in range(1, entities + 1):
            entity_data = nonstationary_panel[nonstationary_panel["Entity"] == entity]
            ax.plot(entity_data["Time"], entity_data["Value"],
                    label=f"Entity {entity}")

        ax.set_title("Non-Stationary Panel Data (Random Walks with Drift)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Unbalanced Panels
    st.markdown("<h3 class='property-header'>Unbalanced Panels</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="definition">
    <p><b>Definition:</b> An unbalanced panel occurs when not all entities are observed in all time periods. 
    This may be due to attrition, late entry, or gaps in data collection.</p>

    <p><b>Examples in Economics:</b></p>
    <ul>
        <li>Firm panels with entry and exit of companies</li>
        <li>Household surveys with non-response or attrition</li>
        <li>Country panels with missing data for certain periods or variables</li>
    </ul>

    <p><b>Why It Matters:</b> Unbalanced panels can create selection bias if the missingness is not random. 
    Some estimation methods require balanced panels or special treatments for unbalanced data.</p>

    <p><b>How to Address:</b></p>
    <ul>
        <li>Use methods that accommodate unbalanced panels (most modern estimators do)</li>
        <li>Test for selection bias</li>
        <li>Imputation techniques for missing data (with caution)</li>
        <li>Inverse probability weighting</li>
        <li>Consider creating a balanced sub-panel if appropriate</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create unbalanced panel visualization
    np.random.seed(123)
    entities = 10
    max_time_periods = 20

    # Initialize empty DataFrame
    unbalanced_panel = pd.DataFrame()
    panel_presence = np.zeros((entities, max_time_periods))

    for e in range(1, entities + 1):
        # Randomly determine when entity enters and exits
        entry_time = np.random.randint(1, 8)
        exit_time = np.random.randint(entry_time + 5, max_time_periods + 1)

        # Mark presence in panel
        panel_presence[e - 1, entry_time - 1:exit_time] = 1

        # Generate some random values
        time_points = range(entry_time, exit_time + 1)
        values = 10 + np.random.normal(0, 1, len(time_points))

        entity_df = pd.DataFrame({
            "Entity": e,
            "Time": time_points,
            "Value": values
        })

        unbalanced_panel = pd.concat([unbalanced_panel, entity_df])

    # Plot panel presence
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(panel_presence, cmap='Blues', aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_yticks(range(entities))
    ax.set_yticklabels([f"Entity {i + 1}" for i in range(entities)])
    ax.set_xticks(range(0, max_time_periods, 2))
    ax.set_xticklabels(range(1, max_time_periods + 1, 2))

    ax.set_xlabel("Time Period")
    ax.set_title("Unbalanced Panel Visualization\n(Blue indicates entity is present in dataset)")

    # Add a colorbar
    plt.colorbar(cax, ticks=[0, 1], orientation='vertical', label='Presence in Panel')

    st.pyplot(fig)

    st.markdown("""
    ### Recommendations for Panel Data Analysis in Economics

    1. **Test for heterogeneity** and use appropriate fixed or random effects models
    2. **Consider the Hausman test** to decide between fixed and random effects
    3. **Test for cross-sectional dependence** in macro panels
    4. **For dynamic relationships**, use appropriate estimators like Arellano-Bond
    5. **With non-stationary panels**, test for unit roots and consider panel cointegration
    6. **For short panels** (small T, large N), traditional panel methods work well
    7. **For long panels** (large T, small N), consider time series properties more carefully
    8. **With unbalanced panels**, check if missingness is random or systematic
    """)

# Cross-Sectional Data
elif page == "Cross-Sectional Data":
    st.markdown("<h2 class='sub-header'>Cross-Sectional Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    Cross-sectional data captures information from multiple entities (individuals, firms, countries)
    at a single point in time. While it lacks the temporal dimension of time series or panel data,
    cross-sectional data has its own unique properties and challenges.
    """)

    # Heteroskedasticity
    st.markdown("<h3 class='property-header'>Heteroskedasticity</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Heteroskedasticity refers to the varying variance of error terms across observations.
        It occurs when the variability of a variable differs across values of a predictor variable.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Variance of consumption often increases with income</li>
            <li>Forecast errors tend to be larger for larger firms</li>
            <li>Price volatility may differ across different market segments</li>
            <li>Error variance in wage equations may vary with education level</li>
        </ul>

        <p><b>Why It Matters:</b> Heteroskedasticity violates the constant variance assumption of OLS.
        While OLS estimates remain unbiased, they are no longer the most efficient, and standard errors become biased,
        affecting hypothesis testing and confidence intervals.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Robust standard errors (White, HC standard errors)</li>
            <li>Weighted Least Squares (WLS)</li>
            <li>Transforming the dependent variable (often log transformation)</li>
            <li>Heteroskedasticity-consistent covariance matrix estimators</li>
            <li>Bootstrapping</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate heteroskedastic data
        np.random.seed(42)
        x = np.linspace(0, 10, 200)
        error_var = 0.1 + 0.5 * x  # Error variance increases with x
        y = 2 + 0.5 * x + np.random.normal(0, np.sqrt(error_var))

        # Plot heteroskedastic data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.6)

        # Add regression line
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax.plot(x, intercept + slope * x, 'r', label='Regression Line')

        ax.set_title("Heteroskedastic Data Example")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Show residuals to highlight heteroskedasticity
        fig, ax = plt.subplots(figsize=(8, 5))
        residuals = y - (intercept + slope * x)
        ax.scatter(x, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='-')

        ax.set_title("Residuals Plot Showing Heteroskedasticity")
        ax.set_xlabel("X")
        ax.set_ylabel("Residuals")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Outliers
    st.markdown("<h3 class='property-header'>Outliers</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Outliers are observations that deviate significantly from the rest of the data.
        They can occur due to measurement errors, unusual events, or legitimate extreme values.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Extremely high income individuals in income distributions</li>
            <li>Unusually large firms in industry data</li>
            <li>Countries with exceptional growth rates or economic conditions</li>
            <li>Extreme price movements during market disruptions</li>
        </ul>

        <p><b>Why It Matters:</b> Outliers can disproportionately influence regression results, distort 
        mean estimates, and affect hypothesis tests. They may lead to incorrect conclusions or model selection.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Robust regression methods (median regression, Huber M-estimation)</li>
            <li>Outlier detection techniques (Cook's distance, leverage plots)</li>
            <li>Winsorizing or trimming extreme values</li>
            <li>Transformation of variables to reduce the impact of outliers</li>
            <li>Analyzing with and without outliers to assess sensitivity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate data with outliers
        np.random.seed(123)
        x = np.random.normal(0, 1, 100)
        y = 3 + 2 * x + np.random.normal(0, 1, 100)

        # Add outliers
        outlier_indices = [10, 30, 80]
        x[outlier_indices] = [3, -3, 4]
        y[outlier_indices] = [15, -10, 20]

        # Plot data with outliers
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.7)
        ax.scatter(x[outlier_indices], y[outlier_indices], color='red', s=100, label='Outliers')

        # Add regression lines with and without outliers
        # With outliers
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y)
        x_range = np.linspace(min(x), max(x), 100)
        ax.plot(x_range, intercept1 + slope1 * x_range, 'r--',
                label='With Outliers')

        # Without outliers
        x_clean = np.delete(x, outlier_indices)
        y_clean = np.delete(y, outlier_indices)
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x_clean, y_clean)
        ax.plot(x_range, intercept2 + slope2 * x_range, 'g-',
                label='Without Outliers')

        ax.set_title("Impact of Outliers on Regression")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Skewness
    st.markdown("<h3 class='property-header'>Skewness</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Skewness measures the asymmetry of a probability distribution. A distribution is skewed 
        when one tail is longer than the other, indicating a concentration of values on one side.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Income distributions (typically right-skewed)</li>
            <li>Firm size distributions (often highly right-skewed)</li>
            <li>Housing prices (usually right-skewed)</li>
            <li>Tax rates (may be left or right-skewed depending on the tax system)</li>
        </ul>

        <p><b>Why It Matters:</b> Skewed distributions violate normality assumptions in many statistical tests.
        Measures of central tendency (mean, median, mode) diverge in skewed distributions, affecting interpretation.
        OLS regression can be less efficient with highly skewed variables.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Transformations (log, square root, Box-Cox) to reduce skewness</li>
            <li>Using robust measures (median instead of mean)</li>
            <li>Non-parametric methods that don't assume normality</li>
            <li>Quantile regression to analyze different parts of the distribution</li>
            <li>Generalized linear models with appropriate link functions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate skewed distributions
        np.random.seed(42)
        # Right-skewed (e.g., income)
        right_skewed = np.random.lognormal(mean=0, sigma=1, size=1000)
        # Left-skewed
        left_skewed = -np.random.lognormal(mean=0, sigma=1, size=1000)
        # Normal for comparison
        normal = np.random.normal(0, 1, 1000)

        # Plot histograms
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        # Right-skewed
        sns.histplot(right_skewed, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f"Right-Skewed Distribution (Skewness: {stats.skew(right_skewed):.2f})")
        axes[0].axvline(np.mean(right_skewed), color='red', linestyle='--', label='Mean')
        axes[0].axvline(np.median(right_skewed), color='green', linestyle='-', label='Median')
        axes[0].legend()

        # Normal
        sns.histplot(normal, kde=True, ax=axes[1], color='lightgreen')
        axes[1].set_title(f"Normal Distribution (Skewness: {stats.skew(normal):.2f})")
        axes[1].axvline(np.mean(normal), color='red', linestyle='--', label='Mean')
        axes[1].axvline(np.median(normal), color='green', linestyle='-', label='Median')
        axes[1].legend()

        # Left-skewed
        sns.histplot(left_skewed, kde=True, ax=axes[2], color='salmon')
        axes[2].set_title(f"Left-Skewed Distribution (Skewness: {stats.skew(left_skewed):.2f})")
        axes[2].axvline(np.mean(left_skewed), color='red', linestyle='--', label='Mean')
        axes[2].axvline(np.median(left_skewed), color='green', linestyle='-', label='Median')
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)

    # Kurtosis
    st.markdown("<h3 class='property-header'>Kurtosis</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Kurtosis measures the "tailedness" of a probability distribution. High kurtosis (leptokurtic)
        indicates heavy tails and more extreme values, while low kurtosis (platykurtic) indicates lighter tails.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Financial returns (typically exhibit excess kurtosis/fat tails)</li>
            <li>Exchange rate changes (often leptokurtic)</li>
            <li>Commodity price fluctuations (frequently show high kurtosis)</li>
            <li>Economic growth rates during periods of stability vs. instability</li>
        </ul>

        <p><b>Why It Matters:</b> High kurtosis means extreme events occur more frequently than predicted by a normal 
        distribution. This affects risk assessment, statistical tests, and confidence intervals. Standard models may 
        underestimate the probability of extreme events.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Using distributions that account for excess kurtosis (t-distribution, generalized error distribution)</li>
            <li>Robust statistical methods less sensitive to outliers</li>
            <li>Transformations to normalize the distribution</li>
            <li>Bootstrap methods for more reliable confidence intervals</li>
            <li>Extreme value theory for modeling tail behavior</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate distributions with different kurtosis
        np.random.seed(42)
        # Normal distribution (kurtosis = 3 for normal)
        normal = np.random.normal(0, 1, 1000)
        # High kurtosis (t-distribution with low df)
        high_kurtosis = np.random.standard_t(df=3, size=1000)
        # Low kurtosis (uniform is platykurtic)
        low_kurtosis = np.random.uniform(-3, 3, 1000)

        # Plot histograms
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        # High kurtosis
        sns.histplot(high_kurtosis, kde=True, ax=axes[0], color='coral')
        axes[0].set_title(
            f"High Kurtosis (t-distribution, Kurtosis: {stats.kurtosis(high_kurtosis, fisher=False):.2f})")

        # Normal
        sns.histplot(normal, kde=True, ax=axes[1], color='skyblue')
        axes[1].set_title(f"Normal Distribution (Kurtosis: {stats.kurtosis(normal, fisher=False):.2f})")

        # Low kurtosis
        sns.histplot(low_kurtosis, kde=True, ax=axes[2], color='lightgreen')
        axes[2].set_title(f"Low Kurtosis (Uniform, Kurtosis: {stats.kurtosis(low_kurtosis, fisher=False):.2f})")

        plt.tight_layout()
        st.pyplot(fig)

    # Multicollinearity
    st.markdown("<h3 class='property-header'>Multicollinearity</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="definition">
        <p><b>Definition:</b> Multicollinearity occurs when independent variables in a regression model are highly 
        correlated with each other, making it difficult to isolate their individual effects on the dependent variable.</p>

        <p><b>Examples in Economics:</b></p>
        <ul>
            <li>Different education measures (years of schooling, degree attainment) in wage equations</li>
            <li>Various measures of firm size (employees, revenue, assets) in productivity studies</li>
            <li>Different macroeconomic indicators that move together (inflation, interest rates)</li>
            <li>Regional characteristics that are closely related (urbanization, income levels, education)</li>
        </ul>

        <p><b>Why It Matters:</b> Multicollinearity increases the variance of coefficient estimates, making them 
        unstable and sensitive to small changes in the model. It can lead to incorrect signs of coefficients, 
        high standard errors, and difficulties in determining significant variables.</p>

        <p><b>How to Address:</b></p>
        <ul>
            <li>Remove some of the correlated variables</li>
            <li>Combine collinear variables into a single index</li>
            <li>Principal Component Analysis (PCA) or factor analysis</li>
            <li>Ridge regression or other regularization techniques</li>
            <li>Increase sample size (when possible)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate multicollinear data
        np.random.seed(42)
        n = 100

        # Create correlated predictors
        x1 = np.random.normal(0, 1, n)
        x2 = 0.9 * x1 + 0.1 * np.random.normal(0, 1, n)  # Highly correlated with x1
        x3 = np.random.normal(0, 1, n)  # Independent variable

        # Create dependent variable influenced by all predictors
        y = 2 * x1 + 3 * x2 + 1.5 * x3 + np.random.normal(0, 1, n)

        # Calculate correlation matrix
        multi_data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})
        corr_matrix = multi_data.corr()

        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix Showing Multicollinearity")
        st.pyplot(fig)

        # Scatter plot of correlated variables
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x1, x2, alpha=0.7)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2 (Highly Correlated with X1)")
        ax.set_title("Scatter Plot of Multicollinear Variables")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Endogeneity
    st.markdown("<h3 class='property-header'>Endogeneity</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="definition">
    <p><b>Definition:</b> Endogeneity occurs when an explanatory variable is correlated with the error term in a regression model.
    This can arise from omitted variables, simultaneity (reverse causality), or measurement error.</p>

    <p><b>Examples in Economics:</b></p>
    <ul>
        <li>Effect of education on income (ability affects both)</li>
        <li>Price and quantity in demand estimation (simultaneity)</li>
        <li>Impact of institutions on economic growth (bidirectional causality)</li>
        <li>Fiscal policy and economic output (policy responds to economic conditions)</li>
    </ul>

    <p><b>Why It Matters:</b> Endogeneity leads to biased and inconsistent coefficient estimates. Standard OLS estimates 
    no longer represent causal effects but mere correlations. Policy recommendations based on such estimates may be misleading.</p>

    <p><b>How to Address:</b></p>
    <ul>
        <li>Instrumental Variables (IV) estimation</li>
        <li>Two-Stage Least Squares (2SLS)</li>
        <li>Generalized Method of Moments (GMM)</li>
        <li>Natural experiments and quasi-experimental designs</li>
        <li>Fixed effects (for certain types of omitted variable bias)</li>
        <li>Difference-in-differences estimation</li>
        <li>Regression discontinuity design</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visual explanation of endogeneity and IV
    np.random.seed(42)
    n = 1000

    # Unobserved variable (e.g., ability)
    ability = np.random.normal(0, 1, n)

    # Instrument (e.g., distance to college) - correlated with education but not directly with income
    instrument = np.random.normal(0, 1, n)

    # Endogenous variable (education) affected by ability and instrument
    education = 0.5 * ability + 0.7 * instrument + np.random.normal(0, 1, n)

    # Outcome (income) affected by education and ability
    income = 3 * education + 2 * ability + np.random.normal(0, 2, n)

    # Create DataFrame
    endo_data = pd.DataFrame({
        'Education': education,
        'Income': income,
        'Ability': ability,
        'Instrument': instrument
    })

    # Plot relationships
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Education vs Income (naive relationship)
    sns.regplot(x='Education', y='Income', data=endo_data, ax=axes[0, 0], line_kws={"color": "red"})
    axes[0, 0].set_title('Naive Regression: Education vs Income')

    # Ability affecting both Education and Income
    axes[0, 1].scatter(ability, education, alpha=0.5, label='Ability → Education')
    axes[0, 1].scatter(ability, income, alpha=0.5, label='Ability → Income')
    axes[0, 1].set_xlabel('Ability (Unobserved)')
    axes[0, 1].set_ylabel('Education / Income')
    axes[0, 1].legend()
    axes[0, 1].set_title('Unobserved Ability Affecting Both Variables')

    # Instrument vs Education (first stage of IV)
    sns.regplot(x='Instrument', y='Education', data=endo_data, ax=axes[1, 0], line_kws={"color": "green"})
    axes[1, 0].set_title('IV First Stage: Instrument vs Education')

    # Instrument vs Income (reduced form)
    sns.regplot(x='Instrument', y='Income', data=endo_data, ax=axes[1, 1], line_kws={"color": "purple"})
    axes[1, 1].set_title('IV Reduced Form: Instrument vs Income')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Recommendations for Cross-Sectional Data Analysis in Economics

    1. **Test for heteroskedasticity** and use robust standard errors when necessary
    2. **Check for outliers** and assess their influence on your results
    3. **Examine distributions** for skewness and kurtosis, consider appropriate transformations
    4. **Be cautious with multicollinearity** when including related economic variables
    5. **Address potential endogeneity** using appropriate causal inference techniques
    6. **Consider sample selection issues** that may affect representativeness
    7. **Use visualization tools** to understand relationships before formal modeling
    8. **Validate model assumptions** through appropriate diagnostic tests
    """)

# Domain-Specific Properties
elif page == "Domain-Specific Properties":
    st.markdown("<h2 class='sub-header'>Domain-Specific Economic Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    Different economic domains have characteristic data properties and challenges that researchers
    should be aware of. This section explores the common features of data in financial economics,
    macroeconomics, and microeconomics.
    """)

    # Financial Data Properties
    st.markdown("<h3 class='property-header'>Financial Data Properties</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="domain-box">
    <p>Financial data often exhibits distinctive properties that require specialized analytical approaches. 
    The high frequency of financial data collection and the presence of market forces create unique patterns.</p>

    <h4>Key Properties:</h4>

    <p><b>1. Volatility Clustering</b></p>
    <ul>
        <li>Periods of high volatility tend to cluster together</li>
        <li>Prevalent in stock returns, exchange rates, and other financial assets</li>
        <li>Requires GARCH-type models or stochastic volatility approaches</li>
    </ul>

    <p><b>2. Fat Tails (Leptokurtosis)</b></p>
    <ul>
        <li>Financial returns show significantly more extreme values than normal distributions would predict</li>
        <li>Kurtosis values often exceed 3 (the normal distribution value)</li>
        <li>Critical for proper risk assessment and stress testing</li>
        <li>Necessitates using t-distributions or other heavy-tailed distributions</li>
    </ul>

    <p><b>3. Leverage Effect</b></p>
    <ul>
        <li>Negative correlation between returns and volatility</li>
        <li>Downward movements tend to increase volatility more than upward movements</li>
        <li>Requires asymmetric volatility models like EGARCH or GJR-GARCH</li>
    </ul>

    <p><b>4. High Frequency/Microstructure Properties</b></p>
    <ul>
        <li>Intraday seasonality in volatility and trading volume</li>
        <li>Bid-ask bounce and other microstructure noise</li>
        <li>Market impact of large trades</li>
        <li>Requires specialized econometric techniques</li>
    </ul>

    <p><b>5. Non-Synchronous Trading</b></p>
    <ul>
        <li>Different assets trade at different times or frequencies</li>
        <li>Creates challenges for correlation estimation</li>
        <li>Especially important in international markets with different trading hours</li>
    </ul>

    <p><b>6. Jumps and Regime Shifts</b></p>
    <ul>
        <li>Sudden, large price movements in response to news or events</li>
        <li>Structural breaks due to regulatory changes or market crises</li>
        <li>May require jump-diffusion models or regime-switching approaches</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Generate financial return data
    np.random.seed(42)
    n = 1000

    # Create data with volatility clustering and leverage effect
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = 0.01

    for i in range(1, n):
        # Conditional variance with leverage effect
        # More impact from negative returns
        if returns[i - 1] < 0:
            volatility[i] = 0.05 + 0.85 * volatility[i - 1] + 0.1 * returns[i - 1] ** 2 + 0.05 * abs(returns[i - 1])
        else:
            volatility[i] = 0.05 + 0.85 * volatility[i - 1] + 0.1 * returns[i - 1] ** 2 - 0.02 * abs(returns[i - 1])

        # Generate return with fat tails (t-distribution)
        returns[i] = np.random.standard_t(df=5) * np.sqrt(volatility[i])

    # Create a financial time series
    dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
    price = 100 * np.exp(np.cumsum(returns))

    financial_data = pd.DataFrame({
        "Date": dates,
        "Price": price,
        "Returns": returns,
        "Volatility": volatility
    })

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Price series
    axes[0].plot(financial_data["Date"], financial_data["Price"])
    axes[0].set_title("Asset Price")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Returns
    axes[1].plot(financial_data["Date"], financial_data["Returns"])
    axes[1].set_title("Returns with Volatility Clustering")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Returns")
    axes[1].grid(True, alpha=0.3)

    # Return distribution
    sns.histplot(financial_data["Returns"], kde=True, ax=axes[2])
    # Add normal distribution for comparison
    x = np.linspace(min(financial_data["Returns"]), max(financial_data["Returns"]), 100)
    mean_ret = np.mean(financial_data["Returns"])
    std_ret = np.std(financial_data["Returns"])
    norm_pdf = stats.norm.pdf(x, mean_ret, std_ret)
    axes[2].plot(x, norm_pdf * len(financial_data["Returns"]) * (x[1] - x[0]), 'r-', label="Normal Distribution")
    axes[2].set_title(f"Return Distribution (Kurtosis: {stats.kurtosis(financial_data['Returns']):.2f})")
    axes[2].set_xlabel("Returns")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Macroeconomic Data Properties
    st.markdown("<h3 class='property-header'>Macroeconomic Data Properties</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="domain-box">
    <p>Macroeconomic data captures broad economic indicators at national or regional levels.
    It has distinctive properties that stem from the aggregate nature of the measures and the
    long-term processes of economic development.</p>

    <h4>Key Properties:</h4>

    <p><b>1. Non-Stationarity and Unit Roots</b></p>
    <ul>
        <li>Many macroeconomic variables like GDP, price levels, and monetary aggregates exhibit trends</li>
        <li>Often contain stochastic trends (unit roots) rather than deterministic trends</li>
        <li>Requires unit root testing, cointegration analysis, and error correction models</li>
    </ul>

    <p><b>2. Structural Breaks</b></p>
    <ul>
        <li>Policy regime changes (e.g., monetary policy frameworks)</li>
        <li>Major economic events (Great Depression, Oil Crises, Financial Crisis)</li>
        <li>Technological revolutions or institutional changes</li>
        <li>Requires testing for and modeling breaks explicitly</li>
    </ul>

    <p><b>3. Seasonality</b></p>
    <ul>
        <li>Regular patterns in economic activity (quarterly, monthly)</li>
        <li>Especially prominent in retail sales, production, employment</li>
        <li>Requires seasonal adjustment or explicit modeling of seasonal components</li>
    </ul>

    <p><b>4. Cointegration</b></p>
    <ul>
        <li>Long-run equilibrium relationships between non-stationary variables</li>
        <li>Common in macroeconomic theory (consumption-income, interest rate parity)</li>
        <li>Requires vector error correction models (VECM) and cointegration tests</li>
    </ul>

    <p><b>5. Low-Frequency Sampling</b></p>
    <ul>
        <li>Many macroeconomic indicators available only quarterly or monthly</li>
        <li>Creates small sample issues in time series analysis</li>
        <li>May require techniques for mixed-frequency data or temporal aggregation</li>
    </ul>

    <p><b>6. Revision of Historical Data</b></p>
    <ul>
        <li>Preliminary estimates often revised substantially</li>
        <li>Creates data uncertainty and real-time forecasting challenges</li>
        <li>Important to use vintage datasets for certain analyses</li>
    </ul>

    <p><b>7. Cross-Country Heterogeneity</b></p>
    <ul>
        <li>Economic relationships vary across countries due to institutional differences</li>
        <li>Requires careful consideration in panel macroeconomic studies</li>
        <li>May need to allow for heterogeneous coefficients</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create macroeconomic data example
    np.random.seed(42)

    # Generate quarterly dates for 30 years
    dates_macro = pd.date_range(start="1990-01-01", periods=120, freq="Q")

    # Create GDP with trend, cycle, seasonal component, and structural break
    t = np.arange(120)

    # Trend component with structural break
    trend = 100 + 0.5 * t
    break_point = 60  # Break at 2005
    trend[break_point:] += 0.3 * (t[break_point:] - t[break_point])

    # Cyclical component (business cycle)
    cycle = 5 * np.sin(2 * np.pi * t / 20)

    # Seasonal component
    seasonal = 3 * np.sin(2 * np.pi * t / 4)

    # Random component
    noise = np.random.normal(0, 2, 120)

    # Combine components
    gdp = trend + cycle + seasonal + noise

    # Create unemployment with inverse relationship to GDP
    unemployment = 10 - 0.03 * (gdp - 100) + np.random.normal(0, 0.5, 120)

    # Create inflation with lag relationship to GDP gaps
    gdp_trend = trend
    gdp_gap = gdp - gdp_trend
    inflation = 2 + 0.2 * np.roll(gdp_gap, 4) + np.random.normal(0, 0.3, 120)
    inflation[:4] = inflation[4]  # Replace initial values

    # Create macro data DataFrame
    macro_data = pd.DataFrame({
        "Date": dates_macro,
        "GDP": gdp,
        "Unemployment": unemployment,
        "Inflation": inflation
    })

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # GDP with components
    axes[0].plot(macro_data["Date"], gdp, label="GDP")
    axes[0].plot(macro_data["Date"], trend, 'r--', label="Trend (with break)")
    axes[0].axvline(x=dates_macro[break_point], color='k', linestyle='-', alpha=0.3, label="Structural Break")
    axes[0].set_title("GDP with Trend and Structural Break")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("GDP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Seasonal component
    quarterly_avg = np.zeros(4)
    for i in range(4):
        quarterly_avg[i] = np.mean(gdp[i::4] - trend[i::4] - cycle[i::4])

    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    axes[1].bar(quarter_labels, quarterly_avg)
    axes[1].set_title("Seasonal Component by Quarter")
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel("Average Seasonal Effect")
    axes[1].grid(True, alpha=0.3)

    # Unemployment and GDP relationship
    axes[2].scatter(gdp, unemployment, alpha=0.6)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(gdp, unemployment)
    x_range = np.linspace(min(gdp), max(gdp), 100)
    axes[2].plot(x_range, intercept + slope * x_range, 'r-')

    axes[2].set_title(f"Okun's Law Relationship (r = {r_value:.2f})")
    axes[2].set_xlabel("GDP")
    axes[2].set_ylabel("Unemployment Rate")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Microeconomic Data Properties
    st.markdown("<h3 class='property-header'>Microeconomic Data Properties</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="domain-box">
    <p>Microeconomic data focuses on individual economic agents (people, households, firms) and their behavior.
    This type of data has specific properties that reflect individual heterogeneity, decision processes, and constraints.</p>

    <h4>Key Properties:</h4>

    <p><b>1. Individual Heterogeneity</b></p>
    <ul>
        <li>Significant variation across individuals in preferences, constraints, and behavior</li>
        <li>Observable and unobservable heterogeneity</li>
        <li>Requires random coefficients models, mixed models, or quantile regression</li>
    </ul>

    <p><b>2. Sample Selection Issues</b></p>
    <ul>
        <li>Non-random selection into observed samples (e.g., wage equations for employed only)</li>
        <li>Participation decisions affecting what we observe</li>
        <li>Requires Heckman selection models or other selection correction approaches</li>
    </ul>

    <p><b>3. Limited Dependent Variables</b></p>
    <ul>
        <li>Binary outcomes (participate/not participate)</li>
        <li>Censored or truncated data (zero consumption, minimum wage)</li>
        <li>Count data (number of children, hospital visits)</li>
        <li>Requires specialized models (probit, logit, tobit, Poisson, etc.)</li>
    </ul>

    <p><b>4. Endogeneity and Self-Selection</b></p>
    <ul>
        <li>Individuals choose education, occupation, location based on expected outcomes</li>
        <li>Treatment effects vary and individuals select based on potential gains</li>
        <li>Requires instrumental variables, natural experiments, or structural approaches</li>
    </ul>

    <p><b>5. Heavy-Tailed Distributions</b></p>
    <ul>
        <li>Income, wealth, and firm size distributions typically have heavy right tails</li>
        <li>Power laws often provide better fit than normal distributions</li>
        <li>Requires appropriate transformations or specialized distributional assumptions</li>
    </ul>

    <p><b>6. Measurement Error</b></p>
    <ul>
        <li>Self-reported data often contains errors (income, hours worked)</li>
        <li>Recall bias in retrospective questions</li>
        <li>Requires error-in-variables models or instrumental variables</li>
    </ul>

    <p><b>7. Panel Attrition</b></p>
    <ul>
        <li>Individuals drop out of panel surveys non-randomly</li>
        <li>Creates potential selection bias in longitudinal studies</li>
        <li>Requires testing for attrition bias and potential correction methods</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create microeconomic data example
    np.random.seed(123)
    n = 1000

    # Generate individual characteristics
    education = np.random.normal(12, 3, n)  # Years of education
    ability = np.random.normal(0, 1, n)  # Unobserved ability
    experience = np.random.gamma(shape=2, scale=5, size=n)  # Work experience
    female = np.random.binomial(1, 0.5, n)  # Gender indicator

    # Generate wage equation with heterogeneous returns
    returns_to_education = 0.08 + 0.02 * ability  # Heterogeneous returns
    log_wage = 1.5 + returns_to_education * education + 0.03 * experience - 0.0005 * experience ** 2 - 0.2 * female + 0.5 * ability + np.random.normal(
        0, 0.3, n)

    # Generate employment status (selection equation)
    emp_propensity = 0.2 + 0.1 * education + 0.05 * experience - 0.5 * female + np.random.normal(0, 1, n)
    employed = (emp_propensity > 0).astype(int)

    # Observed wage (only for employed)
    observed_wage = np.exp(log_wage) * employed
    observed_wage[observed_wage == 0] = np.nan  # Set to NaN for visualization

    # Generate consumption choices (corner solution)
    luxury_propensity = -5 + 0.8 * log_wage + np.random.normal(0, 1, n)
    luxury_spending = np.maximum(0, luxury_propensity)  # Cannot spend negative amounts

    # Create microeconomic DataFrame
    micro_data = pd.DataFrame({
        "Education": education,
        "Experience": experience,
        "Female": female,
        "Ability": ability,  # Unobserved in real data
        "LogWage": log_wage,  # Full wages (unobserved for non-employed)
        "ObservedWage": observed_wage,  # Only observed for employed
        "Employed": employed,
        "LuxurySpending": luxury_spending
    })

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Income distribution (log-normal)
    sns.histplot(np.exp(micro_data["LogWage"]), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Income Distribution (Right-Skewed)")
    axes[0, 0].set_xlabel("Wage")
    axes[0, 0].set_ylabel("Frequency")

    # Education returns conditional on ability
    high_ability = micro_data["Ability"] > 0.5
    low_ability = micro_data["Ability"] < -0.5

    sns.regplot(x="Education", y="LogWage", data=micro_data[high_ability],
                ax=axes[0, 1], scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, label="High Ability")
    sns.regplot(x="Education", y="LogWage", data=micro_data[low_ability],
                ax=axes[0, 1], scatter_kws={"alpha": 0.4}, line_kws={"color": "blue"}, label="Low Ability")

    axes[0, 1].set_title("Heterogeneous Returns to Education")
    axes[0, 1].set_xlabel("Years of Education")
    axes[0, 1].set_ylabel("Log Wage")
    axes[0, 1].legend()

    # Sample selection
    axes[1, 0].scatter(micro_data["Education"], micro_data["ObservedWage"], alpha=0.4, label="Observed Wages")
    axes[1, 0].set_title("Sample Selection in Wage Equations")
    axes[1, 0].set_xlabel("Years of Education")
    axes[1, 0].set_ylabel("Observed Wage (Zero if Unemployed)")
    axes[1, 0].grid(True, alpha=0.3)

    # Corner solution
    axes[1, 1].scatter(np.exp(micro_data["LogWage"]), micro_data["LuxurySpending"], alpha=0.4)
    axes[1, 1].set_title("Corner Solution in Consumption")
    axes[1, 1].set_xlabel("Wage")
    axes[1, 1].set_ylabel("Luxury Spending")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # High-Frequency Data Properties
    st.markdown("<h3 class='property-header'>High-Frequency Financial Data Properties</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="domain-box">
    <p>High-frequency financial data, collected at intraday intervals (minutes, seconds, or even microseconds),
    presents unique challenges and properties that differ from traditional lower frequency data.</p>

    <h4>Key Properties:</h4>

    <p><b>1. Market Microstructure Noise</b></p>
    <ul>
        <li>Bid-ask bounce creating artificial volatility</li>
        <li>Discrete price changes (tick size)</li>
        <li>Trade direction and quote updates</li>
        <li>Requires noise-robust estimation techniques</li>
    </ul>

    <p><b>2. Irregular Spacing</b></p>
    <ul>
        <li>Transactions occur at irregular time intervals</li>
        <li>Information content of time between trades</li>
        <li>Requires point process models or careful aggregation</li>
    </ul>

    <p><b>3. Intraday Patterns</b></p>
    <ul>
        <li>U-shaped volatility pattern (higher at open and close)</li>
        <li>Lunch-time drop in trading activity</li>
        <li>Regular patterns related to market announcements</li>
        <li>Requires seasonal adjustment within the day</li>
    </ul>

    <p><b>4. Price Discreteness</b></p>
    <ul>
        <li>Prices move in discrete increments (ticks)</li>
        <li>Creates rounding effects and constraints on price movements</li>
        <li>May require specialized discrete-price models</li>
    </ul>

    <p><b>5. Long Memory in Volatility</b></p>
    <ul>
        <li>Persistence in volatility at high frequencies</li>
        <li>Slow decay of autocorrelation function</li>
        <li>May require fractionally integrated models</li>
    </ul>

    <p><b>6. Jump Components</b></p>
    <ul>
        <li>Distinction between continuous price movements and jumps</li>
        <li>Jumps often linked to specific news announcements</li>
        <li>Requires specialized jump detection and modeling</li>
    </ul>

    <p><b>7. Data Management Challenges</b></p>
    <ul>
        <li>Extremely large datasets requiring efficient processing</li>
        <li>Cleaning issues (outliers, recording errors)</li>
        <li>Synchronization across multiple assets or markets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create high-frequency data example
    np.random.seed(42)

    # Generate intraday timestamps (1-minute data)
    trading_start = pd.Timestamp("2023-03-17 09:30:00")
    trading_end = pd.Timestamp("2023-03-17 16:00:00")
    timestamps = pd.date_range(start=trading_start, end=trading_end, freq="1min")
    n_periods = len(timestamps)

    # Create intraday volatility pattern (U-shape)
    time_of_day = np.arange(n_periods) / n_periods
    u_shape = 1.5 - 2 * (time_of_day - 0.5) ** 2

    # Add lunch effect
    lunch_time = (timestamps.hour == 12) & (timestamps.minute >= 0) & (timestamps.minute <= 30)
    u_shape[lunch_time] *= 0.7

    # Base volatility and log price
    base_vol = 0.0001  # Base volatility for 1-minute returns
    log_price = np.zeros(n_periods)
    log_price[0] = np.log(100)  # Start at 100

    # Add jumps at specific times
    jump_times = [60, 120, 240, 360]  # Positions for jumps
    jump_sizes = [0.002, -0.003, 0.0025, -0.002]  # Jump sizes

    # Generate returns with intraday pattern and jumps
    returns = np.random.normal(0, np.sqrt(base_vol) * u_shape, n_periods)

    # Add jumps
    for t, size in zip(jump_times, jump_sizes):
        returns[t] += size

    # Construct log price
    for i in range(1, n_periods):
        log_price[i] = log_price[i - 1] + returns[i]

    # Convert to price
    price = np.exp(log_price)

    # Create DataFrame
    hf_data = pd.DataFrame({
        "Timestamp": timestamps,
        "Price": price,
        "Returns": returns,
        "Volatility": u_shape * base_vol
    })

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Price with jumps
    axes[0].plot(hf_data["Timestamp"], hf_data["Price"])
    axes[0].set_title("Intraday Price with Jumps")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Returns
    axes[1].plot(hf_data["Timestamp"], hf_data["Returns"])
    axes[1].set_title("1-Minute Returns")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Returns")
    axes[1].grid(True, alpha=0.3)

    # Intraday volatility pattern
    hour_minute = [f"{t.hour}:{t.minute:02d}" for t in hf_data["Timestamp"]]
    hours = hf_data["Timestamp"].dt.hour + hf_data["Timestamp"].dt.minute / 60

    axes[2].plot(hours, hf_data["Volatility"])
    axes[2].set_title("Intraday Volatility Pattern (U-shape with lunch effect)")
    axes[2].set_xlabel("Hour of Day")
    axes[2].set_ylabel("Volatility")
    axes[2].set_xticks([9.5, 10, 11, 12, 13, 14, 15, 16])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Summary of Domain-Specific Considerations

    When working with economic data, it's essential to understand the specific properties and challenges
    associated with your domain of interest. Here's a quick reference:

    - **Financial Data**: Focus on volatility modeling, fat-tailed distributions, and asymmetric effects.
      Consider using GARCH-type models and non-normal distributions.

    - **Macroeconomic Data**: Test for non-stationarity, structural breaks, and cointegration.
      Consider seasonal adjustment and be aware of data revisions.

    - **Microeconomic Data**: Address heterogeneity, selection issues, and endogeneity.
      Use appropriate limited dependent variable models and consider measurement error.

    - **High-Frequency Data**: Account for market microstructure noise, intraday patterns, and irregular spacing.
      Be prepared for data management challenges and specialized modeling approaches.
    """)

# Statistical Tests
elif page == "Statistical Tests":
    st.markdown("<h2 class='sub-header'>Statistical Tests for Economic Data Properties</h2>", unsafe_allow_html=True)

    st.markdown("""
    This section provides an overview of key statistical tests used to detect and quantify 
    the various properties of economic data. Understanding these tests is essential for selecting 
    appropriate modeling approaches and ensuring valid inference.
    """)

    # Tests for Time Series Properties
    st.markdown("<h3 class='property-header'>Tests for Time Series Properties</h3>", unsafe_allow_html=True)

    # Stationarity Tests
    st.markdown("""
    <div class="definition">
    <h4>Stationarity Tests</h4>

    <p><b>1. Augmented Dickey-Fuller (ADF) Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests the null hypothesis that a unit root is present (series is non-stationary)</li>
        <li><b>Implementation:</b> <code>statsmodels.tsa.stattools.adfuller()</code></li>
        <li><b>Interpretation:</b> Reject the null hypothesis (p < 0.05) to conclude stationarity</li>
        <li><b>Variations:</b> Can include constant, trend, or both</li>
    </ul>

    <p><b>2. KPSS Test (Kwiatkowski–Phillips–Schmidt–Shin)</b></p>
    <ul>
        <li><b>Purpose:</b> Tests the null hypothesis of stationarity (opposite of ADF)</li>
        <li><b>Implementation:</b> <code>statsmodels.tsa.stattools.kpss()</code></li>
        <li><b>Interpretation:</b> Fail to reject the null (p > 0.05) to conclude stationarity</li>
        <li><b>Usage:</b> Often used alongside ADF for confirmatory analysis</li>
    </ul>

    <p><b>3. Phillips-Perron (PP) Test</b></p>
    <ul>
        <li><b>Purpose:</b> Similar to ADF but with non-parametric correction for autocorrelation</li>
        <li><b>Implementation:</b> <code>statsmodels.tsa.stattools.phillips_perron()</code></li>
        <li><b>Advantage:</b> More robust to unspecified autocorrelation and heteroskedasticity</li>
    </ul>

    <p><b>4. Zivot-Andrews Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for unit root allowing for a structural break</li>
        <li><b>Implementation:</b> Available in R or custom Python implementations</li>
        <li><b>Advantage:</b> Accounts for structural breaks that might make a stationary series appear non-stationary</li>
    </ul>
    </div>

    <div class="example-box">
    <h4>Example: ADF Test in Python</h4>

    ```python
    from statsmodels.tsa.stattools import adfuller

    # Run ADF test on time series data
    result = adfuller(time_series_data, autolag='AIC')

    # Extract and print results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f'ADF Statistic: {adf_statistic:.4f}')
    print(f'p-value: {p_value:.4f}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value:.4f}')

    # Interpret results
    if p_value < 0.05:
        print("Reject the null hypothesis. Data is stationary.")
    else:
        print("Fail to reject the null hypothesis. Data contains a unit root (non-stationary).")
    ```
    </div>
    """, unsafe_allow_html=True)

    # Autocorrelation Tests
    st.markdown("""
    <div class="definition">
    <h4>Autocorrelation Tests</h4>

    <p><b>1. Ljung-Box Q Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests the null hypothesis of no autocorrelation up to a specified lag</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.acorr_ljungbox()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude autocorrelation is present</li>
        <li><b>Usage:</b> Commonly used for testing residuals from ARIMA models</li>
    </ul>

    <p><b>2. Breusch-Godfrey Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for higher-order serial correlation in regression residuals</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.acorr_breusch_godfrey()</code></li>
        <li><b>Advantage:</b> Allows for lagged dependent variables and higher-order autocorrelation</li>
    </ul>

    <p><b>3. Durbin-Watson Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for first-order autocorrelation in regression residuals</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.stattools.durbin_watson()</code></li>
        <li><b>Interpretation:</b> Values around 2 suggest no autocorrelation, values toward 0 indicate positive autocorrelation, values toward 4 indicate negative autocorrelation</li>
        <li><b>Limitation:</b> Only tests for first-order autocorrelation</li>
    </ul>
    </div>

    <div class="example-box">
    <h4>Example: Ljung-Box Test in Python</h4>

    ```python
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Test for autocorrelation up to lag 10
    result = acorr_ljungbox(residuals, lags=10)

    # Extract results
    lb_statistics = result[0]
    p_values = result[1]

    # Print results
    print("Ljung-Box Test Results:")
    for i, (stat, p) in enumerate(zip(lb_statistics, p_values), 1):
        print(f"Lag {i}: Statistic={stat:.4f}, p-value={p:.4f}")
        if p < 0.05:
            print("   Significant autocorrelation detected")
        else:
            print("   No significant autocorrelation detected")
    ```
    </div>
    """, unsafe_allow_html=True)

    # Heteroskedasticity Tests
    st.markdown("""
    <div class="definition">
    <h4>Heteroskedasticity Tests</h4>

    <p><b>1. White Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for heteroskedasticity in regression residuals</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.het_white()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude heteroskedasticity is present</li>
        <li><b>Advantage:</b> Doesn't assume a specific form of heteroskedasticity</li>
    </ul>

    <p><b>2. Breusch-Pagan Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for heteroskedasticity based on linear specification</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.het_breuschpagan()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude heteroskedasticity is present</li>
        <li><b>Assumption:</b> Assumes heteroskedasticity is a linear function of the independent variables</li>
    </ul>

    <p><b>3. Engle's ARCH Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for autoregressive conditional heteroskedasticity in time series</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.het_arch()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude ARCH effects are present</li>
        <li><b>Usage:</b> Particularly important for financial time series</li>
    </ul>

    <p><b>4. Goldfeld-Quandt Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests if variances in two subsamples are equal</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.het_goldfeldquandt()</code></li>
        <li><b>Application:</b> Useful when you suspect heteroskedasticity increases with a specific variable</li>
    </ul>
    </div>

    <div class="example-box">
    <h4>Example: Breusch-Pagan Test in Python</h4>

    ```python
    import numpy as np
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm

    # Fit a regression model
    X = sm.add_constant(X_data)
    model = sm.OLS(y_data, X).fit()

    # Run Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, model.model.exog)

    # Extract results
    lm_stat = bp_test[0]
    lm_pvalue = bp_test[1]
    f_stat = bp_test[2]
    f_pvalue = bp_test[3]

    # Print results
    print(f'Breusch-Pagan Test:')
    print(f'Lagrange Multiplier statistic: {lm_stat:.4f}, p-value: {lm_pvalue:.4f}')
    print(f'F-statistic: {f_stat:.4f}, p-value: {f_pvalue:.4f}')

    # Interpret results
    if lm_pvalue < 0.05:
        print("Reject the null hypothesis. Heteroskedasticity is present.")
    else:
        print("Fail to reject the null hypothesis. No evidence of heteroskedasticity.")
    ```
    </div>
    """, unsafe_allow_html=True)

    # Structural Break Tests
    st.markdown("""
    <div class="definition">
    <h4>Structural Break Tests</h4>

    <p><b>1. Chow Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for a structural break at a known point in time</li>
        <li><b>Implementation:</b> No direct StatsModels function, but can be implemented manually</li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude a structural break is present</li>
        <li><b>Limitation:</b> Requires prior knowledge of the breakpoint</li>
    </ul>

    <p><b>2. Quandt-Andrews Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for a structural break at an unknown point in time</li>
        <li><b>Implementation:</b> Available in R, can be manually implemented in Python</li>
        <li><b>Approach:</b> Performs Chow tests at every possible breakpoint and takes the maximum</li>
    </ul>

    <p><b>3. Bai-Perron Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests for multiple structural breaks at unknown points</li>
        <li><b>Implementation:</b> Available in R, limited Python implementations</li>
        <li><b>Advantage:</b> Can identify multiple breakpoints simultaneously</li>
    </ul>

    <p><b>4. CUSUM and CUSUMSQ Tests</b></p>
    <ul>
        <li><b>Purpose:</b> Graphical methods to detect parameter instability over time</li>
        <li><b>Implementation:</b> <code>statsmodels.stats.diagnostic.recursive_olsresiduals()</code></li>
        <li><b>Interpretation:</b> If the cumulative sum crosses confidence bands, it indicates parameter instability</li>
    </ul>
    </div>

    <div class="example-box">
    <h4>Example: Chow Test Implementation in Python</h4>

    ```python
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats

    def chow_test(y, X, breakpoint):
        # Sample size
        n_total = len(y)

        # Split the data
        y1, y2 = y[:breakpoint], y[breakpoint:]
        X1, X2 = X[:breakpoint], X[breakpoint:]

        # Fit the full model and the split models
        X = sm.add_constant(X)
        X1 = sm.add_constant(X1)
        X2 = sm.add_constant(X2)

        model_full = sm.OLS(y, X).fit()
        model_1 = sm.OLS(y1, X1).fit()
        model_2 = sm.OLS(y2, X2).fit()

        # Sum of squared residuals
        ssr_full = sum(model_full.resid**2)
        ssr_1 = sum(model_1.resid**2)
        ssr_2 = sum(model_2.resid**2)
        ssr_restricted = ssr_1 + ssr_2

        # Number of parameters in each model
        k = X.shape[1]

        # Calculate the test statistic
        f_stat = ((ssr_full - ssr_restricted) / k) / (ssr_restricted / (n_total - 2*k))

        # Calculate p-value
        p_value = 1 - stats.f.cdf(f_stat, k, n_total - 2*k)

        return f_stat, p_value

    # Usage
    f_stat, p_value = chow_test(y_data, X_data, breakpoint=100)
    print(f"Chow Test F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Reject the null hypothesis. Structural break is present.")
    else:
        print("Fail to reject the null hypothesis. No evidence of structural break.")
    ```
    </div>
    """, unsafe_allow_html=True)

    import streamlit as st

    st.markdown("""
    <style>
    .definition {
        background-color: #f7f9fc;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4c78a8;
        margin-bottom: 15px;
    }
    .definition h4 {
        color: #2c3e50;
    }
    .definition ul {
        margin-left: 20px;
        margin-bottom: 10px;
    }
    .definition li {
        margin-bottom: 8px;
    }
    </style>

    <div class="definition">
    <h4>Normality and Distribution Tests</h4>

    <p><b>1. Jarque-Bera Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests if sample data has skewness and kurtosis matching a normal distribution</li>
        <li><b>Implementation:</b> <code>scipy.stats.jarque_bera()</code> or <code>statsmodels.stats.stattools.jarque_bera()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude non-normality</li>
        <li><b>Advantage:</b> Specifically focused on skewness and kurtosis</li>
        <li><b>Limitation:</b> Less effective for very small samples</li>
    </ul>

    <p><b>2. Shapiro-Wilk Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests if a sample comes from a normally distributed population</li>
        <li><b>Implementation:</b> <code>scipy.stats.shapiro()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude non-normality</li>
        <li><b>Advantage:</b> Powerful for small to medium-sized samples</li>
        <li><b>Limitation:</b> Not suitable for large samples (n > 5000)</li>
    </ul>

    <p><b>3. Kolmogorov-Smirnov Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests if a sample follows a specified continuous distribution (normal, exponential, uniform, etc.)</li>
        <li><b>Implementation:</b> <code>scipy.stats.kstest()</code></li>
        <li><b>Interpretation:</b> Reject the null (p < 0.05) to conclude the sample does not match the distribution</li>
        <li><b>Advantage:</b> Applicable to any continuous distribution</li>
        <li><b>Limitation:</b> Less sensitive in tails compared to Anderson-Darling test</li>
    </ul>

    <p><b>4. Anderson-Darling Test</b></p>
    <ul>
        <li><b>Purpose:</b> Tests if a sample matches a given distribution, emphasizing tail behavior</li>
        <li><b>Implementation:</b> <code>scipy.stats.anderson()</code></li>
        <li><b>Interpretation:</b> Compares test statistic against critical values; statistic > critical value indicates non-normality</li>
        <li><b>Advantage:</b> Sensitive to deviations in distribution tails</li>
        <li><b>Limitation:</b> Does not directly provide a p-value (uses critical values instead)</li>
    </ul>

    <p><b>General Recommendations:</b></p>
    <ul>
        <li>Always combine these tests with visual methods (Q-Q plots, histograms).</li>
        <li>Shapiro-Wilk is preferred for smaller datasets (n < 5000).</li>
        <li>Jarque-Bera and Kolmogorov-Smirnov are suitable for larger datasets.</li>
        <li>Anderson-Darling is excellent when distribution tails are particularly important.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
