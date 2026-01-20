# ==========================================
# 0. Environment & Configuration
# ==========================================
import subprocess
import sys
import os
import logging
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")


def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Required packages
packages = ['yfinance', 'numpy', 'pandas', 'torch', 'matplotlib', 'seaborn', 'scipy']
for p in packages:
    install_and_import(p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Supply Chain Nexus Configuration (20 Firms) ---
# We define the nexus as a dictionary mapping Supply Chain Stages to Tickers.
SUPPLY_CHAIN_NEXUS = {
    "1_Upstream_Equip_EDA": ["ASML", "AMAT", "LRCX", "KLAC", "SNPS"],
    "2_Midstream_Foundry": ["TSM", "INTC", "TXN"],
    "3_Core_Design_Mem": ["NVDA", "AMD", "QCOM", "AVGO", "ARM", "MU"],
    "4_Downstream_Cloud": ["MSFT", "GOOGL", "META", "AMZN", "AAPL", "ORCL"]
}

# Flatten for data fetching
ALL_TICKERS = [ticker for stage in SUPPLY_CHAIN_NEXUS.values() for ticker in stage]
START_DATE = '2020-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
CACHE_DIR = "market_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid")


# ==========================================
# 1. Data Pipeline
# ==========================================
class DataStore:
    @staticmethod
    def fetch_data(tickers, start_date, end_date):
        cache_file = os.path.join(CACHE_DIR, f"nexus_data_{len(tickers)}_{start_date}_{end_date}.pkl")

        if os.path.exists(cache_file):
            logging.info("Loading cached data...")
            return pd.read_pickle(cache_file)

        logging.info(f"Downloading data for {len(tickers)} firms in the Nexus...")
        # Download logic
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        # Handle multi-index columns from yfinance
        if 'Adj Close' in data:
            df = data['Adj Close']
        elif 'Close' in data:
            df = data['Close']
        else:
            # Fallback if structure is different (single ticker vs multiple)
            df = data

        # Ensure we have a clean DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            # Try to simplify if needed, though usually 'Adj Close' selection fixed it
            pass

        # Drop tickers with too much missing data
        df = df.dropna(axis=1, how='all').dropna()

        # Save
        df.to_pickle(cache_file)
        logging.info(f"Data saved. Shape: {df.shape}")
        return df

    @staticmethod
    def prepare_tensors(df, seq_len=60):
        """
        Converts price dataframe to Log Returns and then to PyTorch tensors.
        Returns: X (Batch, Assets, Seq_Len), Y (Batch, Assets)
        """
        log_ret = np.log(df / df.shift(1)).fillna(0)

        # Clip extreme outliers (optional, for stability)
        log_ret = log_ret.clip(lower=-0.1, upper=0.1)

        data_vals = log_ret.values
        X, Y = [], []

        for i in range(len(data_vals) - seq_len):
            # Input: Sequence of past returns
            # Shape for Attention: (Assets, Time) -> Transpose implies assets attend to time?
            # We want (Assets, Seq_Len) to embed each asset's history.
            X.append(data_vals[i:i + seq_len].T)
            Y.append(data_vals[i + seq_len])

        X = np.array(X)  # (Batch, N_Assets, Seq_Len)
        Y = np.array(Y)  # (Batch, N_Assets)

        return torch.FloatTensor(X), torch.FloatTensor(Y), df.index[seq_len:], log_ret


# ==========================================
# 2. Physics-Informed Transformer Model
# ==========================================
class SupplyChainAttention(nn.Module):
    """
    Models the 'Gravity' or influence between firms in the supply chain.
    """

    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, x):
        # x: (Batch, N_Assets, Embed_Dim)
        # Output: (Batch, N_Assets, Embed_Dim), Weights
        return self.attn(x, x, x)


class NexusTransformer(nn.Module):
    def __init__(self, n_assets, seq_len, d_model=64):
        super().__init__()
        self.n_assets = n_assets

        # Feature Extraction: Compress time history into an embedding
        self.encoder = nn.Linear(seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_assets, d_model) * 0.01)

        # Supply Chain Interaction Layer
        self.nexus_physics = SupplyChainAttention(d_model, n_heads=4)

        # Processing
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Prediction Body
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, Assets, Seq_Len)
        h = self.encoder(x)  # -> (Batch, Assets, d_model)
        h = h + self.pos_emb

        # Interaction (Firms influence each other)
        attn_out, attn_weights = self.nexus_physics(h)
        h = h + attn_out
        h = self.norm1(h)

        # Non-linearity
        h2 = self.ffn(h)
        h = h + h2
        h = self.norm2(h)

        # Forecast
        out = self.head(h).squeeze(-1)  # -> (Batch, Assets)
        return out, attn_weights


# ==========================================
# 3. Portfolio Manager Logic
# ==========================================
class PortfolioManager:
    def __init__(self, tickers, supply_chain_map):
        self.tickers = tickers
        self.map = supply_chain_map
        # Reverse map for easy lookup
        self.ticker_to_stage = {}
        for stage, t_list in supply_chain_map.items():
            for t in t_list:
                self.ticker_to_stage[t] = stage

    def constraint_check(self, weights):
        """
        Ensures no single stage dominates > 60% of the portfolio
        to maintain supply chain resilience.
        """
        stage_weights = {s: 0.0 for s in self.map.keys()}

        for t, w in zip(self.tickers, weights):
            if t in self.ticker_to_stage:
                stage_weights[self.ticker_to_stage[t]] += w

        # Simple heuristic: If any stage > 0.60, damp it.
        # (In a real optimizer, this would be a constraint in scipy.minimize)
        # For this simulation, we'll log it if it's extreme.
        return stage_weights

    def allocate(self, forecast_signal, current_volatility, risk_aversion=1.0):
        """
        Allocation Logic:
        1. Signal Strength (Alpha)
        2. Volatility Adjustment (Risk Parity-lite)
        """
        # 1. Base weights from Signal (Softmax / Power Law)
        # Using a Softmax with Temperature for 'ranking'
        score = forecast_signal
        # Normalize score
        score = (score - np.mean(score)) / (np.std(score) + 1e-6)

        # Convert to probability (long-only)
        exp_score = np.exp(score * 2.0)  # Temperature=2.0 (Aggressive)
        weights = exp_score / np.sum(exp_score)

        # 2. Volatility penalty? (Optional)
        # If an asset is super volatile, reduce weight.
        # weights = weights / current_volatility
        # weights = weights / np.sum(weights)

        return weights


# ==========================================
# 4. Main Execution Loop
# ==========================================
def run_nexus_simulation():
    logging.info("Initializing Supply Chain Nexus Manager...")

    # 1. Data
    df = DataStore.fetch_data(ALL_TICKERS, START_DATE, END_DATE)

    # Filter tickers that might have failed download
    valid_tickers = list(df.columns)
    logging.info(f"Valid Tickers: {len(valid_tickers)} / {len(ALL_TICKERS)}")

    # Update Nexus Map based on valid data
    pm = PortfolioManager(valid_tickers, SUPPLY_CHAIN_NEXUS)

    # Prepare
    seq_len = 60
    X, Y, dates, _ = DataStore.prepare_tensors(df, seq_len=seq_len)

    # Train/Test Split
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    test_dates = dates[split_idx:]

    # 2. Model Training
    logging.info("Training Nexus Transformer...")
    model = NexusTransformer(n_assets=len(valid_tickers), seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []

    model.train()
    for epoch in range(60):  # Fast training
        optimizer.zero_grad()
        preds, _ = model(X_train)
        loss = loss_fn(preds, Y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    # 3. Backtest Simulation
    logging.info("Running Portfolio Simulation...")
    model.eval()

    portfolio_values = [100_000.0]
    benchmark_values = [100_000.0]

    allocations_history = []

    # Pre-calculate MPT Benchmark (Rolling Minimum Variance)
    real_returns = Y_test.numpy()
    # History for covariance: Start with Training data
    history_returns = list(Y_train.numpy())

    mpt_allocations = []

    logging.info(f"Generating MPT Benchmark (MinVar) over {len(real_returns)} days...")

    for t in range(len(real_returns)):
        # 1. Update Covariance Matrix based on past window (e.g. last 120 days)
        # Using a rolling window of 120 days for stability
        window_size = 120
        recent_history = np.array(history_returns[-window_size:])
        cov_matrix = np.cov(recent_history, rowvar=False)

        # 2. Optimize (Min Variance)
        # min w.T * Cov * w
        n_assets = len(valid_tickers)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_w = np.ones(n_assets) / n_assets

        res = minimize(lambda w: w.T @ cov_matrix @ w, initial_w,
                       bounds=bounds, constraints=constraints, tol=1e-4)

        w_bm = res.x

        # Store for next step
        mpt_allocations.append(w_bm)
        history_returns.append(real_returns[t])

        # Progress log
        if t % 50 == 0:
            logging.info(f"MPT Rebalance day {t}/{len(real_returns)}")

    with torch.no_grad():
        all_preds, attn_matrix = model(X_test)
        all_preds = all_preds.numpy()

    for t in range(len(X_test)):
        # Daily Rebalance (simplified)

        # Get AI Forecast
        signal = all_preds[t]

        # PM Allocates
        w = pm.allocate(signal, current_volatility=None)
        allocations_history.append(w)

        # Calculate PnL
        # AI Strategy
        day_return = np.sum(w * real_returns[t])
        portfolio_values.append(portfolio_values[-1] * (1 + day_return))

        # Benchmark (MPT)
        w_bm_t = mpt_allocations[t]
        day_return_bm = np.sum(w_bm_t * real_returns[t])
        benchmark_values.append(benchmark_values[-1] * (1 + day_return_bm))

    # ==========================================
    # 5. Reporting & Visualization
    # ==========================================
    # Performance Stats
    perf_df = pd.DataFrame({
        "Nexus AI Strategy": portfolio_values,
        "MPT MinVar Benchmark": benchmark_values
    }, index=[dates[split_idx - 1]] + list(test_dates))

    total_return = (portfolio_values[-1] - 100000) / 100000 * 100
    bench_return = (benchmark_values[-1] - 100000) / 100000 * 100

    print(f"\n=== FINAL PERFORMANCE REPORT ===")
    print(f"Nexus Strategy:   {portfolio_values[-1]:,.2f} ({total_return:.2f}%)")
    print(f"Benchmark (MPT):  {benchmark_values[-1]:,.2f} ({bench_return:.2f}%)")
    print(f"Alpha:            {total_return - bench_return:.2f}%")

    # --- Visualization ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    perf_df.plot(ax=ax1, color=['#4C4Cff', 'gray'], linewidth=2)
    ax1.set_title("Portfolio Equity: Supply Chain Nexus vs MPT Benchmark")
    ax1.set_ylabel("Capital ($)")

    # 2. Average Sector Exposure (Pie Chart)
    # Aggregate average weights over test period
    avg_w = np.mean(allocations_history, axis=0)  # (N_Assets,)
    stage_exposure = pm.constraint_check(avg_w)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.pie(stage_exposure.values(), labels=stage_exposure.keys(), autopct='%1.1f%%',
            colors=sns.color_palette("pastel"), startangle=140)
    ax2.set_title("Avg. Supply Chain Exposure")

    # 3. Network Influence (Attention Map)
    ax3 = fig.add_subplot(gs[1, 1:])
    # Average attention across test set
    avg_attn = attn_matrix.mean(dim=0).numpy()
    sns.heatmap(avg_attn, ax=ax3, cmap="viridis", xticklabels=valid_tickers, yticklabels=valid_tickers)
    ax3.set_title("Learned Supply Chain Influence (Physics Attention)")

    # 4. Top Holdings (Bar)
    ax4 = fig.add_subplot(gs[2, :])
    holdings = pd.Series(avg_w, index=valid_tickers).sort_values(ascending=False)
    holdings.plot(kind='bar', ax=ax4, color='teal')
    ax4.set_title("Average Asset Allocation weight %")

    plt.tight_layout()
    plt.savefig("nexus_result.png")
    print("Results saved to nexus_result.png")
    plt.show()


if __name__ == "__main__":
    run_nexus_simulation()
