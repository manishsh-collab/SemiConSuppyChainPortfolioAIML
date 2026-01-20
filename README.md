# Supply Chain Nexus — Physics-Informed Transformer for Cross‑Firm Allocation

A research / demo prototype that models interactions across a semiconductor-to-cloud supply‑chain nexus, trains a physics‑informed Transformer to forecast short‑term returns, and runs a simple portfolio simulation that compares an AI-driven allocation versus a Minimum‑Variance (MPT) benchmark.

This repository is an experimental research artifact — educational only, not financial advice.

---

## Contents

- Summary
- Key concepts
- Quickstart
- What the code does (high level)
- Main components / functions
- Configuration & knobs
- Outputs & visualization
- Caveats & recommended improvements
- License

---

## Summary

This single‑script prototype:

- Downloads historical prices for a curated set of firms that form a "Supply Chain Nexus" (20 tickers across upstream, midstream, core design/memory and downstream/cloud).
- Converts prices to log‑returns and prepares sequences for a Transformer model.
- Trains a NexusTransformer with an attention layer that models inter‑firm influence (a "supply‑chain gravity" mechanism).
- Uses model forecasts to produce allocation weights via a PortfolioManager and simulates daily rebalancing P&L.
- Builds an MPT (minimum‑variance) rolling benchmark and compares performance.
- Produces diagnostics including equity curves, attention heatmap, sector exposure pie, and top holdings bar chart.

---

## Key concepts

- Supply Chain Nexus: Firms grouped by role in the hardware → software value chain, used both for modeling and portfolio exposure constraints.
- Physics‑informed attention: The model is structured so assets produce embeddings from their recent price history and then interact through a multi‑head attention module to capture cross‑firm influence.
- Sequence input: For each day we use past `seq_len` log‑returns across all assets to produce asset‑level embeddings and forecasts.
- Simple allocation: Forecasts map to weights using a softmax-based ranking; the PortfolioManager enforces/monitors stage concentration heuristics.
- Benchmark: A rolling Minimum‑Variance portfolio is computed using a covariance window and SciPy optimization.

---

## Quickstart

Requirements
- Python 3.8+
- pip

The script attempts to install missing packages automatically (yfinance, numpy, pandas, torch, matplotlib, seaborn, scipy). For reproducibility, it's recommended to create a virtual environment and install dependencies manually:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install yfinance numpy pandas torch matplotlib seaborn scipy
```

Run the simulation:

```bash
python main.py
```

Output files:
- `nexus_result.png` — composite figure with equity curves, sector exposure pie, attention heatmap, and holdings chart.

---

## What the code does (high level)

1. Builds a list of tickers for each supply‑chain stage and flattens them to fetch historical prices (default from 2020‑01‑01 to today).
2. Caches downloaded price data to `market_data_cache/` for faster repeated runs.
3. Converts prices to clipped log‑returns and builds tensors of shape (Batch, Assets, Seq_Len) for training.
4. Defines a NexusTransformer:
   - Linear encoder compresses time history to an asset embedding.
   - A MultiheadAttention layer models cross‑asset influence.
   - Feed‑forward blocks and a scalar head produce 1‑step (asset) forecasts.
5. Trains the model for a small number of epochs on the training split.
6. Runs a backtest on the test split:
   - Produces model allocations via softmax scoring.
   - Computes daily P&L for AI strategy and a precomputed rolling Minimum‑Variance benchmark.
7. Reports final performance and visualizes results.

---

## Main components

- DataStore.fetch_data(tickers, start, end)
  - Downloads and caches adjusted close (or close) prices; returns a cleaned DataFrame.

- DataStore.prepare_tensors(df, seq_len=60)
  - Converts price series to log‑returns, clips extremes, and returns PyTorch tensors: X (Batch, Assets, Seq_Len), Y (Batch, Assets), dates, and the log_ret DataFrame.

- NexusTransformer (nn.Module)
  - Encodes asset histories into embeddings, applies attention, feed‑forward layers, and predicts next‑step returns per asset. Returns predictions and attention weights.

- PortfolioManager
  - Maps tickers to supply chain stages, computes allocation weights from model forecasts (softmax ranking with temperature), and computes/checks stage exposure for resilience heuristics.

- run_nexus_simulation()
  - Orchestrates the full pipeline: data load, train/test split, model training, MPT benchmark generation, simulated rebalancing, and visualization.

---

## Configuration & knobs

- SUPPLY_CHAIN_NEXUS: Change tickers or add/remove stages to experiment with different universes.
- START_DATE / END_DATE: Time window for historical data.
- CACHE_DIR: Where downloaded data is pickled.
- seq_len (default 60): Number of past days used per input sample.
- Train/Test split ratio: Currently 75% train / 25% test (in code).
- Model hyperparameters:
  - d_model (default 64)
  - attention heads (4)
  - learning rate (0.001)
  - epochs (default 60)
- Allocation logic:
  - Softmax temperature (currently hard-coded as `* 2.0`)
  - Risk aversion or volatility adjustments can be enabled in PortfolioManager.allocate
- Benchmark:
  - Covariance rolling window (currently 120 days)
  - MPT uses bounds (0,1) and full sum-to-1 constraint

To experiment, edit top-level variables or refactor into a small config block / CLI.

---

## Outputs & interpretation

- Printed:
  - Valid tickers count
  - Epoch training loss messages
  - Final performance summary: final capital for AI strategy and MPT benchmark, and alpha (% difference).
- File: `nexus_result.png` — composite visualization:
  - Equity curves (AI vs MPT)
  - Average supply‑chain exposure (pie)
  - Attention heatmap (learned cross‑asset influence)
  - Average asset allocation bar chart

These outputs are diagnostic — look at attention and exposures to understand whether the Transformer learned sensible cross‑firm patterns.

---

## Caveats, limitations & recommended improvements

- Educational demo: No transaction costs, slippage, market impact, shorting/borrowing constraints, or realistic execution model are present.
- Data hygiene: The fetch logic drops NA rows; for robust research consider aligning calendars across tickers and handling corporate actions explicitly.
- Model training is minimalist: a small number of epochs and no validation loop; consider:
  - Time‑aware validation (walk‑forward)
  - Early stopping
  - Hyperparameter search (Optuna)
  - Larger models or regularization
- Allocation is simple and long‑only. Add volatility scaling, maximum position sizes, transaction cost penalties or turnover constraints when moving toward realistic simulations.
- MPT benchmark uses a small sample covariance which can be singular for few samples — regularize the covariance or shrink toward diagonal.
- Reproducibility: The script sets seeds for NumPy and PyTorch but GPU nondeterminism and external library variance can still cause small differences.

---

## Ideas for extension

- Add CLI arguments (click/argparse) and a config file (YAML/JSON).
- Persist trained models and run multiple seeds/experiments storing run metadata.
- Replace softmax allocation with risk‑parity or mean‑variance optimizer that uses model expected returns as inputs.
- Include macro indicators or alternative data (supply chain KPIs, inventories, shipment indices).
- Support short positions and margin constraints.
- Add unit tests for feature processing and the allocation logic.
- Integrate with a backtesting framework (vectorbt, bt, backtrader) for richer analytics.

---

## License

This repository has no license specified. If you intend to reuse or distribute the code, add an explicit license (for example, MIT) by creating a `LICENSE` file.

---

File reference
- Main script: `main.py` (entry point)
- Cache directory: `market_data_cache/`
