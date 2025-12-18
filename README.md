# Single-Asset Vector Backtest Engine

Python backtesting helper focused on **vectorized** single-asset strategies (pandas/numpy). It builds equity/fee series once, then exposes metrics and plots for quick strategy evaluation.

## What it does
- Vectorized P&L with fees: pre-compute `gain`, `fee_cost`, cumulative `total_gain`/`total_gain_after_fee`.
- Two DataFrames for inspection:
  - `bt.df`: full timeline (Datetime index) with positions, P&L, fees.
  - `bt.df2`: trade-only rows (where position changes).
- Metrics: total return (gross/net), profit per trade, trades/day, MDD, Calmar, Sharpe (buy & hold), hit rates (overall/long/short), Ulcer Index, CDaR, Kelly, etc.
- Visualization:
  - `plot_dashboard()`: equity (net), drawdown %, positions (daily resample), optional buy & hold overlay.
  - `plot_monthly_returns_heatmap()`: monthly heatmap.
  - `analyze()`: prints metrics then shows dashboard + heatmap.
- Utilities: `normalize_df` (clean date/time columns), `resample` OHLCV.

## Install
```bash
pip install -r requirements.txt
```
Requires Python 3.12+.

## Quick start
```python
import pandas as pd
from backtest import BacktestInformation, normalize_df

# Load your OHLCV with a Date column and build positions
df = pd.read_csv("your_data.csv")
df = normalize_df(df)  # adds Datetime column sorted ascending

position = ...  # pd.Series of target position per bar (long>0, short<0, flat=0)

bt = BacktestInformation(
    Datetime=df["Datetime"],
    Position=position,
    Close=df["Close"],
    fee=0.001,  # e.g., 0.1% per round trip (split internally)
)

bt.analyze()  # prints metrics, shows dashboard + monthly heatmap

# Access data
timeline = bt.df   # full timeline
trades = bt.df2    # trades only
```

## Main methods (backtest.BacktestInformation)
- `analyze(figsize=(15, 8), show_buy_hold=True)`
- `metrics(window_MA=None, plot=True)`
- `plot_dashboard(figsize=(15, 8), show_buy_hold=True)`
- `plot_monthly_returns_heatmap()`
- Metric helpers: `Margin()`, `MDD()`, `Total_Return_Percent[_After_Fee]()`, `Hitrate[_long/_short]()`, `Profit_per_trade()`, `Ulcer_Index()`, `CDaR()`, `Kelly_Criterion()`, etc.

## Notes
- Positions and prices are expected as aligned Series; fee is split buy/sell internally.
- Resampling inside plots/metrics uses daily frequency to align equity/drawdown.
- If you need to bind to databases or SSH (requirements.txt includes DB/SSH libs), add your own data-loading code before constructing `BacktestInformation`.
