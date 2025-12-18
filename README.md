# Vector-BT: Vectorized Backtesting Framework for Single Assets

Vector-BT is a Python-based framework for performing vectorized backtests on trading strategies for a single asset. It provides tools to calculate key performance metrics, visualize results, and analyze trading performance efficiently using pandas and numpy for vectorized operations.

## Features

- **Vectorized Calculations**: Efficient computation of positions, gains, fees, and metrics using pandas Series operations.
- **Performance Metrics**: Includes calculations for:
  - Total return (before and after fees)
  - Margin
  - Sharpe ratio
  - Maximum Drawdown (MDD)
  - Hit rate (overall, long, short)
  - Average win/loss percentages
  - CAGR/MDD ratio
  - Trading frequency
  - etc.
- **Visualization**:
  - Dashboard with equity curve, drawdown, and position charts
  - Monthly returns heatmap
- **Customizable**: Supports transaction fees and multi-level positions (e.g., DCA).

## Installation

Vector-BT requires Python 3.12+.

### Using pip and requirements.txt

Install dependencies:

```
pip install -r requirements.txt
```

## Quick Start

1. Config:

```python
# config.py
SSH_CONFIG = {
    'host': 'your-ssh-server.com',
    'username': 'your-username',
    'key_path': '~/.ssh/id_rsa',  # hoặc đường dẫn SSH key
    'password': None  # hoặc password nếu không dùng key
}

DATABASE_CONFIG = {
    'historical': {
        'remote_port': 5432,
        'db_name': 'historical_data',
        'db_user': 'your-db-user',
        'db_password': 'your-db-password',
        'max_conn': 10
    },
    'live': {
        'remote_port': 5433,
        'db_name': 'live_data',
        'db_user': 'your-db-user',
        'db_password': 'your-db-password',
        'max_conn': 5
    }
}
```

2. Define your strategy: Generate position signals (e.g., Series of positions: positive for long, negative for short, 0 for neutral).

3. Run backtest:

```python
bt = vbt.BacktestInformation(
    Datetime=df['Datetime'],
    Position=position_series,  # Your position signals
    Close=df['Close'],
    fee=0.001  # 0.1% fee per trade
)

# Display metrics and plots
bt.analyze()
```

This will print key metrics and display:
- Performance dashboard (equity curve, drawdown, positions)
- Monthly returns heatmap

## Trade Log Export

Access detailed trade data through the backtest object's DataFrames:

### Complete Backtest Data (.df)
```python
# Access the complete backtest DataFrame
backtest_df = bt.df

# Key columns include:
# - Datetime: Timestamp of each data point
# - Position: Current position size
# - Close: Asset closing price
# - gain: Position gain before fees
# - fee_cost: Transaction fees
# - gain_after_fee: Position gain after fees
# - total_gain: Cumulative gain before fees
# - total_gain_after_fee: Cumulative gain after fees
```

### Trade-Only Data (.df2)
```python
# Access DataFrame with only trade entries (when positions change)
trades_df = bt.df2

# Key columns include:
# - Datetime: Trade timestamp
# - Position: New position after trade
# - input_pos: Position change (positive for buy, negative for sell)
# - gain_after_fee: Realized gain/loss for that trade
```

## Key Methods

- `analyze()`: Computes all metrics and displays dashboard/heatmap.
- `metrics()`: Prints detailed performance statistics.
- `plot_dashboard()`: Visualizes returns, drawdown, and positions.
- `plot_monthly_returns_heatmap()`: Shows monthly performance heatmap.
- Individual metric getters like `Sharp_after_fee()`, `MDD()`, `Hitrate()`, etc.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.