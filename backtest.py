import numpy as np
import pandas as pd
from time import mktime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import Optional, Tuple
import matplotlib.colors as mcolors

# =====================
# Utility Functions
# =====================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and format date/time columns in the DataFrame."""
    df['Datetime'] = pd.to_datetime(df.Date)
    df = df.sort_values(by='Datetime')
    df['Date'] = df.Datetime.dt.date
    df['Date'] = df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    df['time'] = df.Datetime.dt.time
    df['time'] = df['time'].apply(lambda x: x.strftime('%H:%M:%S'))
    return df

def resample(df: pd.DataFrame, sample_duration: int, type_data: str) -> pd.DataFrame:
    """Resample OHLCV data by a given duration and type."""
    df.Date = pd.to_datetime(df.Date)
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    df = pd.DataFrame(df.resample(f'{sample_duration}{type_data}', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()
    return df

# =====================
# Backtest Information
# =====================
class BacktestInformation:
    """
    Thông tin backtest của chiến thuật giao dịch.
    Input:
        Datetime: Series Datetime
        Position: Series Position
        Close: Series Close
        fee: float (transaction fee)
    """
    def __init__(self, Datetime: pd.Series, Position: pd.Series, Close: pd.Series, fee: float):
        fee = fee / 2
        max_pos = Position.abs().max()
        df = pd.DataFrame()
        df['Datetime'] = pd.to_datetime(Datetime)
        df['Position'] = Position.values #/ max_pos
        df['Close'] = Close.values
        df['input_pos'] = df['Position'].diff().fillna(df['Position'].iloc[0])
        df['gain'] = (df['Position'].shift(1).fillna(0) * df['Close'].diff()).fillna(0)
        df['fee_cost'] = fee * df['Close'] * np.abs(df['input_pos'])
        df['gain_after_fee'] = df['gain'] - df['fee_cost']
        df['total_gain'] = df.gain.cumsum()
        df['total_gain_after_fee'] = df.gain_after_fee.cumsum()

        df = df.set_index(df.Datetime)
        self.df = df
        self.df2 = df[df.input_pos != 0]
        self.fee = fee

    # =====================
    # Buy & Hold Calculations
    # =====================
    def _calculate_buy_hold(self) -> Tuple[pd.Series, float]:
        """Calculate Buy & Hold return and Sharpe ratio."""
        df = self.df.copy()

        # Calculate Buy & Hold returns
        initial_price = df['Close'].iloc[0]
        buy_hold_return = (df['Close'] - initial_price) / initial_price

        # Resample to daily
        buy_hold_daily = buy_hold_return.resample("1D").last().dropna()

        # Calculate Sharpe ratio for Buy & Hold
        buy_hold_daily_diff = buy_hold_daily.diff().dropna()
        buy_hold_daily_diff.iloc[0] = 0 if len(buy_hold_daily_diff) > 0 else 0

        buy_hold_sharpe = 0
        if len(buy_hold_daily_diff) > 0 and buy_hold_daily_diff.std() != 0:
            buy_hold_sharpe = (buy_hold_daily_diff.mean() / buy_hold_daily_diff.std()) * np.sqrt(365)

        return buy_hold_daily * 100, buy_hold_sharpe  # Convert to percentage

    # =====================
    # Dashboard Visualization
    # =====================
    def plot_dashboard(self, figsize: Tuple[int, int] = (15, 8), show_buy_hold: bool = True) -> None:
        """
        Plot a dashboard with equity curve, drawdown (%), and position visualization.
        Uses daily resampled data to match metrics consistency, ensuring proper date range.
        """
        df = self.df.copy()

        # Calculate percentage returns and resample to daily (exactly matching metrics)
        df['gain_per_fee'] = df.gain_after_fee / df.Close
        pnl_perc = df['gain_per_fee'].cumsum()
        pnl_perc_daily = pnl_perc.resample("1D").last().dropna()

        # Calculate drawdown in percentage
        return_perc = pnl_perc_daily * 100  # Convert to percentage
        return_perc_peak = return_perc.cummax()
        drawdown_perc = return_perc - return_perc_peak

        # Position resampled to daily (last position of the day)
        position_daily = df.Position.resample("1D").last().dropna()

        # Filter out any dates before data actually starts (remove 1970 artifacts)
        actual_start_date = df.index.min().normalize()  # Get the actual start date
        pnl_perc_daily = pnl_perc_daily[pnl_perc_daily.index >= actual_start_date]
        drawdown_perc = drawdown_perc[drawdown_perc.index >= actual_start_date]
        position_daily = position_daily[position_daily.index >= actual_start_date]

        # Calculate Buy & Hold (only if requested)
        if show_buy_hold:
            buy_hold_perc, buy_hold_sharpe = self._calculate_buy_hold()
            buy_hold_perc = buy_hold_perc[buy_hold_perc.index >= actual_start_date]

        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

        # 1. Return % curve with Sharpe ratio
        return_perc = pnl_perc_daily * 100  # Convert to percentage
        return_perc_peak = return_perc.cummax()

        # Calculate pre-fee return
        df['gain_per'] = df.gain / df.Close
        pnl_perc_prefee = df['gain_per'].cumsum()
        pnl_perc_prefee_daily = pnl_perc_prefee.resample("1D").last().dropna()
        prefee_perc = pnl_perc_prefee_daily * 100
        prefee_perc = prefee_perc[prefee_perc.index >= actual_start_date]

        # Calculate Sharpe ratios
        sharpe_ratio_after_fee = self.Sharpe_after_fee()
        sharpe_ratio_before_fee = self.Sharpe()

        axs[0].plot(return_perc.index, return_perc, label=f"Return % after fee (Sharpe: {sharpe_ratio_after_fee:.2f})",
                   color="orange", linewidth=1.5)
        axs[0].plot(prefee_perc.index, prefee_perc, label=f"Return % before fee (Sharpe: {sharpe_ratio_before_fee:.2f})", color="tab:blue", linewidth=1.2)
        if show_buy_hold:
            axs[0].plot(buy_hold_perc.index, buy_hold_perc, label=f"Buy & Hold (Sharpe: {buy_hold_sharpe:.2f})",
                       color="gray", linewidth=1.2, linestyle="--")
        axs[0].plot(return_perc.index, return_perc_peak, color='gray', linewidth=0.8, alpha=0.7,
                   label="High-water mark")
        axs[0].set_ylabel("Return (%)")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # 2. Drawdown in percentage
        axs[1].fill_between(drawdown_perc.index, drawdown_perc, 0, color='red', alpha=0.4, label='Drawdown %')
        axs[1].plot(drawdown_perc.index, drawdown_perc, color='darkred', linewidth=1)
        axs[1].set_ylabel("Drawdown (%)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # 3. Position visualization – hỗ trợ multi-level DCA
        max_p = int(self.df['Position'].abs().max())
        # Vẽ vùng long (dương) và short (âm)
        axs[2].fill_between(position_daily.index, 0, position_daily,
                           where=(position_daily > 0), color='#2ca02c', alpha=0.6, label='Long', step='post')
        axs[2].fill_between(position_daily.index, 0, position_daily,
                           where=(position_daily < 0), color='#d62728', alpha=0.6, label='Short', step='post')
        axs[2].axhline(0, color='black', linewidth=0.8, alpha=0.5)
        axs[2].set_ylabel("Position size")
        axs[2].set_xlabel("Date")
        axs[2].set_ylim(-max_p - 0.5, max_p + 0.5)
        # Giảm số lượng y-ticks để tránh dày đặc
        # Giảm số lượng y-ticks để tránh dày đặc
        from matplotlib.ticker import MaxNLocator
        axs[2].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axs[2].legend(title="Position")
        axs[2].grid(True, alpha=0.3)

        # Set x-axis limits to actual data range
        for ax in axs:
            ax.set_xlim(pnl_perc_daily.index.min(), pnl_perc_daily.index.max())

        # Format x-axis dates
        axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        fig.tight_layout()
        plt.show()


    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot monthly returns heatmap with proper data type handling.
        """
        df = self.df.copy()

        # Calculate percentage returns (same as in plot_dashboard)
        df['gain_per_fee'] = df.gain_after_fee / df.Close

        # Resample to monthly returns - fix deprecated 'M' to 'ME'
        monthly_returns = df['gain_per_fee'].resample("ME").sum().dropna() * 100

        # Group by year and month
        monthly_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).sum()

        # Create heatmap data structure with explicit float dtype
        years = sorted(set(monthly_returns.index.year))
        months = range(1, 13)
        heatmap_data = pd.DataFrame(index=years, columns=months, dtype=float).fillna(0.0)

        # Populate heatmap data with explicit float conversion
        for (year, month), value in monthly_data.items():
            heatmap_data.loc[year, month] = float(value)



        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        # Create custom diverging palette: Red (negative) -> White (0) -> Green (positive)
        colors = ['#d62728', '#ffffff', '#2ca02c']  # Red -> White -> Green
        n_bins = 256
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('red_white_green', colors, N=n_bins)

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=custom_cmap,
                    center=0, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')

        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)

        plt.tight_layout()
        plt.show()

    # =====================
    # Combined Function
    # =====================
    def analyze(self, figsize: Tuple[int, int] = (15, 8), show_buy_hold: bool = True) -> None:
        """
        Combined function to display metrics and dashboard visualization.
        """
        # Display metrics
        self.metrics(plot=True)
        print("\n" + "="*50 + "\n")

        # Display dashboard
        self.plot_dashboard(figsize=figsize, show_buy_hold=show_buy_hold)

            # Display monthly returns heatmap
        print("\n" + "="*30 + " Monthly Returns Analysis " + "="*30 + "\n")
        self.plot_monthly_returns_heatmap()

    # =====================
    # Original metrics (kept as requested)
    # =====================
    def metrics(self, window_MA: Optional[int] = None, plot: bool = True) -> None:
        """
        Print summary and plot original PNL charts (kept for compatibility).
        """
        if plot:
            print('Margin:', np.round(self.Margin(), 2))
            print('Margin after fee:', np.round(self.Margin_after_fee(), 2))
            print('MDD: ' + str(self.MDD()[0]) + ' (' + str(np.round(self.MDD()[1], 2)) + '%)')
            print("Calmar:", np.round(self.CAGR_MDD_Ratio(), 2))
            avg_profit_win, avg_loss_loss = self.AvgWinLoss()

            data = [
                ('Total trades', np.round(self.Number_of_trade(), 2)),
                ('Profit per trade', np.round(self.Profit_per_trade(), 2)),
                ('Total Profit', f"{np.round(self.df.total_gain.iloc[-1], 2)} ({np.round(self.Total_Return_Percent(), 2)}%)"),
                ('Profit after fee', f"{np.round(self.Profit_after_fee(), 2)} ({np.round(self.Total_Return_Percent_After_Fee(), 2)}%)"),
                ('Trades per day', np.round(self.Trading_per_day(), 2)),
                ('Profit per year', np.round(self.Profit_per_year(), 2)),
                ('HitRate', str(np.round(self.Hitrate() * 100, 2)) + '%'),
                ('Daily HitRate', str(np.round(self.Hitrate_per_day() * 100, 2)) + '%'),
                ('Average profit when winning', str(np.round(avg_profit_win, 2)) + '%'),
                ('Average loss when losing', str(np.round(avg_loss_loss, 2)) + '%'),
                ('Long', self.Hitrate_long()[1]),
                ('Short', self.Hitrate_short()[1]),
                ('Hitrate long', str(np.round(self.Hitrate_long()[0] * 100, 2)) + '%'),
                ('Hitrate short', str(np.round(self.Hitrate_short()[0] * 100, 2)) + '%'),
                ('Longest consecutive losing days', self.Longest_consecutive_losing_days()),
                ('Ulcer Index', self.Ulcer_Index()),
                ('CDaR', self.CDaR()),
                ('Kelly Criterion', self.Kelly_Criterion()),
            ]
            for row in data:
                print('{:>25}: {:>1}'.format(*row))

    # =====================
    # (Các hàm tính toán giữ nguyên, chỉ bổ sung type-hint, docstring, clean code)
    # =====================
    def Ulcer_Index(self) -> float:
        """Tính Ulcer Index trên daily equity."""
        df = self.df.copy()
        df['gain_per_fee'] = df.gain_after_fee / df.Close
        pnl_perc = df['gain_per_fee'].cumsum()
        pnl_daily = pnl_perc.resample("1D").last().dropna() * 100
        peak = pnl_daily.cummax()
        
        # Xử lý trường hợp peak == 0 để tránh chia cho 0
        dd = pd.Series(index=pnl_daily.index, dtype=float)
        for idx in pnl_daily.index:
            if peak[idx] == 0:
                dd[idx] = 100 if pnl_daily[idx] < 0 else 0  # Gán 100% loss hoặc 0% nếu equity âm/ngang
            else:
                dd[idx] = ((peak[idx] - pnl_daily[idx]) / peak[idx]) * 100  # Drawdown % bình thường
        
        # Tính Ulcer Index chỉ trên các giá trị hợp lệ, bỏ nan/inf
        dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd) == 0:
            return 0.0  # Trả về 0 nếu không có drawdown hợp lệ
        ui = np.sqrt((dd ** 2).mean())
        return np.round(ui, 2)

    def CDaR(self, alpha: float = 0.95) -> float:
        """Tính CDaR trên equity curve phần trăm."""
        df = self.df.copy()
        df['gain_per_fee'] = df.gain_after_fee / df.Close
        pnl_perc = df['gain_per_fee'].cumsum()  # Return tích lũy
        pnl_daily = pnl_perc.resample("1D").last().dropna() * 100  # Phần trăm daily
        peak = pnl_daily.cummax()
        dd = peak - pnl_daily  # Drawdowns
        sorted_dd = dd.sort_values(ascending=False)
        dar = sorted_dd.iloc[int(len(sorted_dd) * (1 - alpha))]  # DaR
        cdd = sorted_dd[sorted_dd >= dar].mean()  # Conditional mean
        return np.round(cdd, 2)

    def Kelly_Criterion(self, fractional: float = 1.0) -> float:
        """
        Tính Kelly fraction dựa trên historical trades.
        
        Parameters:
        - fractional: Scale factor (default 1.0 cho full Kelly, dùng <1 để conservative).
        
        Returns:
        - Kelly fraction (decimal) sau khi scale.
        """
        p = self.Hitrate()  # Win probability
        if p == 0 or p == 1:  # No edge or perfect (rare)
            return 0.0
        
        avg_win_perc, avg_loss_perc = self.AvgWinLoss()
        W = abs(avg_win_perc) / 100.0  # Average win decimal
        L = abs(avg_loss_perc) / 100.0  # Average loss magnitude decimal
        
        if L == 0:  # No losses, infinite b (but cap to avoid overbet)
            return min(p * fractional, 1.0)
        if W == 0:  # No wins
            return 0.0
        
        b = W / L  # Reward/risk ratio
        q = 1 - p
        f_star = p - (q / b)  # Kelly fraction
        f_star = max(f_star, 0.0)  # Không âm
        
        return np.round(f_star * fractional, 4)

    def AvgWinLoss(self) -> Tuple[float, float]:
        """Compute average profit when winning and average loss when losing as percentages"""
        df = self.df.copy()
        trades_list = []
        current_position = 0
        entry_price = 0
        entry_index = None

        for idx, row in df.iterrows():
            if current_position == 0 and row['Position'] != 0:
                # Open new position
                current_position = row['Position']
                entry_price = row['Close']
                entry_index = idx
            elif current_position != 0 and row['Position'] == 0:
                # Close position
                exit_price = row['Close']
                exit_index = idx
                # Calculate percentage gain
                if current_position > 0:
                    # Long position
                    pct_gain = (exit_price - entry_price) / entry_price
                else:
                    # Short position
                    pct_gain = (entry_price - exit_price) / entry_price
                trades_list.append({
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Position': current_position,
                    'Datetime_Entry': entry_index,
                    'Datetime_Exit': exit_index,
                    'Percentage_Gain': pct_gain
                })
                current_position = 0
                entry_price = 0
                entry_index = None
            elif current_position != 0 and row['Position'] != current_position:
                # Position changed (e.g., from long to short)
                # Close existing position
                exit_price = row['Close']
                exit_index = idx
                if current_position > 0:
                    # Long position
                    pct_gain = (exit_price - entry_price) / entry_price
                else:
                    # Short position
                    pct_gain = (entry_price - exit_price) / entry_price
                trades_list.append({
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Position': current_position,
                    'Datetime_Entry': entry_index,
                    'Datetime_Exit': exit_index,
                    'Percentage_Gain': pct_gain
                })
                # Open new position
                current_position = row['Position']
                entry_price = row['Close']
                entry_index = idx

        # After the loop, check if there's an open position
        if current_position != 0 and entry_price != 0:
            # Close the position at the last available price
            exit_price = df['Close'].iloc[-1]
            exit_index = df.index[-1]
            if current_position > 0:
                # Long position
                pct_gain = (exit_price - entry_price) / entry_price
            else:
                # Short position
                pct_gain = (entry_price - exit_price) / entry_price
            trades_list.append({
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Position': current_position,
                'Datetime_Entry': entry_index,
                'Datetime_Exit': exit_index,
                'Percentage_Gain': pct_gain
            })

        trades = pd.DataFrame(trades_list)

        if trades.empty:
            avg_profit_win = 0
            avg_loss_loss = 0
        else:
            winning_trades = trades[trades['Percentage_Gain'] > 0]
            losing_trades = trades[trades['Percentage_Gain'] < 0]
            avg_profit_win = winning_trades['Percentage_Gain'].mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss_loss = losing_trades['Percentage_Gain'].mean() * 100 if len(losing_trades) > 0 else 0

        return avg_profit_win, avg_loss_loss

    def CAGR_MDD_Ratio(self) -> float:
        """Tính tỷ lệ CAGR/MDD."""
        cagr = self.Return()  # Đã là CAGR (annualized return)
        max_drawdown_percent = self.MDD()[1] / 100
        if max_drawdown_percent == 0:
            return np.inf
        return cagr / abs(max_drawdown_percent)

    def Total_Return_Percent(self) -> float:
        """Tính tổng return % (before fee)."""
        df = self.df.copy()
        df['gain_per'] = df.gain / df.Close
        return df['gain_per'].cumsum().iloc[-1] * 100

    def Total_Return_Percent_After_Fee(self) -> float:
        """Tính tổng return % (after fee)."""
        df = self.df.copy()
        df['gain_per'] = df.gain_after_fee / df.Close
        return df['gain_per'].cumsum().iloc[-1] * 100

    def Longest_consecutive_losing_days(self) -> int:
        """Tìm số ngày thua lỗ liên tiếp dài nhất."""
        df = self.df
        df.total_gain = df.gain_after_fee.cumsum()
        pnl_daily = df.total_gain.resample("1D").last().dropna()
        gain_daily = pnl_daily.diff()
        gain_daily = gain_daily.apply(lambda x: 1 if x >= 0 else 0)
        longest_zero_sequence = gain_daily.eq(0).astype(int).groupby(gain_daily.ne(0).cumsum()).sum().max()
        return longest_zero_sequence

    def Count_day_trade(self) -> float:
        df = self.df
        df['abs_pos'] = df['Position'].abs()
        trade = df.abs_pos.resample("1D").sum().dropna()
        return np.round(len(trade[trade != 0]) / len(trade) * 100, 2)

    def Overnight_holding(self) -> float:
        df = self.df
        trade = df.Position.resample("1D").last().dropna()
        return np.round(len(trade[trade != 0]) / len(trade) * 100, 2)

    def Margin(self) -> float:
        df = self.df.copy()
        return (df.total_gain.iloc[-1] / (df.Close * np.abs(df.input_pos)).sum()) * 10000

    def Margin_after_fee(self) -> float:
        df = self.df.copy()
        return (df.total_gain_after_fee.iloc[-1] / (df.Close * np.abs(df.input_pos)).sum()) * 10000

    def Sharpe(self) -> float:
        df = self.df.copy()
        df.gain = df.gain / df.Close
        df.total_gain = df.gain.cumsum()
        pnl_daily = df.total_gain.resample("1D").last().dropna()
        gain_daily = pnl_daily.diff()
        gain_daily.iloc[0:1] = 0
        return (gain_daily.mean()) / gain_daily.std() * np.sqrt(365)

    def Sharpe_after_fee(self) -> float:
        df = self.df.copy()
        df.gain_after_fee = df.gain_after_fee / df.Close
        df.total_gain_after_fee = df.gain_after_fee.cumsum()
        pnl_daily = df.total_gain_after_fee.resample("1D").last().dropna()
        gain_daily = pnl_daily.diff()
        gain_daily.iloc[0:1] = 0
        return (gain_daily.mean()) / gain_daily.std() * np.sqrt(365)

    def MDD(self) -> Tuple[float, float]:
        df = self.df.copy()
        df['gain_per'] = df.gain_after_fee / df.Close
        df['total_gain_max'] = df.total_gain_after_fee.cummax()
        df['dd'] = df.total_gain_max - df.total_gain_after_fee
        df['total_gain_per'] = df.gain_per.cumsum()
        df['total_gain_max_per'] = df.total_gain_per.cummax()
        df['dd_per'] = df.total_gain_max_per - df.total_gain_per
        return np.round(df.dd.max(), 2), df.dd_per.max() * 100

    def Hitrate(self) -> float:
        fee = self.fee
        df = self.df2.copy()
        df = df[df.input_pos != 0]
        df['gain'] = df.Position.shift(1) * df.Close.diff()
        df.loc[df.index[0], 'gain'] = 0
        df['fee_cost'] = fee * df['Close'] * np.abs(df['input_pos'])
        df['gain_after_fee'] = df['gain'] - df['fee_cost']
        df = df.set_index(df.Datetime)
        return len(df[df.gain_after_fee > 0]) / self.Number_of_trade()

    def Hitrate_long(self) -> Tuple[float, int]:
        fee = self.fee
        df = self.df2.copy()
        df = df[df.input_pos != 0]
        df['Position'] = df['Position'].replace(-1, 0)
        number_long = len(df[df['Position'] != 0])
        if number_long == 0:
            return 0, 0
        else:
            df['gain'] = df.Position.shift(1) * df.Close.diff()
            df.loc[df.index[0], 'gain'] = 0
            df['fee_cost'] = fee * df['Close'] * np.abs(df['input_pos'])
            df['gain_after_fee'] = df['gain'] - df['fee_cost']
            df['total_gain_after_fee'] = df.gain_after_fee.cumsum()
            df = df.set_index(df.Datetime)
            return len(df[df.gain_after_fee > 0]) / number_long, number_long

    def Hitrate_short(self) -> Tuple[float, int]:
        fee = self.fee
        df = self.df2.copy()
        df = df[df.input_pos != 0]
        df['Position'] = df['Position'].replace(1, 0)
        number_short = len(df[df['Position'] != 0])
        if number_short == 0:
            return 0, 0
        else:
            df['gain'] = df.Position.shift(1) * df.Close.diff()
            df.loc[df.index[0], 'gain'] = 0
            df['fee_cost'] = fee * df['Close'] * np.abs(df['input_pos'])
            df['gain_after_fee'] = df['gain'] - df['fee_cost']
            df = df.set_index(df.Datetime)
            return len(df[df.gain_after_fee > 0]) / number_short, number_short

    def Number_of_trade(self) -> int:
        df = self.df.copy()
        df = df[df.input_pos != 0]
        return len(df.Position[df.Position != 0])

    def Profit_per_trade(self) -> float:
        df = self.df.copy()
        return df['total_gain_after_fee'].iloc[-1] / self.Number_of_trade()

    def Profit_after_fee(self) -> float:
        df = self.df.copy()
        return df.total_gain_after_fee.iloc[-1]

    def Profit_per_day(self) -> float:
        df = self.df.copy()
        return self.Profit_after_fee() / len(df.resample("1D").last().dropna())

    def Trading_per_day(self) -> float:
        df = self.df.copy()
        return self.Number_of_trade() / len(df.resample("1D").last().dropna())

    def Hitrate_per_day(self) -> float:
        df = self.df.copy()
        gain_daily = df.gain.resample("1D").sum().dropna()
        return len(gain_daily[gain_daily > 0]) / len(gain_daily[gain_daily != 0])

    def Hitrate_per_week(self) -> float:
        df = self.df.copy()
        gain_weekly = df.gain.resample("1W").sum().dropna()
        return len(gain_weekly[gain_weekly > 0]) / len(gain_weekly[gain_weekly != 0])

    def Hitrate_per_month(self) -> float:
        df = self.df.copy()
        gain_monthly = df.gain.resample("1M").sum().dropna()
        return len(gain_monthly[gain_monthly > 0]) / len(gain_monthly[gain_monthly != 0])

    def Return(self) -> float:
        df = self.df.copy()
        df['gain_per'] = df.gain_after_fee / df.Close
        pnl = df['gain_per'].cumsum()
        pnl_daily = pnl.resample("1D").last().dropna()
        r = pnl_daily.diff().dropna()
        return r.mean() * 365

    def Profit_per_year(self) -> float:
        df = self.df.copy()
        pnl = df['gain_after_fee'].cumsum()
        pnl_daily = pnl.resample("1D").last().dropna()
        r = pnl_daily.diff().dropna()
        return r.mean() * 365
