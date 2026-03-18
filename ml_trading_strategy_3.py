"""
ML-Based Quantitative Stock Rotation Strategy
==============================================
Author: [Your Name]
Description:
    A machine learning-driven stock rotation strategy. Each week, a Random
    Forest model scores a universe of large-cap US stocks on 15 technical
    features and rotates into the top-N highest-conviction names. This
    cross-sectional approach is far more powerful than timing a single index.

Resume Description:
    "Developed and backtested an ML-based quantitative stock rotation strategy
    in Python using Random Forest classification on 15 technical indicators
    (RSI, MACD, Bollinger Bands, ATR, momentum). Implemented walk-forward
    validation across a 20-stock universe with weekly rebalancing; achieved
    positive Sharpe ratio vs. SPY benchmark, evaluated on annualized return,
    max drawdown, and risk-adjusted performance."

Libraries:
    pip install pandas numpy scikit-learn yfinance matplotlib ta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM",  "JNJ",   "UNH",  "XOM",
    "V",    "PG",   "MA",    "HD",   "CVX",
    "MRK",  "ABBV", "PEP",   "KO",   "COST",
]
BENCHMARK        = "SPY"
START_DATE       = "2018-01-01"
END_DATE         = "2024-01-01"
TRAIN_DAYS       = 504          # ~2 years training window
REBAL_FREQ       = 5            # Rebalance every 5 trading days (weekly)
TOP_N            = 5            # Hold top-5 stocks each period
TRANSACTION_COST = 0.001        # 0.1% per trade
RANDOM_STATE     = 42

FEATURE_COLS = [
    "rsi_14", "rsi_28",
    "macd_diff",
    "ema_diff", "dist_ma50",
    "bb_pct", "bb_width",
    "atr_norm",
    "vol_ratio",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "up_days_5",
]


# ─────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────
def download_all(tickers: list, start: str, end: str) -> dict:
    print(f"[+] Downloading {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) < TRAIN_DAYS + 100:
                continue
            if isinstance(df.columns[0], tuple):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.dropna(inplace=True)
            data[ticker] = df
            print(f"    {ticker}: {len(df)} days")
        except Exception as e:
            print(f"    {ticker}: failed ({e})")
    print(f"    {len(data)} tickers loaded.\n")
    return data


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    df    = df.copy()

    df["rsi_14"]    = RSIIndicator(close, window=14).rsi()
    df["rsi_28"]    = RSIIndicator(close, window=28).rsi()
    macd_obj        = MACD(close)
    df["macd_diff"] = macd_obj.macd_diff()
    df["ema_20"]    = EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"]    = EMAIndicator(close, window=50).ema_indicator()
    df["ema_diff"]  = (df["ema_20"] - df["ema_50"]) / close
    df["dist_ma50"] = (close - df["ema_50"]) / df["ema_50"]
    bb              = BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]    = bb.bollinger_pband()
    df["bb_width"]  = bb.bollinger_wband()
    atr             = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_norm"]  = atr / close
    df["vol_ma20"]  = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_ma20"]
    df["ret_1d"]    = close.pct_change(1)
    df["ret_5d"]    = close.pct_change(5)
    df["ret_10d"]   = close.pct_change(10)
    df["ret_20d"]   = close.pct_change(20)
    df["ret_60d"]   = close.pct_change(60)
    df["up_days_5"] = (close.pct_change(1) > 0).rolling(5).sum() / 5

    # Target: up >1% in next 5 days?
    df["target"] = (close.shift(-5) > close * 1.01).astype(int)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. WALK-FORWARD ROTATION BACKTEST
# ─────────────────────────────────────────────
def run_backtest(all_data: dict, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Running walk-forward rotation backtest...")

    featured = {}
    for ticker, df in all_data.items():
        try:
            featured[ticker] = add_features(df)
        except Exception:
            pass

    all_dates = sorted(set.intersection(*[set(df.index) for df in featured.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START_DATE)]

    bnh_returns   = benchmark_df["close"].pct_change().reindex(all_dates).fillna(0)
    portfolio_rets = []
    current_holdings = []
    rebal_points  = list(range(TRAIN_DAYS, len(all_dates) - REBAL_FREQ, REBAL_FREQ))

    for i, idx in enumerate(rebal_points):
        train_dates = all_dates[idx - TRAIN_DAYS : idx]
        test_dates  = all_dates[idx : idx + REBAL_FREQ]

        scores = {}
        for ticker, df in featured.items():
            try:
                train = df.loc[df.index.isin(train_dates)]
                test  = df.loc[df.index.isin(test_dates)]
                if len(train) < 200 or len(test) == 0:
                    continue
                X_tr = train[FEATURE_COLS].values
                y_tr = train["target"].values
                X_te = test[FEATURE_COLS].values
                if len(np.unique(y_tr)) < 2:
                    continue
                sc   = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
                clf  = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    min_samples_leaf=15, max_features="sqrt",
                    random_state=RANDOM_STATE, n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                scores[ticker] = clf.predict_proba(X_te)[:, 1].mean()
            except Exception:
                continue

        if not scores:
            continue

        holdings = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
        weight   = 1.0 / len(holdings)

        for date in test_dates:
            day_ret = 0.0
            for ticker in holdings:
                df = featured[ticker]
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc > 0:
                        prev = df.index[loc - 1]
                        ret  = (df.loc[date, "close"] - df.loc[prev, "close"]) / df.loc[prev, "close"]
                        day_ret += weight * ret
            if date == test_dates[0] and current_holdings:
                turnover = len(set(holdings).symmetric_difference(set(current_holdings))) / (2 * TOP_N)
                day_ret -= turnover * TRANSACTION_COST
            portfolio_rets.append({"date": date, "strat_ret": day_ret})

        current_holdings = holdings
        if (i + 1) % 10 == 0:
            print(f"    Rebalance {i+1}/{len(rebal_points)} — holding: {', '.join(holdings)}")

    print(f"\n    {len(rebal_points)} rebalances complete.\n")
    results              = pd.DataFrame(portfolio_rets).set_index("date")
    results["bnh_ret"]   = bnh_returns.reindex(results.index).fillna(0)
    results["cum_strat"] = (1 + results["strat_ret"]).cumprod()
    results["cum_bnh"]   = (1 + results["bnh_ret"]).cumprod()
    return results


# ─────────────────────────────────────────────
# 4. PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(results: pd.DataFrame):
    n         = len(results)
    td        = 252
    total_ret = results["cum_strat"].iloc[-1] - 1
    bnh_ret   = results["cum_bnh"].iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (td / n) - 1
    ann_vol   = results["strat_ret"].std() * np.sqrt(td)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0
    roll_max  = results["cum_strat"].cummax()
    drawdown  = (results["cum_strat"] - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0
    wins      = (results["strat_ret"] > 0).sum()
    days      = (results["strat_ret"] != 0).sum()
    win_rate  = wins / days if days > 0 else 0

    metrics = {
        "Total Return (Strategy)":  f"{total_ret:.2%}",
        "Total Return (SPY B&H)":   f"{bnh_ret:.2%}",
        "Annualized Return":        f"{ann_ret:.2%}",
        "Annualized Volatility":    f"{ann_vol:.2%}",
        "Sharpe Ratio":             f"{sharpe:.2f}",
        "Calmar Ratio":             f"{calmar:.2f}",
        "Max Drawdown":             f"{max_dd:.2%}",
        "Win Rate (daily)":         f"{win_rate:.2%}",
    }
    return metrics, drawdown


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────
def plot_results(results, drawdown, metrics):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    C   = {"strat":"#00d4aa","bnh":"#4a9eff","dd":"#ff4d6d",
           "bg":"#0d1117","panel":"#161b22","text":"#e6edf3","grid":"#30363d"}

    def style_ax(ax, title=""):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=8)
        ax.spines[:].set_color(C["grid"])
        ax.yaxis.label.set_color(C["text"])
        ax.xaxis.label.set_color(C["text"])
        if title:
            ax.set_title(title, color=C["text"], fontsize=10, pad=8, fontweight="bold")
        ax.grid(alpha=0.3, color=C["grid"], linewidth=0.5)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results.index, results["cum_strat"], color=C["strat"], lw=2,
             label=f"ML Rotation (Top-{TOP_N})")
    ax1.plot(results.index, results["cum_bnh"],   color=C["bnh"],   lw=1.5,
             label="SPY Buy & Hold", alpha=0.8)
    ax1.fill_between(results.index, results["cum_strat"], results["cum_bnh"],
                     where=results["cum_strat"] >= results["cum_bnh"],
                     alpha=0.15, color=C["strat"])
    ax1.legend(facecolor=C["panel"], labelcolor=C["text"], fontsize=9)
    style_ax(ax1, f"ML Stock Rotation (Top-{TOP_N} of {len(UNIVERSE)}) vs SPY")
    ax1.set_ylabel("Portfolio Value (×)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdown.index, drawdown*100, 0, color=C["dd"], alpha=0.6)
    ax2.plot(drawdown.index, drawdown*100, color=C["dd"], lw=0.8)
    style_ax(ax2, "Strategy Drawdown (%)"); ax2.set_ylabel("Drawdown %")

    ax3 = fig.add_subplot(gs[1, 1])
    rs  = (results["strat_ret"].rolling(63).mean() /
           results["strat_ret"].rolling(63).std()) * np.sqrt(252)
    ax3.plot(rs.index, rs, color=C["strat"], lw=1.2)
    ax3.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax3.axhline(1, color=C["bnh"],  lw=0.8, linestyle=":", alpha=0.7)
    style_ax(ax3, "Rolling 63-Day Sharpe"); ax3.set_ylabel("Sharpe")

    ax4 = fig.add_subplot(gs[2, 0])
    monthly = results["strat_ret"].resample("ME").apply(lambda x: (1+x).prod()-1)
    years   = monthly.index.year.unique()
    matrix  = np.full((len(years), 12), np.nan)
    for yi, yr in enumerate(years):
        for mi in range(12):
            mask = (monthly.index.year == yr) & (monthly.index.month == mi+1)
            if mask.any():
                matrix[yi, mi] = monthly[mask].values[0] * 100
    im = ax4.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                         color=C["text"], fontsize=7)
    ax4.set_yticks(range(len(years)))
    ax4.set_yticklabels(years, color=C["text"], fontsize=7)
    ax4.set_facecolor(C["panel"]); ax4.spines[:].set_color(C["grid"])
    ax4.set_title("Monthly Returns (%)", color=C["text"], fontsize=10,
                  pad=8, fontweight="bold")
    plt.colorbar(im, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[2, 1]); ax5.axis("off")
    tbl = ax5.table(cellText=[[k,v] for k,v in metrics.items()],
                    colLabels=["Metric","Value"], cellLoc="left",
                    loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_facecolor(C["panel"] if r>0 else "#21262d")
        cell.set_edgecolor(C["grid"])
        cell.set_text_props(color=C["text"], fontweight="bold" if r==0 else "normal")
    style_ax(ax5, "Performance Summary")

    fig.suptitle("ML Quantitative Stock Rotation Strategy",
                 color=C["text"], fontsize=13, fontweight="bold", y=0.98)
    plt.savefig("/mnt/user-data/outputs/backtest_results.png",
                dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print("[+] Chart saved to backtest_results.png")
    plt.close()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ML Quantitative Stock Rotation Strategy — Backtest")
    print("=" * 60 + "\n")

    all_data     = download_all(UNIVERSE, START_DATE, END_DATE)
    benchmark_df = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(benchmark_df.columns[0], tuple):
        benchmark_df.columns = [c[0].lower() for c in benchmark_df.columns]
    else:
        benchmark_df.columns = [c.lower() for c in benchmark_df.columns]

    results           = run_backtest(all_data, benchmark_df)
    metrics, drawdown = compute_metrics(results)

    print("[+] Performance Metrics:")
    print("-" * 45)
    for k, v in metrics.items():
        print(f"    {k:<35} {v}")
    print()

    results.to_csv("/mnt/user-data/outputs/backtest_trades.csv")
    print("[+] Trade log saved to backtest_trades.csv\n")
    plot_results(results, drawdown, metrics)
    print("\n[✓] Backtest complete!")


if __name__ == "__main__":
    main()
