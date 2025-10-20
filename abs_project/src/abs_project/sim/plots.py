# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use("seaborn-v0_8-whitegrid")

import pandas as pd
import numpy as np

def build_waterfall_df(engine):
    """
    Build a DataFrame of all tranche cashflows.
    Handles different tranche lengths by padding shorter ones with zeros.
    """
    data = {}
    max_len = 0

    # find maximum series length
    for tr in engine.tranches:
        total_cf = [i + p for i, p in zip(tr.cash_interest, tr.cash_principal)]
        data[tr.name] = total_cf
        max_len = max(max_len, len(total_cf))

    # add residual equity if present
    if getattr(engine, "residual_cash", []):
        data["Equity_residual"] = engine.residual_cash
        max_len = max(max_len, len(engine.residual_cash))

    # pad all series to same length
    for k, v in data.items():
        if len(v) < max_len:
            data[k] = v + [0.0] * (max_len - len(v))

    df = pd.DataFrame(data)
    df.index = range(1, max_len + 1)
    df.index.name = "Month"
    return df


def plot_waterfall(engine):
    df = build_waterfall_df(engine)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(df.index, [df[c] for c in df.columns], labels=df.columns, alpha=0.9)
    ax.set_title(f"Cashflow Waterfall — {engine.deal.deal_name}", fontsize=15, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash Distribution (€)")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.show()



def plot_tranche_balances(engine):
    """
    Plot cumulative principal repayments for each tranche.
    Handles different tranche lengths by padding shorter ones with zeros.
    """
    # --- 1️⃣ Prepare data ---
    data = {}
    max_len = 0
    for tr in engine.tranches:
        cumul_prin = list(np.cumsum(tr.cash_principal))
        data[tr.name] = cumul_prin
        max_len = max(max_len, len(cumul_prin))

    # --- 2️⃣ Pad all to same length ---
    for k, v in data.items():
        if len(v) < max_len:
            data[k] = v + [v[-1]] * (max_len - len(v))  # keep flat line after last repayment

    # --- 3️⃣ Build DataFrame ---
    df = pd.DataFrame(data)
    df.index = range(1, max_len + 1)
    df.index.name = "Month"

    # --- 4️⃣ Plot ---
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=2)
    plt.title("Cumulative Principal Repayments per Tranche", fontsize=14, weight="bold")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Principal (€)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

