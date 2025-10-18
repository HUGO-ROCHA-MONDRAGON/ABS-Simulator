# abs_simulator.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

# ---------------------------
# Config & core data objects
# ---------------------------

@dataclass
@dataclass
class DealMeta:
    deal_name: str
    issuer: str
    start_date: str
    frequency: str
    periods: int
    currency: str
    benchmark_curve: str
    day_count: str
    waterfall_type: str = "pro_rata"   # 👈 added default

@dataclass
class Pool:
    balance: float
    coupon_annual: float          # collateral coupon (annual)
    amortization: str             # "pass_through"
    seasoning_months: int
    weighted_avg_maturity: int
    weighted_avg_rate: float

@dataclass
class Assumptions:
    CPR_annual: float = 0.0
    CDR_annual: float = 0.0
    recovery_rate: float = 0.0
    recovery_lag_months: int = 0
    delinquency_rate: float = 0.0
    servicing_fee_annual: float = 0.0
    senior_fees_annual: float = 0.0
    scenario_name: str = "Base"

@dataclass
class Tranche:
    name: str
    ttype: str                    # "floating" or "residual"
    notional: float
    price: float                  # clean price (e.g., 100.09)
    legal_final: int              # months
    rating: str
    spread_bps: int               # for floating; residual will ignore
    outstanding: float = field(init=False)
    cash_interest: List[float] = field(default_factory=list)
    cash_principal: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.outstanding = self.notional

# ---------------------------
# Helper functions
# ---------------------------

def monthly_frac(day_count: str) -> float:
    # Keep it simple: monthly accrual fraction ≈ 1/12
    return 1.0 / 12.0

def bisection(f, lo, hi, tol=1e-8, maxit=100):
    flo = f(lo)
    fhi = f(hi)
    if flo * fhi > 0:
        # fallback: widen once
        for _ in range(5):
            lo -= (hi - lo)
            flo = f(lo)
            if flo * fhi <= 0:
                break
        else:
            return None
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

# ---------------------------
# Engine
# ---------------------------

class WaterfallEngine:
    """
    Engine v1.1:
      - Collateral interest: pool_rate/12 * current_pool_balance
      - Scheduled principal: level pass-through (initial_balance / periods)
      - CPR/CDR/prepays/defaults + recoveries with lag
      - Fees (servicing + senior) deducted at collateral level
      - Waterfall:
          Interest: pay senior A -> B -> C (no shortfall carry here for v1)
          Principal: pro-rata across A/B/C by outstanding share
          Residual: all remaining cash (if any) to Equity
    """

    def __init__(self, deal: DealMeta, pool: Pool, tranches: List[Tranche],
                 assumptions: Assumptions, base_index_annual: float = 0.0):
        self.deal = deal
        self.pool = pool
        self.tranches = tranches
        self.ass = assumptions
        self.base_index_annual = base_index_annual
        self.dt = monthly_frac(deal.day_count)
        self.periods = deal.periods
        self.collateral_balance = pool.balance
        self.initial_pool_balance = pool.balance
        self.defaults_history = []  # to compute lagged recoveries

        # Storage
        self.pool_int = []
        self.pool_prin = []
        self.residual_cash = []

    # -------- Collateral schedule --------
    def _collateral_period_cf(self, t: int) -> Tuple[float, float, float]:
        if self.collateral_balance <= 1e-8:
            return 0.0, 0.0, 0.0

        dt = self.dt
        CPR_m = 1 - (1 - self.ass.CPR_annual) ** dt     # annual -> monthly
        CDR_m = 1 - (1 - self.ass.CDR_annual) ** dt

        scheduled_prin = min(self.initial_pool_balance / self.periods, self.collateral_balance)

        # Defaults + Prepayments
        defaults = self.collateral_balance * CDR_m
        prepays  = max(self.collateral_balance - defaults, 0.0) * CPR_m

        # Interest on current balance (simple)
        interest = self.pool.coupon_annual * dt * self.collateral_balance

        # Reduce collateral balance
        total_prin = scheduled_prin + prepays + defaults
        self.collateral_balance = max(self.collateral_balance - total_prin, 0.0)

        # Defaults history for lagged recoveries
        self.defaults_history.append(defaults)

        # Recoveries (if lag applies)
        recov = 0.0
        if t > self.ass.recovery_lag_months:
            lag_index = t - self.ass.recovery_lag_months - 1
            if 0 <= lag_index < len(self.defaults_history):
                recov = self.defaults_history[lag_index] * self.ass.recovery_rate

        # Fees (senior + servicing)
        fees = (self.ass.senior_fees_annual + self.ass.servicing_fee_annual) * dt * self.initial_pool_balance

        # Net available CF to liabilities
        available_cf = interest + total_prin + recov - fees
        return available_cf, interest, total_prin

    # -------- Liability calculations --------
    def _tranche_interest_due(self, tr: Tranche) -> float:
        if tr.ttype == "floating":
            coupon_annual = self.base_index_annual + tr.spread_bps / 10000.0
            return coupon_annual * self.dt * tr.outstanding
        elif tr.ttype == "residual":
            return 0.0
        else:
            # extend later for fixed-rate
            return 0.0

    def _pay_interest_senior(self, cash_avail: float) -> float:
        # Senior order: A -> B -> C; Equity has no stated interest
        seniors = [tr for tr in self.tranches if tr.ttype != "residual"]
        seniors.sort(key=lambda x: x.name)  # assumes A < B < C ...
        for tr in seniors:
            due = self._tranche_interest_due(tr)
            pay = min(cash_avail, due)
            tr.cash_interest.append(pay)
            cash_avail -= pay
        # residual tranche logs 0 interest
        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_interest.append(0.0)
        return cash_avail

    def _pay_principal_prorata(self, cash_avail: float) -> float:
        seniors = [tr for tr in self.tranches if tr.ttype != "residual" and tr.outstanding > 1e-8]
        total_outs = sum(tr.outstanding for tr in seniors)
        if total_outs <= 1e-8 or cash_avail <= 1e-8:
            for tr in self.tranches:
                tr.cash_principal.append(0.0)
            return cash_avail

        for tr in seniors:
            share = tr.outstanding / total_outs
            alloc = min(cash_avail * share, tr.outstanding)
            tr.cash_principal.append(alloc)

        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_principal.append(0.0)

        used = sum(tr.cash_principal[-1] for tr in seniors)
        return cash_avail - used
    
    
    def _pay_principal_sequential(self, cash_avail: float) -> float:
        """Sequential principal: pay A → B → C fully before next tranche."""
        seniors = [tr for tr in self.tranches if tr.ttype != "residual"]
        seniors.sort(key=lambda x: x.name)
        for tr in seniors:
            if cash_avail <= 0:
                tr.cash_principal.append(0.0)
                continue
            pay = min(tr.outstanding, cash_avail)
            tr.cash_principal.append(pay)
            cash_avail -= pay
        # residual tranche
        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_principal.append(0.0)
        return cash_avail


    def simulate(self):
        for t in range(1, self.periods + 1):
            # 1) Collateral cash (already net of fees and including recoveries)
            available_cf, coll_int, coll_prin = self._collateral_period_cf(t)
            self.pool_int.append(coll_int)
            self.pool_prin.append(coll_prin)
            available_cash = available_cf  # <-- do NOT subtract principal again

            # 2) Liabilities – Interest (senior)
            available_cash = self._pay_interest_senior(available_cash)

            # 3) Liabilities – Principal (pro-rata among debt tranches)
            # 3) Liabilities – Principal based on waterfall type
            if getattr(self.deal, "waterfall_type", "pro_rata").lower() == "sequential":
                available_cash = self._pay_principal_sequential(available_cash)
            else:
                available_cash = self._pay_principal_prorata(available_cash)


            # 4) Update tranche notionals
            for tr in self.tranches:
                tr.outstanding = max(tr.outstanding - tr.cash_principal[-1], 0.0)

            # 5) Residual (equity)
            self.residual_cash.append(max(available_cash, 0.0))

    # ---------------------------
    # KPIs
    # ---------------------------

    def tranche_WAL_years(self, tr: Tranche) -> float:
        # WAL = sum(t * principal_t) / notional / 12
        numer = 0.0
        for i, p in enumerate(tr.cash_principal, start=1):
            numer += i * p
        if tr.notional <= 0:
            return 0.0
        return numer / tr.notional / 12.0

    def tranche_DM_bps(self, tr: Tranche) -> float:
        """
        For floating notes, solve for DM such that:
           Price (as clean %) * notional = PV( CFs discounted by (index + DM) )
        Simplification: flat index (base_index_annual), monthly comp with dt = 1/12.
        """
        if tr.ttype == "residual":
            return float('nan')

        price_cash = tr.price / 100.0 * tr.notional
        dt = self.dt
        base = self.base_index_annual

        # Liability CFs actually paid (interest + principal)
        cfs = [tr.cash_interest[i] + tr.cash_principal[i] for i in range(len(tr.cash_interest))]

        def pv_given_dm(dm_annual: float) -> float:
            pv = 0.0
            r = base + dm_annual
            for k, cf in enumerate(cfs, start=1):
                df = 1.0 / ((1.0 + r * dt) ** k)
                pv += cf * df
            return pv

        def objective(dm):
            return pv_given_dm(dm) - price_cash

        # DM search over [-5%, +50%] annual to be safe
        dm = bisection(objective, -0.05, 0.50, tol=1e-8, maxit=200)
        return float('nan') if dm is None else dm * 1e4  # bps

    def results_summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for tr in self.tranches:
            dm = float('nan') if tr.ttype == "residual" else self.tranche_DM_bps(tr)
            out[tr.name] = {
                "notional_start": tr.notional,
                "notional_end": tr.outstanding,
                "WAL_years": self.tranche_WAL_years(tr),
                "DM_bps": dm,
                "int_total": sum(tr.cash_interest),
                "prin_total": sum(tr.cash_principal),
            }
        out["Equity_residual"] = {"total_residual_cash": sum(self.residual_cash)}
        return out

# ============================
# ===  PLOT IMPROVED CFs  ===
# ============================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use("seaborn-v0_8-whitegrid")  # cleaner base

def build_waterfall_df(engine):
    df = pd.DataFrame()
    for tr in engine.tranches:
        total_cf = [i + p for i, p in zip(tr.cash_interest, tr.cash_principal)]
        df[tr.name] = total_cf
    if hasattr(engine, "residual_cash") and len(engine.residual_cash) > 0:
        df["Equity_residual"] = engine.residual_cash
    df.index = range(1, len(df) + 1)
    df.index.name = "Month"
    return df

# --- Improved stacked waterfall (looks like Intex) ---
def plot_waterfall(engine):
    df = build_waterfall_df(engine)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax.stackplot(df.index, [df[c] for c in df.columns],
                 labels=df.columns, alpha=0.9, colors=colors[:len(df.columns)])
    ax.set_title(f"Cashflow Waterfall — {engine.deal.deal_name}", fontsize=16, weight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Monthly Cash Distribution (€)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()

# --- Improved line plot (one curve per tranche) ---
def plot_waterfall_lines(engine):
    df = build_waterfall_df(engine)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col, linewidth=2, color=colors[i % len(colors)])
    ax.set_title(f"Tranche Cashflows — {engine.deal.deal_name}", fontsize=16, weight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Monthly Cash (€)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------
# Loader + Example main
# ---------------------------

def load_from_dict(d: Dict) -> Tuple[DealMeta, Pool, List[Tranche], Assumptions]:
    wf_type = d.get("structure", {}).get("waterfall", {}).get("type", "pro_rata")

    deal = DealMeta(
        deal_name=d["deal"]["deal_name"],
        issuer=d["deal"]["issuer"],
        start_date=d["deal"]["start_date"],
        frequency=d["deal"]["frequency"],
        periods=int(d["deal"]["periods"]),
        currency=d["deal"]["currency"],
        benchmark_curve=d["deal"]["benchmark_curve"],
        day_count=d["deal"]["day_count"],
        waterfall_type=wf_type,   # 👈 add here
    )

    pool = Pool(
        balance=float(d["pool"]["balance"]),
        coupon_annual=float(d["pool"]["coupon_annual"]),
        amortization=d["pool"]["amortization"],
        seasoning_months=int(d["pool"]["seasoning_months"]),
        weighted_avg_maturity=int(d["pool"]["weighted_avg_maturity"]),
        weighted_avg_rate=float(d["pool"]["weighted_avg_rate"]),
    )

    tranches = [
        Tranche(
            name=t["name"],
            ttype=t["type"],
            notional=float(t["notional"]),
            price=float(t["price"]),
            legal_final=int(t["legal_final"]),
            rating=t["rating"],
            spread_bps=int(t.get("spread_bps", 0)),
        )
        for t in d["structure"]["tranches"]
    ]

    assumptions = Assumptions(**d.get("assumptions", {}))
    return deal, pool, tranches, assumptions
