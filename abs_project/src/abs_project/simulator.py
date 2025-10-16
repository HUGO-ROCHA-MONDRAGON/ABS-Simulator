# abs_simulator.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

# ---------------------------
# Config & core data objects
# ---------------------------

@dataclass
class DealMeta:
    deal_name: str
    issuer: str
    start_date: str       # ISO string YYYY-MM-DD
    frequency: str        # "Monthly"
    periods: int          # number of months
    currency: str
    benchmark_curve: str  # e.g., "EURIBOR_3M"
    day_count: str        # "ACT/360" (we use 1/12 simplification for now)

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
    Minimal engine:
      - Collateral interest: pool_rate/12 * current_pool_balance
      - Scheduled principal: level pass-through (initial_balance / periods)
      - Waterfall:
          Interest: pay senior A -> B -> C (no shortfall carry here for v1)
          Principal: pro-rata across A/B/C by outstanding share
          Residual: all remaining cash (if any) to Equity
      - No fees, no defaults/prepays (v1). Easy to add later.
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
        self.defaults_history = []  # <---- ADD THIS LINE
        # Storage
        self.pool_int = []
        self.pool_prin = []
        self.residual_cash = []

    # -------- Collateral schedule (v1: level principal, no losses/prepays) --------
    def _collateral_period_cf(self, t: int) -> Tuple[float, float, float]:
        if self.collateral_balance <= 1e-8:
            return 0.0, 0.0, 0.0

        dt = self.dt
        CPR_m = 1 - (1 - self.ass.CPR_annual) ** dt     # convert annual → monthly
        CDR_m = 1 - (1 - self.ass.CDR_annual) ** dt

        scheduled_prin = min(self.initial_pool_balance / self.periods, self.collateral_balance)

        # Defaults + Prepayments
        defaults = self.collateral_balance * CDR_m
        prepays = (self.collateral_balance - defaults) * CPR_m

        # Interest earned on surviving loans
        interest = self.pool.coupon_annual * dt * self.collateral_balance

        # Reduce collateral balance
        total_prin = scheduled_prin + prepays + defaults
        self.collateral_balance -= total_prin

        # Append defaults to history
        self.defaults_history.append(defaults)

        # Recoveries (if lag applies)
        recov = 0.0
        if t > self.ass.recovery_lag_months:
            lag_index = t - self.ass.recovery_lag_months - 1
            if lag_index < len(self.defaults_history):
                recov = self.defaults_history[lag_index] * self.ass.recovery_rate

        # Fees (senior + servicing)
        fees = (self.ass.senior_fees_annual + self.ass.servicing_fee_annual) * dt * self.initial_pool_balance

        # Net available CF
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
        # sort by name A,B,C (simple; adapt if you add explicit seniority ranks)
        seniors.sort(key=lambda x: x.name)
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
            # append zeros
            for tr in self.tranches:
                tr.cash_principal.append(0.0)
            return cash_avail

        for tr in seniors:
            share = tr.outstanding / total_outs
            alloc = min(cash_avail * share, tr.outstanding)
            tr.cash_principal.append(alloc)
        # residual tranche gets no principal
        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_principal.append(0.0)

        used = sum(tr.cash_principal[-1] for tr in seniors)
        return cash_avail - used

    def simulate(self):
        for t in range(1, self.periods + 1):
            # 1) Collateral cash
            coll_int, coll_prin = self._collateral_period_cf(t)
            self.pool_int.append(coll_int)
            self.pool_prin.append(coll_prin)
            self.collateral_balance -= coll_prin
            available_cash = coll_int + coll_prin

            # 2) Liabilities – Interest (senior)
            available_cash = self._pay_interest_senior(available_cash)

            # 3) Liabilities – Principal (pro-rata among debt tranches)
            available_cash = self._pay_principal_prorata(available_cash)

            # 4) Update tranche notionals
            for tr in self.tranches:
                tr.outstanding -= tr.cash_principal[-1]
                tr.outstanding = max(tr.outstanding, 0.0)

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

        # Build projected liability CFs (interest + principal actually paid)
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
            if tr.ttype == "residual":
                dm = float('nan')
            else:
                dm = self.tranche_DM_bps(tr)
            out[tr.name] = {
                "notional_start": tr.notional,
                "notional_end": tr.outstanding,
                "WAL_years": self.tranche_WAL_years(tr),
                "DM_bps": dm,
                "int_total": sum(tr.cash_interest),
                "prin_total": sum(tr.cash_principal),
            }
        out["Equity_residual"] = {
            "total_residual_cash": sum(self.residual_cash)
        }
        return out

# ---------------------------
# Loader from your YAML-like dict
# ---------------------------

def load_from_dict(d: Dict) -> Tuple[DealMeta, Pool, List[Tranche], Assumptions]:
    deal = DealMeta(
        deal_name=d["deal"]["deal_name"],
        issuer=d["deal"]["issuer"],
        start_date=d["deal"]["start_date"],
        frequency=d["deal"]["frequency"],
        periods=int(d["deal"]["periods"]),
        currency=d["deal"]["currency"],
        benchmark_curve=d["deal"]["benchmark_curve"],
        day_count=d["deal"]["day_count"],
    )
    pool = Pool(
        balance=float(d["pool"]["balance"]),
        coupon_annual=float(d["pool"]["coupon_annual"]),
        amortization=d["pool"]["amortization"],
        seasoning_months=int(d["pool"]["seasoning_months"]),
        weighted_avg_maturity=int(d["pool"]["weighted_avg_maturity"]),
        weighted_avg_rate=float(d["pool"]["weighted_avg_rate"]),
    )
    tranches = []
    for t in d["structure"]["tranches"]:
        tranches.append(
            Tranche(
                name=t["name"],
                ttype=t["type"],
                notional=float(t["notional"]),
                price=float(t["price"]),
                legal_final=int(t["legal_final"]),
                rating=t["rating"],
                spread_bps=int(t.get("spread_bps", 0)),
            )
        )
    assumptions = Assumptions(**d.get("assumptions", {}))
    return deal, pool, tranches, assumptions




# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Paste your dict here (converted from YAML). Example:
    data = {
        "deal": {
            "deal_name": "AUTOFR 2022-1",
            "issuer": "BNP Paribas Personal Finance",
            "start_date": "2025-01-01",
            "frequency": "Monthly",
            "periods": 72,
            "currency": "EUR",
            "benchmark_curve": "EURIBOR_3M",
            "day_count": "ACT/360",
        },
        "pool": {
            "balance": 142000000,
            "coupon_annual": 0.0341,
            "amortization": "pass_through",
            "seasoning_months": 10,
            "weighted_avg_maturity": 71,
            "weighted_avg_rate": 0.0351,
        },
        "structure": {
            "waterfall": {"type": "pro_rata"},
            "tranches": [
                {"name": "A", "type": "floating", "notional": 112062125.65, "price": 100.09, "legal_final": 48, "rating": "AAA", "spread_bps": 193},
                {"name": "B", "type": "floating", "notional": 10804048.51, "price": 97.91, "legal_final": 48, "rating": "A", "spread_bps": 198},
                {"name": "C", "type": "floating", "notional": 5312690.76, "price": 99.89, "legal_final": 72, "rating": "BB+", "spread_bps": 147},
                {"name": "Equity", "type": "residual", "notional": 13821135.08, "price": 98.18, "legal_final": 72, "rating": "NR"},
            ],
        },
    }

deal, pool, tranches, ass = load_from_dict(data)
engine = WaterfallEngine(deal, pool, tranches, ass, base_index_annual=0.026)
engine.simulate()
summary = engine.results_summary()


# ============================
# ===  PLOT WATERFALL CFs  ===
# ============================

import pandas as pd
import matplotlib.pyplot as plt

# --- Build DataFrame of monthly CFs per tranche ---
def build_waterfall_df(engine):
    df = pd.DataFrame()
    for tr in engine.tranches:
        total_cf = [i + p for i, p in zip(tr.cash_interest, tr.cash_principal)]
        df[tr.name] = total_cf
    # add equity residual if exists
    if hasattr(engine, "residual_cash") and len(engine.residual_cash) > 0:
        df["Equity_residual"] = engine.residual_cash
    df.index = range(1, len(df) + 1)
    df.index.name = "Month"
    return df

# --- Generate the plot ---
def plot_waterfall(engine):
    df = build_waterfall_df(engine)
    plt.figure(figsize=(10, 6))
    plt.stackplot(df.index, [df[c] for c in df.columns],
                  labels=df.columns, alpha=0.8)
    plt.title(f"Cashflow Waterfall — {engine.deal.deal_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Monthly Cash Distribution (€)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Run ---
plot_waterfall(engine)

def plot_waterfall_lines(engine):
    df = build_waterfall_df(engine)
    plt.figure(figsize=(10,6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=1.8)
    plt.title(f"Tranche Cashflows — {engine.deal.deal_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Monthly Cash (€)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_waterfall_lines(engine)
