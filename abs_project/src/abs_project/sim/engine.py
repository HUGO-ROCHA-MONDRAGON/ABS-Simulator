# engine.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable
import math

# ============================================================
# ===              1️⃣  CORE DATA STRUCTURES                ===
# ============================================================

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
    waterfall_type: str = "pro_rata"

@dataclass
class Pool:
    balance: float
    coupon_annual: float
    amortization: str
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
    ttype: str
    notional: float
    price: float
    legal_final: int
    rating: str
    spread_bps: int
    outstanding: float = field(init=False)
    cash_interest: List[float] = field(default_factory=list)
    cash_principal: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.outstanding = self.notional


# ============================================================
# ===                   2️⃣  HELPERS                       ===
# ============================================================

def monthly_frac(day_count: str) -> float:
    return 1.0 / 12.0

def bisection(f: Callable[[float], float], lo: float, hi: float, tol=1e-8, maxit=100):
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
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
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# ============================================================
# ===                 3️⃣  WATERFALL ENGINE                ===
# ============================================================

class WaterfallEngine:
    """
    Cashflow engine for ABS waterfall simulation.
    Handles both pro-rata and sequential structures.
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
        self.defaults_history = []

        self.pool_int, self.pool_prin, self.residual_cash = [], [], []

    # ---------- Collateral ----------
    def _collateral_period_cf(self, t: int) -> Tuple[float, float, float]:
        if self.collateral_balance <= 1e-8:
            return 0.0, 0.0, 0.0

        dt = self.dt
        CPR_m = 1 - (1 - self.ass.CPR_annual) ** dt
        CDR_m = 1 - (1 - self.ass.CDR_annual) ** dt

        scheduled_prin = min(self.initial_pool_balance / self.periods, self.collateral_balance)
        defaults = self.collateral_balance * CDR_m
        prepays = max(self.collateral_balance - defaults, 0.0) * CPR_m
        interest = self.pool.coupon_annual * dt * self.collateral_balance
        total_prin = scheduled_prin + prepays + defaults
        self.collateral_balance = max(self.collateral_balance - total_prin, 0.0)

        self.defaults_history.append(defaults)

        recov = 0.0
        if t > self.ass.recovery_lag_months:
            lag_index = t - self.ass.recovery_lag_months - 1
            if 0 <= lag_index < len(self.defaults_history):
                recov = self.defaults_history[lag_index] * self.ass.recovery_rate

        fees = (self.ass.senior_fees_annual + self.ass.servicing_fee_annual) * dt * self.initial_pool_balance
        available_cf = interest + total_prin + recov - fees
        return available_cf, interest, total_prin

    # ---------- Liability ----------
    def _tranche_interest_due(self, tr: Tranche) -> float:
        if tr.ttype == "floating":
            coupon_annual = self.base_index_annual + tr.spread_bps / 10000.0
            return coupon_annual * self.dt * tr.outstanding
        return 0.0

    def _pay_interest_senior(self, cash_avail: float) -> float:
        seniors = [tr for tr in self.tranches if tr.ttype != "residual"]
        seniors.sort(key=lambda x: x.name)
        for tr in seniors:
            due = self._tranche_interest_due(tr)
            pay = min(cash_avail, due)
            tr.cash_interest.append(pay)
            cash_avail -= pay
        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_interest.append(0.0)
        return cash_avail

    def _pay_principal(self, cash_avail: float) -> float:
        wf_type = getattr(self.deal, "waterfall_type", "pro_rata").lower()
        seniors = [tr for tr in self.tranches if tr.ttype != "residual" and tr.outstanding > 1e-8]

        if wf_type == "sequential":
            seniors.sort(key=lambda x: x.name)
            for tr in seniors:
                pay = min(tr.outstanding, cash_avail)
                tr.cash_principal.append(pay)
                cash_avail -= pay
        else:  # pro-rata
            total_outs = sum(tr.outstanding for tr in seniors)
            for tr in seniors:
                share = tr.outstanding / total_outs
                alloc = min(cash_avail * share, tr.outstanding)
                tr.cash_principal.append(alloc)
            used = sum(tr.cash_principal[-1] for tr in seniors)
            cash_avail -= used

        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_principal.append(0.0)
        return cash_avail

    # ---------- Simulation ----------
    def simulate(self):
        for t in range(1, self.periods + 1):
            avail_cf, coll_int, coll_prin = self._collateral_period_cf(t)
            self.pool_int.append(coll_int)
            self.pool_prin.append(coll_prin)

            cash = self._pay_interest_senior(avail_cf)
            cash = self._pay_principal(cash)

            for tr in self.tranches:
                tr.outstanding = max(tr.outstanding - tr.cash_principal[-1], 0.0)
            self.residual_cash.append(max(cash, 0.0))

    # ---------- KPIs ----------
    def tranche_WAL_years(self, tr: Tranche) -> float:
        numer = sum(i * p for i, p in enumerate(tr.cash_principal, start=1))
        return 0.0 if tr.notional <= 0 else numer / tr.notional / 12.0

    def tranche_DM_bps(self, tr: Tranche) -> float:
        if tr.ttype == "residual":
            return float("nan")
        price_cash = tr.price / 100.0 * tr.notional
        dt, base = self.dt, self.base_index_annual
        cfs = [i + p for i, p in zip(tr.cash_interest, tr.cash_principal)]

        def pv(dm_annual):
            return sum(cf / ((1 + (base + dm_annual) * dt) ** k) for k, cf in enumerate(cfs, start=1))

        obj = lambda dm: pv(dm) - price_cash
        dm = bisection(obj, -0.05, 0.50)
        return float("nan") if dm is None else dm * 1e4

    def results_summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for tr in self.tranches:
            dm = float("nan") if tr.ttype == "residual" else self.tranche_DM_bps(tr)
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
