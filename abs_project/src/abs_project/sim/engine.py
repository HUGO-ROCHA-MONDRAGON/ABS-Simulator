# engine.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional

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
    waterfall_type: str = "pro_rata"   # "pro_rata" or "sequential"


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

    # 🔽 NEW (optional) — to make pro-rata more realistic
    stepdown_month: Optional[int] = None     # e.g., 24 (switch after month 24)
    oc_trigger: Optional[float] = None       # e.g., 1.10 means 110% OC ratio required


@dataclass
class Tranche:
    name: str
    ttype: str               # "floating" or "residual"
    notional: float
    price: float             # price as % of par (e.g., 100)
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
    # simple ACT/12 style for now
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


# ---- Yield-curve math (forwards from zero) -----------------

def _df_from_zero_annual(z_annual: float, t_years: float) -> float:
    """Discount factor from an annual simple zero rate using monthly comp."""
    return 1.0 / ((1.0 + z_annual / 12.0) ** (t_years * 12.0))


def _fwd_1m_from_zero(curve_fn: Callable[[float], float], t_prev: float, t_curr: float) -> float:
    """
    One-month simple forward rate implied by zero curve between (t_prev, t_curr).
    Returns annualized simple rate (e.g., 0.041 = 4.1%).
    """
    z_prev = float(curve_fn(max(t_prev, 1e-9)))
    z_curr = float(curve_fn(max(t_curr, 1e-9)))
    DF_prev = _df_from_zero_annual(z_prev, t_prev)
    DF_curr = _df_from_zero_annual(z_curr, t_curr)
    # forward over one month, annualized simple
    return 12.0 * (DF_prev / DF_curr - 1.0)


# ============================================================
# ===                 3️⃣  WATERFALL ENGINE                ===
# ============================================================

class WaterfallEngine:
    """
    Cashflow engine for ABS waterfall simulation.
    - Supports pro-rata and sequential structures.
    - Floating coupons accrue at period FORWARD rates from a zero curve.
    - Discounting uses zero(yf) + DM with monthly compounding.
    - Optional step-down triggers (month and/or OC ratio) to switch pro-rata → sequential.
    """

    def __init__(
        self,
        deal: DealMeta,
        pool: Pool,
        tranches: List[Tranche],
        assumptions: Assumptions,
        base_index_annual: float = 0.0,
        curve_fn: Optional[Callable[[float], float]] = None,
    ):
        self.deal = deal
        self.pool = pool
        self.tranches = tranches
        self.ass = assumptions

        self.base_index_annual = base_index_annual  # fallback if no curve_fn
        self.curve_fn = curve_fn

        self.dt = monthly_frac(deal.day_count)
        self.periods = deal.periods
        self.collateral_balance = pool.balance
        self.initial_pool_balance = pool.balance
        self.defaults_history: List[float] = []

        self.pool_int, self.pool_prin, self.residual_cash = [], [], []

    # ---------- Collateral (assets) ----------
    def _collateral_period_cf(self, t: int) -> Tuple[float, float, float]:
        if self.collateral_balance <= 1e-8:
            return 0.0, 0.0, 0.0

        dt = self.dt
        CPR_m = 1 - (1 - self.ass.CPR_annual) ** dt
        CDR_m = 1 - (1 - self.ass.CDR_annual) ** dt

        # Simple equal-scheduled amortization of pool principal + prepay + default
        scheduled_prin = min(self.initial_pool_balance / self.periods, self.collateral_balance)
        defaults = self.collateral_balance * CDR_m
        prepays = max(self.collateral_balance - defaults, 0.0) * CPR_m

        # Asset interest at the pool coupon (fixed WAC)
        interest = self.pool.coupon_annual * dt * self.collateral_balance

        total_prin = scheduled_prin + prepays + defaults
        self.collateral_balance = max(self.collateral_balance - total_prin, 0.0)
        self.defaults_history.append(defaults)

        # Recoveries with lag
        recov = 0.0
        if t > self.ass.recovery_lag_months:
            lag_index = t - self.ass.recovery_lag_months - 1
            if 0 <= lag_index < len(self.defaults_history):
                recov = self.defaults_history[lag_index] * self.ass.recovery_rate

        # Fees (simple)
        fees = (self.ass.senior_fees_annual + self.ass.servicing_fee_annual) * dt * self.initial_pool_balance

        available_cf = interest + total_prin + recov - fees
        return available_cf, interest, total_prin

    # ---------- Liability (notes) ----------
    def _base_rate_for_period(self, t_month: int) -> float:
        """Annualized base rate applicable to the coupon accrual for [t-1, t]."""
        if self.curve_fn:
            t_prev = (t_month - 1) / 12.0
            t_curr = t_month / 12.0
            return _fwd_1m_from_zero(self.curve_fn, t_prev, t_curr)  # annualized simple
        return self.base_index_annual

    def _tranche_interest_due(self, tr: Tranche, t_month: int) -> float:
        if tr.ttype != "floating":
            return 0.0
        base_rate = self._base_rate_for_period(t_month)  # annualized simple
        coupon_annual = base_rate + tr.spread_bps / 1e4
        return coupon_annual * self.dt * tr.outstanding

    def _pay_interest_senior(self, cash_avail: float, t_month: int) -> float:
        seniors = [tr for tr in self.tranches if tr.ttype != "residual"]
        seniors.sort(key=lambda x: x.name)  # simple seniority by name
        for tr in seniors:
            due = self._tranche_interest_due(tr, t_month)
            pay = min(cash_avail, due)
            tr.cash_interest.append(pay)
            cash_avail -= pay
        for tr in self.tranches:
            if tr.ttype == "residual":
                tr.cash_interest.append(0.0)
        return cash_avail

    def _current_oc_ratio(self) -> float:
        """OC ratio ≈ collateral / total notes outstanding (excl. residual)."""
        notes_out = sum(tr.outstanding for tr in self.tranches if tr.ttype != "residual")
        if notes_out <= 1e-12:
            return float("inf")
        return self.collateral_balance / notes_out

    def _effective_waterfall_type(self, t_month: int) -> str:
        """Apply optional step-down triggers to switch pro-rata → sequential."""
        wf_type = getattr(self.deal, "waterfall_type", "pro_rata").lower()
        if wf_type != "pro_rata":
            return wf_type

        stepdown_hit = False
        if self.ass.stepdown_month is not None and t_month >= self.ass.stepdown_month:
            stepdown_hit = True
        if self.ass.oc_trigger is not None and self._current_oc_ratio() >= self.ass.oc_trigger:
            stepdown_hit = True

        return "sequential" if stepdown_hit else "pro_rata"

    def _pay_principal(self, cash_avail: float, t_month: int) -> float:
        wf_type = self._effective_waterfall_type(t_month)
        seniors = [tr for tr in self.tranches if tr.ttype != "residual" and tr.outstanding > 1e-8]

        if wf_type == "sequential":
            seniors.sort(key=lambda x: x.name)
            for tr in seniors:
                pay = min(tr.outstanding, cash_avail)
                tr.cash_principal.append(pay)
                cash_avail -= pay
        else:  # pro-rata (exact proportional allocation)
            total_outs = sum(tr.outstanding for tr in seniors)
            if total_outs <= 1e-12:
                for tr in seniors:
                    tr.cash_principal.append(0.0)
            else:
                # Strict proportional split (servicer rounding not modeled)
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

            cash = self._pay_interest_senior(avail_cf, t)
            cash = self._pay_principal(cash, t)

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
        cfs = [i + p for i, p in zip(tr.cash_interest, tr.cash_principal)]

        def pv(dm_annual: float) -> float:
            pv_total = 0.0
            for k, cf in enumerate(cfs, start=1):
                yf = k / 12.0
                base_rate = self.curve_fn(yf) if self.curve_fn else self.base_index_annual  # annual simple rate
                # Discount with monthly comp on (zero + DM)
                df = 1.0 / ((1.0 + (base_rate + dm_annual) / 12.0) ** k)
                pv_total += cf * df
            return pv_total

        dm = bisection(lambda x: pv(x) - price_cash, -0.05, 0.50)
        return float("nan") if dm is None else dm * 1e4

    def results_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
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
