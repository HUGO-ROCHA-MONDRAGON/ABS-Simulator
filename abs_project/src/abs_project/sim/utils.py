# utils.py
def tranche_repayment_summary(engine):
    """
    Print when each tranche is fully repaid (in months and years).
    """
    print("\n💰 Tranche Repayment Summary")
    for tr in engine.tranches:
        if sum(tr.cash_principal) <= 0:
            print(f"  • {tr.name}: no repayments recorded")
            continue
        last_nonzero = max(i for i, p in enumerate(tr.cash_principal, start=1) if p > 1e-6)
        print(f"  • {tr.name}: fully repaid after {last_nonzero} months ({last_nonzero/12:.2f} years)")


from .plots import plot_waterfall, plot_tranche_balances
from .utils import tranche_repayment_summary

def show_scenario_details(engine):
    """
    Display repayment summary + waterfall plots for a single simulated scenario.
    """
    print("\n=============================")
    print(f"📘 Scenario details: {engine.ass.scenario_name}")
    print("=============================")
    tranche_repayment_summary(engine)
    plot_waterfall(engine)
    plot_tranche_balances(engine)
