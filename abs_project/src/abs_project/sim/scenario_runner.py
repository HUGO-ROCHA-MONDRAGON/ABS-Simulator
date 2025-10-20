# scenario_runner.py
import pandas as pd
import matplotlib.pyplot as plt
from .engine import WaterfallEngine, Assumptions
from .loader import load_from_dict
from .plots import plot_waterfall


def run_scenarios(abs_data, scenario_list, base_index_annual=0.026, plot=False):
    """
    Runs multiple scenarios on the same ABS deal and returns a summary DataFrame.
    
    Parameters
    ----------
    abs_data : dict
        Parsed YAML dictionary describing the ABS deal.
    scenario_list : list[Assumptions]
        List of Assumptions objects to run.
    base_index_annual : float, optional
        Base index rate (e.g., EURIBOR), by default 0.026.
    plot : bool, optional
        Whether to plot the waterfall for each scenario.
    """
    results = []

    for ass in scenario_list:
        deal, pool, tranches, _ = load_from_dict(abs_data)
        engine = WaterfallEngine(deal, pool, tranches, ass, base_index_annual)
        engine.simulate()

        summary = engine.results_summary()
        for tr_name, metrics in summary.items():
            row = {
                "Scenario": ass.scenario_name,
                "Tranche": tr_name,
                "WAL (yrs)": round(metrics.get("WAL_years", 0), 2),
                "DM (bps)": (
                    round(metrics.get("DM_bps", 0), 1)
                    if not pd.isna(metrics.get("DM_bps", 0))
                    else None
                ),
                "Total Interest (€)": round(metrics.get("int_total", 0), 0),
                "Total Principal (€)": round(metrics.get("prin_total", 0), 0),
                "Total Residual (€)": round(
                    metrics.get("total_residual_cash", 0), 0
                ),
                "CPR (%)": round(ass.CPR_annual * 100, 1),
                "CDR (%)": round(ass.CDR_annual * 100, 1),
                "Recovery (%)": round(ass.recovery_rate * 100, 1),
            }
            results.append(row)

        if plot:
            print(f"\n📊 Scenario: {ass.scenario_name}")
            plot_waterfall(engine)

    df = pd.DataFrame(results)
    return df


def style_results(df: pd.DataFrame):
    """
    Return a nicely formatted styled DataFrame for Jupyter display.
    """
    styled = (
        df.style.hide(axis="index")
        .format({
            "WAL (yrs)": "{:.2f}",
            "DM (bps)": "{:.0f}",
            "Total Interest (€)": "{:,.0f}",
            "Total Principal (€)": "{:,.0f}",
            "Total Residual (€)": "{:,.0f}",
            "CPR (%)": "{:.1f}",
            "CDR (%)": "{:.1f}",
            "Recovery (%)": "{:.1f}",
        })
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "bold"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
        .background_gradient(subset=["WAL (yrs)", "DM (bps)"], cmap="Blues")
    )
    return styled




def plot_scenario_summary(df: pd.DataFrame):
    """
    Compare key metrics (WAL, DM, Loss) across scenarios and tranches.
    Expects the DataFrame returned by run_scenarios().
    """
    if df.empty:
        print("⚠️ No data to plot.")
        return

    # --- Ensure proper order ---
    metrics = ["WAL (yrs)", "DM (bps)", "Total Principal (€)"]
    df_group = df.groupby(["Scenario", "Tranche"])[metrics].mean().reset_index()

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for tr in df_group["Tranche"].unique():
            sub = df_group[df_group["Tranche"] == tr]
            ax.plot(
                sub["Scenario"],
                sub[metric],
                marker="o",
                linewidth=2,
                label=tr,
            )
        ax.set_title(metric, fontsize=12, weight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Scenario Comparison: WAL / DM / Principal", fontsize=14, weight="bold")
    fig.tight_layout()
    plt.show()

