from typing import Dict, List, Tuple
from .engine import DealMeta, Pool, Tranche, Assumptions


def load_from_dict(d: Dict) -> Tuple[DealMeta, Pool, List[Tranche], Assumptions]:
    """
    Load deal, pool, tranches, and assumptions from a YAML dictionary.
    Handles key renames such as 'type' -> 'ttype'.
    """

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
        waterfall_type=wf_type,
    )

    pool = Pool(**d["pool"])

    tranches = []
    for t in d["structure"]["tranches"]:
        # --- Explicit rename for YAML compatibility ---
        if "type" in t:
            t["ttype"] = t.pop("type")

        # --- Fill missing optional fields safely ---
        t.setdefault("spread_bps", 0)
        t.setdefault("price", 100.0)
        t.setdefault("legal_final", deal.periods)
        t.setdefault("rating", "NR")

        tranches.append(Tranche(**t))

    assumptions = Assumptions(**d.get("assumptions", {}))
    return deal, pool, tranches, assumptions
