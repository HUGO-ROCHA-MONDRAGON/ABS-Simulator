# abs_deal_generator.py
import random
import datetime
import yaml
import pandas as pd
from pathlib import Path


# ============================================================
# 1️⃣ DEAL STRUCTURE
# ============================================================
def random_deal_structure():
    return {
        "deal_name": f"AUTOFR {random.randint(2020, 2026)}-{random.randint(1, 3)}",
        "issuer": random.choice([
            "Santander Consumer France",
            "BNP Paribas Personal Finance",
            "Volkswagen Bank",
            "Crédit Agricole Consumer Finance",
            "RCI Banque"
        ]),
        "start_date": str(datetime.date(2025, 1, 1)),
        "frequency": "Monthly",
        "periods": random.choice([48, 60, 72]),
        "currency": "EUR",
        "benchmark_curve": random.choice(["EURIBOR_1M", "EURIBOR_3M"]),
        "day_count": "ACT/360",
    }


# ============================================================
# 2️⃣ POOL (ACTIF)
# ============================================================
def random_pool():
    balance = random.randint(80, 150) * 1_000_000
    coupon = round(random.uniform(0.02, 0.06), 4)
    wam = random.randint(36, 72)
    war = coupon + random.uniform(-0.002, 0.004)
    return {
        "balance": balance,
        "coupon_annual": coupon,
        "amortization": random.choice(["level_payment", "pass_through"]),
        "seasoning_months": random.randint(0, 12),
        "weighted_avg_maturity": wam,
        "weighted_avg_rate": round(war, 4),
    }


# ============================================================
# 3️⃣ TRANCHES
# ============================================================
def random_tranche(name, tranche_type, pool_balance, notional):
    # Ratings hiérarchiques
    if name == "A":
        rating = "AAA"
    elif name == "B":
        rating = "A"
    elif name == "C":
        rating = "BB+"
    else:
        rating = "NR"

    tranche = {
        "name": name,
        "type": tranche_type,
        "notional": notional,
        "price": round(random.uniform(95, 101), 2),
        "legal_final": random.choice([48, 60, 72]),
        "rating": rating,
    }

    # Conditions de taux selon le type
    if tranche_type == "fixed":
        tranche["coupon_annual"] = round(random.uniform(0.03, 0.07), 4)
    elif tranche_type == "floating":
        tranche["spread_bps"] = random.randint(80, 200)

    return tranche


# ============================================================
# 4️⃣ STRUCTURE DU PASSIF (TRANCHES + WATERFALL)
# ============================================================
def random_passive_structure(pool_balance):
    # Parts décroissantes (A > B > C > Equity)
    share_A = random.uniform(0.75, 0.90)
    share_B = random.uniform(0.05, 0.12)
    share_C = random.uniform(0.02, 0.06)

    total = share_A + share_B + share_C
    if total > 0.97:
        factor = 0.97 / total
        share_A *= factor
        share_B *= factor
        share_C *= factor

    # Calcul des notionals
    notional_A = round(pool_balance * share_A, 2)
    notional_B = round(pool_balance * share_B, 2)
    notional_C = round(pool_balance * share_C, 2)
    notional_E = round(pool_balance - (notional_A + notional_B + notional_C), 2)

    # --- Generate consistent spreads and prices ---
    # Spread_A < Spread_B < Spread_C
    spread_A = random.randint(80, 130)
    spread_B = spread_A + random.randint(40, 80)
    spread_C = spread_B + random.randint(80, 150)

    # Price_A > Price_B > Price_C
    price_A = round(random.uniform(100, 101), 2)
    price_B = round(random.uniform(97, price_A - 0.5), 2)
    price_C = round(random.uniform(94, price_B - 0.5), 2)

    # Création des tranches
    tranche_A = {
        **random_tranche("A", "floating", pool_balance, notional_A),
        "spread_bps": spread_A,
        "price": price_A
    }

    tranche_B = {
        **random_tranche("B", "floating", pool_balance, notional_B),
        "spread_bps": spread_B,
        "price": price_B
    }

    tranche_C = {
        **random_tranche("C", "floating", pool_balance, notional_C),
        "spread_bps": spread_C,
        "price": price_C
    }

    tranche_E = random_tranche("Equity", "residual", pool_balance, notional_E)

    return {
        "waterfall": {"type": random.choice(["sequential", "pro_rata"])},
        "tranches": [tranche_A, tranche_B, tranche_C, tranche_E],
    }


# ============================================================
# 5️⃣ ASSEMBLAGE COMPLET DU DEAL
# ============================================================
def generate_abs_deal():
    deal = random_deal_structure()
    pool = random_pool()
    passive = random_passive_structure(pool["balance"])
    abs_deal = {"deal": deal, "pool": pool, "structure": passive}
    return abs_deal


# ============================================================
# 6️⃣ EXPORT EXCEL
# ============================================================
def export_to_excel(abs_deal):
    deal_name = abs_deal["deal"]["deal_name"].replace(" ", "_")
    output_dir = Path(__file__).resolve().parents[1] / "data" / "deals"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_dir / f"{deal_name}.xlsx"

    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        # Onglet Deal Structure
        pd.DataFrame(abs_deal["deal"].items(), columns=["Field", "Value"]).to_excel(
            writer, sheet_name="Deal_Structure", index=False
        )

        # Onglet Pool
        pd.DataFrame(abs_deal["pool"].items(), columns=["Field", "Value"]).to_excel(
            writer, sheet_name="Pool", index=False
        )

        # Onglet Tranches
        tranches_df = pd.DataFrame(abs_deal["structure"]["tranches"])
        tranches_df.to_excel(writer, sheet_name="Tranches", index=False)

        # Onglet Waterfall
        pd.DataFrame(
            abs_deal["structure"]["waterfall"].items(), columns=["Field", "Value"]
        ).to_excel(writer, sheet_name="Waterfall", index=False)

    print(f"✅ Excel file created: {file_name}")
