import numpy as np
import matplotlib.pyplot as plt

def simulate_asset_pool(cpr, initial_balance=1_000_000, annual_rate=0.05, term=360):
    """
    Simulate an amortizing asset pool with CPR prepayment.
    
    Parameters:
      cpr            : Annual conditional prepayment rate (as a decimal)
      initial_balance: Total initial pool balance
      annual_rate    : Annual interest rate
      term           : Number of months in the simulation (e.g. 360 for 30 years)
      
    Returns:
      A list of dictionaries containing monthly cash flow details.
    """
    r = annual_rate / 12.0  # monthly interest rate
    # Compute constant mortgage payment
    payment = initial_balance * r / (1 - (1 + r) ** (-term))
    balance = initial_balance
    cashflows = []
    # Convert annual CPR to a monthly Single Monthly Mortality (SMM)
    SMM = 1 - (1 - cpr) ** (1 / 12.0)
    
    for m in range(1, term + 1):
        if balance <= 0:
            break
        interest = balance * r
        scheduled_principal = payment - interest
        # Adjust if the scheduled principal exceeds the remaining balance.
        scheduled_principal = min(scheduled_principal, balance)
        # Prepayment calculated on the remaining balance after scheduled principal
        prepayment = SMM * (balance - scheduled_principal)
        total_principal = scheduled_principal + prepayment
        total_principal = min(total_principal, balance)  # final period adjustment
        cashflows.append({
            'month': m,
            'payment': payment if balance > payment else balance + interest,
            'interest': interest,
            'scheduled_principal': scheduled_principal,
            'prepayment': prepayment,
            'total_principal': total_principal,
            'balance': balance - total_principal
        })
        balance -= total_principal
    return cashflows

def waterfall_distribution(cashflows, waterfall_type, pool=1_000_000, senior_pct=0.7):
    """
    Distribute the monthly principal cash flows between two tranches.
    
    Parameters:
      cashflows    : List of monthly cash flows (from simulate_asset_pool)
      waterfall_type: "sequential" or "prorata"
      pool         : Total initial pool balance
      senior_pct   : Percentage of the pool allocated to the senior tranche
      
    Returns:
      Two lists of monthly payments allocated to the senior and junior tranches.
    """
    # Define tranche notionals.
    senior_notional = pool * senior_pct
    junior_notional = pool * (1 - senior_pct)
    senior_balance = senior_notional
    junior_balance = junior_notional
    
    senior_payments = []
    junior_payments = []
    
    for cf in cashflows:
        available = cf['total_principal']
        if waterfall_type == "sequential":
            # Sequential: pay the senior tranche first.
            senior_payment = min(available, senior_balance)
            senior_balance -= senior_payment
            available -= senior_payment
            junior_payment = min(available, junior_balance)
            junior_balance -= junior_payment
        elif waterfall_type == "prorata":
            # Pro-rata: distribute in proportion to remaining balances.
            total_balance = senior_balance + junior_balance
            if total_balance > 0:
                senior_payment = min(available * (senior_balance / total_balance), senior_balance)
                junior_payment = min(available * (junior_balance / total_balance), junior_balance)
                senior_balance -= senior_payment
                junior_balance -= junior_payment
            else:
                senior_payment, junior_payment = 0, 0
        else:
            raise ValueError("Unknown waterfall type. Use 'sequential' or 'prorata'.")
        
        senior_payments.append(senior_payment)
        junior_payments.append(junior_payment)
    
    return senior_payments, junior_payments

def compute_WAL(payments, cashflow_months, tranche_notional):
    """
    Compute the Weighted Average Life (WAL) of a tranche.
    
    Parameters:
      payments         : List of monthly principal payments for the tranche
      cashflow_months  : List of month numbers corresponding to the payments
      tranche_notional : Original notional of the tranche
      
    Returns:
      WAL in years.
    """
    # Multiply each payment by the time (in years) at which it was received.
    total_weighted_time = sum(p * (m / 12.0) for p, m in zip(payments, cashflow_months))
    return total_weighted_time / tranche_notional

def simulate_ABS(cpr, waterfall_type, initial_balance=1_000_000, annual_rate=0.05, term=360, senior_pct=0.7):
    """
    Runs the full ABS simulation for a given CPR and waterfall structure.
    
    Returns:
      A dictionary including the cashflows, allocated payments, and computed WALs.
    """
    cashflows = simulate_asset_pool(cpr, initial_balance, annual_rate, term)
    senior_payments, junior_payments = waterfall_distribution(cashflows, waterfall_type, pool=initial_balance, senior_pct=senior_pct)
    months = [cf['month'] for cf in cashflows]
    
    senior_notional = initial_balance * senior_pct
    junior_notional = initial_balance * (1 - senior_pct)
    
    WAL_senior = compute_WAL(senior_payments, months, senior_notional)
    WAL_junior = compute_WAL(junior_payments, months, junior_notional)
    
    return {
        'cashflows': cashflows,
        'senior_payments': senior_payments,
        'junior_payments': junior_payments,
        'WAL_senior': WAL_senior,
        'WAL_junior': WAL_junior
    }
import matplotlib.pyplot as plt

def plot_waterfall_curves(cpr=0.06, waterfall_type='sequential', initial_balance=1_000_000, annual_rate=0.05, term=360, senior_pct=0.7):
    """
    Plot the waterfall payment curves (monthly principal splits) for a given CPR and waterfall type.
    
    Parameters:
      cpr             : Annual CPR (e.g., 0.06 for 6%)
      waterfall_type  : 'sequential' or 'prorata'
      initial_balance : Initial asset pool balance
      annual_rate     : Annual interest rate of the pool
      term            : Loan term in months
      senior_pct      : Percentage of the pool allocated to the senior tranche
    """
    # Run the ABS simulation for the selected waterfall type.
    simulation = simulate_ABS(cpr, waterfall_type, initial_balance, annual_rate, term, senior_pct)
    cashflows = simulation['cashflows']
    months = [cf['month'] for cf in cashflows]
    senior_payments = simulation['senior_payments']
    junior_payments = simulation['junior_payments']
    
    # Create line plots for each tranche.
    plt.figure(figsize=(12, 6))
    plt.plot(months, senior_payments, label='Senior Payment', marker='o')
    plt.plot(months, junior_payments, label='Junior Payment', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Principal Payment')
    plt.title(f'Waterfall Payment Curves ({waterfall_type.capitalize()} Waterfall) at CPR = {cpr:.2%}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_waterfall_curves(cpr=0.06, waterfall_type='sequential')


# To plot the waterfall for a pro-rata structure, simply call:
# plot_waterfall_distribution(cpr=0.06, waterfall_type='prorata')

def run_simulation():
    """
    Runs the ABS simulation over a range of CPR values for both waterfall types and plots the resulting WAL.
    """
    # Define a range of CPR values from 0% to 10%.
    cpr_values = np.linspace(0.0, 0.1, 11)
    
    WAL_seq_senior, WAL_seq_junior = [], []
    WAL_pro_senior, WAL_pro_junior = [], []
    
    for cpr in cpr_values:
        # Simulate sequential waterfall.
        seq_results = simulate_ABS(cpr, "sequential")
        # Simulate pro-rata waterfall.
        pro_results = simulate_ABS(cpr, "prorata")
        
        WAL_seq_senior.append(seq_results['WAL_senior'])
        WAL_seq_junior.append(seq_results['WAL_junior'])
        WAL_pro_senior.append(pro_results['WAL_senior'])
        WAL_pro_junior.append(pro_results['WAL_junior'])
    
    # Plotting WAL vs CPR for Sequential Waterfall
    plt.figure(figsize=(10, 5))
    plt.plot(cpr_values * 100, WAL_seq_senior, marker='o', label='Senior WAL (Sequential)')
    plt.plot(cpr_values * 100, WAL_seq_junior, marker='o', label='Junior WAL (Sequential)')
    plt.xlabel('CPR (%)')
    plt.ylabel('WAL (years)')
    plt.title('WAL vs CPR - Sequential Waterfall')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plotting WAL vs CPR for Pro-rata Waterfall
    plt.figure(figsize=(10, 5))
    plt.plot(cpr_values * 100, WAL_pro_senior, marker='o', label='Senior WAL (Pro-rata)')
    plt.plot(cpr_values * 100, WAL_pro_junior, marker='o', label='Junior WAL (Pro-rata)')
    plt.xlabel('CPR (%)')
    plt.ylabel('WAL (years)')
    plt.title('WAL vs CPR - Pro-rata Waterfall')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_simulation()
