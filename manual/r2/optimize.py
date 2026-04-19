import numpy as np
from scipy.optimize import minimize

def research_score(x_research):
    """Calculates the Research edge (logarithmic growth)."""
    return 200_000 * np.log(1 + x_research) / np.log(1 + 100)

def scale_score(x_scale):
    """Calculates the Scale breadth (linear growth)."""
    return 7 * (x_scale / 100.0)

def expected_speed_multiplier(x_speed):
    """
    Estimates the Speed multiplier based on your investment.
    
    Since this is an adversarial rank-based game, this function is an ASSUMPTION.
    Here, we assume a uniform distribution of opponents: 
    - 0 invested gives you the bottom rank (0.1)
    - 100 invested guarantees the top rank (0.9)
    - It scales linearly in between.
    
    You should tweak this function based on how competitive you think the field is.
    For example, if you think everyone will invest at least 20 in speed, 
    anything below 20 should return 0.1.
    """
    # Simple linear assumption:
    return 0.1 + 0.8 * (x_speed / 100.0)

def calculate_pnl(x):
    """
    Objective function to calculate PnL.
    x is an array: [Research%, Scale%, Speed%]
    """
    x_research, x_scale, x_speed = x
    
    # Calculate individual pillar outcomes
    research = research_score(x_research)
    scale = scale_score(x_scale)
    speed_mult = expected_speed_multiplier(x_speed)
    
    # Calculate budget used (total budget is 50,000 for 100%)
    total_percentage = x_research + x_scale + x_speed
    budget_used = (total_percentage / 100.0) * 50_000
    
    # Gross PnL
    gross_pnl = research * scale * speed_mult
    
    # Net PnL
    return gross_pnl - budget_used

def objective(x):
    """
    SciPy minimizes functions, so we return the negative PnL 
    to maximize our profit.
    """
    return -calculate_pnl(x)

def optimize_allocation():
    # Initial guess (even split)
    x0 = np.array([33.3, 33.3, 33.3])
    
    # Bounds for each percentage [0, 100]
    bounds = [(0, 100), (0, 100), (43, 43)]
    
    # Constraint: The sum of percentages must be <= 100
    # Inequality constraints are strictly non-negative in SciPy: fun(x) >= 0
    constraints = {'type': 'ineq', 'fun': lambda x: 100.0 - np.sum(x)}
    
    # Run the SLSQP optimizer
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if result.success:
        optimal_x = result.x
        max_pnl = calculate_pnl(optimal_x)
        
        print("Optimization Successful!")
        print("-" * 30)
        print(f"Optimal Research Allocation : {optimal_x[0]:.2f}%")
        print(f"Optimal Scale Allocation    : {optimal_x[1]:.2f}%")
        print(f"Optimal Speed Allocation    : {optimal_x[2]:.2f}%")
        print(f"Total Allocation Used       : {np.sum(optimal_x):.2f}%")
        print("-" * 30)
        print(f"Expected Maximum PnL        : {max_pnl:,.2f} XIReCs")
    else:
        print("Optimization failed:", result.message)

if __name__ == "__main__":
    optimize_allocation()