import numpy as np
import pandas as pd

def simulate_avellaneda_stoikov():
    # Model Parameters
    T = 1.0           # Total time (e.g., 1 trading day)
    steps = 100       # Number of steps in the simulation
    dt = T / steps    # Time step
    sigma = 2.0       # Volatility
    gamma = 0.1       # Risk aversion
    kappa = 1.5       # Order book liquidity profile
    s0 = 100.0        # Initial mid-price
    
    # Initialize state variables
    mid_price = s0
    inventory = 0
    history = []

    for i in range(steps):
        t = i * dt
        time_left = T - t
        
        # 1. Calculate Reservation Price (r)
        # If inventory > 0 (long), reservation price drops below mid-price.
        # If inventory < 0 (short), reservation price rises above mid-price.
        reservation_price = mid_price - (inventory * gamma * (sigma**2) * time_left)
        
        # 2. Calculate Optimal Spread (delta)
        # Note: In this specific formulation, the spread is constant over time, 
        # though advanced variations make it time-dependent.
        spread = (2 / gamma) * np.log(1 + (gamma / kappa))
        
        # 3. Calculate Bid and Ask Quotes
        bid = reservation_price - (spread / 2)
        ask = reservation_price + (spread / 2)
        
        # Store the current state
        history.append({
            'Step': i,
            'Mid Price': round(mid_price, 2),
            'Inventory': inventory,
            'Res Price': round(reservation_price, 2),
            'Bid': round(bid, 2),
            'Ask': round(ask, 2)
        })
        
        # 4. Simulate Price Movement (Brownian Motion / Random Walk)
        mid_price += np.random.normal(0, sigma * np.sqrt(dt))
        
        # 5. Simulate Order Fills (Simplified Probability Model)
        # In reality, fill probability depends on distance from mid-price.
        # Here, we use a simplified logic: if we are long, we are more desperate to sell,
        # our ask drops, making a sell fill more likely.
        
        # Base fill probability
        prob_buy_fill = 0.5 - (0.05 * inventory) 
        prob_sell_fill = 0.5 + (0.05 * inventory)
        
        # Clip probabilities to stay between 0 and 1
        prob_buy_fill = max(0.0, min(1.0, prob_buy_fill))
        prob_sell_fill = max(0.0, min(1.0, prob_sell_fill))
        
        rand_val = np.random.uniform(0, 1)
        
        if rand_val < prob_buy_fill:
            inventory += 1  # Our bid was hit, we bought 1 unit
        elif rand_val > (1 - prob_sell_fill):
            inventory -= 1  # Our ask was lifted, we sold 1 unit

    # Output the results
    df = pd.DataFrame(history)
    return df

# Run the simulation
results = simulate_avellaneda_stoikov()

# Display the first and last few steps to see the shift
print("--- Early Steps ---")
print(results.head())
print("\n--- Final Steps ---")
print(results.tail())