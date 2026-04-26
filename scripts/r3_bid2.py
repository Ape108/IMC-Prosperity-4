def calculate_bid_2_pnl(our_bid, predicted_avg_b2):
    """
    Calculates the Expected PnL for Bid 2 given a specific bid and a predicted global average.
    """
    min_reserve = 670
    max_reserve = 920
    increment = 5
    sell_price = 920

    # Generate the distribution of counterparties (670, 675, ..., 920)
    reserve_prices = list(range(min_reserve, max_reserve + increment, increment))

    # Calculate Volume Won (our bid must strictly beat their reserve)
    volume_won = sum(1 for reserve in reserve_prices if our_bid > reserve)
    
    # Base Profit Calculation
    margin = sell_price - our_bid
    base_pnl = volume_won * margin

    # Apply the game theory penalty if we fail to beat the global average
    if our_bid <= predicted_avg_b2:
        # Avoid division by zero in the rare case someone bids exactly 920
        if our_bid == sell_price:
            return 0
            
        penalty_multiplier = ((sell_price - predicted_avg_b2) / (sell_price - our_bid)) ** 3
        final_pnl = base_pnl * penalty_multiplier
    else:
        final_pnl = base_pnl

    return final_pnl

def optimize_bid_2(predicted_avg_b2):
    """
    Runs a grid search to find the optimal Bid 2 value for a given market prediction.
    """
    best_bid = 0
    max_pnl = -1
    
    # Test all logical integer bids from the minimum up to just below the sell price
    for test_bid in range(671, 920):
        expected_pnl = calculate_bid_2_pnl(test_bid, predicted_avg_b2)
        
        if expected_pnl > max_pnl:
            max_pnl = expected_pnl
            best_bid = test_bid
            
    return best_bid, max_pnl

if __name__ == "__main__":
    # --- TUNE THIS VARIABLE ---
    # What do you expect the average of all OTHER players' second bids to be?
    # Knowing that 791 is the mathematically optimal baseline, it will likely be higher.
    predicted_market_average = 800 
    
    optimal_bid, expected_return = optimize_bid_2(predicted_market_average)
    
    print("--- Bid 2 Simulation & Optimization ---")
    print(f"Your Predicted Global Average: {predicted_market_average}")
    print(f"Optimal Bid to Submit:       {optimal_bid}")
    print(f"Expected Final PnL:          {expected_return:,.2f}")
    
    print("\n--- Contextual Breakdown ---")
    # Show what happens if you bid the absolute baseline (791) in this environment
    baseline_pnl = calculate_bid_2_pnl(791, predicted_market_average)
    print(f"PnL if you bid the 791 baseline: {baseline_pnl:,.2f}")