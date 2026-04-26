def optimize_bid_1():
    # Trading parameters based on the challenge rules
    min_reserve = 670
    max_reserve = 920
    increment = 5
    sell_price = 920

    # Generate the exact distribution of counterparty reserve prices
    # range() is exclusive at the top end, so we add the increment to include 920
    reserve_prices = list(range(min_reserve, max_reserve + increment, increment))

    best_bid = 0
    max_expected_pnl = 0
    best_volume_won = 0

    # Test every logical integer bid from just above the minimum up to the sell price
    for bid in range(min_reserve + 1, sell_price):
        
        # Count how many trades we win (our bid must be strictly greater than their reserve)
        volume_won = sum(1 for reserve in reserve_prices if bid > reserve)
        
        # Calculate our profit margin per pod
        margin = sell_price - bid
        
        # Calculate total Expected PnL for this bid
        expected_pnl = volume_won * margin

        # If this bid yields a higher PnL than our previous best, update our records
        if expected_pnl > max_expected_pnl:
            max_expected_pnl = expected_pnl
            best_bid = bid
            best_volume_won = volume_won

    return best_bid, best_volume_won, max_expected_pnl

if __name__ == "__main__":
    best_bid, volume, max_pnl = optimize_bid_1()
    
    print("--- Bid 1 Optimization Results ---")
    print(f"Optimal Bid: {best_bid}")
    print(f"Volume Won (Trades Executed): {volume}")
    print(f"Profit Margin per Trade: {920 - best_bid}")
    print(f"Total Expected PnL: {max_pnl}")