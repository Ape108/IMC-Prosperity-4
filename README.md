# IMC Prosperity 4 Trading Strategy

This repository contains a trading strategy implementation for the IMC Prosperity 4 challenge, a simulated trading competition where participants develop algorithms to trade various financial instruments in a realistic market environment.

## Overview

IMC Prosperity is an annual algorithmic trading competition organized by IMC Trading. Participants are provided with historical market data and must implement trading strategies that maximize profit while managing risk. The challenge simulates real-world trading conditions with multiple products, order books, and market dynamics.

## Project Structure

```
├── README.md                 # Project documentation
├── strategy.py              # Main trading strategy implementation
└── data/
    └── round_0/            # Historical market data
        ├── prices_round_0_day_-1.csv    # Order book data
        ├── prices_round_0_day_-2.csv    # Order book data
        ├── trades_round_0_day_-1.csv    # Executed trades
        └── trades_round_0_day_-2.csv    # Executed trades
```

## Data Format

### Prices Data (`prices_round_0_day_X.csv`)
Contains order book snapshots with the following columns:
- `day`: Trading day number
- `timestamp`: Time in milliseconds since market open
- `product`: Financial instrument (e.g., TOMATOES, EMERALDS)
- `bid_price_1/2/3`: Best bid prices (highest 3 levels)
- `bid_volume_1/2/3`: Corresponding bid volumes
- `ask_price_1/2/3`: Best ask prices (lowest 3 levels)
- `ask_volume_1/2/3`: Corresponding ask volumes
- `mid_price`: Midpoint between best bid and ask
- `profit_and_loss`: Cumulative P&L for the strategy

### Trades Data (`trades_round_0_day_X.csv`)
Contains executed trades with the following columns:
- `timestamp`: Time of trade execution
- `buyer`: Buying party
- `seller`: Selling party
- `symbol`: Trading symbol
- `currency`: Currency used
- `price`: Execution price
- `quantity`: Trade volume

## Strategy Implementation

The main trading logic is implemented in `strategy.py`. The strategy should:

1. Analyze market data (order book and trade history)
2. Generate trading signals based on market conditions
3. Submit limit orders to buy/sell positions
4. Manage inventory and risk

### Key Components

- **Market Analysis**: Process order book data to identify trends and opportunities
- **Signal Generation**: Implement logic to determine when to buy/sell
- **Order Management**: Submit appropriate orders to the market
- **Risk Management**: Monitor position sizes and P&L

## Requirements

- Python 3.8+
- pandas (for data manipulation)
- numpy (for numerical computations)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ape108/IMC-Prosperity-4.git
cd IMC-Prosperity-4
```

2. Install dependencies:
```bash
pip install pandas numpy
```

## Usage

1. Implement your trading strategy in `strategy.py`
2. Run the strategy against historical data:
```bash
python strategy.py
```

## Evaluation

Strategies are evaluated based on:
- **Profit & Loss**: Total returns generated
- **Sharpe Ratio**: Risk-adjusted returns
- **Market Impact**: How much the strategy moves prices
- **Execution Quality**: Ability to get favorable prices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your strategy improvements
4. Test thoroughly with historical data
5. Submit a pull request

## License

This project is for educational and competitive purposes. Please refer to IMC Prosperity's terms and conditions for usage rights.

## Acknowledgments

- IMC Trading for organizing the Prosperity challenge
- The trading community for sharing insights and strategies