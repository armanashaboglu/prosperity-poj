DESCRIPTION OF CURRENT & PAST STRATEGIES:


ROUND 0 (tutorial):

 ARCHITECTURE OVERVIEW:
 --------------------
The code uses an object-oriented design with an inheritance hierarchy (inspired from jmerle's repo):
- Base Strategy class: Defines common operations for all strategies
 - MarketMakingStrategy: Implements sophisticated market-making logic with position management
 - Product-specific strategies: Implement custom fair value calculations for each product

KEY COMPONENTS:
-------------
1. STRATEGY HIERARCHY:
   - Strategy (base class): Provides common methods like buy(), sell() and run()
   - MarketMakingStrategy: Implements core market-making logic with position management
   - RainforestResinStrategy: Uses stable true value (10000) for RAINFOREST_RESIN
   - KelpStrategy: Uses dynamic true value calculation for KELP based on market data
 2. POSITION MANAGEMENT:
    - Window tracking: Maintains a sliding window (deque) of recent positions to detect if stuck at limits
    - Liquidation modes: 
      * Soft liquidation: Triggered when position is at limit for 50%+ of recent history
      * Hard liquidation: Triggered when position is at limit for entire history
    - Dynamic pricing: Adjusts max buy/sell prices based on current position 
      (more conservative when position is already skewed)

3. ORDER EXECUTION PRIORITY:
   - First phase: Take existing favorable orders in the market
   - Second phase: If stuck at limits, place aggressive orders to rebalance position
   - Third phase: Place regular market-making orders at competitive prices

4. PRICE CALCULATION:
   - RAINFOREST_RESIN: Assumes stable true value of 10000
   - KELP: Dynamic calculation based on most popular price points (sum of highest volume bid and ask prices divided by 2 )
 5. STATE PERSISTENCE:
   - save() and load() methods allow strategies to maintain state between rounds
   - Critical for tracking position history and detecting stuck positions

TRADING LOGIC DETAILS:
--------------------
1. FAIR VALUE CALCULATION:
   - Each product determines its own "true value" (fair price)
   - True value serves as an anchor for buy/sell decisions

2. BUYING/SELLING LOGIC (3-phase approach):
   - Phase 1: Take existing favorable orders in the market
     * For buying: Take sell orders with price <= max_buy_price
     * For selling: Take buy orders with price >= min_sell_price
     * Quantity limited by both available volume and position capacity
   
   - Phase 2: Special handling for stuck positions
     * If hard_liquidate: Trade aggressively at true value with 50% of remaining capacity
     * If soft_liquidate: Trade somewhat aggressively (true_value±2) with 50% of remaining capacity
   
   - Phase 3: Place new orders at competitive prices
     * Find popular price points based on volume
     * Place remaining capacity at competitive price relative to existing orders

3. POSITION MANAGEMENT:
   - Dynamic max_buy_price and min_sell_price based on position
      * When long (positive position), lower max_buy_price to reduce buying
     * When short (negative position), raise min_sell_price to reduce selling
   - Sliding window tracks if positions are consistently at limits
   - Liquidation modes trigger more aggressive pricing to escape stuck positions

ROUND 1:

RAINFOREST\RESIN Strategy (V3RainforestResinStrategy):
Type: Market Making (based on V3MarketMakingStrategy).
Fair Value: Assumes a static fair value of 10000 (defined in PARAMS).
Logic:
Phase 1 (Take): Takes existing sell orders priced at or below 10000 and buy orders priced at or above 10000, respecting position limits.
Phase 2 (Make): Places new buy/sell orders based on the remaining order book after Phase 1. It aims to place orders one tick away from the best remaining bid/ask (below/above 10000 respectively), but adjusts the price based on the volume at the best level (if volume <= 6, it uses the best level's price directly). It also includes specific rules to avoid placing orders at 9997, 9999, 10001, and 10003.
Position Management: Inherits position window tracking and liquidation logic from V3MarketMakingStrategy, but this logic doesn't appear to be actively used in the V3RainforestResinStrategy's specific act method implementation; the primary logic focuses on the two phases described above.
KELP Strategy (PrototypeKelpStrategy):
Type: Market Making (based on PrototypeMarketMakingStrategy).
Fair Value: Dynamically calculated. It prioritizes the average of the highest-volume bid and ask prices. If these aren't available or valid (ask <= bid), it falls back to the midpoint of the best bid/ask. If only one side exists, it uses that price.
Logic (inherits from PrototypeMarketMakingStrategy):
Phase 1 (Take): Takes existing orders within calculated price boundaries (max_buy_price, min_sell_price), which are adjusted based on current position (more conservative when skewed).
Phase 2 (Liquidation): If the position is stuck at the limit (checked using a 4-tick sliding window self.window), it places aggressive orders to rebalance:
Hard Liquidation (stuck 4/4 ticks): Trades 50% of capacity at the calculated true value.
Soft Liquidation (stuck >= 2/4 ticks & last tick): Trades 50% of capacity at true_value - 2 if buying (stuck short) or true_value + 2 if selling (stuck long). Note: The code explicitly implements the buy-side liquidation but seems to omit the sell-side logic.
Phase 3 (Make): Places remaining capacity based on popular prices (highest volume bid/ask) or derived from the true value if the book is thin.
State: Uses save/load to persist the self.window deque for liquidation logic.
SQUID\INK Strategy (SquidInkRsiStrategy):
Type: Technical Indicator (Relative Strength Index - RSI).
Parameters: Uses rsi_window, rsi_overbought, rsi_oversold from the PARAMS dictionary.
Logic:
Calculates the mid-price (average of best bid and ask).
Calculates RSI using Wilder's smoothing method based on the history of mid-prices.
Signal:
If RSI > rsi_overbought: Places an aggressive sell order for the full remaining capacity at best_bid_price - 1.
If RSI < rsi_oversold: Places an aggressive buy order for the full remaining capacity at best_ask_price + 1.
Does not actively make markets or consider fair value beyond the RSI signal.
State: Uses save/load to persist mid_price_history, avg_gain, avg_loss, and rsi_initialized for continuous RSI calculation.

ROUND 2:


