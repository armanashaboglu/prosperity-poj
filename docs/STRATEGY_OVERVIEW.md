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
