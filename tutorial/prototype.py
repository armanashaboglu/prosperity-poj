"""
#=========================================================================
# ALGORITHMIC TRADING STRATEGY - IMC PROSPERITY COMPETITION
#=========================================================================
#
# CONTEXT & GOALS:
# ---------------
# This trading algorithm implements a sophisticated market-making strategy for the IMC Prosperity competition,
# handling multiple products with different trading characteristics:
# - RAINFOREST_RESIN: A stable-priced product (similar to Amethysts from the previous competition)
# - KELP: A more volatile product with changing price dynamics (similar to Starfruit)
#
# The primary goal is to maximize profit while managing risk and position limits through:
# 1. Efficient price discovery using product-specific fair value calculations
# 2. Smart order execution prioritizing favorable existing orders before placing new ones
# 3. Sophisticated position management to avoid getting stuck at position limits
# 4. Adaptable pricing based on current market conditions and position
#
# ARCHITECTURE OVERVIEW:
# --------------------
# The code uses an object-oriented design with an inheritance hierarchy:
# - Base Strategy class: Defines common operations for all strategies
# - MarketMakingStrategy: Implements sophisticated market-making logic with position management
# - Product-specific strategies: Implement custom fair value calculations for each product
#
# KEY COMPONENTS:
# -------------
# 1. STRATEGY HIERARCHY:
#    - Strategy (base class): Provides common methods like buy(), sell() and run()
#    - MarketMakingStrategy: Implements core market-making logic with position management
#    - RainforestResinStrategy: Uses stable true value (10000) for RAINFOREST_RESIN
#    - KelpStrategy: Uses dynamic true value calculation for KELP based on market data
#
# 2. POSITION MANAGEMENT:
#    - Window tracking: Maintains a sliding window (deque) of recent positions to detect if stuck at limits
#    - Liquidation modes: 
#      * Soft liquidation: Triggered when position is at limit for 50%+ of recent history
#      * Hard liquidation: Triggered when position is at limit for entire history
#    - Dynamic pricing: Adjusts max buy/sell prices based on current position 
#      (more conservative when position is already skewed)
#
# 3. ORDER EXECUTION PRIORITY:
#    - First phase: Take existing favorable orders in the market
#    - Second phase: If stuck at limits, place aggressive orders to rebalance position
#    - Third phase: Place regular market-making orders at competitive prices
#
# 4. PRICE CALCULATION:
#    - RAINFOREST_RESIN: Assumes stable true value of 10000
#    - KELP: Dynamic calculation based on most popular price points (highest volume)
#
# 5. STATE PERSISTENCE:
#    - save() and load() methods allow strategies to maintain state between rounds
#    - Critical for tracking position history and detecting stuck positions
#
# TRADING LOGIC DETAILS:
# --------------------
# 1. FAIR VALUE CALCULATION:
#    - Each product determines its own "true value" (fair price)
#    - True value serves as an anchor for buy/sell decisions
#
# 2. BUYING/SELLING LOGIC (3-phase approach):
#    - Phase 1: Take existing favorable orders in the market
#      * For buying: Take sell orders with price <= max_buy_price
#      * For selling: Take buy orders with price >= min_sell_price
#      * Quantity limited by both available volume and position capacity
#    
#    - Phase 2: Special handling for stuck positions
#      * If hard_liquidate: Trade aggressively at true value with 50% of remaining capacity
#      * If soft_liquidate: Trade somewhat aggressively (true_value±2) with 50% of remaining capacity
#    
#    - Phase 3: Place new orders at competitive prices
#      * Find popular price points based on volume
#      * Place remaining capacity at competitive price relative to existing orders
#
# 3. POSITION MANAGEMENT:
#    - Dynamic max_buy_price and min_sell_price based on position
#      * When long (positive position), lower max_buy_price to reduce buying
#      * When short (negative position), raise min_sell_price to reduce selling
#    - Sliding window tracks if positions are consistently at limits
#    - Liquidation modes trigger more aggressive pricing to escape stuck positions
#
# This strategy improves upon basic market-making by:
# - Taking any favorable prices, not just specific fixed ones
# - Actively managing positions to prevent getting stuck
# - Dynamically adjusting pricing based on market conditions
# - Prioritizing immediate execution when favorable
#=========================================================================
"""
###TO DO LATER##
# - Analyze the log file and the csv file carefully to see if we can detect any anomalies
# - Try to record the trades we did only with hidden bots between iterations (timestamps)
# - Right now, trading works this way: At the start of each iteration, we observe the order book data for each product (after some bots have traded). 
# - Then, we send our orders to the exchange(matching engine) and record the orders.
#   After that, orders that get matched are filled, the rest or the ones that were partially filled are left in the order book.
#   At the end of the iteration, other bots trade with the remaining orders in the order book and our orders that do not get filled drop out of the order book, 
#   partially filled orders are left in the order book.
#   So the order looks like this each timestamp(iteration):  Bots -> Trader -> some other bots
#   there is x bots with priority over us and y bots with no priority over us, these numbers are constant across all iterations and products.
#   The problem with this is that we don't know which trades were done by the hidden bots.
#   We need to find a way to detect the hidden bots and predict their trades.
# - We can try to detect the hidden bots by analyzing the trades that happen between iterations.
#   If a trade happens between iterations, it is likely to be done by a hidden bot.
#   We can then predict the trades of the hidden bots and trade against them.
#----------------------------
# Turns out we receive market_trades dictionary
# This variable contains all trades AFTER you on previous iteration and BEFORE you in current one
# there can be no trades on specific product. Data could come from previous iterations ( for example if a market trade received at timestamp 0 for product A, 
# it could be that this trade is still in market_trades for product A after n iterations if no other trades were made on product A)
# So we need to find a way utilize this data to our advantage.
# certain factors affect these hidden bots trades, but trades done on previous iterations do not affect current iteration trades (admin said this)
# Current iter order book data probably affect bot trades, so we need to find a way to predict the trades of the hidden bots.
# Todo 2: visualizing the exectued trades of hidden bots might help us figuring out the true fair price of each asset per iteration (since bots know the fair price)
# Also, a visualizing the histogram of bid ask orders might be useful



import json
from typing import Any, Dict, List
from abc import abstractmethod
from collections import deque

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        """Implement the strategy logic here"""
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        """Helper method to add a buy order"""
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"{self.symbol} BUY {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        """Helper method to add a sell order"""
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"{self.symbol} SELL {quantity}x{price}")

    def save(self) -> dict:
        """Return data to be saved between rounds"""
        return {}

    def load(self, data: dict) -> None:
        """Load saved data from previous rounds"""
        pass


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window = deque()
        self.window_size = 10
        
    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        """Determine the fair price for this product"""
        raise NotImplementedError()
        
    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
        
        order_depth = state.order_depths[self.symbol]
        
        # Skip if no orders on either side
        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.print(f"Skipping {self.symbol}: No orders on one or both sides of the book")
            return
            
        # Sort orders for processing
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # Get position and calculate available space
        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position
        
        # Calculate true value
        true_value = self.get_true_value(state)
        logger.print(f"{self.symbol} - true value: {true_value}, position: {position}/{self.position_limit}")
        
        # Track positions at limits to detect if we're stuck
        self.window.append(abs(position) == self.position_limit)
        if len(self.window) > self.window_size:
            self.window.popleft()
            
        # Determine if we need to take more aggressive action to manage position
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)
        
        # Adjust max buy price and min sell price based on position
        max_buy_price = true_value - 1 if position > self.position_limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.position_limit * 0.5 else true_value
        
        logger.print(f"{self.symbol} - max buy: {max_buy_price}, min sell: {min_sell_price}")
        
        # BUYING LOGIC: First take existing favorable sell orders
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
                
        # If we're stuck at limit, get more aggressive with buying
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity
            
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity
            
        # Place remaining buy orders at competitive price
        if to_buy > 0:
            # Find the most popular buy price point
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] if buy_orders else (true_value - 2)
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)
            
        # SELLING LOGIC: First take existing favorable buy orders
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
                
        # If we're stuck at limit, get more aggressive with selling
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity
            
        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity
            
        # Place remaining sell orders at competitive price
        if to_sell > 0:
            # Find the most popular sell price point
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else (true_value + 2)
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)
            
    def save(self) -> dict:
        return {"window": list(self.window)}
        
    def load(self, data: dict) -> None:
        if data and "window" in data:
            self.window = deque(data["window"])


class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # For RAINFOREST_RESIN, we assume a stable true value of 10000
        # This is similar to the competitor's Amethysts strategy
        return 10000


class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # For KELP, we calculate true value dynamically based on market data
        # This is similar to the competitor's Starfruit strategy
        order_depth = state.order_depths[self.symbol]
        
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # Find the most popular price points (highest volume)
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] if buy_orders else 0
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else 0
        
        # If we have valid prices, use their midpoint
        if popular_buy_price > 0 and popular_sell_price > 0:
            return round((popular_buy_price + popular_sell_price) / 2)
        
        # Fallback to best bid/ask midpoint
        best_bid = buy_orders[0][0] if buy_orders else 0
        best_ask = sell_orders[0][0] if sell_orders else 0
        
        if best_bid > 0 and best_ask > 0:
            return round((best_bid + best_ask) / 2)
            
        # Last resort: just use midpoint of available orders
        return round((best_bid + best_ask) / 2) if best_bid > 0 or best_ask > 0 else 0


class Trader:
    def __init__(self):
        # Define position limits for each product
        self.position_limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50
        }
        
        # Create strategy instances for each product
        self.strategies = {
            "KELP": KelpStrategy("KELP", self.position_limits["KELP"]),
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", self.position_limits["RAINFOREST_RESIN"])
        }
        
        self.iteration = 0
        
    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main method called by the backtester
        """
        self.iteration += 1
        
        # Initialize the result dictionary
        result = {}
        
        # Create logs dictionary for tracking metrics
        logs_dict = {
            "iteration": self.iteration,
            "timestamp": state.timestamp,
            "products": list(state.order_depths.keys()),
            "positions": state.position
        }
        
        # Log basic information
        logger.print(f"Iteration: {self.iteration}, Timestamp: {state.timestamp}")
        logger.print(f"Products: {list(state.order_depths.keys())}")
        logger.print(f"Positions: {state.position}")
        
        # Load previous trading data if available
        old_trader_data = json.loads(state.traderData) if state.traderData and state.traderData != "" else {}
        new_trader_data = {}
        
        # Process each product using the appropriate strategy
        for product, strategy in self.strategies.items():
            if product in old_trader_data:
                strategy.load(old_trader_data[product])
                
            if product in state.order_depths:
                orders = strategy.run(state)
                if orders:
                    result[product] = orders
                    
            new_trader_data[product] = strategy.save()
            
        # Prepare trader data to save state between rounds
        trader_data = json.dumps({**logs_dict, **new_trader_data})
        
        # No conversions in this implementation
        conversions = 0
        
        # Flush logs for visualization
        logger.flush(state, result, conversions, trader_data)
                
        return result, conversions, trader_data
