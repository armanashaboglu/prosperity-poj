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
import copy

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder


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


# === START NEW PARAMS STRUCTURE ===
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000
    },
    Product.KELP: {
        "ema_alpha": 0.7000000000000001,
        "imbalance_levels": 3,
        "imbalance_factor": 1.0,
        "vol_threshold": 1.5,
        "vol_spread_add": 3,
        "vol_size_divisor": 1,
        "inv_factor": 0.2,
        "base_spread": 2,
        "min_spread": 1,
        "max_spread": 8,
        "inventory_warn_level_spread": 29,
        "inv_spread_add": 2,
        "base_size": 20,
        "inventory_warn_level_size": 36,
        "inv_size_factor": 0.05
    },
    Product.SQUID_INK: {
        "ema_alpha": 0.9,
        "imbalance_levels": 5,
        "imbalance_factor": 0.30000000000000004,
        "vol_threshold": 2.0,
        "vol_spread_add": 3,
        "vol_size_divisor": 1,
        "inv_factor": 0.2,
        "base_spread": 1,
        "min_spread": 1,
        "max_spread": 15,
        "inventory_warn_level_spread": 45,
        "inv_spread_add": 0,
        "base_size": 15,
        "inventory_warn_level_size": 34,
        "inv_size_factor": 0.15000000000000002
    }
}
# === END NEW PARAMS STRUCTURE ===


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
        self.window_size = 4
        
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

    # Override act to implement the simple Take + Make strategy (with adjustments)
    def act(self, state: TradingState) -> None:
        self.orders = [] # Clear orders specific to this strategy run
        true_value = 10000

        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return # No orders for this product

        # Ensure order books are dictionaries & Create deep copies for simulation
        initial_buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        initial_sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
        sim_buy_orders = copy.deepcopy(initial_buy_orders)
        sim_sell_orders = copy.deepcopy(initial_sell_orders)

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        logger.print(f"{self.symbol} (DynMM) -> TV: {true_value}, Pos: {position}/{self.position_limit}, BuyCap: {to_buy}, SellCap: {to_sell}")

        # Phase 1: Take orders strictly better than or equal to fair value
        # Buy from asks <= 10000
        asks_sorted = sorted(sim_sell_orders.items())
        for price, volume in asks_sorted:
            if to_buy <= 0: break # No more capacity
            volume = -volume # Make positive
            if price <= true_value:
                qty_to_take = min(to_buy, volume)
                self.buy(price, qty_to_take)
                to_buy -= qty_to_take
                # Update simulated book
                sim_sell_orders[price] += qty_to_take # Add taken qty (becomes less negative)
                if sim_sell_orders[price] == 0:
                    del sim_sell_orders[price]
            else:
                break # Price too high

        # Sell to bids >= 10000
        bids_sorted = sorted(sim_buy_orders.items(), reverse=True)
        for price, volume in bids_sorted:
            if to_sell <= 0: break # No more capacity
            if price >= true_value:
                qty_to_take = min(to_sell, volume)
                self.sell(price, qty_to_take)
                to_sell -= qty_to_take
                # Update simulated book
                sim_buy_orders[price] -= qty_to_take
                if sim_buy_orders[price] == 0:
                    del sim_buy_orders[price]
            else:
                break # Price too low

        logger.print(f"{self.symbol} (DynMM) Phase 1 DONE - RemBuyCap: {to_buy}, RemSellCap: {to_sell}")

        # Phase 2: Make markets based on remaining book (sim_buy_orders, sim_sell_orders)

        # Determine Bid Price
        make_bid_price = true_value - 1 # Default bid
        bids_below_10k = {p: v for p, v in sim_buy_orders.items() if p < true_value}
        if bids_below_10k:
            best_bid_after_take = max(bids_below_10k.keys())
            best_bid_vol = bids_below_10k[best_bid_after_take]
            # Pennying/Joining logic (example: join if volume <= 5)
            if best_bid_vol <= 5:
                 make_bid_price = best_bid_after_take # Join small volume
            else:
                 make_bid_price = best_bid_after_take + 1 # Penny large volume


        # Determine Ask Price
        make_ask_price = true_value + 1 # Default ask
        asks_above_10k = {p: v for p, v in sim_sell_orders.items() if p > true_value}
        if asks_above_10k:
            best_ask_after_take = min(asks_above_10k.keys())
            best_ask_vol = abs(asks_above_10k[best_ask_after_take])
             # Pennying/Joining logic
            if best_ask_vol <= 5:
                make_ask_price = best_ask_after_take # Join small volume
            else:
                make_ask_price = best_ask_after_take - 1 # Penny large volume

        # Avoid specific undesirable price levels
        if make_bid_price == 9999: make_bid_price = 9998
        elif make_bid_price == 9997: make_bid_price = 9996
        if make_ask_price == 10001: make_ask_price = 10002
        elif make_ask_price == 10003: make_ask_price = 10004

        # Ensure bid < ask
        if make_bid_price >= make_ask_price:
            logger.print(f"Warning: Bid >= Ask ({make_bid_price} >= {make_ask_price}) after adjustments, fixing.")
            make_ask_price = make_bid_price + 1
            # Re-check avoidance after fixing inversion
            if make_ask_price == 10001: make_ask_price = 10002
            elif make_ask_price == 10003: make_ask_price = 10004

        # Place Make Orders
        if to_buy > 0:
            self.buy(make_bid_price, to_buy)
        if to_sell > 0:
            self.sell(make_ask_price, to_sell)

        logger.print(f"{self.symbol} (DynMM) Phase 2 DONE - Placed Make orders at {make_bid_price}/{make_ask_price}")


class KelpStrategy(Strategy): # Inherit directly from base Strategy
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {}) # Load KELP specific params
        if not self.params:
             logger.print(f"ERROR: Parameters for {self.symbol} not found in PARAMS.")
             # Handle error state - maybe raise exception or use defaults
             self.params = {} # Use empty dict to avoid errors, but strategy will fail

        # Initialize state variables
        self.prev_ema: float | None = None
        self.prev_mid: float | None = None
        # Add other state if needed (e.g., recent prices for better volatility)

    def act(self, state: TradingState) -> None:
        self.orders = [] # Reset orders
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            logger.print(f"No order depth for {self.symbol}")
            return

        buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}

        # --- 1. Read current market state ---
        best_bid_price = max(buy_orders.keys()) if buy_orders else None
        best_ask_price = min(sell_orders.keys()) if sell_orders else None

        if best_bid_price is None or best_ask_price is None:
             logger.print(f"Missing best bid or ask for {self.symbol}")
             # Cannot calculate mid, potentially skip or use EMA if available
             if self.prev_ema is None: return # Cannot proceed without a price reference
             mid_price = self.prev_ema # Fallback to last EMA
             current_spread = self.params.get("base_spread", 2) # Fallback spread
             logger.print(f"Using prev EMA {mid_price:.2f} as mid fallback.")
        else:
             mid_price = (best_bid_price + best_ask_price) / 2
             current_spread = best_ask_price - best_bid_price

        position = state.position.get(self.symbol, 0)
        logger.print(f"{self.symbol} -> Mid: {mid_price:.2f}, Spread: {current_spread}, Pos: {position}/{self.position_limit}")

        # --- 2. Update dynamic metrics (EMA) ---
        alpha = self.params.get("ema_alpha", 0.3)
        if self.prev_ema is None:
             ema_price = mid_price # Initialize EMA with current mid
        else:
             ema_price = alpha * mid_price + (1 - alpha) * self.prev_ema
        logger.print(f"{self.symbol} -> EMA: {ema_price:.2f} (Prev: {self.prev_ema})")
        # Will store ema_price in self.prev_ema at the end for next iteration

        # --- 3. Calculate imbalance ---
        imbalance = 0.0
        imb_levels = self.params.get("imbalance_levels", 5)
        # Sort bids descending, asks ascending
        sorted_bids = sorted(buy_orders.items(), reverse=True)
        sorted_asks = sorted(sell_orders.items())
        total_bid_vol = sum(vol for _, vol in sorted_bids[:imb_levels])
        total_ask_vol = sum(-vol for _, vol in sorted_asks[:imb_levels]) # use absolute volume

        if total_bid_vol + total_ask_vol > 0:
             imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        logger.print(f"{self.symbol} -> Imbalance ({imb_levels} levels): {imbalance:.3f} (BidVol: {total_bid_vol}, AskVol: {total_ask_vol})")


        # --- 4. Estimate short-term volatility (Simple 1-tick move) ---
        is_volatile = False
        price_move = 0.0
        if self.prev_mid is not None:
            price_move = abs(mid_price - self.prev_mid)
            vol_threshold = self.params.get("vol_threshold", 2.0)
            if price_move > vol_threshold:
                is_volatile = True
        logger.print(f"{self.symbol} -> Price Move: {price_move:.2f}, Volatile: {is_volatile}")
        # Will store mid_price in self.prev_mid at the end

        # --- 5. Determine target price (reservation price) ---
        base_price = ema_price # Use EMA as the base fair value estimate
        inv_factor = self.params.get("inv_factor", 0.1)
        imbalance_factor = self.params.get("imbalance_factor", 0.2)

        inv_skew = inv_factor * position
        # Scale imbalance skew by current spread to make it relative
        imb_skew = imbalance_factor * imbalance * current_spread
        target_price = base_price - inv_skew + imb_skew
        logger.print(f"{self.symbol} -> Target Price: {target_price:.2f} (Base: {base_price:.2f}, InvSkew: {-inv_skew:.2f}, ImbSkew: {imb_skew:.2f})")


        # --- 6. Determine spread ---
        spread = self.params.get("base_spread", 2)
        vol_spread_add = self.params.get("vol_spread_add", 1)
        inv_warn_level_spread = self.params.get("inventory_warn_level_spread", 40)
        inv_spread_add = self.params.get("inv_spread_add", 1)
        min_spread = self.params.get("min_spread", 2)
        max_spread = self.params.get("max_spread", 8)

        if is_volatile:
             spread += vol_spread_add
             logger.print(f"{self.symbol} -> Spread Add (Vol): +{vol_spread_add}")
        if abs(position) > inv_warn_level_spread:
             spread += inv_spread_add
             logger.print(f"{self.symbol} -> Spread Add (Inv): +{inv_spread_add}")

        spread = max(min_spread, min(max_spread, spread))
        logger.print(f"{self.symbol} -> Final Spread: {spread}")


        # --- 7. Calculate quote prices ---
        # Use integer prices
        bid_quote = int(round(target_price - spread / 2))
        ask_quote = int(round(target_price + spread / 2))

        # Ensure they don't invert
        if bid_quote >= ask_quote:
             logger.print(f"Warning: Calculated quotes inverted ({bid_quote} >= {ask_quote}). Using BBO.")
             # Fallback to BBO if inversion happens
             bid_quote = best_bid_price if best_bid_price is not None else int(round(target_price - min_spread/2))
             ask_quote = best_ask_price if best_ask_price is not None else int(round(target_price + min_spread/2))
             # Ensure fallback doesn't invert
             if bid_quote >= ask_quote: ask_quote = bid_quote + 1


        # Ensure not crossing existing best prices too aggressively
        # Allow placing inside the spread, but not beyond the opposite best price
        if best_ask_price is not None:
            bid_quote = min(bid_quote, best_ask_price - 1)
        if best_bid_price is not None:
            ask_quote = max(ask_quote, best_bid_price + 1)

        # Final check for inversion after BBO constraints
        if bid_quote >= ask_quote:
             logger.print(f"Warning: Quotes inverted ({bid_quote} >= {ask_quote}) after BBO check. Adjusting.")
             # Could reset to target +/- min_spread/2 or just offset
             ask_quote = bid_quote + 1

        logger.print(f"{self.symbol} -> Quotes: Bid {bid_quote}, Ask {ask_quote}")


        # --- 8. Determine order sizes ---
        base_size = self.params.get("base_size", 10)
        inv_warn_level_size = self.params.get("inventory_warn_level_size", 30)
        inv_size_factor = self.params.get("inv_size_factor", 0.1)
        vol_size_divisor = self.params.get("vol_size_divisor", 2)

        bid_size = base_size
        ask_size = base_size

        # Adjust for position limits first
        if position + bid_size > self.position_limit:
             bid_size = max(0, self.position_limit - position)
        if position - ask_size < -self.position_limit:
             ask_size = max(0, self.position_limit + position)

        # Scale with inventory level (Reduce size proportionally beyond warning level)
        if position > inv_warn_level_size:
             reduction_factor = (position - inv_warn_level_size) * inv_size_factor
             bid_size = max(1, int(round(bid_size * (1 - reduction_factor)))) # Reduce buy size if long
        elif position < -inv_warn_level_size:
             reduction_factor = (abs(position) - inv_warn_level_size) * inv_size_factor
             ask_size = max(1, int(round(ask_size * (1 - reduction_factor)))) # Reduce sell size if short

        # Scale with volatility
        if is_volatile:
             bid_size = max(1, bid_size // vol_size_divisor)
             ask_size = max(1, ask_size // vol_size_divisor)

        # Final check against position limits after scaling
        bid_size = max(0, min(bid_size, self.position_limit - position))
        ask_size = max(0, min(ask_size, self.position_limit + position))

        logger.print(f"{self.symbol} -> Sizes: Bid {bid_size}, Ask {ask_size}")


        # --- 9. Place or update orders ---
        if bid_size > 0:
             # Use helper to place and log
             self.buy(bid_quote, bid_size) # buy() handles logging

        if ask_size > 0:
             # Use helper to place and log
             self.sell(ask_quote, ask_size) # sell() handles logging

        # --- Update State for next iteration ---
        self.prev_ema = ema_price
        self.prev_mid = mid_price

    def save(self) -> dict:
        # Save the state variables needed for the next round
        return {"prev_ema": self.prev_ema, "prev_mid": self.prev_mid}

    def load(self, data: dict) -> None:
        # Load state variables, handling potential missing keys
        self.prev_ema = data.get("prev_ema", None)
        self.prev_mid = data.get("prev_mid", None)
        # Log loaded state? Optional.
        # logger.print(f"Loaded state for {self.symbol}: prev_ema={self.prev_ema}, prev_mid={self.prev_mid}")


class Trader:
    def __init__(self):
        # Define position limits for each product
        self.position_limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50  # Add SQUID_INK limit
        }

        # Create strategy instances for each product
        # Ensure correct parameters are passed if PARAMS structure changed
        self.strategies = {
            "KELP": KelpStrategy("KELP", self.position_limits["KELP"]), # Now uses the new complex strategy
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", self.position_limits["RAINFOREST_RESIN"]),
            # SQUID_INK needs its own strategy or to be updated. Currently points to old MM strategy.
            # For now, let's create an instance of the *old* MarketMakingStrategy for SQUID_INK
            # to avoid breaking the Trader class structure.
            "SQUID_INK": KelpStrategy("SQUID_INK", self.position_limits["SQUID_INK"])
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