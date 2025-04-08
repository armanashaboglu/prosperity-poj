"""
#=========================================================================
# ALGORITHMIC TRADING STRATEGY - IMC PROSPERITY COMPETITION - Round 1 V1
#=========================================================================
#
# CONTEXT & GOALS:
# ---------------
# Strategy for Round 1, handling KELP, RAINFOREST_RESIN, and SQUID_INK.
# Based on the prototype strategy, incorporating insights from historical data analysis
# and improvements inspired by the v6 competitor strategy (Prosperity 2).
#
# KEY CHANGES from Prototype:
# 1. Added SQUID_INK strategy (initially mirroring KELP).
# 2. Adopted PARAMS dictionary structure for easier tuning.
# 3. Refined Trading Logic (3 phases: Take, Clear, Make):
#    - Take: Use `take_width`, add `prevent_adverse` for volatile products.
#    - Clear: Added explicit step to clear positions exceeding soft limits using `clear_width`.
#    - Make: Implemented pennying/joining logic (`disregard_edge`, `join_edge`, `default_edge`).
#             Added position skewing based on `soft_position_limit`.
# 4. RAINFOREST_RESIN Specific Logic: Avoid placing orders at ±1, ±3 from fair value (10000).
# 5. State Management: Simplified using `jsonpickle` directly in Trader class (removed Logger).
# 6. Fair Value: Stable 10000 for Resin, popular price midpoint for Kelp/Squid Ink.
#
# FUTURE IMPROVEMENTS:
# - Tune PARAMS based on backtesting/live results.
# - Consider more sophisticated fair value for Kelp/Squid Ink (e.g., incorporating `market_trades`).
# - Potentially add logic based on Squid Ink popular mid-price trigger.
# - Refine position clearing logic (e.g., dynamic amount based on `window` state).
#=========================================================================
"""

import json
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
from collections import deque
import jsonpickle # Use jsonpickle for state serialization
import math

# Ensure datamodel contents are accessible (adjust import if needed based on project structure)
# If datamodel.py is in the same directory or a standard location:
from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder
# If datamodel.py is in the parent directory (strategy/)
# import sys
# sys.path.append('..')
# from datamodel import OrderDepth, UserId, TradingState, Order, Symbol


# --- Logger Class (Use exact prototype version) ---
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
        if max_item_length < 0: max_item_length = 0 # Ensure non-negative length

        print(
            self.to_json(
                [
                    # Pass truncated state.traderData here
                    self.compress_state(state, self.truncate(state.traderData if state.traderData else "", max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length), # Pass truncated new trader_data
                    self.truncate(self.logs, max_item_length), # Pass truncated logs
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
        # Check if trades is not None and is a dict before iterating
        if trades and isinstance(trades, dict):
            for arr in trades.values():
                 # Check if arr is not None and is a list/tuple before iterating
                 if arr and isinstance(arr, (list, tuple)):
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
        # Check if observations and conversionObservations exist and are dicts
        if observations and hasattr(observations, 'conversionObservations') and isinstance(observations.conversionObservations, dict):
            for product, observation in observations.conversionObservations.items():
                # Check if observation object itself is not None
                if observation:
                     conversion_observations[product] = [
                         # Use getattr for safety in case fields are missing
                         getattr(observation, 'bidPrice', None),
                         getattr(observation, 'askPrice', None),
                         getattr(observation, 'transportFees', None),
                         getattr(observation, 'exportTariff', None),
                         getattr(observation, 'importTariff', None),
                         getattr(observation, 'sugarPrice', None),
                         getattr(observation, 'sunlightIndex', None),
                     ]
        # Check if plainValueObservations exists
        plain_obs = observations.plainValueObservations if observations and hasattr(observations, 'plainValueObservations') else {}
        return [plain_obs, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        # Check if orders is not None and is a dict
        if orders and isinstance(orders, dict):
            for arr in orders.values():
                # Check if arr is not None and is a list/tuple
                if arr and isinstance(arr, (list, tuple)):
                    for order in arr:
                        compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        # Ensure value is a string before checking length
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


# Define Product constants for clarity
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

# Define PARAMS dictionary for strategy configuration
PARAMS = {
    Product.KELP: {
        "take_width": 1,        # Take orders within fair_value +/- this width
        "clear_width": 1,       # Clear positions aggressively at fair_value +/- this width
        "prevent_adverse": True,# Prevent taking large orders that might move the price against us
        "adverse_volume": 15,   # Volume threshold considered potentially adverse
        # Making parameters
        "disregard_edge": 1,    # Ignore levels within fair +/- this for pennying/joining
        "join_edge": 0,         # Join levels within fair +/- this (0 means never join, only penny)
        "default_edge": 2,      # Default edge if no levels to penny/join
        "soft_position_limit": 30, # Skew pricing when position exceeds this
        "window_size": 5        # Size of the window for tracking stuck positions
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,    # Assumed stable fair value
        "take_width": 0,        # Take orders AT or better than fair (<= 10000 or >= 10000)
        "clear_width": 1,       # Clear aggressively at 9999 / 10001 if needed
        "prevent_adverse": False,# Less relevant for stable product
        "adverse_volume": 0,
        # Making parameters - These are now overridden by the specialized _make_orders
        # "disregard_edge": 0, # Not used
        # "join_edge": 0,    # Not used
        # "default_edge": 1, # Not used (will make at 9999/10001)
        "soft_position_limit": 40, # Skew pricing only when very close to limit (reduce chance of crossing)
        "window_size": 5
    },
    Product.SQUID_INK: { # Start by mirroring KELP parameters
        "take_width": 1,
        "clear_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 15,
        # Making parameters
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 2,
        "soft_position_limit": 30,
        "window_size": 5
    },
}

class BaseStrategy:
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        self.symbol = symbol
        self.params = params
        self.position_limit = position_limit
        self.orders: List[Order] = []
        # Use deque for position window tracking
        self.position_window = deque(maxlen=params.get("window_size", 5))

    @abstractmethod
    def get_fair_value(self, state: TradingState) -> float | None:
        """Determine the fair price for this product"""
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        """Main logic runner for the strategy"""
        self.orders = [] # Reset orders for this iteration
        self.act(state)
        return self.orders

    def act(self, state: TradingState) -> None:
        """Implements the core Take, Clear, Make logic"""
        if self.symbol not in state.order_depths:
            logger.print(f"Warning: No order depth data for {self.symbol}") # Use logger
            return

        order_depth = state.order_depths[self.symbol]
        # Create copies to avoid modifying the original state unintentionally during simulation
        current_buy_orders = order_depth.buy_orders.copy() if isinstance(order_depth.buy_orders, dict) else {}
        current_sell_orders = order_depth.sell_orders.copy() if isinstance(order_depth.sell_orders, dict) else {}
        sim_order_depth = OrderDepth()
        sim_order_depth.buy_orders = current_buy_orders
        sim_order_depth.sell_orders = current_sell_orders

        position = state.position.get(self.symbol, 0)
        buy_order_volume = 0
        sell_order_volume = 0

        # 0. Update position window
        self.position_window.append(abs(position) == self.position_limit)
        # Determine stuck status (can be used in clearing/making)
        is_stuck_soft = len(self.position_window) == self.position_window.maxlen and sum(self.position_window) >= self.position_window.maxlen / 2 and self.position_window[-1]
        is_stuck_hard = len(self.position_window) == self.position_window.maxlen and all(self.position_window)

        # 1. Calculate Fair Value
        fair_value = self.get_fair_value(state)
        if fair_value is None:
            logger.print(f"Warning: Could not calculate fair value for {self.symbol}. Skipping.") # Use logger
            return
        logger.print(f"{self.symbol} - Fair Value: {fair_value:.2f}, Position: {position}/{self.position_limit}") # Use logger

        # --- Trading Phases ---
        # Phase 1: Take favorable orders
        take_orders, buy_order_volume, sell_order_volume = self._take_orders(
            sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume
        )
        self.orders.extend(take_orders)

        # Phase 2: Clear position if outside soft limits (aggressively)
        clear_orders, buy_order_volume, sell_order_volume = self._clear_orders(
            sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume, is_stuck_hard # Pass stuck status
        )
        self.orders.extend(clear_orders)

        # Phase 3: Make markets (place new resting orders)
        make_orders, _, _ = self._make_orders(
            sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume
        )
        self.orders.extend(make_orders)

    def _take_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        """Takes orders within the take_width"""
        orders = []
        take_width = self.params["take_width"]
        prevent_adverse = self.params.get("prevent_adverse", False)
        adverse_volume = self.params.get("adverse_volume", 0)
        position_limit = self.position_limit

        # Take Sells (Buy from market)
        if isinstance(order_depth.sell_orders, dict): # Check if dict before sorting
            sell_order_prices = sorted(order_depth.sell_orders.keys())
            for price in sell_order_prices:
                if price <= fair_value - take_width:
                    available_volume = -order_depth.sell_orders[price]
                    if prevent_adverse and available_volume > adverse_volume:
                        logger.print(f"Skipping adverse sell order at {price} ({available_volume} > {adverse_volume})") # Use logger
                        continue # Skip potentially adverse large orders

                    qty_to_buy = min(available_volume, position_limit - (position + buy_volume))
                    if qty_to_buy <= 0: break # Cannot buy more

                    logger.print(f"{self.symbol} - Taking Sell: {qty_to_buy}x{price}") # Use logger
                    orders.append(Order(self.symbol, price, qty_to_buy))
                    buy_volume += qty_to_buy
                    # Simulate update to order depth
                    order_depth.sell_orders[price] += qty_to_buy
                    if order_depth.sell_orders[price] == 0: del order_depth.sell_orders[price]
                else:
                    break # Prices are sorted, no need to check further

        # Take Buys (Sell to market)
        if isinstance(order_depth.buy_orders, dict): # Check if dict before sorting
             buy_order_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
             for price in buy_order_prices:
                 if price >= fair_value + take_width:
                     available_volume = order_depth.buy_orders[price]
                     if prevent_adverse and available_volume > adverse_volume:
                         logger.print(f"Skipping adverse buy order at {price} ({available_volume} > {adverse_volume})") # Use logger
                         continue # Skip potentially adverse large orders

                     qty_to_sell = min(available_volume, position_limit + (position - sell_volume))
                     if qty_to_sell <= 0: break # Cannot sell more

                     logger.print(f"{self.symbol} - Taking Buy: {qty_to_sell}x{price}") # Use logger
                     orders.append(Order(self.symbol, price, -qty_to_sell))
                     sell_volume += qty_to_sell
                     # Simulate update
                     order_depth.buy_orders[price] -= qty_to_sell
                     if order_depth.buy_orders[price] == 0: del order_depth.buy_orders[price]
                 else:
                     break # Prices are sorted

        return orders, buy_volume, sell_volume

    def _clear_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int, is_stuck: bool) -> tuple[List[Order], int, int]:
        """Clears position if it exceeds soft limit by taking aggressively"""
        orders = []
        soft_limit = self.params["soft_position_limit"]
        clear_width = self.params["clear_width"]
        position_after_take = position + buy_volume - sell_volume
        position_limit = self.position_limit

        # If long position exceeds soft limit, sell aggressively
        if position_after_take > soft_limit:
            qty_to_clear = position_after_take - soft_limit
            if is_stuck: qty_to_clear = position_after_take # Clear harder if stuck
            sell_price_target = round(fair_value - clear_width) # Sell lower to clear
            logger.print(f"{self.symbol} - Clearing Long Position: Attempting {qty_to_clear} @ >= {sell_price_target}") # Use logger

            if isinstance(order_depth.buy_orders, dict): # Check if dict
                buy_order_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                for price in buy_order_prices:
                    if price >= sell_price_target and qty_to_clear > 0:
                        available_volume = order_depth.buy_orders[price]
                        sell_now = min(qty_to_clear, available_volume, position_limit + (position - sell_volume))
                        if sell_now <= 0: continue

                        logger.print(f"{self.symbol} - Clearing Sell: {sell_now}x{price}") # Use logger
                        orders.append(Order(self.symbol, price, -sell_now))
                        sell_volume += sell_now
                        qty_to_clear -= sell_now
                        # Simulate update
                        order_depth.buy_orders[price] -= sell_now
                        if order_depth.buy_orders[price] == 0: del order_depth.buy_orders[price]
                    if qty_to_clear <= 0: break

        # If short position exceeds soft limit, buy aggressively
        elif position_after_take < -soft_limit:
            qty_to_clear = abs(position_after_take) - soft_limit
            if is_stuck: qty_to_clear = abs(position_after_take) # Clear harder if stuck
            buy_price_target = round(fair_value + clear_width) # Buy higher to clear
            logger.print(f"{self.symbol} - Clearing Short Position: Attempting {qty_to_clear} @ <= {buy_price_target}") # Use logger

            if isinstance(order_depth.sell_orders, dict): # Check if dict
                sell_order_prices = sorted(order_depth.sell_orders.keys())
                for price in sell_order_prices:
                    if price <= buy_price_target and qty_to_clear > 0:
                        available_volume = -order_depth.sell_orders[price]
                        buy_now = min(qty_to_clear, available_volume, position_limit - (position + buy_volume))
                        if buy_now <= 0: continue

                        logger.print(f"{self.symbol} - Clearing Buy: {buy_now}x{price}") # Use logger
                        orders.append(Order(self.symbol, price, buy_now))
                        buy_volume += buy_now
                        qty_to_clear -= buy_now
                        # Simulate update
                        order_depth.sell_orders[price] += buy_now
                        if order_depth.sell_orders[price] == 0: del order_depth.sell_orders[price]
                    if qty_to_clear <= 0: break

        return orders, buy_volume, sell_volume

    def _make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        """Places new bid and ask orders based on pennying/joining logic"""
        orders = []
        disregard_edge = self.params["disregard_edge"]
        join_edge = self.params["join_edge"]
        default_edge = self.params["default_edge"]
        soft_limit = self.params["soft_position_limit"]
        position_limit = self.position_limit

        # Find relevant price levels for pennying/joining
        # Ensure we use the *current* state of the order book after takes/clears
        asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge] if isinstance(order_depth.sell_orders, dict) else []
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge] if isinstance(order_depth.buy_orders, dict) else []


        best_ask_to_consider = min(asks_above_fair) if asks_above_fair else None
        best_bid_to_consider = max(bids_below_fair) if bids_below_fair else None

        # Determine Ask Price
        ask_price = round(fair_value + default_edge)
        if best_ask_to_consider is not None:
            if abs(best_ask_to_consider - fair_value) <= join_edge:
                ask_price = best_ask_to_consider  # Join
            else:
                ask_price = best_ask_to_consider - 1  # Penny

        # Determine Bid Price
        bid_price = round(fair_value - default_edge)
        if best_bid_to_consider is not None:
            if abs(fair_value - best_bid_to_consider) <= join_edge:
                bid_price = best_bid_to_consider # Join
            else:
                bid_price = best_bid_to_consider + 1 # Penny

        # Skew pricing based on position relative to soft limit
        # Position here should reflect the state *before* this making step
        current_position_after_clears = position + buy_volume - sell_volume # Position before making step
        if current_position_after_clears > soft_limit:
             bid_price -= 1 # Less willing to buy more
             ask_price -= 1 # More willing to sell
        elif current_position_after_clears < -soft_limit:
             bid_price += 1 # More willing to buy
             ask_price += 1 # Less willing to sell more


        # Ensure bid < ask
        bid_price = min(bid_price, ask_price - 1)

        # Place Buy Order
        # Calculate remaining buy capacity *after* takes and clears
        buy_capacity = position_limit - current_position_after_clears # Use current position
        if buy_capacity > 0:
            final_bid_price = self._adjust_price_level(bid_price) # Adjust for specific product rules
            logger.print(f"{self.symbol} - Making Buy: {buy_capacity}x{final_bid_price}") # Use logger
            orders.append(Order(self.symbol, final_bid_price, buy_capacity))
            # Don't update buy_volume here, it's for tracking fills in this iteration

        # Place Sell Order
        # Calculate remaining sell capacity *after* takes and clears
        sell_capacity = position_limit + current_position_after_clears # Use current position
        if sell_capacity > 0:
            final_ask_price = self._adjust_price_level(ask_price) # Adjust for specific product rules
            # Ensure final ask is still > final bid after adjustment
            if final_ask_price <= final_bid_price:
                 final_ask_price = final_bid_price + 1
                 # Re-adjust if the increment pushed it into a forbidden zone (e.g., Resin 10001)
                 final_ask_price = self._adjust_price_level(final_ask_price)

            # Ensure ask is still reasonable (e.g., not below fair value due to adjustments)
            # final_ask_price = max(final_ask_price, round(fair_value + 1))

            logger.print(f"{self.symbol} - Making Sell: {sell_capacity}x{final_ask_price}") # Use logger
            orders.append(Order(self.symbol, final_ask_price, -sell_capacity))
            # Don't update sell_volume here

        return orders, buy_volume, sell_volume

    def _adjust_price_level(self, price: int) -> int:
        """Hook for subclasses to adjust price levels (e.g., for Resin)"""
        return price # Default: no adjustment

    def save(self) -> Dict:
        """Return data to be saved between rounds"""
        # Save the position window as a list for JSON compatibility
        return {"position_window": list(self.position_window)}

    def load(self, data: Dict) -> None:
        """Load saved data from previous rounds"""
        if data and "position_window" in data and isinstance(data["position_window"], list):
            # Load window, ensuring maxlen is respected
            loaded_window = data["position_window"]
            # Start from the end of the loaded window to get the most recent states
            start_index = max(0, len(loaded_window) - self.position_window.maxlen)
            # Clear current deque and extend with loaded data
            self.position_window.clear()
            self.position_window.extend(loaded_window[start_index:])
            # logger.print(f"Loaded window for {self.symbol}: {list(self.position_window)}") # Reduced verbosity
        # else: # Reduced verbosity
            # logger.print(f"No valid window data found for {self.symbol} in loaded data.")

# --- Product Specific Strategies --- #

class RainforestResinStrategy(BaseStrategy):
    def get_fair_value(self, state: TradingState) -> float | None:
        return self.params.get("fair_value", 10000) # Use configured fair value

    def _adjust_price_level(self, price: int) -> int:
        # No adjustment needed for the simple 9999/10001 strategy
        return price

    # Override _make_orders for the specific Resin logic
    def _make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        """Places orders directly at 9999 and 10001 for Rainforest Resin."""
        orders = []
        position_limit = self.position_limit
        soft_limit = self.params["soft_position_limit"]
        make_bid_price = int(fair_value - 1) # Target 9999
        make_ask_price = int(fair_value + 1) # Target 10001

        # Calculate position *before* this making step
        current_position_after_clears = position + buy_volume - sell_volume

        # --- Apply Position Skewing --- (Optional but recommended)
        # If heavily long, slightly discourage buying more by lowering bid, encourage selling by lowering ask
        if current_position_after_clears > soft_limit:
            make_bid_price -= 1 # Bid at 9998 instead of 9999
            make_ask_price -= 1 # Ask at 10000 instead of 10001 (more likely to sell)
        # If heavily short, slightly discourage selling more by raising ask, encourage buying by raising bid
        elif current_position_after_clears < -soft_limit:
            make_bid_price += 1 # Bid at 10000 instead of 9999 (more likely to buy)
            make_ask_price += 1 # Ask at 10002 instead of 10001

        # Ensure bid < ask after skewing
        make_bid_price = min(make_bid_price, make_ask_price - 1)

        # Place Buy Order
        buy_capacity = position_limit - current_position_after_clears
        if buy_capacity > 0:
            logger.print(f"{self.symbol} - Making Buy: {buy_capacity}x{make_bid_price}")
            orders.append(Order(self.symbol, make_bid_price, buy_capacity))

        # Place Sell Order
        sell_capacity = position_limit + current_position_after_clears
        if sell_capacity > 0:
             # Ensure ask price is reasonable after skewing (e.g., not <= fair value)
             final_ask_price = max(make_ask_price, int(fair_value + 1))
             if final_ask_price <= make_bid_price: # Double check bid < ask
                  final_ask_price = make_bid_price + 1

             logger.print(f"{self.symbol} - Making Sell: {sell_capacity}x{final_ask_price}")
             orders.append(Order(self.symbol, final_ask_price, -sell_capacity))

        return orders, buy_volume, sell_volume


class VolatileStrategy(BaseStrategy):
    """Strategy for volatile products like KELP and SQUID_INK using popular price midpoint"""
    def get_fair_value(self, state: TradingState) -> float | None:
        if self.symbol not in state.order_depths:
            return None
        order_depth = state.order_depths[self.symbol]

        if not order_depth.buy_orders and not order_depth.sell_orders:
            # Attempt fallback using market trades if available?
            # For now, return None if book is empty
            return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        # Find most popular bid price (price with highest volume)
        popular_buy_price = 0
        max_buy_vol = 0
        if buy_orders:
            # Create list of (price, volume) tuples for easier sorting if needed
            buy_levels = list(buy_orders.items())
            # Sort primarily by volume (desc), secondarily by price (desc)
            buy_levels.sort(key=lambda x: (x[1], x[0]), reverse=True)
            popular_buy_price = buy_levels[0][0] # Price of the level with highest volume
            # for price, volume in buy_orders.items():
            #     if volume > max_buy_vol:
            #         max_buy_vol = volume
            #         popular_buy_price = price
            #     elif volume == max_buy_vol and price > popular_buy_price: # Tie break with higher price
            #         popular_buy_price = price
        else: # Handle case with no buy orders
            if not sell_orders: return None # No orders at all
            # Use best ask as a reference if no buys
            popular_buy_price = min(sell_orders.keys()) - 1 # Crude estimate


        # Find most popular sell price (price with highest absolute volume)
        popular_sell_price = 0
        max_sell_vol = 0
        if sell_orders:
            sell_levels = list(sell_orders.items())
            # Sort primarily by absolute volume (desc), secondarily by price (asc)
            sell_levels.sort(key=lambda x: (abs(x[1]), -x[0]), reverse=True)
            popular_sell_price = sell_levels[0][0]
            # for price, volume in sell_orders.items():
            #     abs_vol = abs(volume)
            #     if abs_vol > max_sell_vol:
            #         max_sell_vol = abs_vol
            #         popular_sell_price = price
            #     elif abs_vol == max_sell_vol and price < popular_sell_price: # Tie break with lower price
            #         popular_sell_price = price
        else: # Handle case with no sell orders
            if not buy_orders: return None # No orders at all
             # Use best bid as a reference if no sells
            popular_sell_price = max(buy_orders.keys()) + 1 # Crude estimate

        # Calculate fair value based on popular prices
        # Ensure popular prices are valid before calculating midpoint
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
             return (popular_buy_price + popular_sell_price) / 2.0
        else: # Fallback if popular prices are crossed or invalid
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / 2.0
            elif best_bid > 0:
                return best_bid # Use best bid if no asks
            elif best_ask > 0:
                return best_ask # Use best ask if no bids

        return None # Should not be reached if there are orders

class KelpStrategy(VolatileStrategy):
    pass # Inherits fair value calc from VolatileStrategy

class SquidInkStrategy(VolatileStrategy):
    pass # Inherits fair value calc from VolatileStrategy

# --- Trader Class --- #

class Trader:
    def __init__(self):
        # Define position limits for each product (Ensure these match competition rules)
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50
        }

        # Create strategy instances for each product, passing relevant params
        self.strategies = {
            Product.KELP: KelpStrategy(Product.KELP, PARAMS[Product.KELP], self.position_limits[Product.KELP]),
            Product.RAINFOREST_RESIN: RainforestResinStrategy(Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN], self.position_limits[Product.RAINFOREST_RESIN]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, PARAMS[Product.SQUID_INK], self.position_limits[Product.SQUID_INK])
        }

        logger.print("Trader Initialized with strategies for:", list(self.strategies.keys())) # Use logger

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main method called by the competition runner.
        """
        logger.print(f"--- Iteration {state.timestamp} ---") # Use logger
        result: Dict[Symbol, List[Order]] = {}
        trader_data_for_next_round = {}
        conversions = 0 # Initialize conversions

        # Load state from previous round if available
        try:
            # Use json.loads now as we expect standard JSON
            loaded_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(loaded_data, dict):
                logger.print("Warning: traderData was not a dict, ignoring.") # Use logger
                loaded_data = {}
        except Exception as e:
            logger.print(f"Error decoding traderData with json.loads: {e}. Starting fresh.") # Use logger
            loaded_data = {}

        # Run strategy for each product
        for product, strategy in self.strategies.items():
            logger.print(f"Running strategy for {product}...") # Use logger
            # Load strategy-specific state
            if product in loaded_data and isinstance(loaded_data[product], dict):
                strategy.load(loaded_data[product])
            else:
                 logger.print(f"No previous state found for {product}.") # Use logger

            # Execute strategy logic
            try:
                # Only run if product data exists in current state
                if product in state.order_depths:
                    orders = strategy.run(state)
                    if orders:
                        result[product] = orders
                        logger.print(f"  {product} Orders: {[(o.price, o.quantity) for o in orders]}") # Use logger
                else:
                     logger.print(f"  Skipping {product}, no order depth data in current state.") # Use logger
            except Exception as e:
                logger.print(f"*** ERROR running strategy for {product} at timestamp {state.timestamp}: {e} ***") # Use logger
                import traceback
                logger.print(traceback.format_exc()) # Log traceback

            # Save strategy state for next round
            trader_data_for_next_round[product] = strategy.save()

        # Encode the combined state for the next round using json.dumps
        try:
            # Convert the dictionary to a JSON string
            traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"))
        except Exception as e:
             logger.print(f"Error encoding traderData with json.dumps: {e}. Sending empty data.") # Use logger
             traderData_encoded = "{}"


        # --- Flush Logs ---
        # The traderData_encoded calculated above is the data for the *next* iteration.
        # state.traderData contains the data *from* the *previous* iteration.
        # logger.flush will handle compressing/truncating state, orders, conversions, new traderData, and logs.
        logger.flush(state, result, conversions, traderData_encoded)

        # Return the orders, conversions, and the new encoded trader data string for the runner
        # Note: The runner expects the *string* traderData, not the dictionary.
        return result, conversions, traderData_encoded 