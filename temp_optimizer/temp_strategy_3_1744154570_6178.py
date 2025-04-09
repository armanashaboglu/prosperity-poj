# Create strategy/round1/round1_v2.py

import json
from typing import Any, Dict, List, Tuple, Deque # Added Deque
from abc import abstractmethod
from collections import deque
import math
import copy # For deep copying order depths if needed

# Assuming datamodel.py contains these - adjust import if needed
from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# --- Logger Class (Copied from working version) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Slightly reduced for safety margin

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Creates a dummy state and orders to calculate base length without sensitive data
        dummy_state_data = [ state.timestamp, "", [], {}, [], [], {}, [] ]
        dummy_orders = []

        try:
            base_json = self.to_json([dummy_state_data, dummy_orders, conversions, "", ""])
            base_length = len(base_json)

            # Calculate max length for traderData, new_trader_data, and logs
            max_item_length = (self.max_log_length - base_length) // 3
            if max_item_length < 0: max_item_length = 0

            # Prepare the actual data payload, truncating as needed
            compressed_state = self.compress_state(state, self.truncate(state.traderData if state.traderData else "", max_item_length))
            compressed_orders = self.compress_orders(orders)
            truncated_trader_data = self.truncate(trader_data, max_item_length)
            truncated_logs = self.truncate(self.logs, max_item_length)

            full_log_entry = [
                compressed_state,
                compressed_orders,
                conversions,
                truncated_trader_data,
                truncated_logs,
            ]

            print(self.to_json(full_log_entry))

        except Exception as e:
            print(f"Error during log flushing: {e}")
            # Fallback to printing minimal info if encoding fails
            print(json.dumps([state.timestamp, conversions, f"Log Flush Error: {e}"]))


        self.logs = "" # Reset logs

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
         # Ensure order depths are compressed correctly (dictionaries)
         compressed_order_depths = self.compress_order_depths(state.order_depths)
         return [
             state.timestamp,
             trader_data, # Already truncated
             self.compress_listings(state.listings),
             compressed_order_depths, # Pass the correctly compressed dict
             self.compress_trades(state.own_trades),
             self.compress_trades(state.market_trades),
             state.position,
             self.compress_observations(state.observations),
         ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings and isinstance(listings, dict):
            for listing in listings.values():
                if listing:
                     compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths and isinstance(order_depths, dict):
            for symbol, order_depth in order_depths.items():
                # Ensure we are storing the dictionaries directly, not lists of tuples
                 if order_depth and isinstance(order_depth, OrderDepth):
                     compressed[symbol] = [
                         order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {},
                         order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                     ]
        return compressed


    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades and isinstance(trades, dict):
            for arr in trades.values():
                 if arr and isinstance(arr, (list, tuple)):
                    for trade in arr:
                        if trade:
                            compressed.append([
                                getattr(trade, 'symbol', None),
                                getattr(trade, 'price', None),
                                getattr(trade, 'quantity', None),
                                getattr(trade, 'buyer', None),
                                getattr(trade, 'seller', None),
                                getattr(trade, 'timestamp', None),
                            ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        plain_obs = {}
        if observations:
            if hasattr(observations, 'conversionObservations') and isinstance(observations.conversionObservations, dict):
                for product, observation in observations.conversionObservations.items():
                    if observation:
                         conversion_observations[product] = [
                             getattr(observation, 'bidPrice', None), getattr(observation, 'askPrice', None),
                             getattr(observation, 'transportFees', None), getattr(observation, 'exportTariff', None),
                             getattr(observation, 'importTariff', None), getattr(observation, 'sunlightIndex', None), # Removed SugarPrice as it might not always be present
                             # getattr(observation, 'humidity', None) # Example if humidity is added later
                         ]
            if hasattr(observations, 'plainValueObservations') and isinstance(observations.plainValueObservations, dict):
                 plain_obs = observations.plainValueObservations

        return [plain_obs, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders and isinstance(orders, dict):
            for arr in orders.values():
                if arr and isinstance(arr, (list, tuple)):
                    for order in arr:
                         if order:
                             compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        # Use default=str to handle potential non-serializable types gracefully
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"), default=str)


    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): value = str(value) # Ensure string
        if len(value) <= max_length:
            return value
        # Ensure max_length is not negative after subtracting 3
        if max_length < 3: return value[:max_length] # Just truncate if space is very limited
        return value[: max_length - 3] + "..."

logger = Logger()

# --- Constants and Parameters ---
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

# --- Updated PARAMS to match prototype.py logic ---
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000
    },
    Product.KELP: {
        "ema_alpha": 0.1,
        "imbalance_levels": 2,
        "imbalance_factor": 0.5,
        "vol_threshold": 4.0,
        "vol_spread_add": 2,
        "vol_size_divisor": 3,
        "inv_factor": 0.4,
        "base_spread": 1,
        "min_spread": 1,
        "max_spread": 10,
        "inventory_warn_level_spread": 36,
        "inv_spread_add": 1,
        "base_size": 17,
        "inventory_warn_level_size": 35,
        "inv_size_factor": 0.15000000000000002
    },
    Product.SQUID_INK: {
        "ema_alpha": 0.5,
        "imbalance_levels": 10,
        "imbalance_factor": 0.8,
        "vol_threshold": 2.5,
        "vol_spread_add": 3,
        "vol_size_divisor": 3,
        "inv_factor": 0.45,
        "base_spread": 2,
        "min_spread": 1,
        "max_spread": 10,
        "inventory_warn_level_spread": 39,
        "inv_spread_add": 0,
        "base_size": 11,
        "inventory_warn_level_size": 38,
        "inv_size_factor": 0.15000000000000002
    }
}
# --- End Updated PARAMS ---

# --- Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []
        # Get global params, ensure defaults match the new PARAMS["global"] if needed
        self.global_params = PARAMS.get("global", {
            "window_size": 4, "soft_limit_factor": 0.5, "skew_offset": 1,
            "soft_liq_offset": 2, "hard_liq_offset": 0, "liq_pct": 0.5, "make_offset": 1
        })

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def _place_buy(self, price: int, quantity: int) -> None:
        if quantity <= 0: return
        price = int(round(price)) # Ensure integer price
        quantity = int(quantity) # Ensure integer quantity
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}")

    def _place_sell(self, price: int, quantity: int) -> None:
        if quantity <= 0: return
        price = int(round(price)) # Ensure integer price
        quantity = int(quantity) # Ensure integer quantity
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}")

    # Default save/load, subclasses can override
    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

# --- Market Making Strategy ---
class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = params # Product-specific params
        # Ensure window maxlen uses the global param correctly
        self.window: Deque[bool] = deque(maxlen=self.global_params["window_size"])

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]

        # Ensure order books are dictionaries before proceeding
        buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}

        if not buy_orders_dict and not sell_orders_dict:
            logger.print(f"No orders for {self.symbol}, skipping.")
            return

        # --- Get position, capacity, true value (Same as prototype) ---
        position = state.position.get(self.symbol, 0)
        # Use variable names matching prototype.py for clarity during replication
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        true_value = self.get_true_value(state)
        if true_value is None:
            logger.print(f"Could not determine true value for {self.symbol}, skipping.")
            return

        logger.print(f"{self.symbol} -> TV: {true_value:.2f}, Pos: {position}/{self.position_limit}, BuyCap: {to_buy}, SellCap: {to_sell}")

        # --- Position Management & Liquidation Flags (Same as prototype) ---
        self.window.append(abs(position) == self.position_limit)
        # Note: prototype.py used self.window_size directly, we use global_params["window_size"]
        # but they should be the same value (4) based on PARAMS.
        window_size = self.global_params["window_size"]
        is_full_window = len(self.window) == window_size
        stuck_count = sum(self.window)
        soft_liquidate = is_full_window and stuck_count >= window_size / 2 and self.window[-1]
        hard_liquidate = is_full_window and stuck_count == window_size

        # --- Calculate Buy/Sell Boundaries (Same as prototype) ---
        # Use the 50% threshold from prototype explicitly
        soft_limit_threshold = self.position_limit * 0.5
        max_buy_price = true_value - 1 if position > soft_limit_threshold else true_value
        min_sell_price = true_value + 1 if position < -soft_limit_threshold else true_value
        logger.print(f"{self.symbol} - max_buy_price(take/make): {max_buy_price:.2f}, min_sell_price(take/make): {min_sell_price:.2f}")


        # --- Phase 1: Take Existing Favorable Orders (prototype logic) ---
        # Convert dicts to sorted lists like prototype does *for this phase*
        sell_orders_list = sorted(sell_orders_dict.items()) if sell_orders_dict else []
        buy_orders_list = sorted(buy_orders_dict.items(), reverse=True) if buy_orders_dict else []

        # Buy from market (take asks)
        for price, volume in sell_orders_list:
            volume = -volume # Make positive
            # Check capacity and price boundary
            if to_buy > 0 and price <= max_buy_price:
                qty_to_take = min(to_buy, volume)
                # Use _place_buy which logs correctly
                self._place_buy(price, qty_to_take)
                to_buy -= qty_to_take
            # No else break needed here, keep checking if capacity remains

        # Sell to market (take bids)
        for price, volume in buy_orders_list:
             # Check capacity and price boundary
             if to_sell > 0 and price >= min_sell_price:
                 qty_to_take = min(to_sell, volume)
                 # Use _place_sell which logs correctly
                 self._place_sell(price, qty_to_take)
                 to_sell -= qty_to_take
            # No else break needed here

        logger.print(f"{self.symbol} Phase 1 DONE - RemBuyCap: {to_buy}, RemSellCap: {to_sell}")

        # --- Phase 2: Liquidation if Stuck (prototype logic) ---
        # Note: Using integer division // like prototype
        # Using hard/soft liq offsets from global_params (0 and 2, matching prototype)
        hard_liq_offset = self.global_params["hard_liq_offset"] # Should be 0
        soft_liq_offset = self.global_params["soft_liq_offset"] # Should be 2

        if to_buy > 0 and hard_liquidate: # Stuck short, need to buy
            quantity = to_buy // 2
            # prototype uses true_value +/- 0
            liq_price = true_value + hard_liq_offset
            self._place_buy(liq_price, quantity)
            to_buy -= quantity
            logger.print(f"{self.symbol} Hard Liq Buy {quantity}x{liq_price:.0f}")
        elif to_buy > 0 and soft_liquidate: # Stuck short, need to buy
            quantity = to_buy // 2
            # prototype uses true_value - 2 (mistake?) / + 2
            # Replicating prototype: buy at true_value - 2 if soft liquidating short? <= This seems wrong in prototype, should be TV+2. Let's use TV+offset
            liq_price = true_value + soft_liq_offset # Buy higher
            self._place_buy(liq_price, quantity)
            to_buy -= quantity
            logger.print(f"{self.symbol} Soft Liq Buy {quantity}x{liq_price:.0f}")

        if to_sell > 0 and hard_liquidate: # Stuck long, need to sell
            quantity = to_sell // 2
            # prototype uses true_value +/- 0
            liq_price = true_value - hard_liq_offset
            self._place_sell(liq_price, quantity)
            to_sell -= quantity
            logger.print(f"{self.symbol} Hard Liq Sell {quantity}x{liq_price:.0f}")
        elif to_sell > 0 and soft_liquidate: # Stuck long, need to sell
            quantity = to_sell // 2
            # prototype uses true_value + 2 (mistake?) / - 2
            # Replicating prototype: sell at true_value + 2 if soft liquidating long? <= This seems wrong in prototype, should be TV-2. Let's use TV-offset
            liq_price = true_value - soft_liq_offset # Sell lower
            self._place_sell(liq_price, quantity)
            to_sell -= quantity
            logger.print(f"{self.symbol} Soft Liq Sell {quantity}x{liq_price:.0f}")

        logger.print(f"{self.symbol} Phase 2 DONE - RemBuyCap: {to_buy}, RemSellCap: {to_sell}")

        # --- Phase 3: Place New Market Making Orders (prototype logic) ---
        if to_buy > 0:
            # Find the most popular buy price point (highest volume)
            # Need to re-fetch/re-sort buy_orders_list as prototype did (implicitly uses original state)
            buy_orders_list_for_make = sorted(buy_orders_dict.items(), reverse=True) if buy_orders_dict else []
            # Default fallback from prototype: true_value - 2
            popular_buy_price = max(buy_orders_list_for_make, key=lambda tup: tup[1])[0] if buy_orders_list_for_make else (true_value - 2)
            # Calculate make price (min of boundary and popular+1)
            make_price = min(max_buy_price, popular_buy_price + 1)
            self._place_buy(make_price, to_buy)

        if to_sell > 0:
            # Find the most popular sell price point (highest *negative* volume = lowest abs vol?) <= prototype key is tup[1], which is negative for sells. min() finds lowest neg value = highest abs vol. CORRECT.
            # Need to re-fetch/re-sort sell_orders_list as prototype did
            sell_orders_list_for_make = sorted(sell_orders_dict.items()) if sell_orders_dict else []
            # Default fallback from prototype: true_value + 2
            # In prototype: min(sell_orders, key=lambda tup: tup[1]) -> finds price with most negative volume (largest sell volume)
            popular_sell_price = min(sell_orders_list_for_make, key=lambda tup: tup[1])[0] if sell_orders_list_for_make else (true_value + 2)
            # Calculate make price (max of boundary and popular-1)
            make_price = max(min_sell_price, popular_sell_price - 1)
            self._place_sell(make_price, to_sell)

        logger.print(f"{self.symbol} Phase 3 DONE - Placed remaining capacity.")

    # _get_rsi_skew hook remains for structure, but unused by current strategies
    def _get_rsi_skew(self, state: TradingState) -> int:
        return 0

    def save(self) -> dict:
        # Save position window
        return {"window": list(self.window)}

    def load(self, data: dict) -> None:
        # Load position window
        if data and "window" in data and isinstance(data["window"], list):
            loaded_window = data["window"]
            # Use the maxlen from global_params when loading
            start_index = max(0, len(loaded_window) - self.global_params["window_size"])
            self.window.clear()
            self.window.extend(loaded_window[start_index:])


# --- Product Specific Strategies ---

# Changed RainforestResinStrategy to inherit from MarketMakingStrategy
class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float | None:
        # Use configured fair value
        return self.params.get("fair_value", 10000)
    # Removed the overridden _make_orders method - now uses base class logic

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float | None:
        # Dynamic calculation based on most popular price points (highest volume)
        # Adjusted to exactly match prototype.py logic
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders: return None # No data

        # --- Replicate prototype.py logic exactly ---
        # Sort orders by price first
        buy_orders_list = sorted(buy_orders.items(), reverse=True) if buy_orders else []
        sell_orders_list = sorted(sell_orders.items()) if sell_orders else []

        # Find popular price using max/min on sorted list (handles tie-breaking like prototype)
        popular_buy_price = max(buy_orders_list, key=lambda tup: tup[1])[0] if buy_orders_list else 0
        popular_sell_price = min(sell_orders_list, key=lambda tup: tup[1])[0] if sell_orders_list else 0
        # --- End replication ---

        # Calculate midpoint, with fallbacks - same as before
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            return (popular_buy_price + popular_sell_price) / 2.0
        else: # Fallback to best bid/ask midpoint
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / 2.0
            elif best_bid > 0: return float(best_bid)
            elif best_ask > 0: return float(best_ask)
            else: return None # Cannot determine value

# Changed SquidInkStrategy to remove RSI logic
class SquidInkStrategy(MarketMakingStrategy): # Inherits from MM directly
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        super().__init__(symbol, params, position_limit)
        # Removed RSI state variables: self.price_history, self.rsi_value

    def get_true_value(self, state: TradingState) -> float | None:
        # Use the same logic as KelpStrategy
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders: return None

        popular_buy_price = 0
        max_buy_vol = 0
        if buy_orders:
             for price, volume in buy_orders.items():
                 if volume > max_buy_vol:
                     max_buy_vol = volume
                     popular_buy_price = price
             if max_buy_vol <= 0:
                 popular_buy_price = max(buy_orders.keys()) if buy_orders else 0
        else:
             if not sell_orders: return None
             popular_buy_price = min(sell_orders.keys()) - 1

        popular_sell_price = 0
        max_sell_vol = 0
        if sell_orders:
             for price, volume in sell_orders.items():
                  abs_vol = abs(volume)
                  if abs_vol > max_sell_vol:
                     max_sell_vol = abs_vol
                     popular_sell_price = price
             if max_sell_vol <= 0:
                 popular_sell_price = min(sell_orders.keys()) if sell_orders else 0
        else:
             if not buy_orders: return None
             popular_sell_price = max(buy_orders.keys()) + 1

        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            return (popular_buy_price + popular_sell_price) / 2.0
        else:
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0: return (best_bid + best_ask) / 2.0
            elif best_bid > 0: return float(best_bid)
            elif best_ask > 0: return float(best_ask)
            else: return None

    # Removed _calculate_rsi and _get_rsi_skew methods

    def save(self) -> dict:
        # Only save base class state (window)
        return super().save()
        # Removed price_history and rsi_value saving

    def load(self, data: dict) -> None:
        # Only load base class state
        super().load(data)
        # Removed price_history and rsi_value loading


# --- Trader Class ---
class Trader:
    def __init__(self):
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50
        }

        # Create strategy instances, passing product-specific params
        self.strategies = {
            Product.KELP: KelpStrategy(
                Product.KELP, PARAMS[Product.KELP], self.position_limits[Product.KELP]
            ),
            Product.RAINFOREST_RESIN: RainforestResinStrategy( # Now uses base MM logic
                Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN], self.position_limits[Product.RAINFOREST_RESIN]
            ),
            # Product.SQUID_INK: SquidInkStrategy( # Now uses Kelp-like logic without RSI
            #     Product.SQUID_INK, PARAMS[Product.SQUID_INK], self.position_limits[Product.SQUID_INK]
            # )
        }
        logger.print("Trader Initialized (v2 - prototype aligned) with strategies for:", list(self.strategies.keys()))

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        logger.print(f"--- Round 1 V2 (Proto Aligned) | Timestamp: {state.timestamp} ---") # Updated log message slightly
        result: Dict[Symbol, List[Order]] = {}
        trader_data_for_next_round = {}
        conversions = 0

        try:
            loaded_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e:
            logger.print(f"Error decoding traderData: {e}. Starting fresh.")
            loaded_data = {}

        for product, strategy in self.strategies.items():
            product_state = loaded_data.get(product, {})
            # Ensure loaded state is dict before passing
            if isinstance(product_state, dict):
                 strategy.load(product_state)
            else:
                 logger.print(f"Warning: Invalid state data for {product}, ignoring.")
                 strategy.load({}) # Load default empty state

            try:
                orders = strategy.run(state)
                if orders:
                    result[product] = orders
            except Exception as e:
                logger.print(f"*** ERROR running strategy for {product} at timestamp {state.timestamp}: {e} ***")
                import traceback
                logger.print(traceback.format_exc())

            trader_data_for_next_round[product] = strategy.save()

        try:
            # Use default=str for robustness during encoding
            traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"), default=str)
        except Exception as e:
             logger.print(f"Error encoding traderData: {e}. Sending empty data.")
             traderData_encoded = "{}"

        logger.flush(state, result, conversions, traderData_encoded)

        return result, conversions, traderData_encoded
