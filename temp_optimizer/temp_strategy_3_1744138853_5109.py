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

PARAMS = {
    "global": {
        "window_size": 7,
        "soft_limit_factor": 0.7,
        "skew_offset": 1,
        "soft_liq_offset": 4,
        "hard_liq_offset": 1,
        "liq_pct": 0.2,
        "make_offset": 2
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000
    },
    Product.KELP: {},
    Product.SQUID_INK: {
        "rsi_period": 10,
        "rsi_oversold": 20,
        "rsi_overbought": 70,
        "rsi_skew_factor": 3,
        "price_history_len": 43
    }
}

# --- Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []
        # Get global params, default to reasonable values if not found
        self.global_params = PARAMS.get("global", {
            "window_size": 5, "soft_limit_factor": 0.7, "skew_offset": 1,
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
        self.window: Deque[bool] = deque(maxlen=self.global_params["window_size"])

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]
        # Make deep copies to simulate changes without affecting other strategies using the state
        sim_buy_orders = copy.deepcopy(order_depth.buy_orders) if isinstance(order_depth.buy_orders, dict) else {}
        sim_sell_orders = copy.deepcopy(order_depth.sell_orders) if isinstance(order_depth.sell_orders, dict) else {}

        if not sim_buy_orders and not sim_sell_orders:
            logger.print(f"No orders for {self.symbol}, skipping.")
            return

        position = state.position.get(self.symbol, 0)
        # Calculate remaining capacity *before* any actions this round
        remaining_buy_capacity = self.position_limit - position
        remaining_sell_capacity = self.position_limit + position

        true_value = self.get_true_value(state)
        if true_value is None:
            logger.print(f"Could not determine true value for {self.symbol}, skipping.")
            return

        logger.print(f"{self.symbol} -> TV: {true_value:.2f}, Pos: {position}/{self.position_limit}, BuyCap: {remaining_buy_capacity}, SellCap: {remaining_sell_capacity}")

        # --- Position Management ---
        self.window.append(abs(position) == self.position_limit)
        is_full_window = len(self.window) == self.window.maxlen
        stuck_count = sum(self.window)
        soft_limit_threshold = self.position_limit * self.global_params["soft_limit_factor"]

        is_stuck_soft = is_full_window and stuck_count >= self.window.maxlen / 2 and self.window[-1]
        is_stuck_hard = is_full_window and stuck_count == self.window.maxlen

        # --- Price Skewing ---
        # Base prices before skewing/liquidation adjustments
        base_max_buy_price = true_value
        base_min_sell_price = true_value

        # Initial skew based on position relative to soft limit
        skew = 0
        if position > soft_limit_threshold:
             skew = -self.global_params["skew_offset"] # Position is long -> less willing to buy, more willing to sell
        elif position < -soft_limit_threshold:
             skew = self.global_params["skew_offset"]  # Position is short -> more willing to buy, less willing to sell

        # Allow subclasses (like SquidInk) to add RSI-based skew
        rsi_skew = self._get_rsi_skew(state)
        total_skew = skew + rsi_skew

        current_max_buy_price = base_max_buy_price + total_skew
        current_min_sell_price = base_min_sell_price + total_skew

        logger.print(f"{self.symbol} PosSkew: {skew}, RsiSkew: {rsi_skew}, TotalSkew: {total_skew} -> BuyPx: {current_max_buy_price:.2f}, SellPx: {current_min_sell_price:.2f}")

        # --- Phase 1: Take Existing Favorable Orders ---
        # Simulate order book changes within this phase
        taken_buy_vol = 0
        taken_sell_vol = 0

        # Buy from market (take asks)
        asks = sorted(sim_sell_orders.items())
        for price, volume in asks:
            volume = -volume # Make positive
            if price <= current_max_buy_price and remaining_buy_capacity > 0:
                qty_to_take = min(volume, remaining_buy_capacity)
                self._place_buy(price, qty_to_take)
                remaining_buy_capacity -= qty_to_take
                taken_buy_vol += qty_to_take
                sim_sell_orders[price] += qty_to_take # Add back positive volume to simulate fill
                if sim_sell_orders[price] == 0:
                    del sim_sell_orders[price]
            else:
                # Asks are sorted low to high, if this one isn't good, none further will be
                break

        # Sell to market (take bids)
        bids = sorted(sim_buy_orders.items(), reverse=True)
        for price, volume in bids:
             if price >= current_min_sell_price and remaining_sell_capacity > 0:
                 qty_to_take = min(volume, remaining_sell_capacity)
                 self._place_sell(price, qty_to_take)
                 remaining_sell_capacity -= qty_to_take
                 taken_sell_vol += qty_to_take
                 sim_buy_orders[price] -= qty_to_take # Remove volume to simulate fill
                 if sim_buy_orders[price] == 0:
                     del sim_buy_orders[price]
             else:
                 # Bids are sorted high to low
                 break

        logger.print(f"{self.symbol} Phase 1 DONE - Bought: {taken_buy_vol}, Sold: {taken_sell_vol}, RemBuyCap: {remaining_buy_capacity}, RemSellCap: {remaining_sell_capacity}")

        # --- Phase 2: Liquidation if Stuck ---
        # Use capacity remaining *after* phase 1
        liq_buy_vol = 0
        liq_sell_vol = 0
        liq_pct = self.global_params["liq_pct"]
        hard_liq_offset = self.global_params["hard_liq_offset"]
        soft_liq_offset = self.global_params["soft_liq_offset"]

        if is_stuck_hard:
            if position > 0 and remaining_sell_capacity > 0: # Stuck long, need to sell
                qty_to_liq = math.floor(remaining_sell_capacity * liq_pct)
                liq_price = true_value - hard_liq_offset # Aggressive sell price
                self._place_sell(liq_price, qty_to_liq)
                liq_sell_vol += qty_to_liq
                remaining_sell_capacity -= qty_to_liq
                logger.print(f"{self.symbol} Hard Liq Sell {qty_to_liq}x{liq_price}")
            elif position < 0 and remaining_buy_capacity > 0: # Stuck short, need to buy
                qty_to_liq = math.floor(remaining_buy_capacity * liq_pct)
                liq_price = true_value + hard_liq_offset # Aggressive buy price
                self._place_buy(liq_price, qty_to_liq)
                liq_buy_vol += qty_to_liq
                remaining_buy_capacity -= qty_to_liq
                logger.print(f"{self.symbol} Hard Liq Buy {qty_to_liq}x{liq_price}")
        elif is_stuck_soft:
             if position > 0 and remaining_sell_capacity > 0: # Stuck long, need to sell
                qty_to_liq = math.floor(remaining_sell_capacity * liq_pct)
                liq_price = true_value - soft_liq_offset # Less aggressive sell
                self._place_sell(liq_price, qty_to_liq)
                liq_sell_vol += qty_to_liq
                remaining_sell_capacity -= qty_to_liq
                logger.print(f"{self.symbol} Soft Liq Sell {qty_to_liq}x{liq_price}")
             elif position < 0 and remaining_buy_capacity > 0: # Stuck short, need to buy
                qty_to_liq = math.floor(remaining_buy_capacity * liq_pct)
                liq_price = true_value + soft_liq_offset # Less aggressive buy
                self._place_buy(liq_price, qty_to_liq)
                liq_buy_vol += qty_to_liq
                remaining_buy_capacity -= qty_to_liq
                logger.print(f"{self.symbol} Soft Liq Buy {qty_to_liq}x{liq_price}")

        # --- Phase 3: Place New Market Making Orders ---
        # Use capacity remaining *after* phase 1 and 2
        make_offset = self.global_params["make_offset"]

        # Find best prices in the *simulated* order book (after our takes)
        best_sim_ask = min(sim_sell_orders.keys()) if sim_sell_orders else None
        best_sim_bid = max(sim_buy_orders.keys()) if sim_buy_orders else None

        # Determine desired bid price (penny the best bid or place relative to fair value)
        make_bid_price = true_value - make_offset # Default if no competing bid
        if best_sim_bid is not None:
             make_bid_price = best_sim_bid + make_offset # Penny the best bid + offset

        # Determine desired ask price
        make_ask_price = true_value + make_offset # Default if no competing ask
        if best_sim_ask is not None:
            make_ask_price = best_sim_ask - make_offset # Penny the best ask - offset

        # Apply skew again (could be different if position changed due to fills)
        # Using the same skew calculation as before for simplicity, based on initial position
        final_bid_price = make_bid_price + total_skew
        final_ask_price = make_ask_price + total_skew

        # Ensure bid < ask
        final_bid_price = min(final_bid_price, final_ask_price - 1)

        # Place orders with remaining capacity
        if remaining_buy_capacity > 0:
            self._place_buy(final_bid_price, remaining_buy_capacity)

        if remaining_sell_capacity > 0:
             # Ensure ask is still reasonable after skew
             final_ask_price = max(final_ask_price, final_bid_price + 1)
             self._place_sell(final_ask_price, remaining_sell_capacity)

        logger.print(f"{self.symbol} Phase 3 DONE - Placed Bid: {remaining_buy_capacity}x{final_bid_price:.0f}, Placed Ask: {remaining_sell_capacity}x{final_ask_price:.0f}")


    # Hook for RSI skew, default is 0
    def _get_rsi_skew(self, state: TradingState) -> int:
        return 0

    def save(self) -> dict:
        # Save position window
        return {"window": list(self.window)}

    def load(self, data: dict) -> None:
        # Load position window
        if data and "window" in data and isinstance(data["window"], list):
            loaded_window = data["window"]
            start_index = max(0, len(loaded_window) - self.window.maxlen)
            self.window.clear()
            self.window.extend(loaded_window[start_index:])

# --- Product Specific Strategies ---

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float | None:
        # Use configured fair value
        return self.params.get("fair_value", 10000)

    # Override act to use simpler 10000-based logic if needed, or let MarketMakingStrategy handle it
    # For now, let the base class handle it with skewing based on params.

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float | None:
        # Dynamic calculation based on most popular price points (highest volume)
        # This logic is similar to prototype's KelpStrategy
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders: return None # No data

        # Find most popular buy price (highest volume, tie-break higher price)
        popular_buy_price = 0
        if buy_orders:
            buy_levels = sorted(buy_orders.items(), key=lambda item: (item[1], item[0]), reverse=True)
            popular_buy_price = buy_levels[0][0]
        else: # Estimate if no buy orders
             if not sell_orders: return None
             popular_buy_price = min(sell_orders.keys()) - 1 # Crude

        # Find most popular sell price (highest volume, tie-break lower price)
        popular_sell_price = 0
        if sell_orders:
             # Sort by abs(volume) desc, price asc
             sell_levels = sorted(sell_orders.items(), key=lambda item: (abs(item[1]), -item[0]), reverse=True)
             popular_sell_price = sell_levels[0][0]
        else: # Estimate if no sell orders
             if not buy_orders: return None
             popular_sell_price = max(buy_orders.keys()) + 1 # Crude

        # Calculate midpoint, with fallbacks
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

class SquidInkStrategy(MarketMakingStrategy):
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        super().__init__(symbol, params, position_limit)
        # State for RSI calculation
        history_len = self.params.get("price_history_len", 20)
        self.price_history: Deque[float] = deque(maxlen=history_len)
        self.rsi_value: float | None = None # Store calculated RSI

    def get_true_value(self, state: TradingState) -> float | None:
        # Same fair value logic as Kelp for now
        # Could be refined later (e.g., using VWAP from market trades)
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders: return None

        popular_buy_price = 0
        if buy_orders:
            buy_levels = sorted(buy_orders.items(), key=lambda item: (item[1], item[0]), reverse=True)
            popular_buy_price = buy_levels[0][0]
        else:
             if not sell_orders: return None
             popular_buy_price = min(sell_orders.keys()) - 1

        popular_sell_price = 0
        if sell_orders:
             sell_levels = sorted(sell_orders.items(), key=lambda item: (abs(item[1]), -item[0]), reverse=True)
             popular_sell_price = sell_levels[0][0]
        else:
             if not buy_orders: return None
             popular_sell_price = max(buy_orders.keys()) + 1

        fair_value = None
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            fair_value = (popular_buy_price + popular_sell_price) / 2.0
        else:
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0:
                fair_value = (best_bid + best_ask) / 2.0
            elif best_bid > 0: fair_value = float(best_bid)
            elif best_ask > 0: fair_value = float(best_ask)

        # Store the calculated fair value for RSI
        if fair_value is not None:
            self.price_history.append(fair_value)

        return fair_value

    def _calculate_rsi(self) -> float | None:
        period = self.params.get("rsi_period", 14)
        if len(self.price_history) < period + 1: # Need enough data
            return None

        prices = list(self.price_history)
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # Calculate gains and losses for the RSI period
        period_changes = changes[-period:] # Look at the last 'period' changes
        gains = [c for c in period_changes if c > 0]
        losses = [abs(c) for c in period_changes if c < 0]

        if not losses: # Avoid division by zero if no losses
             if gains: return 100.0 # All gains = RSI 100
             else: return 50.0 # No changes = RSI 50 (neutral)

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period # Already checked losses is not empty

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        self.rsi_value = rsi # Store for logging/potential use
        return rsi

    # Override _get_rsi_skew to use calculated RSI
    def _get_rsi_skew(self, state: TradingState) -> int:
        rsi = self._calculate_rsi()
        if rsi is None:
            logger.print(f"{self.symbol} RSI: Not enough data")
            return 0

        oversold = self.params.get("rsi_oversold", 30)
        overbought = self.params.get("rsi_overbought", 70)
        skew_factor = self.params.get("rsi_skew_factor", 1)
        rsi_skew = 0

        if rsi > overbought:
            rsi_skew = -skew_factor # Overbought -> price likely to drop -> skew sell price lower, buy price lower
            logger.print(f"{self.symbol} RSI: {rsi:.2f} (> {overbought}) -> Overbought, Skew: {rsi_skew}")
        elif rsi < oversold:
            rsi_skew = skew_factor  # Oversold -> price likely to rise -> skew buy price higher, sell price higher
            logger.print(f"{self.symbol} RSI: {rsi:.2f} (< {oversold}) -> Oversold, Skew: {rsi_skew}")
        else:
             logger.print(f"{self.symbol} RSI: {rsi:.2f} (Neutral)")

        return rsi_skew

    def save(self) -> dict:
        # Save base class state (window) and price history
        data = super().save()
        data["price_history"] = list(self.price_history)
        data["rsi_value"] = self.rsi_value # Save last calculated RSI for info
        return data

    def load(self, data: dict) -> None:
        # Load base class state
        super().load(data)
        # Load price history
        if data and "price_history" in data and isinstance(data["price_history"], list):
            loaded_history = data["price_history"]
            # Ensure maxlen is respected when loading
            start_index = max(0, len(loaded_history) - self.price_history.maxlen)
            self.price_history.clear()
            self.price_history.extend(loaded_history[start_index:])
        if data and "rsi_value" in data:
             self.rsi_value = data["rsi_value"]


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
            Product.RAINFOREST_RESIN: RainforestResinStrategy(
                Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN], self.position_limits[Product.RAINFOREST_RESIN]
            ),
            Product.SQUID_INK: SquidInkStrategy(
                Product.SQUID_INK, PARAMS[Product.SQUID_INK], self.position_limits[Product.SQUID_INK]
            )
        }
        logger.print("Trader Initialized (v2) with strategies for:", list(self.strategies.keys()))

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        logger.print(f"--- Round 1 V2 | Timestamp: {state.timestamp} ---")
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
            traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"))
        except Exception as e:
             logger.print(f"Error encoding traderData: {e}. Sending empty data.")
             traderData_encoded = "{}"

        logger.flush(state, result, conversions, traderData_encoded)

        return result, conversions, traderData_encoded
