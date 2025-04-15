import json
import numpy as np
from typing import Any, Dict, List, Deque, Optional, Tuple
from abc import abstractmethod
from collections import deque, defaultdict
import math
import copy

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# --- Logger Class --- Slightly simplified from competition provided one
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        # Truncate state.traderData, trader_data, and self.logs to fit the log limit
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

# --- Product Enums ---
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    B1B2_DEVIATION = "B1B2_DEVIATION"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# --- Basket Definitions ---
BASKET_COMPONENTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
}

# --- Theoretical Spread Definition ---
# B1 - B2 = (6C + 3J + 1D) - (4C + 2J) = 2C + 1J + 1D
B1B2_THEORETICAL_COMPONENTS = {
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: 1
}

# --- Strategy Parameters (Optimized for B1 only) ---
PARAMS = {
    "SQUID_INK": {
        "rsi_window": 96,
        "rsi_overbought": 56,
        "rsi_oversold": 39,
        "price_offset": 1
    },
    "VOLCANIC_ROCK": {
        "rsi_window": 54,
        "rsi_overbought": 57,
        "rsi_oversold": 41,
        "price_offset": 0
    }
}

# Add after PARAMS dictionary
VOUCHER_SLOPE_PARAMS = PARAMS["VOUCHER_SLOPE_REVERSION"]
VOUCHER_10500_PARAMS = PARAMS["VOUCHER_10500_STRATEGY"]

# --- Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] 
        self.act(state)
        return self.orders

    def _place_buy_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}") # Keep logs minimal

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}") # Keep logs minimal

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None: return best_bid 
        elif best_ask is not None: return best_ask 
        else: return None

# --- V3 Strategies (Base for Resin) ---
class V3Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"{self.symbol} BUY {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"{self.symbol} SELL {quantity}x{price}")

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

class V3MarketMakingStrategy(V3Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window = deque(maxlen=4)
        self.window_size = 4

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None: # Overridden by Resin Strategy
        if self.symbol not in state.order_depths: return

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders: return

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        true_value = self.get_true_value(state)
        self.window.append(abs(position) == self.position_limit)

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.position_limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.position_limit * 0.5 else true_value

        # Buy logic
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity
        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] if buy_orders else (true_value - 2)
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        # Sell logic
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity
        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity
        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else (true_value + 2)
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> dict:
        return {"window": list(self.window)}

    def load(self, data: dict) -> None:
        if data and "window" in data:
            loaded_window = data["window"]
            start_index = max(0, len(loaded_window) - self.window.maxlen)
            self.window.clear()
            self.window.extend(loaded_window[start_index:])

class V3RainforestResinStrategy(V3MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return PARAMS.get(Product.RAINFOREST_RESIN, {}).get("fair_value", 10000)

    def act(self, state: TradingState) -> None:
        self.orders = [] 
        true_value = self.get_true_value(state)
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return

        initial_buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        initial_sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
        sim_buy_orders = copy.deepcopy(initial_buy_orders)
        sim_sell_orders = copy.deepcopy(initial_sell_orders)

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        #slurp
        asks_sorted = sorted(sim_sell_orders.items())
        for price, volume in asks_sorted:
            if to_buy <= 0: break
            volume = -volume
            if price <= true_value:
                qty_to_take = min(to_buy, volume)
                self.buy(price, qty_to_take)
                to_buy -= qty_to_take
                sim_sell_orders[price] += qty_to_take
                if sim_sell_orders[price] == 0: del sim_sell_orders[price]
            else: break

        bids_sorted = sorted(sim_buy_orders.items(), reverse=True)
        for price, volume in bids_sorted:
            if to_sell <= 0: break
            if price >= true_value:
                qty_to_take = min(to_sell, volume)
                self.sell(price, qty_to_take)
                to_sell -= qty_to_take
                sim_buy_orders[price] -= qty_to_take
                if sim_buy_orders[price] == 0: del sim_buy_orders[price]
            else: break

        # mm
        make_bid_price = true_value - 1
        bids_below_10k = {p: v for p, v in sim_buy_orders.items() if p < true_value}
        if bids_below_10k:
            best_bid_after_take = max(bids_below_10k.keys())
            best_bid_vol = bids_below_10k[best_bid_after_take]
            if best_bid_vol <= 6: make_bid_price = best_bid_after_take
            else: make_bid_price = best_bid_after_take + 1

        make_ask_price = true_value + 1
        asks_above_10k = {p: v for p, v in sim_sell_orders.items() if p > true_value}
        if asks_above_10k:
            best_ask_after_take = min(asks_above_10k.keys())
            best_ask_vol = abs(asks_above_10k[best_ask_after_take])
            if best_ask_vol <= 6: make_ask_price = best_ask_after_take
            else: make_ask_price = best_ask_after_take - 1

        if make_bid_price >= make_ask_price: make_ask_price = make_bid_price + 1

        if to_buy > 0: self.buy(make_bid_price, to_buy)
        if to_sell > 0: self.sell(make_ask_price, to_sell)


# --- Prototype Strategies (Base for Kelp) ---
class PrototypeMarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window_size = 4
        self.window: Deque[bool] = deque(maxlen=self.window_size) 

    @abstractmethod
    def get_true_value(self, state: TradingState) -> Optional[int]:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        self.orders = [] 
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return

        buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
        if not buy_orders_dict and not sell_orders_dict: return

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        true_value = self.get_true_value(state)
        if true_value is None: return

        soft_limit_threshold = self.position_limit * 0.5
        max_buy_price = true_value - 1 if position > soft_limit_threshold else true_value
        min_sell_price = true_value + 1 if position < -soft_limit_threshold else true_value

        # Phase 1: Take orders
        sell_orders_list = sorted(sell_orders_dict.items()) 
        for price, volume in sell_orders_list:
            volume = -volume 
            if to_buy > 0 and price <= max_buy_price:
                qty_to_take = min(to_buy, volume)
                self._place_buy_order(price, qty_to_take)
                to_buy -= qty_to_take

        buy_orders_list = sorted(buy_orders_dict.items(), reverse=True) 
        for price, volume in buy_orders_list:
            if to_sell > 0 and price >= min_sell_price:
                qty_to_take = min(to_sell, volume)
                self._place_sell_order(price, qty_to_take)
                to_sell -= qty_to_take

        # Phase 2: Liquidation if stuck at limits
        self.window.append(abs(position) == self.position_limit)
        is_full_window = len(self.window) == self.window_size 
        stuck_count = sum(self.window)
        soft_liquidate = is_full_window and stuck_count >= self.window_size / 2 and self.window[-1]
        hard_liquidate = is_full_window and stuck_count == self.window_size

        if hard_liquidate:
            if to_buy > 0: # Stuck short
                quantity = to_buy // 2
                self._place_buy_order(true_value, quantity)
                to_buy -= quantity
            elif to_sell > 0: # Stuck long
                 quantity = to_sell // 2
                 self._place_sell_order(true_value, quantity)
                 to_sell -= quantity
        elif soft_liquidate:
            if to_buy > 0: # Stuck short
                quantity = to_buy // 2
                liq_price = true_value - 2 
                self._place_buy_order(liq_price, quantity)
                to_buy -= quantity
            elif to_sell > 0: # Stuck long
                quantity = to_sell // 2
                liq_price = true_value + 2 
                self._place_sell_order(liq_price, quantity)
                to_sell -= quantity

        # Phase 3: Make remaining orders
        if to_buy > 0:
            popular_buy_price = max(buy_orders_list, key=lambda tup: tup[1])[0] if buy_orders_list else (true_value - 2)
            make_price = min(max_buy_price, popular_buy_price + 1)
            self._place_buy_order(make_price, to_buy)
        if to_sell > 0:
            popular_sell_price = min(sell_orders_list, key=lambda tup: tup[1])[0] if sell_orders_list else (true_value + 2)
            make_price = max(min_sell_price, popular_sell_price - 1)
            self._place_sell_order(make_price, to_sell)

    def save(self) -> dict:
        return {"window": list(self.window)}

    def load(self, data: dict) -> None:
        if data and "window" in data and isinstance(data["window"], list):
            loaded_window = data["window"]
            self.window = deque(loaded_window, maxlen=self.window_size)

class PrototypeKelpStrategy(PrototypeMarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> Optional[int]:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
        buy_levels = sorted(buy_orders.items(), reverse=True) if buy_orders else []
        sell_levels = sorted(sell_orders.items()) if sell_orders else []

        # Prioritize popular prices (highest volume)
        popular_buy_price = max(buy_levels, key=lambda tup: tup[1])[0] if buy_levels else 0
        popular_sell_price = min(sell_levels, key=lambda tup: tup[1])[0] if sell_levels else 0

        final_value = None
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            final_value = (popular_buy_price + popular_sell_price) / 2
        else: # Fallback to BBO midpoint or single side
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0 and best_ask > best_bid: final_value = (best_bid + best_ask) / 2
            elif best_bid > 0: final_value = best_bid
            elif best_ask > 0: final_value = best_ask

        return round(final_value) if final_value is not None else None

# --- NEW: Voucher Slope Reversion Strategy State ---
class SlopeReversionStrategyState:
    """Holds the state for a single adjacent voucher pair slope."""
    def __init__(self, window_size: int):
        self.slope_history: Deque[float] = deque(maxlen=window_size)
        self.rolling_mean: Optional[float] = None
        self.rolling_std: Optional[float] = None
        self.lower_band: Optional[float] = None
        self.upper_band: Optional[float] = None
        self.current_slope: Optional[float] = None # Store the latest calculated slope
        # --- Position Tracking ---
        # 0 = Flat, 1 = Long Slope (Bought Low Strike, Sold High Strike), -1 = Short Slope (Sold Low Strike, Bought High Strike)
        self.current_spread_direction: int = 0

    def update(self, current_slope: Optional[float], k: float, min_history: int):
        """Updates history and recalculates stats if slope is valid."""
        self.current_slope = current_slope # Store latest slope even if None

        if current_slope is None:
            # Don't add None to history, keep existing stats until a valid slope is seen
            return

        self.slope_history.append(current_slope)

        if len(self.slope_history) >= min_history:
            history_arr = np.array(list(self.slope_history))
            self.rolling_mean = np.mean(history_arr)
            # Use ddof=1 for sample standard deviation, although for large N it matters less
            self.rolling_std = np.std(history_arr, ddof=1)

            if self.rolling_std is not None and self.rolling_std > 1e-6: # Avoid issues with zero/tiny std dev
                self.lower_band = self.rolling_mean - k * self.rolling_std
                self.upper_band = self.rolling_mean + k * self.rolling_std
            else:
                # If std dev is zero or near-zero, bands are effectively just the mean
                self.lower_band = self.rolling_mean
                self.upper_band = self.rolling_mean
        else:
            # Not enough history yet
            self.rolling_mean = None
            self.rolling_std = None
            self.lower_band = None
            self.upper_band = None

    def to_dict(self) -> Dict:
        """Serializes state for traderData."""
        return {
            "slope_history": list(self.slope_history),
            "current_spread_direction": self.current_spread_direction
            # Recalculate mean/std/bands on load based on history
        }

    @classmethod
    def from_dict(cls, data: Dict, window_size: int, k: float, min_history: int, recalc: bool) -> 'SlopeReversionStrategyState':
        """Deserializes state from traderData."""
        instance = cls(window_size)
        if "slope_history" in data and isinstance(data["slope_history"], list):
            history = data["slope_history"]
            # Ensure history respects maxlen on load
            start_index = max(0, len(history) - window_size)
            instance.slope_history.extend(history[start_index:])

            # Optionally recalculate stats based on loaded history
            if recalc and instance.slope_history:
                # Use the last known slope to trigger recalculation
                last_slope = instance.slope_history[-1]
                instance.update(last_slope, k, min_history)

        # Load position state
        instance.current_spread_direction = data.get("current_spread_direction", 0)
        if not isinstance(instance.current_spread_direction, int):
             instance.current_spread_direction = 0 # Reset if invalid

        return instance

# --- NEW: Voucher Slope Reversion Strategy ---
class VoucherSlopeReversionStrategy(Strategy):
    """Trades the reversion of the slope between adjacent voucher strike prices."""
    def __init__(self, symbol: str, position_limits: Dict[str, int], strikes_map: Dict[str, int], all_voucher_list: List[str]):
        # Use a unique symbol key for this strategy, position limit isn't directly used by base class
        super().__init__(symbol, 0)
        self.pos_limits = position_limits # Store limits for all voucher products
        # --- Store strikes and voucher list ---
        self.strikes = strikes_map
        self.all_vouchers = all_voucher_list
        # --- Load Params ---
        self.params = PARAMS.get(self.symbol, PARAMS["VOUCHER_SLOPE_REVERSION"]) # Use default if specific key missing
        if not self.params: # Fallback if default also missing (shouldn't happen with above)
             self.params = {
                 "rolling_window": 100, "std_dev_multiplier": 1.5, "trade_size": 5,
                 "min_history": 50, "exit_target": "mean", "recalc_stats_on_load": True,
                 "liquidity_capture_pct": 0.90, "min_trade_size": 5,
                 "allowed_pairs": ["9500-9750", "10000-10250"]  # Default allowed pairs
             }
             logger.print(f"Warning: {self.symbol} params not found, using defaults.")

        # Load parameters
        self.window_size = self.params.get("rolling_window", 100)
        self.k = self.params.get("std_dev_multiplier", 1.5)
        self.trade_size = self.params.get("trade_size", 5)
        self.min_history = self.params.get("min_history", 50)
        self.exit_target = self.params.get("exit_target", "mean")
        self.recalc_on_load = self.params.get("recalc_stats_on_load", True)
        
        # Simplified liquidity parameters
        self.liquidity_capture_pct = self.params.get("liquidity_capture_pct", 0.90)
        self.min_trade_size = self.params.get("min_trade_size", 5)
        
        # New parameter - allowed pairs to trade (to avoid conflicting signals)
        self.allowed_pairs = self.params.get("allowed_pairs", ["9500-9750", "10000-10250"])

        # Strategy state per pair
        self.pair_states: Dict[str, SlopeReversionStrategyState] = {} # Key: "K1-K2" string

        # Define adjacent pairs based on sorted strikes
        sorted_vouchers = sorted(self.all_vouchers, key=lambda p: self.strikes[p])
        self.voucher_pairs: List[Tuple[str, str]] = []
        for i in range(len(sorted_vouchers) - 1):
            # Ensure prod1 always has the lower strike
            prod1 = sorted_vouchers[i]
            prod2 = sorted_vouchers[i+1]
            pair_label = self._get_pair_label_from_strikes(self.strikes[prod1], self.strikes[prod2])
            
            # Only add pairs that are in our allowed list
            if pair_label in self.allowed_pairs:
                self.voucher_pairs.append((prod1, prod2)) # Store as (low_strike_prod, high_strike_prod)
                self.pair_states[pair_label] = SlopeReversionStrategyState(self.window_size)
        
        logger.print(f"{self.symbol}: Trading only these pairs: {[self._get_pair_label(p1, p2) for p1, p2 in self.voucher_pairs]} (from allowed: {self.allowed_pairs})")

    def _get_pair_label_from_strikes(self, strike1: int, strike2: int) -> str:
        """Generates a consistent label like '9500-9750' from strike values."""
        return f"{min(strike1, strike2)}-{max(strike1, strike2)}"
        
    def _get_pair_label(self, prod1: str, prod2: str) -> str:
        """Generates a consistent label like '9500-9750'."""
        k1 = self.strikes[prod1]
        k2 = self.strikes[prod2]
        return self._get_pair_label_from_strikes(k1, k2)

    def _get_mid_price_safe(self, symbol: str, state: TradingState) -> Optional[float]:
        """Safely calculates the mid-price for a symbol."""
        # Use the base class method if available, or implement locally
        depth = state.order_depths.get(symbol)
        if not depth: return None
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        # Fallback logic if one side is missing (optional, can return None)
        # elif best_bid is not None: return best_bid
        # elif best_ask is not None: return best_ask
        else: return None

    def _calculate_slope(self, low_strike_prod: str, high_strike_prod: str, state: TradingState) -> Optional[float]:
        """Calculates slope C(Low)-C(High) / High-Low for an adjacent pair."""
        k1 = self.strikes[low_strike_prod]
        k2 = self.strikes[high_strike_prod]
        price1 = self._get_mid_price_safe(low_strike_prod, state)
        price2 = self._get_mid_price_safe(high_strike_prod, state)
        if price1 is None or price2 is None: return None
        strike_diff = k2 - k1
        if abs(strike_diff) < 1e-6: return None
        slope = (price1 - price2) / strike_diff
        return slope
        
    def _calculate_available_liquidity(self, product: str, side: str, state: TradingState) -> int:
        """Calculate available liquidity for a product on buy/sell side."""
        depth = state.order_depths.get(product)
        if not depth: return 0
        
        if side.lower() == "buy":
            # Liquidity for buying = sum of sell orders volumes
            if not depth.sell_orders: return 0
            return sum(abs(vol) for vol in depth.sell_orders.values())
        elif side.lower() == "sell":
            # Liquidity for selling = sum of buy orders volumes
            if not depth.buy_orders: return 0
            return sum(abs(vol) for vol in depth.buy_orders.values())
        return 0
        
    def _calculate_position_capacity(self, product: str, direction: int, state: TradingState) -> int:
        """Calculate how many more units we can buy/sell based on position limits."""
        current_pos = state.position.get(product, 0)
        limit = self.pos_limits.get(product, 0)
        
        if direction > 0:  # Buying
            return max(0, limit - current_pos)
        else:  # Selling
            return max(0, limit + current_pos)
    
    def _calculate_trade_size(self, 
                              low_strike_prod: str, 
                              high_strike_prod: str, 
                              direction: int,  # +1 for Buy Spread, -1 for Sell Spread
                              state: TradingState) -> int:
        """Calculate maximum possible trade size based on limits and available liquidity."""
        # For Buy Spread (direction=+1): Sell low_strike, Buy high_strike
        # For Sell Spread (direction=-1): Buy low_strike, Sell high_strike
        
        # Determine products to buy/sell
        sell_prod = low_strike_prod if direction > 0 else high_strike_prod
        buy_prod = high_strike_prod if direction > 0 else low_strike_prod
        
        # 1. Check position capacity for both legs
        sell_capacity = self._calculate_position_capacity(sell_prod, -1, state)
        buy_capacity = self._calculate_position_capacity(buy_prod, 1, state)
        position_limit = min(sell_capacity, buy_capacity)
        
        # 2. Check available liquidity for both legs
        sell_liquidity = self._calculate_available_liquidity(sell_prod, "sell", state)
        buy_liquidity = self._calculate_available_liquidity(buy_prod, "buy", state)
        
        # Apply liquidity capture percentage (avoid taking all available liquidity)
        sell_liquidity = int(sell_liquidity * self.liquidity_capture_pct)
        buy_liquidity = int(buy_liquidity * self.liquidity_capture_pct)
        
        liquidity_limit = min(sell_liquidity, buy_liquidity)
        
        # 3. Calculate the final trade size - either our base size or maximum available
        trade_size = min(self.trade_size, position_limit, liquidity_limit)
        
        # 4. Enforce minimum trade size
        if trade_size < self.min_trade_size:
            logger.print(f"  Trade size {trade_size} below minimum {self.min_trade_size}. Liquidity: {liquidity_limit}, Position: {position_limit}")
            return 0
            
        logger.print(f"  Calculated trade size: {trade_size} (Liq={liquidity_limit}, Pos={position_limit}, Base={self.trade_size})")
        return trade_size

    def _place_voucher_spread_order(self,
                                  low_strike_prod: str, high_strike_prod: str,
                                  direction: int, # +1 for Buy Spread (Sell Low, Buy High), -1 for Sell Spread (Buy Low, Sell High)
                                  quantity: int, state: TradingState) -> bool:
        """ Places aggressive orders for a voucher spread. Returns True if orders placed. """
        pair_label = self._get_pair_label(low_strike_prod, high_strike_prod)
        abs_quantity = abs(quantity)
        if abs_quantity == 0: return False

        trade_description = "Buy Spread (Sell Low, Buy High)" if direction > 0 else "Sell Spread (Buy Low, Sell High)"
        logger.print(f"Attempting {trade_description} for {pair_label}, Qty: {abs_quantity}")

        # Determine which product to buy/sell for this direction
        sell_prod = low_strike_prod if direction > 0 else high_strike_prod
        buy_prod = high_strike_prod if direction > 0 else low_strike_prod

        # --- Basic Capacity Check ---
        current_pos_sell = state.position.get(sell_prod, 0)
        current_pos_buy = state.position.get(buy_prod, 0)
        limit_sell = self.pos_limits.get(sell_prod, 0)
        limit_buy = self.pos_limits.get(buy_prod, 0)

        if (current_pos_sell - abs_quantity) < -limit_sell:
             logger.print(f"  SKIP {trade_description} {pair_label}: Selling {abs_quantity} of {sell_prod} (pos {current_pos_sell}) would exceed short limit {-limit_sell}.")
             return False
        if (current_pos_buy + abs_quantity) > limit_buy:
             logger.print(f"  SKIP {trade_description} {pair_label}: Buying {abs_quantity} of {buy_prod} (pos {current_pos_buy}) would exceed long limit {limit_buy}.")
             return False

        # --- Get Prices for Aggressive Orders ---
        depth_sell = state.order_depths.get(sell_prod)
        depth_buy = state.order_depths.get(buy_prod)

        # Price to SELL sell_prod (hit the best bid)
        sell_price = None
        if depth_sell and depth_sell.buy_orders:
             sell_price = max(depth_sell.buy_orders.keys())
        else:
             logger.print(f"  WARN {pair_label}: No buy orders for {sell_prod} to determine aggressive sell price.")
             return False # Require liquidity to execute aggressivley

        # Price to BUY buy_prod (hit the best ask)
        buy_price = None
        if depth_buy and depth_buy.sell_orders:
             buy_price = min(depth_buy.sell_orders.keys())
        else:
             logger.print(f"  WARN {pair_label}: No sell orders for {buy_prod} to determine aggressive buy price.")
             return False # Require liquidity

        # --- Place Orders ---
        # Sell Leg
        logger.print(f"  PLACING {pair_label}: SELL {sell_prod} {abs_quantity}x{sell_price}")
        self.orders.append(Order(sell_prod, sell_price, -abs_quantity))

        # Buy Leg
        logger.print(f"  PLACING {pair_label}: BUY {buy_prod} {abs_quantity}x{buy_price}")
        self.orders.append(Order(buy_prod, buy_price, abs_quantity))

        return True # Indicate success

    def act(self, state: TradingState) -> None:
        self.orders = [] # Clear orders at the start of each act call

        for low_strike_prod, high_strike_prod in self.voucher_pairs:
            pair_label = self._get_pair_label(low_strike_prod, high_strike_prod)
            pair_state = self.pair_states.get(pair_label) # Use .get for safety
            if pair_state is None: continue # Should not happen

            # 1. Calculate current slope
            current_slope = self._calculate_slope(low_strike_prod, high_strike_prod, state)

            # 2. Update state (history, MA, STD, Bands)
            pair_state.update(current_slope, self.k, self.min_history)

            # Log state if possible
            if pair_state.lower_band is not None: 
                logger.print(f" {self.symbol} {pair_label}: Slp={current_slope:.3f}, M={pair_state.rolling_mean:.3f}, Sd={pair_state.rolling_std:.3f}, Lb={pair_state.lower_band:.3f}, Pos={pair_state.current_spread_direction}")
            elif current_slope is not None: 
                logger.print(f" {self.symbol} {pair_label}: Slp={current_slope:.3f}, Hist={len(pair_state.slope_history)}/{self.min_history}, Pos={pair_state.current_spread_direction}")

            # 3. Check Trading Signals (Only if bands are calculated)
            if current_slope is not None and pair_state.lower_band is not None and pair_state.upper_band is not None and pair_state.rolling_mean is not None:

                # --- Entry Logic ---
                # If flat or existing position AND slope < Lower Band -> Enter/add to LONG Slope position (+1)
                # We expect slope to increase (revert). Profit by Buying Low Strike, Selling High Strike.
                if pair_state.current_spread_direction >= 0 and current_slope < pair_state.lower_band:
                    logger.print(f"  ENTRY Signal {pair_label}: Slope {current_slope:.3f} < LBand {pair_state.lower_band:.3f}. Target: Long Slope.")
                    
                    # Calculate maximum possible trade size
                    trade_size = self._calculate_trade_size(
                        low_strike_prod, high_strike_prod, 
                        -1,  # Sell Spread direction 
                        state
                    )
                    
                    if trade_size > 0:
                        # Place Sell Spread (direction=-1) -> Buys LowK, Sells HighK
                        if self._place_voucher_spread_order(low_strike_prod, high_strike_prod, -1, trade_size, state):
                            # If we were flat before, now we're long, otherwise we're adding to long
                            pair_state.current_spread_direction = +1

                # If flat or existing position AND slope > Upper Band -> Enter/add to SHORT Slope position (-1)
                # We expect slope to decrease (revert). Profit by Selling Low Strike, Buying High Strike.
                elif pair_state.current_spread_direction <= 0 and current_slope > pair_state.upper_band:
                    logger.print(f"  ENTRY Signal {pair_label}: Slope {current_slope:.3f} > UBand {pair_state.upper_band:.3f}. Target: Short Slope.")
                    
                    # Calculate maximum possible trade size
                    trade_size = self._calculate_trade_size(
                        low_strike_prod, high_strike_prod, 
                        +1,  # Buy Spread direction
                        state
                    )
                    
                    if trade_size > 0:
                        # Place Buy Spread (direction=+1) -> Sells LowK, Buys HighK
                        if self._place_voucher_spread_order(low_strike_prod, high_strike_prod, +1, trade_size, state):
                            # If we were flat before, now we're short, otherwise we're adding to short
                            pair_state.current_spread_direction = -1


                # --- Exit Logic ---
                # If SHORT Slope (-1) AND slope crossed back BELOW target -> Exit position (0)
                # Trade: Buy low strike, Sell high strike (opposite of entry)
                elif pair_state.current_spread_direction == -1:
                    exit_level = pair_state.rolling_mean # Default target: mean
                    if self.exit_target != "mean": # Allow for fixed level exit? (e.g., 0)
                        try: exit_level = float(self.exit_target)
                        except ValueError: pass # Keep mean if target is not a valid float

                    # Exit if slope falls back below the exit level
                    if current_slope < exit_level:
                         logger.print(f"  EXIT Signal {pair_label}: Slope {current_slope:.3f} < Exit Target {exit_level:.3f}. Closing Short Slope.")
                         
                         # Get current positions to determine exit size
                         low_pos = state.position.get(low_strike_prod, 0)
                         high_pos = state.position.get(high_strike_prod, 0)
                         
                         # For short slope: we're short low_strike and long high_strike
                         exit_size = min(abs(low_pos), abs(high_pos))
                         
                         if exit_size > 0:
                             # Place Sell Spread (direction=-1) -> Buys LowK, Sells HighK
                             if self._place_voucher_spread_order(low_strike_prod, high_strike_prod, -1, exit_size, state):
                                  pair_state.current_spread_direction = 0 # Update state

                # If LONG Slope (+1) AND slope crossed back ABOVE target -> Exit position (0)
                # Trade: Sell low strike, Buy high strike (opposite of entry)
                elif pair_state.current_spread_direction == +1:
                    exit_level = pair_state.rolling_mean # Default target: mean
                    if self.exit_target != "mean":
                        try: exit_level = float(self.exit_target)
                        except ValueError: pass

                    # Exit if slope rises back above the exit level
                    if current_slope > exit_level:
                         logger.print(f"  EXIT Signal {pair_label}: Slope {current_slope:.3f} > Exit Target {exit_level:.3f}. Closing Long Slope.")
                         
                         # Get current positions to determine exit size
                         low_pos = state.position.get(low_strike_prod, 0)
                         high_pos = state.position.get(high_strike_prod, 0)
                         
                         # For long slope: we're long low_strike and short high_strike
                         exit_size = min(abs(low_pos), abs(high_pos))
                         
                         if exit_size > 0:
                             # Place Buy Spread (direction=+1) -> Sells LowK, Buys HighK
                             if self._place_voucher_spread_order(low_strike_prod, high_strike_prod, +1, exit_size, state):
                                  pair_state.current_spread_direction = 0 # Update state


    def save(self) -> dict:
        # Save the state of each pair
        data_to_save = {}
        for pair_label, pair_state_obj in self.pair_states.items():
             data_to_save[pair_label] = pair_state_obj.to_dict()
        return data_to_save

    def load(self, data: dict) -> None:
        # Load the state for each pair
        logger.print(f"{self.symbol}: Loading state...")
        if isinstance(data, dict):
            for pair_label, pair_state_data in data.items():
                if pair_label in self.pair_states:
                    try:
                         self.pair_states[pair_label] = SlopeReversionStrategyState.from_dict(
                             pair_state_data, self.window_size, self.k, self.min_history, self.recalc_on_load
                         )
                         # logger.print(f"  Loaded state for {pair_label}: Pos={self.pair_states[pair_label].current_spread_direction}")
                    except Exception as e:
                         logger.print(f"Error loading state dict for pair {pair_label}: {e}")
                         self.pair_states[pair_label] = SlopeReversionStrategyState(self.window_size) # Reset if error
                # else: logger.print(f"Warning: Loaded state for unknown pair '{pair_label}'. Ignoring.")


# --- NEW: Generic RSI Strategy ---
class RsiStrategy(Strategy):
    """A generic RSI strategy applicable to different products."""
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {}) # Load params specific to self.symbol
        if not self.params:
            logger.print(f"ERROR: Parameters for RSI strategy on {self.symbol} not found in PARAMS. Using defaults.")
            self.params = {"rsi_window": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0, "price_offset": 0} # Fallback defaults

        # Load RSI parameters from self.params
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2:
            logger.print(f"Warning: RSI window {self.window} too small for {self.symbol}, setting to 2.")
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.price_offset = self.params.get("price_offset", 0)  # New parameter with default 0 (no offset)

        # State variables for RSI calculation
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False

        logger.print(f"Initialized RsiStrategy for {self.symbol}: Win={self.window}, OB={self.overbought_threshold}, OS={self.oversold_threshold}, Offset={self.price_offset}")

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        """Calculates RSI using Wilder's smoothing method."""
        self.mid_price_history.append(current_mid_price)

        if len(self.mid_price_history) < self.window + 1:
            return None # Need enough data points

        prices = list(self.mid_price_history)
        # Need at least 2 prices to calculate 1 change
        if len(prices) < 2: 
            return None 
        
        changes = np.diff(prices) # Use numpy.diff for efficient calculation
        
        # Ensure changes array is not empty
        if changes.size == 0: 
            return None

        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))

        # Ensure we have enough data points for the initial calculation
        if len(gains) < self.window: 
            return None 

        if not self.rsi_initialized or self.avg_gain is None or self.avg_loss is None:
            # First calculation: Use simple average over the window
            # Slice to get exactly 'window' number of changes
            self.avg_gain = np.mean(gains[-self.window:]) 
            self.avg_loss = np.mean(losses[-self.window:])
            self.rsi_initialized = True
            # logger.print(f" {self.symbol} (RSI): Initialized avg_gain={self.avg_gain:.4f}, avg_loss={self.avg_loss:.4f}")
        else:
            # Subsequent calculations: Use Wilder's smoothing
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss is not None and self.avg_loss < 1e-9: # Check for near-zero loss
             # Avoid division by zero or extreme RSI; RSI is 100 if avg_loss is 0
             return 100.0
        elif self.avg_gain is None or self.avg_loss is None:
             # Should not happen if initialized correctly, but safety check
             return None 
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth: 
            return

        # Use the base class helper to get mid price
        current_mid_price = self._get_mid_price(self.symbol, state)
        if current_mid_price is None:
             # logger.print(f"{self.symbol} (RSI): Missing mid-price.")
             return

        # Calculate RSI
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            # logger.print(f" {self.symbol} (RSI): Not enough history yet.")
            return
        # logger.print(f" {self.symbol} RSI: Mid={current_mid_price:.1f}, RSI({self.window})={rsi_value:.2f}") # Verbose logging

        # Generate Signal & Trade
        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position

        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Signal: Sell when RSI is overbought
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            if best_bid_price is not None: # Need a bid to hit
                size_to_sell = to_sell_capacity # Sell max capacity
                # Apply price offset for selling (negative direction)
                aggressive_sell_price = best_bid_price - self.price_offset
                if aggressive_sell_price <= 0: # Ensure price is positive
                    aggressive_sell_price = best_bid_price
                # logger.print(f" {self.symbol} -> RSI SELL Signal (RSI: {rsi_value:.2f} > {self.overbought_threshold}) Qty: {size_to_sell} @ Aggressive Price {aggressive_sell_price}")
                self._place_sell_order(aggressive_sell_price, size_to_sell)

        # Signal: Buy when RSI is oversold
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            if best_ask_price is not None: # Need an ask to hit
                size_to_buy = to_buy_capacity # Buy max capacity
                # Apply price offset for buying (positive direction)
                aggressive_buy_price = best_ask_price + self.price_offset
                # logger.print(f" {self.symbol} -> RSI BUY Signal (RSI: {rsi_value:.2f} < {self.oversold_threshold}) Qty: {size_to_buy} @ Aggressive Price {aggressive_buy_price}")
                self._place_buy_order(aggressive_buy_price, size_to_buy)

    def save(self) -> dict:
        # Save strategy state - include RSI state
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        }

    def load(self, data: dict) -> None:
        # Load strategy state - include RSI state
        loaded_history = data.get("mid_price_history", [])
        if isinstance(loaded_history, list):
             # Ensure history respects maxlen on load
             start_index = max(0, len(loaded_history) - (self.window + 1))
             self.mid_price_history = deque(loaded_history[start_index:], maxlen=self.window + 1)
        else:
             self.mid_price_history = deque(maxlen=self.window + 1) # Reset if invalid

        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        # Ensure avg_gain/loss are floats if loaded
        if self.avg_gain is not None:
            try: 
                self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError): 
                self.avg_gain = None
        if self.avg_loss is not None:
            try: 
                self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError): 
                self.avg_loss = None

        self.rsi_initialized = data.get("rsi_initialized", False)
        if not isinstance(self.rsi_initialized, bool): 
            self.rsi_initialized = False

        # Reset if essential state is inconsistent
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            # logger.print(f"Warning: RSI avg gain/loss invalid after loading for {self.symbol}. Resetting RSI calc state.")
            self.rsi_initialized = False
            self.avg_gain = None
            self.avg_loss = None
            # History might be okay, don't clear it unless needed

class B1B2DeviationStrategy(Strategy):
    """Trades the deviation between the actual B1-B2 spread and the theoretical B1-B2 spread."""
    def __init__(self, symbol: str, position_limits: Dict[str, int]) -> None:
        # Use the B1B2_DEVIATION symbol for the strategy key, position limit is not directly used here
        super().__init__(symbol, 0)
        self.pos_limits = position_limits # Store limits for all relevant products
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
            self.params = { # Fallback defaults
                "deviation_mean": 0, "deviation_std_window": 500,
                "zscore_threshold_entry": 2.0, "zscore_threshold_exit": 0.5,
                "target_deviation_spread_size": 10
            }
            logger.print(f"Warning: {self.symbol} params not found, using defaults.")

        # Load parameters
        self.deviation_mean = self.params.get("deviation_mean", 0)
        self.deviation_std_window = self.params.get("deviation_std_window", 500)
        if self.deviation_std_window < 10: self.deviation_std_window = 10
        self.zscore_entry = abs(self.params.get("zscore_threshold_entry", 2.0))
        self.zscore_exit = abs(self.params.get("zscore_threshold_exit", 0.5))
        self.target_size = abs(self.params.get("target_deviation_spread_size", 10))

        # State variables
        self.deviation_history: Deque[float] = deque(maxlen=self.deviation_std_window)
        # Represents the net number of deviation units we are holding
        # +ve means: Long B1, Short B2, Short 2C, Short 1J, Short 1D
        # -ve means: Short B1, Long B2, Long 2C, Long 1J, Long 1D
        self.current_effective_deviation_pos: int = 0

    # Override _place_buy/sell_order to ensure they operate on self.orders directly
    def _place_order(self, product_symbol: Symbol, price: int, quantity: int) -> None:
         if quantity == 0: return
         if price <= 0: return
         self.orders.append(Order(product_symbol, price, quantity))
         # logger.print(f"   PLACE {product_symbol} {quantity}x{price}")

    def _get_mid_price_safe(self, product: Symbol, state: TradingState) -> Optional[float]:
        """Safely gets mid price, returning None if unavailable."""
        return super()._get_mid_price(product, state) # Use base class method

    def act(self, state: TradingState) -> None:
        self.orders = [] # Crucial: Clear orders at the start of each act call for THIS strategy

        # 1. Calculate required mid-prices
        mid_b1 = self._get_mid_price_safe(Product.PICNIC_BASKET1, state)
        mid_b2 = self._get_mid_price_safe(Product.PICNIC_BASKET2, state)
        mid_c = self._get_mid_price_safe(Product.CROISSANTS, state)
        mid_j = self._get_mid_price_safe(Product.JAMS, state)
        mid_d = self._get_mid_price_safe(Product.DJEMBES, state)

        required_products_present = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2,
                                     Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        if any(prod not in state.order_depths for prod in required_products_present):
             logger.print(f"{self.symbol}: Missing order depth for one or more products. Skipping.")
             return

        if None in [mid_b1, mid_b2, mid_c, mid_j, mid_d]:
            logger.print(f"{self.symbol}: Missing mid-price for one or more components/baskets. Skipping.")
            return # Cannot calculate deviation

        # 2. Calculate actual and theoretical spreads, and the deviation
        actual_spread = mid_b1 - mid_b2
        theoretical_spread = (B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * mid_c +
                              B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * mid_j +
                              B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * mid_d)
        deviation = actual_spread - theoretical_spread
        self.deviation_history.append(deviation)

        # 3. Check if history is sufficient for Z-score calculation
        # Use a smaller fraction for min_periods check to start trading sooner
        if len(self.deviation_history) < self.deviation_std_window // 4:
             # logger.print(f"{self.symbol}: Insufficient history ({len(self.deviation_history)}/{self.deviation_std_window})")
             return

        # 4. Calculate Z-score
        current_deviation_history = list(self.deviation_history)
        deviation_std = np.std(current_deviation_history)
        if deviation_std < 1e-6: # Avoid division by zero
            logger.print(f"{self.symbol}: Deviation std dev too low ({deviation_std:.4f}). Skipping.")
            return

        z_score = (deviation - self.deviation_mean) / deviation_std
        logger.print(f"{self.symbol}: Dev={deviation:.2f}, Mean={self.deviation_mean:.2f}, Std={deviation_std:.2f}, Z={z_score:.2f}, CurrEffPos={self.current_effective_deviation_pos}")

        # 5. Determine desired position based on Z-score
        desired_effective_deviation_pos = self.current_effective_deviation_pos # Default: no change

        if z_score >= self.zscore_entry:
            # Deviation is high -> Sell Deviation (-target_size)
            desired_effective_deviation_pos = -self.target_size
        elif z_score <= -self.zscore_entry:
            # Deviation is low -> Buy Deviation (+target_size)
            desired_effective_deviation_pos = self.target_size
        else:
            # Check for exit signal ONLY if we hold a position
            if self.current_effective_deviation_pos > 0 and z_score >= -self.zscore_exit:
                # Holding Long, Z moved back up -> Close
                desired_effective_deviation_pos = 0
                logger.print(f"{self.symbol}: Exit Long Deviation signal (Z={z_score:.2f} >= {-self.zscore_exit:.2f})")
            elif self.current_effective_deviation_pos < 0 and z_score <= self.zscore_exit:
                # Holding Short, Z moved back down -> Close
                desired_effective_deviation_pos = 0
                logger.print(f"{self.symbol}: Exit Short Deviation signal (Z={z_score:.2f} <= {self.zscore_exit:.2f})")
            # Otherwise, stay in current position if between exit and entry thresholds

        # 6. Execute trades if desired position changed
        if desired_effective_deviation_pos != self.current_effective_deviation_pos:
            # logger.print(f"{self.symbol}: Target state change: {self.current_effective_deviation_pos} -> {desired_effective_deviation_pos}")
            self._execute_deviation_trade(state, desired_effective_deviation_pos)
        # else: logger.print(f"{self.symbol}: Holding position {self.current_effective_deviation_pos}")

    def _calculate_max_deviation_spread_size(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on all 5 product limits."""
        if direction == 0: return 0

        # Define quantity changes PER UNIT of deviation spread trade
        # direction > 0 (Buy Deviation): +B1, -B2, -2C, -1J, -1D
        # direction < 0 (Sell Deviation): -B1, +B2, +2C, +1J, +1D
        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction,
            Product.PICNIC_BASKET2: -direction,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * direction,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * direction,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * direction,
        }

        max_units = float('inf')

        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue # Should not happen with current def

            current_pos = state.position.get(product, 0)
            limit = self.pos_limits.get(product)
            if limit is None: # Should have limit defined
                logger.print(f"Error: Missing position limit for {product}")
                return 0
            if limit == 0: # Safety check
                logger.print(f"Warning: Limit for {product} is 0. Cannot trade deviation.")
                return 0

            if qty_change > 0: # Need to BUY this product
                capacity = limit - current_pos
            else: # Need to SELL this product
                capacity = limit + current_pos # Capacity is positive number

            if capacity < 0: capacity = 0 # Already over limit in the wrong direction

            # How many units can we trade based on this product's capacity?
            max_units_for_product = capacity // abs(qty_change)
            max_units = min(max_units, max_units_for_product)
            # logger.print(f"  Limit Check {product}: Curr={current_pos}, Lim={limit}, Chg/Unit={qty_change}, Cap={capacity}, MaxUnits={max_units_for_product}")

        final_max = max(0, int(max_units))
        # logger.print(f" Max Deviation Units Calculation (Dir={direction}): Max={final_max}")
        return final_max

    def _calculate_market_liquidity_limit(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on TOTAL market liquidity for each leg."""
        if direction == 0: return 0

        order_depths = state.order_depths
        max_units = float('inf')

        # Define quantity changes PER UNIT of deviation spread trade
        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction,
            Product.PICNIC_BASKET2: -direction,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * direction,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * direction,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * direction,
        }

        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue

            od = order_depths.get(product)
            if not od: return 0 # Cannot trade if any order book is missing

            total_available_volume = 0
            if qty_change > 0: # Need to BUY this product -> sum volume from SELL side
                if not od.sell_orders: return 0 # No liquidity
                total_available_volume = sum(abs(vol) for vol in od.sell_orders.values())
            else: # Need to SELL this product -> sum volume from BUY side
                if not od.buy_orders: return 0 # No liquidity
                total_available_volume = sum(abs(vol) for vol in od.buy_orders.values())

            if total_available_volume <= 0:
                return 0 # One leg has no liquidity

            units_fillable_for_product = total_available_volume // abs(qty_change)
            max_units = min(max_units, units_fillable_for_product)
            # logger.print(f"  Liq Check {product}: Chg/Unit={qty_change}, AvailVol={total_available_volume}, UnitsFillable={units_fillable_for_product}")

        final_max_liq = max(0, int(max_units))
        # logger.print(f" Max Deviation Units From Liquidity (Dir={direction}): Max={final_max_liq}")
        return final_max_liq

    def _place_aggressive_orders_for_leg(self, product_symbol: Symbol, total_quantity_needed: int, order_depth: OrderDepth):
        """Places orders for one leg, consuming book levels until quantity is met or liquidity runs out."""
        if total_quantity_needed == 0: return

        orders_to_place = []
        remaining_qty = abs(total_quantity_needed)

        if total_quantity_needed > 0: # Need to BUY
            if not order_depth.sell_orders: return # No liquidity
            sorted_levels = sorted(order_depth.sell_orders.items()) # Sort asks by price ascending
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break

        else: # Need to SELL
            if not order_depth.buy_orders: return # No liquidity
            sorted_levels = sorted(order_depth.buy_orders.items(), reverse=True) # Sort bids by price descending
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, -int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break

        # Add collected orders to the strategy's main list
        self.orders.extend(orders_to_place)

    def _execute_deviation_trade(self, state: TradingState, target_effective_pos: int):
        """Calculates executable size considering limits & liquidity, then places aggressive orders."""

        qty_units_to_trade = target_effective_pos - self.current_effective_deviation_pos
        if qty_units_to_trade == 0:
            return

        direction = 1 if qty_units_to_trade > 0 else -1

        # 1. Check Position Limit Constraint
        max_units_pos = self._calculate_max_deviation_spread_size(state, direction)
        if max_units_pos <= 0:
            logger.print(f"Execute: Cannot trade {direction} unit(s), blocked by position limit.")
            return

        # 2. Check Market Liquidity Constraint
        max_units_liq = self._calculate_market_liquidity_limit(state, direction)
        if max_units_liq <= 0:
            logger.print(f"Execute: Cannot trade {direction} unit(s), blocked by market liquidity.")
            return

        # 3. Determine Actual Executable Units
        actual_units_to_trade = direction * min(abs(qty_units_to_trade), max_units_pos, max_units_liq)

        if actual_units_to_trade == 0:
            # This case should ideally be caught by the checks above, but added for safety
            logger.print(f"Execute: Calculated 0 actual units to trade (Target: {target_effective_pos}, Current: {self.current_effective_deviation_pos}, MaxPos: {max_units_pos}, MaxLiq: {max_units_liq}).")
            return

        logger.print(f"Execute: Attempting to trade {actual_units_to_trade} deviation units (LimitPos: {max_units_pos}, LimitLiq: {max_units_liq}).")

        # 4. Define quantity changes based on the ACTUAL units we will trade
        final_qty_changes = {
            Product.PICNIC_BASKET1: +actual_units_to_trade,
            Product.PICNIC_BASKET2: -actual_units_to_trade,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * actual_units_to_trade,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * actual_units_to_trade,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * actual_units_to_trade,
        }

        # 5. Place aggressive orders for each leg
        order_depths = state.order_depths
        for product, final_qty_int in final_qty_changes.items():
            final_qty = int(round(final_qty_int)) # Ensure integer
            if final_qty == 0: continue

            od = order_depths.get(product)
            if not od:
                logger.print(f"Error: Order depth for {product} disappeared before placing orders!")
                # Note: This might mean the overall trade isn't perfectly hedged anymore.
                # Could potentially cancel already added orders, but becomes complex.
                continue # Skip this leg

            self._place_aggressive_orders_for_leg(product, final_qty, od)

        # 6. Update internal state *after* attempting to place all orders
        # Assume the orders will fill aggressivley up to the calculated limit
        self.current_effective_deviation_pos += actual_units_to_trade
        logger.print(f"Execute: Aggressive orders placed. New effective pos: {self.current_effective_deviation_pos}")

    def save(self) -> dict:
        # Need to save history and current effective position
        return {
            "deviation_history": list(self.deviation_history),
            "current_effective_deviation_pos": self.current_effective_deviation_pos
        }

    def load(self, data: dict) -> None:
        # Load history
        loaded_history = data.get("deviation_history", [])
        if isinstance(loaded_history, list):
             # Ensure deque uses correct maxlen from params
             self.deviation_history = deque(loaded_history, maxlen=self.deviation_std_window)
        else:
             self.deviation_history = deque(maxlen=self.deviation_std_window)

        # Load effective position
        loaded_pos = data.get("current_effective_deviation_pos", 0)
        if isinstance(loaded_pos, (int, float)):
            self.current_effective_deviation_pos = int(loaded_pos)
        else:
            self.current_effective_deviation_pos = 0
        # logger.print(f" Loaded state for {self.symbol}: History size {len(self.deviation_history)}, Eff Pos {self.current_effective_deviation_pos}")

# --- NEW: Strategy for 10500 Voucher based on Volcanic Rock RSI ---
class Voucher10500Strategy(Strategy):
    """Strategy for trading the 10500 voucher based on Volcanic Rock RSI when price is high enough."""
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        
        # Load config parameters
        self.params = PARAMS.get("VOUCHER_10500_STRATEGY", {})
        if not self.params:
            logger.print(f"Warning: Parameters for Voucher10500Strategy not found. Using defaults.")
            self.params = {
                "min_rock_price": 10200,
                "rsi_window": 106,
                "rsi_overbought": 60,
                "rsi_oversold": 40,
                "max_position_pct": 1.0,
            }
        
        # Extract parameters
        self.min_rock_price = self.params.get("min_rock_price", 10200)
        self.rsi_window = self.params.get("rsi_window", 106)
        self.rsi_overbought = self.params.get("rsi_overbought", 60)
        self.rsi_oversold = self.params.get("rsi_oversold", 40)
        self.max_position_pct = self.params.get("max_position_pct", 1.0)
        
        # RSI calculation state
        self.rock_price_history = deque(maxlen=self.rsi_window + 1)
        self.avg_gain = None
        self.avg_loss = None
        self.rsi_initialized = False
        
        logger.print(f"Initialized Voucher10500Strategy: MinRockPrice={self.min_rock_price}, RSI({self.rsi_window}) OB={self.rsi_overbought}/OS={self.rsi_oversold}, MaxPosPct={self.max_position_pct}")

    def _calculate_rsi(self, current_price: float) -> Optional[float]:
        """Calculates RSI using Wilder's smoothing method."""
        self.rock_price_history.append(current_price)

        if len(self.rock_price_history) < self.rsi_window + 1:
            return None  # Need enough data points

        prices = list(self.rock_price_history)
        if len(prices) < 2:
            return None

        changes = np.diff(prices)
        if changes.size == 0:
            return None

        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))

        if len(gains) < self.rsi_window:
            return None

        if not self.rsi_initialized or self.avg_gain is None or self.avg_loss is None:
            # First calculation: Use simple average over the window
            self.avg_gain = np.mean(gains[-self.rsi_window:])
            self.avg_loss = np.mean(losses[-self.rsi_window:])
            self.rsi_initialized = True
        else:
            # Subsequent calculations: Use Wilder's smoothing
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.rsi_window - 1)) + current_gain) / self.rsi_window
            self.avg_loss = ((self.avg_loss * (self.rsi_window - 1)) + current_loss) / self.rsi_window

        if self.avg_loss is not None and self.avg_loss < 1e-9:
            return 100.0
        elif self.avg_gain is None or self.avg_loss is None:
            return None
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        # Check if we have both products in order depths
        if (Product.VOLCANIC_ROCK not in state.order_depths or 
            self.symbol not in state.order_depths):
            return
        
        # Get mid prices
        volcanic_rock_mid = self._get_mid_price(Product.VOLCANIC_ROCK, state)
        voucher_mid = self._get_mid_price(self.symbol, state)
        
        if volcanic_rock_mid is None or voucher_mid is None:
            return
        
        # Check if volcanic rock price is high enough
        if volcanic_rock_mid < self.min_rock_price:
            logger.print(f"Voucher10500Strategy: Volcanic rock price {volcanic_rock_mid} below minimum {self.min_rock_price}. Not trading.")
            return
        
        # Calculate RSI on volcanic rock price
        rsi_value = self._calculate_rsi(volcanic_rock_mid)
        if rsi_value is None:
            return
        
        # Get current position and calculate target position
        current_position = state.position.get(self.symbol, 0)
        max_position = int(self.position_limit * self.max_position_pct)
        
        # Get order depth for voucher to place orders
        voucher_depth = state.order_depths.get(self.symbol)
        if not voucher_depth:
            return
            
        logger.print(f"Voucher10500Strategy: Rock Price={volcanic_rock_mid}, RSI={rsi_value:.1f}, Current Pos={current_position}/{max_position}")
        
        # Determine target position based on RSI signal
        target_position = 0  # Default neutral position
        
        if rsi_value < self.rsi_oversold:
            # RSI oversold - Go LONG - Target 100% of max position
            target_position = max_position
            logger.print(f"Voucher10500Strategy: LONG signal (RSI={rsi_value:.1f} < {self.rsi_oversold}). Target position: {target_position}")
        elif rsi_value > self.rsi_overbought:
            # RSI overbought - Go SHORT - Target -100% of max position
            target_position = -max_position
            logger.print(f"Voucher10500Strategy: SHORT signal (RSI={rsi_value:.1f} > {self.rsi_overbought}). Target position: {target_position}")
        else:
            # RSI in neutral zone - Exit any positions if we have them
            if current_position != 0:
                logger.print(f"Voucher10500Strategy: NEUTRAL signal (RSI={rsi_value:.1f}). Exiting position: {current_position}")
                target_position = 0
            else:
                # No position and no signal - do nothing
                return
                
        # Calculate quantity needed to reach target position
        quantity_needed = target_position - current_position
        
        if quantity_needed == 0:
            # Already at target position
            return
            
        # Execute the order to reach target position
        if quantity_needed > 0:
            # Need to BUY
            if voucher_depth.sell_orders:
                best_ask = min(voucher_depth.sell_orders.keys())
                logger.print(f"Voucher10500Strategy: BUYING {quantity_needed} @ {best_ask} to reach target position {target_position}")
                self._place_buy_order(best_ask, quantity_needed)
        else:
            # Need to SELL
            if voucher_depth.buy_orders:
                best_bid = max(voucher_depth.buy_orders.keys())
                logger.print(f"Voucher10500Strategy: SELLING {-quantity_needed} @ {best_bid} to reach target position {target_position}")
                self._place_sell_order(best_bid, -quantity_needed)

    def save(self) -> dict:
        return {
            "rock_price_history": list(self.rock_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        }

    def load(self, data: dict) -> None:
        if not data:
            return
            
        # Load history
        loaded_history = data.get("rock_price_history", [])
        if isinstance(loaded_history, list):
            start_index = max(0, len(loaded_history) - (self.rsi_window + 1))
            self.rock_price_history = deque(loaded_history[start_index:], maxlen=self.rsi_window + 1)
        
        # Load gains/losses
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        
        # Convert to proper types
        if self.avg_gain is not None:
            try:
                self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError):
                self.avg_gain = None
                
        if self.avg_loss is not None:
            try:
                self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError):
                self.avg_loss = None
        
        self.rsi_initialized = data.get("rsi_initialized", False)
        if not isinstance(self.rsi_initialized, bool):
            self.rsi_initialized = False

# --- Trader Class ---
class Trader:
    def __init__(self):
        self.position_limits = {
            Product.KELP: 50, Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50,
            Product.CROISSANTS: 250, Product.JAMS: 350, Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60, Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }

        # Define the strategy key
        VOUCHER_SLOPE_STRATEGY_KEY = "VOUCHER_SLOPE_REVERSION"

        # --- Define constants locally within __init__ ---
        LOCAL_STRIKES = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        
        # Filter out 10500 from the voucher list used for the slope strategy
        LOCAL_SLOPE_VOUCHERS = [
            Product.VOLCANIC_ROCK_VOUCHER_9500, 
            Product.VOLCANIC_ROCK_VOUCHER_9750, 
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250
        ]
        # --- End local definitions ---

        self.strategies = {
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]),
            Product.KELP: PrototypeKelpStrategy(Product.KELP, self.position_limits[Product.KELP]),
            # --- Use Generic RSI Strategy ---
            Product.SQUID_INK: RsiStrategy(Product.SQUID_INK, self.position_limits[Product.SQUID_INK]),
            Product.B1B2_DEVIATION: B1B2DeviationStrategy(Product.B1B2_DEVIATION, self.position_limits),
            VOUCHER_SLOPE_STRATEGY_KEY: VoucherSlopeReversionStrategy(
                VOUCHER_SLOPE_STRATEGY_KEY,
                self.position_limits,
                LOCAL_STRIKES, # Pass the local STRIKES map
                LOCAL_SLOPE_VOUCHERS # Pass only vouchers used by slope strategy (excludes 10500)
            ),
            # --- Use Generic RSI Strategy ---
            Product.VOLCANIC_ROCK: RsiStrategy(Product.VOLCANIC_ROCK, self.position_limits[Product.VOLCANIC_ROCK]),
            # --- Special strategy for 10500 voucher ---
            Product.VOLCANIC_ROCK_VOUCHER_10500: Voucher10500Strategy(
                Product.VOLCANIC_ROCK_VOUCHER_10500, 
                self.position_limits[Product.VOLCANIC_ROCK_VOUCHER_10500]
            )
        }
        logger.print("Trader Initialized with Modified Strategies: Slope pairs (9500-9750, 10000-10250), Special 10500 Voucher Strategy")

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
         # --- Existing Run Logic ---
         all_orders: List[Order] = [] # Collect all orders here
         conversions = 0
         trader_data_for_next_round = {}

         try:
             loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
             if not isinstance(loaded_data, dict): loaded_data = {}
         except Exception as e:
             logger.print(f"Error loading traderData: {e}")
             loaded_data = {}

         # Run strategies and collect their orders
         for strategy_key, strategy in self.strategies.items():
             # Load state for this specific strategy
             strategy_state = loaded_data.get(str(strategy_key), {}) # Use str(key) just in case
             if isinstance(strategy_state, dict):
                 try: strategy.load(strategy_state)
                 except Exception as e: logger.print(f"Error loading state for {strategy_key}: {e}")
             else: strategy.load({}) # Load empty state if not dict

             # Check if market data is available for the strategy
             market_data_available = True
             required_products = []
             if isinstance(strategy, B1B2DeviationStrategy):
                 required_products = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
             elif isinstance(strategy, VoucherSlopeReversionStrategy):
                 # Use the list stored in the strategy instance, not the global one
                 required_products = strategy.all_vouchers 
             elif str(strategy_key) in self.position_limits: # Assume other keys are single products if in limits
                 required_products = [str(strategy_key)]
             else:
                  # If strategy_key isn't a Product enum or one of the special keys, assume no data needed? Or log warning.
                  logger.print(f"Warning: Unsure how to check market data availability for strategy key: {strategy_key}")
                  # market_data_available = False # Be conservative?

             if any(prod not in state.order_depths for prod in required_products):
                 if required_products: # Only log if we actually knew which products were needed
                     logger.print(f"Strategy {strategy_key}: Market data missing for required products ({[p for p in required_products if p not in state.order_depths]}). Skipping run.")
                 market_data_available = False

             if market_data_available:
                 try:
                     # Strategy.run populates strategy.orders
                     strategy.run(state) # This calls the strategy's act method
                     all_orders.extend(strategy.orders) # Add this strategy's orders to the main list
                 except Exception as e:
                     logger.print(f"*** ERROR running {strategy_key} strategy: {e} ***");
                     import traceback; logger.print(traceback.format_exc())

             # Save state for this strategy
             try: trader_data_for_next_round[str(strategy_key)] = strategy.save()
             except Exception as e:
                  logger.print(f"Error saving state for {strategy_key}: {e}")
                  trader_data_for_next_round[str(strategy_key)] = {}

         # Group orders by symbol for the final output format
         final_result: Dict[Symbol, List[Order]] = defaultdict(list)
         for order in all_orders:
             # --- Ensure quantity is integer ---
             if not isinstance(order.quantity, int):
                 logger.print(f"Warning: Order quantity was not int for {order.symbol}: {order.quantity}. Rounding.")
                 order.quantity = int(round(order.quantity))
             # --- Ensure price is integer ---
             if not isinstance(order.price, int):
                 logger.print(f"Warning: Order price was not int for {order.symbol}: {order.price}. Rounding.")
                 order.price = int(round(order.price))

             if order.quantity != 0: # Don't submit zero quantity orders
                 final_result[order.symbol].append(order)

         # Encode data & flush logs
         try:
              # Ensure keys are strings for JSON
              trader_data_to_encode = {str(k): v for k, v in trader_data_for_next_round.items()}
              traderData_encoded = json.dumps(trader_data_to_encode, separators=(",", ":"), cls=ProsperityEncoder)
         except Exception as e:
             logger.print(f"Error encoding traderData: {e}")
             traderData_encoded = "{}"

         # Use the imported logger instance
         logger.flush(state, dict(final_result), conversions, traderData_encoded) # Pass the dict version
         return dict(final_result), conversions, traderData_encoded 