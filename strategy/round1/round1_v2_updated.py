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
# 6. Fair Value: Stable 10000 for Resin, popular price midpoint for Kelp/Squid Ink with enhanced MA/EWMA.
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
import jsonpickle
import math

# Ensure datamodel contents are accessible
from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# --- Logger Class (Unchanged) ---
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

        max_item_length = (self.max_log_length - base_length) // 3
        if max_item_length < 0: max_item_length = 0

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData if state.traderData else "", max_item_length)),
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
        if trades and isinstance(trades, dict):
            for arr in trades.values():
                if arr and isinstance(arr, (list, tuple)):
                    for trade in arr:
                        compressed.append([
                            trade.symbol,
                            trade.price,
                            trade.quantity,
                            trade.buyer,
                            trade.seller,
                            trade.timestamp,
                        ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        if observations and hasattr(observations, 'conversionObservations') and isinstance(observations.conversionObservations, dict):
            for product, observation in observations.conversionObservations.items():
                if observation:
                    conversion_observations[product] = [
                        getattr(observation, 'bidPrice', None),
                        getattr(observation, 'askPrice', None),
                        getattr(observation, 'transportFees', None),
                        getattr(observation, 'exportTariff', None),
                        getattr(observation, 'importTariff', None),
                        getattr(observation, 'sugarPrice', None),
                        getattr(observation, 'sunlightIndex', None),
                    ]
        plain_obs = observations.plainValueObservations if observations and hasattr(observations, 'plainValueObservations') else {}
        return [plain_obs, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders and isinstance(orders, dict):
            for arr in orders.values():
                if arr and isinstance(arr, (list, tuple)):
                    for order in arr:
                        compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
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

# Updated PARAMS dictionary
PARAMS = {
    Product.KELP: {
        "take_width": 2,        # Wider take to capture more opportunities
        "clear_width": 0.5,     # Tighter clear to reduce holding costs
        "prevent_adverse": True,
        "adverse_volume": 20,   # Slightly higher threshold
        "disregard_edge": 0.5,  # Tighter edge for pennying/joining
        "join_edge": 0.5,       # Allow joining closer to fair value
        "default_edge": 1,      # Tighter spread for aggressive market-making
        "soft_position_limit": 25, # Lower to trigger adjustments quicker
        "window_size": 10        # Larger window for trend detection
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 0,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 0,
        "soft_position_limit": 40,
        "window_size": 5
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "disregard_edge": 0.5,
        "join_edge": 0.5,
        "default_edge": 1,
        "soft_position_limit": 25,
        "window_size": 10
    },
}

class BaseStrategy:
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        self.symbol = symbol
        self.params = params
        self.position_limit = position_limit
        self.orders: List[Order] = []
        self.position_window = deque(maxlen=params.get("window_size", 5))

    @abstractmethod
    def get_fair_value(self, state: TradingState) -> float | None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            logger.print(f"Warning: No order depth data for {self.symbol}")
            return

        order_depth = state.order_depths[self.symbol]
        current_buy_orders = order_depth.buy_orders.copy() if isinstance(order_depth.buy_orders, dict) else {}
        current_sell_orders = order_depth.sell_orders.copy() if isinstance(order_depth.sell_orders, dict) else {}
        sim_order_depth = OrderDepth()
        sim_order_depth.buy_orders = current_buy_orders
        sim_order_depth.sell_orders = current_sell_orders

        position = state.position.get(self.symbol, 0)
        buy_order_volume = 0
        sell_order_volume = 0

        self.position_window.append(abs(position) == self.position_limit)
        is_stuck_soft = len(self.position_window) == self.position_window.maxlen and sum(self.position_window) >= self.position_window.maxlen / 2 and self.position_window[-1]
        is_stuck_hard = len(self.position_window) == self.position_window.maxlen and all(self.position_window)

        fair_value = self.get_fair_value(state)
        if fair_value is None:
            logger.print(f"Warning: Could not calculate fair value for {self.symbol}. Skipping.")
            return
        logger.print(f"{self.symbol} - Fair Value: {fair_value:.2f}, Position: {position}/{self.position_limit}")

        take_orders, buy_order_volume, sell_order_volume = self._take_orders(sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume)
        self.orders.extend(take_orders)

        clear_orders, buy_order_volume, sell_order_volume = self._clear_orders(sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume, is_stuck_hard)
        self.orders.extend(clear_orders)

        make_orders, _, _ = self._make_orders(sim_order_depth, fair_value, position, buy_order_volume, sell_order_volume)
        self.orders.extend(make_orders)

    def _take_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        orders = []
        take_width = self.params["take_width"]
        prevent_adverse = self.params.get("prevent_adverse", False)
        adverse_volume = self.params.get("adverse_volume", 0)
        position_limit = self.position_limit

        if isinstance(order_depth.sell_orders, dict):
            sell_order_prices = sorted(order_depth.sell_orders.keys())
            for price in sell_order_prices:
                if price <= fair_value - take_width:
                    available_volume = -order_depth.sell_orders[price]
                    if prevent_adverse and available_volume > adverse_volume:
                        logger.print(f"Skipping adverse sell order at {price} ({available_volume} > {adverse_volume})")
                        continue

                    qty_to_buy = min(available_volume, position_limit - (position + buy_volume))
                    if qty_to_buy <= 0: break

                    logger.print(f"{self.symbol} - Taking Sell: {qty_to_buy}x{price}")
                    orders.append(Order(self.symbol, price, qty_to_buy))
                    buy_volume += qty_to_buy
                    order_depth.sell_orders[price] += qty_to_buy
                    if order_depth.sell_orders[price] == 0: del order_depth.sell_orders[price]
                else:
                    break

        if isinstance(order_depth.buy_orders, dict):
            buy_order_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for price in buy_order_prices:
                if price >= fair_value + take_width:
                    available_volume = order_depth.buy_orders[price]
                    if prevent_adverse and available_volume > adverse_volume:
                        logger.print(f"Skipping adverse buy order at {price} ({available_volume} > {adverse_volume})")
                        continue

                    qty_to_sell = min(available_volume, position_limit + (position - sell_volume))
                    if qty_to_sell <= 0: break

                    logger.print(f"{self.symbol} - Taking Buy: {qty_to_sell}x{price}")
                    orders.append(Order(self.symbol, price, -qty_to_sell))
                    sell_volume += qty_to_sell
                    order_depth.buy_orders[price] -= qty_to_sell
                    if order_depth.buy_orders[price] == 0: del order_depth.buy_orders[price]
                else:
                    break

        return orders, buy_volume, sell_volume

    def _clear_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int, is_stuck: bool) -> tuple[List[Order], int, int]:
        orders = []
        soft_limit = self.params["soft_position_limit"]
        clear_width = self.params["clear_width"]
        position_after_take = position + buy_volume - sell_volume
        position_limit = self.position_limit

        if position_after_take > soft_limit:
            qty_to_clear = position_after_take - soft_limit
            if is_stuck: qty_to_clear = position_after_take
            sell_price_target = round(fair_value - clear_width)
            logger.print(f"{self.symbol} - Clearing Long Position: Attempting {qty_to_clear} @ >= {sell_price_target}")

            if isinstance(order_depth.buy_orders, dict):
                buy_order_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                for price in buy_order_prices:
                    if price >= sell_price_target and qty_to_clear > 0:
                        available_volume = order_depth.buy_orders[price]
                        sell_now = min(qty_to_clear, available_volume, position_limit + (position - sell_volume))
                        if sell_now <= 0: continue

                        logger.print(f"{self.symbol} - Clearing Sell: {sell_now}x{price}")
                        orders.append(Order(self.symbol, price, -sell_now))
                        sell_volume += sell_now
                        qty_to_clear -= sell_now
                        order_depth.buy_orders[price] -= sell_now
                        if order_depth.buy_orders[price] == 0: del order_depth.buy_orders[price]
                    if qty_to_clear <= 0: break

        elif position_after_take < -soft_limit:
            qty_to_clear = abs(position_after_take) - soft_limit
            if is_stuck: qty_to_clear = abs(position_after_take)
            buy_price_target = round(fair_value + clear_width)
            logger.print(f"{self.symbol} - Clearing Short Position: Attempting {qty_to_clear} @ <= {buy_price_target}")

            if isinstance(order_depth.sell_orders, dict):
                sell_order_prices = sorted(order_depth.sell_orders.keys())
                for price in sell_order_prices:
                    if price <= buy_price_target and qty_to_clear > 0:
                        available_volume = -order_depth.sell_orders[price]
                        buy_now = min(qty_to_clear, available_volume, position_limit - (position + buy_volume))
                        if buy_now <= 0: continue

                        logger.print(f"{self.symbol} - Clearing Buy: {buy_now}x{price}")
                        orders.append(Order(self.symbol, price, buy_now))
                        buy_volume += buy_now
                        qty_to_clear -= buy_now
                        order_depth.sell_orders[price] += buy_now
                        if order_depth.sell_orders[price] == 0: del order_depth.sell_orders[price]
                    if qty_to_clear <= 0: break

        return orders, buy_volume, sell_volume

    def _make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        orders = []
        disregard_edge = self.params["disregard_edge"]
        join_edge = self.params["join_edge"]
        default_edge = self.params["default_edge"]
        soft_limit = self.params["soft_position_limit"]
        position_limit = self.position_limit

        asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge] if isinstance(order_depth.sell_orders, dict) else []
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge] if isinstance(order_depth.buy_orders, dict) else []

        best_ask_to_consider = min(asks_above_fair) if asks_above_fair else None
        best_bid_to_consider = max(bids_below_fair) if bids_below_fair else None

        ask_price = round(fair_value + default_edge)
        if best_ask_to_consider is not None:
            if abs(best_ask_to_consider - fair_value) <= join_edge:
                ask_price = best_ask_to_consider
            else:
                ask_price = best_ask_to_consider - 1

        bid_price = round(fair_value - default_edge)
        if best_bid_to_consider is not None:
            if abs(fair_value - best_bid_to_consider) <= join_edge:
                bid_price = best_bid_to_consider
            else:
                bid_price = best_bid_to_consider + 1

        current_position_after_clears = position + buy_volume - sell_volume
        if current_position_after_clears > soft_limit:
            bid_price -= 1
            ask_price -= 1
        elif current_position_after_clears < -soft_limit:
            bid_price += 1
            ask_price += 1

        bid_price = min(bid_price, ask_price - 1)

        buy_capacity = position_limit - current_position_after_clears
        if buy_capacity > 0:
            final_bid_price = self._adjust_price_level(bid_price)
            logger.print(f"{self.symbol} - Making Buy: {buy_capacity}x{final_bid_price}")
            orders.append(Order(self.symbol, final_bid_price, buy_capacity))

        sell_capacity = position_limit + current_position_after_clears
        if sell_capacity > 0:
            final_ask_price = self._adjust_price_level(ask_price)
            if final_ask_price <= final_bid_price:
                final_ask_price = final_bid_price + 1
            final_ask_price = self._adjust_price_level(final_ask_price)

            logger.print(f"{self.symbol} - Making Sell: {sell_capacity}x{final_ask_price}")
            orders.append(Order(self.symbol, final_ask_price, -sell_capacity))

        return orders, buy_volume, sell_volume

    def _adjust_price_level(self, price: int) -> int:
        return price

    def save(self) -> Dict:
        return {"position_window": list(self.position_window)}

    def load(self, data: Dict) -> None:
        if data and "position_window" in data and isinstance(data["position_window"], list):
            loaded_window = data["position_window"]
            start_index = max(0, len(loaded_window) - self.position_window.maxlen)
            self.position_window.clear()
            self.position_window.extend(loaded_window[start_index:])

class RainforestResinStrategy(BaseStrategy):
    def get_fair_value(self, state: TradingState) -> float | None:
        return self.params.get("fair_value", 10000)

    def _adjust_price_level(self, price: int) -> int:
        return price

    def _make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> tuple[List[Order], int, int]:
        orders = []
        position_limit = self.position_limit
        soft_limit = self.params["soft_position_limit"]
        make_bid_price = int(fair_value - 1)
        make_ask_price = int(fair_value + 1)

        current_position_after_clears = position + buy_volume - sell_volume

        if current_position_after_clears > soft_limit:
            make_bid_price -= 1
            make_ask_price -= 1
        elif current_position_after_clears < -soft_limit:
            make_bid_price += 1
            make_ask_price += 1

        make_bid_price = min(make_bid_price, make_ask_price - 1)

        buy_capacity = position_limit - current_position_after_clears
        if buy_capacity > 0:
            logger.print(f"{self.symbol} - Making Buy: {buy_capacity}x{make_bid_price}")
            orders.append(Order(self.symbol, make_bid_price, buy_capacity))

        sell_capacity = position_limit + current_position_after_clears
        if sell_capacity > 0:
            final_ask_price = max(make_ask_price, int(fair_value + 1))
            if final_ask_price <= make_bid_price:
                final_ask_price = make_bid_price + 1

            logger.print(f"{self.symbol} - Making Sell: {sell_capacity}x{final_ask_price}")
            orders.append(Order(self.symbol, final_ask_price, -sell_capacity))

        return orders, buy_volume, sell_volume

class VolatileStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict, position_limit: int) -> None:
        super().__init__(symbol, params, position_limit)
        self.price_history = deque(maxlen=params["window_size"])

    def get_fair_value(self, state: TradingState) -> float | None:
        if self.symbol not in state.order_depths:
            return None
        order_depth = state.order_depths[self.symbol]

        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid = max(buy_orders.keys()) if buy_orders else None
        best_ask = min(sell_orders.keys()) if sell_orders else None

        if best_bid is None or best_ask is None:
            return None

        mid_price = (best_bid + best_ask) / 2.0
        self.price_history.append(mid_price)

        if len(self.price_history) < 2:
            return mid_price

        window = self.params["window_size"]
        ma = sum(self.price_history) / len(self.price_history) if self.price_history else mid_price

        alpha = 2 / (self.params["window_size"] + 1)
        ewma = mid_price
        for price in reversed(list(self.price_history)[:-1]):
            ewma = alpha * price + (1 - alpha) * ewma

        fair_value = 0.6 * ma + 0.4 * ewma

        imbalance = self._calculate_imbalance(order_depth)
        if imbalance is not None:
            fair_value += imbalance * 0.1

        return fair_value

    def _calculate_imbalance(self, order_depth: OrderDepth) -> float | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        total_bid_vol = sum(order_depth.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        if total_bid_vol + total_ask_vol == 0:
            return 0

        return (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

class KelpStrategy(VolatileStrategy):
    pass

class SquidInkStrategy(VolatileStrategy):
    pass

class Trader:
    def __init__(self):
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50
        }

        self.strategies = {
            Product.KELP: KelpStrategy(Product.KELP, PARAMS[Product.KELP], self.position_limits[Product.KELP]),
            Product.RAINFOREST_RESIN: RainforestResinStrategy(Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN], self.position_limits[Product.RAINFOREST_RESIN]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, PARAMS[Product.SQUID_INK], self.position_limits[Product.SQUID_INK])
        }

        logger.print("Trader Initialized with strategies for:", list(self.strategies.keys()))

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        logger.print(f"--- Iteration {state.timestamp} ---")
        result: Dict[Symbol, List[Order]] = {}
        trader_data_for_next_round = {}
        conversions = 0

        try:
            loaded_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(loaded_data, dict):
                logger.print("Warning: traderData was not a dict, ignoring.")
                loaded_data = {}
        except Exception as e:
            logger.print(f"Error decoding traderData with json.loads: {e}. Starting fresh.")
            loaded_data = {}

        for product, strategy in self.strategies.items():
            logger.print(f"Running strategy for {product}...")
            if product in loaded_data and isinstance(loaded_data[product], dict):
                strategy.load(loaded_data[product])
            else:
                logger.print(f"No previous state found for {product}.")

            if product in state.order_depths:
                orders = strategy.run(state)
                if orders:
                    result[product] = orders
                    logger.print(f"  {product} Orders: {[(o.price, o.quantity) for o in orders]}")
            else:
                logger.print(f"  Skipping {product}, no order depth data in current state.")

            trader_data_for_next_round[product] = strategy.save()

        try:
            traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"))
        except Exception as e:
            logger.print(f"Error encoding traderData with json.dumps: {e}. Sending empty data.")
            traderData_encoded = "{}"

        logger.flush(state, result, conversions, traderData_encoded)

        return result, conversions, traderData_encoded