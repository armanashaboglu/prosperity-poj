from datamodel import Order, OrderDepth, TradingState, Symbol
import math
import numpy as np
import json
import jsonpickle
from typing import List, Tuple, Dict, Any
from collections import deque

# ---------------------------
# Logger (based on round1rsi implementation)
# ---------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, traderData: str) -> None:
        # For simplicity, encode state info along with orders and logs.
        output = jsonpickle.encode(
            [state.timestamp, orders, conversions, traderData, self.logs],
            unpicklable=False
        )
        print(output)
        self.logs = ""

logger = Logger()

# ---------------------------
# Product and Parameters
# ---------------------------
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    Product.KELP: {
        "take_width": 1,
        "position_limit": 50,
        "min_volume_filter": 20,
        "spread_edge": 1,
        "default_fair_method": "vwap_with_vol_filter",
    },
    # SQUID_INK parameters from round1v3:
    Product.SQUID_INK: {
        "rsi_period": 37,
        "rsi_overbought": 65,
        "rsi_oversold": 28,
        "rsi_trade_size": 18,
    },
}

LIMIT = {
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
}

# ---------------------------
# RAINFOREST_RESIN Strategy Functions (from round1rsi)
# ---------------------------
def resin_take_orders(
    order_depth: OrderDepth, fair_value: float, position: int, position_limit: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0

    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_ask, quantity))
                buy_order_volume += quantity

    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_bid, -quantity))
                sell_order_volume += quantity

    return orders, buy_order_volume, sell_order_volume

def resin_clear_orders(
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = position_limit - (position + buy_order_volume)
    sell_quantity = position_limit + (position - sell_order_volume)

    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)

    return orders, buy_order_volume, sell_order_volume

def resin_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
    bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
    baaf = min(aaf) if aaf else fair_value + 2
    bbbf_val = max(bbbf) if bbbf else fair_value - 2

    buy_quantity = position_limit - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, bbbf_val + 1, buy_quantity))

    sell_quantity = position_limit + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, baaf - 1, -sell_quantity))

    return orders

# ---------------------------
# KELP Strategy Functions (from round1rsi)
# ---------------------------
def kelp_fair_value(
    order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20
) -> float:
    if method == "mid_price":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif method == "mid_price_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
        return (best_ask + best_bid) / 2
    elif method == "vwap":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        if volume == 0:
            return (best_ask + best_bid) / 2
        return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    elif method == "vwap_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    else:
        raise ValueError("Unknown fair value method specified.")

def kelp_take_orders(
    order_depth: OrderDepth, fair_value: float, params: dict, position: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0

    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
            quantity = min(ask_amount, params["position_limit"] - position)
            if quantity > 0:
                orders.append(Order(Product.KELP, best_ask, quantity))
                buy_order_volume += quantity

    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        bid_amount = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
            quantity = min(bid_amount, params["position_limit"] + position)
            if quantity > 0:
                orders.append(Order(Product.KELP, best_bid, -quantity))
                sell_order_volume += quantity

    return orders, buy_order_volume, sell_order_volume

def kelp_clear_orders(
    order_depth: OrderDepth,
    position: int,
    params: dict,
    fair_value: float,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    sell_quantity = params["position_limit"] + (position - sell_order_volume)

    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(Product.KELP, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.KELP, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)

    return orders, buy_order_volume, sell_order_volume

def kelp_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    params: dict,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    edge = params["spread_edge"]
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
    bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
    baaf = min(aaf) if aaf else fair_value + edge + 1
    bbbf = max(bbf) if bbf else fair_value - edge - 1

    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.KELP, bbbf + 1, buy_quantity))

    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.KELP, baaf - 1, -sell_quantity))

    return orders

# ---------------------------
# SQUID_INK Strategy using RSI (from round1v3)
# ---------------------------
class BaseStrategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    def _place_buy_order(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}")

    def _place_sell_order(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}")

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

class SquidInkRSIStrategy(BaseStrategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.rsi_params = PARAMS[Product.SQUID_INK]
        self.price_history: deque[float] = deque(maxlen=self.rsi_params["rsi_period"] + 1)
        self.rsi_value: float = 50.0

    def _calculate_rsi(self) -> float:
        period = self.rsi_params["rsi_period"]
        if len(self.price_history) <= period:
            return 50.0
        prices = np.array(list(self.price_history))
        deltas = np.diff(prices)
        if len(deltas) < period:
            return 50.0
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
        order_depth = state.order_depths[self.symbol]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid = max(buy_orders.keys()) if buy_orders else None
        best_ask = min(sell_orders.keys()) if sell_orders else None

        current_price = None
        if best_bid is not None and best_ask is not None:
            current_price = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            current_price = float(best_bid)
        elif best_ask is not None:
            current_price = float(best_ask)

        if current_price is None:
            logger.print(f"Cannot determine price for {self.symbol}, skipping RSI step.")
            return

        self.price_history.append(current_price)
        self.rsi_value = self._calculate_rsi()
        logger.print(f"{self.symbol} - RSI({self.rsi_params['rsi_period']}): {self.rsi_value:.2f}")

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.position_limit - position
        sell_capacity = self.position_limit + position
        trade_size = self.rsi_params["rsi_trade_size"]

        overbought_level = self.rsi_params["rsi_overbought"]
        oversold_level = self.rsi_params["rsi_oversold"]

        if self.rsi_value > overbought_level and sell_capacity > 0 and best_bid is not None:
            qty = min(trade_size, sell_capacity)
            logger.print(f"{self.symbol} RSI Overbought ({self.rsi_value:.1f} > {overbought_level}), Selling {qty} at best bid {best_bid}")
            self._place_sell_order(best_bid, qty)
        elif self.rsi_value < oversold_level and buy_capacity > 0 and best_ask is not None:
            qty = min(trade_size, buy_capacity)
            logger.print(f"{self.symbol} RSI Oversold ({self.rsi_value:.1f} < {oversold_level}), Buying {qty} at best ask {best_ask}")
            self._place_buy_order(best_ask, qty)

    def run(self, state: TradingState) -> List[Order]:
        # Reset orders each run
        self.orders = []
        self.act(state)
        return self.orders

    def save(self) -> dict:
        return {"price_history": list(self.price_history), "rsi_value": self.rsi_value}

    def load(self, data: dict) -> None:
        if data and "price_history" in data:
            loaded_history = data["price_history"]
            maxlen = self.rsi_params["rsi_period"] + 1
            start_index = max(0, len(loaded_history) - maxlen)
            self.price_history = deque(loaded_history[start_index:], maxlen=maxlen)
        self.rsi_value = data.get("rsi_value", 50.0)

# ---------------------------
# Unified Trader for round1v4
# ---------------------------
class Trader:
    def __init__(self, params: Dict[str, Any] = None) -> None:
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = LIMIT
        # Instantiate the SQUID_INK strategy from round1v3 (with RSI)
        self.squid_strategy = SquidInkRSIStrategy(Product.SQUID_INK, self.LIMIT[Product.SQUID_INK])

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # --- RAINFOREST_RESIN using round1rsi strategy functions ---
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_params = self.params[Product.RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]

            orders_take, bo, so = resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RAINFOREST_RESIN]
            )
            orders_clear, bo, so = resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, self.LIMIT[Product.RAINFOREST_RESIN], bo, so
            )
            orders_make = resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RAINFOREST_RESIN], bo, so
            )
            result[Product.RAINFOREST_RESIN] = orders_take + orders_clear + orders_make

        # --- KELP using round1rsi strategy functions ---
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_params = self.params[Product.KELP]
            kelp_order_depth = state.order_depths[Product.KELP]
            kelp_fair = kelp_fair_value(
                kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"]
            )
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, kelp_fair, kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, kelp_fair, bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, kelp_fair, kelp_position, kelp_params, bo, so)
            result[Product.KELP] = kelp_take + kelp_clear + kelp_make

        # --- SQUID_INK using the RSI strategy ---
        if Product.SQUID_INK in state.order_depths:
            try:
                loaded_data = json.loads(state.traderData) if state.traderData else {}
                squid_data = loaded_data.get(Product.SQUID_INK, {})
                self.squid_strategy.load(squid_data)
            except Exception as e:
                logger.print(f"Error loading SQUID_INK strategy state: {e}")
                self.squid_strategy.load({})
            orders_squid = self.squid_strategy.run(state)
            result[Product.SQUID_INK] = orders_squid

        traderData = jsonpickle.encode(
            {Product.SQUID_INK: self.squid_strategy.save()}, unpicklable=False
        )
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
