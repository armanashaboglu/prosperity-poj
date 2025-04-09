import json
import numpy as np
from typing import Any, Dict, List, Deque, Optional
from abc import abstractmethod
from collections import deque
import math
import copy

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# --- Logger Class (Copied from v3) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        dummy_state_data = [ state.timestamp, "", [], {}, [], [], {}, [] ]
        dummy_orders = []

        try:
            base_json = self.to_json([dummy_state_data, dummy_orders, conversions, "", ""])
            base_length = len(base_json)

            max_item_length = (self.max_log_length - base_length) // 3
            if max_item_length < 0: max_item_length = 0

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
            print(json.dumps([state.timestamp, conversions, f"Log Flush Error: {e}"])) # Fallback

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
         compressed_order_depths = self.compress_order_depths(state.order_depths)
         return [
             state.timestamp,
             trader_data,
             self.compress_listings(state.listings),
             compressed_order_depths,
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
                            getattr(observation, 'bidPrice', None),
                            getattr(observation, 'askPrice', None),
                            getattr(observation, 'transportFees', None),
                            getattr(observation, 'exportTariff', None),
                            getattr(observation, 'importTariff', None),
                            getattr(observation, 'sunlightIndex', None),
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
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"), default=str)

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length:
            return value
        if max_length < 3: return value[:max_length]
        return value[: max_length - 3] + "..."

logger = Logger()

# --- Product Class ---
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

# --- Parameters ---
PARAMS = {
    Product.SQUID_INK: {
        "rsi_period": 29,
        "rsi_overbought": 74,
        "rsi_oversold": 29,
        "rsi_trade_size": 5
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000
    }
}

# --- Base Strategy (Generic - kept for RSI) ---
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
        logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}")

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}")

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

# --- V3 Strategy Classes (Copied from round1_v3.py and renamed) ---
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
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"{self.symbol} BUY {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
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

    def act(self, state: TradingState) -> None:
        # Base MM act from v3 - included for completeness but overridden by V3RainforestResinStrategy
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.print(f"Skipping {self.symbol}: No orders on one or both sides of the book")
            return

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        true_value = self.get_true_value(state)
        logger.print(f"{self.symbol} - true value: {true_value}, position: {position}/{self.position_limit}")

        self.window.append(abs(position) == self.position_limit)

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.position_limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.position_limit * 0.5 else true_value

        logger.print(f"{self.symbol} - max buy: {max_buy_price}, min sell: {min_sell_price}")

        # BUYING LOGIC
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

        # SELLING LOGIC
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
        if not order_depth:
            return

        initial_buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        initial_sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
        sim_buy_orders = copy.deepcopy(initial_buy_orders)
        sim_sell_orders = copy.deepcopy(initial_sell_orders)

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        logger.print(f"{self.symbol} (V3 Logic) -> TV: {true_value}, Pos: {position}/{self.position_limit}, BuyCap: {to_buy}, SellCap: {to_sell}")

        # Phase 1: Take orders strictly better than or equal to fair value
        asks_sorted = sorted(sim_sell_orders.items())
        for price, volume in asks_sorted:
            if to_buy <= 0: break
            volume = -volume
            if price <= true_value:
                qty_to_take = min(to_buy, volume)
                self.buy(price, qty_to_take)
                to_buy -= qty_to_take
                sim_sell_orders[price] += qty_to_take
                if sim_sell_orders[price] == 0:
                    del sim_sell_orders[price]
            else:
                break

        bids_sorted = sorted(sim_buy_orders.items(), reverse=True)
        for price, volume in bids_sorted:
            if to_sell <= 0: break
            if price >= true_value:
                qty_to_take = min(to_sell, volume)
                self.sell(price, qty_to_take)
                to_sell -= qty_to_take
                sim_buy_orders[price] -= qty_to_take
                if sim_buy_orders[price] == 0:
                    del sim_buy_orders[price]
            else:
                break

        logger.print(f"{self.symbol} (V3 Logic) Phase 1 DONE - RemBuyCap: {to_buy}, RemSellCap: {to_sell}")

        # Phase 2: Make markets based on remaining simulated book
        make_bid_price = true_value - 1
        bids_below_10k = {p: v for p, v in sim_buy_orders.items() if p < true_value}
        if bids_below_10k:
            best_bid_after_take = max(bids_below_10k.keys())
            best_bid_vol = bids_below_10k[best_bid_after_take]
            if best_bid_vol <= 6:
                 make_bid_price = best_bid_after_take
            else:
                 make_bid_price = best_bid_after_take + 1

        make_ask_price = true_value + 1
        asks_above_10k = {p: v for p, v in sim_sell_orders.items() if p > true_value}
        if asks_above_10k:
            best_ask_after_take = min(asks_above_10k.keys())
            best_ask_vol = abs(asks_above_10k[best_ask_after_take])
            if best_ask_vol <= 6:
                make_ask_price = best_ask_after_take
            else:
                make_ask_price = best_ask_after_take - 1

        # Avoid specific price levels
        if make_bid_price == 9999: make_bid_price = 9998
        elif make_bid_price == 9997: make_bid_price = 9996
        if make_ask_price == 10001: make_ask_price = 10002
        elif make_ask_price == 10003: make_ask_price = 10004

        # Ensure bid < ask
        if make_bid_price >= make_ask_price:
            make_ask_price = make_bid_price + 1
            if make_ask_price == 10001: make_ask_price = 10002
            elif make_ask_price == 10003: make_ask_price = 10004

        if to_buy > 0:
            self.buy(make_bid_price, to_buy)
        if to_sell > 0:
            self.sell(make_ask_price, to_sell)

        logger.print(f"{self.symbol} (V3 Logic) Phase 2 DONE - Placed Make orders at {make_bid_price}/{make_ask_price}")
# --- End V3 Strategy Classes ---


# --- Prototype Market Making Strategy (For Kelp) ---
class PrototypeMarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window_size = 4 # Hardcoded from prototype
        self.window: Deque[bool] = deque()

    def buy(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PROTO_BUY {self.symbol} {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PROTO_SELL {self.symbol} {quantity}x{price}")

    @abstractmethod
    def get_true_value(self, state: TradingState) -> Optional[int]:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            logger.print(f"No order depth data for {self.symbol} (Proto Logic), skipping.")
            return

        buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}

        if not buy_orders_dict and not sell_orders_dict:
            logger.print(f"No orders for {self.symbol} (Proto Logic), skipping.")
            return

        position = state.position.get(self.symbol, 0)
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        true_value = self.get_true_value(state)
        if true_value is None:
             logger.print(f"Could not determine true value for {self.symbol} (Proto Logic), skipping.")
             return

        logger.print(f"{self.symbol} (Proto Logic) -> TV: {true_value}, Pos: {position}/{self.position_limit}, BuyCap: {to_buy}, SellCap: {to_sell}")

        soft_limit_threshold = self.position_limit * 0.5
        max_buy_price = true_value - 1 if position > soft_limit_threshold else true_value
        min_sell_price = true_value + 1 if position < -soft_limit_threshold else true_value
        logger.print(f"{self.symbol} (Proto Logic) -> Boundaries - MaxBuy: {max_buy_price}, MinSell: {min_sell_price}")

        # Phase 1: Take orders
        sell_orders_list = sorted(sell_orders_dict.items()) if sell_orders_dict else []
        for price, volume in sell_orders_list:
            volume = -volume
            if to_buy > 0 and price <= max_buy_price:
                qty_to_take = min(to_buy, volume)
                self.buy(price, qty_to_take)
                to_buy -= qty_to_take

        buy_orders_list = sorted(buy_orders_dict.items(), reverse=True) if buy_orders_dict else []
        for price, volume in buy_orders_list:
            if to_sell > 0 and price >= min_sell_price:
                qty_to_take = min(to_sell, volume)
                self.sell(price, qty_to_take)
                to_sell -= qty_to_take

        # Phase 2: Liquidation
        self.window.append(abs(position) == self.position_limit)
        # Check if window has correct size before using maxlen - use self.window_size
        while len(self.window) > self.window_size:
            self.window.popleft()

        is_full_window = len(self.window) == self.window_size
        stuck_count = sum(self.window)
        soft_liquidate = is_full_window and stuck_count >= self.window_size / 2 and self.window[-1]
        hard_liquidate = is_full_window and stuck_count == self.window_size

        hard_liq_offset = 0
        soft_liq_offset = 2

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            liq_price = true_value + hard_liq_offset
            self.buy(liq_price, quantity)
            to_buy -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Hard Liq Buy {quantity}x{liq_price:.0f}")
        elif to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            liq_price = true_value + soft_liq_offset
            self.buy(liq_price, quantity)
            to_buy -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Soft Liq Buy {quantity}x{liq_price:.0f}")

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            liq_price = true_value - hard_liq_offset
            self.sell(liq_price, quantity)
            to_sell -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Hard Liq Sell {quantity}x{liq_price:.0f}")
        elif to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            liq_price = true_value - soft_liq_offset
            self.sell(liq_price, quantity)
            to_sell -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Soft Liq Sell {quantity}x{liq_price:.0f}")


        # Phase 3: Make orders
        if to_buy > 0:
            buy_orders_list_for_make = sorted(buy_orders_dict.items(), reverse=True) if buy_orders_dict else []
            popular_buy_price = max(buy_orders_list_for_make, key=lambda tup: tup[1])[0] if buy_orders_list_for_make else (true_value - 2)
            make_price = min(max_buy_price, popular_buy_price + 1)
            self.buy(make_price, to_buy)

        if to_sell > 0:
            sell_orders_list_for_make = sorted(sell_orders_dict.items()) if sell_orders_dict else []
            popular_sell_price = min(sell_orders_list_for_make, key=lambda tup: tup[1])[0] if sell_orders_list_for_make else (true_value + 2)
            make_price = max(min_sell_price, popular_sell_price - 1)
            self.sell(make_price, to_sell)

    def save(self) -> dict:
        return {"window": list(self.window)}

    def load(self, data: dict) -> None:
        if data and "window" in data and isinstance(data["window"], list):
            loaded_window = data["window"]
            self.window = deque(loaded_window) # Re-initialize deque
            while len(self.window) > self.window_size:
                 self.window.popleft()

# --- Prototype Kelp Strategy (inherits from Prototype MM) ---
class PrototypeKelpStrategy(PrototypeMarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> Optional[int]:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}

        buy_levels = sorted(buy_orders.items(), reverse=True) if buy_orders else []
        sell_levels = sorted(sell_orders.items()) if sell_orders else []

        popular_buy_price = max(buy_levels, key=lambda tup: tup[1])[0] if buy_levels else 0
        popular_sell_price = min(sell_levels, key=lambda tup: tup[1])[0] if sell_levels else 0

        final_value = None
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            final_value = (popular_buy_price + popular_sell_price) / 2
        else:
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0 and best_ask > best_bid:
                final_value = (best_bid + best_ask) / 2
            elif best_bid > 0: final_value = best_bid
            elif best_ask > 0: final_value = best_ask

        return round(final_value) if final_value is not None else None

# --- Squid Ink RSI Strategy --- 
class SquidInkRSIStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.rsi_params = PARAMS[Product.SQUID_INK]
        self.price_history: Deque[float] = deque(maxlen=self.rsi_params["rsi_period"] + 1)
        self.rsi_value: Optional[float] = None

    def _calculate_rsi(self) -> Optional[float]:
        period = self.rsi_params["rsi_period"]
        if len(self.price_history) <= period:
            return None

        prices = np.array(list(self.price_history))
        deltas = np.diff(prices)

        if len(deltas) < period:
             return None

        relevant_deltas = deltas[-period:]

        gains = relevant_deltas[relevant_deltas > 0].sum()
        losses = -relevant_deltas[relevant_deltas < 0].sum()

        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths: return
        order_depth = state.order_depths[self.symbol]
        buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
        sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}

        best_bid = max(buy_orders.keys()) if buy_orders else None
        best_ask = min(sell_orders.keys()) if sell_orders else None

        current_price = None
        if best_bid is not None and best_ask is not None: current_price = (best_bid + best_ask) / 2.0
        elif best_bid is not None: current_price = float(best_bid)
        elif best_ask is not None: current_price = float(best_ask)

        if current_price is None:
             logger.print(f"Cannot determine price for {self.symbol}, skipping RSI step.")
             return

        self.price_history.append(current_price)
        self.rsi_value = self._calculate_rsi()

        if self.rsi_value is not None:
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

    def save(self) -> dict:
        return {"price_history": list(self.price_history), "rsi_value": self.rsi_value}

    def load(self, data: dict) -> None:
        if data and "price_history" in data and isinstance(data["price_history"], list):
             loaded_history = data["price_history"]
             maxlen = self.rsi_params["rsi_period"] + 1
             start_index = max(0, len(loaded_history) - maxlen)
             self.price_history.clear()
             self.price_history.extend(loaded_history[start_index:])
        self.rsi_value = data.get("rsi_value", None) if isinstance(data, dict) else None


# --- Trader Class --- 
class Trader:
    def __init__(self):
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50
        }
        self.strategies = {
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(
                Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]
            ),
            Product.KELP: PrototypeKelpStrategy(
                Product.KELP, self.position_limits[Product.KELP]
            ),
            Product.SQUID_INK: SquidInkRSIStrategy(
                 Product.SQUID_INK, self.position_limits[Product.SQUID_INK]
            )
        }
        logger.print("Trader Initialized (Fallback Strategy - Corrected)")
        logger.print(f"Using PARAMS: {json.dumps(PARAMS)}")

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        logger.print(f"--- Fallback Strategy (Corrected) | Timestamp: {state.timestamp} ---")
        result: Dict[Symbol, List[Order]] = {}
        trader_data_for_next_round = {}
        conversions = 0

        try:
            loaded_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(loaded_data, dict):
                loaded_data = {}
        except Exception as e:
            logger.print(f"Error decoding traderData: {e}. Starting fresh.")
            logger.print(f"Raw traderData: {state.traderData[:200]}...")
            loaded_data = {}

        for product, strategy in self.strategies.items():
            product_state = loaded_data.get(product, {})
            if isinstance(product_state, dict):
                 strategy.load(product_state)
            else:
                 logger.print(f"Warning: Invalid state data for {product}, loading default.")
                 strategy.load({})

            try:
                if product in state.order_depths:
                    strategy.run(state)
                    if strategy.orders:
                        result[product] = strategy.orders
                else:
                     logger.print(f"  Skipping {product}, no order depth data in current state.")
            except Exception as e:
                logger.print(f"*** ERROR running strategy for {product} at timestamp {state.timestamp}: {e} ***")
                import traceback
                logger.print(traceback.format_exc())

            trader_data_for_next_round[product] = strategy.save()

        try:
            traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"), default=str)
        except Exception as e:
             logger.print(f"Error encoding traderData: {e}. Sending empty data.")
             traderData_encoded = "{}"

        logger.flush(state, result, conversions, traderData_encoded)

        return result, conversions, traderData_encoded
