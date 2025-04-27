
import numpy as np #type: ignore
from typing import Any, Dict, List, Deque, Optional
from abc import abstractmethod
from collections import deque
import math
import copy

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# --- Logger Class (Copied from v3) ---
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


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

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
    "SQUID_INK": {
        
        "rsi_window": 32,           # Lookback period for RSI
        "rsi_overbought": 73,     # RSI level considered overbought
        "rsi_oversold": 38,       # RSI level considered oversold
        
    },
    "RAINFOREST_RESIN": { # Params for V3 Resin Strategy
        "fair_value": 10000,
    },
    # KELP uses the PrototypeKelpStrategy which doesn't read from PARAMS
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
        #if make_bid_price == 9991: make_bid_price = 9997
        #if make_ask_price == 10007: make_ask_price = 10003
        

        # Ensure bid < ask
        #if make_bid_price >= make_ask_price:
        #    make_ask_price = make_bid_price + 1
        #    if make_ask_price == 10001: make_ask_price = 10002
        #    elif make_ask_price == 10003: make_ask_price = 10004

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
        price = int(round(price))
        quantity = int(round(quantity))
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PROTO_BUY {self.symbol} {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        quantity = int(round(quantity))
        if quantity <= 0: return
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

        # Calculate flags based on current window (before potential pop)
        is_full_window = len(self.window) == self.window_size
        stuck_count = sum(self.window)
        soft_liquidate = is_full_window and stuck_count >= self.window_size / 2 and self.window[-1]
        hard_liquidate = is_full_window and stuck_count == self.window_size

        # Enforce window size *after* calculating flags (matches prototype timing)
        if len(self.window) > self.window_size:
            self.window.popleft()

        # Correct the liquidation prices to match prototype exactly
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity) # Hard liq buy at TV
            to_buy -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Hard Liq Buy {quantity}x{true_value:.0f}")
        elif to_buy > 0 and soft_liquidate: # Stuck short, need to buy
            quantity = to_buy // 2
            # Prototype buys at TV - 2 for soft liq when short (maybe typo in proto, but replicating)
            liq_price = true_value - 2
            self.buy(liq_price, quantity)
            to_buy -= quantity
            logger.print(f"{self.symbol} (Proto Logic) Soft Liq Buy {quantity}x{liq_price:.0f}")


        # Phase 3: Make orders
        if to_buy > 0:
            # Pass the original list here
            popular_buy_price = max(buy_orders_list, key=lambda tup: tup[1])[0] if buy_orders_list else (true_value - 2)
            make_price = min(max_buy_price, popular_buy_price + 1)
            self.buy(make_price, to_buy)

        if to_sell > 0:
            # Pass the original list here
            popular_sell_price = min(sell_orders_list, key=lambda tup: tup[1])[0] if sell_orders_list else (true_value + 2)
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

# --- NEW: Squid Ink RSI Strategy ---
class SquidInkRsiStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
             logger.print(f"ERROR: Parameters for {self.symbol} not found in PARAMS. Using defaults.")
             self.params = {"rsi_window": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0}

        # Load RSI parameters
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2: # RSI requires at least 2 periods for changes
            logger.print(f"Warning: RSI window {self.window} too small, setting to 2.")
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)

        # State variables for RSI calculation
        # Need window + 1 prices to calculate window changes
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False

        logger.print(f"Initialized SquidInkRsiStrategy with window={self.window}, OB={self.overbought_threshold}, OS={self.oversold_threshold}")

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        """Calculates RSI using Wilder's smoothing method."""
        self.mid_price_history.append(current_mid_price)

        if len(self.mid_price_history) < self.window + 1:
            # Need enough data points to calculate initial average gain/loss
            return None

        prices = list(self.mid_price_history)
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))] # len(prices) = window + 1, so len(changes) = window

        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]

        if not self.rsi_initialized:
            # First calculation: Use simple average over the window
            self.avg_gain = sum(gains) / self.window
            self.avg_loss = sum(losses) / self.window
            self.rsi_initialized = True
            logger.print(f" {self.symbol} (RSI): Initialized avg_gain={self.avg_gain:.4f}, avg_loss={self.avg_loss:.4f}")
        else:
            # Subsequent calculations: Use Wilder's smoothing
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss == 0:
             # Avoid division by zero; RSI is 100 if avg_loss is 0
             return 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)

        if not order_depth:
            logger.print(f" {self.symbol} (RSI): No order depth data.")
            return

        # --- 1. Calculate Current Mid-Price --- #
        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid_price is None or best_ask_price is None:
            logger.print(f"{self.symbol} (RSI): Missing BBO, cannot calculate mid-price or RSI.")
            return

        current_mid_price = (best_bid_price + best_ask_price) / 2.0

        # --- 2. Calculate RSI --- #
        rsi_value = self._calculate_rsi(current_mid_price)

        if rsi_value is None:
            logger.print(f" {self.symbol} (RSI): Not enough history to calculate RSI.")
            return # Wait for more data
        else:
            logger.print(f" {self.symbol} (RSI): Mid={current_mid_price:.2f}, RSI({self.window})={rsi_value:.2f}")

        # --- 3. Generate Signal & Trade --- #
        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position

        # Signal: Sell when RSI is overbought
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            size_to_sell = to_sell_capacity # Sell max capacity
            aggressive_sell_price = best_bid_price - 1 # Place 1 tick below best bid
            logger.print(f" {self.symbol} -> RSI SELL Signal (RSI: {rsi_value:.2f} > {self.overbought_threshold}) Max Qty: {size_to_sell} @ Aggressive Price {aggressive_sell_price}")
            self._place_sell_order(aggressive_sell_price, size_to_sell)

        # Signal: Buy when RSI is oversold
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            size_to_buy = to_buy_capacity # Buy max capacity
            aggressive_buy_price = best_ask_price + 1 # Place 1 tick above best ask
            logger.print(f" {self.symbol} -> RSI BUY Signal (RSI: {rsi_value:.2f} < {self.oversold_threshold}) Max Qty: {size_to_buy} @ Aggressive Price {aggressive_buy_price}")
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
             self.mid_price_history = deque(loaded_history, maxlen=self.window + 1)
        else:
             self.mid_price_history = deque(maxlen=self.window + 1) # Reset if invalid

        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)

        # Type/validity checking
        if self.avg_gain is not None:
            try: self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError): self.avg_gain = None
        if self.avg_loss is not None:
            try: self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError): self.avg_loss = None
        if not isinstance(self.rsi_initialized, bool):
             self.rsi_initialized = False

        # Reset if essential state is inconsistent
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            logger.print(f"Warning: RSI avg gain/loss invalid after loading for {self.symbol}. Resetting RSI calc state.")
            self.rsi_initialized = False
            self.avg_gain = None
            self.avg_loss = None
            # History might be okay, don't clear it unless needed


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
            # --- Use the new RSI strategy ---
            Product.SQUID_INK: SquidInkRsiStrategy( # Use the new class
                 Product.SQUID_INK,
                 self.position_limits[Product.SQUID_INK]
            )
        }
        logger.print("Trader Initialized (Fallback Strategy - With RSI Squid)") # Updated log message
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
