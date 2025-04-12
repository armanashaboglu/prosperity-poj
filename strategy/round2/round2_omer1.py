import json
import numpy as np
from typing import Any, Dict, List, Deque, Optional
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

# --- Strategy Parameters (Optimized for B1 only) ---
PARAMS = {
    "SQUID_INK": { 
        "rsi_window": 106,
        "rsi_overbought": 52,
        "rsi_oversold": 41,
    },
    "RAINFOREST_RESIN": { 
        "fair_value": 10000,
    },
    Product.PICNIC_BASKET1: {
        "default_spread_mean": 48.7,
        "spread_std_window": 83,       
        "zscore_threshold": 24,       
        "target_position": 60,
        "close_threshold": 0.35
    },
    Product.PICNIC_BASKET2: { # arman bunda spread reversion trade ediyoruz ama ADF testine gore spread non-stationary, notebooks&analysis\round2_Basket_Spread.ipynb
        "default_spread_mean": 30.2,    
        "spread_std_window": 159,       
        "zscore_threshold": 5,       
        "target_position": 97,
        "close_threshold": 0.6
    }
}

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
        # logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}") # Keep logs minimal

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        # logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}") # Keep logs minimal

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
        # logger.print(f"{self.symbol} BUY {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        # logger.print(f"{self.symbol} SELL {quantity}x{price}")

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

        # Phase 1: Take orders <= true value
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

        # Phase 2: Make markets based on remaining book
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

# --- Squid Ink RSI Strategy ---
class SquidInkRsiStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {})
        if not self.params: self.params = {"rsi_window": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0}

        self.window = self.params.get("rsi_window", 14)
        if self.window < 2: self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)

        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        self.mid_price_history.append(current_mid_price)
        if len(self.mid_price_history) < self.window + 1: return None

        prices = list(self.mid_price_history)
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))] 
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]

        if not self.rsi_initialized:
            self.avg_gain = sum(gains) / self.window
            self.avg_loss = sum(losses) / self.window
            self.rsi_initialized = True
        else:
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss == 0: return 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth: return

        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid_price is None or best_ask_price is None: return
        current_mid_price = (best_bid_price + best_ask_price) / 2.0

        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None: return

        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position

        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            size_to_sell = to_sell_capacity
            aggressive_sell_price = best_bid_price - 1
            self._place_sell_order(aggressive_sell_price, size_to_sell)
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            size_to_buy = to_buy_capacity
            aggressive_buy_price = best_ask_price + 1
            self._place_buy_order(aggressive_buy_price, size_to_buy)

    def save(self) -> dict:
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        }

    def load(self, data: dict) -> None:
        loaded_history = data.get("mid_price_history", [])
        if isinstance(loaded_history, list):
             self.mid_price_history = deque(loaded_history, maxlen=self.window + 1)
        else:
             self.mid_price_history = deque(maxlen=self.window + 1)

        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)

        # valcheck
        if self.avg_gain is not None: 
            try: self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError): self.avg_gain = None
        if self.avg_loss is not None: 
            try: self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError): self.avg_loss = None
        if not isinstance(self.rsi_initialized, bool): self.rsi_initialized = False
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            self.rsi_initialized = False; self.avg_gain = None; self.avg_loss = None

# --- Z-Score Spread Trading Strategy ---
class ZScoreSpreadStrategy(Strategy):
    def get_swmid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders: return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        if best_bid_vol + best_ask_vol == 0: return (best_bid + best_ask) / 2
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_order_depth = OrderDepth()
        components = BASKET_COMPONENTS.get(self.symbol)
        if not components: return synthetic_order_depth

        component_best_bids = {}
        component_best_asks = {}
        component_bid_volumes = {}
        component_ask_volumes = {}
        possible = True

        for product, quantity_per_basket in components.items():
            if quantity_per_basket == 0: continue
            comp_depth = order_depths.get(product)
            if not comp_depth: possible = False; break
            if comp_depth.buy_orders: 
                best_bid = max(comp_depth.buy_orders.keys()); component_best_bids[product] = best_bid; component_bid_volumes[product] = comp_depth.buy_orders[best_bid]
            else: possible = False; break
            if comp_depth.sell_orders: 
                best_ask = min(comp_depth.sell_orders.keys()); component_best_asks[product] = best_ask; component_ask_volumes[product] = abs(comp_depth.sell_orders[best_ask])
            else: possible = False; break

        if not possible: return synthetic_order_depth

        # Implied bid
        implied_bid_price = 0; implied_bid_volume = float('inf')
        for product, quantity_per_basket in components.items():
            if quantity_per_basket == 0: continue
            implied_bid_price += component_best_bids[product] * quantity_per_basket
            max_baskets_for_comp = component_bid_volumes[product] // quantity_per_basket
            implied_bid_volume = min(implied_bid_volume, max_baskets_for_comp)
        if implied_bid_volume != float('inf') and implied_bid_volume > 0:
            synthetic_order_depth.buy_orders[int(round(implied_bid_price))] = int(implied_bid_volume)

        # Implied ask
        implied_ask_price = 0; implied_ask_volume = float('inf')
        for product, quantity_per_basket in components.items():
            if quantity_per_basket == 0: continue
            implied_ask_price += component_best_asks[product] * quantity_per_basket
            max_baskets_for_comp = component_ask_volumes[product] // quantity_per_basket
            implied_ask_volume = min(implied_ask_volume, max_baskets_for_comp)
        if implied_ask_volume != float('inf') and implied_ask_volume > 0:
            synthetic_order_depth.sell_orders[int(round(implied_ask_price))] = -int(implied_ask_volume)

        return synthetic_order_depth

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = defaultdict(list)
        components = BASKET_COMPONENTS.get(self.symbol)
        if not components: return dict(component_orders)

        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        synth_best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        synth_best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float('inf')

        for order in synthetic_orders:
            if order.symbol != "SYNTHETIC": continue
            price = order.price; quantity = order.quantity
            component_prices = {}; possible = True

            # Determine component prices based on synthetic order direction & validity
            if quantity > 0 and price >= synth_best_ask: # Buy synthetic -> use component ASKS
                for product in components.keys():
                    comp_depth = order_depths.get(product)
                    if not comp_depth or not comp_depth.sell_orders: possible = False; break
                    component_prices[product] = min(comp_depth.sell_orders.keys())
            elif quantity < 0 and price <= synth_best_bid: # Sell synthetic -> use component BIDS
                for product in components.keys():
                    comp_depth = order_depths.get(product)
                    if not comp_depth or not comp_depth.buy_orders: possible = False; break
                    component_prices[product] = max(comp_depth.buy_orders.keys())
            else: continue # Skip invalid or zero-qty orders

            if not possible: continue

            # Create component orders
            for product, quantity_per_basket in components.items():
                 if quantity_per_basket == 0: continue
                 component_price = component_prices.get(product)
                 if component_price is None: continue
                 component_quantity = int(round(quantity * quantity_per_basket))
                 if component_quantity != 0:
                    component_order = Order(product, component_price, component_quantity)
                    component_orders[product].append(component_order)
        return dict(component_orders)

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]) -> Optional[Dict[str, List[Order]]]:
        if target_position == basket_position: return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths.get(self.symbol)
        if not basket_order_depth: return None

        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if not synthetic_order_depth.buy_orders and not synthetic_order_depth.sell_orders: return None

        aggregate_orders = defaultdict(list)
        final_execute_volume = 0

        if target_position > basket_position: # Need to BUY basket, SELL synthetic
            if not basket_order_depth.sell_orders or not synthetic_order_depth.buy_orders: return None
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders.get(basket_ask_price, 0))
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders.get(synthetic_bid_price, 0))
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            final_execute_volume = min(orderbook_volume, target_quantity)

            if final_execute_volume > 0:
                aggregate_orders[self.symbol].append(Order(self.symbol, basket_ask_price, final_execute_volume))
                synthetic_orders = [Order("SYNTHETIC", synthetic_bid_price, -final_execute_volume)]
                component_orders_to_add = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
                for product, orders_list in component_orders_to_add.items(): aggregate_orders[product].extend(orders_list)
            else: return None

        elif target_position < basket_position: # Need to SELL basket, BUY synthetic
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders: return None
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders.get(basket_bid_price, 0))
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders.get(synthetic_ask_price, 0))
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            final_execute_volume = min(orderbook_volume, target_quantity)

            if final_execute_volume > 0:
                aggregate_orders[self.symbol].append(Order(self.symbol, basket_bid_price, -final_execute_volume))
                synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, final_execute_volume)]
                component_orders_to_add = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
                for product, orders_list in component_orders_to_add.items(): aggregate_orders[product].extend(orders_list)
            else: return None

        return dict(aggregate_orders) if final_execute_volume > 0 else None

    def __init__(self, symbol: str, position_limit: int, position_limits: Dict[str, int]) -> None:
        super().__init__(symbol, position_limit)
        self.pos_limits = position_limits
        self.params = PARAMS.get(self.symbol, {})
        if not self.params: self.params = {"default_spread_mean": 0, "spread_std_window": 50, "zscore_threshold": 1.5, "target_position": 10}

        self.components = BASKET_COMPONENTS.get(self.symbol)
        if not self.components: self.components = {}

        self.default_spread_mean = self.params.get("default_spread_mean", 0)
        self.spread_std_window = self.params.get("spread_std_window", 50)
        if self.spread_std_window < 2: self.spread_std_window = 2
        self.zscore_threshold = self.params.get("zscore_threshold", 1.5)
        self.target_position = self.params.get("target_position", 10)

        self.spread_history: Deque[float] = deque(maxlen=self.spread_std_window)

    def _get_synthetic_mid_price(self, state: TradingState) -> Optional[float]:
        synthetic_price = 0.0
        if not self.components: return None
        for component, quantity in self.components.items():
            component_mid_price = self._get_mid_price(component, state)
            if component_mid_price is None: return None
            synthetic_price += quantity * component_mid_price
        return synthetic_price

    def _get_limit_synthetic_volume(self, state: TradingState, direction: int) -> int:
        max_synthetic_volume = float('inf')
        if not self.components: return 0

        for component, quantity_per_basket in self.components.items():
            comp_order_depth = state.order_depths.get(component)
            if not comp_order_depth: return 0

            if direction > 0: # Buy synthetic -> need component BIDS
                 if not comp_order_depth.buy_orders: return 0
                 best_comp_bid_price = max(comp_order_depth.buy_orders.keys())
                 comp_volume_at_best = comp_order_depth.buy_orders[best_comp_bid_price]
                 tradable_comp_baskets = comp_volume_at_best // quantity_per_basket
                 max_synthetic_volume = min(max_synthetic_volume, tradable_comp_baskets)
            else: # Sell synthetic -> need component ASKS
                 if not comp_order_depth.sell_orders: return 0
                 best_comp_ask_price = min(comp_order_depth.sell_orders.keys())
                 comp_volume_at_best = abs(comp_order_depth.sell_orders[best_comp_ask_price])
                 tradable_comp_baskets = comp_volume_at_best // quantity_per_basket
                 max_synthetic_volume = min(max_synthetic_volume, tradable_comp_baskets)

        return int(max_synthetic_volume) if max_synthetic_volume != float('inf') else 0

    def _execute_spread_trade(self, state: TradingState, desired_position: int):
        current_position = state.position.get(self.symbol, 0)
        qty_to_trade = desired_position - current_position
        if qty_to_trade == 0: return

        trade_direction_basket = 1 if qty_to_trade > 0 else -1
        trade_qty_basket = abs(qty_to_trade)
        order_depth_basket = state.order_depths.get(self.symbol)
        if not order_depth_basket: return

        # Determine aggressive price
        if trade_direction_basket > 0: basket_price = min(order_depth_basket.sell_orders.keys()) if order_depth_basket.sell_orders else None
        else: basket_price = max(order_depth_basket.buy_orders.keys()) if order_depth_basket.buy_orders else None
        if basket_price is None: return

        # Calculate volume limits
        # 1. Basket book liquidity
        if trade_direction_basket > 0: basket_book_volume = abs(order_depth_basket.sell_orders.get(basket_price, 0))
        else: basket_book_volume = abs(order_depth_basket.buy_orders.get(basket_price, 0))
        # 2. Synthetic liquidity
        synthetic_volume_limit = self._get_limit_synthetic_volume(state, -trade_direction_basket)
        # 3. Basket position limit
        basket_pos_limit = self.pos_limits.get(self.symbol, 0)
        if trade_direction_basket > 0: limit_from_basket_pos = basket_pos_limit - current_position
        else: limit_from_basket_pos = basket_pos_limit + current_position
        # 4. Component position limits
        limit_from_comp_pos = float('inf')
        for component, comp_qty_per_basket in self.components.items():
            component_pos = state.position.get(component, 0)
            component_limit = self.pos_limits.get(component, 0)
            component_direction = -trade_direction_basket
            if component_direction > 0: comp_capacity = component_limit - component_pos
            else: comp_capacity = component_limit + component_pos
            if comp_qty_per_basket > 0: 
                 max_baskets_for_comp = comp_capacity // comp_qty_per_basket
                 limit_from_comp_pos = min(limit_from_comp_pos, max_baskets_for_comp)
            elif comp_qty_per_basket < 0: logger.print(f"WARNING: Negative quantity per basket ({comp_qty_per_basket}) for {component} not fully handled in limit check.")

        # 5. Final volume
        execute_volume = min(trade_qty_basket, basket_book_volume, synthetic_volume_limit, limit_from_basket_pos, limit_from_comp_pos)

        if execute_volume <= 0: return

        # Place orders
        self._place_order(basket_price, execute_volume * trade_direction_basket)
        for component, comp_qty_per_basket in self.components.items():
             order_depth_comp = state.order_depths.get(component)
             if not order_depth_comp: continue
             component_trade_qty = execute_volume * comp_qty_per_basket
             component_direction = -trade_direction_basket
             if component_direction > 0: comp_price = min(order_depth_comp.sell_orders.keys()) if order_depth_comp.sell_orders else None
             else: comp_price = max(order_depth_comp.buy_orders.keys()) if order_depth_comp.buy_orders else None
             if comp_price is None: continue
             final_component_qty = int(round(component_trade_qty * component_direction))
             if final_component_qty != 0: self._place_order_component(component, comp_price, final_component_qty)

    # Helper for component orders
    def _place_order_component(self, symbol: Symbol, price: float, quantity: float) -> None:
         if quantity == 0: return
         price = int(round(price))
         quantity = int(round(quantity))
         if quantity == 0: return
         self.orders.append(Order(symbol, price, quantity))

    # Helper for basket orders
    def _place_order(self, price: float, quantity: float) -> None:
         if quantity == 0: return
         price = int(round(price))
         quantity = int(round(quantity))
         if quantity == 0: return
         self.orders.append(Order(self.symbol, price, quantity))

    def act(self, state: TradingState) -> None:
        self.orders = []
        order_depths = state.order_depths
        basket_position = state.position.get(self.symbol, 0)

        required_products = list(BASKET_COMPONENTS.get(self.symbol, {}).keys()) + [self.symbol]
        if any(prod not in order_depths for prod in required_products): return

        basket_order_depth = order_depths[self.symbol]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)

        if basket_swmid is None or synthetic_swmid is None: return

        spread = basket_swmid - synthetic_swmid
        self.spread_history.append(spread)
        if len(self.spread_history) < self.spread_std_window: return

        current_spread_history = list(self.spread_history)
        spread_std = np.std(current_spread_history)
        if spread_std < 1e-6: return

        z_score = (spread - self.default_spread_mean) / spread_std
        # logger.print(f" {self.symbol} (Act): Spread={spread:.2f}, FixedMean={self.default_spread_mean}, RollStd={spread_std:.2f}, Z={z_score:.2f}")

        target_pos = basket_position
        orders_to_place = None

        if z_score >= self.zscore_threshold: target_pos = -self.target_position # High Z -> Short
        elif z_score <= -self.zscore_threshold: target_pos = self.target_position # Low Z -> Long
        else:
            # Check if we should close the position based on Z-score normalizing
            if self.symbol == Product.PICNIC_BASKET1:
                close_threshold = self.params.get("close_threshold", 0.1) # Get from params, default 0.1
            elif self.symbol == Product.PICNIC_BASKET2:
                close_threshold = self.params.get("close_threshold", 0.0) # Get from params, default 0.0
            else:
                close_threshold = 0.0 # Default for any other symbol if strategy applied elsewhere

            if (basket_position > 0 and z_score <= close_threshold) or \
               (basket_position < 0 and z_score >= -close_threshold):
                desired_position = 0
                logger.print(f"{self.symbol}: Z-score ({z_score:.2f}) crossed threshold ({close_threshold:.2f}). Closing position.")

        if basket_position != target_pos:
            orders_to_place = self.execute_spread_orders(target_pos, basket_position, order_depths)

        if orders_to_place:
            self.orders.extend([order for order_list in orders_to_place.values() for order in order_list])

    def save(self) -> dict:
        return {"spread_history": list(self.spread_history)}

    def load(self, data: dict) -> None:
        if data and "spread_history" in data and isinstance(data["spread_history"], list):
            loaded_history = data["spread_history"]
            start_index = max(0, len(loaded_history) - self.spread_history.maxlen)
            self.spread_history.clear()
            self.spread_history.extend(loaded_history[start_index:])

# --- Trader Class ---
class Trader:
    def __init__(self):
        self.position_limits = {
            Product.KELP: 50, Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50,
            Product.CROISSANTS: 250, Product.JAMS: 350, Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60, Product.PICNIC_BASKET2: 100,
        }
        # Redundant component limits dict removed

        self.strategies = {
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]),
            Product.KELP: PrototypeKelpStrategy(Product.KELP, self.position_limits[Product.KELP]),
            Product.SQUID_INK: SquidInkRsiStrategy(Product.SQUID_INK, self.position_limits[Product.SQUID_INK]),
            Product.PICNIC_BASKET1: ZScoreSpreadStrategy(Product.PICNIC_BASKET1, self.position_limits[Product.PICNIC_BASKET1], self.position_limits),
            Product.PICNIC_BASKET2: ZScoreSpreadStrategy(Product.PICNIC_BASKET2, self.position_limits[Product.PICNIC_BASKET2], self.position_limits), # Still commented out
        }
        # logger.print("Trader Initialized") # Simplified log

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
         aggregated_orders: Dict[Symbol, Dict[int, int]] = {}
         conversions = 0
         trader_data_for_next_round = {}

         try:
             loaded_data = json.loads(state.traderData) if state.traderData else {}
             if not isinstance(loaded_data, dict): loaded_data = {}
         except Exception as e: loaded_data = {}

         # Run strategies and aggregate orders
         for product, strategy in self.strategies.items():
             product_state = loaded_data.get(product, {}) 
             if isinstance(product_state, dict): 
                 try: strategy.load(product_state)
                 except Exception: strategy.load({})
             else: strategy.load({}) 
             
             market_data_available = product in state.order_depths
             if market_data_available:
                 try:
                     strategy.run(state)
                     for order in strategy.orders:
                         symbol = order.symbol; price = order.price; quantity = order.quantity
                         if symbol not in aggregated_orders: aggregated_orders[symbol] = {}
                         aggregated_orders[symbol][price] = aggregated_orders[symbol].get(price, 0) + quantity
                 except Exception as e: logger.print(f"*** ERROR running {product} strategy: {e} ***"); import traceback; logger.print(traceback.format_exc())
             
             try: trader_data_for_next_round[product] = strategy.save()
             except Exception: trader_data_for_next_round[product] = {}

         # Final order creation with position limit checks
         final_result: Dict[Symbol, List[Order]] = {}
         for symbol, price_qty_map in aggregated_orders.items():
             orders_for_symbol = []
             current_position = state.position.get(symbol, 0)
             limit = self.position_limits.get(symbol, 0)
             sorted_prices = sorted(price_qty_map.keys())

             for price in sorted_prices:
                 quantity = price_qty_map[price]
                 if quantity == 0: continue

                 potential_position = current_position + quantity
                 if quantity > 0 and potential_position > limit:
                     quantity = limit - current_position
                 elif quantity < 0 and potential_position < -limit:
                     quantity = -limit - current_position
                 
                 if quantity != 0:
                     orders_for_symbol.append(Order(symbol, price, int(quantity)))
                     current_position += quantity

             if orders_for_symbol: final_result[symbol] = orders_for_symbol

         # Encode data & flush logs
         try: traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"), default=str)
         except Exception: traderData_encoded = "{}"
         logger.flush(state, final_result, conversions, traderData_encoded)
         return final_result, conversions, traderData_encoded 