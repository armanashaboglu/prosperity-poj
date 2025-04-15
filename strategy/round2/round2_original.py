import json
import math
from typing import Any, Dict, List, Optional, Deque
import copy
from collections import defaultdict, deque
from abc import abstractmethod
from datamodel import (
    Listing,
    Observation,
    OrderDepth,
    UserId,
    TradingState,
    Order,
    Symbol,
    Trade,
    ProsperityEncoder
)

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        base_json = self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ])
        base_len = len(base_json)
        max_item_length = (self.max_log_length - base_len)//3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length)
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations)
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        arr = []
        for lst in listings.values():
            arr.append([lst.symbol, lst.product, lst.denomination])
        return arr

    def compress_order_depths(self, od_map: Dict[Symbol, OrderDepth]) -> Dict[str, List[Any]]:
        c = {}
        for sym, od in od_map.items():
            c[sym] = [od.buy_orders, od.sell_orders]
        return c

    def compress_trades(self, trades_map: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        arr = []
        for tlist in trades_map.values():
            for t in tlist:
                arr.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return arr

    def compress_observations(self, obs: Observation) -> List[Any]:
        conv_obs = {}
        for product, ob in obs.conversionObservations.items():
            conv_obs[product] = [
                ob.bidPrice, ob.askPrice,
                ob.transportFees, ob.exportTariff,
                ob.importTariff, ob.sugarPrice, ob.sunlightIndex
            ]
        return [obs.plainValueObservations, conv_obs]

    def compress_orders(self, orders_map: Dict[str, List[Order]]) -> List[List[Any]]:
        arr = []
        for sym_list in orders_map.values():
            for o in sym_list:
                arr.append([o.symbol, o.price, o.quantity])
        return arr

    def to_json(self, val: Any) -> str:
        return json.dumps(val, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, s: str, maxlen: int) -> str:
        if len(s)<=maxlen:
            return s
        return s[:maxlen-3]+"..."

logger = Logger()


class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.BASKET1: 60,
    Product.BASKET2: 100,
    Product.KELP: 50,
    Product.RAINFOREST_RESIN: 50,
    Product.SQUID_INK: 50
}

PARAMS = {
    "SQUID_INK": { 
        "rsi_window": 96,  
        "rsi_overbought": 56, 
        "rsi_oversold": 39, 
    },
    "RAINFOREST_RESIN": { 
        "fair_value": 10000,
    }}
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

# <<< ADDED: Squid Ink RSI Strategy >>>
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

class Trader:
    def __init__(self):
        logger.print("Initializing Trader: Running individual strategies + Round 2 Arbitrage Logic")

        self.pos_limits = POSITION_LIMITS

        # --- Arbitrage parameters (kept from original) --- 
        self.diff_threshold_b1 = 200
        self.diff_threshold_b2 = 120
        self.diff_threshold_b1_b2 = 60
        self.max_arb_lot = 5 # Defaulting to 5 as it wasn't a member variable before

        # --- Initialize individual product strategies --- 
        self.strategies: Dict[Symbol, Strategy] = {}
        # Define which strategy class corresponds to which product
        strategy_map = {
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy,
            Product.KELP: PrototypeKelpStrategy,
            Product.SQUID_INK: SquidInkRsiStrategy,
        }

        for product, strategy_class in strategy_map.items():
            limit = self.pos_limits.get(product)
            if limit is not None:
                 try: 
                     self.strategies[product] = strategy_class(product, limit)
                     logger.print(f"  - Initialized strategy for {product}")
                 except Exception as e: 
                     logger.print(f"*** ERROR initializing strategy for {product}: {e} ***")
            else: 
                 logger.print(f"Warning: Position limit not found for {product}. Strategy not initialized.")

        logger.print(f"Trader Initialized with {len(self.strategies)} individual strategies + Basket Arbitrage.")
        logger.print(f"Using PARAMS: {json.dumps(PARAMS)}")


    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        logger.print(f"--- Trader Run | Timestamp: {state.timestamp} ---")

        # Use defaultdict for easier order aggregation
        all_orders: Dict[Symbol, List[Order]] = defaultdict(list)
        conversions = 0
        trader_data_for_next_round = {} # Store state for strategies

        # --- Load state for individual strategies --- 
        try:
            loaded_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(loaded_data, dict):
                logger.print(f"Warning: TraderData was not a dict: {state.traderData[:100]}... Resetting state.")
                loaded_data = {}
        except Exception as e:
            logger.print(f"Error decoding traderData: {e}. Starting fresh state.")
            loaded_data = {}

        # --- Run individual product strategies --- 
        logger.print("Running individual strategies...")
        for product, strategy in self.strategies.items():
            product_key_str = str(product) # Use string representation for JSON keys
            product_state = loaded_data.get(product_key_str, {})
            if not isinstance(product_state, dict): 
                logger.print(f"Warning: Invalid state data for {product}, loading default.")
                product_state = {}
            
            try:
                # Load the saved state into the strategy object
                strategy.load(product_state) 
            except Exception as e:
                logger.print(f"*** ERROR loading state for {product}: {e} ***")
                # Attempt to continue with default state if load fails

            try:
                # Check if the product is active in the current state (has order depth)
                if product in state.order_depths: 
                    # Strategy's run method should execute its logic and return List[Order]
                    # Base Strategy.run() calls strategy.act() and returns self.orders
                    strategy_orders = strategy.run(state) 
                    if strategy_orders: # If the strategy generated orders
                        # Append these orders to our main dictionary
                        all_orders[product].extend(strategy_orders) 
                else:
                     logger.print(f"  Skipping {product} strategy run, no order depth data.")
            except Exception as e:
                logger.print(f"*** ERROR running strategy for {product} at timestamp {state.timestamp}: {e} ***")
                import traceback
                logger.print(traceback.format_exc()) # Log full traceback for debugging

            try:
                # Save the current state of the strategy for the next round
                trader_data_for_next_round[product_key_str] = strategy.save()
            except Exception as e:
                logger.print(f"*** ERROR saving state for {product}: {e} ***")

        logger.print("Individual strategies finished.")

        # --- Run Arbitrage Logic (Basket Trading) --- 
        logger.print("Running Basket Arbitrage logic...")
        try:
            # Calculate mid prices (existing logic from original run method)
            relevant = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.BASKET1, Product.BASKET2]
            best_bid = {}
            best_ask = {}
            mid_price = {}
            for r in relevant:
                od = state.order_depths.get(r)
                if od:
                    best_bid[r] = max(od.buy_orders.keys()) if od.buy_orders else None
                    best_ask[r] = min(od.sell_orders.keys()) if od.sell_orders else None
                    if best_bid[r] is not None and best_ask[r] is not None: mid_price[r] = 0.5*(best_bid[r] + best_ask[r])
                    elif best_bid[r] is not None: mid_price[r] = best_bid[r]
                    elif best_ask[r] is not None: mid_price[r] = best_ask[r]
                    else: mid_price[r] = None
                else:
                    best_bid[r], best_ask[r], mid_price[r] = None, None, None

            c = mid_price.get(Product.CROISSANTS)
            j = mid_price.get(Product.JAMS)
            d = mid_price.get(Product.DJEMBES)
            b1 = mid_price.get(Product.BASKET1)
            b2 = mid_price.get(Product.BASKET2)

            # Arbitrage Logic 1: B1 vs B2 Composition (existing logic)
            # Pass all_orders dict to the helper function so it can append orders
            if (b1 is not None) and (b2 is not None) and (c is not None) and (j is not None) and (d is not None):
                implied_b1_from_b2 = b2 + 2*c + 1*j + 1*d
                diff_comp = b1 - implied_b1_from_b2
                logger.print(f"  B1 vs B2+Comp Arb: B1={b1:.1f}, B2plus={implied_b1_from_b2:.1f}, diff={diff_comp:.1f}, Thr={self.diff_threshold_b1_b2}")
                if diff_comp > self.diff_threshold_b1_b2:
                    self.execute_b1_vs_b2_composition(all_orders, state, -1, +1, +2, +1, +1, self.max_arb_lot)
                elif diff_comp < -self.diff_threshold_b1_b2: # Original logic didn't have offset here
                    self.execute_b1_vs_b2_composition(all_orders, state, +1, -1, -2, -1, -1, self.max_arb_lot)

            # Arbitrage Logic 2: B1 vs Items (existing logic)
            # Pass all_orders dict to the helper function
            if (c is not None) and (j is not None) and (d is not None) and (b1 is not None):
                fair_b1 = 6*c + 3*j + 1*d
                diff1 = b1 - fair_b1
                logger.print(f"  B1 vs Items Arb: B1={b1:.1f}, Fair={fair_b1:.1f}, diff={diff1:.1f}, Thr={self.diff_threshold_b1}")
                if diff1 > self.diff_threshold_b1:
                    self.execute_spread_trade(all_orders, state, Product.BASKET1, -1, {Product.CROISSANTS: +6, Product.JAMS:+3, Product.DJEMBES:+1}, self.max_arb_lot)
                elif diff1 < -self.diff_threshold_b1 : # Keep original offset logic
                    self.execute_spread_trade(all_orders, state, Product.BASKET1, +1, {Product.CROISSANTS: -6, Product.JAMS:-3, Product.DJEMBES:-1}, self.max_arb_lot)

            # Arbitrage Logic 3: B2 vs Items (existing logic)
            # Pass all_orders dict to the helper function
            if (c is not None) and (j is not None) and (b2 is not None):
                fair_b2 = 4*c + 2*j
                diff2 = b2 - fair_b2
                logger.print(f"  B2 vs Items Arb: B2={b2:.1f}, Fair={fair_b2:.1f}, diff={diff2:.1f}, Thr={self.diff_threshold_b2}")
                if diff2 > self.diff_threshold_b2:
                    self.execute_spread_trade(all_orders, state, Product.BASKET2, -1, {Product.CROISSANTS:+4, Product.JAMS:+2}, self.max_arb_lot)
                elif diff2 < -self.diff_threshold_b2 : # Keep original offset logic
                    self.execute_spread_trade(all_orders, state, Product.BASKET2, +1, {Product.CROISSANTS:-4, Product.JAMS:-2}, self.max_arb_lot)
        except Exception as e:
            logger.print(f"*** ERROR during Basket Arbitrage logic: {e} ***")
            import traceback
            logger.print(traceback.format_exc())

        logger.print("Basket Arbitrage logic finished.")

        # --- Finalize and Return --- 
        # Convert defaultdict back to dict for Prosperity system
        final_orders = dict(all_orders) 

        try:
            # Encode the collected strategy states
            trader_data_out = json.dumps(trader_data_for_next_round, separators=(",", ":"), default=str) 
        except Exception as e:
             logger.print(f"Error encoding final traderData: {e}. Sending empty data.")
             trader_data_out = "{}"

        logger.flush(state, final_orders, conversions, trader_data_out)
        # Return the combined orders from all strategies and arbitrage logic
        return final_orders, conversions, trader_data_out


    ############################################
    # synergy function: basket vs items
    ############################################
    def execute_spread_trade(
        self,
        orders_dict: Dict[str, List[Order]],
        state: TradingState,
        basket_symbol: str,
        basket_side: int,
        components_sides: Dict[str,int],
        max_single_lot: int=5
    ):
        # same logic as your old code
        current_pos = state.position.get(basket_symbol, 0)
        limit_basket = self.pos_limits[basket_symbol]

        if basket_side>0:
            capacity = limit_basket - current_pos
        else:
            capacity = limit_basket + current_pos

        if capacity<=0:
            logger.print(f"No capacity synergy on {basket_symbol}, side={basket_side}. skip.")
            return

        trade_qty = min(max_single_lot, capacity)
        if trade_qty<=0:
            return

        # check item capacity
        for comp_sym, comp_side in components_sides.items():
            comp_pos = state.position.get(comp_sym, 0)
            limit_comp = self.pos_limits[comp_sym]
            needed = abs(comp_side)* trade_qty
            if comp_side>0:
                cap_comp = limit_comp - comp_pos
            else:
                cap_comp = limit_comp + comp_pos
            if cap_comp< needed:
                logger.print(f"No item capacity => {comp_sym}, side={comp_side}, skip synergy.")
                return

        od_basket = state.order_depths.get(basket_symbol)
        if not od_basket:
            logger.print(f"No od for {basket_symbol}, skip synergy.")
            return

        # place basket order
        if basket_side>0:
            # buy => best ask
            if od_basket.sell_orders:
                best_ask = min(od_basket.sell_orders.keys())
                px = best_ask+1
                orders_dict[basket_symbol].append(Order(basket_symbol, px, trade_qty))
                logger.print(f" => BUY {basket_symbol} {trade_qty}x{px}")
        else:
            # sell => best bid
            if od_basket.buy_orders:
                best_bid = max(od_basket.buy_orders.keys())
                px = best_bid-1
                orders_dict[basket_symbol].append(Order(basket_symbol, px, -trade_qty))
                logger.print(f" => SELL {basket_symbol} {trade_qty}x{px}")

        # place items in opposite direction
        for comp_sym, comp_side in components_sides.items():
            full_qty = comp_side* trade_qty
            if full_qty==0:
                continue
            od_comp = state.order_depths.get(comp_sym)
            if not od_comp:
                logger.print(f"No od for {comp_sym}, skip synergy side.")
                continue
            if full_qty>0:
                # buy
                if od_comp.sell_orders:
                    best_ask = min(od_comp.sell_orders.keys())
                    px = best_ask+1
                    orders_dict[comp_sym].append(Order(comp_sym, px, full_qty))
                    logger.print(f" => BUY {comp_sym} {full_qty}x{px}")
            else:
                # sell
                if od_comp.buy_orders:
                    best_bid = max(od_comp.buy_orders.keys())
                    px= best_bid-1
                    orders_dict[comp_sym].append(Order(comp_sym, px, full_qty))
                    logger.print(f" => SELL {comp_sym} {abs(full_qty)}x{px}")

    ############################################
    # synergy function: basket1 vs basket2 + items
    ############################################
    def execute_b1_vs_b2_composition(
        self,
        orders_dict: Dict[str,List[Order]],
        state: TradingState,
        side_b1: int,
        side_b2: int,
        side_c: int,
        side_j: int,
        side_d: int,
        max_single_lot: int=5
    ):
        """
        The difference: B1 minus B2 => 2C + 1J + 1D
        We'll do 1-lot synergy if capacity. If side_b1= -1 => short B1, side_b2=+1 => buy B2, side_c=+2 => buy Croissants, etc.
        """
        # capacity checks
        def check_capacity(prod, side_val):
            ppos = state.position.get(prod, 0)
            lim = self.pos_limits[prod]
            if side_val>0:
                return lim - ppos
            else:
                return lim + ppos

        cap_b1 = check_capacity(Product.BASKET1, side_b1)
        cap_b2 = check_capacity(Product.BASKET2, side_b2)
        cap_c = check_capacity(Product.CROISSANTS, side_c)
        cap_j = check_capacity(Product.JAMS, side_j)
        cap_d = check_capacity(Product.DJEMBES, side_d)

        # each synergy-lot requires 1 B1, 1 B2, 2 or 1 items, etc.
        # figure out how many synergy-lots we can do
        def lot_capacity(side_val, cap_val):
            if side_val==0:
                return float('inf')
            need = abs(side_val)
            return cap_val//need

        possible_b1 = lot_capacity(side_b1, cap_b1)
        possible_b2 = lot_capacity(side_b2, cap_b2)
        possible_c  = lot_capacity(side_c,  cap_c)
        possible_j  = lot_capacity(side_j,  cap_j)
        possible_d  = lot_capacity(side_d,  cap_d)

        synergy_possible = min(possible_b1, possible_b2, possible_c, possible_j, possible_d, max_single_lot)
        if synergy_possible<=0:
            logger.print("No capacity for B1 vs B2 composition synergy, skip.")
            return

        # do 1-lot synergy
        lot = 1

        # place B1
        od_b1 = state.order_depths.get(Product.BASKET1)
        if not od_b1:
            logger.print("No od for B1, skip synergy.")
            return
        if side_b1>0:
            # buy B1 => best ask
            if od_b1.sell_orders:
                best_ask = min(od_b1.sell_orders.keys())
                px = best_ask+1
                orders_dict[Product.BASKET1].append(Order(Product.BASKET1, px, lot))
                logger.print(f" => BUY B1 {lot}x{px}")
        else:
            # sell B1 => best bid
            if od_b1.buy_orders:
                best_bid = max(od_b1.buy_orders.keys())
                px= best_bid-1
                orders_dict[Product.BASKET1].append(Order(Product.BASKET1, px, -lot))
                logger.print(f" => SELL B1 {lot}x{px}")

        # place B2
        od_b2 = state.order_depths.get(Product.BASKET2)
        if not od_b2:
            logger.print("No od for B2, skip synergy.")
            return
        if side_b2>0:
            # buy B2
            if od_b2.sell_orders:
                best_ask = min(od_b2.sell_orders.keys())
                px= best_ask+1
                orders_dict[Product.BASKET2].append(Order(Product.BASKET2, px, lot))
                logger.print(f" => BUY B2 {lot}x{px}")
        else:
            # sell B2
            if od_b2.buy_orders:
                best_bid = max(od_b2.buy_orders.keys())
                px= best_bid-1
                orders_dict[Product.BASKET2].append(Order(Product.BASKET2, px, -lot))
                logger.print(f" => SELL B2 {lot}x{px}")

        # place items
        def place_item(prod, side_val, count):
            if side_val==0:
                return
            odx = state.order_depths.get(prod)
            if not odx:
                logger.print(f"No od for {prod}, skip synergy item.")
                return
            needed = side_val* count
            if needed>0:
                # buy
                if odx.sell_orders:
                    best_ask = min(odx.sell_orders.keys())
                    px = best_ask+1
                    orders_dict[prod].append(Order(prod, px, needed))
                    logger.print(f" => BUY {prod} {needed}x{px}")
            else:
                # sell
                if odx.buy_orders:
                    best_bid = max(odx.buy_orders.keys())
                    px= best_bid-1
                    orders_dict[prod].append(Order(prod, px, needed))
                    logger.print(f" => SELL {prod} {abs(needed)}x{px}")

        place_item(Product.CROISSANTS, side_c, lot)
        place_item(Product.JAMS, side_j, lot)
        place_item(Product.DJEMBES, side_d, lot)