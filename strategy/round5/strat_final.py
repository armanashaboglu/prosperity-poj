import json
import numpy as np # type: ignore
from typing import Any, Dict, List, Deque, Optional, Tuple
from abc import abstractmethod
from collections import deque, defaultdict
import math
import copy
from enum import Enum

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder


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

# --- Strategy Parameters (Optimized for B1 only) ---
PARAMS = {
    "SQUID_INK": {
        "rsi_window": 85,
        "rsi_overbought": 56,
        "rsi_oversold": 44,
        "price_offset": 1
    },
    "VOLCANIC_ROCK": {
        "rsi_window": 85,
        "rsi_overbought": 52,
        "rsi_oversold": 42,
        "price_offset": 0
    },

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


class RsiStrategy(Strategy):
    """A generic RSI strategy applicable to different products."""
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {}) # Load params specific to self.symbol
        if not self.params:
            logger.print(f"ERROR: Parameters for RSI strategy on {self.symbol} not found in PARAMS. Using defaults.")
            self.params = {"rsi_window": 85, "rsi_overbought": 52.0, "rsi_oversold": 42.0, "price_offset": 0} # Fallback defaults

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
             
             return

        # Calculate RSI
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            
            return
        

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
                
                self._place_sell_order(aggressive_sell_price, size_to_sell)

        # Signal: Buy when RSI is oversold
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            if best_ask_price is not None: # Need an ask to hit
                size_to_buy = to_buy_capacity # Buy max capacity
                # Apply price offset for buying (positive direction)
                aggressive_buy_price = best_ask_price + self.price_offset
                
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
        super().load(data)
        self.ema = data.get("ema", None)

# --- Croissant Strategy --- (NEW)
class CroissantCaesarOliviaStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        # Use a simple state: 0 (initial/undetermined), 1 (target LONG), -1 (target SHORT)
        self.target_state = 0
        logger.print(f"Initialized CroissantCaesarOliviaStrategy for {symbol}")

    def act(self, state: TradingState) -> None:
        # 1. Check market trades to potentially update the target state
        croissant_trades = state.market_trades.get(self.symbol, [])
        signal_found = False
        for trade in croissant_trades:
            caesar = "Caesar"
            olivia = "Olivia"

            if trade.buyer == caesar and trade.seller == olivia:
                if self.target_state != -1:
                    logger.print(f"CROISSANT State Update: Caesar bought from Olivia -> TARGET SHORT")
                    self.target_state = -1
                signal_found = True
                break # Prioritize the first signal found in this batch?

            elif trade.buyer == olivia and trade.seller == caesar:
                if self.target_state != 1:
                    logger.print(f"CROISSANT State Update: Olivia bought from Caesar -> TARGET LONG")
                    self.target_state = 1
                signal_found = True
                break # Prioritize the first signal found in this batch?

        # 2. Execute orders based on the current target state
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol)

        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders):
            logger.print(f"CROISSANT: No order depth or orders available. Cannot trade.")
            return

        if self.target_state == -1:
            # Target: Full Short (-limit)
            needed_to_sell = self.position_limit + position # Qty to sell to reach -limit
            if needed_to_sell > 0:
                sell_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                if sell_price is not None:
                    logger.print(f"CROISSANT: TARGET SHORT, current pos {position}, selling {needed_to_sell} towards {-self.position_limit}")
                    # Limit order size to available bid volume if needed, or just place full order
                    # For simplicity, place full needed order for now
                    self._place_sell_order(sell_price, needed_to_sell)
                else:
                     logger.print(f"CROISSANT: TARGET SHORT, but no buy orders to hit.")
            # else: logger.print(f"CROISSANT: TARGET SHORT, already at or below limit {position}")

        elif self.target_state == 1:
            # Target: Full Long (+limit)
            needed_to_buy = self.position_limit - position # Qty to buy to reach +limit
            if needed_to_buy > 0:
                buy_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                if buy_price is not None:
                    logger.print(f"CROISSANT: TARGET LONG, current pos {position}, buying {needed_to_buy} towards {self.position_limit}")
                    # Limit order size to available ask volume if needed, or just place full order
                    self._place_buy_order(buy_price, needed_to_buy)
                else:
                    logger.print(f"CROISSANT: TARGET LONG, but no sell orders to hit.")
            # else: logger.print(f"CROISSANT: TARGET LONG, already at or above limit {position}")

        # else: target_state is 0 (initial), do nothing until a signal is seen

    def save(self) -> dict:
        # Save the target state (-1, 0, or 1)
        return {"target_state": self.target_state}

    def load(self, data: dict) -> None:
        # Load the target state, default to 0 if not found or invalid
        loaded_state = data.get("target_state")
        if loaded_state in [-1, 0, 1]:
            self.target_state = loaded_state
        else:
            self.target_state = 0 # Default to initial state
        logger.print(f"Loaded state for {self.symbol}: TargetState={self.target_state}")


# --- Trader Class --- (MODIFIED)
class Trader:
    def __init__(self):
        # --- Position Limits --- (UPDATED)
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
        }

        # --- Strategy Definitions --- (UPDATED)
        self.strategies = {
            # Existing Strategies
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]),
            Product.KELP: PrototypeKelpStrategy(Product.KELP, self.position_limits[Product.KELP]),
            Product.SQUID_INK: RsiStrategy(Product.SQUID_INK, self.position_limits[Product.SQUID_INK]),
            Product.VOLCANIC_ROCK: RsiStrategy(Product.VOLCANIC_ROCK, self.position_limits[Product.VOLCANIC_ROCK]),

            # New Strategies based on Gameplan
            Product.CROISSANTS: CroissantCaesarOliviaStrategy(Product.CROISSANTS, self.position_limits[Product.CROISSANTS]),
            #Product.JAMS: PicnicBasketStrategy(Product.JAMS, self.position_limits[Product.JAMS]),
            #Product.DJEMBES: PicnicBasketStrategy(Product.DJEMBES, self.position_limits[Product.DJEMBES]),
            #Product.PICNIC_BASKET1: PicnicBasketStrategy(Product.PICNIC_BASKET1, self.position_limits[Product.PICNIC_BASKET1]),
            #Product.PICNIC_BASKET2: PicnicBasketStrategy(Product.PICNIC_BASKET2, self.position_limits[Product.PICNIC_BASKET2]),
        }

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
         # --- Existing Run Logic ---
         all_orders: List[Order] = []
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
             strategy_state = loaded_data.get(str(strategy_key), {})
             if isinstance(strategy_state, dict):
                 try: strategy.load(strategy_state)
                 except Exception as e: logger.print(f"Error loading state for {strategy_key}: {e}")
             else: strategy.load({}) 

             # Check if market data is available for the strategy
             market_data_available = True
             required_products = []
             # Simplified check now that VolatilitySmile is removed
             if str(strategy_key) in self.position_limits: 
                 required_products = [str(strategy_key)]
             # Add checks for other multi-product strategies if needed
             else:
                  logger.print(f"Warning: Unsure how to check market data availability for strategy key: {strategy_key}")

             # Ensure all required products for the *current strategy* have order depths
             current_strategy_symbol = strategy.symbol # Assumes base Strategy has .symbol
             if current_strategy_symbol not in state.order_depths:
                  logger.print(f"Strategy {strategy_key}: Market data missing for its own symbol '{current_strategy_symbol}'. Skipping run.")
                  market_data_available = False
             elif any(prod not in state.order_depths for prod in required_products):
                  missing_prods = [p for p in required_products if p not in state.order_depths]
                  logger.print(f"Strategy {strategy_key}: Market data missing for required products {missing_prods}. Skipping run.")
                  market_data_available = False

             if market_data_available:
                 try:
                     # Simplified run logic 
                     strategy.run(state)
                     all_orders.extend(strategy.orders)
                 except Exception as e:
                     logger.print(f"*** ERROR running {strategy_key} strategy: {e} ***");
                     import traceback; logger.print(traceback.format_exc())

             # Save state for this strategy
             try:
                 # Simplified save logic
                 trader_data_for_next_round[str(strategy_key)] = strategy.save()
             except Exception as e:
                  logger.print(f"Error saving state for {strategy_key}: {e}")
                  trader_data_for_next_round[str(strategy_key)] = {}

         # Group orders by symbol for the final output format
         final_result: Dict[Symbol, List[Order]] = defaultdict(list)
         for order in all_orders:
            if not isinstance(order.quantity, int):
                 order.quantity = int(round(order.quantity))
            if not isinstance(order.price, int):
                 order.price = int(round(order.price))
            if order.quantity != 0:
                 final_result[order.symbol].append(order)
         try:
              trader_data_to_encode = {str(k): v for k, v in trader_data_for_next_round.items()}
              traderData_encoded = json.dumps(trader_data_to_encode, separators=(",", ":"), cls=ProsperityEncoder)
         except Exception as e:
             logger.print(f"Error encoding traderData: {e}"); traderData_encoded = "{}"
         logger.flush(state, dict(final_result), conversions, traderData_encoded)
         return dict(final_result), conversions, traderData_encoded 