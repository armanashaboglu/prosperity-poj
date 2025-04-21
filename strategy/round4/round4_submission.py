import json
import numpy as np
from typing import Any, Dict, List, Deque, Optional, Tuple
from abc import abstractmethod
from collections import deque, defaultdict
import math
import copy

from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, Symbol, Trade, ProsperityEncoder

# Use the Logger from round4_macarons as it includes observation compression for R4
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

    def compress_listings(self, listings: dict[Symbol, Any]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Any) -> list[Any]:
        conversion_observations = {}
        # R4 update: Check if observations and conversionObservations exist
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
            for product, observation in observations.conversionObservations.items():
                # Check for R4 specific fields
                sugar_price = getattr(observation, 'sugarPrice', None)
                sunlight_index = getattr(observation, 'sunlightIndex', None)
                humidity = getattr(observation, 'humidity', None) # Add humidity for R4/R5
                
                compressed_obs = [
                    getattr(observation, 'bidPrice', None),
                    getattr(observation, 'askPrice', None),
                    getattr(observation, 'transportFees', None),
                    getattr(observation, 'exportTariff', None),
                    getattr(observation, 'importTariff', None),
                ]
                # Only add R4 fields if they exist
                if sugar_price is not None: compressed_obs.append(sugar_price)
                if sunlight_index is not None: compressed_obs.append(sunlight_index)
                if humidity is not None: compressed_obs.append(humidity) # Add humidity

                conversion_observations[product] = compressed_obs

        # R4 update: Check for plainValueObservations
        plain_value_obs = {}
        if hasattr(observations, 'plainValueObservations') and observations.plainValueObservations:
           plain_value_obs = observations.plainValueObservations

        return [plain_value_obs, conversion_observations]


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

# --- Product Enums (Combined) ---
class Product:
    # Round 3 Products (excluding SQUID_INK)
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    # SQUID_INK = "SQUID_INK" # Excluded
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    B1B2_DEVIATION = "B1B2_DEVIATION" # Virtual product for spread trading
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    # Round 4 Products
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS" # From R4 file

# --- Round 3 Basket Definitions ---
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

# --- Round 3 Theoretical Spread Definition ---
B1B2_THEORETICAL_COMPONENTS = {
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: 1
}

# --- Strategy Parameters (Combined, excluding SQUID_INK) ---
PARAMS = {
    # "SQUID_INK": { # Excluded
    #     "rsi_window": 96, "rsi_overbought": 56, "rsi_oversold": 39, "price_offset": 1
    # },
    "VOLCANIC_ROCK": { # RSI strategy from R3
        "rsi_window": 85, "rsi_overbought": 52, "rsi_oversold": 42, "price_offset": 0
    },
    Product.B1B2_DEVIATION: { # Basket deviation strategy from R3
        "deviation_mean": 0, # From notebook analysis (or optimize)
        "deviation_std_window": 140, # Rolling window for std dev calc
        "zscore_threshold_entry": 13.6, # Z-score to enter
        "zscore_threshold_exit": 0.11, # Z-score to exit towards mean
        "target_deviation_spread_size": 60, # Target units of deviation spread to hold
    },
    # Add other parameters needed for R3 strategies (e.g., Resin MM fair value)
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000 # Example fair value for Resin MM
    }
    # Note: Kelp strategy gets fair value dynamically, Resin uses this parameter
    # Note: Macarons strategy has its params internally managed for now
}

# --- Round 3 Voucher-specific constants ---
STRIKES = {
    Product.VOUCHER_9500: 9500,
    Product.VOUCHER_9750: 9750,
    Product.VOUCHER_10000: 10000,
    Product.VOUCHER_10250: 10250,
    Product.VOUCHER_10500: 10500,
}

# --- Round 3 Black-Scholes functions ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def calculate_option_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0: return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def calculate_implied_volatility(option_price, S, K, T, r=0, initial_vol=0.3, max_iterations=50, precision=0.0001):
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0: return 0.0
    intrinsic_value = max(0, S - K)
    if option_price <= intrinsic_value: return 0.0
    vol = initial_vol
    for i in range(max_iterations):
        price = calculate_option_price(S, K, T, r, vol)
        diff = option_price - price
        if abs(diff) < precision: return vol
        try:
            d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
            vega = S * math.sqrt(T) * norm_pdf(d1)
            if vega == 0: return vol
            vol = vol + diff / vega
            if vol <= 0: vol = 0.0001
            elif vol > 5: vol = 5.0
        except (ValueError, OverflowError): # Catch potential math errors
             # If calculation fails, return current vol or fallback
             logger.print(f"Warning: Math error in IV calc (vol={vol}), returning current/fallback vol.")
             return vol # Return the last valid vol, or initial_vol/fallback if error happens early
    return vol # Return last calculated vol if max_iterations reached

# --- Round 3 Volatility Smile Strategy ---
class VolatilitySmileStrategy:
    # ... (Keep the entire class definition from round3_submission.py) ...
    def __init__(self) -> None:
        """Initialize the Volatility Smile trader for all vouchers"""
        self.position_limits = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200,
            Product.VOUCHER_10000: 200,
            Product.VOUCHER_10250: 200,
            Product.VOUCHER_10500: 200,
        }
        self.voucher_symbols = [
            Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000,
            Product.VOUCHER_10250, Product.VOUCHER_10500
        ]
        self.short_ewma_span = 10
        self.long_ewma_span = 51
        self.rolling_window = 30
        self.zscore_upper_threshold = 0.5
        self.zscore_lower_threshold = -2.8
        self.trade_size = 11
        self.base_iv_history = deque(maxlen=200)
        self.short_ewma_base_iv = None
        self.long_ewma_first = None
        self.long_ewma_base_iv = None
        self.ewma_diff_history = deque(maxlen=200)
        self.zscore_history = deque(maxlen=100)
        self.day = 4 # Should be set appropriately based on simulation day
        self.last_timestamp = None
        self.orders = {} # Note: VolSmile manages orders differently

    def update_time_to_expiry(self, timestamp):
        base_tte = 8 - self.day
        iteration = (timestamp % 1000000) // 100
        iteration_adjustment = iteration / 10000
        tte = (base_tte - iteration_adjustment) / 365
        return max(0.0001, tte)

    def get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        elif best_bid is not None: return best_bid
        elif best_ask is not None: return best_ask
        else: return None

    def calculate_order_size(self, symbol: Symbol, zscore: float, state: TradingState) -> int:
        current_position = state.position.get(symbol, 0)
        position_limit = self.position_limits.get(symbol, 0)
        fixed_size = self.trade_size
        if zscore > 0:
            if current_position - fixed_size >= -position_limit: return -fixed_size
            else: return 0
        else:
            if current_position + fixed_size <= position_limit: return fixed_size
            else: return 0

    def place_order(self, orders_dict, symbol, price, quantity):
        if quantity == 0: return
        price = int(round(price)) # Ensure integer price
        quantity = int(round(quantity)) # Ensure integer quantity
        if quantity == 0: return # Check again after rounding
        if symbol not in orders_dict: orders_dict[symbol] = []
        orders_dict[symbol].append(Order(symbol, price, quantity))
        # logger.print(f"PLACE {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)}x{price}")

    def update_ewma(self, current_value, previous_ewma, span):
        if previous_ewma is None: return current_value
        alpha = 2 / (span + 1)
        return alpha * current_value + (1 - alpha) * previous_ewma

    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders_dict = {}
        self.last_timestamp = state.timestamp
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)
        if not rock_price:
            # logger.print("No price available for VOLCANIC_ROCK, skipping VolSmile iteration")
            return orders_dict

        # Risk Management: Flatten deep OTM positions
        for voucher in self.voucher_symbols:
            strike = STRIKES[voucher]
            if rock_price - strike <= -250:
                current_position = state.position.get(voucher, 0)
                if current_position != 0:
                    voucher_price = self.get_mid_price(voucher, state)
                    order_depth = state.order_depths.get(voucher)
                    if voucher_price and order_depth:
                        if current_position > 0 and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            self.place_order(orders_dict, voucher, best_bid, -current_position)
                        elif current_position < 0 and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            self.place_order(orders_dict, voucher, best_ask, -current_position)

        moneyness_values = []
        iv_values = []
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price: continue
            strike = STRIKES[voucher]
            if time_to_expiry <= 0: continue # Avoid division by zero
            try:
                moneyness = math.log(strike / rock_price) / math.sqrt(time_to_expiry)
                impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, time_to_expiry)
                if impl_vol > 0:
                    moneyness_values.append(moneyness)
                    iv_values.append(impl_vol)
            except (ValueError, OverflowError):
                # logger.print(f"Math error calculating moneyness/IV for {voucher}. Skipping.")
                continue


        if len(moneyness_values) >= 3:
            try:
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                base_iv = c
                # logger.print(f"Base IV (ATM): {base_iv:.6f}")
                self.base_iv_history.append(base_iv)

                self.short_ewma_base_iv = self.update_ewma(base_iv, self.short_ewma_base_iv, self.short_ewma_span)
                self.long_ewma_first = self.update_ewma(base_iv, self.long_ewma_first, self.long_ewma_span)
                self.long_ewma_base_iv = self.update_ewma(self.long_ewma_first, self.long_ewma_base_iv, self.long_ewma_span)

                if len(self.base_iv_history) >= self.rolling_window and self.short_ewma_base_iv is not None and self.long_ewma_base_iv is not None:
                    ewma_diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                    if not hasattr(self, 'ewma_diff_history'): self.ewma_diff_history = deque(maxlen=200)
                    self.ewma_diff_history.append(ewma_diff)

                    if len(self.ewma_diff_history) >= self.rolling_window:
                        recent_ewma_diffs = list(self.ewma_diff_history)[-self.rolling_window:]
                        rolling_std = np.std(recent_ewma_diffs)
                    else:
                        rolling_std = np.std(list(self.ewma_diff_history)) if self.ewma_diff_history else 0

                    zscore = ewma_diff / rolling_std if rolling_std > 1e-9 else 0
                    self.zscore_history.append(zscore)
                    # logger.print(f"Z-score: {zscore:.4f}, Upper: {self.zscore_upper_threshold}, Lower: {self.zscore_lower_threshold}")

                    # Trading Logic
                    for voucher in self.voucher_symbols:
                        # Skip deep OTM
                        strike = STRIKES[voucher]
                        if rock_price - strike <= -250: continue

                        order_depth = state.order_depths.get(voucher)
                        if not order_depth: continue

                        order_size = self.calculate_order_size(voucher, zscore, state)
                        if order_size == 0: continue

                        if zscore > self.zscore_upper_threshold: # Sell Signal
                            if order_depth.buy_orders:
                                best_bid = max(order_depth.buy_orders.keys())
                                self.place_order(orders_dict, voucher, best_bid, order_size) # order_size is negative
                            # else: logger.print(f"No best bid for {voucher} on sell signal.")
                        elif zscore < self.zscore_lower_threshold: # Buy Signal
                            if order_depth.sell_orders:
                                best_ask = min(order_depth.sell_orders.keys())
                                self.place_order(orders_dict, voucher, best_ask, order_size) # order_size is positive
                            # else: logger.print(f"No best ask for {voucher} on buy signal.")

            except Exception as e:
                # logger.print(f"Error in VolSmile polyfit/trading logic: {e}")
                import traceback
                # logger.print(traceback.format_exc())
                pass # Avoid crashing the whole trader

        self.orders = orders_dict # Internal tracking for the strategy if needed
        return orders_dict

    def save_state(self) -> dict:
        return {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history) if hasattr(self, 'ewma_diff_history') else [],
            "zscore_history": list(self.zscore_history),
            "last_timestamp": self.last_timestamp
        }

    def load_state(self, data: dict) -> None:
        if not data: return
        self.day = data.get("day", self.day)
        self.base_iv_history = deque(data.get("base_iv_history", []), maxlen=200)
        self.short_ewma_base_iv = data.get("short_ewma_base_iv")
        self.long_ewma_first = data.get("long_ewma_first")
        self.long_ewma_base_iv = data.get("long_ewma_base_iv")
        self.ewma_diff_history = deque(data.get("ewma_diff_history", []), maxlen=200)
        self.zscore_history = deque(data.get("zscore_history", []), maxlen=100)
        self.last_timestamp = data.get("last_timestamp")
    # --- End of VolatilitySmileStrategy ---

# --- Round 3 Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()

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
        # logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}")

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        # logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}")

    def save(self) -> dict: return {}
    def load(self, data: dict) -> None: pass

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        elif best_bid is not None: return best_bid
        elif best_ask is not None: return best_ask
        else: return None
    # --- End of Base Strategy ---

# --- Round 3 V3 Strategies (Base for Resin) ---
class V3Strategy(Strategy): # Inherits from R3 Strategy base
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)

    # act is abstract in parent

    # run is inherited from parent

    def buy(self, price: int, quantity: int) -> None: # Specific buy helper for V3
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, quantity))
        # logger.print(f"{self.symbol} BUY {quantity}x{price}")

    def sell(self, price: int, quantity: int) -> None: # Specific sell helper for V3
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        # logger.print(f"{self.symbol} SELL {quantity}x{price}")

    # save/load inherited from parent

class V3MarketMakingStrategy(V3Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window = deque(maxlen=4)
        self.window_size = 4

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int: raise NotImplementedError()

    # Original MM act logic (kept for reference, overridden by Resin)
    def _base_act(self, state: TradingState) -> None:
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
        if to_buy > 0 and hard_liquidate: self.buy(true_value, to_buy // 2); to_buy -= to_buy // 2
        if to_buy > 0 and soft_liquidate: self.buy(true_value - 2, to_buy // 2); to_buy -= to_buy // 2
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
        if to_sell > 0 and hard_liquidate: self.sell(true_value, to_sell // 2); to_sell -= to_sell // 2
        if to_sell > 0 and soft_liquidate: self.sell(true_value + 2, to_sell // 2); to_sell -= to_sell // 2
        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else (true_value + 2)
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    # This act method will be used by Resin
    def act(self, state: TradingState) -> None: raise NotImplementedError("Subclasses should implement act")

    def save(self) -> dict:
        data = super().save()
        data["window"] = list(self.window)
        return data

    def load(self, data: dict) -> None:
        super().load(data)
        if data and "window" in data:
            loaded_window = data["window"]
            start_index = max(0, len(loaded_window) - self.window.maxlen)
            self.window.clear()
            self.window.extend(loaded_window[start_index:])

class V3RainforestResinStrategy(V3MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # Use parameter if available, else default
        return PARAMS.get(self.symbol, {}).get("fair_value", 10000)

    def act(self, state: TradingState) -> None: # Overrides base MM act
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

        # Slurp
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

        # Market Make
        make_bid_price = true_value - 1
        bids_below_tv = {p: v for p, v in sim_buy_orders.items() if p < true_value}
        if bids_below_tv:
            best_bid_after_take = max(bids_below_tv.keys())
            best_bid_vol = bids_below_tv[best_bid_after_take]
            if best_bid_vol <= 6: make_bid_price = best_bid_after_take
            else: make_bid_price = best_bid_after_take + 1
        make_ask_price = true_value + 1
        asks_above_tv = {p: v for p, v in sim_sell_orders.items() if p > true_value}
        if asks_above_tv:
            best_ask_after_take = min(asks_above_tv.keys())
            best_ask_vol = abs(asks_above_tv[best_ask_after_take])
            if best_ask_vol <= 6: make_ask_price = best_ask_after_take
            else: make_ask_price = best_ask_after_take - 1
        if make_bid_price >= make_ask_price: make_ask_price = make_bid_price + 1
        if to_buy > 0: self.buy(make_bid_price, to_buy)
        if to_sell > 0: self.sell(make_ask_price, to_sell)
    # --- End of V3 Strategies ---


# --- Round 3 Prototype Strategies (Base for Kelp) ---
class PrototypeMarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window_size = 4
        self.window: Deque[bool] = deque(maxlen=self.window_size)

    @abstractmethod
    def get_true_value(self, state: TradingState) -> Optional[int]: raise NotImplementedError()

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
            if to_buy > 0: quantity = to_buy // 2; self._place_buy_order(true_value, quantity); to_buy -= quantity
            elif to_sell > 0: quantity = to_sell // 2; self._place_sell_order(true_value, quantity); to_sell -= quantity
        elif soft_liquidate:
            if to_buy > 0: quantity = to_buy // 2; self._place_buy_order(true_value - 2, quantity); to_buy -= quantity
            elif to_sell > 0: quantity = to_sell // 2; self._place_sell_order(true_value + 2, quantity); to_sell -= quantity

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
        data = super().save()
        data["window"] = list(self.window)
        return data

    def load(self, data: dict) -> None:
        super().load(data)
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
        popular_buy_price = max(buy_levels, key=lambda tup: tup[1])[0] if buy_levels else 0
        popular_sell_price = min(sell_levels, key=lambda tup: tup[1])[0] if sell_levels else 0
        final_value = None
        if popular_buy_price > 0 and popular_sell_price > 0 and popular_sell_price > popular_buy_price:
            final_value = (popular_buy_price + popular_sell_price) / 2
        else:
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 0
            if best_bid > 0 and best_ask > 0 and best_ask > best_bid: final_value = (best_bid + best_ask) / 2
            elif best_bid > 0: final_value = best_bid
            elif best_ask > 0: final_value = best_ask
        return round(final_value) if final_value is not None else None
    # --- End of Prototype Strategies ---


# --- Round 3 RSI Strategy ---
class RsiStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
            # logger.print(f"ERROR: RSI params for {self.symbol} not found. Using defaults.")
            self.params = {"rsi_window": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0, "price_offset": 0}
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2: self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.price_offset = self.params.get("price_offset", 0)
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False
        # logger.print(f"Initialized RsiStrategy for {self.symbol}: Win={self.window}, OB={self.overbought_threshold}, OS={self.oversold_threshold}, Offset={self.price_offset}")

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        self.mid_price_history.append(current_mid_price)
        if len(self.mid_price_history) < self.window + 1: return None
        prices = list(self.mid_price_history)
        if len(prices) < 2: return None
        changes = np.diff(prices)
        if changes.size == 0: return None
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        if len(gains) < self.window: return None

        if not self.rsi_initialized or self.avg_gain is None or self.avg_loss is None:
            self.avg_gain = np.mean(gains[-self.window:])
            self.avg_loss = np.mean(losses[-self.window:])
            self.rsi_initialized = True
        else:
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss is not None and self.avg_loss < 1e-9: return 100.0
        elif self.avg_gain is None or self.avg_loss is None: return None
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth: return
        current_mid_price = self._get_mid_price(self.symbol, state)
        if current_mid_price is None: return
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None: return

        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position
        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            if best_bid_price is not None:
                size_to_sell = to_sell_capacity
                aggressive_sell_price = best_bid_price - self.price_offset
                if aggressive_sell_price <= 0: aggressive_sell_price = best_bid_price
                self._place_sell_order(aggressive_sell_price, size_to_sell)
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            if best_ask_price is not None:
                size_to_buy = to_buy_capacity
                aggressive_buy_price = best_ask_price + self.price_offset
                self._place_buy_order(aggressive_buy_price, size_to_buy)

    def save(self) -> dict:
        data = super().save()
        data.update({
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        })
        return data

    def load(self, data: dict) -> None:
        super().load(data)
        loaded_history = data.get("mid_price_history", [])
        if isinstance(loaded_history, list):
             start_index = max(0, len(loaded_history) - (self.window + 1))
             self.mid_price_history = deque(loaded_history[start_index:], maxlen=self.window + 1)
        else: self.mid_price_history = deque(maxlen=self.window + 1)
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        if self.avg_gain is not None:
            try: self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError): self.avg_gain = None
        if self.avg_loss is not None:
            try: self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError): self.avg_loss = None
        self.rsi_initialized = data.get("rsi_initialized", False)
        if not isinstance(self.rsi_initialized, bool): self.rsi_initialized = False
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            self.rsi_initialized = False; self.avg_gain = None; self.avg_loss = None
    # --- End of RSI Strategy ---


# --- Round 3 B1B2 Deviation Strategy ---
class B1B2DeviationStrategy(Strategy):
    def __init__(self, symbol: str, position_limits: Dict[str, int]) -> None:
        super().__init__(symbol, 0) # Virtual product, no direct limit
        self.pos_limits = position_limits
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
            self.params = {"deviation_mean": 0, "deviation_std_window": 500, "zscore_threshold_entry": 2.0, "zscore_threshold_exit": 0.5, "target_deviation_spread_size": 10}
            # logger.print(f"Warning: {self.symbol} params not found, using defaults.")
        self.deviation_mean = self.params.get("deviation_mean", 0)
        self.deviation_std_window = self.params.get("deviation_std_window", 500)
        if self.deviation_std_window < 10: self.deviation_std_window = 10
        self.zscore_entry = abs(self.params.get("zscore_threshold_entry", 2.0))
        self.zscore_exit = abs(self.params.get("zscore_threshold_exit", 0.5))
        self.target_size = abs(self.params.get("target_deviation_spread_size", 10))
        self.deviation_history: Deque[float] = deque(maxlen=self.deviation_std_window)
        self.current_effective_deviation_pos: int = 0

    def _place_order(self, product_symbol: Symbol, price: int, quantity: int) -> None:
         if quantity == 0: return
         if price <= 0: return
         self.orders.append(Order(product_symbol, int(round(price)), int(round(quantity)))) # Ensure int

    def _get_mid_price_safe(self, product: Symbol, state: TradingState) -> Optional[float]:
        return super()._get_mid_price(product, state)

    def act(self, state: TradingState) -> None:
        self.orders = []
        mid_b1 = self._get_mid_price_safe(Product.PICNIC_BASKET1, state)
        mid_b2 = self._get_mid_price_safe(Product.PICNIC_BASKET2, state)
        mid_c = self._get_mid_price_safe(Product.CROISSANTS, state)
        mid_j = self._get_mid_price_safe(Product.JAMS, state)
        mid_d = self._get_mid_price_safe(Product.DJEMBES, state)
        required_products = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        if any(prod not in state.order_depths for prod in required_products): return
        if None in [mid_b1, mid_b2, mid_c, mid_j, mid_d]: return

        actual_spread = mid_b1 - mid_b2
        theoretical_spread = (B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * mid_c +
                              B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * mid_j +
                              B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * mid_d)
        deviation = actual_spread - theoretical_spread
        self.deviation_history.append(deviation)

        if len(self.deviation_history) < self.deviation_std_window // 4: return

        current_deviation_history = list(self.deviation_history)
        deviation_std = np.std(current_deviation_history)
        if deviation_std < 1e-6: return

        z_score = (deviation - self.deviation_mean) / deviation_std
        # logger.print(f"{self.symbol}: Dev={deviation:.2f}, Z={z_score:.2f}, CurrEffPos={self.current_effective_deviation_pos}")

        desired_effective_deviation_pos = self.current_effective_deviation_pos
        if z_score >= self.zscore_entry: desired_effective_deviation_pos = -self.target_size
        elif z_score <= -self.zscore_entry: desired_effective_deviation_pos = self.target_size
        else:
            if self.current_effective_deviation_pos > 0 and z_score >= -self.zscore_exit: desired_effective_deviation_pos = 0
            elif self.current_effective_deviation_pos < 0 and z_score <= self.zscore_exit: desired_effective_deviation_pos = 0

        if desired_effective_deviation_pos != self.current_effective_deviation_pos:
            self._execute_deviation_trade(state, desired_effective_deviation_pos)

    def _calculate_max_deviation_spread_size(self, state: TradingState, direction: int) -> int:
        if direction == 0: return 0
        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction, Product.PICNIC_BASKET2: -direction,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * direction,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * direction,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * direction,
        }
        max_units = float('inf')
        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue
            current_pos = state.position.get(product, 0)
            limit = self.pos_limits.get(product)
            if limit is None: return 0 # Should have limit
            if limit == 0: return 0
            capacity = (limit - current_pos) if qty_change > 0 else (limit + current_pos)
            if capacity < 0: capacity = 0
            max_units_for_product = capacity // abs(qty_change) if abs(qty_change) > 0 else float('inf')
            max_units = min(max_units, max_units_for_product)
        return max(0, int(max_units))

    def _calculate_market_liquidity_limit(self, state: TradingState, direction: int) -> int:
        if direction == 0: return 0
        order_depths = state.order_depths
        max_units = float('inf')
        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction, Product.PICNIC_BASKET2: -direction,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * direction,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * direction,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * direction,
        }
        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue
            od = order_depths.get(product)
            if not od: return 0
            total_available_volume = 0
            if qty_change > 0: # Need to BUY -> sum SELL side volume
                if not od.sell_orders: return 0
                total_available_volume = sum(abs(vol) for vol in od.sell_orders.values())
            else: # Need to SELL -> sum BUY side volume
                if not od.buy_orders: return 0
                total_available_volume = sum(abs(vol) for vol in od.buy_orders.values())
            if total_available_volume <= 0: return 0
            units_fillable = total_available_volume // abs(qty_change) if abs(qty_change) > 0 else float('inf')
            max_units = min(max_units, units_fillable)
        return max(0, int(max_units))

    def _place_aggressive_orders_for_leg(self, product_symbol: Symbol, total_quantity_needed: int, order_depth: OrderDepth):
        if total_quantity_needed == 0: return
        orders_to_place = []
        remaining_qty = abs(total_quantity_needed)
        if total_quantity_needed > 0: # Need to BUY
            if not order_depth.sell_orders: return
            sorted_levels = sorted(order_depth.sell_orders.items())
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break
        else: # Need to SELL
            if not order_depth.buy_orders: return
            sorted_levels = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, -int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break
        self.orders.extend(orders_to_place) # Add to main list for the strategy

    def _execute_deviation_trade(self, state: TradingState, target_effective_pos: int):
        qty_units_to_trade = target_effective_pos - self.current_effective_deviation_pos
        if qty_units_to_trade == 0: return
        direction = 1 if qty_units_to_trade > 0 else -1
        max_units_pos = self._calculate_max_deviation_spread_size(state, direction)
        if max_units_pos <= 0: return
        max_units_liq = self._calculate_market_liquidity_limit(state, direction)
        if max_units_liq <= 0: return
        actual_units_to_trade = direction * min(abs(qty_units_to_trade), max_units_pos, max_units_liq)
        if actual_units_to_trade == 0: return
        # logger.print(f"Execute: Attempting to trade {actual_units_to_trade} deviation units (LimitPos: {max_units_pos}, LimitLiq: {max_units_liq}).")
        final_qty_changes = {
            Product.PICNIC_BASKET1: +actual_units_to_trade, Product.PICNIC_BASKET2: -actual_units_to_trade,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * actual_units_to_trade,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * actual_units_to_trade,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * actual_units_to_trade,
        }
        order_depths = state.order_depths
        for product, final_qty_int in final_qty_changes.items():
            final_qty = int(round(final_qty_int))
            if final_qty == 0: continue
            od = order_depths.get(product)
            if not od:
                # logger.print(f"Error: Order depth for {product} disappeared in B1B2 trade!")
                continue
            self._place_aggressive_orders_for_leg(product, final_qty, od)
        self.current_effective_deviation_pos += actual_units_to_trade
        # logger.print(f"Execute: Aggressive orders placed for B1B2. New effective pos: {self.current_effective_deviation_pos}")

    def save(self) -> dict:
        data = super().save()
        data.update({
            "deviation_history": list(self.deviation_history),
            "current_effective_deviation_pos": self.current_effective_deviation_pos
        })
        return data

    def load(self, data: dict) -> None:
        super().load(data)
        loaded_history = data.get("deviation_history", [])
        if isinstance(loaded_history, list):
             self.deviation_history = deque(loaded_history, maxlen=self.deviation_std_window)
        else: self.deviation_history = deque(maxlen=self.deviation_std_window)
        loaded_pos = data.get("current_effective_deviation_pos", 0)
        if isinstance(loaded_pos, (int, float)): self.current_effective_deviation_pos = int(loaded_pos)
        else: self.current_effective_deviation_pos = 0
    # --- End of B1B2 Deviation Strategy ---


# --- Round 4 Macarons Strategy ---
class MacaronsStrategy:
    # ... (Keep the entire class definition from round4_macarons.py) ...
    def __init__(self) -> None:
        self.symbol = Product.MAGNIFICENT_MACARONS
        self.position_limit = 75
        self.conversion_limit = 10
        self.price_history = deque(maxlen=200)
        self.mid_price_history = deque(maxlen=200)
        self.sunlight_history = deque(maxlen=5)
        self.orders = []
        self.fair_value_lower = 550
        self.fair_value_upper = 700
        self.slope_lower_threshold = -0.011
        self.slope_upper_threshold = 0.011
        self.rsi_period = 130
        self.rsi_oversold = 40
        self.rsi_overbought = 58
        self.sunlight_value_threshold = 45
        self.current_sunlight_index = None
        self.trade_size_pct = 1.0
        self.target_position = 0
        self.rsi_target_units = 20
        self.active_signal = None

    def get_position_adjusted_volumes(self, state: TradingState) -> Tuple[int, int]:
        position = state.position.get(self.symbol, 0)
        return self.position_limit - position, self.position_limit + position

    def update_state(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths: return
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders: return
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        self.mid_price_history.append(mid_price)
        self.price_history.append(mid_price)

        sunlight_found = False
        if hasattr(state, 'observations') and state.observations:
            if hasattr(state.observations, 'plainValueObservations') and state.observations.plainValueObservations:
                 # R4: Sunlight is usually in plainValueObservations
                 sunlight_keys = ['sunlightIndex', 'SUNLIGHT_INDEX', 'sunlight_index', 'SunlightIndex']
                 for key in sunlight_keys:
                      if key in state.observations.plainValueObservations:
                           self.current_sunlight_index = state.observations.plainValueObservations[key]
                           self.sunlight_history.append(self.current_sunlight_index)
                           sunlight_found = True
                           break
            # Fallback to conversionObservations if not found in plain (less likely for sunlight)
            if not sunlight_found and hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations:
                 for product, obs_data in state.observations.conversionObservations.items():
                      if hasattr(obs_data, 'sunlightIndex') and obs_data.sunlightIndex is not None:
                           self.current_sunlight_index = obs_data.sunlightIndex
                           self.sunlight_history.append(self.current_sunlight_index)
                           sunlight_found = True
                           break # Found it, stop checking conversions

        if not sunlight_found and not self.sunlight_history: # Only if truly missing and history is empty
             dummy_value = 6750 # Use a neutral-ish dummy value based on typical range
             self.sunlight_history.append(dummy_value)
             self.current_sunlight_index = dummy_value


    def calculate_sunlight_slope(self) -> Optional[float]:
        if len(self.sunlight_history) < 2: return None
        try:
            current = self.sunlight_history[-1]
            previous = self.sunlight_history[-2]
            # Avoid using dummy if real data exists
            if len(self.sunlight_history) > 2 and abs(previous - 6750) < 1:
                previous = self.sunlight_history[-3] # Use older real data if possible
            if abs(current - 6750) < 1 and abs(previous - 6750) < 1:
                return 0.0 # Still only have dummy values
            slope = current - previous
            return slope
        except Exception: return 0.0 # Default to no slope on error

    def calculate_rsi(self) -> Optional[float]:
        if len(self.price_history) < self.rsi_period + 1: return None
        prices = list(self.price_history)[-self.rsi_period-1:]
        deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        # Ensure we have enough actual gain/loss data points for the period
        if len(gains) < self.rsi_period: return None

        avg_gain = sum(gains[:self.rsi_period]) / self.rsi_period
        avg_loss = sum(losses[:self.rsi_period]) / self.rsi_period
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _place_buy_order(self, price: float, quantity: int) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(round(quantity)) # Round quantity too
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, quantity))

    def _place_sell_order(self, price: float, quantity: int) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(round(quantity)) # Round quantity too
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))

    def act(self, state: TradingState) -> Tuple[List[Order], int]:
        self.orders = []
        conversion_request = 0
        if self.symbol not in state.order_depths: return self.orders, conversion_request
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders: return self.orders, conversion_request

        self.update_state(state)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        current_position = state.position.get(self.symbol, 0)
        available_to_buy, available_to_sell = self.get_position_adjusted_volumes(state)

        # Continue executing previous signal if needed
        if self.active_signal and current_position != self.target_position:
            direction = 1 if self.target_position > current_position else -1
            units_to_trade = abs(self.target_position - current_position)
            
            if direction > 0: # Need to buy more
                units_to_buy = min(available_to_buy, units_to_trade)
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                trade_units = min(units_to_buy, ask_volume)
                if trade_units > 0:
                    self._place_buy_order(best_ask, trade_units)
                    units_to_buy -= trade_units
                if units_to_buy > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_buy, self.conversion_limit - abs(conversion_request)) # Respect limit
                    conversion_request += conv_units
            else: # Need to sell more
                units_to_sell = min(available_to_sell, units_to_trade)
                bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                trade_units = min(units_to_sell, bid_volume)
                if trade_units > 0:
                    self._place_sell_order(best_bid, trade_units)
                    units_to_sell -= trade_units
                if units_to_sell > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_sell, self.conversion_limit - abs(conversion_request)) # Respect limit
                    conversion_request -= conv_units

            return self.orders, int(round(conversion_request)) # Return early

        # New signal detection
        slope = self.calculate_sunlight_slope()
        sunlight_value = self.current_sunlight_index
        signal_detected = False
        new_target_position = current_position # Default to holding

        if slope is not None and sunlight_value is not None:
             # Check slope/sunlight conditions...
             if slope < self.slope_lower_threshold:
                  new_target_position = self.position_limit
                  self.active_signal = "SLOPE_LONG"
                  signal_detected = True
             elif slope > self.slope_upper_threshold and current_mid > self.fair_value_upper:
                  new_target_position = -self.position_limit
                  self.active_signal = "SLOPE_SHORT"
                  signal_detected = True
             elif sunlight_value <= self.sunlight_value_threshold and slope < 0:
                  new_target_position = self.position_limit
                  self.active_signal = "SUNLIGHT_LONG"
                  signal_detected = True
             elif sunlight_value >= self.sunlight_value_threshold and slope > 0 and current_mid > self.fair_value_upper:
                  new_target_position = -self.position_limit
                  self.active_signal = "SUNLIGHT_SHORT"
                  signal_detected = True

        # If no slope/sunlight signal, consider RSI
        if not signal_detected:
            if self.fair_value_lower <= current_mid <= self.fair_value_upper:
                rsi = self.calculate_rsi()
                if rsi is not None:
                    if rsi < self.rsi_oversold:
                        new_target_position = min(current_position + self.rsi_target_units, self.position_limit)
                        self.active_signal = "RSI_LONG"
                        signal_detected = True # RSI signal detected
                    elif rsi > self.rsi_overbought:
                        new_target_position = max(current_position - self.rsi_target_units, -self.position_limit)
                        self.active_signal = "RSI_SHORT"
                        signal_detected = True # RSI signal detected
                    else: # RSI is neutral
                         self.active_signal = None # Reset signal if RSI is neutral
                         new_target_position = current_position # Stay put
                         signal_detected = False # No active trading signal
            else: # Outside fair value, no RSI check
                 self.active_signal = None
                 new_target_position = current_position
                 signal_detected = False

        # Update target position if a new signal was detected
        if signal_detected:
            self.target_position = new_target_position

            # Start executing the new signal immediately
            units_needed = abs(self.target_position - current_position)
            direction = 1 if self.target_position > current_position else -1

            if direction > 0: # Need to buy
                units_to_buy = min(available_to_buy, units_needed)
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                trade_units = min(units_to_buy, ask_volume)
                if trade_units > 0:
                    self._place_buy_order(best_ask, trade_units)
                    units_to_buy -= trade_units
                if units_to_buy > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_buy, self.conversion_limit - abs(conversion_request))
                    conversion_request += conv_units
            elif direction < 0: # Need to sell
                 units_to_sell = min(available_to_sell, units_needed)
                 bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                 trade_units = min(units_to_sell, bid_volume)
                 if trade_units > 0:
                      self._place_sell_order(best_bid, trade_units)
                      units_to_sell -= trade_units
                 if units_to_sell > 0 and abs(conversion_request) < self.conversion_limit:
                      conv_units = min(units_to_sell, self.conversion_limit - abs(conversion_request))
                      conversion_request -= conv_units

        return self.orders, int(round(conversion_request))


    def run(self, state: TradingState) -> Tuple[List[Order], int]: # Match R3 return type
        return self.act(state) # run just calls act

    def save(self) -> dict:
        return {
            "price_history": list(self.price_history),
            "mid_price_history": list(self.mid_price_history),
            "sunlight_history": list(self.sunlight_history),
            "current_sunlight_index": self.current_sunlight_index,
            "target_position": self.target_position,
            "active_signal": self.active_signal
        }

    def load(self, data: dict) -> None:
        if not data: return
        self.price_history = deque(data.get("price_history", []), maxlen=200)
        self.mid_price_history = deque(data.get("mid_price_history", []), maxlen=200)
        self.sunlight_history = deque(data.get("sunlight_history", []), maxlen=5)
        self.current_sunlight_index = data.get("current_sunlight_index")
        self.target_position = data.get("target_position", 0)
        self.active_signal = data.get("active_signal")
    # --- End of Macarons Strategy ---

# --- Trader Class (Combined) ---
class Trader:
    def __init__(self):
        # Combined Position Limits
        self.position_limits = {
            # R3 Products (excluding SQUID_INK)
            Product.KELP: 50, Product.RAINFOREST_RESIN: 50,
            Product.CROISSANTS: 250, Product.JAMS: 350, Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60, Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOUCHER_9500: 200, Product.VOUCHER_9750: 200, Product.VOUCHER_10000: 200,
            Product.VOUCHER_10250: 200, Product.VOUCHER_10500: 200,
            # R4 Products
            Product.MAGNIFICENT_MACARONS: 75
        }

        # Instantiate all strategies
        self.strategies = {
            # R3 Strategies (excluding SQUID_INK)
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]),
            Product.KELP: PrototypeKelpStrategy(Product.KELP, self.position_limits[Product.KELP]),
            Product.B1B2_DEVIATION: B1B2DeviationStrategy(Product.B1B2_DEVIATION, self.position_limits), # Pass full limits dict
            Product.VOLCANIC_ROCK: RsiStrategy(Product.VOLCANIC_ROCK, self.position_limits[Product.VOLCANIC_ROCK]),
            "VOLATILITY_SMILE": VolatilitySmileStrategy(), # For vouchers
            # R4 Strategies
            Product.MAGNIFICENT_MACARONS: MacaronsStrategy(),
            
        }

        # Store strategy-specific states (e.g., macaron state uses different keys)
        self.strategy_states = {key: {} for key in self.strategies.keys()}
        self.strategy_states["VOLATILITY_SMILE"] = {} # VolSmile also needs its state tracked


    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
         all_orders: Dict[Symbol, List[Order]] = defaultdict(list)
         total_conversions = 0 # Macarons strategy returns conversions
         trader_data_for_next_round = {}

         # Load All Strategy States
         try:
             loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
             if not isinstance(loaded_data, dict): loaded_data = {}
             # Load state for each strategy using its specific key/method
             for key, strategy in self.strategies.items():
                  strategy_state_data = loaded_data.get(str(key), {})
                  try:
                      if isinstance(strategy, VolatilitySmileStrategy):
                          strategy.load_state(strategy_state_data)
                      elif isinstance(strategy, MacaronsStrategy):
                          strategy.load(strategy_state_data) # Macarons uses load()
                      else: # Default load for R3 strategies
                          strategy.load(strategy_state_data)
                  except Exception as e:
                      logger.print(f"Error loading state for {key}: {e}")
         except Exception as e:
             logger.print(f"Error loading traderData: {e}")
             loaded_data = {}


         # Run strategies and collect outputs
         for strategy_key, strategy in self.strategies.items():
             market_data_available = True
             required_products = []

             # Define required products based on strategy type
             if isinstance(strategy, B1B2DeviationStrategy):
                 required_products = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
             elif isinstance(strategy, VolatilitySmileStrategy):
                 required_products = [Product.VOLCANIC_ROCK] + strategy.voucher_symbols
             elif isinstance(strategy, MacaronsStrategy):
                  required_products = [strategy.symbol]
             elif hasattr(strategy, 'symbol') and str(strategy.symbol) in self.position_limits: # Standard single-product strategies
                 required_products = [str(strategy.symbol)]
             else:
                 logger.print(f"Warning: Unsure how to check market data for strategy key: {strategy_key}")

             # Check data availability
             missing_data = [p for p in required_products if p not in state.order_depths]
             if missing_data:
                 # logger.print(f"Strategy {strategy_key}: Market data missing for {missing_data}. Skipping run.")
                 market_data_available = False

             if market_data_available:
                 try:
                     strategy_orders = []
                     strategy_conversions = 0 # Default to 0 conversions

                     if isinstance(strategy, VolatilitySmileStrategy):
                         # VolSmile returns a dict of orders {symbol: [Order, ...]}
                         orders_dict = strategy.run(state)
                         for symbol, orders_list in orders_dict.items():
                             all_orders[symbol].extend(orders_list)
                     elif isinstance(strategy, MacaronsStrategy):
                         # Macarons returns (list[Order], int_conversions)
                         strategy_orders, strategy_conversions = strategy.run(state)
                         if strategy_orders: # Check if list is not empty
                             all_orders[strategy.symbol].extend(strategy_orders)
                         total_conversions += strategy_conversions # Add macaron conversions
                     elif isinstance(strategy, B1B2DeviationStrategy):
                        # B1B2 Deviation strategy populates strategy.orders with orders for MULTIPLE symbols
                        strategy.run(state) # Populates strategy.orders
                        if strategy.orders:
                            for order in strategy.orders:
                                all_orders[order.symbol].append(order) # Add order under its ACTUAL symbol
                     else:
                         # Default R3 strategies (like Kelp, Resin, Volc Rock RSI) return list[Order] via strategy.orders
                         strategy.run(state) # Populates strategy.orders
                         if strategy.orders: # Check if list is not empty
                             # These strategies trade their own symbol
                             all_orders[strategy.symbol].extend(strategy.orders)

                 except Exception as e:
                     logger.print(f"*** ERROR running {strategy_key} strategy: {e} ***");
                     import traceback; logger.print(traceback.format_exc())

             # Save state for this strategy
             try:
                 if isinstance(strategy, VolatilitySmileStrategy):
                     trader_data_for_next_round[str(strategy_key)] = strategy.save_state()
                 elif isinstance(strategy, MacaronsStrategy):
                      trader_data_for_next_round[str(strategy_key)] = strategy.save() # Macarons uses save()
                 else: # Default save for R3 strategies
                     trader_data_for_next_round[str(strategy_key)] = strategy.save()
             except Exception as e:
                  logger.print(f"Error saving state for {strategy_key}: {e}")
                  trader_data_for_next_round[str(strategy_key)] = {} # Save empty dict on error


         # Final cleanup of orders (ensure ints, non-zero quantity)
         final_result: Dict[Symbol, List[Order]] = defaultdict(list)
         for symbol, orders_list in all_orders.items():
             for order in orders_list:
                 try:
                     order.price = int(round(order.price))
                     order.quantity = int(round(order.quantity))
                     if order.quantity != 0:
                         final_result[symbol].append(order)
                 except Exception as e:
                     logger.print(f"Error cleaning order for {symbol}: {order}. Error: {e}")


         # Encode final trader data & flush logs
         try:
              traderData_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"), cls=ProsperityEncoder)
         except Exception as e:
             logger.print(f"Error encoding traderData: {e}")
             traderData_encoded = "{}"

         logger.flush(state, dict(final_result), total_conversions, traderData_encoded)
         return dict(final_result), total_conversions, traderData_encoded
