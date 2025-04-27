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
    # Add Volcanic Rock Vouchers for Round 3
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    
# --- Signal Enum --- (RE-ADDED)
class Signal(Enum):
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    
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
        "rsi_window": 85,
        "rsi_overbought": 52,
        "rsi_oversold": 42,
        "price_offset": 0
    },

}

# --- Add Voucher-specific constants
STRIKES = {
    Product.VOUCHER_9500: 9500,
    Product.VOUCHER_9750: 9750,
    Product.VOUCHER_10000: 10000,
    Product.VOUCHER_10250: 10250,
    Product.VOUCHER_10500: 10500,
}

# --- Add Black-Scholes functions for option pricing ---
def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    """Standard normal probability density function."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def calculate_option_price(S, K, T, r, sigma):
    """Black-Scholes formula for call option price."""
    if sigma <= 0 or T <= 0:
        return max(0, S - K)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def calculate_implied_volatility(option_price, S, K, T, r=0, initial_vol=0.3, max_iterations=50, precision=0.0001):
    """Newton-Raphson method to find implied volatility - optimized version."""
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    
    intrinsic_value = max(0, S - K)
    if option_price <= intrinsic_value:
        return 0.0
    
    vol = initial_vol
    for i in range(max_iterations):
        price = calculate_option_price(S, K, T, r, vol)
        diff = option_price - price
        
        if abs(diff) < precision:
            return vol
        
        d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        vega = S * math.sqrt(T) * norm_pdf(d1)
        
        if vega == 0:
            return vol
     
        vol = vol + diff / vega
        
        if vol <= 0:
            vol = 0.0001
        elif vol > 5:  
            vol = 5.0
    
    return vol

# --- Add Volatility Smile Trader Strategy Class ---
class VolatilitySmileStrategy:
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
            Product.VOUCHER_9500, 
            Product.VOUCHER_9750, 
            Product.VOUCHER_10000, 
            Product.VOUCHER_10250, 
            Product.VOUCHER_10500
        ]
        
        # Parameters (fixed thresholds)
        self.short_ewma_span = 41
        self.long_ewma_span = 149
        self.rolling_window = 81
        self.zscore_upper_threshold = 2.9000000000000004
        self.zscore_lower_threshold = -1.0

        # --- Dynamic Threshold Parameters & State --- ADDED BACK
        self.threshold_adjustment_rate = 0.0014 # Optimizable rate
        self.current_intended_state = Signal.NEUTRAL # Tracks LONG/SHORT 
        self.time_in_state = 0 # Counter for timestamps in LONG/SHORT state
        
        # State Variables
        self.base_iv_history = deque(maxlen=200)
        self.short_ewma_base_iv = None
        self.long_ewma_first = None  
        self.long_ewma_base_iv = None  
        self.ewma_diff_history = deque(maxlen=200)
        self.zscore_history = deque(maxlen=100)
        self.day = 3
        self.last_timestamp = None
        self.orders = {}
    
    def update_time_to_expiry(self, timestamp):
        """Calculate time to expiry based on the current timestamp."""
        base_tte = 8 - self.day
        iteration = (timestamp % 1000000) // 100
        iteration_adjustment = iteration / 10000
        tte = (base_tte - iteration_adjustment) / 365
        return max(0.0001, tte)  
    
    def get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        """Get the mid price for a given symbol from the order depth."""
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            return None
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        else:
            return None
    
    def place_order(self, orders_dict, symbol, price, quantity):
        """Add an order to the orders dictionary."""

        if symbol in [Product.VOUCHER_10250, Product.VOUCHER_10500]:
            return

        if quantity == 0:
            return
        
        if symbol not in orders_dict:
            orders_dict[symbol] = []
        
        orders_dict[symbol].append(Order(symbol, price, quantity))
        logger.print(f"PLACE {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)}x{price}")
    
    def update_ewma(self, current_value, previous_ewma, span):
        """Calculate EWMA (Exponentially Weighted Moving Average)."""
        if previous_ewma is None:
            return current_value
        alpha = 2 / (span + 1)
        return alpha * current_value + (1 - alpha) * previous_ewma
        
    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """Execute the volatility smile trading strategy with dynamic FLIP thresholds."""
        orders_dict = {}
        self.last_timestamp = state.timestamp
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)

        if not rock_price:
            logger.print("No price available for VOLCANIC_ROCK, skipping iteration")
            self.orders = orders_dict
            return orders_dict
        
        # --- OTM Flattening Logic --- 
        is_otm_flat = False
        for voucher in self.voucher_symbols:
            strike = STRIKES[voucher]
            if rock_price - strike <= -250:
                current_position = state.position.get(voucher, 0)
                if current_position != 0:
                    logger.print(f"FLATTENING {voucher} - Too far out of money (Price: {rock_price}, Strike: {strike}) Pos: {current_position}")
                    is_otm_flat = True
                    order_depth = state.order_depths.get(voucher)
                    if current_position > 0: # Need to sell
                        if order_depth and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            self.place_order(orders_dict, voucher, best_bid, -current_position)
                    elif current_position < 0: # Need to buy
                        if order_depth and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            self.place_order(orders_dict, voucher, best_ask, -current_position)
                            
        if is_otm_flat:
             logger.print("Skipping main Volatility Smile logic due to OTM flattening.")
             self.orders = orders_dict
             return orders_dict
        # --- End OTM Flattening Logic --- 
        
        # --- Calculate Z-Score --- 
        zscore = 0 
        calculated_zscore_is_valid = False
        moneyness_values = []
        iv_values = []
        voucher_data = {}
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price: continue
            strike = STRIKES[voucher]
            # Ensure time_to_expiry is valid
            valid_tte = max(1e-9, time_to_expiry) # Use a very small positive number if zero
            moneyness_log_arg = strike / rock_price
            if moneyness_log_arg <= 0 or valid_tte <= 0:
                 logger.print(f"Skipping {voucher} due to invalid args for log/sqrt: strike={strike}, rock_price={rock_price}, tte={valid_tte}")
                 continue 
            moneyness = math.log(moneyness_log_arg) / math.sqrt(valid_tte)
            impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, valid_tte)
            if impl_vol > 0: 
                moneyness_values.append(moneyness)
                iv_values.append(impl_vol)
                voucher_data[voucher] = {'moneyness': moneyness, 'iv': impl_vol}
        
        if len(moneyness_values) >= 3: # Need enough points to fit polynomial
            try:
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                base_iv = c
                self.base_iv_history.append(base_iv)
                self.short_ewma_base_iv = self.update_ewma(base_iv, self.short_ewma_base_iv, self.short_ewma_span)
                self.long_ewma_first = self.update_ewma(base_iv, self.long_ewma_first, self.long_ewma_span)
                self.long_ewma_base_iv = self.update_ewma(self.long_ewma_first, self.long_ewma_base_iv, self.long_ewma_span)
                
                if len(self.base_iv_history) >= self.rolling_window and self.short_ewma_base_iv is not None and self.long_ewma_base_iv is not None:
                    ewma_diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                    if not hasattr(self, 'ewma_diff_history'): self.ewma_diff_history = deque(maxlen=200)
                    self.ewma_diff_history.append(ewma_diff)
                    rolling_std = 0
                    if len(self.ewma_diff_history) >= self.rolling_window:
                        rolling_std = np.std(list(self.ewma_diff_history)[-self.rolling_window:])
                    elif len(self.ewma_diff_history) > 1:
                        rolling_std = np.std(list(self.ewma_diff_history))
                    
                    if rolling_std > 1e-9:
                        zscore = ewma_diff / rolling_std
                        calculated_zscore_is_valid = True
                        self.zscore_history.append(zscore)
                    else:
                         logger.print("Warning: Rolling std dev near zero for Z-score.")
            except Exception as e:
                 logger.print(f"Error calculating Z-score: {e}")

        # --- State & Threshold Adjustment --- 
        if not calculated_zscore_is_valid:
            logger.print("Holding positions due to invalid Z-score calc.")
            self.orders = orders_dict 
            return orders_dict

        logger.print(f"Z-score: {zscore:.4f}, Orig Upper: {self.zscore_upper_threshold:.4f}, Orig Lower: {self.zscore_lower_threshold:.4f}")

        # Increment timer first if already in a non-neutral state
        if self.current_intended_state != Signal.NEUTRAL:
            self.time_in_state += 1

        # Calculate adjusted thresholds based on time in the *current* state
        adjusted_upper_threshold = self.zscore_upper_threshold
        adjusted_lower_threshold = self.zscore_lower_threshold
        min_threshold_gap = 0.1 

        if self.current_intended_state == Signal.LONG:
            adjustment = self.time_in_state * self.threshold_adjustment_rate
            adjusted_upper_threshold = self.zscore_upper_threshold - adjustment
            adjusted_upper_threshold = max(adjusted_upper_threshold, self.zscore_lower_threshold + min_threshold_gap)
            logger.print(f"In LONG state for {self.time_in_state} steps. Adjusted Exit (Upper) Threshold: {adjusted_upper_threshold:.4f}")
        elif self.current_intended_state == Signal.SHORT:
            adjustment = self.time_in_state * self.threshold_adjustment_rate
            adjusted_lower_threshold = self.zscore_lower_threshold + adjustment
            adjusted_lower_threshold = min(adjusted_lower_threshold, self.zscore_upper_threshold - min_threshold_gap)
            logger.print(f"In SHORT state for {self.time_in_state} steps. Adjusted Exit (Lower) Threshold: {adjusted_lower_threshold:.4f}")

        # --- Determine Target State based on Z-score vs ADJUSTED Thresholds --- 
        new_target_state = self.current_intended_state # Assume state doesn't change

        if zscore < adjusted_lower_threshold:
            if self.current_intended_state != Signal.LONG:
                 logger.print(f"STATE FLIP -> LONG: Z-score {zscore:.4f} < adjusted lower {adjusted_lower_threshold:.4f}")
                 new_target_state = Signal.LONG
                 self.time_in_state = 0 # Reset timer on state change
        elif zscore > adjusted_upper_threshold:
            if self.current_intended_state != Signal.SHORT:
                logger.print(f"STATE FLIP -> SHORT: Z-score {zscore:.4f} > adjusted upper {adjusted_upper_threshold:.4f}")
                new_target_state = Signal.SHORT
                self.time_in_state = 0 # Reset timer on state change
        # else: zscore is between adjusted thresholds, state remains the same, timer increments (already done)

        self.current_intended_state = new_target_state # Update the state for this iteration and the next

        # --- Execute Orders Based on Target State --- 
        logger.print(f"Current Target State: {self.current_intended_state}")

        if self.current_intended_state == Signal.LONG:
            # Target: Full Long (+limit)
            for voucher in self.voucher_symbols:
                current_position = state.position.get(voucher, 0)
                position_limit = self.position_limits.get(voucher, 0)
                needed_to_buy = position_limit - current_position
                if needed_to_buy > 0:
                            order_depth = state.order_depths.get(voucher)
                            if order_depth and order_depth.sell_orders:
                                price = min(order_depth.sell_orders.keys())
                                logger.print(f"Targeting LONG {voucher}: Placing BUY {needed_to_buy} at {price}")
                                self.place_order(orders_dict, voucher, price, needed_to_buy)
                            else:
                                 logger.print(f"Want to BUY {voucher}, but no asks.")

        elif self.current_intended_state == Signal.SHORT:
            # Target: Full Short (-limit)
            for voucher in self.voucher_symbols:
                current_position = state.position.get(voucher, 0)
                position_limit = self.position_limits.get(voucher, 0)
                needed_to_sell = position_limit + current_position
                if needed_to_sell > 0:
                    order_depth = state.order_depths.get(voucher)
                    if order_depth and order_depth.buy_orders:
                        price = max(order_depth.buy_orders.keys())
                        logger.print(f"Targeting SHORT {voucher}: Placing SELL {needed_to_sell} at {price}")
                        self.place_order(orders_dict, voucher, price, -needed_to_sell)
                    else:
                        logger.print(f"Want to SELL {voucher}, but no bids.")

        # else: State is NEUTRAL (only at start), do nothing until a threshold is crossed.
        
        self.orders = orders_dict
        return orders_dict
    
    def save_state(self) -> dict:
        """Save the current state of the trader."""
        state_data = {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history) if hasattr(self, 'ewma_diff_history') else [],
            "zscore_history": list(self.zscore_history),
            "zscore_upper_threshold": self.zscore_upper_threshold,
            "zscore_lower_threshold": self.zscore_lower_threshold,
            # Add new state
            "threshold_adjustment_rate": self.threshold_adjustment_rate,
            "current_intended_state": self.current_intended_state.value,
            "time_in_state": self.time_in_state,
            "last_timestamp": self.last_timestamp
        }
        return {k: v for k, v in state_data.items() if v is not None}

    
    def load_state(self, data: dict) -> None:
        """Load the state of the trader."""
        if not data:
            return
            
        self.day = data.get("day", self.day)
        self.base_iv_history = deque(data.get("base_iv_history", []), maxlen=200)
        self.short_ewma_base_iv = data.get("short_ewma_base_iv")
        self.long_ewma_first = data.get("long_ewma_first")
        self.long_ewma_base_iv = data.get("long_ewma_base_iv")
        self.ewma_diff_history = deque(data.get("ewma_diff_history", []), maxlen=200)
        self.zscore_history = deque(data.get("zscore_history", []), maxlen=100)

        # Load original thresholds (these might be optimized)
        self.zscore_upper_threshold = data.get("zscore_upper_threshold", self.zscore_upper_threshold)
        self.zscore_lower_threshold = data.get("zscore_lower_threshold", self.zscore_lower_threshold)

        # Load dynamic state variables
        self.threshold_adjustment_rate = data.get("threshold_adjustment_rate", 0.005) # Default if not found
        loaded_state_value = data.get("current_intended_state")
        try:
            self.current_intended_state = Signal(loaded_state_value) if loaded_state_value is not None else Signal.NEUTRAL
        except ValueError:
            self.current_intended_state = Signal.NEUTRAL
        self.time_in_state = data.get("time_in_state", 0)
        
        self.last_timestamp = data.get("last_timestamp")
        logger.print(f"Loaded state: IntendedState={self.current_intended_state}, TimeInState={self.time_in_state}")


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
            pass # Use pass if you need an empty block structurally

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
            pass

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


# --- Signal Strategy Base Class --- (NEW, adapted from gameplan)
class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.signal = Signal.NEUTRAL # Default state
        self.thresholds = self._get_thresholds() # Load thresholds specific to symbol
        logger.print(f"Initialized SignalStrategy for {symbol} with thresholds: {self.thresholds}")

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        # Implementation specific to derived class (e.g., PicnicBasketStrategy)
        raise NotImplementedError()

    def _get_thresholds(self) -> Tuple[Optional[float], Optional[float]]:
        # Thresholds defined in gameplan
        threshold_map = {
            Product.JAMS: (195, 485),
            Product.DJEMBES: (325, 370),
            Product.PICNIC_BASKET1: (290, 355),
            Product.PICNIC_BASKET2: (290, 355),
            # Add other symbols here if this strategy is reused
        }
        return threshold_map.get(self.symbol, (None, None)) # Return None if symbol not found

    def act(self, state: TradingState) -> None:
        # Update signal based on derived class logic
        new_signal = self.get_signal(state)
        if new_signal is not None and new_signal != self.signal:
            logger.print(f"{self.symbol}: Signal changed from {self.signal} to {new_signal}")
            self.signal = new_signal
        elif new_signal is None:
             logger.print(f"{self.symbol}: get_signal returned None, maintaining signal {self.signal}")

        # Execute based on the current signal
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol)

        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders):
            logger.print(f"{self.symbol}: No order depth or orders available. Cannot trade.")
            return

        if self.signal == Signal.NEUTRAL:
            if position != 0:
                logger.print(f"{self.symbol}: Signal NEUTRAL, liquidating position {position}")
                if position < 0:
                    buy_price = self.get_buy_price(order_depth)
                    if buy_price is not None:
                         self._place_buy_order(buy_price, -position)
                elif position > 0:
                    sell_price = self.get_sell_price(order_depth)
                    if sell_price is not None:
                        self._place_sell_order(sell_price, position)
            # else: logger.print(f"{self.symbol}: Signal NEUTRAL, already flat.")

        elif self.signal == Signal.SHORT:
            needed_to_sell = self.position_limit + position
            if needed_to_sell > 0:
                sell_price = self.get_sell_price(order_depth)
                if sell_price is not None:
                    logger.print(f"{self.symbol}: Signal SHORT, current pos {position}, selling {needed_to_sell} to reach {-self.position_limit}")
                    self._place_sell_order(sell_price, needed_to_sell)
                # else: logger.print(f"{self.symbol}: Signal SHORT, but no buy orders to hit.")
            # else: logger.print(f"{self.symbol}: Signal SHORT, already at or below limit {position}")

        elif self.signal == Signal.LONG:
            needed_to_buy = self.position_limit - position
            if needed_to_buy > 0:
                buy_price = self.get_buy_price(order_depth)
                if buy_price is not None:
                    logger.print(f"{self.symbol}: Signal LONG, current pos {position}, buying {needed_to_buy} to reach {self.position_limit}")
                    self._place_buy_order(buy_price, needed_to_buy)
                # else: logger.print(f"{self.symbol}: Signal LONG, but no sell orders to hit.")
            # else: logger.print(f"{self.symbol}: Signal LONG, already at or above limit {position}")

    def get_buy_price(self, order_depth: OrderDepth) -> Optional[int]:
        # Hit best ask if available
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    def get_sell_price(self, order_depth: OrderDepth) -> Optional[int]:
        # Hit best bid if available
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    def save(self) -> dict:
        # Save the current signal state
        return {"signal": self.signal.value}

    def load(self, data: dict) -> None:
        # Load the signal state
        loaded_signal_value = data.get("signal")
        if loaded_signal_value is not None:
            try:
                self.signal = Signal(loaded_signal_value)
            except ValueError:
                 logger.print(f"Warning: Invalid signal value {loaded_signal_value} found in trader data for {self.symbol}. Resetting to NEUTRAL.")
                 self.signal = Signal.NEUTRAL
        else:
            self.signal = Signal.NEUTRAL # Default if not found
        logger.print(f"Loaded state for {self.symbol}: Signal={self.signal}")


# --- Picnic Basket Strategy --- (NEW, adapted from gameplan)

# --- Trader Class --- (MODIFIED)
class Trader:
    def __init__(self):
        # --- Position Limits --- (UPDATED)
        self.position_limits = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250, # Added
            Product.JAMS: 350, # Added
            Product.DJEMBES: 60, # Added
            Product.PICNIC_BASKET1: 60, # Added
            Product.PICNIC_BASKET2: 100, # Added
            Product.VOLCANIC_ROCK: 400,
            Product.VOUCHER_9500: 200, 
            Product.VOUCHER_9750: 200, 
            Product.VOUCHER_10000: 200, 
            Product.VOUCHER_10250: 200, 
            Product.VOUCHER_10500: 200
        }

        # --- Strategy Definitions --- (UPDATED)
        self.strategies = {
            # Existing Strategies
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy(Product.RAINFOREST_RESIN, self.position_limits[Product.RAINFOREST_RESIN]),
            Product.KELP: PrototypeKelpStrategy(Product.KELP, self.position_limits[Product.KELP]),
            Product.SQUID_INK: RsiStrategy(Product.SQUID_INK, self.position_limits[Product.SQUID_INK]),
            Product.VOLCANIC_ROCK: RsiStrategy(Product.VOLCANIC_ROCK, self.position_limits[Product.VOLCANIC_ROCK]),
            "VOLATILITY_SMILE": VolatilitySmileStrategy(), # Handles all vouchers

            # New Strategies based on Gameplan
            Product.CROISSANTS: CroissantCaesarOliviaStrategy(Product.CROISSANTS, self.position_limits[Product.CROISSANTS]),
        }

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
         # --- Existing Run Logic --- (No changes needed here)
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
             strategy_state = loaded_data.get(str(strategy_key), {}) # Use str(key) just in case
             if isinstance(strategy_state, dict):
                 try: strategy.load(strategy_state)
                 except Exception as e: logger.print(f"Error loading state for {strategy_key}: {e}")
             else: strategy.load({}) # Load empty state if not dict

             # Check if market data is available for the strategy
             market_data_available = True
             required_products = []
             if isinstance(strategy, VolatilitySmileStrategy):
                 # Check for Volcanic Rock and at least one voucher
                 required_products = [Product.VOLCANIC_ROCK]
                 voucher_data_available = any(voucher in state.order_depths for voucher in 
                     [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500])
                 if not voucher_data_available:
                     market_data_available = False
             elif str(strategy_key) in self.position_limits: # Assume other keys are single products if in limits
                 required_products = [str(strategy_key)]
             else:
                  
                  logger.print(f"Warning: Unsure how to check market data availability for strategy key: {strategy_key}")
                  # market_data_available = False # Be conservative?

             if any(prod not in state.order_depths for prod in required_products):
                 if required_products: # Only log if we actually knew which products were needed
                     logger.print(f"Strategy {strategy_key}: Market data missing for required products ({[p for p in required_products if p not in state.order_depths]}). Skipping run.")
                 market_data_available = False

             if market_data_available:
                 try:
                     if isinstance(strategy, VolatilitySmileStrategy):
                         # For VolatilitySmileStrategy, we call run with a different interface
                         orders_from_strategy = strategy.run(state)
                         for symbol, orders in orders_from_strategy.items():
                             all_orders.extend(orders)
                     else:
                         # For other strategies, use the original approach
                         strategy.run(state) # This calls the strategy's act method
                         all_orders.extend(strategy.orders) # Add this strategy's orders to the main list
                 except Exception as e:
                     logger.print(f"*** ERROR running {strategy_key} strategy: {e} ***");
                     import traceback; logger.print(traceback.format_exc())

             # Save state for this strategy
             try: 
                 if isinstance(strategy, VolatilitySmileStrategy):
                     trader_data_for_next_round[str(strategy_key)] = strategy.save_state()
                 else:
                     trader_data_for_next_round[str(strategy_key)] = strategy.save()
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