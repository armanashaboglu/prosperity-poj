#STRATEGY FRAMEWORK


# TimeToExpiry schedule is given as input to us. 
# Day 0.csv initial timestamp starts with 8 DTE which decays linearly throughout the timestamps 
# and comes down to 7 DTE at day 1.csv initial timestamp and so on. Each day is 1 000 000M timestamps, with each timestamp 
# increasing by 100 increments. So a total day is 10 000 iterations.
# For calculaating TTE, introduce a hard coded variable called DAY, that indicated which day this algorithm will be ran on.
# TTE will decay linearly from 8 DTE to 7 DTE at day 1.csv initial timestamp and so on. But also in each iteration, 
# TTE will decay by 1/10000 a day.


#1 — at every timestamp
# # For every voucher:
# moneyness = np.log(strike / rock_price) / np.sqrt(TTE)
# IV = ImpliedVol(option_price, rock_price, strike, TTE) 

#2 — smile fit per timestamp
# Fit quadratic: IV = a * m^2 + b * m + c
# base IV = fitted IV at m=0 → equal  c

#base IV mean reversion signal
# store base IV time series
#base_iv_mean = smoothed base IV(double ewma long term)
#base_iv_std  = Base_IV.rolling(N).std()
#zscore_base = (Base_IV (short term ewma) - base_iv_mean) / base_iv_std
# We use one short term ewma, and one long term double ewma to filter out noise.
#short_ewma_span = 20  # for now
#long_ewma_span = 100 # for now
#rolling_window = 50 # for now
# trading logic:
#if zscore_base > upper_threshold:
#SELL all options trade size = 1

#if zscore_base < lower_threshold:
#buy all options trade size = 1


#intuition behind the strategy:
    # Base IV is the implied volatility of a option when perfectly at the money. Strike = Spot price.
    # Therefore, the implied volatility of this option is the most "pure" measure of expected volatility
    # According to our analysis, the Base IV is mean reverting.
    # Therefore, if we can SELL the base IV when above a threshold, and BUY the base IV when below a threshold,
    # However, base IV data is noisy, so we use a double ewma to smooth the data.
    # We use one short term ewma, and one long term double ewma to filter out noise.
    # We use a zscore to determine when to SELL or BUY the base IV.

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
from collections import deque, defaultdict
import math
from datamodel import Listing, OrderDepth, TradingState, Order, Symbol, Trade, ProsperityEncoder

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

    def compress_observations(self, observations: Any) -> list[Any]:
        if not observations:
            return [[], {}]
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

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

STRIKES = {
    Product.VOUCHER_9500: 9500,
    Product.VOUCHER_9750: 9750,
    Product.VOUCHER_10000: 10000,
    Product.VOUCHER_10250: 10250,
    Product.VOUCHER_10500: 10500,
}

POSITION_LIMITS = {
    Product.VOLCANIC_ROCK: 400,
    Product.VOUCHER_9500: 200,
    Product.VOUCHER_9750: 200,
    Product.VOUCHER_10000: 200,
    Product.VOUCHER_10250: 200,
    Product.VOUCHER_10500: 200,
}

HISTORICAL_MEAN_RESIDUALS = {
    Product.VOUCHER_9500: -0.000372,
    Product.VOUCHER_9750: 0.000809,
    Product.VOUCHER_10000: -0.000169,
    Product.VOUCHER_10250: -0.000618,
    Product.VOUCHER_10500: 0.000346,
}

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

class VolatilitySmileTrader:
    def __init__(self) -> None:
        """Initialize the Volatility Smile trader for all vouchers"""
        self.position_limits = POSITION_LIMITS
        
        # Voucher symbols
        self.voucher_symbols = [
            Product.VOUCHER_9500, 
            Product.VOUCHER_9750, 
            Product.VOUCHER_10000, 
            Product.VOUCHER_10250, 
            Product.VOUCHER_10500
        ]
        
        # Strategy parameters for base IV mean reversion
        self.short_ewma_span = 37  # First level EWMA span for Base IV
        self.long_ewma_span = 68  # Second level EWMA span for double EWMA
        self.rolling_window = 48   # Window for rolling standard deviation
        
        # Z-score thresholds for trading signals
        self.zscore_upper_threshold = 0.5  # Z-score threshold for sell signals
        self.zscore_lower_threshold = -2.8  # Z-score threshold for buy signals
        
        
        self.trade_size = 22
        
        
        self.base_iv_history = deque(maxlen=200)
        
       
        self.short_ewma_base_iv = None
        self.long_ewma_first = None  
        self.long_ewma_base_iv = None  
        
        
        self.ewma_diff_history = deque(maxlen=200)
        
        
        self.zscore_history = deque(maxlen=100)
        
        
        self.day = 2  # set to 3 when sub 
        self.last_timestamp = None
    
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
    
    def calculate_order_size(self, symbol: Symbol, zscore: float, state: TradingState) -> int:
        """Calculate order size based on z-score sign and current position."""
        current_position = state.position.get(symbol, 0)
        position_limit = self.position_limits.get(symbol, 0)
        
       
        fixed_size = self.trade_size
        
        
        if zscore > 0:  
           
            if current_position - fixed_size >= -position_limit:
                return -fixed_size  
            else:
                return 0  
        else: 
            if current_position + fixed_size <= position_limit:
                return fixed_size 
            else:
                return 0  
    
    def place_order(self, orders_dict, symbol, price, quantity):
        """Add an order to the orders dictionary."""
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
        """Execute the volatility smile trading strategy using base IV mean reversion signals."""
        orders_dict = {}
        
        self.last_timestamp = state.timestamp
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)
        if not rock_price:
            logger.print("No price available for VOLCANIC_ROCK, skipping iteration")
            return orders_dict
        
        # Add risk management check - if deep out of the money, go flat
        for voucher in self.voucher_symbols:
            strike = STRIKES[voucher]
            # Check if underlying - strike is <= -250 (deep out of the money)
            if rock_price - strike <= -250:
                current_position = state.position.get(voucher, 0)
                if current_position != 0:
                    voucher_price = self.get_mid_price(voucher, state)
                    if voucher_price and current_position > 0:
                        # Have long position, need to sell to go flat
                        order_depth = state.order_depths.get(voucher)
                        if order_depth and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            # Place sell order at best bid to flatten position
                            self.place_order(orders_dict, voucher, best_bid, -current_position)
                            logger.print(f"RISK MGMT - FLATTEN: {voucher} SELL {current_position}x{best_bid} (rock:{rock_price}, strike:{strike})")
                    elif voucher_price and current_position < 0:
                        # Have short position, need to buy to go flat
                        order_depth = state.order_depths.get(voucher)
                        if order_depth and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            # Place buy order at best ask to flatten position
                            self.place_order(orders_dict, voucher, best_ask, -current_position)
                            logger.print(f"RISK MGMT - FLATTEN: {voucher} BUY {-current_position}x{best_ask} (rock:{rock_price}, strike:{strike})")
        
        moneyness_values = []
        iv_values = []
        voucher_data = {}
        
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price:
                continue
                
            strike = STRIKES[voucher]
            
           
            moneyness = math.log(strike / rock_price) / math.sqrt(time_to_expiry)
            
            
            impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, time_to_expiry)
            
            if impl_vol > 0: 
                moneyness_values.append(moneyness)
                iv_values.append(impl_vol)
                voucher_data[voucher] = {'moneyness': moneyness, 'iv': impl_vol}
        
        
        if len(moneyness_values) >= 3:
            try:
                
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                
               
                base_iv = c
                logger.print(f"Base IV (ATM): {base_iv:.6f}")
                
                
                self.base_iv_history.append(base_iv)
                
                self.short_ewma_base_iv = self.update_ewma(
                    base_iv, 
                    self.short_ewma_base_iv, 
                    self.short_ewma_span
                )
                
                # Update
                self.long_ewma_first = self.update_ewma(
                    base_iv,
                    self.long_ewma_first,
                    self.long_ewma_span
                )
                
                # 5. Update second l
                self.long_ewma_base_iv = self.update_ewma(
                    self.long_ewma_first,
                    self.long_ewma_base_iv,
                    self.long_ewma_span
                )
                
                
                if len(self.base_iv_history) >= self.rolling_window and self.short_ewma_base_iv is not None and self.long_ewma_base_iv is not None:
                    
                    ewma_diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                    
                    
                    if not hasattr(self, 'ewma_diff_history'):
                        self.ewma_diff_history = deque(maxlen=200)
                    self.ewma_diff_history.append(ewma_diff)
                    
                    
                    
                    if len(self.ewma_diff_history) >= self.rolling_window:
                        recent_ewma_diffs = list(self.ewma_diff_history)[-self.rolling_window:]
                        rolling_std = np.std(recent_ewma_diffs)
                    else:
                        rolling_std = np.std(list(self.ewma_diff_history))
                    
                
                    if rolling_std > 0:
                        zscore = ewma_diff / rolling_std
                    else:
                        zscore = 0
                    
                    self.zscore_history.append(zscore)
                    
                    logger.print(f"Base IV: {base_iv:.6f}, Short EWMA: {self.short_ewma_base_iv:.6f}, Long EWMA: {self.long_ewma_base_iv:.6f}")
                    logger.print(f"EWMA Diff: {ewma_diff:.6f}, Rolling StdDev: {rolling_std:.6f}")
                    logger.print(f"Z-score: {zscore:.4f}, Upper Threshold: {self.zscore_upper_threshold}, Lower Threshold: {self.zscore_lower_threshold}")
                    
                    
                    if zscore > self.zscore_upper_threshold:
                        logger.print(f"SELL SIGNAL - Z-score: {zscore:.4f} > {self.zscore_upper_threshold}")
                        
                        for voucher in self.voucher_symbols:
                            # Check if deep out of the money before placing orders
                            strike = STRIKES[voucher]
                            if rock_price - strike <= -250:
                                logger.print(f"SKIPPING {voucher} - Too far out of money: rock:{rock_price}, strike:{strike}")
                                continue
                                
                            order_depth = state.order_depths.get(voucher)
                            # Ensure we have an order depth and buy orders to determine the best bid
                            if order_depth and order_depth.buy_orders:
                                best_bid = max(order_depth.buy_orders.keys())
                                price_to_sell = best_bid # Place sell order at best bid
                            
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size != 0:
                                    self.place_order(orders_dict, voucher, price_to_sell, order_size)
                            else:
                                logger.print(f"No best bid found for {voucher} to place sell order.") # Log if no bid exists
                    
                    elif zscore < self.zscore_lower_threshold:
                        logger.print(f"BUY SIGNAL - Z-score: {zscore:.4f} < {self.zscore_lower_threshold}")
                        
                        for voucher in self.voucher_symbols:
                            # Check if deep out of the money before placing orders
                            strike = STRIKES[voucher]
                            if rock_price - strike <= -250:
                                logger.print(f"SKIPPING {voucher} - Too far out of money: rock:{rock_price}, strike:{strike}")
                                continue
                                
                            order_depth = state.order_depths.get(voucher)
                            # Ensure we have an order depth and sell orders to determine the best ask
                            if order_depth and order_depth.sell_orders:
                                best_ask = min(order_depth.sell_orders.keys())
                                price_to_buy = best_ask # Place buy order at best ask
                                
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size != 0:
                                    self.place_order(orders_dict, voucher, price_to_buy, order_size)
                            else:
                                logger.print(f"No best ask found for {voucher} to place buy order.") # Log if no ask exists
            
            except Exception as e:
                logger.print(f"Error: {e}")
                import traceback
                logger.print(traceback.format_exc())
        
        return orders_dict
    
    def save_state(self) -> dict:
        """Save the current state of the trader."""
        return {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history) if hasattr(self, 'ewma_diff_history') else [],
            "zscore_history": list(self.zscore_history),
            "zscore_upper_threshold": self.zscore_upper_threshold,
            "zscore_lower_threshold": self.zscore_lower_threshold,
            "last_timestamp": self.last_timestamp
        }
    
    def load_state(self, state_data: dict) -> None:
        """Load the state of the trader."""
        if not state_data:
            return
            
        self.day = state_data.get("day", self.day)
        
        
        base_iv_history = state_data.get("base_iv_history", [])
        self.base_iv_history = deque(base_iv_history, maxlen=200)
        
        
        self.short_ewma_base_iv = state_data.get("short_ewma_base_iv")
        self.long_ewma_first = state_data.get("long_ewma_first")
        self.long_ewma_base_iv = state_data.get("long_ewma_base_iv")
        
       
        ewma_diff_history = state_data.get("ewma_diff_history", [])
        self.ewma_diff_history = deque(ewma_diff_history, maxlen=200)
        
      
        zscore_history = state_data.get("zscore_history", [])
        self.zscore_history = deque(zscore_history, maxlen=100)
        
        
        self.zscore_upper_threshold = state_data.get("zscore_upper_threshold", self.zscore_upper_threshold)
        self.zscore_lower_threshold = state_data.get("zscore_lower_threshold", self.zscore_lower_threshold)
        
        
        self.last_timestamp = state_data.get("last_timestamp")

class Trader:
    def __init__(self):
        self.voucher_trader = VolatilitySmileTrader()
    
    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        """Main trader entry point."""
        # Load previous state
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                self.voucher_trader.load_state(trader_data.get("voucher_trader", {}))
            except Exception as e:
                logger.print(f"Error loading state: {e}")
        
        # Run the voucher trading strategy
        orders = self.voucher_trader.run(state)
        
        # Save state for next round
        new_trader_data = {
            "voucher_trader": self.voucher_trader.save_state()
        }
        
        # No conversions in this strategy
        conversions = 0
        
        # Encode state data
        try:
            traderData = json.dumps(new_trader_data)
        except Exception as e:
            logger.print(f"Error encoding state: {e}")
            traderData = "{}"
        
        # Use the imported logger instance
        logger.flush(state, orders, conversions, traderData)
        return orders, conversions, traderData