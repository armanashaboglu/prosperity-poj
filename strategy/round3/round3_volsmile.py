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
# Get fitted IV for each voucher
# Residual IV = Actual IV - Fitted IV

# base IV = fitted IV at m=0 → equal  c

# 3 — resid z-Score trading signal

#hitorical mean of resids
#rolling_std  = Residual_IV.rolling(N).std()
#zscore = (Residual_IV - rolling_mean) / rolling_std
# trading logic:
#if zscore > upper_threshold:
#    SELL Option (IV is too high)
#if zscore < lower_threshold:
#    BUY Option (IV is too low)

# 4 — base IV mean reversion signal(?)
# store base IV time series
#base_iv_mean = smoothed base IV(lowess)
#base_iv_std  = Base_IV.rolling(N).std()
#zscore_base = (Base_IV - base_iv_mean) / base_iv_std
# trading logic:
#if zscore_base > upper_threshold:
#SELL all options (?)

#if zscore_base < lower_threshold:
#buy all options (?)

# NOISE REDUCTION needed for base iv trading. maybe even for residual iv trading.
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
        
        # Strategy parameters
        self.window_size = 500  # optimize later
        self.residual_upper_threshold = 2.5  
        self.residual_lower_threshold = -2.5
        self.close_position_threshold = 0.15  # threshold for closing positions
        self.order_size_factor = 0.2  # factor to determine order size
        
        self.voucher_symbols = [
            Product.VOUCHER_9500, 
            Product.VOUCHER_9750, 
            Product.VOUCHER_10000, 
            Product.VOUCHER_10250, 
            Product.VOUCHER_10500
        ]
        
        
        self.residual_ivs = {symbol: deque(maxlen=self.window_size) for symbol in self.voucher_symbols}
        
        # store residuals for position tracking
        self.ewma_residuals = deque(maxlen=self.window_size)
        
        
        self.day = 2  # CHANGE TO 3 WHEN SUBMITTING!
        self.last_timestamp = None
    
    def update_time_to_expiry(self, timestamp):
        """Calculate time to expiry based on the current timestamp."""
        
        base_tte = 8 - self.day
        
        
        iteration = (timestamp % 1000000) // 100
        
        # linear decay
        iteration_adjustment = iteration / 10000
        
        # 
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
    
    def calculate_order_size(self, symbol: Symbol, z_score: float, state: TradingState) -> int:
        """Calculate order size based on z-score magnitude and position limits."""
        
        current_position = state.position.get(symbol, 0)
        position_limit = self.position_limits.get(symbol, 0)
        
        # Calculate available capacity
        if z_score > 0:  
            available_capacity = position_limit + current_position
        else:  
            available_capacity = position_limit - current_position
        
        # base size on z-score magnitude and position limit
        threshold = abs(self.residual_upper_threshold)
        size_pct = min(1.0, abs(z_score) / threshold) * self.order_size_factor
        
        size = int(size_pct * position_limit)
        
        
        size = min(size, available_capacity)
        
        #
        return size if z_score < 0 else -size
    
    def calculate_close_position_size(self, symbol: Symbol, state: TradingState) -> int:
        """Calculate size needed to close current position."""
        current_position = state.position.get(symbol, 0)
        if current_position == 0:
            return 0
            
        return -current_position
    
    def place_order(self, orders_dict, symbol, price, quantity):
        """Add an order to the orders dictionary."""
        if quantity == 0:
            return
        
        if symbol not in orders_dict:
            orders_dict[symbol] = []
        
        orders_dict[symbol].append(Order(symbol, price, quantity))
        logger.print(f"PLACE {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)}x{price}")
    
    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """Execute the volatility smile trading strategy."""
        orders_dict = {}
        
        
        self.last_timestamp = state.timestamp
        
        
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        
        
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)
        if not rock_price:
            logger.print("No price available for VOLCANIC_ROCK, skipping iteration")
            return orders_dict
        
        # calculate moneyness and implied volatility for each voucher
        moneyness_values = []
        iv_values = []
        voucher_data = {}
        
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price:
                continue
                
            strike = STRIKES[voucher]
            
            # moneyness
            moneyness = math.log(strike / rock_price) / math.sqrt(time_to_expiry)
            
            # implied volatility
            impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, time_to_expiry)
            
            if impl_vol > 0: 
                moneyness_values.append(moneyness)
                iv_values.append(impl_vol)
                voucher_data[voucher] = {'moneyness': moneyness, 'iv': impl_vol}
        
        #  Smile Fitting 
        if len(moneyness_values) >= 3:
            try:
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                
                # Calculate residual IV for each voucher
                for voucher, data in voucher_data.items():
                    moneyness = data['moneyness']
                    actual_iv = data['iv']
                    fitted_iv = a * moneyness**2 + b * moneyness + c
                    residual_iv = actual_iv - fitted_iv
                    self.residual_ivs[voucher].append(residual_iv)
                
                # Trading logic
                for voucher in self.voucher_symbols:
                    if voucher in voucher_data and len(self.residual_ivs[voucher]) >= 10:
                        residuals = list(self.residual_ivs[voucher])
                        current_position = state.position.get(voucher, 0)
                        
                        historical_mean = HISTORICAL_MEAN_RESIDUALS.get(voucher, 0.0)
                        std_residual = np.std(residuals) if np.std(residuals) > 0 else 1.0
                        
                        current_residual = self.residual_ivs[voucher][-1]
                        zscore = (current_residual - historical_mean) / std_residual
                        
                        logger.print(f"{voucher} - Current Residual: {current_residual:.6f}, Historical Mean: {historical_mean:.6f}, Z-Score: {zscore:.2f}, Position: {current_position}")
                        
                        # Close positions
                        #if current_position > 0 and abs(zscore) <= self.close_position_threshold:
                            #logger.print(f"CLOSING LONG POSITION FOR {voucher} - Z-Score: {zscore:.2f} in range [-{self.close_position_threshold}, {self.close_position_threshold}]")
                            #voucher_price = self.get_mid_price(voucher, state)
                            #if voucher_price:
                                #price = int(round(voucher_price - 1))
                                #order_size = self.calculate_close_position_size(voucher, state)
                                #if order_size != 0:
                                    #self.place_order(orders_dict, voucher, price, order_size)
                        
                        #elif current_position < 0 and abs(zscore) <= self.close_position_threshold:
                        #    logger.print(f"CLOSING SHORT POSITION FOR {voucher} - Z-Score: {zscore:.2f} in range [-{self.close_position_threshold}, {self.close_position_threshold}]")
                        #    voucher_price = self.get_mid_price(voucher, state)
                        #    if voucher_price:
                                #price = int(round(voucher_price + 1))
                                #order_size = self.calculate_close_position_size(voucher, state)
                                #if order_size != 0:
                                    #self.place_order(orders_dict, voucher, price, order_size)
                        
                        
                        
                            
                        if zscore > abs(self.residual_upper_threshold):
                                logger.print(f"INDIVIDUAL SELL SIGNAL FOR {voucher} - Z-Score: {zscore:.2f} > {abs(self.residual_upper_threshold)}")
                                voucher_price = self.get_mid_price(voucher, state)
                                if voucher_price:
                                    price = int(round(voucher_price - 1))
                                    order_size = self.calculate_order_size(voucher, zscore, state)
                                    if order_size != 0:
                                        self.place_order(orders_dict, voucher, price, order_size)
                            
                        elif zscore < self.residual_lower_threshold:
                                logger.print(f"INDIVIDUAL BUY SIGNAL FOR {voucher} - Z-Score: {zscore:.2f} < {self.residual_lower_threshold}")
                                voucher_price = self.get_mid_price(voucher, state)
                                if voucher_price:
                                    price = int(round(voucher_price + 1))
                                    order_size = self.calculate_order_size(voucher, zscore, state)
                                    if order_size != 0:
                                        self.place_order(orders_dict, voucher, price, order_size)
            
            except Exception as e:
                logger.print(f"Error: {e}")
                import traceback
                logger.print(traceback.format_exc())
        
        return orders_dict
    
    def save_state(self) -> dict:
        """Save the current state of the trader."""
        return {
            "day": self.day,
            "residual_ivs": {symbol: list(values) for symbol, values in self.residual_ivs.items()},
            "ewma_residuals": list(self.ewma_residuals),
            "last_timestamp": self.last_timestamp
        }
    
    def load_state(self, state_data: dict) -> None:
        """Load the state of the trader."""
        if not state_data:
            return
            
        self.day = state_data.get("day", self.day)
        
        # Load residual IVs for individual vouchers
        residual_ivs = state_data.get("residual_ivs", {})
        for symbol, values in residual_ivs.items():
            if symbol in self.residual_ivs:
                self.residual_ivs[symbol] = deque(values, maxlen=self.window_size)
        
        # Load other residuals
        self.ewma_residuals = deque(state_data.get("ewma_residuals", []), maxlen=self.window_size)
        
        # Load timestamp
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