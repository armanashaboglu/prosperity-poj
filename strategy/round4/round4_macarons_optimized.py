import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque, Any
from collections import deque
import math

from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Trade, OrderDepth


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
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice if hasattr(observation, 'sugarPrice') else None,
                observation.sunlightIndex if hasattr(observation, 'sunlightIndex') else None,
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


class MacaronsStrategy:
    def __init__(self) -> None:
        self.symbol = "MAGNIFICENT_MACARONS"
        self.position_limit = 75
        self.conversion_limit = 10  # Maximum conversion per timestamp
        
        # Increased history size to ensure enough data for RSI
        self.price_history = deque(maxlen=200)
        self.mid_price_history = deque(maxlen=200)
        self.sunlight_history = deque(maxlen=5) # Keep short for slope calculation
        self.orders = []
        
        self.fair_value_lower = 550
        self.fair_value_upper = 700
        self.slope_lower_threshold = -0.011
        self.slope_upper_threshold = 0.011
        
        self.rsi_period = 149
        # Corrected RSI levels: oversold < overbought
        self.rsi_oversold = 39
        self.rsi_overbought = 64
        self.sunlight_value_threshold = 45
        
        # Store latest sunlight index value
        self.current_sunlight_index = None
        
        # Trade size percentage for RSI strategy
        self.trade_size_pct = 0.7
        
        # NEW: Target position tracking
        self.target_position = 0  # Target position we want to reach
        self.rsi_target_units = 20  # Units we target to buy/sell in RSI strategy
        self.active_signal = None  # Tracks current active signal (e.g., "SLOPE_LONG", "RSI_SHORT", etc.)
        
    def get_position_adjusted_volumes(self, state: TradingState) -> Tuple[int, int]:
        position = state.position.get(self.symbol, 0)
        available_to_buy = self.position_limit - position
        available_to_sell = self.position_limit + position
        return available_to_buy, available_to_sell
        
    def update_state(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
            
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        self.mid_price_history.append(mid_price)
        self.price_history.append(mid_price)
        
        sunlight_found = False
        if hasattr(state, 'observations') and state.observations is not None:
            if hasattr(state.observations, 'plainValueObservations') and state.observations.plainValueObservations is not None:
                for possible_key in ['sunlightIndex', 'SUNLIGHT_INDEX', 'sunlight_index', 'SunlightIndex']:
                    if possible_key in state.observations.plainValueObservations:
                        self.current_sunlight_index = state.observations.plainValueObservations[possible_key]
                        self.sunlight_history.append(self.current_sunlight_index)
                        sunlight_found = True
                        
                        break
                if not sunlight_found:
                    if hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations:
                        for product, observation in state.observations.conversionObservations.items():
                            if hasattr(observation, 'sunlightIndex') and observation.sunlightIndex is not None:
                                self.current_sunlight_index = observation.sunlightIndex
                                self.sunlight_history.append(self.current_sunlight_index)
                                sunlight_found = True
                                
                                break
                    if not sunlight_found:
                         logger.print(f"T={state.timestamp}: No sunlight index found in any observations")
            else:
                logger.print(f"T={state.timestamp}: plainValueObservations not available")
        else:
            logger.print(f"T={state.timestamp}: No observations available")
            
        if not sunlight_found and len(self.sunlight_history) == 0:
            logger.print(f"T={state.timestamp}: Initializing sunlight history with dummy value")
            dummy_value = 0.5 # Use a neutral dummy value
            self.sunlight_history.append(dummy_value)
            self.current_sunlight_index = dummy_value # Also set current value to dummy initially
    
    def calculate_sunlight_slope(self) -> Optional[float]:
        if len(self.sunlight_history) < 2:
            logger.print(f"Not enough sunlight data points: {len(self.sunlight_history)}/2 required")
            return None
            
        try:
            current = self.sunlight_history[-1]
            previous = self.sunlight_history[-2]
            
            # Avoid calculating slope based on dummy values if possible
            if abs(current - 0.5) < 0.001 and abs(previous - 0.5) < 0.001 and len(self.sunlight_history) < 3:
                 logger.print(f"Using initial dummy values, returning slope 0.0")
                 return 0.0
                 
            slope = current - previous
            logger.print(f"Calculated slope: {slope:.4f} (current={current:.4f}, previous={previous:.4f})")
            return slope
        except Exception as e:
            logger.print(f"Error calculating slope: {str(e)}")
            return 0.0
    
    def calculate_rsi(self) -> Optional[float]:
        
        
        if len(self.price_history) < self.rsi_period + 1:
            logger.print(f"Not enough price data for RSI: {len(self.price_history)}/{self.rsi_period + 1} required")
            return None
            
        prices = list(self.price_history)[-self.rsi_period-1:]
        deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        
        avg_gain = sum(gains[:self.rsi_period]) / self.rsi_period
        avg_loss = sum(losses[:self.rsi_period]) / self.rsi_period

        
        if avg_loss == 0:
            return 100 # RSI is 100 if there are no losses
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        logger.print(f"Calculated RSI: {rsi:.2f} (Avg Gain: {avg_gain:.2f}, Avg Loss: {avg_loss:.2f})")
        return rsi
    
    def act(self, state: TradingState) -> Tuple[List[Order], int]:
        self.orders = []
        conversion_request = 0
        
        if self.symbol not in state.order_depths:
            logger.print(f"T={state.timestamp}: No order depth for {self.symbol}")
            return self.orders, conversion_request
            
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.print(f"T={state.timestamp}: Incomplete order book for {self.symbol}")
            return self.orders, conversion_request
            
        self.update_state(state)
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        
        current_position = state.position.get(self.symbol, 0)
        available_to_buy, available_to_sell = self.get_position_adjusted_volumes(state)
        
        
        # Check if we need to continue executing a previous signal
        if self.active_signal and current_position != self.target_position:
            
            
            if self.target_position > current_position:
                # We need to buy more
                units_to_buy = min(available_to_buy, abs(self.target_position - current_position))
                
                # First try to buy via trading
                bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                
                # Determine trading volume (limited by market liquidity)
                trade_units = min(units_to_buy, ask_volume)
                if trade_units > 0:
                    self._place_buy_order(best_ask, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ Continuing BUY execution: {trade_units} @ {best_ask}")
                    units_to_buy -= trade_units
                
                # Then use conversions if needed and possible (max 10 per timestamp)
                if units_to_buy > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_buy, self.conversion_limit)
                    conversion_request = conv_units
                    logger.print(f"T={state.timestamp}: ✅ Requesting IMPORT conversion: {conv_units} units")
                
            else:
                # We need to sell more
                units_to_sell = min(available_to_sell, abs(current_position - self.target_position))
                
                # First try to sell via trading
                bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                
                # Determine trading volume (limited by market liquidity)
                trade_units = min(units_to_sell, bid_volume)
                if trade_units > 0:
                    self._place_sell_order(best_bid, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ Continuing SELL execution: {trade_units} @ {best_bid}")
                    units_to_sell -= trade_units
                
                # Then use conversions if needed and possible (max 10 per timestamp)
                if units_to_sell > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_sell, self.conversion_limit)
                    conversion_request = -conv_units
                    logger.print(f"T={state.timestamp}: ✅ Requesting EXPORT conversion: {conv_units} units")
            
            return self.orders, conversion_request
        
        # Calculate slope and get current sunlight value
        slope = self.calculate_sunlight_slope()
        sunlight_value = self.current_sunlight_index
        
        # New signal detection
        signal_detected = False
        
        if slope is not None:
            logger.print(f"T={state.timestamp}: Sunlight slope: {slope:.4f}, Sunlight Value: {sunlight_value}")

            # 1. Strong Negative Slope
            if slope < self.slope_lower_threshold:
                logger.print(f"T={state.timestamp}: CONDITION 1 MET: Strong negative slope ({slope:.4f} < {self.slope_lower_threshold}) - Going fully long")
                self.target_position = self.position_limit
                self.active_signal = "SLOPE_LONG"
                signal_detected = True

            # 2. Strong Positive Slope & Above Fair Value
            elif slope > self.slope_upper_threshold and current_mid > self.fair_value_upper:
                logger.print(f"T={state.timestamp}: CONDITION 2 MET: Strong positive slope ({slope:.4f} > {self.slope_upper_threshold}) and Price above fair value ({current_mid:.2f} > {self.fair_value_upper}) - Going fully short")
                self.target_position = -self.position_limit
                self.active_signal = "SLOPE_SHORT"
                signal_detected = True

            # 3. Low Sunlight Index & Negative Slope
            elif sunlight_value is not None and sunlight_value <= self.sunlight_value_threshold and slope < 0:
                logger.print(f"T={state.timestamp}: CONDITION 3 MET: Low sunlight value ({sunlight_value} <= {self.sunlight_value_threshold}) and negative slope ({slope:.4f} < 0) - Going fully long")
                self.target_position = self.position_limit
                self.active_signal = "SUNLIGHT_LONG"
                signal_detected = True
                
            # 4. High Sunlight Index & Positive Slope & Above Fair Value
            elif sunlight_value is not None and sunlight_value >= self.sunlight_value_threshold and slope > 0 and current_mid > self.fair_value_upper:
                logger.print(f"T={state.timestamp}: CONDITION 4 MET: High sunlight value ({sunlight_value} >= {self.sunlight_value_threshold}) and positive slope ({slope:.4f} > 0) and Price above fair value ({current_mid:.2f} > {self.fair_value_upper}) - Going fully short")
                self.target_position = -self.position_limit
                self.active_signal = "SUNLIGHT_SHORT"
                signal_detected = True
        else:
            logger.print(f"T={state.timestamp}: Slope calculation failed or not enough data.")

        # If no signal detected, check if we should apply RSI strategy or stay put
        if not signal_detected:
            # Only apply RSI if price is within fair value range
            if self.fair_value_lower <= current_mid <= self.fair_value_upper:
                logger.print(f"T={state.timestamp}: CONDITIONS 1-4 NOT MET, Price in fair value range ({self.fair_value_lower}-{self.fair_value_upper}) - Applying RSI strategy")
                self._apply_rsi_strategy(state, best_bid, best_ask, current_mid, available_to_buy, available_to_sell, conversion_request)
            else:
                # Reset active signal as we're staying put
                self.active_signal = None
                self.target_position = current_position  # Maintain current position
                logger.print(f"T={state.timestamp}: CONDITIONS 1-4 NOT MET, Price outside fair value range - Staying put")
            
            return self.orders, conversion_request
        
        # Start executing the new signal
        if signal_detected:
            units_needed = abs(self.target_position - current_position)
            direction = 1 if self.target_position > current_position else -1
            
            if direction > 0 and available_to_buy > 0:
                # Buy direction
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                trade_units = min(units_needed, ask_volume, available_to_buy)
                
                if trade_units > 0:
                    self._place_buy_order(best_ask, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ Initial BUY ORDER: {trade_units} @ {best_ask}")
                    units_needed -= trade_units
                
                # Use conversions if needed
                if units_needed > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_needed, self.conversion_limit)
                    conversion_request = conv_units
                    logger.print(f"T={state.timestamp}: ✅ Requesting IMPORT conversion: {conv_units} units")
                    
            elif direction < 0 and available_to_sell > 0:
                # Sell direction
                bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                trade_units = min(units_needed, bid_volume, available_to_sell)
                
                if trade_units > 0:
                    self._place_sell_order(best_bid, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ Initial SELL ORDER: {trade_units} @ {best_bid}")
                    units_needed -= trade_units
                
                # Use conversions if needed
                if units_needed > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_needed, self.conversion_limit)
                    conversion_request = -conv_units
                    logger.print(f"T={state.timestamp}: ✅ Requesting EXPORT conversion: {conv_units} units")
        
        return self.orders, conversion_request
        
    def _apply_rsi_strategy(self, state: TradingState, best_bid: float, best_ask: float, 
                          current_mid: float, available_to_buy: int, available_to_sell: int, conversion_request: int) -> None:
        rsi = self.calculate_rsi()
        
        if rsi is not None:
            logger.print(f"T={state.timestamp}: IMPLEMENTING RSI STRATEGY - RSI: {rsi:.2f} (Oversold < {self.rsi_oversold}, Overbought > {self.rsi_overbought})")
            
            current_position = state.position.get(self.symbol, 0)
            order_depth = state.order_depths[self.symbol]
            
            if rsi < self.rsi_oversold and available_to_buy > 0:
                # RSI oversold - time to buy
                self.target_position = min(current_position + self.rsi_target_units, self.position_limit)
                self.active_signal = "RSI_LONG"
                
                # Determine how many units to buy
                units_to_buy = self.target_position - current_position
                
                # First try market orders
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                trade_units = min(units_to_buy, ask_volume)
                
                if trade_units > 0:
                    self._place_buy_order(best_ask, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ RSI OVERSOLD - BUY ORDER: {trade_units} @ {best_ask}")
                    units_to_buy -= trade_units
                
                # Then use conversions if possible
                if units_to_buy > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_buy, self.conversion_limit)
                    conversion_request = conv_units
                    logger.print(f"T={state.timestamp}: ✅ RSI OVERSOLD - IMPORT CONVERSION: {conv_units} units")
                
            elif rsi > self.rsi_overbought and available_to_sell > 0:
                # RSI overbought - time to sell
                self.target_position = max(current_position - self.rsi_target_units, -self.position_limit)
                self.active_signal = "RSI_SHORT"
                
                # Determine how many units to sell
                units_to_sell = current_position - self.target_position
                
                # First try market orders
                bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
                trade_units = min(units_to_sell, bid_volume)
                
                if trade_units > 0:
                    self._place_sell_order(best_bid, trade_units)
                    logger.print(f"T={state.timestamp}: ✅ RSI OVERBOUGHT - SELL ORDER: {trade_units} @ {best_bid}")
                    units_to_sell -= trade_units
                
                # Then use conversions if possible
                if units_to_sell > 0 and abs(conversion_request) < self.conversion_limit:
                    conv_units = min(units_to_sell, self.conversion_limit)
                    conversion_request = -conv_units
                    logger.print(f"T={state.timestamp}: ✅ RSI OVERBOUGHT - EXPORT CONVERSION: {conv_units} units")
                
            else:
                # Reset active signal if RSI is neutral
                if current_position == self.target_position:
                    self.active_signal = None
                logger.print(f"T={state.timestamp}: ❌ RSI NEUTRAL ({self.rsi_oversold} <= {rsi:.2f} <= {self.rsi_overbought}) - NO NEW TRADE")
        else:
            logger.print(f"T={state.timestamp}: ❌ NOT ENOUGH DATA FOR RSI - NO TRADE")
    
    def _place_buy_order(self, price: float, quantity: int) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        self.orders.append(Order(self.symbol, price, quantity))
        
    def _place_sell_order(self, price: float, quantity: int) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        self.orders.append(Order(self.symbol, price, -quantity))
    
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        self.orders = []
        orders, conversion_request = self.act(state)
        return orders, conversion_request
        
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
        if not data:
            return
            
        if "price_history" in data and isinstance(data["price_history"], list):
            self.price_history = deque(data["price_history"], maxlen=200)
            
        if "mid_price_history" in data and isinstance(data["mid_price_history"], list):
            self.mid_price_history = deque(data["mid_price_history"], maxlen=200)
            
        if "sunlight_history" in data and isinstance(data["sunlight_history"], list):
            self.sunlight_history = deque(data["sunlight_history"], maxlen=5)
            
        if "current_sunlight_index" in data:
            self.current_sunlight_index = data["current_sunlight_index"]
            
        if "target_position" in data:
            self.target_position = data["target_position"]
            
        if "active_signal" in data:
            self.active_signal = data["active_signal"]


class Trader:
    def __init__(self):
        self.macarons_strategy = MacaronsStrategy()
    
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        
        trader_data = {}
        try:
            if state.traderData and state.traderData != "\"\"":
                trader_data = json.loads(state.traderData)
        except Exception:
            pass
            
        self.macarons_strategy.load(trader_data.get("macarons", {}))
        
        macarons_orders, conversions = self.macarons_strategy.run(state)
        if macarons_orders:
            result[self.macarons_strategy.symbol] = macarons_orders
        
        new_trader_data = {
            "macarons": self.macarons_strategy.save()
        }
        
        encoded_trader_data = json.dumps(new_trader_data, cls=ProsperityEncoder)
        logger.flush(state, result, conversions, encoded_trader_data)
        return result, conversions, encoded_trader_data