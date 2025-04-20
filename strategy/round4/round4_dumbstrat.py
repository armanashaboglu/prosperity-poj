import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

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


class BotObserverStrategy:
    def __init__(self) -> None:
        self.symbol = "MAGNIFICENT_MACARONS"
        self.position_limit = 75
        
        # Explicitly track the current mode (0 = short sell, 1 = convert)
        self.current_mode = 0  # Start with short selling
        
        # Units to trade each time
        self.trade_units = 10
        
        # Track timestamps for debugging
        self.last_timestamp = -1
        
    def short_sell(self, state: TradingState) -> List[Order]:
        """Execute short sell logic"""
        orders = []
        
        if self.symbol not in state.order_depths:
            logger.print(f"T={state.timestamp}: No order depth for {self.symbol} - cannot short sell")
            return orders
            
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders:
            logger.print(f"T={state.timestamp}: No buy orders in the book - cannot short sell")
            return orders
            
        # Get current position
        current_position = state.position.get(self.symbol, 0)
        
        # Calculate best bid price (market price)
        best_bid = max(order_depth.buy_orders.keys())
        sell_price = best_bid + 1
        
        # Check position limits before placing order
        if current_position - self.trade_units >= -self.position_limit:
            # Place short sell order at market price (bid + 1)
            orders.append(Order(self.symbol, sell_price, -self.trade_units))
            logger.print(f"T={state.timestamp}: ✅ SHORT SELLING {self.trade_units} units at {sell_price}")
            logger.print(f"T={state.timestamp}: Current position: {current_position}, After trade: {current_position - self.trade_units}")
        else:
            logger.print(f"T={state.timestamp}: ❌ Cannot short sell - would exceed position limit")
            logger.print(f"T={state.timestamp}: Current position: {current_position}, Limit: {self.position_limit}")
            
        return orders
        
    def convert(self, state: TradingState) -> int:
        """Execute conversion request logic"""
        # Log the conversion request
        logger.print(f"T={state.timestamp}: ✅ REQUESTING CONVERSION of {self.trade_units} units")
        logger.print(f"T={state.timestamp}: Current position: {state.position.get(self.symbol, 0)}")
        
        # Return the number of units to convert
        return self.trade_units
    
    def act(self, state: TradingState) -> Tuple[List[Order], int]:
        """Main strategy logic"""
        orders = []
        conversion_request = 0
        
        # Debug timestamp progression
        logger.print(f"T={state.timestamp}: Last timestamp was {self.last_timestamp}")
        logger.print(f"T={state.timestamp}: Current mode is {self.current_mode} (0=short sell, 1=convert)")
        
        # Execute strategy based on current mode
        if self.current_mode == 0:
            # Short sell mode
            orders = self.short_sell(state)
            # Switch to convert mode for next timestamp
            self.current_mode = 1
            logger.print(f"T={state.timestamp}: Switching to CONVERT mode for next timestamp")
        else:
            # Convert mode
            conversion_request = self.convert(state)
            # Switch to short sell mode for next timestamp
            self.current_mode = 0
            logger.print(f"T={state.timestamp}: Switching to SHORT SELL mode for next timestamp")
        
        # Update last timestamp
        self.last_timestamp = state.timestamp
        
        return orders, conversion_request
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        orders, conversion_request = self.act(state)
        return orders, conversion_request
        
    def save(self) -> dict:
        return {
            "current_mode": self.current_mode,
            "last_timestamp": self.last_timestamp
        }
        
    def load(self, data: dict) -> None:
        if not data:
            return
            
        if "current_mode" in data:
            self.current_mode = data["current_mode"]
            
        if "last_timestamp" in data:
            self.last_timestamp = data["last_timestamp"]


class Trader:
    def __init__(self):
        self.bot_observer_strategy = BotObserverStrategy()
    
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        
        trader_data = {}
        try:
            if state.traderData and state.traderData != "\"\"":
                trader_data = json.loads(state.traderData)
        except Exception:
            pass
            
        self.bot_observer_strategy.load(trader_data.get("bot_observer", {}))
        
        orders, conversions = self.bot_observer_strategy.run(state)
        if orders:
            result[self.bot_observer_strategy.symbol] = orders
        
        new_trader_data = {
            "bot_observer": self.bot_observer_strategy.save()
        }
        
        encoded_trader_data = json.dumps(new_trader_data, cls=ProsperityEncoder)
        logger.flush(state, result, conversions, encoded_trader_data)
        return result, conversions, encoded_trader_data
