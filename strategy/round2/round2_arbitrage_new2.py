import json
import math
from typing import Any, Dict, List, Deque, Optional
from collections import defaultdict, deque
from abc import abstractmethod
import copy
import numpy as np

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
        try:
            safe_orders = {str(k): v for k, v in orders.items()} if isinstance(orders, dict) else {}
            safe_state_positions = {str(k): v for k, v in state.position.items()} if hasattr(state, 'position') and isinstance(state.position, dict) else {}
            minimal_state_repr = self.compress_state(state, "")

            base_json = self.to_json([
                minimal_state_repr,
                self.compress_orders(safe_orders),
                conversions,
                "",
                ""
            ])
            base_len = len(base_json)
            max_item_length = max(0, (self.max_log_length - base_len)//3)
            safe_trader_data = trader_data if isinstance(trader_data, str) else "{}"
            safe_state_trader_data = state.traderData if hasattr(state, 'traderData') and isinstance(state.traderData, str) else ""

            print(self.to_json([
                self.compress_state(state, self.truncate(safe_state_trader_data, max_item_length)),
                self.compress_orders(safe_orders),
                conversions,
                self.truncate(safe_trader_data, max_item_length),
                self.truncate(self.logs, max_item_length)
            ]))
        except Exception as e:
            print(json.dumps([state.timestamp if hasattr(state, 'timestamp') else 0, conversions, f"LogFlushError: {e}"]))
        finally:
            self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        ts = getattr(state, 'timestamp', 0)
        listings = getattr(state, 'listings', {}) or {}
        order_depths = getattr(state, 'order_depths', {}) or {}
        own_trades = getattr(state, 'own_trades', {}) or {}
        market_trades = getattr(state, 'market_trades', {}) or {}
        position = getattr(state, 'position', {}) or {}
        observations = getattr(state, 'observations', Observation({}, {}))
        return [
            ts, trader_data,
            self.compress_listings(listings),
            self.compress_order_depths(order_depths),
            self.compress_trades(own_trades),
            self.compress_trades(market_trades),
            {str(k): v for k, v in position.items()},
            self.compress_observations(observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        arr = []
        if isinstance(listings, dict):
            for lst in listings.values():
                if all(hasattr(lst, attr) for attr in ['symbol', 'product', 'denomination']):
                    arr.append([lst.symbol, lst.product, lst.denomination])
        return arr

    def compress_order_depths(self, od_map: Dict[Symbol, OrderDepth]) -> Dict[str, List[Any]]:
        c = {}
        if isinstance(od_map, dict):
            for sym, od in od_map.items():
                buy_orders = od.buy_orders if hasattr(od, 'buy_orders') and isinstance(od.buy_orders, dict) else {}
                sell_orders = od.sell_orders if hasattr(od, 'sell_orders') and isinstance(od.sell_orders, dict) else {}
                safe_buy = {str(p): v for p, v in buy_orders.items()}
                safe_sell = {str(p): v for p, v in sell_orders.items()}
                c[str(sym)] = [safe_buy, safe_sell]
        return c

    def compress_trades(self, trades_map: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        arr = []
        if isinstance(trades_map, dict):
            for tlist in trades_map.values():
                if isinstance(tlist, list):
                    for t in tlist:
                        if all(hasattr(t, attr) for attr in ['symbol', 'price', 'quantity', 'buyer', 'seller', 'timestamp']):
                            arr.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return arr

    def compress_observations(self, obs: Observation) -> List[Any]:
        conv_obs = {}
        if hasattr(obs, 'conversionObservations') and isinstance(obs.conversionObservations, dict):
            for product, ob in obs.conversionObservations.items():
                conv_obs[str(product)] = [
                    getattr(ob, 'bidPrice', None), getattr(ob, 'askPrice', None),
                    getattr(ob, 'transportFees', None), getattr(ob, 'exportTariff', None),
                    getattr(ob, 'importTariff', None)
                ]
        plain_obs = getattr(obs, 'plainValueObservations', {}) or {}
        safe_plain_obs = {str(k): v for k, v in plain_obs.items()}
        return [safe_plain_obs, conv_obs]

    def compress_orders(self, orders_map: Dict[str, List[Order]]) -> List[List[Any]]:
        arr = []
        if isinstance(orders_map, dict):
            for sym_list in orders_map.values():
                if isinstance(sym_list, list):
                    for o in sym_list:
                        if isinstance(o, Order) and hasattr(o, 'symbol') and hasattr(o, 'price') and hasattr(o, 'quantity'):
                            arr.append([o.symbol, o.price, o.quantity])
        return arr

    def to_json(self, val: Any) -> str:
        try: return json.dumps(val, cls=ProsperityEncoder, separators=(",", ":"))
        except Exception: return json.dumps(val, separators=(",", ":"), default=str)

    def truncate(self, s: str, maxlen: int) -> str:
        maxlen = max(0, maxlen)
        if not isinstance(s, str): s = str(s)
        if len(s) <= maxlen: return s
        if maxlen < 3: return s[:maxlen]
        return s[:maxlen-3]+"..."

logger = Logger()

class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"


PARAMS = {
    # "SQUID_INK": { 
    #     "rsi_window": 96,  
    #     "rsi_overbought": 56, 
    #     "rsi_oversold": 39, 
    # }, <<< REMOVED RSI PARAMS
    Product.SQUID_INK: { # <<< ADDED SMA Z-Score Params
        "sma_window": 125,       # Window for Simple Moving Average
        "zscore_window": 125,    # Window for Z-Score calculation (of deviation)
        "zscore_entry_long": -2.5, # Z-score threshold to initiate long position
        "zscore_entry_short": 2.5, # Z-score threshold to initiate short position
        # "zscore_exit_long": 0.0,   # <<< REMOVED
        # "zscore_exit_short": 0.0,  # <<< REMOVED
    },
    "RAINFOREST_RESIN": { 
        "fair_value": 10000,
    },}
