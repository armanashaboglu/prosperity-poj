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


POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    Product.SQUID_INK: 50,
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50
}

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
        try:
            if self.symbol in state.order_depths:
                self.act(state)
        except Exception as e: logger.print(f"ERROR in act() for {self.symbol}: {e}")
        return self.orders

    def _place_buy_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        try:
            price_int = int(round(float(price)))
            quantity_int = int(math.floor(float(quantity)))
            if quantity_int <= 0 or price_int <= 0: return
            self.orders.append(Order(self.symbol, price_int, quantity_int))
        except Exception as e: logger.print(f"Error placing BUY {self.symbol}: {e}")

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        try:
            price_int = int(round(float(price)))
            quantity_int = int(math.floor(float(quantity)))
            if quantity_int <= 0 or price_int <= 0: return
            self.orders.append(Order(self.symbol, price_int, -quantity_int))
        except Exception as e: logger.print(f"Error placing SELL {self.symbol}: {e}")

    def save(self) -> dict: return {}
    def load(self, data: dict) -> None: pass

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        try:
            od = state.order_depths.get(symbol)
            if not od: return None
            buy_orders = od.buy_orders if isinstance(od.buy_orders, dict) else {}
            sell_orders = od.sell_orders if isinstance(od.sell_orders, dict) else {}
            numeric_bids = {p: v for p, v in buy_orders.items() if isinstance(p, (int, float))}
            numeric_asks = {p: v for p, v in sell_orders.items() if isinstance(p, (int, float))}

            best_bid = max(numeric_bids.keys()) if numeric_bids else None
            best_ask = min(numeric_asks.keys()) if numeric_asks else None

            if best_bid is not None and best_ask is not None: return (float(best_bid) + float(best_ask)) / 2.0
            elif best_bid is not None: return float(best_bid)
            elif best_ask is not None: return float(best_ask)
            else: return None
        except Exception as e: logger.print(f"Error getting mid price {symbol}: {e}"); return None

class V3Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []
    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()
    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        try:
            if self.symbol in state.order_depths: self.act(state)
        except Exception as e: logger.print(f"ERROR in V3 act {self.symbol}: {e}")
        return self.orders
    def buy(self, price: int, quantity: int) -> None:
        if quantity <= 0 or price <= 0: return
        try: self.orders.append(Order(self.symbol, int(price), int(quantity)))
        except Exception as e: logger.print(f"Error V3 buy {self.symbol}: {e}")
    def sell(self, price: int, quantity: int) -> None:
        if quantity <= 0 or price <= 0: return
        try: self.orders.append(Order(self.symbol, int(price), -int(quantity)))
        except Exception as e: logger.print(f"Error V3 sell {self.symbol}: {e}")
    def save(self) -> dict: return {}
    def load(self, data: dict) -> None: pass

class V3MarketMakingStrategy(V3Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window_size = 4
        self.window: Deque[bool] = deque(maxlen=max(1, self.window_size))
    @abstractmethod
    def get_true_value(self, state: TradingState) -> int: raise NotImplementedError()
    def act(self, state: TradingState) -> None: pass
    def save(self) -> dict:
        state_data = super().save()
        try: state_data["window"] = list(self.window)
        except Exception as e: logger.print(f"Error save V3MM window: {e}")
        return state_data
    def load(self, data: dict) -> None:
        super().load(data)
        try:
            maxlen = max(1, self.window_size)
            if not hasattr(self, 'window') or not isinstance(self.window, deque) or self.window.maxlen != maxlen:
                self.window = deque(maxlen=maxlen)
            else: self.window.clear()
            if isinstance(data, dict) and "window" in data and isinstance(data["window"], list):
                loaded = data["window"]
                valid = [item for item in loaded if isinstance(item, bool)]
                self.window.extend(valid[max(0, len(valid) - maxlen):])
        except Exception as e:
            logger.print(f"Error load V3MM window: {e}")
            maxlen = max(1, self.window_size)
            if not hasattr(self, 'window') or not isinstance(self.window, deque) or self.window.maxlen != maxlen:
                 self.window = deque(maxlen=maxlen)
            else: self.window.clear()

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

class PrototypeMarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.window_size = 4
        self.window: Deque[bool] = deque(maxlen=max(1, self.window_size))

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

        self.window.append(abs(position) == self.position_limit)
        is_full_window = len(self.window) == self.window_size
        stuck_count = sum(self.window)
        soft_liquidate = is_full_window and stuck_count >= self.window_size / 2 and self.window[-1]
        hard_liquidate = is_full_window and stuck_count == self.window_size

        if hard_liquidate:
            if to_buy > 0:
                quantity = to_buy // 2
                self._place_buy_order(true_value, quantity)
                to_buy -= quantity
            elif to_sell > 0:
                 quantity = to_sell // 2
                 self._place_sell_order(true_value, quantity)
                 to_sell -= quantity
        elif soft_liquidate:
            if to_buy > 0:
                quantity = to_buy // 2
                liq_price = true_value - 2
                self._place_buy_order(liq_price, quantity)
                to_buy -= quantity
            elif to_sell > 0:
                quantity = to_sell // 2
                liq_price = true_value + 2
                self._place_sell_order(liq_price, quantity)
                to_sell -= quantity

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
            self.window = deque(loaded_window, maxlen=max(1, self.window_size))

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

# <<< --- NEW STRATEGY: Squid Ink SMA Z-Score --- >>>

class SquidInkSmaZscoreStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {})
        if not isinstance(self.params, dict): self.params = {}
        if not self.params:
            # Default no longer needs exit params
            default = {"sma_window": 50, "zscore_window": 20, "zscore_entry_long": -2.0, "zscore_entry_short": 2.0}
            logger.print(f"Warn: Using default SMA Z-Score params {symbol}: {default}"); self.params = default

        self.sma_window = max(2, int(self.params.get("sma_window", 50)))
        self.zscore_window = max(2, int(self.params.get("zscore_window", 20)))
        self.zscore_entry_long = float(self.params.get("zscore_entry_long", -2.0))
        self.zscore_entry_short = float(self.params.get("zscore_entry_short", 2.0))
        # # <<< REMOVED: Load exit parameters >>>
        # self.zscore_exit_long = float(self.params.get("zscore_exit_long", 0.0))
        # self.zscore_exit_short = float(self.params.get("zscore_exit_short", 0.0))

        # --- Validation --- 
        if self.zscore_entry_long > 0: self.zscore_entry_long *= -1
        if self.zscore_entry_short < 0: self.zscore_entry_short *= -1
        if self.zscore_entry_short <= abs(self.zscore_entry_long):
            logger.print(f"Warn: Short entry {self.zscore_entry_short} <= abs(Long entry {self.zscore_entry_long}). Adjusting short entry.")
            self.zscore_entry_short = abs(self.zscore_entry_long) + 0.1
        # <<< REMOVED Exit threshold validation >>>
        # --- End Validation ---

        self.history_maxlen = max(self.sma_window, self.zscore_window)
        self.mid_price_history: deque[float] = deque(maxlen=self.history_maxlen)
        self.current_sma: Optional[float] = None

        # <<< ADDED: State to track intended position (1=long, -1=short, 0=initial) >>>
        self.intended_state: int = 0

        logger.print(f"Init {self.symbol}: SMA Win={self.sma_window}, Z Win={self.zscore_window}, Z Entry(L/S)={self.zscore_entry_long:.2f}/{self.zscore_entry_short:.2f}, Hist Maxlen={self.history_maxlen}") # Simplified log

    def _calculate_sma(self) -> Optional[float]:
        if len(self.mid_price_history) < self.sma_window:
            return None
        try:
            relevant_history = list(self.mid_price_history)[-self.sma_window:]
            return np.mean(relevant_history)
        except Exception as e:
            logger.print(f"Error calculating SMA for {self.symbol}: {e}")
            return None

    def act(self, state: TradingState) -> None:
        self.orders = []
        mid_price = self._get_mid_price(self.symbol, state)
        if mid_price is None: return
        self.mid_price_history.append(mid_price)

        self.current_sma = self._calculate_sma()
        if self.current_sma is None: return

        z_score: Optional[float] = None
        rolling_std: Optional[float] = None
        if len(self.mid_price_history) >= self.zscore_window:
            try:
                relevant_history = list(self.mid_price_history)[-self.zscore_window:]
                rolling_std = np.std(relevant_history)
                if rolling_std is not None and rolling_std < 1e-6:
                    rolling_std = None
                elif rolling_std is not None:
                    z_score = (mid_price - self.current_sma) / rolling_std
            except Exception as e:
                logger.print(f"Error calculating Price STD/Z-score for {self.symbol}: {e}")
                rolling_std = None; z_score = None

        if z_score is None: return

        # --- Determine Intended State based on Z-Score --- 
        if z_score <= self.zscore_entry_long:
            self.intended_state = 1 # Signal Long
        elif z_score >= self.zscore_entry_short:
            self.intended_state = -1 # Signal Short
        # If between thresholds, self.intended_state *does not change*
        # --- End Intended State Logic ---
        
        # --- Determine Target Position based on Intended State --- 
        if self.intended_state == 1:
            target_position = self.position_limit
        elif self.intended_state == -1:
            target_position = -self.position_limit
        else: # intended_state == 0 (only at the very beginning)
            target_position = 0
        # --- End Target Position Logic ---

        # Execute Trades to Reach Target Position
        current_position = state.position.get(self.symbol, 0)
        qty_to_trade = target_position - current_position

        # Log details
        log_sma = f"{self.current_sma:.2f}" if self.current_sma is not None else "N/A"
        log_std = f"{rolling_std:.2f}" if rolling_std is not None else "N/A"
        log_z = f"{z_score:.2f}"
        logger.print(f"{self.symbol}: Mid={mid_price:.2f}, SMA={log_sma}, PriceStd={log_std}, Z={log_z}, IntendedState={self.intended_state}, CurrPos={current_position}, TargetPos={target_position}, TradeQty={qty_to_trade}")

        if qty_to_trade == 0: return

        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return

        if qty_to_trade > 0: # Need to BUY
            best_ask = min((p for p in (order_depth.sell_orders or {}).keys() if isinstance(p, (int, float))), default=None)
            if best_ask is not None:
                price = float(best_ask + 1)
                self._place_buy_order(price, float(qty_to_trade))
            else:
                logger.print(f"Warn: Cannot place BUY for {self.symbol}, no asks found.")
        elif qty_to_trade < 0: # Need to SELL
            best_bid = max((p for p in (order_depth.buy_orders or {}).keys() if isinstance(p, (int, float))), default=None)
            if best_bid is not None:
                price = float(best_bid - 1)
                self._place_sell_order(price, float(abs(qty_to_trade)))
            else:
                logger.print(f"Warn: Cannot place SELL for {self.symbol}, no bids found.")

    def save(self) -> dict:
        return {
            "mid_price_history": list(self.mid_price_history),
            "intended_state": self.intended_state # <<< ADDED
        }

    def load(self, data: dict) -> None:
        if not isinstance(data, dict): data = {}
        # Load mid price history
        loaded_mid_hist = data.get("mid_price_history", [])
        if not hasattr(self, 'mid_price_history') or not isinstance(self.mid_price_history, deque) or self.mid_price_history.maxlen != self.history_maxlen:
            self.mid_price_history = deque(maxlen=self.history_maxlen)
        else: self.mid_price_history.clear()
        if isinstance(loaded_mid_hist, list):
            valid_mid = [p for p in loaded_mid_hist if isinstance(p, (int, float))]
            self.mid_price_history.extend(valid_mid[max(0, len(valid_mid) - self.history_maxlen):])
        
        # <<< ADDED: Load intended state >>>
        loaded_state = data.get("intended_state", 0)
        if isinstance(loaded_state, int) and loaded_state in [-1, 0, 1]:
            self.intended_state = loaded_state
        else:
            self.intended_state = 0 # Default to 0 if invalid data

        # Recalculate SMA based on loaded history if possible
        self.current_sma = self._calculate_sma()

# <<< --- END NEW STRATEGY --- >>>

# --- Arbitrage Parameters (Moved to Global Scope) ---
ARB_PARAMS = {
    "diff_threshold_b1": 213,
    "diff_threshold_b2": 126,
    "max_arb_lot": 9
}

# --- Z-Score Arbitrage Parameters (NEW) ---
# Defaults inspired by optimizer_r2.py
ZSCORE_ARB_PARAMS = {
    "deviation_mean": 18.5, # Should be analyzed/optimized
    "deviation_std_window": 44, # Example starting value
    "zscore_threshold_entry_long": 15.8, # <<< ADDED: Threshold to BUY deviation (when Z < -threshold)
    "zscore_threshold_entry_short": 24.2, # <<< ADDED: Threshold to SELL deviation (when Z > threshold)
    "target_deviation_spread_size": 2, # Example starting value (max units to target)
}

# --- Theoretical Spread Definition (for Z-Score Arb) ---
B1B2_THEORETICAL_COMPONENTS = {
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: 1
}

class Trader:
    def __init__(self):
        # Arbitrage parameters (reference the global dict)
        # self.arb_params is no longer needed
        # self.arb_params = {
        #     "diff_threshold_b1": 200, "diff_threshold_b2": 120,
        #     "diff_threshold_b1_b2": 60, "max_arb_lot": 5
        # }
        # Use global limits
        self.pos_limits = POSITION_LIMITS

        self.strategies: Dict[Symbol, Strategy] = {}
        strategy_map = {
            Product.RAINFOREST_RESIN: V3RainforestResinStrategy,
            Product.KELP: PrototypeKelpStrategy,
            # Product.SQUID_INK: SquidInkRsiStrategy, # <<< REMOVED RSI Strategy
            Product.SQUID_INK: SquidInkSmaZscoreStrategy, # <<< ADDED SMA Z-Score Strategy
        }

        for product, strategy_class in strategy_map.items():
            limit = self.pos_limits.get(product)
            if limit is not None:
                 try: self.strategies[product] = strategy_class(product, limit)
                 except Exception as e: logger.print(f"Error init {product}: {e}")
            else: logger.print(f"Warn: Limit missing {product}")

        logger.print(f"Trader Initialized: {len(self.strategies)} indiv strats + Basket Arb.")

        # --- Initialize Z-Score Arb State ---
        self.zscore_params = ZSCORE_ARB_PARAMS # Load params
        self.deviation_std_window = max(10, self.zscore_params.get("deviation_std_window", 100))
        self.deviation_history: Deque[float] = deque(maxlen=self.deviation_std_window)
        self.current_effective_deviation_pos: int = 0
        # <<< ADDED: Load asymmetric entry thresholds >>>
        self.zscore_threshold_entry_long = abs(self.zscore_params.get("zscore_threshold_entry_long", 2.0))
        self.zscore_threshold_entry_short = abs(self.zscore_params.get("zscore_threshold_entry_short", 2.0))
        logger.print(f"Z-Score Arb Initialized: Window={self.deviation_std_window}, Mean={self.zscore_params.get('deviation_mean')}, Z-LongEntry={self.zscore_threshold_entry_long}, Z-ShortEntry={self.zscore_threshold_entry_short}")
    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        all_orders: Dict[Symbol, List[Order]] = defaultdict(list)
        conversions = 0
        trader_data_next = {}

        try:
            loaded_data = json.loads(state.traderData if state.traderData else "{}")
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e: logger.print(f"Err load traderData: {e}"); loaded_data = {}

        # --- Load Z-Score Arb State --- 
        zscore_arb_state = loaded_data.get("zscore_arb_state", {})
        if isinstance(zscore_arb_state, dict):
            loaded_history = zscore_arb_state.get("deviation_history", [])
            if isinstance(loaded_history, list):
                # Ensure deque uses correct maxlen from params
                self.deviation_history = deque(loaded_history, maxlen=self.deviation_std_window)
            else: # Fallback if data is corrupted
                self.deviation_history = deque(maxlen=self.deviation_std_window)
            
            loaded_pos = zscore_arb_state.get("current_effective_deviation_pos", 0)
            if isinstance(loaded_pos, (int, float)):
                self.current_effective_deviation_pos = int(loaded_pos)
            else:
                self.current_effective_deviation_pos = 0
        else: # Fallback if state format incorrect
            self.deviation_history = deque(maxlen=self.deviation_std_window)
            self.current_effective_deviation_pos = 0
        # logger.print(f"Loaded Z-Score State: Hist Len={len(self.deviation_history)}, Eff Pos={self.current_effective_deviation_pos}")

        for key, strat in self.strategies.items():
            strat_key_str = str(key)
            strat_state = loaded_data.get(strat_key_str, {})
            if not isinstance(strat_state, dict): strat_state = {}

            try: strat.load(strat_state)
            except Exception as e: logger.print(f"Err load state {key}: {e}")

            try:
                orders = strat.run(state)
                if isinstance(orders, list):
                    for order in orders:
                         if isinstance(order, Order) and hasattr(order, 'symbol'):
                             all_orders[order.symbol].append(order)
            except Exception as e: logger.print(f"ERR run {key}: {e}")

            try: trader_data_next[strat_key_str] = strat.save()
            except Exception as e: logger.print(f"Err save state {key}: {e}")

        try:
            positions = getattr(state, 'position', {}) or {}
            relevant = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]
            swmid = {r: self.get_swmid(state.order_depths[r]) for r in relevant}
            c, j, d, b1, b2 = [swmid.get(p) for p in relevant]

            # --- Arb 1: Z-Score Deviation Logic (NEW) ---
            if all(p is not None for p in [b1, b2, c, j, d]):
                try:
                    # Calculate actual and theoretical spreads, and the deviation
                    actual_spread = float(b1) - float(b2)
                    theoretical_spread = (B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * float(c) +
                                          B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * float(j) +
                                          B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * float(d))
                    deviation = actual_spread - theoretical_spread
                    self.deviation_history.append(deviation)

                    # Check if history is sufficient for Z-score calculation
                    if len(self.deviation_history) >= self.deviation_std_window // 4: # Start trading sooner
                        # Calculate Z-score
                        current_deviation_history = list(self.deviation_history)
                        deviation_std = np.std(current_deviation_history)
                        
                        if deviation_std > 1e-6: # Avoid division by zero or near-zero std dev
                            deviation_mean = self.zscore_params.get("deviation_mean", 0)
                            z_score = (deviation - deviation_mean) / deviation_std
                            logger.print(f"ZScoreArb: Dev={deviation:.2f}, Mean={deviation_mean:.2f}, Std={deviation_std:.2f}, Z={z_score:.2f}, CurrEffPos={self.current_effective_deviation_pos}")

                            # Determine desired position based on Z-score
                            # <<< REMOVED old symmetric entry threshold >>>
                            # zscore_entry = abs(self.zscore_params.get("zscore_threshold_entry", 2.0))
                            # Use loaded asymmetric thresholds
                            target_size = abs(self.zscore_params.get("target_deviation_spread_size", 10))

                            desired_effective_deviation_pos = self.current_effective_deviation_pos # Default: no change

                            # <<< MODIFIED: Use asymmetric entry thresholds >>>
                            if z_score >= self.zscore_threshold_entry_short:
                                # Deviation is high -> Sell Deviation (-target_size)
                                desired_effective_deviation_pos = -target_size
                            elif z_score <= -self.zscore_threshold_entry_long:
                                # Deviation is low -> Buy Deviation (+target_size)
                                desired_effective_deviation_pos = target_size
                            else: # <<< ADDED: If not entering long or short, desired position is flat >>>
                                desired_effective_deviation_pos = 0

                            # Execute trades if desired position changed
                            qty_units_to_trade = desired_effective_deviation_pos - self.current_effective_deviation_pos
                            if qty_units_to_trade != 0:
                                logger.print(f"ZScoreArb: Target state change: {self.current_effective_deviation_pos} -> {desired_effective_deviation_pos}")
                                actual_units_traded = self.execute_zscore_deviation_trade(state, all_orders, qty_units_to_trade)
                                # Update internal state *after* attempting trade
                                self.current_effective_deviation_pos += actual_units_traded
                                logger.print(f"ZScoreArb: Attempted {qty_units_to_trade}, Actual Trade={actual_units_traded}. New effective pos: {self.current_effective_deviation_pos}")
                        else:
                           logger.print(f"ZScoreArb: Deviation std dev too low ({deviation_std:.4f}). Skipping Z-score calc.")
                    # else: logger.print(f"ZScoreArb: Insufficient history ({len(self.deviation_history)}/{self.deviation_std_window}) for Z-score.")

                except Exception as z_err:
                     logger.print(f"ERR Z-Score Arb Logic: {z_err}")
            
            
            # Arb 2: B1 vs Items
            if all(p is not None for p in [b1, c, j, d]):
                fair = 6*float(c) + 3*float(j) + 1*float(d)
                diff = float(b1) - fair
                # Use global ARB_PARAMS
                thr_pos = ARB_PARAMS["diff_threshold_b1"]
                thr_neg = -thr_pos + 50 # Original offset
                logger.print(f"B1vItems: B1={b1:.1f} Fair={fair:.1f} Diff={diff:.1f} ThrPos={thr_pos} ThrNeg={thr_neg}")
                if diff > thr_pos:
                    # Short B1, Buy Items
                    self.execute_spread_trade(all_orders, state, Product.PICNIC_BASKET1, -1, {Product.CROISSANTS: +6, Product.JAMS:+3, Product.DJEMBES:+1}, ARB_PARAMS["max_arb_lot"])
                elif diff < thr_neg:
                    # Buy B1, Short Items
                    self.execute_spread_trade(all_orders, state, Product.PICNIC_BASKET1, +1, {Product.CROISSANTS: -6, Product.JAMS:-3, Product.DJEMBES:-1}, ARB_PARAMS["max_arb_lot"])

            # Arb 3: B2 vs Items
            if all(p is not None for p in [b2, c, j]):
                fair = 4*float(c) + 2*float(j)
                diff = float(b2) - fair
                # Use global ARB_PARAMS
                thr_pos = ARB_PARAMS["diff_threshold_b2"]
                thr_neg = -thr_pos + 30 # Original offset
                logger.print(f"B2vItems: B2={b2:.1f} Fair={fair:.1f} Diff={diff:.1f} ThrPos={thr_pos} ThrNeg={thr_neg}")
                if diff > thr_pos:
                    # Short B2, Buy Items
                    self.execute_spread_trade(all_orders, state, Product.PICNIC_BASKET2, -1, {Product.CROISSANTS:+4, Product.JAMS:+2}, ARB_PARAMS["max_arb_lot"])
                elif diff < thr_neg:
                    # Buy B2, Short Items
                    self.execute_spread_trade(all_orders, state, Product.PICNIC_BASKET2, +1, {Product.CROISSANTS:-4, Product.JAMS:-2}, ARB_PARAMS["max_arb_lot"])

        except Exception as e: logger.print(f"ERR Basket Arb: {e}")

        # --- Save Z-Score Arb State --- 
        trader_data_next["zscore_arb_state"] = {
            "deviation_history": list(self.deviation_history),
            "current_effective_deviation_pos": self.current_effective_deviation_pos
        }

        final_orders = dict(all_orders)

        try: trader_data_encoded = json.dumps(trader_data_next, separators=(",", ":"), default=str)
        except Exception as e: logger.print(f"Err encode traderData: {e}"); trader_data_encoded = "{}"

        logger.flush(state, final_orders, conversions, trader_data_encoded)
        return final_orders, conversions, trader_data_encoded


    def execute_spread_trade(self, orders_dict, state, basket_symbol, basket_side, components_sides, max_single_lot):
        try:
            positions = getattr(state, 'position', {}) or {}
            current_pos = positions.get(basket_symbol, 0)
            limit_bkt = self.pos_limits.get(basket_symbol, 0)

            capacity = (limit_bkt - current_pos) if basket_side > 0 else (limit_bkt + current_pos)
            if capacity <= 0:
                return

            trade_qty = int(min(max_single_lot, capacity))
            if trade_qty <= 0: return

            for comp, side_per in components_sides.items():
                comp_pos = positions.get(comp, 0); limit_comp = self.pos_limits.get(comp, 0)
                needed = abs(side_per * trade_qty)
                cap_comp = (limit_comp - comp_pos) if side_per > 0 else (limit_comp + comp_pos)
                if cap_comp < needed:
                    return

            od_bkt = state.order_depths.get(basket_symbol)
            if not od_bkt: return
            buy_bkt = {p: v for p, v in (od_bkt.buy_orders or {}).items() if isinstance(p, (int, float))}
            sell_bkt = {p: v for p, v in (od_bkt.sell_orders or {}).items() if isinstance(p, (int, float))}

            if basket_side > 0:
                 if sell_bkt:
                      best_ask = min(sell_bkt.keys())
                      px = int(best_ask+1)
                      orders_dict[basket_symbol].append(Order(basket_symbol, px, trade_qty))
            else:
                 if buy_bkt:
                      best_bid = max(buy_bkt.keys())
                      px = int(best_bid-1)
                      orders_dict[basket_symbol].append(Order(basket_symbol, px, -trade_qty))

            # Place component orders
            for comp, side_per in components_sides.items():
                qty = int(side_per * trade_qty)
                if qty == 0: continue

                od_comp = state.order_depths.get(comp)
                if not od_comp: continue
                buy_comp = {p: v for p, v in (od_comp.buy_orders or {}).items() if isinstance(p, (int, float))}
                sell_comp = {p: v for p, v in (od_comp.sell_orders or {}).items() if isinstance(p, (int, float))}

                if qty > 0:
                     if sell_comp:
                          best_ask = min(sell_comp.keys())
                          px = int(best_ask+1)
                          orders_dict[comp].append(Order(comp, px, qty))
                else:
                     if buy_comp:
                          best_bid = max(buy_comp.keys())
                          px= int(best_bid-1)
                          orders_dict[comp].append(Order(comp, px, qty))

        except Exception as e: logger.print(f"Error exec spread {basket_symbol}: {e}")

    def execute_zscore_deviation_trade(self, state: TradingState, orders_dict: Dict[Symbol, List[Order]], qty_units_to_trade: int) -> int:
        """Calculates executable size considering limits & liquidity, then places aggressive orders.
           Returns the actual number of units traded.
        """
        if qty_units_to_trade == 0: return 0

        direction = 1 if qty_units_to_trade > 0 else -1
        max_abs_units_to_trade = abs(qty_units_to_trade)

        # 1. Check Position Limit Constraint
        max_units_pos = self._calculate_max_deviation_spread_size(state, direction)
        if max_units_pos <= 0:
            logger.print(f"ZScore Execute: Cannot trade {direction} unit(s), blocked by position limit.")
            return 0 # Return 0 units traded

        # 2. Check Market Liquidity Constraint
        max_units_liq = self._calculate_market_liquidity_limit(state, direction)
        if max_units_liq <= 0:
            logger.print(f"ZScore Execute: Cannot trade {direction} unit(s), blocked by market liquidity.")
            return 0 # Return 0 units traded

        # 3. Determine Actual Executable Units
        actual_units_to_trade = direction * min(max_abs_units_to_trade, max_units_pos, max_units_liq)

        if actual_units_to_trade == 0:
            logger.print(f"ZScore Execute: Calculated 0 actual units to trade (Target Chg: {qty_units_to_trade}, MaxPos: {max_units_pos}, MaxLiq: {max_units_liq}).")
            return 0 # Return 0 units traded

        logger.print(f"ZScore Execute: Attempting to trade {actual_units_to_trade} deviation units (LimitPos: {max_units_pos}, LimitLiq: {max_units_liq}).")

        # 4. Define quantity changes based on the ACTUAL units we will trade
        final_qty_changes = {
            Product.PICNIC_BASKET1: +actual_units_to_trade,
            Product.PICNIC_BASKET2: -actual_units_to_trade,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * actual_units_to_trade,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * actual_units_to_trade,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * actual_units_to_trade,
        }

        # 5. Place aggressive orders for each leg
        order_depths = state.order_depths
        for product, final_qty_float in final_qty_changes.items():
            final_qty = int(round(final_qty_float)) # Ensure integer
            if final_qty == 0: continue

            od = order_depths.get(product)
            if not od:
                logger.print(f"Error: ZScore - Order depth for {product} disappeared before placing orders!")
                # Note: This leg fails, trade might be partially hedged.
                # Consider if we should cancel previous legs (complex) or proceed.
                continue # Skip this leg for now
            
            # Use the helper to place orders for this leg, appending to the main orders_dict
            self._place_aggressive_orders_for_leg(product, final_qty, od, orders_dict)

        # 6. Return the actual number of units traded (signed)
        return actual_units_to_trade

    def _calculate_max_deviation_spread_size(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on all 5 product limits."""
        if direction == 0: return 0

        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction,
            Product.PICNIC_BASKET2: -direction,
            Product.CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[Product.CROISSANTS] * direction,
            Product.JAMS: -B1B2_THEORETICAL_COMPONENTS[Product.JAMS] * direction,
            Product.DJEMBES: -B1B2_THEORETICAL_COMPONENTS[Product.DJEMBES] * direction,
        }
        max_units = float('inf')

        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue
            current_pos = state.position.get(product, 0)
            limit = self.pos_limits.get(product)
            if limit is None: logger.print(f"Error: Missing pos limit for {product}"); return 0
            if limit == 0: logger.print(f"Warning: Limit for {product} is 0."); return 0

            capacity = (limit - current_pos) if qty_change > 0 else (limit + current_pos)
            if capacity < 0: capacity = 0

            max_units_for_product = capacity // abs(qty_change)
            max_units = min(max_units, max_units_for_product)
            # logger.print(f"  Limit Check {product}: Curr={current_pos}, Lim={limit}, Chg/Unit={qty_change}, Cap={capacity}, MaxUnits={max_units_for_product}")

        final_max = max(0, int(max_units))
        # logger.print(f" Max Deviation Units Calculation (Dir={direction}): Max={final_max}")
        return final_max

    def _calculate_market_liquidity_limit(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on TOTAL market liquidity."""
        if direction == 0: return 0
        order_depths = state.order_depths
        max_units = float('inf')
        qty_changes_per_unit = {
            Product.PICNIC_BASKET1: +direction,
            Product.PICNIC_BASKET2: -direction,
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
                sell_orders = od.sell_orders if isinstance(od.sell_orders, dict) else {}
                if not sell_orders: return 0
                total_available_volume = sum(abs(vol) for vol in sell_orders.values())
            else: # Need to SELL -> sum BUY side volume
                buy_orders = od.buy_orders if isinstance(od.buy_orders, dict) else {}
                if not buy_orders: return 0
                total_available_volume = sum(abs(vol) for vol in buy_orders.values())

            if total_available_volume <= 0: return 0
            units_fillable_for_product = total_available_volume // abs(qty_change)
            max_units = min(max_units, units_fillable_for_product)
            # logger.print(f"  Liq Check {product}: Chg/Unit={qty_change}, AvailVol={total_available_volume}, UnitsFillable={units_fillable_for_product}")

        final_max_liq = max(0, int(max_units))
        # logger.print(f" Max Deviation Units From Liquidity (Dir={direction}): Max={final_max_liq}")
        return final_max_liq

    def _place_aggressive_orders_for_leg(self, product_symbol: Symbol, total_quantity_needed: int, order_depth: OrderDepth, orders_dict: Dict[Symbol, List[Order]]):
        """Places orders for one leg, consuming book levels, appending to orders_dict."""
        if total_quantity_needed == 0: return

        remaining_qty = abs(total_quantity_needed)

        if total_quantity_needed > 0: # Need to BUY
            sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
            if not sell_orders: logger.print(f"Warning: No sell liquidity for {product_symbol} to place BUY order."); return
            sorted_levels = sorted(sell_orders.items())
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_dict[product_symbol].append(Order(product_symbol, int(price), int(qty_at_this_level)))
                    logger.print(f"    => ZScore-BUY {product_symbol} {int(qty_at_this_level)}x{int(price)}")
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break

        else: # Need to SELL
            buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
            if not buy_orders: logger.print(f"Warning: No buy liquidity for {product_symbol} to place SELL order."); return
            sorted_levels = sorted(buy_orders.items(), reverse=True)
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_dict[product_symbol].append(Order(product_symbol, int(price), -int(qty_at_this_level)))
                    logger.print(f"    => ZScore-SELL {product_symbol} {int(qty_at_this_level)}x{int(price)}")
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break
