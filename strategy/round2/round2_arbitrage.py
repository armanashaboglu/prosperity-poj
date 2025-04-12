import json
import math
from typing import Any, Dict, List
from collections import defaultdict

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
                observation.sugarPrice,
                observation.sunlightIndex,
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
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"

POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.BASKET1: 60,
    Product.BASKET2: 100
}

class Trader:
    def __init__(self):
        logger.print("round2_arbitrage Trader Initialized (with corrected product names).")

        # Store position limits
        self.pos_limits = POSITION_LIMITS

        # A threshold for difference that triggers a trade
<<<<<<< HEAD
        self.diff_threshold = 200  # Adjust as needed
        self.diff_threshold_basket2 = 120 # Separate threshold for Basket 2
=======
        self.diff_threshold = 30  # Adjust as needed
>>>>>>> 8cc6356232fd44f77f14673590fa4d755781b603

    def run(self, state: TradingState):
        """
        Called every iteration with updated TradingState.
        Returns (dict_of_orders, conversions, traderData_str).
        """
        logger.print(f"--- round2_arbitrage Strategy | Timestamp: {state.timestamp} ---")

        from collections import defaultdict
        orders = defaultdict(list)
        conversions = 0
        trader_data_out = "{}"  # If we want to store any persistent state

        # 1) Current positions
        positions = state.position

        def get_pos(prod: str) -> int:
            return positions.get(prod, 0)

        # 2) Extract best bid/ask => mid_price for each relevant product
        relevant_products = [
            Product.CROISSANTS,
            Product.JAMS,
            Product.DJEMBES,
            Product.BASKET1,
            Product.BASKET2
        ]

        best_bid = {}
        best_ask = {}
        mid_price = {}

        for prod in relevant_products:
            od = state.order_depths.get(prod)
            if od is None:
                continue  # no data
            if od.buy_orders:
                best_bid[prod] = max(od.buy_orders.keys())
            else:
                best_bid[prod] = None
            if od.sell_orders:
                best_ask[prod] = min(od.sell_orders.keys())
            else:
                best_ask[prod] = None

            if best_bid[prod] is not None and best_ask[prod] is not None:
                mid_price[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
            elif best_bid[prod] is not None:
                mid_price[prod] = best_bid[prod]
            elif best_ask[prod] is not None:
                mid_price[prod] = best_ask[prod]
            else:
                mid_price[prod] = None

        # 3) Check synergy for Basket1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBES
        c = mid_price.get(Product.CROISSANTS, None)
        j = mid_price.get(Product.JAMS, None)
        d = mid_price.get(Product.DJEMBES, None)
        b1 = mid_price.get(Product.BASKET1, None)
        b2 = mid_price.get(Product.BASKET2, None)

        if (c is not None) and (j is not None) and (d is not None) and (b1 is not None):
            fair_b1 = 6*c + 3*j + 1*d
            diff1 = b1 - fair_b1
            logger.print(f"Basket1 synergy: B1_mid={b1:.1f}, sum_components={fair_b1:.1f}, diff={diff1:.1f}")

            if diff1 > self.diff_threshold:
                # Overpriced basket => SELL basket, BUY items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET1, basket_side=-1,
                    components_sides={
                        Product.CROISSANTS: +6,
                        Product.JAMS: +3,
                        Product.DJEMBES: +1
                    }
                )
<<<<<<< HEAD
            elif diff1 < -self.diff_threshold + 50:
=======
            elif diff1 < -self.diff_threshold:
>>>>>>> 8cc6356232fd44f77f14673590fa4d755781b603
                # Underpriced => BUY basket, SELL items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET1, basket_side=+1,
                    components_sides={
                        Product.CROISSANTS: -6,
                        Product.JAMS: -3,
                        Product.DJEMBES: -1
                    }
                )

        # 4) Check synergy for Basket2 = 4 CROISSANTS + 2 JAMS
        if (c is not None) and (j is not None) and (b2 is not None):
            fair_b2 = 4*c + 2*j
            diff2 = b2 - fair_b2
            logger.print(f"Basket2 synergy: B2_mid={b2:.1f}, sum_components={fair_b2:.1f}, diff={diff2:.1f}")

<<<<<<< HEAD
            if diff2 > self.diff_threshold_basket2:
=======
            if diff2 > self.diff_threshold:
>>>>>>> 8cc6356232fd44f77f14673590fa4d755781b603
                # Overpriced => SELL basket2, BUY items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET2, basket_side=-1,
                    components_sides={
                        Product.CROISSANTS: +4,
                        Product.JAMS: +2
                    }
                )
<<<<<<< HEAD
            elif diff2 < -self.diff_threshold_basket2 + 30:   # meani ekledim
=======
            elif diff2 < -self.diff_threshold:
>>>>>>> 8cc6356232fd44f77f14673590fa4d755781b603
                # Underpriced => BUY basket2, SELL items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET2, basket_side=+1,
                    components_sides={
                        Product.CROISSANTS: -4,
                        Product.JAMS: -2
                    }
                )

        # Convert orders => normal dict
        final_orders = {}
        for k, v in orders.items():
            final_orders[k] = v

        logger.flush(state, final_orders, conversions, trader_data_out)
        return (final_orders, conversions, trader_data_out)

    def execute_spread_trade(
        self,
        orders_dict: Dict[str, List[Order]],
        state: TradingState,
        basket_symbol: str,
        basket_side: int,
        components_sides: Dict[str, int]
    ):
        """
        Attempt a 1-basket trade, plus the corresponding component trades.
        basket_side = +1 means BUY the basket, -1 means SELL the basket
        components_sides = dict of {component_symbol: +X or -X} for each 1 basket
        """
        # 1) Check capacity on basket
        current_basket_pos = state.position.get(basket_symbol, 0)
        limit_basket = self.pos_limits[basket_symbol]

        if basket_side > 0:
            # Buying the basket => max we can buy is limit_basket - current_basket_pos
            capacity_basket = limit_basket - current_basket_pos
        else:
            # Selling => max is limit_basket + current_basket_pos
            capacity_basket = limit_basket + current_basket_pos

        if capacity_basket <= 0:
            logger.print(f"No capacity to trade basket {basket_symbol} side={basket_side}. Skipping.")
            return

<<<<<<< HEAD
        trade_basket_qty = min(5, capacity_basket)  # do a 1-lot if possible
=======
        trade_basket_qty = min(1, capacity_basket)  # do a 1-lot if possible
>>>>>>> 8cc6356232fd44f77f14673590fa4d755781b603

        # 2) Check each component capacity likewise
        for comp_symbol, comp_side in components_sides.items():
            comp_pos = state.position.get(comp_symbol, 0)
            limit_comp = self.pos_limits[comp_symbol]

            if comp_side > 0:
                # means buy comp_side * trade_basket_qty
                capacity_comp = limit_comp - comp_pos
            else:
                # means sell
                capacity_comp = limit_comp + comp_pos

            needed = abs(comp_side) * trade_basket_qty
            if capacity_comp < needed:
                logger.print(f"No capacity for {comp_symbol} side={comp_side} * {trade_basket_qty}. Skipping entire spread.")
                return

        logger.print(f"ArbTrade: 1-lot => {basket_symbol} side={basket_side}, items={components_sides}")

        # 3) Place the basket order near best bid/ask
        od_basket = state.order_depths.get(basket_symbol)
        if not od_basket:
            logger.print(f"No order depth for {basket_symbol}. Cannot trade it.")
            return

        if basket_side > 0:
            # buy at best ask or best ask+1
            if od_basket.sell_orders:
                best_ask = min(od_basket.sell_orders.keys())
                price = best_ask + 1
                orders_dict[basket_symbol].append(Order(basket_symbol, price, trade_basket_qty))
                logger.print(f" -> BUY {basket_symbol} {trade_basket_qty}x{price}")
        else:
            # sell at best bid or best bid-1
            if od_basket.buy_orders:
                best_bid = max(od_basket.buy_orders.keys())
                price = best_bid - 1
                orders_dict[basket_symbol].append(Order(basket_symbol, price, -trade_basket_qty))
                logger.print(f" -> SELL {basket_symbol} {trade_basket_qty}x{price}")

        # 4) Place the component orders in opposite direction
        for comp_symbol, comp_side in components_sides.items():
            full_qty = comp_side * trade_basket_qty
            if full_qty == 0:
                continue

            od_comp = state.order_depths.get(comp_symbol)
            if not od_comp:
                logger.print(f"No order depth for {comp_symbol}. Skipping comp side.")
                continue

            if full_qty > 0:
                # buy
                if od_comp.sell_orders:
                    best_ask = min(od_comp.sell_orders.keys())
                    price = best_ask + 1
                    orders_dict[comp_symbol].append(Order(comp_symbol, price, full_qty))
                    logger.print(f" -> BUY {comp_symbol} {full_qty}x{price}")
            else:
                # sell
                if od_comp.buy_orders:
                    best_bid = max(od_comp.buy_orders.keys())
                    price = best_bid - 1
                    orders_dict[comp_symbol].append(Order(comp_symbol, price, full_qty))
                    logger.print(f" -> SELL {comp_symbol} {abs(full_qty)}x{price}")
