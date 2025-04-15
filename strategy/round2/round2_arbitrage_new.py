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

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        base_json = self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ])
        base_len = len(base_json)
        max_item_length = (self.max_log_length - base_len)//3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length)
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations)
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        arr = []
        for lst in listings.values():
            arr.append([lst.symbol, lst.product, lst.denomination])
        return arr

    def compress_order_depths(self, od_map: Dict[Symbol, OrderDepth]) -> Dict[str, List[Any]]:
        c = {}
        for sym, od in od_map.items():
            c[sym] = [od.buy_orders, od.sell_orders]
        return c

    def compress_trades(self, trades_map: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        arr = []
        for tlist in trades_map.values():
            for t in tlist:
                arr.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return arr

    def compress_observations(self, obs: Observation) -> List[Any]:
        conv_obs = {}
        for product, ob in obs.conversionObservations.items():
            conv_obs[product] = [
                ob.bidPrice, ob.askPrice,
                ob.transportFees, ob.exportTariff,
                ob.importTariff, ob.sugarPrice, ob.sunlightIndex
            ]
        return [obs.plainValueObservations, conv_obs]

    def compress_orders(self, orders_map: Dict[str, List[Order]]) -> List[List[Any]]:
        arr = []
        for sym_list in orders_map.values():
            for o in sym_list:
                arr.append([o.symbol, o.price, o.quantity])
        return arr

    def to_json(self, val: Any) -> str:
        return json.dumps(val, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, s: str, maxlen: int) -> str:
        if len(s)<=maxlen:
            return s
        return s[:maxlen-3]+"..."

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
        logger.print("round2_arbitrage extended with cross-basket synergy (composition).")

        self.pos_limits = POSITION_LIMITS

        # synergy thresholds for B1 & B2 vs items
        self.diff_threshold_b1 = 200
        self.diff_threshold_b2 = 120

        # synergy threshold for B1 vs B2 with composition
        # i.e. B1 minus (B2 + 2C + 1J + 1D)
        self.diff_threshold_b1_b2 = 50

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        logger.print(f"--- synergy with cross composition | Timestamp: {state.timestamp} ---")

        from collections import defaultdict
        orders = defaultdict(list)
        conversions = 0
        trader_data_out = "{}"

        positions = state.position

        def get_pos(prod):
            return positions.get(prod, 0)

        # 1) best bid/ask => mid price for relevant
        relevant = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.BASKET1, Product.BASKET2]
        best_bid = {}
        best_ask = {}
        mid_price = {}
        for r in relevant:
            od = state.order_depths.get(r)
            if od:
                if od.buy_orders:
                    best_bid[r] = max(od.buy_orders.keys())
                else:
                    best_bid[r] = None

                if od.sell_orders:
                    best_ask[r] = min(od.sell_orders.keys())
                else:
                    best_ask[r] = None

                if best_bid[r] is not None and best_ask[r] is not None:
                    mid_price[r] = 0.5*(best_bid[r]+ best_ask[r])
                elif best_bid[r] is not None:
                    mid_price[r] = best_bid[r]
                elif best_ask[r] is not None:
                    mid_price[r] = best_ask[r]
                else:
                    mid_price[r] = None
            else:
                best_bid[r] = None
                best_ask[r] = None
                mid_price[r] = None

        # 2) synergy B1 vs items
        c = mid_price.get(Product.CROISSANTS)
        j = mid_price.get(Product.JAMS)
        d = mid_price.get(Product.DJEMBES)
        b1 = mid_price.get(Product.BASKET1)
        b2 = mid_price.get(Product.BASKET2)

        if (c is not None) and (j is not None) and (d is not None) and (b1 is not None):
            fair_b1 = 6*c + 3*j + 1*d
            diff1 = b1 - fair_b1
            logger.print(f"BASKET1 synergy: B1={b1:.1f}, sum={fair_b1:.1f}, diff={diff1:.1f}")
            if diff1> self.diff_threshold_b1:
                # overpriced => SELL B1, BUY items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET1, basket_side=-1,
                    components_sides={Product.CROISSANTS:+6, Product.JAMS:+3, Product.DJEMBES:+1}
                )
            elif diff1< -self.diff_threshold_b1+50:
                # under => BUY B1, SELL items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET1, basket_side=+1,
                    components_sides={Product.CROISSANTS:-6, Product.JAMS:-3, Product.DJEMBES:-1}
                )

        # 3) synergy B2 vs items
        if (c is not None) and (j is not None) and (b2 is not None):
            fair_b2 = 4*c + 2*j
            diff2 = b2 - fair_b2
            logger.print(f"BASKET2 synergy: B2={b2:.1f}, sum={fair_b2:.1f}, diff={diff2:.1f}")
            if diff2> self.diff_threshold_b2:
                # overpriced => SELL B2, BUY items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET2, basket_side=-1,
                    components_sides={Product.CROISSANTS:+4, Product.JAMS:+2}
                )
            elif diff2< -self.diff_threshold_b2+30:
                # under => BUY B2, SELL items
                self.execute_spread_trade(
                    orders, state,
                    basket_symbol=Product.BASKET2, basket_side=+1,
                    components_sides={Product.CROISSANTS:-4, Product.JAMS:-2}
                )

        # 4) synergy B1 vs B2 with composition difference
        # B1 - B2 = 2C + 1J + 1D
        # So the "theoretical" price of B1 ~ B2 + cost(2C + 1J + 1D).
        # We'll compare B1's mid to B2 + 2*C +1*J +1*D
        if (b1 is not None) and (b2 is not None) and (c is not None) and (j is not None) and (d is not None):
            # implied_B1_from_B2 = b2 + 2c + 1j + 1d
            implied_b1_from_b2 = b2 + 2*c + 1*j + 1*d
            diff_comp = b1 - implied_b1_from_b2
            logger.print(f"B1 vs B2+2C+1J+1D synergy: B1={b1:.1f}, B2plus={implied_b1_from_b2:.1f}, diff={diff_comp:.1f}")

            if diff_comp> self.diff_threshold_b1_b2:
                # B1 overpriced => short 1 B1, buy 1 B2, buy 2C, 1J, 1D
                self.execute_b1_vs_b2_composition(
                    orders, state,
                    side_b1=-1, side_b2=+1,   # short B1, buy B2
                    side_c=+2, side_j=+1, side_d=+1
                )
            elif diff_comp< -self.diff_threshold_b1_b2:
                # B2+2C+1J+1D overpriced => short B2+2C+1J+1D, buy B1
                self.execute_b1_vs_b2_composition(
                    orders, state,
                    side_b1=+1, side_b2=-1,   # short B2
                    side_c=-2, side_j=-1, side_d=-1
                )

        final_orders = {}
        for sym, ord_list in orders.items():
            final_orders[sym] = ord_list

        logger.flush(state, final_orders, conversions, trader_data_out)
        return final_orders, conversions, trader_data_out

    #######################################
    # synergy for basket vs items
    #######################################
    def execute_spread_trade(
        self,
        orders_dict: Dict[str, List[Order]],
        state: TradingState,
        basket_symbol: str,
        basket_side: int,
        components_sides: Dict[str,int],
        max_single_lot: int=5
    ):
        """
        Copied from your round2_arbitrage approach: 
        Do a single-lot synergy if there's capacity for basket + items.
        """
        current_basket_pos = state.position.get(basket_symbol, 0)
        limit_basket = self.pos_limits[basket_symbol]

        if basket_side>0:
            capacity_basket = limit_basket - current_basket_pos
        else:
            capacity_basket = limit_basket + current_basket_pos

        if capacity_basket<=0:
            logger.print(f"No capacity for synergy on {basket_symbol}, side={basket_side}. Skipping.")
            return

        trade_basket_qty = min(max_single_lot, capacity_basket)
        if trade_basket_qty<=0:
            return

        # check each item capacity
        for comp_sym, comp_side in components_sides.items():
            comp_pos = state.position.get(comp_sym, 0)
            limit_comp = self.pos_limits[comp_sym]
            needed = abs(comp_side)* trade_basket_qty
            if comp_side>0:
                capacity_comp = limit_comp - comp_pos
            else:
                capacity_comp = limit_comp + comp_pos
            if capacity_comp< needed:
                logger.print(f"No item capacity for synergy => {comp_sym}, need={needed}, have={capacity_comp}. skip.")
                return

        # place the basket order
        od_basket = state.order_depths.get(basket_symbol)
        if not od_basket:
            logger.print(f"No order depth for {basket_symbol}. skip synergy.")
            return

        if basket_side>0:
            if od_basket.sell_orders:
                best_ask = min(od_basket.sell_orders.keys())
                px = best_ask+1
                orders_dict[basket_symbol].append(Order(basket_symbol, px, trade_basket_qty))
                logger.print(f" => BUY {basket_symbol} {trade_basket_qty}x{px}")
        else:
            if od_basket.buy_orders:
                best_bid = max(od_basket.buy_orders.keys())
                px = best_bid-1
                orders_dict[basket_symbol].append(Order(basket_symbol, px, -trade_basket_qty))
                logger.print(f" => SELL {basket_symbol} {trade_basket_qty}x{px}")

        # place items in opposite direction
        for comp_sym, comp_side in components_sides.items():
            full_qty = comp_side* trade_basket_qty
            if full_qty==0:
                continue
            od_comp = state.order_depths.get(comp_sym)
            if not od_comp:
                logger.print(f"No depth for {comp_sym}, skip synergy side.")
                continue
            if full_qty>0:
                # buy
                if od_comp.sell_orders:
                    best_ask = min(od_comp.sell_orders.keys())
                    px = best_ask+1
                    orders_dict[comp_sym].append(Order(comp_sym, px, full_qty))
                    logger.print(f" => BUY {comp_sym} {full_qty}x{px}")
            else:
                # sell
                if od_comp.buy_orders:
                    best_bid = max(od_comp.buy_orders.keys())
                    px = best_bid-1
                    orders_dict[comp_sym].append(Order(comp_sym, px, full_qty))
                    logger.print(f" => SELL {comp_sym} {abs(full_qty)}x{px}")

    #######################################
    # synergy for basket1 vs basket2 with item difference
    #######################################
    def execute_b1_vs_b2_composition(
        self,
        orders_dict: Dict[str,List[Order]],
        state: TradingState,
        side_b1: int,
        side_b2: int,
        side_c: int,
        side_j: int,
        side_d: int,
        max_single_lot: int=5
    ):
        """
        We do a 1-lot synergy if there's capacity. 
        'side_b1' is +1 means buy B1, -1 means sell B1, etc.
        Also for the difference items: side_c, side_j, side_d.
        We'll check capacity for B1, B2, Croissants, Jams, Djembes. 
        Then place the trades at best quotes +/- 1.

        ex: If B1 is overpriced => side_b1=-1, side_b2=+1, side_c=+2, side_j=+1, side_d=+1
        """
        # capacity for B1
        b1_pos = state.position.get(Product.BASKET1,0)
        b1_lim = self.pos_limits[Product.BASKET1]
        if side_b1>0:
            cap_b1 = b1_lim - b1_pos
        else:
            cap_b1 = b1_lim + b1_pos

        # capacity for B2
        b2_pos = state.position.get(Product.BASKET2,0)
        b2_lim = self.pos_limits[Product.BASKET2]
        if side_b2>0:
            cap_b2 = b2_lim - b2_pos
        else:
            cap_b2 = b2_lim + b2_pos

        # capacity for Croissants
        c_pos = state.position.get(Product.CROISSANTS,0)
        c_lim = self.pos_limits[Product.CROISSANTS]
        if side_c>0:
            cap_c = c_lim - c_pos
        else:
            cap_c = c_lim + c_pos

        # capacity for Jams
        j_pos = state.position.get(Product.JAMS,0)
        j_lim = self.pos_limits[Product.JAMS]
        if side_j>0:
            cap_j = j_lim - j_pos
        else:
            cap_j = j_lim + j_pos

        # capacity for Djembes
        d_pos = state.position.get(Product.DJEMBES,0)
        d_lim = self.pos_limits[Product.DJEMBES]
        if side_d>0:
            cap_d = d_lim - d_pos
        else:
            cap_d = d_lim + d_pos

        # see how many "lots" we can do
        # each lot = 1 B1 (side_b1) + 1 B2 (side_b2) + 2/1/1 items
        # The code above is for a single-lot synergy. We'll do at most 'max_single_lot' if capacity allows
        # but each is 1 "basket-lot" so we require min( cap_B1, cap_B2, etc ) across all needed items.

        # if side_c != 0 => each synergy-lot needs abs(side_c) capacity in Croissants
        # so the synergy-lot needed for c = abs(side_c)
        # capacity = floor(cap_c / abs(side_c))

        # We'll define a helper for each:
        def lot_capacity(side_val, cap_val):
            if side_val==0:
                return float('inf')
            need = abs(side_val)
            return cap_val//need

        possible_b1 = lot_capacity(side_b1, cap_b1)
        possible_b2 = lot_capacity(side_b2, cap_b2)
        possible_c = lot_capacity(side_c, cap_c)
        possible_j = lot_capacity(side_j, cap_j)
        possible_d = lot_capacity(side_d, cap_d)

        total_capacity = min(possible_b1, possible_b2, possible_c, possible_j, possible_d, max_single_lot)
        if total_capacity<=0:
            logger.print(f"No capacity for B1 vs B2 composition synergy. skipping.")
            return

        # place 1-lot synergy
        # We'll do 'total_capacity' times if you want multiple-lot, or just do 1-lot. Up to you.
        # We'll do just 1-lot for clarity
        synergy_lots = 1  
        # To do multiple-lots, you'd do a for-loop up to total_capacity

        # place B1
        od_b1 = state.order_depths.get(Product.BASKET1)
        if not od_b1:
            logger.print("No depth for BASKET1, skip synergy B1-B2 comp.")
            return
        if side_b1>0:
            # buy B1 => best ask
            if od_b1.sell_orders:
                b1_ask = min(od_b1.sell_orders.keys())
                px= b1_ask+1
                orders_dict[Product.BASKET1].append(Order(Product.BASKET1, px, synergy_lots))
                logger.print(f" => BUY {synergy_lots}x BASKET1 @ {px}")
        else:
            # sell B1 => best bid
            if od_b1.buy_orders:
                b1_bid = max(od_b1.buy_orders.keys())
                px= b1_bid-1
                orders_dict[Product.BASKET1].append(Order(Product.BASKET1, px, -synergy_lots))
                logger.print(f" => SELL {synergy_lots}x BASKET1 @ {px}")

        # place B2
        od_b2 = state.order_depths.get(Product.BASKET2)
        if not od_b2:
            logger.print("No depth for BASKET2, skip synergy B1-B2 comp.")
            return
        if side_b2>0:
            if od_b2.sell_orders:
                b2_ask = min(od_b2.sell_orders.keys())
                px= b2_ask+1
                orders_dict[Product.BASKET2].append(Order(Product.BASKET2, px, synergy_lots))
                logger.print(f" => BUY {synergy_lots}x BASKET2 @ {px}")
        else:
            if od_b2.buy_orders:
                b2_bid = max(od_b2.buy_orders.keys())
                px= b2_bid-1
                orders_dict[Product.BASKET2].append(Order(Product.BASKET2, px, -synergy_lots))
                logger.print(f" => SELL {synergy_lots}x BASKET2 @ {px}")

        # place items
        def place_item(sym, side_val, lots):
            if side_val==0:
                return
            odx = state.order_depths.get(sym)
            if not odx:
                logger.print(f"No depth for {sym}, skip synergy item side.")
                return
            full_qty= side_val* lots
            if full_qty>0:
                # buy
                if odx.sell_orders:
                    askp= min(odx.sell_orders.keys())
                    px= askp+1
                    orders_dict[sym].append(Order(sym, px, full_qty))
                    logger.print(f" => BUY {sym} {full_qty}x{px}")
            else:
                # sell
                if odx.buy_orders:
                    bidp= max(odx.buy_orders.keys())
                    px= bidp-1
                    orders_dict[sym].append(Order(sym, px, full_qty))
                    logger.print(f" => SELL {sym} {abs(full_qty)}x{px}")

        place_item(Product.CROISSANTS, side_c, synergy_lots)
        place_item(Product.JAMS, side_j, synergy_lots)
        place_item(Product.DJEMBES, side_d, synergy_lots)
