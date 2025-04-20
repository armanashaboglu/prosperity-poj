SUMMARY OF RULES, MECHANICS AND LINKS TO USEFUL REPOS


What is Prosperity? 

Prosperity is a 15-day long trading challenge happening somewhere in a near - utopian - future. You’re in control of an island in an archipelago and your goal is to bring your island to prosperity. You do so by earning as many SeaShells as possible; the main currency in the archipelago. The more SeaShells you earn, the more your island will prosper. 

During your 15 days on the island, your trading abilities will be tested through a variety of trading challenges. It’s up to you to develop a successful trading strategy. You will be working on a Python script to handle algorithmic trades on your behalf. Every round also presents a manual trading challenge; a one off challenge that is separate from your algorithmic trading and could yield some additional profits. Your success depends on both these algorithmic and manual trades.

Take a deep breath and smell the opportunities drifting by with the ocean breeze. It’s up to you to seize those opportunities and lead your island into prosperity. At times, it will be challenging. Hard even. But for those who know how to work the market to their favor, the reward will be bountiful.

High-level storyline: 

Welcome to Prosperity! You’re in control of an island in an archipelago somewhere in a near (utopian) future. Your goal is as clear as day: bring your island to prosperity. You do so by earning as many SeaShells as possible; the main currency in the archipelago. The more SeaShells you earn, the more your island will grow and prosper.

The main character in the game is a cockatoo that lives in the archipelago. Besides a cockatoo he is also the news anchor for Tropical TV. Every round he’ll give you the latest headlines; all you need to know to progress in the game. The news broadcasts from Tropical TV will play an important role during our 15-day trading challenge. The cockatoo will give a short update on what happened in the previous round, before he’ll disclose this round’s assignment and tradable goods. 

During the trading season, you’ll meet all kinds of interesting characters. Some from within our own archipelago and some that we consider our neighbours. All of them have their own unique story. You should definitely keep your ears peeled for those stories, because every real trading tycoon knows that within every story there is an opportunity to be found. You just have to know how to turn that compelling story into a profitable strategy.


Game mechanics overview: 

# Rounds

The 15 days of simulation of Prosperity are divided into 5 rounds. Each round lasts 72 hours. At the end of every round - before the timer runs out - all teams will have to submit their algorithmic and manual trades to be processed. The algorithms will then participate in a full day of trading against the Prosperity trading bots. Note that all algorithms are trading separately, there is no interaction between the algorithms of different players. When a new round starts, the results of the previous round will be disclosed and the leaderboard will be updated accordingly. During the game, you can always visit previous rounds in the dashboard, to review information and results. But once a round is closed, you can no longer change your submitted trades for that round. When round 5 ends, the final results will be processed and the winner of the Prosperity trading challenge will be announced.

### Algorithmic trading

Every round contains an algorithmic trading challenge. You will have 72 hours to submit your (final) Python program. When the round ends, the last successfully processed submission will be locked in and processed for results.

### Manual trading

All rounds contain a manual trading challenge. Just like the algorithmic challenge, manual trading challenges last 72 hours to submit your (final) trade. When the round ends, the last submission will be locked in and processed for results. During the tutorial round, manual trading is inactive. Note that manual trades have no effect on your algorithmic trade and can be seen as separate challenges to gain additional profits. 

### Round timings

The challenge will start on April 7th, 2025 at 13:00 PM **CET**. There are five rounds and each round lasts 72 hours.

- **Tutorial: February 24, 13:00 → April 7, 13:00**
- **Round 1: April 7, 13:00 → April 10, 13:00**
- **Round 2: April 10, 13:00 → April 13, 13:00**
- **Round 3: April 13, 13:00 → April 16, 13:00**
- **Round 4: April 16, 13:00 → April 19, 13:00**
- **Round 5: April 19, 13:00 → April 22, 13:00**

# Dashboard

## Algorithmic trading submissions

Submitting your algorithmic trading program (your Python script) is easily done through the dashboard, by clicking the “Upload algorithm” button. A window will open where you can drag & drop your file or search for it through a file browser. Here, you can also find all the previously uploaded programs with their respective status and who uploaded them. You can even download the debug logs. Handy!

# Tropical TV

Tropical TV is the archipelago’s daily news broadcast, hosted by a chatty cockatoo. The news broadcasts will keep you up to date with all developments around the archipelago. It contains all the information you need to navigate your way through the trading challenges that are thrown at you. It’s a great way to start your day!

Tropical TV episodes will be available at the beginning of every round. You will be notified as soon as a new Tropical TV episode becomes available.

# Archipelago

The archipelago is where you will find your island amidst the islands of all the other participating teams. The position of your island in the archipelago corresponds to your island’s performance throughout the game. The best-performing island will be at the dead center of the archipelago. The least performing islands will be at the outer rims of the archipelago. The current standing of an island - along with the island name and current amount of SeaShells - is also displayed in the small indicator above the island.

# Leaderboard

Next to the archipelago, a handy leaderboard is available as well. Simply click on the Leaderboard button and you will see the top performing teams and, of course, your own island’s position amongst your competitors. Handy, for if you quickly want to see how many more SeaShells you need to overtake the next team.

Full guide on the algorithmic trading part: 

# The Challenge

For the algorithmic trading challenge, you will be writing and uploading a trading algorithm class in Python, which will then be set loose on the island exchange. On this exchange, the algorithm will trade against a number of bots, with the aim of earning as many SeaShells (the currency of the archipelago) as possible. The algorithmic trading challenge consists of several rounds, that take place on different days of the challenge. At the beginning of each round, it is disclosed which new products will be available for trading on that day. Sample data for these products is provided that players can use to get a better understanding of the price dynamics of these products, and consequently build a better algorithm for trading them. While most days will feature new products, the old products will also still be tradable in the rounds after which they are introduced. This means that based on the result of the previous rounds, players also have the opportunity to analyse and optimise their trading strategies for these “old” products. 

The format for the trading algorithm will be a predefined `Trader` class, which has a single method called `run` which contains all the trading logic coded up by the trader. Once the algorithm is uploaded it will be run in the simulation environment. The simulation consists of a large number of iterations. During each iteration the run method will be called and provided with a `TradingState` object. This object contains an overview of all the trades that have happened since the last iteration, both the algorithms own trades as well as trades that happened between other market participants. Even more importantly, the `TradingState` will contain a per product overview of all the outstanding buy and sell orders (also called “quotes”) originating from the bots. Based on the logic in the `run` method the algorithm can then decide to either send orders that will fully or partially match with the existing orders, e.g. sending a buy (sell) order with a price equal to or higher (lower) than one of the outstanding bot quotes, which will result in a trade. If the algorithm sends a buy (sell) order with an associated quantity that is larger than the bot sell (buy) quote that it is matched to, the remaining quantity will be left as an outstanding buy (sell) quote with which the trading bots will then potentially trade. When the next iteration begins, the `TradingState` will then reveal whether any of the bots decided to “trade on” the player’s outstanding quote. If none of the bots trade on an outstanding player quote, the quote is automatically cancelled at the end of the iteration.

Every trade done by the algorithm in a certain product changes the “position” of the algorithm in that product. E.g. if the initial position in product X was 2 and the algorithm buys an additional quantity of 3, the position in product X is then 5. If the algorithm then subsequently sells a quantity of 7, the position in product X will be -2, called “short 2”. Like in the real world, the algorithms are restricted by per product position limits, which define the absolute position (long or short) that the algorithm is not allowed to exceed. If the aggregated quantity of all the buy (sell) orders an algorithm sends during a certain iteration would, if all fully matched, result in the algorithm obtaining a long (short) position exceeding the position limit, all the orders are cancelled by the exchange.

In the first section, the general outline of the `Trader` class that the player will be creating is outlined.

# Overview of the `Trader` class

Below an abstract representation of what the trader class should look like is shown. The class only requires a single method called `run`, which is called by the simulation every time a new `TraderState` is available. The logic within this `run` method is written by the player and determines the behaviour of the algorithm. The output of the method is a dictionary named `result`, which contains all the orders that the algorithm decides to send based on this logic.

```python
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
```

Example implementation above presents placing order idea as well.

When you send the Trader implementation there is always submission identifier generated. It’s UUID value similar to “59f81e67-f6c6-4254-b61e-39661eac6141”. Should any questions arise on the results, feel free to communicate on Discord channels. Identifier is absolutely essential to answer questions. Please put it in the message.

Technical implementation for the trading container is based on Amazon Web Services Lambda function. Based on the fact that Lambda is stateless AWS can not guarantee any class or global variables will stay in place on subsequent calls. We provide possibility of defining a traderData string value as an opportunity to keep the state details. Any Python variable could be serialised into string with jsonpickle library and deserialised on the next call based on TradingState.traderData property. Container will not interfere with the content. 

To get a better feel for what this `TradingState` object is exactly and how players can use it, a description of the class is provided below.

# Overview of the `TradingState` class

The `TradingState` class holds all the important market information that an algorithm needs to make decisions about which orders to send. Below the definition is provided for the `TradingState` class:

```python
Time = int
Symbol = str
Product = str
Position = int

class TradingState(object):
   def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
```

The most important properties

- own_trades: the trades the algorithm itself has done since the last `TradingState` came in. This property is a dictionary of `Trade` objects with key being a product name. The definition of the `Trade` class is provided in the subsections below.
- market_trades: the trades that other market participants have done since the last `TradingState` came in. This property is also a dictionary of `Trade` objects with key being a product name.
- position: the long or short position that the player holds in every tradable product. This property is a dictionary with the product as the key for which the value is a signed integer denoting the position.
- order_depths: all the buy and sell orders per product that other market participants have sent and that the algorithm is able to trade with. This property is a dict where the keys are the products and the corresponding values are instances of the `OrderDepth` class. This `OrderDepth` class then contains all the buy and sell orders. An overview of the `OrderDepth` class is also provided in the subsections below.

## `Trade` class

Both the own_trades property and the market_trades property provide the traders with a list of trades per products. Every individual trade in each of these lists is an instance of the `Trade` class.

```python
Symbol = str
UserId = str

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None, timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```

These trades have five distinct properties:

1. The symbol/product that the trade corresponds to (i.e. are we exchanging apples or oranges)
2. The price at which the product was exchanged
3. The quantity that was exchanged
4. The identity of the buyer in the transaction
5. The identity of the seller in this transaction

On the island exchange, like on most real-world exchanges, counterparty information is typically not disclosed. Therefore properties 4 and 5 will only be non-empty strings if the algorithm itself is the buyer (4 will be “SUBMISSION”) or the seller (5 will be “SUBMISSION”).

## `OrderDepth` class

Provided by the `TradingState` class is also the `OrderDepth` per symbol. This object contains the collection of all outstanding buy and sell orders, or “quotes” that were sent by the trading bots, for a certain symbol. 

```python
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}
```

All the orders on a single side (buy or sell) are aggregated in a dict, where the keys indicate the price associated with the order, and the corresponding keys indicate the total volume on that price level. For example, if the buy_orders property would look like this for a certain product `{9: 5, 10: 4}` That would mean that there is a total buy order quantity of 5 at the price level of 9, and a total buy order quantity of 4 at a price level of 10. Players should note that in the sell_orders property, the quantities specified will be negative. E.g., `{12: -3, 11: -2}` would mean that the aggregated sell order volume at price level 12 is 3, and 2 at price level 11.

Every price level at which there are buy orders should always be strictly lower than all the levels at which there are sell orders. If not, then there is a potential match between buy and sell orders, and a trade between the bots should have happened.

## `Observation` class

Observation details help to decide on eventual orders or conversion requests. There are two items delivered inside the TradingState instance:

1. Simple product to value dictionary inside plainValueObservations
2. Dictionary of complex **ConversionObservation** values for respective products. Used to place conversion requests from Trader class. Structure visible below.

```python
class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
```

In case you decide to place conversion request on product listed integer number should be returned as “conversions” value from run() method. Based on logic defined inside Prosperity container it will convert positions acquired by submitted code. There is a number of conditions for conversion to  happen:

- You need to obtain either long or short position earlier.
- Conversion request cannot exceed possessed items count.
- In case you have 10 items short (-10) you can only request from 1 to 10. Request for 11 or more will be fully ignored.
- While conversion happens you will need to cover transportation and import/export tariff.
- Conversion request is not mandatory. You can send 0 or None as value.

# How to send orders using the `Order` class

After performing logic on the incoming order state, the `run` method defined by the player should output a dictionary containing the orders that the algorithm wants to send. The keys of this dictionary should be all the products that the algorithm wishes to send orders for. These orders should be instances of the `Order` class. Each order has three important properties. These are:

1. The symbol of the product for which the order is sent.
2. The price of the order: the maximum price at which the algorithm wants to buy in case of a BUY order, or the minimum price at which the algorithm wants to sell in case of a SELL order.
3. The quantity of the order: the maximum quantity that the algorithm wishes to buy or sell. If the sign of the quantity is positive, the order is a buy order, if the sign of the quantity is negative it is a sell order.

```python
Symbol = str

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```

If there are active orders from counterparties for the same product against which the algorithms’ orders can be matched, the algorithms’ order will be (partially) executed right away. If no immediate or partial execution is possible, the remaining order quantity will be visible for the bots in the market, and it might be that one of them sees it as a good trading opportunity and will trade against it. If none of the bots decides to trade against the remaining order quantity, it is cancelled. Note that after cancellation of the algorithm’s orders but before the next `Tradingstate` comes in, bots might also trade with each other.

Note that on the island exchange players’ execution is infinitely fast, which means that all their orders arrive in the exchange matching engine without any delay. Therefore, all the orders that a player sends that can be immediately matched with an order from one of the bots, will be matched and result in a trade. In other words, none of the bots can send an order that is faster than the player’s order and get the opportunity instead.


## Position Limits

Just like in the real world of trading, there are position limits, i.e. limits to the size of the position that the algorithm can trade into in a single product. These position limits are defined on a per-product basis, and refer to the absolute allowable position size. So for a hypothetical position limit of 10, the position can neither be greater than 10 (long) nor less than -10 (short). On the Prosperity Island exchange, this position limit is enforced by the exchange. If at any iteration, the player’s algorithm tries to send buy (sell) orders for a product with an aggregated quantity that would cause the player to go over long (short) position limits if all orders would be fully executed, all orders will be rejected automatically. For example, the position limit in product X is 30 and the current position is -5, then any aggregated buy order volume exceeding 30 - (-5) = 35 would result in an order rejection.

For an overview of the per-product position limit players are referred to the ‘Rounds’ section on [Prosperity 3 Wiki](https://www.notion.so/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4?pvs=21).

Below two example iterations are provided to give an idea what the simulation behaviour looks like.


github repos of high-performing competitors in Prosperity 2 and Prosperity 1.

@https://github.com/jmerle/imc-prosperity-2?tab=readme-ov-file (currently this is what our strategy is based on)

@https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal (Prosperity 1, 2nd place)

@https://github.com/ericcccsliu/imc-prosperity-2/tree/main (Prosperity 2, 2n place)

Someone also created a backtester tool so we can locally test our algorithms before uploading it to the website. 
@https://github.com/jmerle/imc-prosperity-3-backtester/tree/master 

Every time we upload our algorithm to the website or locally backtest, we get a .log file that contains sandbox logs , activities log which show the LOB for each product at every timestamp, and trade history which contains every trade happened in each timestamp.

There is also a visualizer tool made by someone which is open source. It works by just uploading the log file to the visualiser website and it generates various graphs.

visualiser tool: @https://jmerle.github.io/imc-prosperity-3-visualizer/ 

ROUND 0 (tutorial):
In the tutorial round there are two tradable goods: `Rainforest Resin` and `Kelp`. While the value of the `Rainforest Resin` has been stable throughout the history of the archipelago, the value of `Kelp` has been going up and down over time. 

Position limits for the newly introduced products:

- `RAINFOREST_RESIN`: 50
- `KELP`: 50

ROUND 1:
## Algorithm challenge

The first three tradable products are introduced: : `Rainforest Resin` , `Kelp`, and `Squid Ink`. The value of the `Rainforest Resin` has been stable throughout the history of the archipelago, the value of `Kelp` has been going up and down over time, and the value of `Squid Ink` can also swing a bit, but some say there is a pattern to be discovered in its prize progression. All algorithms uploaded in the tutorial round will be processed and generate results instantly, so you can experiment with different programs and strategies.

Position limits for the newly introduced products:

- `RAINFOREST_RESIN`: 50
- `KELP`: 50
- `SQUID_INK`: 50

## Manual challenge

You get the chance to do a series of trades in some foreign island currencies. The first trade is a conversion of your SeaShells into a foreign currency, the last trade is a conversion from a foreign currency into SeaShells. Everything in between is up to you. Give some thought to what series of trades you would like to do, as there might be an opportunity to walk away with more shells than you arrived with.

ROUND 2:

## Algorithm challenge

In this second round, you’ll find that everybody on the archipelago loves to picnic. Therefore, in addition to the products from round one, two Picnic Baskets are now available as a tradable good. 

`PICNIC_BASKET1` contains three products: 

1. Six (6) `CROISSANTS`
2. Three (3) `JAMS`
3. One (1) `DJEMBE`

`PICNIC_BASKET2` contains just two products: 

1. Four (4) `CROISSANTS`
2. Two (2) `JAMS`

Aside from the Picnic Baskets, you can now also trade the three products individually on the island exchange. 

Position limits for the newly introduced products:

- `CROISSANT`: 250
- `JAM`: 350
- `DJEMBE`: 60
- `PICNIC_BASKET1`: 60
- `PICNIC_BASKET2`: 100

## Manual challenge

Some shipping containers with valuables inside washed ashore. You get to choose a maximum of two containers to open and receive the valuable contents from. The first container you open is free of charge, but for the second one you will have to pay some SeaShells. Keep in mind that you are not the only one choosing containers and making a claim on its contents. You will have to split the spoils with all others that choose the same container. So, choose carefully. 

Here's a breakdown of how your profit from a container will be computed:
Every container has its **treasure multiplier** (up to 90) and number of **inhabitants** (up to 10) that will be choosing that particular container. The container’s total treasure is the product of the **base treasure** (10 000, same for all containers) and the container’s specific treasure multiplier. However, the resulting amount is then divided by the sum of the inhabitants that choose the same container and the percentage of opening this specific container of the total number of times a container has been opened (by all players). 

For example, if **5 inhabitants** choose a container, and **this container was chosen** **10% of the total number of times a container has been opened** (by all players), the prize you get from that container will be divided by 15. After the division, **costs for opening a container** apply (if there are any), and profit is what remains.

ROUND 3:

## Algorithm challenge

Our inhabitants really like volcanic rock. So much even, that they invented a new tradable good, `VOLCANIC ROCK VOUCHERS`. The vouchers will give you the right to buy `VOLCANIC ROCK` at a certain price by the end of the round and can be traded as a separate item on the island’s exchange. Of course you will have to pay a premium for these coupons, but if your strategy is solid as a rock, SeaShells spoils will be waiting for you on the horizon. 

There are five Volcanic Rock Vouchers, each with their own **strike price** and **premium.** 

**At beginning of Round 1, all the Vouchers have 7 trading days to expire. By end of Round 5, vouchers will have 2 trading days left to expire.**

Position limits for the newly introduced products:

- `VOLCANIC_ROCK`: 400

`VOLCANIC_ROCK_VOUCHER_9500` :

- Position Limit: 200
- Strike Price: 9,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_9750` :

- Position Limit: 200
- Strike Price: 9,750 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10000` :

- Position Limit: 200
- Strike Price: 10,000 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10250` :

- Position Limit: 200
- Strike Price: 10,250 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10500` :

- Position Limit: 200
- Strike Price: 10,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

## Manual challenge

A big group of Sea Turtles is visiting our shores, bringing with them an opportunity to acquire some top grade `FLIPPERS`. You only have two chances to offer a good price. Each one of the Sea Turtles will accept the lowest bid that is over their reserve price. 

The distribution of reserve prices is uniform between 160–200 and 250–320, but none of the Sea Turtles will trade between 200 and 250 due to some ancient superstition.

For your second bid, they also take into account the average of the second bids by other traders in the archipelago. They’ll trade with you when your offer is above the average of all second bids. But if you end up under the average, the probability of a deal decreases rapidly. 

To simulate this probability, the PNL obtained from trading with a fish for which your second bid is under the average of all second bids will be scaled by a factor *p*:

$$
p = (\frac{320 – \text{average bid}}{320 – \text{your bid}})^3
$$

You know there’s a constant desire for Flippers on the archipelago. So, at the end of the round, you’ll be able to sell them for 320 SeaShells ****a piece.

Think hard about how you want to set your two bids, place your feet firmly in the sand and brace yourself, because this could get messy.

ROUND 4:

## Algorithm challenge

In this fourth round of Prosperity a new luxury product is introduced: `MAGNIFICENT MACARONS`. `MAGNIFICENT MACARONS` are a delicacy and their value is dependent on all sorts of observable factors like hours of sun light, sugar prices, shipping costs, in- & export tariffs and suitable storage space. Can you find the right connections to optimize your program? 

Position limits for the newly introduced products:

- `MAGNIFICENT_MACARONS`: 75
- Conversion Limit for `MAGNIFICENT_MACARONS` = 10

## Additional trading microstructure information:

1. ConversionObservation (detailed in “[Writing an Algorithm in Python](https://www.notion.so/17be8453a09381988c6ed45b1b597049?pvs=21)” under E-learning center) shows quotes of `MAGNIFICENT_MACARONS` offered by the chefs from Pristine Cuisine
2. To purchase 1 unit of `MAGNIFICENT_MACARONS` from Pristine Cuisine, you will purchase at askPrice, pay `TRANSPORT_FEES` and `IMPORT_TARIFF`
3. To sell 1 unit of `MAGNIFICENT_MACARONS` to Pristine Cuisine, you will sell at bidPrice, pay `TRANSPORT_FEES` and `EXPORT_TARIFF`
4. You can ONLY trade with Pristine Cuisine via the conversion request with applicable conditions as mentioned in the wiki
5. For every 1 unit of `MAGNIFICENT_MACARONS` net long position, storage cost of 0.1 Seashells per timestamp will be applied for the duration that position is held. No storage cost applicable to net short position

ROUND 5:

## Algorithm challenge

The final round of the challenge is already here! And surprise, no new products are introduced for a change. Dull? Probably not, as you do get another treat. The island exchange now discloses to you who the counterparty is you have traded against. This means that the counter_party property of the OwnTrade object is now populated. Perhaps interesting to see if you can leverage this information to make your algorithm even more profitable?

```python
class OwnTrade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, counter_party: UserId = None) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.counter_party = counter_party


