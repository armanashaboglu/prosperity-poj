-EXPLANATION OF CLASSES IN DATAMODEL-

TradingState: 
The TradingState class holds all the important market information that an algorithm needs to make decisions about which orders to send.
The most important properties

- own_trades: the trades the algorithm itself has done since the last `TradingState` came in. This property is a dictionary of `Trade` objects with key being a product name. The definition of the `Trade` class is provided in the subsections below.
- market_trades: the trades that other market participants have done since the last `TradingState` came in. This property is also a dictionary of `Trade` objects with key being a product name.
- position: the long or short position that the player holds in every tradable product. This property is a dictionary with the product as the key for which the value is a signed integer denoting the position.
- order_depths: all the buy and sell orders per product that other market participants have sent and that the algorithm is able to trade with. This property is a dict where the keys are the products and the corresponding values are instances of the `OrderDepth` class.
- This `OrderDepth` class then contains all the buy and sell orders. An overview of the `OrderDepth` class is also provided in the subsections below.

OrderDepth:
Provided by the TradingState class is also the OrderDepth per symbol. This object contains the collection of all outstanding buy and sell orders, or “quotes” that were sent by the trading bots, for a certain symbol.
All the orders on a single side (buy or sell) are aggregated in a dict, where the keys indicate the price associated with the order, and the corresponding keys indicate the total volume on that price level. 
For example, if the buy_orders property would look like this for a certain product `{9: 5, 10: 4}` That would mean that there is a total buy order quantity of 5 at the price level of 9, and a total buy order quantity of 4 at a price level of 10. 
Players should note that in the sell_orders property, the quantities specified will be negative. E.g., `{12: -3, 11: -2}` would mean that the aggregated sell order volume at price level 12 is 3, and 2 at price level 11.
Every price level at which there are buy orders should always be strictly lower than all the levels at which there are sell orders. If not, then there is a potential match between buy and sell orders, and a trade between the bots should have happened.

Trader: 
The class only requires a single method called run, which is called by the simulation every time a new TraderState is available. 
The logic within this run method is written by the player and determines the behaviour of the algorithm. 
The output of the method is a dictionary named result, which contains all the orders that the algorithm decides to send based on this logic.

Trade:
Both the own_trades property and the market_trades property provide the traders with a list of trades per products. Every individual trade in each of these lists is an instance of the Trade class.
These trades have five distinct properties:

1. The symbol/product that the trade corresponds to (i.e. are we exchanging apples or oranges)
2. The price at which the product was exchanged
3. The quantity that was exchanged
4. The identity of the buyer in the transaction
5. The identity of the seller in this transaction

On the island exchange, like on most real-world exchanges, counterparty information is typically not disclosed. Therefore properties 4 and 5 will only be non-empty strings if the algorithm itself is the buyer (4 will be “SUBMISSION”) or the seller (5 will be “SUBMISSION”)

Observation:
Observation details help to decide on eventual orders or conversion requests. There are two items delivered inside the TradingState instance:

1. Simple product to value dictionary inside plainValueObservations
2. Dictionary of complex **ConversionObservation** values for respective products. Used to place conversion requests from Trader class.

In case you decide to place conversion request on product listed integer number should be returned as “conversions” value from run() method. Based on logic defined inside Prosperity container it will convert positions acquired by submitted code. There is a number of conditions for conversion to  happen:

- You need to obtain either long or short position earlier.
- Conversion request cannot exceed possessed items count.
- In case you have 10 items short (-10) you can only request from 1 to 10. Request for 11 or more will be fully ignored.
- While conversion happens you will need to cover transportation and import/export tariff.
- Conversion request is not mandatory. You can send 0 or None as value.
