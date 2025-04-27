import dash # type: ignore
from dash import dcc, html, Input, Output, State, dash_table # type: ignore

import plotly.graph_objects as go  # type: ignore
import plotly.express as px # type: ignore
import pandas as pd # type: ignore
import json
import os 
from .log_parser import parse_log_file # Use relative import
from .historical_data_loader import load_historical_data # Added import for historical data

# --- Configuration ---
LOG_FILE_PATH = '/Users/omersen/prosperity-poj/dashboard/1dea0e49-af48-4ade-b6d7-f8bc1a152dd2.log' # Or make this configurable
HISTORICAL_DATA_DIR = '/Users/omersen/prosperity-poj/strategy/round5/resources/all_data' # Added path for historical data
ROUND_4_PRODUCTS = ["KELP", "RAINFOREST_RESIN", "SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES", "BASKET1", "BASKET2", "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500", "VOLCANIC_ROCK_VOUCHER_10750", "VOLCANIC_ROCK_VOUCHER_11000", "MAGNIFICENT_MACARONS"] # Added list of R1 products
POSITION_LIMITS = { # Hardcode for now, 
    "KELP": 50,
    "RAINFOREST_RESIN": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "BASKET1": 60,
    "BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "VOLCANIC_ROCK_VOUCHER_10750": 200,
    "VOLCANIC_ROCK_VOUCHER_11000": 200,
    "MAGNIFICENT_MACARONS": 75
}

# --- Load and Prepare Data ---
print("Loading log data...")
lambda_state_df, activities_df, trade_history_df = parse_log_file(LOG_FILE_PATH)
print("Log data loaded.")

# Basic Data Preparation (add more as needed)
# Ensure timestamp is numeric and sort activities if needed
activities_df['timestamp'] = pd.to_numeric(activities_df['timestamp'])
activities_df = activities_df.sort_values(by='timestamp')
lambda_state_df['timestamp'] = pd.to_numeric(lambda_state_df['timestamp'])
lambda_state_df = lambda_state_df.sort_values(by='timestamp')
trade_history_df['timestamp'] = pd.to_numeric(trade_history_df['timestamp'])
trade_history_df = trade_history_df.sort_values(by='timestamp')

# Extract unique timestamps for slider/input
timestamps = sorted(lambda_state_df['timestamp'].unique())
min_ts = timestamps[0]
max_ts = timestamps[-1]
ts_map = {i: ts for i, ts in enumerate(timestamps)} # Map index to timestamp
ts_rev_map = {ts: i for i, ts in enumerate(timestamps)} # Map timestamp to index

# --- Load Historical Data ---
print("Loading historical data...")
historical_data = load_historical_data(HISTORICAL_DATA_DIR)
print(f"Historical data loaded for days: {list(historical_data.keys())}")
# Define available historical days for selector
historical_days = sorted([day for day in historical_data.keys() if 'prices' in historical_data[day] or 'trades' in historical_data[day]])

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True) # Suppress exceptions for initial layout
server = app.server # Expose server variable for deployments

# --- App Layout ---
app.layout = html.Div([
    html.H1("IMC Prosperity Dashboard"),
    html.Hr(),

    dcc.Tabs(id="analysis-tabs", value='tab-log-analysis', children=[
        dcc.Tab(label='Log File Analysis', value='tab-log-analysis', children=[
            html.Div([
                html.H2("Log File Analysis"),
                html.P(f"Displaying data from: {os.path.basename(LOG_FILE_PATH)}"),

                # Dynamic container for Log Price Graphs
                html.Div(id='log-price-graphs-container'),

                # Graphs Row 2 (Positions & PnL - FROM LOGS) - Keep these separate for layout
                html.Div([
                     html.Div(dcc.Graph(id='position-graph'), style={'width': '49%', 'display': 'inline-block'}),
                     html.Div(dcc.Graph(id='pnl-graph'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '20px'}),

                # Timestamp Controls
                html.Div([
                    html.H3("Select Timestamp for Detailed State:"),
                    html.Label("Timestamp:"),
                    dcc.Slider(
                        id='timestamp-slider',
                        min=0,
                        max=len(timestamps) - 1,
                        value=0,
                        marks={i: str(ts) for i, ts in ts_map.items() if i % (len(timestamps)//10) == 0}, # Mark ~10 timestamps
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    dcc.Input(
                        id='timestamp-input',
                        type='number',
                        value=min_ts,
                        min=min_ts,
                        max=max_ts,
                        step=100 # Typical step in competition logs
                    ),
                    html.Div(id='timestamp-display')
                ], style={'marginBottom': '40px', 'marginTop': '20px'}),

                # Data Display Row (Trader Data, LOBs, Trades, etc. - FROM LOGS)
                html.Div([
                    html.H3("State at Selected Timestamp (from Log File)"),
                    # Left Column: Trader Data & LOBs
                    html.Div([
                        html.H4("Trader Data"),
                        html.Pre(id='trader-data-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '300px'}),
                        html.Hr(),
                        html.H4("Order Books"),
                        # Dynamic container for LOB displays
                        html.Div(id='log-lob-container'),
                    ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    # Right Column: Trades, Orders, Logs, etc.
                    html.Div([
                        html.H4("Orders Sent"),
                        html.Div(id='orders-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '150px'}),
                        html.H4("Own Trades at Timestamp"),
                        html.Div(id='own-trades-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '150px'}),
                        html.H4("Market Trades at Timestamp"),
                        html.Div(id='market-trades-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '150px'}),
                        html.H4("Algorithm Logs"),
                        html.Pre(id='algo-logs-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '150px'}),
                        html.H4("Current State"),
                        html.Pre(id='current-state-display', style={'border': '1px solid lightgrey', 'padding': '10px'}),
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'})
                ])
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Historical Data Analysis', value='tab-historical-analysis', children=[
             html.Div([
                html.H2("Historical Data Analysis (Round 1 CSVs)"),
                html.Div([
                    html.H3("Select Historical Day:"),
                    dcc.RadioItems(
                        id='day-selector',
                        options=[{'label': f'Day {day}', 'value': day} for day in historical_days],
                        value=historical_days[0] if historical_days else None, # Default to first available day
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                ], style={'marginBottom': '20px'}),

                # Historical Graphs (Generated Dynamically by Callback)
                html.Div(id='historical-graphs-container'),
             ], style={'padding': '20px'}) # Add padding to tab content
        ]),
    ]),

    # dcc.Store(id='selected-timestamp-store', data=min_ts) # Maybe use later if needed

], style={'padding': '0px 20px 20px 20px'}) # Adjust main padding slightly

# --- Helper Functions ---
def calculate_popular_mid_price(row):
    """Calculates mid-price based on highest volume bid/ask levels."""
    bid_prices = ['bid_price_1', 'bid_price_2', 'bid_price_3']
    ask_prices = ['ask_price_1', 'ask_price_2', 'ask_price_3']
    bid_volumes = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    ask_volumes = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']

    # Find best bid price (highest price with non-zero volume)
    best_bid_vol = 0
    best_bid_price = 0
    for bp, bv in zip(bid_prices, bid_volumes):
        if row[bv] > best_bid_vol and pd.notna(row[bp]):
            best_bid_vol = row[bv]
            best_bid_price = row[bp]
        elif row[bv] == best_bid_vol and pd.notna(row[bp]) and row[bp] > best_bid_price: # Tie-break with higher price
             best_bid_price = row[bp]


    # Find best ask price (lowest price with non-zero volume)
    best_ask_vol = 0
    best_ask_price = float('inf')
    for ap, av in zip(ask_prices, ask_volumes):
         if row[av] > best_ask_vol and pd.notna(row[ap]):
            best_ask_vol = row[av]
            best_ask_price = row[ap]
         elif row[av] == best_ask_vol and pd.notna(row[ap]) and row[ap] < best_ask_price: # Tie-break with lower price
             best_ask_price = row[ap]


    if best_bid_price > 0 and best_ask_price != float('inf'):
        return (best_bid_price + best_ask_price) / 2
    else:
        # Fallback or handle cases where one side might be empty
        # Use standard midprice if popular price cannot be calculated
        if pd.notna(row.get('bid_price_1')) and pd.notna(row.get('ask_price_1')):
             return (row['bid_price_1'] + row['ask_price_1']) / 2
        else:
            return None # Indicate failure to calculate

# --- Callbacks ---
# Callback to synchronize slider and input box, and display current timestamp
@app.callback(
    [Output('timestamp-display', 'children'),
     Output('timestamp-input', 'value'),
     Output('timestamp-slider', 'value')],
    [Input('timestamp-slider', 'value'),
     Input('timestamp-input', 'value')],
    prevent_initial_call=True
)
def sync_timestamp_controls(slider_idx, input_ts):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    current_ts = min_ts
    current_idx = 0

    if trigger_id == 'timestamp-slider':
        current_idx = slider_idx if slider_idx is not None else 0
        current_ts = ts_map.get(current_idx, min_ts)
    elif trigger_id == 'timestamp-input':
        current_ts = input_ts if input_ts is not None else min_ts
        # Find closest index for the slider
        if current_ts in ts_rev_map:
            current_idx = ts_rev_map[current_ts]
        else:
            # If exact timestamp not found (e.g., input halfway), find nearest
            closest_ts = min(timestamps, key=lambda ts: abs(ts - current_ts))
            current_idx = ts_rev_map.get(closest_ts, 0)
            current_ts = closest_ts # Snap input to the closest valid TS

    display_text = f"Selected Timestamp: {current_ts}"
    return display_text, current_ts, current_idx

# Callback for LOG Price Graphs (Now dynamic, includes Spread)
@app.callback(
    Output('log-price-graphs-container', 'children'), # Output to the container
    Input('timestamp-input', 'value') # Trigger on timestamp change
)
def update_log_price_graphs(selected_ts):
    price_graph_children = []

    # Check if dataframes are empty
    if lambda_state_df.empty or activities_df.empty or trade_history_df.empty:
         return html.P('No Log Data Loaded to generate price graphs.')

    # Merge calculated fair values from lambda_state_df into activities_df for plotting
    potential_fair_value_cols = {f'fair_value_{p}': p for p in activities_df['product'].unique()}
    lambda_df_copy = lambda_state_df.copy() # Work on a copy
    for col, prod in potential_fair_value_cols.items():
        if 'calculated_fair_values' in lambda_df_copy.columns:
             lambda_df_copy[col] = lambda_df_copy['calculated_fair_values'].apply(lambda x: x.get(prod) if isinstance(x, dict) else None)
        else:
             lambda_df_copy[col] = None # Add column with None if fair values missing

    fair_value_cols_to_merge = [col for col in potential_fair_value_cols if col in lambda_df_copy.columns]
    fair_value_data = lambda_df_copy[['timestamp'] + fair_value_cols_to_merge]

    # Merge with activities_df - use outer join to keep all timestamps
    activities_with_fair = pd.merge(activities_df, fair_value_data, on='timestamp', how='left')

    products_in_log = sorted(activities_with_fair['product'].unique()) # Get unique products from activities

    for i, product in enumerate(products_in_log):
        fig_price = go.Figure()
        fig_spread = go.Figure() # Figure for spread
        product_price_data = activities_with_fair[activities_with_fair['product'] == product].copy() # Use copy
        product_trade_data = trade_history_df[trade_history_df['symbol'] == product]
        fair_value_col_name = f'fair_value_{product}'

        if not product_price_data.empty:
            # --- Price Graph Logic ---
            # (Keep existing price graph logic: Bids, Asks, Mid, Fair Value, Trades, Vline)
            # Bids
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['bid_price_1'], mode='lines', name='Bid 1', line=dict(color='lightgreen')))
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['bid_price_2'], mode='lines', name='Bid 2', line=dict(color='lightgreen', dash='dot'), visible='legendonly'))
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['bid_price_3'], mode='lines', name='Bid 3', line=dict(color='lightgreen', dash='dashdot'), visible='legendonly'))
            # Asks
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['ask_price_1'], mode='lines', name='Ask 1', line=dict(color='lightcoral')))
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['ask_price_2'], mode='lines', name='Ask 2', line=dict(color='lightcoral', dash='dot'), visible='legendonly'))
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['ask_price_3'], mode='lines', name='Ask 3', line=dict(color='lightcoral', dash='dashdot'), visible='legendonly'))
            # Mid Price
            fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data['mid_price'], mode='lines', name='Mid Price', line=dict(color='lightblue')))
            # Fair Value
            if fair_value_col_name in product_price_data.columns and not product_price_data[fair_value_col_name].isnull().all():
                fig_price.add_trace(go.Scatter(x=product_price_data['timestamp'], y=product_price_data[fair_value_col_name], mode='lines', name='Fair Value (Algo)', line=dict(color='orange', dash='longdash'), visible='legendonly'))
            # Trades
            own_buy = product_trade_data[product_trade_data['buyer'] == 'SUBMISSION']
            own_sell = product_trade_data[product_trade_data['seller'] == 'SUBMISSION']
            market_trades = product_trade_data[(product_trade_data['buyer'] != 'SUBMISSION') & (product_trade_data['seller'] != 'SUBMISSION')]
            fig_price.add_trace(go.Scatter(x=own_buy['timestamp'], y=own_buy['price'], mode='markers', name='Own Buy', marker=dict(color='cyan', symbol='triangle-up', size=8), visible='legendonly', hovertemplate="Buy Qty: %{customdata}<extra></extra>", customdata=own_buy['quantity']))
            fig_price.add_trace(go.Scatter(x=own_sell['timestamp'], y=own_sell['price'], mode='markers', name='Own Sell', marker=dict(color='yellow', symbol='triangle-down', size=8), visible='legendonly', hovertemplate="Sell Qty: %{customdata}<extra></extra>", customdata=own_sell['quantity']))
            fig_price.add_trace(go.Scatter(x=market_trades['timestamp'], y=market_trades['price'], mode='markers', name='Market Trades', marker=dict(color='magenta', symbol='x', size=6), visible='legendonly', hovertemplate="Market Qty: %{customdata}<extra></extra>", customdata=market_trades['quantity']))
            # Vline
            fig_price.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")
            fig_price.update_layout(
                title=f'{product} Price Levels & Trades (Log Data)',
                xaxis_title='Timestamp', yaxis_title='Price',
                hovermode='x unified', template='plotly_dark'
            )

            # --- Spread Graph Logic ---
            if 'ask_price_1' in product_price_data.columns and 'bid_price_1' in product_price_data.columns:
                product_price_data['spread'] = product_price_data['ask_price_1'] - product_price_data['bid_price_1']
                # Handle potential NaNs if calculation results in them
                product_price_data['spread'] = product_price_data['spread'].fillna(method='ffill').fillna(method='bfill') # Forward fill then back fill

                fig_spread.add_trace(go.Scatter(
                    x=product_price_data['timestamp'],
                    y=product_price_data['spread'],
                    mode='lines', name='Bid-Ask Spread', line=dict(color='lightyellow')
                ))
                fig_spread.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")
                fig_spread.update_layout(
                    title=f'{product} Bid-Ask Spread (Log Data)',
                    xaxis_title='Timestamp', yaxis_title='Spread (Ask1 - Bid1)',
                    hovermode='x unified', template='plotly_dark'
                )
            else:
                fig_spread.update_layout(title=f'{product} Spread (Missing Data)', template='plotly_dark')

        else:
             # Handle case where product is in activities but has no price data
             fig_price.update_layout(title=f'{product} (No Price Data)', template='plotly_dark')
             fig_spread.update_layout(title=f'{product} Spread (No Price Data)', template='plotly_dark')

        # --- Layout for Product ---
        # Determine layout style based on number of products
        width_style = '99%' if len(products_in_log) == 1 else '49%'
        float_style = 'none' if i % 2 == 0 else 'right'
        style = {'width': width_style, 'display': 'inline-block', 'verticalAlign': 'top'}
        if len(products_in_log) > 1 and i % 2 != 0:
            style['float'] = float_style

        # Create the div containing both price and spread graphs for this product
        product_graph_div = html.Div([
            dcc.Graph(figure=fig_price),
            dcc.Graph(figure=fig_spread, style={'marginTop': '5px'}) # Add small margin
        ], style=style)

        price_graph_children.append(product_graph_div)

        # Add a clearing div after every pair of graphs if more than 1 product
        if len(products_in_log) > 1 and i % 2 != 0:
            price_graph_children.append(html.Div(style={'clear': 'both', 'height': '20px'})) # Add height for spacing

    return price_graph_children

# Callback for Position Graph (Added robustness for missing limits)
@app.callback(
    Output('position-graph', 'figure'),
    Input('timestamp-input', 'value')
)
def update_position_graph(selected_ts):
    fig = go.Figure()
    if lambda_state_df.empty:
         fig.update_layout(title='Product Positions (No Log Data)', template='plotly_dark')
         return fig

    # Plot positions for each product IN THE LOG DATA
    log_products = []
    if not lambda_state_df.empty and not lambda_state_df['positions'].iloc[0] is None:
         # Get products from the first row's position dict keys, handle potential None
         first_pos = lambda_state_df['positions'].iloc[0]
         if isinstance(first_pos, dict):
             log_products = list(first_pos.keys())

    if not log_products: # If no positions found in first row, try unique products from activities
        if not activities_df.empty:
            log_products = activities_df['product'].unique()

    plotted_products = []
    for product in log_products:
        limit = POSITION_LIMITS.get(product)
        if limit is None:
             print(f"Warning: Position limit not defined for '{product}' in POSITION_LIMITS. Cannot plot % limit.")
             continue # Skip this product if limit is not defined

        # Extract position for this product - careful with missing keys
        positions = lambda_state_df['positions'].apply(lambda p: p.get(product, 0) if isinstance(p, dict) else 0) # Default to 0

        # Calculate position as % of limit
        # Avoid division by zero if limit is 0 for some reason
        if limit != 0:
            position_pct = (positions / limit) * 100
            fig.add_trace(go.Scatter(
                x=lambda_state_df['timestamp'],
                y=position_pct,
                mode='lines',
                name=f'{product} Position (% Limit)'
            ))
            plotted_products.append(product)
        else:
            print(f"Warning: Position limit is 0 for '{product}'. Cannot plot % limit.")

    # Add vertical line for selected timestamp
    fig.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        title='Product Positions (% of Limit) (Log Data)' + (f' - Displaying: {", ".join(plotted_products)}' if plotted_products else ' - No Positions Plotable'),
        xaxis_title='Timestamp',
        yaxis_title='% of Position Limit',
        hovermode='x unified',
        template='plotly_dark',
        yaxis_range=[-105, 105]
    )
    return fig

# Callback for PnL Graph (Adapt for new products)
@app.callback(
    Output('pnl-graph', 'figure'),
    Input('timestamp-input', 'value') # Trigger on timestamp change for vertical line
)
def update_pnl_graph(selected_ts):
    fig = go.Figure()
    total_pnl_col = 'Total PnL'
    if activities_df.empty:
        fig.update_layout(title='Profit & Loss (No Log Data)', template='plotly_dark')
        return fig

    # Calculate Total PnL across products per timestamp
    try:
        pnl_data = activities_df.pivot(index='timestamp', columns='product', values='profit_and_loss')
        pnl_data = pnl_data.fillna(0) # Fill NaNs in PnL if a product wasn't present at a timestamp
        pnl_data[total_pnl_col] = pnl_data.sum(axis=1)
        pnl_data = pnl_data.reset_index() # Make timestamp a column again
    except Exception as e:
        print(f"Error pivoting PnL data: {e}")
        fig.update_layout(title='Profit & Loss (Error Processing)', template='plotly_dark')
        return fig

    # Plot PnL for each product found in the activities log
    products_in_log = activities_df['product'].unique()
    for product in products_in_log:
        if product in pnl_data.columns: # Ensure product column exists after pivot
             fig.add_trace(go.Scatter(
                 x=pnl_data['timestamp'],
                 y=pnl_data[product],
                 mode='lines',
                 name=f'{product} PnL',
                 visible='legendonly' # Hide individual PnLs initially
             ))

    # Plot Total PnL (visible by default)
    if total_pnl_col in pnl_data.columns:
        fig.add_trace(go.Scatter(
            x=pnl_data['timestamp'],
            y=pnl_data[total_pnl_col],
            mode='lines',
            name=total_pnl_col,
            line=dict(color='gold', width=3)
        ))

    # Add vertical line for selected timestamp
    fig.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        title='Profit & Loss Over Time (Log Data)',
        xaxis_title='Timestamp', 
        yaxis_title='PnL',
        hovermode='x unified', 
        template='plotly_dark'
    )
    return fig

# Combined callback to update timestamp-specific details (LOBs now dynamic)
@app.callback(
    [Output('trader-data-display', 'children'),
     Output('log-lob-container', 'children'), # LOBs now go here
     Output('orders-display', 'children'),
     Output('own-trades-display', 'children'),
     Output('market-trades-display', 'children'),
     Output('algo-logs-display', 'children'),
     Output('current-state-display', 'children')],
    Input('timestamp-input', 'value')
)
def update_timestamp_details(selected_ts):
    # Initial check for data
    if lambda_state_df.empty or selected_ts is None or selected_ts not in lambda_state_df['timestamp'].values:
        no_data_msg = "Select a valid timestamp or No Log Data."
        # Return message for all outputs (7 outputs now)
        lob_container_msg = html.P(no_data_msg)
        return no_data_msg, lob_container_msg, no_data_msg, no_data_msg, no_data_msg, no_data_msg, no_data_msg

    # --- Get Data for Timestamp ---
    state_row = lambda_state_df[lambda_state_df['timestamp'] == selected_ts].iloc[0]
    order_depths = state_row['order_depths']
    orders_list = state_row['compressed_orders_list']
    own_trades_list = state_row['own_trades']
    market_trades_list = state_row['market_trades']
    logs_str = state_row['logs_str']
    positions_dict = state_row['positions']
    trader_data_str = state_row['trader_data_str']

    # --- 1. Trader Data ---
    trader_data_display = "No Trader Data logged for this timestamp."
    if trader_data_str:
        try:
            # Attempt to pretty-print if it's JSON
            trader_data_dict = json.loads(trader_data_str)
            trader_data_display = json.dumps(trader_data_dict, indent=4)
        except Exception:
             # Otherwise, display as raw string
             trader_data_display = trader_data_str

    # --- 2. LOB Displays (Dynamic) ---
    lob_children = []
    if isinstance(order_depths, dict):
        products_in_state = sorted(list(order_depths.keys()))
        for product in products_in_state:
            product_lob_content = None
            # Check if the order depth structure is as expected (list/tuple of two dicts)
            if not isinstance(order_depths[product], (list, tuple)) or len(order_depths[product]) != 2:
                 product_lob_content = html.P(f"Unexpected order depth format for {product}")
            else:
                buy_orders = order_depths[product][0]
                sell_orders = order_depths[product][1]

                # Ensure buy_orders and sell_orders are dictionaries
                if not isinstance(buy_orders, dict) or not isinstance(sell_orders, dict):
                     product_lob_content = html.P(f"Invalid order book format (buy/sell not dicts) for {product}")
                else:
                    # Ensure keys are convertible to int before sorting
                    try:
                         buy_levels = sorted([(int(p), v) for p, v in buy_orders.items() if v > 0], reverse=True) # Filter 0 vol
                         sell_levels = sorted([(int(p), -v) for p, v in sell_orders.items() if v < 0]) # Filter 0 vol, lowest ask first
                    except (ValueError, TypeError) as e:
                         product_lob_content = html.P(f"Error processing LOB prices for {product}: {e}")

                    if product_lob_content is None: # If no error so far
                        depth = 5
                        data_for_table = []

                        # Add top N asks in REVERSE order (lowest price first in sell_levels)
                        for i in range(min(depth, len(sell_levels))):
                            price, volume = sell_levels[i]
                            data_for_table.append({'Bid Vol': '', 'Price': price, 'Ask Vol': volume})
                        data_for_table.reverse() # Show highest ask price first

                        # Add top N bids (highest price first)
                        for i in range(min(depth, len(buy_levels))):
                            price, volume = buy_levels[i]
                            data_for_table.append({'Bid Vol': volume, 'Price': price, 'Ask Vol': ''})

                        if not data_for_table:
                             product_lob_content = html.P(f"Empty order book for {product}")
                        else:
                            lob_df = pd.DataFrame(data_for_table)
                            table = dash_table.DataTable(
                                columns=[{"name": i, "id": i} for i in lob_df.columns],
                                data=lob_df.to_dict('records'),
                                style_cell={
                                    'textAlign': 'center', 'padding': '5px', 'fontFamily': 'Arial, sans-serif',
                                    'border': '1px solid grey'
                                },
                                style_header={
                                    'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'
                                },
                                style_data={'color': 'black','backgroundColor': 'white'},
                                style_data_conditional=[
                                    {'if': {'column_id': 'Bid Vol', 'filter_query': '{Bid Vol} != ""'}, 'backgroundColor': 'rgb(189, 215, 231)', 'fontWeight': 'bold'},
                                    {'if': {'column_id': 'Ask Vol', 'filter_query': '{Ask Vol} != ""'}, 'backgroundColor': 'rgb(231, 189, 189)', 'fontWeight': 'bold'}
                                ]
                            )
                            product_lob_content = table

            # Add the generated content (Table or Error P) for the product to the list
            lob_children.append(html.Div([
                html.H4(f"{product} Order Book", style={'marginTop': '15px'}),
                product_lob_content
            ]))

    else:
        lob_children = [html.P("Order depth data is not in the expected format (dictionary).")]

    # --- 3. Orders Table ---
    orders_display = "No orders sent at this timestamp."
    if orders_list and isinstance(orders_list, list) and len(orders_list) > 0:
        try:
            # Check if first element is a list/tuple with 3 items
            if isinstance(orders_list[0], (list, tuple)) and len(orders_list[0]) == 3:
                 orders_df = pd.DataFrame(orders_list, columns=['Symbol', 'Price', 'Quantity'])
                 orders_df['Type'] = orders_df['Quantity'].apply(lambda q: 'BUY' if q > 0 else 'SELL')
                 orders_df['Quantity'] = orders_df['Quantity'].abs()
                 orders_display = dash_table.DataTable(
                     columns=[{"name": i, "id": i} for i in ['Symbol', 'Type', 'Price', 'Quantity']],
                     data=orders_df.to_dict('records'),
                     style_cell={'textAlign': 'left', 'padding': '5px'},
                     style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
                     style_data={'color': 'black','backgroundColor': 'white'}
                 )
            else:
                 orders_display = "Orders data has unexpected format."
        except Exception as e:
            orders_display = f"Error creating orders table: {e}"
    elif not orders_list:
         orders_display = "No orders sent at this timestamp." # Explicitly handle empty list
    else:
         orders_display = "Orders data has unexpected format (not a list)."


    # --- 4. Own Trades Table ---
    own_trades_display = "No own trades at this timestamp."
    if own_trades_list and isinstance(own_trades_list, list) and len(own_trades_list) > 0:
        try:
             # Check format of first trade entry
             if isinstance(own_trades_list[0], (list, tuple)) and len(own_trades_list[0]) == 6:
                 own_trades_df = pd.DataFrame(own_trades_list, columns=['Symbol', 'Price', 'Quantity', 'Buyer', 'Seller', 'Trade Ts'])
                 own_trades_df['Type'] = own_trades_df.apply(lambda row: 'BUY' if row['Buyer'] == 'SUBMISSION' else 'SELL', axis=1)
                 own_trades_df = own_trades_df[['Trade Ts', 'Symbol', 'Type', 'Price', 'Quantity']]
                 own_trades_display = dash_table.DataTable(
                     columns=[{"name": i, "id": i} for i in own_trades_df.columns],
                     data=own_trades_df.to_dict('records'),
                     style_cell={'textAlign': 'left', 'padding': '5px'},
                     style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
                     style_data={'color': 'black','backgroundColor': 'white'}
                 )
             else:
                 own_trades_display = "Own trades data has unexpected format."
        except Exception as e:
            own_trades_display = f"Error creating own trades table: {e}"
    elif not own_trades_list:
         own_trades_display = "No own trades at this timestamp." # Explicitly handle empty list
    else:
         own_trades_display = "Own trades data has unexpected format (not a list)."

    # --- 5. Market Trades Table ---
    market_trades_display = "No market trades at this timestamp."
    if market_trades_list and isinstance(market_trades_list, list) and len(market_trades_list) > 0:
        try:
             if isinstance(market_trades_list[0], (list, tuple)) and len(market_trades_list[0]) == 6:
                 market_trades_df = pd.DataFrame(market_trades_list, columns=['Symbol', 'Price', 'Quantity', 'Buyer', 'Seller', 'Trade Ts'])
                 market_trades_df = market_trades_df[['Trade Ts', 'Symbol', 'Price', 'Quantity', 'Buyer', 'Seller']]
                 market_trades_display = dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in market_trades_df.columns],
                    data=market_trades_df.to_dict('records'),
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
                    style_data={'color': 'black','backgroundColor': 'white'}
                 )
             else:
                  market_trades_display = "Market trades data has unexpected format."
        except Exception as e:
            market_trades_display = f"Error creating market trades table: {e}"
    elif not market_trades_list:
         market_trades_display = "No market trades at this timestamp." # Explicitly handle empty list
    else:
         market_trades_display = "Market trades data has unexpected format (not a list)."


    # --- 6. Algo Logs ---
    algo_logs_display = logs_str if logs_str else "No algorithm logs for this timestamp."

    # --- 7. Current State ---
    current_pnl_dict = {}
    # Make sure activities_df is not empty before filtering
    if not activities_df.empty:
        pnl_row = activities_df[activities_df['timestamp'] == selected_ts]
        if not pnl_row.empty:
            for product in pnl_row['product'].unique():
                product_pnl = pnl_row[pnl_row['product'] == product]['profit_and_loss'].iloc[0]
                current_pnl_dict[product] = round(product_pnl, 2)

    current_state_lines = []
    current_state_lines.append("Positions:")
    if isinstance(positions_dict, dict):
        for product, pos in positions_dict.items():
             limit = POSITION_LIMITS.get(product, 'N/A') # Keep using N/A if limit not defined
             current_state_lines.append(f"  {product}: {pos} / {limit}")
    else:
         current_state_lines.append("  (Position data not available or not a dictionary)")

    current_state_lines.append("\nPnL at Timestamp:")
    if current_pnl_dict:
        for product, pnl in current_pnl_dict.items(): current_state_lines.append(f"  {product}: {pnl}")
    else:
        current_state_lines.append("  (N/A)")

    current_state_display = "\n".join(current_state_lines)

    # --- Return all updated components ---
    return (
        trader_data_display,
        lob_children, # Return the list of LOB divs
        orders_display,
        own_trades_display,
        market_trades_display,
        algo_logs_display,
        current_state_display
    )

# --- NEW Callback for Historical Graphs (includes Spread) ---
@app.callback(
    Output('historical-graphs-container', 'children'),
    Input('day-selector', 'value')
)
def update_historical_graphs(selected_day):
    if selected_day is None or selected_day not in historical_data:
        return html.P(f"Please select a valid historical day with data. Available: {list(historical_data.keys())}")

    day_data = historical_data[selected_day]
    prices_df = day_data.get('prices')
    trades_df = day_data.get('trades')

    if prices_df is None or prices_df.empty:
        return html.P(f"No historical price data found for Day {selected_day}.")

    # --- Data Preprocessing ---
    try:
        prices_df['timestamp'] = pd.to_numeric(prices_df['timestamp'])
        prices_df = prices_df.sort_values(by='timestamp')
        if trades_df is not None and not trades_df.empty:
            trades_df['timestamp'] = pd.to_numeric(trades_df['timestamp'])
            trades_df = trades_df.sort_values(by='timestamp')
            if 'symbol' not in trades_df.columns:
                 print("Warning: 'symbol' column missing in historical trades data.")
                 trades_df = None # Cannot use trades data without symbol

        # Calculate popular mid-price and standard mid-price
        prices_df['popular_mid_price'] = prices_df.apply(calculate_popular_mid_price, axis=1)
        prices_df['standard_mid_price'] = (prices_df.get('bid_price_1') + prices_df.get('ask_price_1')) / 2
        # Calculate spread
        if 'ask_price_1' in prices_df.columns and 'bid_price_1' in prices_df.columns:
             prices_df['spread'] = prices_df['ask_price_1'] - prices_df['bid_price_1']
             prices_df['spread'] = prices_df['spread'].fillna(method='ffill').fillna(method='bfill')
        else:
             prices_df['spread'] = None # Or np.nan

    except Exception as e:
        print(f"Error during historical data preprocessing for day {selected_day}: {e}")
        return html.P(f"Error processing data for Day {selected_day}. Check data format.")


    # --- Generate Graphs ---
    graph_children = []
    products_in_day = prices_df['product'].unique()

    for product in products_in_day:
        # Filter data for the current product
        product_prices = prices_df[prices_df['product'] == product].copy()
        product_trades = pd.DataFrame() # Default to empty
        if trades_df is not None and 'symbol' in trades_df.columns: # Ensure symbol exists before filtering
             product_trades = trades_df[trades_df['symbol'] == product].copy()
             # Clean trades timestamp and price
             if not product_trades.empty:
                 product_trades['timestamp'] = pd.to_numeric(product_trades['timestamp'], errors='coerce')
                 product_trades['price'] = pd.to_numeric(product_trades['price'], errors='coerce')
                 product_trades.dropna(subset=['timestamp', 'price'], inplace=True)


        # --- Create Price Graph ---
        fig_price = go.Figure()
        fig_spread = go.Figure()
        fig_hist = go.Figure()

        # Check if there is any price data for this product on this day
        if product_prices.empty:
            print(f"Warning: No historical price data found for product '{product}' on day {selected_day}. Skipping graphs.")
            # Create placeholder graphs
            fig_price.update_layout(title=f'{product} - No Historical Price Data (Day {selected_day})', template='plotly_dark')
            fig_spread.update_layout(title=f'{product} - No Spread Data (Day {selected_day})', template='plotly_dark')
            fig_hist.update_layout(title=f'{product} - No Volume Data (Day {selected_day})', template='plotly_dark')
        else:
            # --- Clean Price Data ---
            # Ensure timestamp is numeric and drop NaNs
            product_prices['timestamp'] = pd.to_numeric(product_prices['timestamp'], errors='coerce')
            product_prices.dropna(subset=['timestamp'], inplace=True)
            if product_prices.empty: # Check again after dropping NaN timestamps
                 print(f"Warning: No valid timestamps for product '{product}' on day {selected_day} after cleaning. Skipping graphs.")
                 fig_price.update_layout(title=f'{product} - No Valid Timestamps (Day {selected_day})', template='plotly_dark')
                 fig_spread.update_layout(title=f'{product} - No Valid Timestamps (Day {selected_day})', template='plotly_dark')
                 fig_hist.update_layout(title=f'{product} - No Valid Timestamps (Day {selected_day})', template='plotly_dark')
                 # Add these placeholder graphs to children and continue loop
                 graph_children.append(html.Div([
                     html.Div(dcc.Graph(figure=fig_price), style={'width': '100%', 'marginBottom': '5px'}),
                     html.Div(dcc.Graph(figure=fig_spread), style={'width': '100%', 'marginBottom': '5px'}),
                     html.Div(dcc.Graph(figure=fig_hist), style={'width': '100%'})
                 ], style={'marginBottom': '20px', 'border': '1px solid #555', 'padding': '10px'}))
                 continue # Skip to the next product


            # Price Graph Logic
            # Bids & Asks (Level 1 visible, 2 & 3 hidden) - Clean data before plotting each
            for i in range(1, 4):
                bid_col_p = f'bid_price_{i}'
                ask_col_p = f'ask_price_{i}'
                if bid_col_p in product_prices.columns:
                     bid_data = pd.to_numeric(product_prices[bid_col_p], errors='coerce').ffill().bfill()
                     if not bid_data.isnull().all():
                         fig_price.add_trace(go.Scatter(x=product_prices['timestamp'], y=bid_data, mode='lines', name=f'Bid {i}', line=dict(color='lightgreen', dash=('solid' if i == 1 else ('dot' if i==2 else 'dashdot'))), visible=(True if i == 1 else 'legendonly')))
                if ask_col_p in product_prices.columns:
                     ask_data = pd.to_numeric(product_prices[ask_col_p], errors='coerce').ffill().bfill()
                     if not ask_data.isnull().all():
                         fig_price.add_trace(go.Scatter(x=product_prices['timestamp'], y=ask_data, mode='lines', name=f'Ask {i}', line=dict(color='lightcoral', dash=('solid' if i == 1 else ('dot' if i==2 else 'dashdot'))), visible=(True if i == 1 else 'legendonly')))

            # Mid Prices - Clean data before plotting
            mid_price_col = 'standard_mid_price'
            if mid_price_col in product_prices.columns:
                 mid_price_data = pd.to_numeric(product_prices[mid_price_col], errors='coerce').ffill().bfill()
                 if not mid_price_data.isnull().all():
                     fig_price.add_trace(go.Scatter(x=product_prices['timestamp'], y=mid_price_data, mode='lines', name='Mid Price (B1/A1)', line=dict(color='lightblue')))

            pop_mid_price_col = 'popular_mid_price'
            if pop_mid_price_col in product_prices.columns:
                 pop_mid_price_data = pd.to_numeric(product_prices[pop_mid_price_col], errors='coerce').ffill().bfill()
                 if not pop_mid_price_data.isnull().all():
                     fig_price.add_trace(go.Scatter(x=product_prices['timestamp'], y=pop_mid_price_data, mode='lines', name='Mid Price (Popular Vol)', line=dict(color='yellow', dash='dash'), visible='legendonly'))

            # Market Trades - Already cleaned earlier, keep as Scattergl for markers if desired, or change too
            if not product_trades.empty:
                fig_price.add_trace(go.Scatter(x=product_trades['timestamp'], y=product_trades['price'], mode='markers', name='Market Trades', marker=dict(color='magenta', symbol='x', size=6), hovertemplate="Trade Qty: %{customdata}<extra></extra>", customdata=product_trades['quantity']))

            fig_price.update_layout(
                title=f'{product} - Historical Prices & Trades (Day {selected_day})',
                xaxis_title='Timestamp', yaxis_title='Price',
                hovermode='x unified', template='plotly_dark', legend_title_text='Price Levels'
            )

            # Spread Graph Logic - Clean data before plotting
            spread_col = 'spread'
            if spread_col in product_prices.columns:
                 spread_data = pd.to_numeric(product_prices[spread_col], errors='coerce').ffill().bfill()
                 if not spread_data.isnull().all():
                     fig_spread.add_trace(go.Scatter(
                         x=product_prices['timestamp'], y=spread_data,
                         mode='lines', name='Bid-Ask Spread', line=dict(color='lightyellow')
                     ))
                     fig_spread.update_layout(
                         title=f'{product} - Historical Bid-Ask Spread (Day {selected_day})',
                         xaxis_title='Timestamp', yaxis_title='Spread (Ask1 - Bid1)',
                         hovermode='x unified', template='plotly_dark'
                     )
                 else:
                      fig_spread.update_layout(title=f'{product} Spread (No Valid Data)', template='plotly_dark')
            else:
                 fig_spread.update_layout(title=f'{product} Spread (Column Missing)', template='plotly_dark')


            # Volume Histogram Logic - Clean data before plotting
            has_volume_data = False
            volume_cols = [f'{side}_volume_{i}' for side in ['bid', 'ask'] for i in range(1, 4)]
            colors = px.colors.qualitative.Plotly
            for i, col in enumerate(volume_cols):
                if col in product_prices.columns:
                    try:
                        # Ensure numeric, convert errors, drop NaNs
                        volumes = pd.to_numeric(product_prices[col], errors='coerce').dropna()
                        valid_volumes = volumes[volumes > 0]
                        if not valid_volumes.empty:
                           fig_hist.add_trace(go.Histogram(x=valid_volumes, name=col.replace('_', ' ').title(), marker_color=colors[i % len(colors)], opacity=0.75))
                           has_volume_data = True # Mark that we found some volume
                    except Exception as e:
                         print(f"Error processing/plotting histogram for '{col}', product '{product}': {e}")

            if has_volume_data:
                fig_hist.update_layout(
                    title=f'{product} - Historical Volume Distribution (Day {selected_day})',
                    xaxis_title='Volume per Price Level', yaxis_title='Frequency (Count of Timestamps)',
                    barmode='overlay', template='plotly_dark', legend_title_text='Volume Levels'
                )
                fig_hist.update_traces(opacity=0.75)
            else:
                fig_hist.update_layout(title=f'{product} Volume (No Valid Data)', template='plotly_dark')

        # Add graphs to the children list (stacked layout)
        graph_children.append(html.Div([
            html.Div(dcc.Graph(figure=fig_price), style={'width': '100%', 'marginBottom': '5px'}), # Price graph
            html.Div(dcc.Graph(figure=fig_spread), style={'width': '100%', 'marginBottom': '5px'}), # Spread graph
            html.Div(dcc.Graph(figure=fig_hist), style={'width': '100%'})  # Histogram
        ], style={'marginBottom': '20px', 'border': '1px solid #555', 'padding': '10px'})) # Add border around product section


    return graph_children

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True) 