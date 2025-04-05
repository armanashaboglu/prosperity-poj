import dash
from dash import dcc, html, Input, Output, State, dash_table
# from plotly_resampler import FigureResampler as go # Reverting this
import plotly.graph_objects as go # Use standard go
import plotly.express as px
import pandas as pd
import json
from .log_parser import parse_log_file # Use relative import

# --- Configuration ---
LOG_FILE_PATH = 'C:/Users/Admin/projects/prosperity-poj/strategy/tutorial/prototype.log' # Or make this configurable
POSITION_LIMITS = { # Hardcode for now, ideally load dynamically later
    "KELP": 50,
    "RAINFOREST_RESIN": 50
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

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True) # Suppress exceptions for initial layout
server = app.server # Expose server variable for deployments

# --- App Layout ---
app.layout = html.Div([
    html.H1("IMC Prosperity Log Visualizer"),

    # Graphs Row 1 (Price & Volume)
    html.Div([
        html.Div(dcc.Graph(id='kelp-price-graph'), style={'width': '49%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='kelp-volume-graph'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Div(dcc.Graph(id='resin-price-graph'), style={'width': '49%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='resin-volume-graph'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
    ], style={'marginBottom': '20px'}),

    # Graphs Row 2 (Positions & PnL)
    html.Div([
         html.Div(dcc.Graph(id='position-graph'), style={'width': '49%', 'display': 'inline-block'}),
         html.Div(dcc.Graph(id='pnl-graph'), style={'width': '49%', 'display': 'inline-block', 'float': 'right'}) # Add PnL graph
    ], style={'marginBottom': '20px'}),

    # Timestamp Controls (Moved Below Graphs)
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
        html.Div(id='timestamp-display') # To show the selected timestamp clearly
    ], style={'marginBottom': '40px', 'marginTop': '20px'}), # Added spacing

    # Data Display Row (Trader Data, LOBs, Trades, etc.)
    html.Div([
        html.H3("State at Selected Timestamp"),
        # Left Column: Trader Data & LOBs
        html.Div([
            html.H4("Trader Data"),
            html.Pre(id='trader-data-display', style={'border': '1px solid lightgrey', 'padding': '10px', 'overflowX': 'scroll', 'maxHeight': '300px'}), # Added max height
            html.Hr(),
            html.Div([
                 html.H4("KELP Order Book", style={'display': 'inline-block', 'marginRight': '10px'}),
                 html.Div(id='kelp-lob-display', style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '45%'}) # Placeholder for KELP LOB
            ]),
             html.Div([
                 html.H4("RESIN Order Book", style={'display': 'inline-block', 'marginRight': '10px'}),
                 html.Div(id='resin-lob-display', style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '45%'}) # Placeholder for RESIN LOB
            ]),
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

    # dcc.Store(id='selected-timestamp-store', data=min_ts) # Maybe use later if needed

], style={'padding': '20px'})

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

# Callback for Price Graphs (full history, Scattergl, reduced initial visibility, trade markers)
@app.callback(
    [Output('kelp-price-graph', 'figure'),
     Output('resin-price-graph', 'figure')],
    [Input('timestamp-input', 'value')] # Trigger on timestamp change
)
def update_price_graphs(selected_ts):
    # Create figures using standard go
    fig_kelp = go.Figure()
    fig_resin = go.Figure()

    # Merge calculated fair values from lambda_state_df into activities_df for plotting
    # We need to handle the dictionary structure in 'calculated_fair_values'
    # Create temporary fair value columns for easier merging
    lambda_state_df['fair_value_KELP'] = lambda_state_df['calculated_fair_values'].apply(lambda x: x.get('KELP'))
    lambda_state_df['fair_value_RAINFOREST_RESIN'] = lambda_state_df['calculated_fair_values'].apply(lambda x: x.get('RAINFOREST_RESIN'))
    
    # Select only necessary columns for merge to avoid conflicts
    fair_value_data = lambda_state_df[['timestamp', 'fair_value_KELP', 'fair_value_RAINFOREST_RESIN']]
    
    # Merge with activities_df - use outer join to keep all timestamps
    activities_with_fair = pd.merge(activities_df, fair_value_data, on='timestamp', how='left')

    # --- KELP --- 
    kelp_price_data = activities_with_fair[activities_with_fair['product'] == 'KELP'] 
    kelp_trade_data = trade_history_df[trade_history_df['symbol'] == 'KELP']
    
    if not kelp_price_data.empty:
        # Plot price levels using Scattergl
        # Bids (Bid 1 visible, 2 & 3 hidden)
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['bid_price_1'], mode='lines', name='Bid 1', line=dict(color='lightgreen')))
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['bid_price_2'], mode='lines', name='Bid 2', line=dict(color='lightgreen', dash='dot'), visible='legendonly'))
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['bid_price_3'], mode='lines', name='Bid 3', line=dict(color='lightgreen', dash='dashdot'), visible='legendonly'))
        # Asks (Ask 1 visible, 2 & 3 hidden)
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['ask_price_1'], mode='lines', name='Ask 1', line=dict(color='lightcoral')))
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['ask_price_2'], mode='lines', name='Ask 2', line=dict(color='lightcoral', dash='dot'), visible='legendonly'))
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['ask_price_3'], mode='lines', name='Ask 3', line=dict(color='lightcoral', dash='dashdot'), visible='legendonly'))
        # Mid Price (Visible)
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['mid_price'], mode='lines', name='Mid Price', line=dict(color='lightblue')))

        # Add Fair Value trace (initially hidden)
        fig_kelp.add_trace(go.Scattergl(x=kelp_price_data['timestamp'], y=kelp_price_data['fair_value_KELP'], mode='lines', name='Fair Value (Algo)', line=dict(color='orange', dash='longdash'), visible='legendonly'))

        # Add Trade Markers (using trade_history_df) - hidden initially
        kelp_own_buy = kelp_trade_data[kelp_trade_data['buyer'] == 'SUBMISSION']
        kelp_own_sell = kelp_trade_data[kelp_trade_data['seller'] == 'SUBMISSION']
        kelp_market_trades = kelp_trade_data[(kelp_trade_data['buyer'] != 'SUBMISSION') & (kelp_trade_data['seller'] != 'SUBMISSION')]

        fig_kelp.add_trace(go.Scattergl(x=kelp_own_buy['timestamp'], y=kelp_own_buy['price'], mode='markers', name='Own Buy', marker=dict(color='cyan', symbol='triangle-up', size=8), visible='legendonly', hovertemplate="Buy Qty: %{customdata}<extra></extra>", customdata=kelp_own_buy['quantity']))
        fig_kelp.add_trace(go.Scattergl(x=kelp_own_sell['timestamp'], y=kelp_own_sell['price'], mode='markers', name='Own Sell', marker=dict(color='yellow', symbol='triangle-down', size=8), visible='legendonly', hovertemplate="Sell Qty: %{customdata}<extra></extra>", customdata=kelp_own_sell['quantity']))
        fig_kelp.add_trace(go.Scattergl(x=kelp_market_trades['timestamp'], y=kelp_market_trades['price'], mode='markers', name='Market Trades', marker=dict(color='magenta', symbol='x', size=6), visible='legendonly', hovertemplate="Market Qty: %{customdata}<extra></extra>", customdata=kelp_market_trades['quantity']))

        # Add vertical line for selected timestamp
        fig_kelp.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")

        fig_kelp.update_layout(
            title='KELP Price Levels & Trades',
            xaxis_title='Timestamp', 
            yaxis_title='Price',
            hovermode='x unified', 
            template='plotly_dark'
        )

    # --- RAINFOREST_RESIN --- 
    resin_price_data = activities_with_fair[activities_with_fair['product'] == 'RAINFOREST_RESIN']
    resin_trade_data = trade_history_df[trade_history_df['symbol'] == 'RAINFOREST_RESIN']
    
    if not resin_price_data.empty:
        # Plot price levels using Scattergl
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['bid_price_1'], mode='lines', name='Bid 1', line=dict(color='lightgreen')))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['bid_price_2'], mode='lines', name='Bid 2', line=dict(color='lightgreen', dash='dot'), visible='legendonly'))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['bid_price_3'], mode='lines', name='Bid 3', line=dict(color='lightgreen', dash='dashdot'), visible='legendonly'))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['ask_price_1'], mode='lines', name='Ask 1', line=dict(color='lightcoral')))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['ask_price_2'], mode='lines', name='Ask 2', line=dict(color='lightcoral', dash='dot'), visible='legendonly'))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['ask_price_3'], mode='lines', name='Ask 3', line=dict(color='lightcoral', dash='dashdot'), visible='legendonly'))
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['mid_price'], mode='lines', name='Mid Price', line=dict(color='lightblue')))

        # Add Fair Value trace (initially hidden)
        fig_resin.add_trace(go.Scattergl(x=resin_price_data['timestamp'], y=resin_price_data['fair_value_RAINFOREST_RESIN'], mode='lines', name='Fair Value (Algo)', line=dict(color='orange', dash='longdash'), visible='legendonly'))

        # Add Trade Markers - hidden initially
        resin_own_buy = resin_trade_data[resin_trade_data['buyer'] == 'SUBMISSION']
        resin_own_sell = resin_trade_data[resin_trade_data['seller'] == 'SUBMISSION']
        resin_market_trades = resin_trade_data[(resin_trade_data['buyer'] != 'SUBMISSION') & (resin_trade_data['seller'] != 'SUBMISSION')]

        fig_resin.add_trace(go.Scattergl(x=resin_own_buy['timestamp'], y=resin_own_buy['price'], mode='markers', name='Own Buy', marker=dict(color='cyan', symbol='triangle-up', size=8), visible='legendonly', hovertemplate="Buy Qty: %{customdata}<extra></extra>", customdata=resin_own_buy['quantity']))
        fig_resin.add_trace(go.Scattergl(x=resin_own_sell['timestamp'], y=resin_own_sell['price'], mode='markers', name='Own Sell', marker=dict(color='yellow', symbol='triangle-down', size=8), visible='legendonly', hovertemplate="Sell Qty: %{customdata}<extra></extra>", customdata=resin_own_sell['quantity']))
        fig_resin.add_trace(go.Scattergl(x=resin_market_trades['timestamp'], y=resin_market_trades['price'], mode='markers', name='Market Trades', marker=dict(color='magenta', symbol='x', size=6), visible='legendonly', hovertemplate="Market Qty: %{customdata}<extra></extra>", customdata=resin_market_trades['quantity']))
        
        # Add vertical line for selected timestamp
        fig_resin.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")

        fig_resin.update_layout(
            title='RESIN Price Levels & Trades',
            xaxis_title='Timestamp',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark'
        )

    return fig_kelp, fig_resin

# Callback for Position Graph 
@app.callback(
    Output('position-graph', 'figure'),
    Input('timestamp-input', 'value') # Trigger on timestamp change for vertical line
)
def update_position_graph(selected_ts):
    fig = go.Figure()
    
    # Plot positions for each product
    for product, limit in POSITION_LIMITS.items():
        # Extract position for this product - careful with missing keys
        positions = lambda_state_df['positions'].apply(lambda p: p.get(product, 0)) # Default to 0 if product not in dict
        # Calculate position as % of limit
        position_pct = (positions / limit) * 100
        
        fig.add_trace(go.Scattergl(
            x=lambda_state_df['timestamp'], 
            y=position_pct, 
            mode='lines', 
            name=f'{product} Position (% Limit)'
        ))
        
    # Add vertical line for selected timestamp
    fig.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")
    
    fig.update_layout(
        title='Product Positions (% of Limit)',
        xaxis_title='Timestamp', 
        yaxis_title='% of Position Limit',
        hovermode='x unified', 
        template='plotly_dark',
        yaxis_range=[-105, 105] # Set fixed y-axis range slightly beyond +/- 100%
    )
    return fig

# Callback for PnL Graph
@app.callback(
    Output('pnl-graph', 'figure'),
    Input('timestamp-input', 'value') # Trigger on timestamp change for vertical line
)
def update_pnl_graph(selected_ts):
    fig = go.Figure()
    total_pnl_col = 'Total PnL'
    
    # Calculate Total PnL across products per timestamp
    pnl_data = activities_df.pivot(index='timestamp', columns='product', values='profit_and_loss')
    pnl_data[total_pnl_col] = pnl_data.sum(axis=1)
    pnl_data = pnl_data.reset_index() # Make timestamp a column again

    # Plot PnL for each product
    for product in activities_df['product'].unique():
        fig.add_trace(go.Scattergl(
            x=pnl_data['timestamp'], 
            y=pnl_data[product], 
            mode='lines', 
            name=f'{product} PnL',
            visible='legendonly' # Hide individual PnLs initially
        ))
        
    # Plot Total PnL (visible by default)
    fig.add_trace(go.Scattergl(
        x=pnl_data['timestamp'], 
        y=pnl_data[total_pnl_col], 
        mode='lines', 
        name=total_pnl_col,
        line=dict(color='gold', width=3)
    ))
        
    # Add vertical line for selected timestamp
    fig.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="grey")
    
    fig.update_layout(
        title='Profit & Loss Over Time',
        xaxis_title='Timestamp', 
        yaxis_title='PnL',
        hovermode='x unified', 
        template='plotly_dark'
    )
    return fig

# Combined callback to update all timestamp-specific details
@app.callback(
    [Output('trader-data-display', 'children'),
     Output('kelp-lob-display', 'children'),
     Output('resin-lob-display', 'children'),
     Output('orders-display', 'children'),
     Output('own-trades-display', 'children'),
     Output('market-trades-display', 'children'),
     Output('algo-logs-display', 'children'),
     Output('current-state-display', 'children')],
    Input('timestamp-input', 'value')
)
def update_timestamp_details(selected_ts):
    if selected_ts is None or selected_ts not in lambda_state_df['timestamp'].values:
        no_data_msg = "Select a valid timestamp."
        # Return message for all outputs
        return [no_data_msg] * 8 

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
            trader_data_dict = json.loads(trader_data_str)
            trader_data_display = json.dumps(trader_data_dict, indent=4)
        except Exception as e:
            trader_data_display = f"Error decoding Trader Data: {e}\n{trader_data_str}"

    # --- 2. LOB Displays --- 
    lob_tables = {}
    for product in ['KELP', 'RAINFOREST_RESIN']:
        if product not in order_depths:
            lob_tables[product] = html.P(f"No order depth data for {product}")
            continue
        
        buy_orders = order_depths[product][0]  
        sell_orders = order_depths[product][1] 

        buy_levels = sorted([(int(p), v) for p, v in buy_orders.items()], reverse=True) 
        sell_levels = sorted([(int(p), -v) for p, v in sell_orders.items()]) # Lowest ask first

        depth = 5 
        data_for_table = []
        
        # Add top N asks in REVERSE order (highest price first)
        for i in range(min(depth, len(sell_levels)) - 1, -1, -1):
            price, volume = sell_levels[i]
            data_for_table.append({'Bid Vol': '', 'Price': price, 'Ask Vol': volume})
                
        # Add top N bids (highest price first)
        for i in range(min(depth, len(buy_levels))):
            price, volume = buy_levels[i]
            data_for_table.append({'Bid Vol': volume, 'Price': price, 'Ask Vol': ''})

        if not data_for_table:
             lob_tables[product] = html.P(f"Empty order book for {product}")
             continue

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
        lob_tables[product] = table
    
    kelp_lob_display = lob_tables.get('KELP', html.P("Error generating KELP LOB."))
    resin_lob_display = lob_tables.get('RAINFOREST_RESIN', html.P("Error generating RESIN LOB."))

    # --- 3. Orders Table --- 
    orders_display = "No orders sent at this timestamp."
    if orders_list:
        try:
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
        except Exception as e:
            orders_display = f"Error creating orders table: {e}"

    # --- 4. Own Trades Table --- 
    own_trades_display = "No own trades at this timestamp."
    if own_trades_list:
        try:
            # Expected structure: [symbol, price, quantity, buyer, seller, timestamp]
            own_trades_df = pd.DataFrame(own_trades_list, columns=['Symbol', 'Price', 'Quantity', 'Buyer', 'Seller', 'Trade Ts'])
            own_trades_df['Type'] = own_trades_df.apply(lambda row: 'BUY' if row['Buyer'] == 'SUBMISSION' else 'SELL', axis=1)
            # Include 'Trade Ts' in the selection
            own_trades_df = own_trades_df[['Trade Ts', 'Symbol', 'Type', 'Price', 'Quantity']]
            own_trades_display = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in own_trades_df.columns],
                data=own_trades_df.to_dict('records'),
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
                style_data={'color': 'black','backgroundColor': 'white'}
            )
        except Exception as e:
            own_trades_display = f"Error creating own trades table: {e}"
            
    # --- 5. Market Trades Table ---
    market_trades_display = "No market trades at this timestamp."
    if market_trades_list:
        try:
             # Expected structure: [symbol, price, quantity, buyer, seller, timestamp]
            market_trades_df = pd.DataFrame(market_trades_list, columns=['Symbol', 'Price', 'Quantity', 'Buyer', 'Seller', 'Trade Ts'])
            # Include 'Trade Ts' in the selection
            market_trades_df = market_trades_df[['Trade Ts', 'Symbol', 'Price', 'Quantity', 'Buyer', 'Seller']]
            market_trades_display = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in market_trades_df.columns],
                data=market_trades_df.to_dict('records'),
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
                style_data={'color': 'black','backgroundColor': 'white'}
            )
        except Exception as e:
            market_trades_display = f"Error creating market trades table: {e}"

    # --- 6. Algo Logs --- 
    algo_logs_display = logs_str if logs_str else "No algorithm logs for this timestamp."

    # --- 7. Current State --- 
    current_pnl_dict = {}
    pnl_row = activities_df[activities_df['timestamp'] == selected_ts]
    if not pnl_row.empty:
        for product in pnl_row['product'].unique():
            product_pnl = pnl_row[pnl_row['product'] == product]['profit_and_loss'].iloc[0]
            current_pnl_dict[product] = round(product_pnl, 2)
            
    current_state_lines = []
    current_state_lines.append("Positions:")
    for product, pos in positions_dict.items():
         limit = POSITION_LIMITS.get(product, 'N/A')
         current_state_lines.append(f"  {product}: {pos} / {limit}")
    if not positions_dict: current_state_lines.append("  (None)")
        
    current_state_lines.append("\nPnL at Timestamp:")
    for product, pnl in current_pnl_dict.items(): current_state_lines.append(f"  {product}: {pnl}")
    if not current_pnl_dict: current_state_lines.append("  (N/A)")
         
    current_state_display = "\n".join(current_state_lines)

    # --- Return all updated components ---
    return (
        trader_data_display,
        kelp_lob_display,
        resin_lob_display,
        orders_display,
        own_trades_display,
        market_trades_display,
        algo_logs_display,
        current_state_display
    )

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True) 