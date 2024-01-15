__version__ = "0.1.4.8"
__author__="NutInSpace"

from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import concurrent.futures
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
import os
import re
import requests
import pickle
import hashlib
import datetime
import logging

from functools import lru_cache

# Configure logging with precision in the timestamp
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Adding milliseconds for precision
)
logger = logging.getLogger(__name__)

meta = [
    'logId',             # name of log
    'logDate',           # last modified date
    'md5',               # md5 sum of the log
]
columns = [
    'marketId',         # 1  : from log
    'orderId',          # 0  : from log
    'itemType',         # 2  : from log (id of item)
    'buyQuantity',      # 3  : from log
    'expirationDate',   # 4  : from log
    'updateDate',       # 5  : from log
    'unitPrice',        # 6  : from log
    'ownerId',          # 7  : needs cleanup
    'ownerName',        # 8  : fixup 1, from log may contain Entity
    'itemName',         # 9  : fixup 2, lookUp
    'marketName',       # 10 : fixup 3, lookUp
    'orderValue',       # 11 : fixup 4, Maths
    'orderMass',        # 12 : fixup 4, Maths + lookUp
    'orderVol',         # 13 : fixup 4, Maths + lookUp
    'tier',             # 14 : fixup 4, lookUp
    'size',             # 15 : fixup 4, lookUp
    "classId",          #16 : fixup 4, lookUp
    "displayClassId",   #17 : fixup 4, lookup
    'icon',             #18 : fixup 4, lookUp
    ### 'description',      #19 : fixup 4, lookUp
]
markets = {'46174' : 'Market 12', '46129' : 'Market 6', '46136' : 'Aegis'}
num = {'specifier' : ',.2f'}
currency = {'specifier': '$,.2f'}
display_columns = [
    {'id' : 'orderId', 'name' : 'Order Id'},
    {'id' : 'classId', 'name' : 'Class Id'},
    {'id' : 'displayClassId', 'name' : 'Display Id'},
    {'id' : 'logDate', 'name' : 'Log Date'},
    {'id' : 'md5', 'name' : 'Log'},
    {'id' : 'tier', 'name' : 'Tier', 'type': 'numeric'},
    {'id' : 'size', 'name' : 'Size'},
    {'id' : 'itemName', 'name' : 'Item'}, 
    {'id' : 'marketName', 'name' : 'Market'}, 
    {'id' : 'orderMass', 'name' : 'Order Mas (kTon)', 'format': num, 'type': 'numeric'}, 
    {'id' : 'orderVol', 'name' : 'Order Volume (kl)', 'format': num, 'type': 'numeric'}, 
    {'id' : 'buyQuantity', 'name' : 'Order Quantity', 'type': 'numeric', 'format': num},
    {'id' : 'expirationDate', 'name' : 'Expires'},
    {'id' : 'updateDate', 'name' : 'Updated'}, 
    {'id' : 'ownerName', 'name' : 'Seller'},
    {'id' : 'unitPrice', 'name' : 'Unit Price', 'format': currency, 'type': 'numeric'},
    {'id' : 'orderValue', 'name' : 'Order Value', 'format': currency, 'type': 'numeric'}, 
]

metric_columns = [
    {'id' : 'orderType', 'name' : 'Order'},
    {'id' : 'classId', 'name' : 'Class Id'},
    {'id' : 'tier', 'name' : 'Tier', 'type': 'numeric'},
    {'id' : 'size', 'name' : 'Size'},
    {'id' : 'itemName', 'name' : 'Item'}, 
    {'id' : 'Sum Total', 'name' : 'Sum Total', 'format': currency, 'type': 'numeric'},
    {'id' : 'Min Price', 'name' : 'Minimum', 'format': currency, 'type': 'numeric'}, 
    {'id' : 'Weighted Mean', 'name' : 'Average Price', 'format': currency, 'type': 'numeric'},
    {'id' : 'Average Order Volume', 'name' : 'Avg. Order Vol.', 'format': num, 'type': 'numeric'},
    {'id' : 'Max Price', 'name' : 'Maximum', 'format': currency, 'type': 'numeric'},
]

# Regular expression pattern
market_pattern = r'MarketOrder:\[marketId = (\d+), orderId = (\d+), itemType = (\d+), buyQuantity = (.*?), expirationDate = @\(\d+\) (.*?), updateDate = @\(\d+\) (.*?), unitPrice = Currency:\[amount = (\d+).*?](?:, ownerId = EntityId:\[(.*?)\], ownerName = (.*?))?\]'
owner_pattern = r"ownerId = EntityId:\[playerId = (\d+) organizationId = (\d+)\] ownerName = (.+)"

def parse_fix_owner(items):
    fixed = []
    for item in items:
        try:
            item = list(item)
            if "ownerId" in item[5]: 
                data = item[5].split(",")
                item[5] = data[0]
                rem = "".join(data[1:])
                owners = re.findall(owner_pattern, rem)
                owners = list(owners[0])

                # player + org 
                playerId = int(owners[0]) + int(owners[1])
                # shrugs
                playerName = owners[2]
                item[7] = playerId
                item[8] = playerName
                    
            item = tuple(item)
        except Exception as e:
            logger.error(e)
        fixed.append(item)
    return fixed

def parse_add_itemName(items):
    updated = []
    for item in items:
        try:
            item = list(item)
            # itemName, index 9
            item.append(lookupItem(item[2], 'displayNameWithSize'))
            item = tuple(item)
            updated.append(item)
        except Exception as e:
            logger.error(e)
    return updated

@lru_cache(maxsize=None)
def load_items_cache(local_file_path, source_url):
    if os.path.exists(local_file_path):
        logger.info("Loading data from local file")
        with open(local_file_path, 'r') as file:
            return json.load(file)
    else:
        logger.info("Fetching data from the web")
        response = requests.get(source_url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch data from the web: {response.status_code}")
            return None

def lookupItem(id, key='displayNameWithSize', local_file_path='items.json'):
    source_url = 'https://raw.githubusercontent.com/NutInSpace/DualUniverse-GPT/main/items.json'
    #logger.debug(f"lookupItem: {id}")

    items_cache = load_items_cache(local_file_path, source_url)

    if items_cache is None:
        return "Error fetching data", None

    # Search in the cached data
    for search in items_cache:
        if str(search.get('id')) == str(id):
            return search.get(key, f"Unknown {key}")

    return f"Unknown {key}", items_cache
 
def parse_add_itemName(items):
    updated = []
    for item in items:
        item = list(item)
        # itemName, index 9
        item.append(lookupItem(item[2]))
        item = tuple(item)
        updated.append(item)
    return updated
 
def parse_add_orderValue(items):
    updated = []
    for item in items:
        item = list(item)
        # buyQuantity * unitPrice
        value = item[3] * item[6] / 100
        item.append(value)
        item = tuple(item)
        updated.append(item)
    return updated

def parse_add_marketName(items):
    updated = []
    for item in items:
        item = list(item)
        try:
            name = markets[item[0]]
            item.append(name)
        except KeyError:
            item.append("UnknownMarket")
        item = tuple(item)
        updated.append(item)
    return updated

def parse_add_orderValue(items):
    updated = []
    progress = 0
    logger.info("Processing items :")
    for item in items:
        # Track progress
        progress += 1
        print(f" Progress: {(progress/len(items)*100):.2f} %\r",end="")
        
        item = list(item)
        i = item[2] 
        q = item[3]
        
        # 11) orderValue = buyQuantity * unitPrice
        value = int(q) * int(item[6]) / 100
        item.append(value)
        
        # 12) orderMass = buyQuantity * unitMass  
        # 13) orderVol = buyQuantity * unitVol  
        unit_mass = lookupItem(i, 'unitMass')
        unit_vol = lookupItem(i, 'unitVolume')
        mass = int(q) * float(unit_mass) / 1000
        volume = int(q) * float(unit_vol) / 1000
        item.append(mass)
        item.append(volume)

        # 14) tier
        tier = lookupItem(i, 'tier')
        item.append(tier)
        
        # 15) size
        size = lookupItem(i, 'size')
        item.append(size)
        
        # 16) classId
        classId = lookupItem(i, 'classId')
        item.append(classId)
        
        # 17) displayClassId
        displayClassId = lookupItem(i, 'displayClassId')
        item.append(displayClassId)
        
        # 18) icon
        icon = lookupItem(i, 'iconPath')
        item.append(icon)
        
        # 19) description
        ##desc = lookupItem(i, 'description')
        ###item.append(desc)
                
        # Repackage
        item = tuple(item)
        updated.append(item)
    return updated
       
def process_file(file_path):
    
    compiled_pattern = re.compile(market_pattern)
    matches = []
    logger.info(f"parer: reading {file_path}")
    with open(file_path, 'r') as file:
        content = file.read()
        logger.info(f"parser: matching {file_path}")
        matches.extend(re.findall(market_pattern, content))
        
    # Fixup 1, Index 8
    logger.info(f"parer fixup-1: {len(matches)} items")
    matches = parse_fix_owner(matches)
    
    # Fixup 2, Index 9
    logger.info(f"parser fixup-2: {len(matches)} items")
    matches = parse_add_itemName(matches)
    
    # Fixup 3, Index 10
    logger.info(f"parer fixup-3:  {len(matches)} items")
    matches = parse_add_marketName(matches)
    
    # Fixup 9, Index 11
    logger.info(f"parer fixup-4:  {len(matches)} items")
    matches = parse_add_orderValue(matches)

    return matches

def generate_md5(filename):
    """ Generate MD5 checksum for a file. """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_file_data(file_data, log_file):
    df = pd.DataFrame(file_data, columns=columns)
    df = df.drop_duplicates()
    df['logId'] = os.path.basename(log_file)
    # Convert timestamp to human-readable format
    timestamp = os.path.getmtime(log_file)
    human_readable_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    df['logDate'] = human_readable_date

    df['md5'] = generate_md5(log_file)
    return df

def load_or_create_cache(file_path, cache_path):
    if os.path.exists(cache_path):
        logger.info(f"Loading cache {cache_path}")
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    
    file_data = process_file(file_path)
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(file_data, cache_file)
    logger.info(f"Saved Cache file {cache_path}")
    return file_data

import concurrent.futures

def process_log_file(log_file, cache_folder_path):
    checksum = generate_md5(log_file)
    cache_file_path = os.path.join(cache_folder_path, checksum + __version__.replace(".", "_") + '.pkl')
    file_data = load_or_create_cache(log_file, cache_file_path)
    
    df = pd.DataFrame(file_data, columns=columns)
    df = df.drop_duplicates()
    df['logId'] = os.path.basename(log_file)
    # Convert timestamp to human-readable format
    timestamp = os.path.getmtime(log_file)
    human_readable_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    df['logDate'] = human_readable_date
    df['md5'] = checksum
    return df

def create_dataframe_from_log_files_multithreaded(log_files, cache_folder_path):
    frames = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submitting all log files to the executor
        future_to_file = {executor.submit(process_log_file, log_file, cache_folder_path): log_file for log_file in log_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            log_file = future_to_file[future]
            try:
                df = future.result()
                frames.append(df)
                logger.debug(f"Processed file {os.path.basename(log_file)}")
            except Exception as exc:
                logger.error(f"File {os.path.basename(log_file)} generated an exception: {exc}")

    return pd.concat(frames, ignore_index=True)

def update():
    logger.info("Starting update process")

    # Setup directories
    cwd = os.getcwd()
    cache_folder_path = os.path.join(cwd, 'cache')
    os.makedirs(cache_folder_path, exist_ok=True)
    logger.info(f"Cache directory: {cache_folder_path}")

    log_directory = os.path.expandvars(r'%localappdata%\NQ\DualUniverse\log')
    log_files = [os.path.join(log_directory, lf) for lf in os.listdir(log_directory)]
    logger.info(f"Log Directory: {log_directory}")
    logger.info("Log files found")

    # Process log files and create DataFrame
    final_df = create_dataframe_from_log_files_multithreaded(log_files, cache_folder_path)

    # Clean
    final_df.drop_duplicates(inplace=True)

    # Save the DataFrame
    final_df.to_csv('market_orders.csv', index=False)
    logger.info("DataFrame saved as 'market_orders.csv'")

es = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize the Dash app with suppressed callback exceptions
app = Dash("Dual Universe Market Orders mk1", external_stylesheets=es, suppress_callback_exceptions=True)

# Matrix Reloaded style theme
matrix_style = {
    'backgroundColor': '#110000',
    'color': '#00FF00',
    'fontFamily': 'Monospace'
}

tab_selected_style  = {
    'borderBottom': '1px solid #00FF00',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_style = {
    'borderTop': '1px solid #00FF00',
    'borderBottom': '1px solid #00FF00',
    'backgroundColor': '#110000',
    'color': 'white',
    'padding': '6px'
}

# Tabs definition
tabs1 = dcc.Tabs(
    id='tabs', 
    style=matrix_style, 
    value='buy-tab', 
    children=[
        dcc.Tab(label='All Orders', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Buy Orders', value='buy-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Sell Orders', value='sell-tab', style=tab_style, selected_style=tab_selected_style)
    ]
)

tabs2 = dcc.Tabs(
    id='tabs2', 
    style=matrix_style, 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Tiers', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Basic', value='basic-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Uncommon', value='uncommon-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Advanced', value='advanced-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Rare', value='rare-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Exotic', value='exotic-tab', style=tab_style, selected_style=tab_selected_style),
    ]
)

tabs3 = dcc.Tabs(
    id='tabs3', 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Sizes', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Unit', value='liter-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='XSmall', value='xsmall-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Small', value='small-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Medium', value='medium-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Large', value='large-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='XLarge', value='xlarge-tab', style=tab_style, selected_style=tab_selected_style),
    ],
    style=matrix_style
)

tabs4 = dcc.Tabs(
    id='tabs4', 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Markets', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Market 6', value='market6-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Market 12', value='market12-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Aegis', value='aegis-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Other', value='other-tab', style=tab_style, selected_style=tab_selected_style),
    ],
    style=matrix_style
)

tabs5 = dcc.Tabs(
    id='tabs5', 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Items', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Ore', value=312610052, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Pure', value=1161217162, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Product', value=2533089091, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Container', value=703994582, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Industry', value=3943695040, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Engine', value=2294552479, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Other', value='other-tab', style=tab_style, selected_style=tab_selected_style),
    ],
    style=matrix_style
)

# Sidebar style
sidebar_style = {
    'backgroundColor': '#111111',
    'color': '#00FF00',
    'padding': '1px',
    'width': '1%',
    'display': 'flex',
    'flexDirection': 'column',
    'height': '100vh',
    'overflowX': 'auto'
}

# Content style
content_style = {
    'margin-left': '1%',
    'margin-right': '1%',
    'padding': '2px',
    'width': '98%',
    'backgroundColor': '#111111',
    'color': '#00FF00',
    'fontFamily': 'Monospace'
}

# Layout of the app
app.layout = html.Div(style=matrix_style, children=[
    dcc.Store(id='intermediate-value'),
        dcc.Checklist(
            id='toggle-expired',
            options=[{'label': ' Hide Expired Orders', 'value': 'hide_expired'}],
            value=[]
        ),
    html.Div(id='content', style=content_style, children=[
        html.H1("Dual Universe Market Intel"),
        html.H3(f"Build 1/13/2024 Version: {__version__}"),
        html.H4("Market Filters (Location, Order Type, Tier, Size, Item Class)"),
        tabs4,
        tabs1,
        tabs2,
        tabs3,
        tabs5,
        html.H4("Metrics Report"),
        dash_table.DataTable(
            id='metrics-table',
            columns=metric_columns,
            data=[], #init-empty
            filter_action='native',
            sort_action='native',
            page_action='native',
            page_size=20,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#1c1c1c', 'color': '#00FF00'},
                {'if': {'column_id': 'some_column'}, 'backgroundColor': '#262626', 'color': '#00FF00'}
            ],
            style_cell={
                'backgroundColor': '#111111',
                'color': '#00FF00',
                'border': '1px solid #00FF00'
            }
        ),
        html.H3("Orders Data"),
        dash_table.DataTable(
            id='table',
            columns=display_columns,
            data=[],
            filter_action='native',
            sort_action='native',
            page_action='native',
            page_size=20,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#1c1c1c', 'color': '#00FF00'},
                {'if': {'column_id': 'some_column'}, 'backgroundColor': '#262626', 'color': '#00FF00'}
            ],
            style_cell={
                'backgroundColor': '#111111',
                'color': '#00FF00',
                'border': '1px solid #00FF00'
            }
        ),
    ])
])

# Callbacks
@app.callback(
    Output('intermediate-value', 'data'),
    [Input('toggle-expired', 'value')],
    prevent_initial_call=False
)
def load_clean_data(toggle_expired):
    # update()  # Rebuild the CSV, parsing live game data
    df = pd.read_csv('market_orders.csv')
    df['Value'] = df['orderValue'].apply(lambda x: f"${x:,.2f}")
    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    df['unitPrice'] = df['unitPrice'] / 100
    if toggle_expired and 'hide_expired' in toggle_expired:
        update()
        current_date_time = pd.Timestamp.now()
        df = df[df['expirationDate'] >= current_date_time]
    #if 'hide_spam' in toggle_spam:
    df = df[(abs(df['unitPrice']) > 2) & (abs(df['orderValue']) > 1000)]
    
    # Fix
    df.fillna({'size': 'Unit'}, inplace=True)
    df.fillna({'ownerName': 'OwnerUnknown'}, inplace=True)
    df.fillna(value=0, inplace=True)  # Replaces all NaN values with 0
    
    # Add tracker for order type
    df['orderType'] = ['Buy' if qty > 0 else 'Sell' for qty in df['buyQuantity']]
    
    #logger.debug("clean orders:")
    #logger.debug(df)
    
    # Convert updateDate to datetime and extract the date
    df['updateDateDay'] = pd.to_datetime(df['updateDate']).dt.date
    
    # Cache local file
    df.to_csv("clean_orders.csv", index=False)
    
    return df.to_json(index=False)

@app.callback(
    Output('table', 'data'),
    [Input('tabs', 'value'), Input('tabs2', 'value'), Input('tabs3', 'value'), Input('tabs4', 'value'), Input('tabs5', 'value')],
    prevent_initial_call=False
)
def update_table(tab_value, tier_tab_value, size_tab_value, market_tab_value, class_tab_value):
    if os.path.exists('clean_orders.csv'):
        logger.debug("table-update: cache reload")
        df = pd.read_csv('clean_orders.csv')
        #logger.debug(df)
    else:
        logger.debug("table-update: full reload")
        df = load_clean_data(None)
        #logger.debug(df)
    
    # Filter Order
    if tab_value == 'buy-tab':
        filtered_df = df[df['buyQuantity'] > 0]
    elif tab_value == 'sell-tab':   
        filtered_df = df[df['buyQuantity'] <= 0]
    else:
        filtered_df = df
    # Save
    df = filtered_df
    
    # Filter Tier
    tab_value = tier_tab_value
    if tab_value == 'all-tab':
        pass #return df.to_dict('records')
    elif tab_value == 'basic-tab':   
        filtered_df = df[df['tier'] == 1]
    elif tab_value == 'uncommon-tab':   
        filtered_df = df[df['tier'] == 2]
    elif tab_value == 'advanced-tab':   
        filtered_df = df[df['tier'] == 3]
    elif tab_value == 'rare-tab':   
        filtered_df = df[df['tier'] == 4]
    elif tab_value == 'exotic-tab':   
        filtered_df = df[df['tier'] == 5]
    # Save
    df = filtered_df
        
    # Define a mapping between tab values and market IDs
    market_id_map = {
        'market6-tab': 46129,
        'market12-tab': 46174,
        'aegis-tab': 46136
    }
    tab_value = market_tab_value

    if tab_value == 'all-tab':
        pass  # Use df as is
    elif tab_value in market_id_map:
        # Filter df based on the market ID from the map
        filtered_df = df[df['marketId'] == market_id_map[tab_value]]
    elif tab_value == 'other-tab':
        # Filter out the specified market IDs
        excluded_ids = [market_id_map['aegis-tab'], market_id_map['market12-tab'], market_id_map['market6-tab']]
        filtered_df = df[~df['marketId'].isin(excluded_ids)]
    else:
        filtered_df = df  # Default case if tab_value is not recognized
        
    tab_choices = [312610052, 1161217162, 2533089091, 703994582, 3943695040, 2294552479]
    tab_value = class_tab_value
    if tab_value == 'all-tab':
        pass  # Use df as is
    elif tab_value in tab_choices:
        # Filter df based on the market ID from the map
        filtered_df = df[df['classId'] == tab_value]
    elif tab_value == 'other-tab':
        filtered_df = df[~df['classId'].isin(tab_choices)]
    else:
        filtered_df = df  # Default case if tab_value is not recognized

    # Save
    df = filtered_df
    
    # Filter Size
    tab_value = size_tab_value
    if tab_value == 'all-tab':
        pass #return df.to_dict('records')
    elif tab_value == 'liter-tab':   
        filtered_df = df[df['size'].isin(["Unit", "L", "unit", "liter"])]
    elif tab_value == 'xsmall-tab':   
        filtered_df = df[df['size'].isin(["xsmall", "xs"])]
    elif tab_value == 'small-tab':   
        filtered_df = df[df['size'].isin(["small", "s"])]
    elif tab_value == 'medium-tab':   
        filtered_df = df[df['size'].isin(["medium", "m"])]
    elif tab_value == 'large-tab':   
        filtered_df = df[df['size'].isin(["large", "l"])]
    elif tab_value == 'xlarge-tab':   
        filtered_df = df[df['size'].isin(["xlarge", "xxlarge", "xl", "xxl"])]
    # Save
    df = filtered_df
        
    # Cache Local File
    filtered_df.to_csv('filtered_clean_orders.csv')

    return filtered_df.to_dict('records')

@app.callback(
    Output('metrics-table', 'data'),
    [Input('table', 'data')]
)
def update_metrics(data):
    df = pd.DataFrame(data)
    if df.empty:
        logger.info("Warning: empty dataframe in update_metrics")
        return None

    # Filter duplicate orders
    df.drop_duplicates(inplace=True, subset='orderId')

    # Grouping by 'itemName', 'itemType', 'size', and 'tier'
    grouped_df = df.groupby(['classId', 'orderType', 'itemName', 'itemType', 'size', 'tier'])

    # Use agg to perform all aggregations in one step
    aggregated_df = grouped_df.agg({
        'orderValue': 'sum',
        'unitPrice': ['min', 'max'],
        'orderVol': 'mean'
    }).reset_index()

    # Rename columns for clarity
    aggregated_df.columns = ['classId', 'orderType', 'itemName', 'itemType', 'size', 'tier', 'Sum Total', 'Min Price', 'Max Price', 'Average Order Volume']

    # Calculate the weighted mean for each group using vectorized operations
    weighted_sums = (df['unitPrice'] * df['buyQuantity']).groupby([df['classId'], df['orderType'], df['itemName'], df['itemType'], df['size'], df['tier']]).sum()
    total_quantity = df['buyQuantity'].groupby([df['classId'], df['orderType'], df['itemName'], df['itemType'], df['size'], df['tier']]).sum()
    weighted_mean = weighted_sums / total_quantity
    weighted_mean_df = weighted_mean.reset_index(name='Weighted Mean')

    # Merging the results into a single DataFrame
    results_df = pd.merge(aggregated_df, weighted_mean_df, on=['classId', 'orderType', 'itemName', 'itemType', 'size', 'tier'])

    # Sorting the results
    results_df = results_df.sort_values(by='tier')
    return results_df.to_dict('records')


# Run the app
if __name__ == '__main__':
    #update()
    app.run_server()
