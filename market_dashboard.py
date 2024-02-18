
__version__ = "0.1.6.4"
__author__= "NutInSpace"

import cProfile
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
sortby = SortKey.CUMULATIVE

from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import plotly.express as px
import scipy.stats as stats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import json
import os
import re
import requests
import pickle
import hashlib
import datetime
import logging

from functools import lru_cache

import aiohttp
import asyncio

# Globals
items_cache_json = None
known_items = {}

# Configure logging with precision in the timestamp
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to DEBUG
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
    #{'id' : 'displayClassId', 'name' : 'Display Id'},
    #{'id' : 'logDate', 'name' : 'Log Date'},
    #{'id' : 'md5', 'name' : 'Log'},
    #{'id' : 'tier', 'name' : 'Tier', 'type': 'numeric'},
    #{'id' : 'size', 'name' : 'Size'},
    {'id' : 'itemName', 'name' : 'Item'}, 
    {'id' : 'marketName', 'name' : 'Market'}, 
    {'id' : 'orderMass', 'name' : 'Order Mass (kTon)', 'format': num, 'type': 'numeric'}, 
    {'id' : 'orderVol', 'name' : 'Order Volume (kl)', 'format': num, 'type': 'numeric'}, 
    {'id' : 'buyQuantity', 'name' : 'Order Quantity', 'type': 'numeric', 'format': num},
    {'id' : 'expirationDate', 'name' : 'Expires'},
    {'id' : 'updateDate', 'name' : 'Updated'}, 
    {'id' : 'ownerName', 'name' : 'Seller'},
    {'id' : 'unitPrice', 'name' : 'Unit Price', 'format': currency, 'type': 'numeric'},
    {'id' : 'orderValue', 'name' : 'Order Value', 'format': currency, 'type': 'numeric'}, 
]

metric_columns = [
    {'id' : 'trend', 'name' : 'Trend','format': num, 'type': 'numeric'},
    {'id' : 'updateDateDay', 'name' : 'Updated'}, 
    {'id' : 'orderType', 'name' : 'Order'},
    #{'id' : 'classId', 'name' : 'Class Id'},
    #{'id' : 'tier', 'name' : 'Tier', 'type': 'numeric'},
    #{'id' : 'size', 'name' : 'Size'},
    {'id' : 'itemName', 'name' : 'Item'}, 
    {'id' : 'Sum Total', 'name' : 'Sum Total', 'format': currency, 'type': 'numeric'},
    {'id' : 'unitPrice', 'name' : 'Mean Price Per', 'format': currency, 'type': 'numeric'}
    #{'id' : 'orderVolume', 'name' : 'Value', 'format': currency, 'type': 'numeric'}
    #{'id' : 'Min Price', 'name' : 'Minimum', 'format': currency, 'type': 'numeric'}, 
    #{'id' : 'Weighted Mean', 'name' : 'Average Price', 'format': currency, 'type': 'numeric'},
    #{'id' : 'Average Order Volume', 'name' : 'Avg. Order Vol.', 'format': num, 'type': 'numeric'},
    #{'id' : 'Max Price', 'name' : 'Maximum', 'format': currency, 'type': 'numeric'},
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
        item = list(item)
        # itemName, index 9
        item.append(lookupItem(item[2], 'displayNameWithSize'))
        item = tuple(item)
        updated.append(item)
    return updated

def lookupItem(item_id, key='displayNameWithSize'):
    # Use the cached value if available
    if item_id in known_items and key in known_items[item_id]:
        return known_items[item_id][key]
    
    # Load the item's display name and cache it
    known_items[item_id] = known_items.get(item_id, {})
    known_items[item_id][key] = fetchItem(item_id, key)
    return known_items[item_id][key]

def fetchItem(item_id, key='displayNameWithSize', local_file_path='items.json'):
    global items_cache_json
    # Load the JSON file into cache if it's not already loaded
    if items_cache_json is None:
        logger.info("Loading items from file")
        with open(local_file_path, 'r') as file:
            items_cache_json = json.load(file)
    
    # Search for the item in the cached data
    for item in items_cache_json:
        if str(item.get('id')) == str(item_id):
            return item.get(key, f"Unknown {key}")
    return f"Unknown {key}"  # Default return if item or key not found


''' 
def parse_add_itemName(items):
    updated = []
    for item in items:
        item = list(item)
        # itemName, index 9
        item.append(lookupItem(item[2]))
        item = tuple(item)
        updated.append(item)
    return updated
'''
 
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
    #logger.info(f"Updating items: {len(items)}")
    for item in items:
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
    logger.info(file_path)
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
    df.drop_duplicates(inplace=True, subset='orderId')
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

def process_log_file(log_file, cache_folder_path):
    checksum = generate_md5(log_file)
    cache_file_path = os.path.join(cache_folder_path, checksum + __version__.replace(".", "_") + '.pkl')
    file_data = load_or_create_cache(log_file, cache_file_path)
    
    df = pd.DataFrame(file_data, columns=columns)
    df.drop_duplicates(inplace=True, subset='orderId')
    df['logId'] = os.path.basename(log_file)
    # Convert timestamp to human-readable format
    timestamp = os.path.getmtime(log_file)
    human_readable_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    df['logDate'] = human_readable_date
    df['md5'] = checksum
    return df

def create_dataframe_from_log_files_debug(log_files, cache_folder_path):
    frames = []
    
    for log_file in log_files:
        data = process_log_file(log_file, cache_folder_path)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)

def create_dataframe_from_log_files_multithreaded(log_files, cache_folder_path):
    frames = []

    # Dynamically set max_workers based on system resources
    cpu_count = os.cpu_count()
    max_workers = min(len(log_files), cpu_count - 1 or 1)  # Reserve 1 CPU for system processes, min of 1 worker

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Fetch and log system information in a non-blocking way
        system_info_future = executor.submit(fetch_system_info)

        # Submit log file processing tasks
        future_to_file = {executor.submit(process_log_file, log_file, cache_folder_path): log_file for log_file in log_files}

        # Wait for system info logging to complete
        log_system_info(system_info_future.result())

        # Process futures as they complete
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
                logging.debug(f"Processed file {os.path.basename(log_file)}")
            except Exception as exc:
                logging.error(f"File {os.path.basename(log_file)} generated an exception: {exc}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_system_info():
    memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    return os.cpu_count(), memory, disk_usage

def log_system_info(info):
    cpu_count, memory, disk_usage = info
    logging.info(f"System has {cpu_count} CPUs, " +
                 f"Total memory: {memory.total / (1024 ** 3):.2f} GB, " +
                 f"Available memory: {memory.available / (1024 ** 3):.2f} GB, " +
                 f"Disk usage: {disk_usage.percent}%")

def update():
    logger.info("Starting update process")

    # Setup directories
    cwd = os.getcwd()
    cache_folder_path = os.path.join(cwd, 'cache')
    os.makedirs(cache_folder_path, exist_ok=True)
    logger.info(f"Cache directory: {cache_folder_path}")

    log_directory = os.path.expandvars(r'%localappdata%\NQ\DualUniverse\log')
    
    # Retrieve log files and sort them from newest to oldest
    log_files = [os.path.join(log_directory, lf) for lf in os.listdir(log_directory)]
    log_files.sort(key=os.path.getmtime, reverse=True)

    # Process log files and create DataFrame
    final_df = create_dataframe_from_log_files_multithreaded(log_files, cache_folder_path)
    #final_df = create_dataframe_from_log_files_debug(log_files, cache_folder_path)

    # Clean
    final_df.drop_duplicates(inplace=True, subset='orderId')

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
# Define a mapping between tab values and market IDs
market_id_map = {
    'market6-tab': 46129,
    'market12-tab': 46174,
    'aegis-tab': 46136
}
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

class_id_map = {
    'ore-tab' : 312610052,
    'pure-tab' : 1161217162,
    'product-tab' : 2533089091,
    'container-tab' : 703994582,
    'industry-tab' : 3943695040,
    'engine-tab' : 2294552479
}

tabs5 = dcc.Tabs(
    id='tabs5', 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Items', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Ore', value='ore-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Pure', value='pure-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Product', value='product-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Container', value='container-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Industry', value='industry-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Engine', value='engine-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Other', value='other-tab', style=tab_style, selected_style=tab_selected_style),
    ],
    style=matrix_style,
)
tabs6 = dcc.Tabs(
    id='tabs6', 
    value='all-tab', 
    children=[
        dcc.Tab(label='All Trends', value='all-tab', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Strongly Downward', value='strongly-down', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Moderately Downward', value='moderately-down', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Slightly Downward', value='slightly-down', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Stable', value='stable', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Slightly Upward', value='slightly-up', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Moderately Upward', value='moderately-up', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Strongly Upward', value='strongly-up', style=tab_style, selected_style=tab_selected_style),
    ],
    style=matrix_style,
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
    dcc.Store(id=__version__.replace(".", "_")),
    #dcc.Store(id='metrics-table'),
        dcc.Checklist(
            id='toggle-expired',
            options=[{'label': ' Hide Expired Orders', 'value': 'hide_expired'}],
            value=[]
        ),
        dcc.Checklist(
            id='toggle-filter',
            options=[{'label': ' Filter Prices by Std.Dev', 'value': 'filter_orders'}],
            value=[]
        ),
    html.Div(id='content', style=content_style, children=[
        html.H4(f"Dual Universe Market Intel Build 1/13/2024 Version: {__version__}"),
        #html.H4("Market Filters (Location, Order Type, Tier, Size, Item Class)"),
        tabs4,
        tabs1,
        tabs2,
        tabs3,
        tabs5,
        tabs6,
        html.Div([  # Parent div for the graph and the table
        html.Div([  # Div for the graph
                dash_table.DataTable(
                    id='head-table',
                    columns=metric_columns,
                    data=[],  # init-empty
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
                )
            ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),  # Adjust width as needed
            
        html.Div([  # Div for the table
                dcc.Graph(id="price_graph", style={'width': '100%','height': '670px', 'display': 'inline-block', 'vertical-align': 'top'})
        ], style={'width': '50%', 'display': 'inline-block'}),  # Adjust width as needed
        ]),
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
        html.H4("Orders Data"),
        dash_table.DataTable(
            id='table',
            columns=display_columns,
            data=[],
            filter_action='native',
            sort_action='native',
            page_action='native',
            page_size=10,
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

from pandas.tseries.offsets import MonthEnd
def round_to_nearest_month(date):
    # Round down to the first of the month
    first_of_month = date - pd.offsets.MonthBegin(1)
    
    # Check if date is closer to the start or end of the month
    if date - first_of_month < first_of_month + MonthEnd(1) - date:
        return first_of_month
    else:
        return first_of_month + MonthEnd(1)


# Callbacks
@app.callback(
    Output(__version__.replace(".", "_"), 'data'),
    [Input('toggle-expired', 'value')],
    prevent_initial_call=False
)
def load_clean_data(toggle_expired):
    #update()  # Rebuild the CSV, parsing live game data
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
    
    # Apply the rounding function
    # df['updateDateMonth'] = df['updateDate'].apply(round_to_nearest_month)
    
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
    # save
    df = filtered_df
        
    tab_value = class_tab_value
    if tab_value == 'all-tab':
        pass  # Use df as is
    elif tab_value in class_id_map:
        # Filter df based on the market ID from the map
        filtered_df = df[df['classId'] == class_id_map[tab_value]]
    elif tab_value == 'other-tab':
        # Filter out the specified market IDs
        excluded_ids = [312610052,1161217162,2533089091,703994582,3943695040,2294552479]
        filtered_df = df[~df['classId'].isin(excluded_ids)]
    else:
        filtered_df = df  # Default case if tab_value is not recognized
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
    filtered_df.to_csv('filtered_clean_orders.csv', index=False)

    return filtered_df.to_dict('records')

@app.callback(
    Output('metrics-table', 'data'),
    [Input('table', 'data'), Input('toggle-filter', 'value'), Input('tabs6', 'value')]
)
def update_metrics(data, filter_std_dev=True, trend_tab_value=None):
    logger.info("Update metrics called.")
    if not data:
        logger.info("Warning: empty data in update_metrics")
        return []

    # Creating DataFrame and removing duplicates
    df = pd.DataFrame(data).drop_duplicates(subset='orderId')

    # Filtering based on standard deviation for each group if enabled
    if filter_std_dev:
        def filter_by_std(group):
            mean = group['unitPrice'].mean()
            std = group['unitPrice'].std()
            return group[group['unitPrice'].between(mean - std, mean + std)]

        df = df.groupby(['orderType', 'itemName']).apply(filter_by_std).reset_index(drop=True)

    # Grouping and aggregating data
    agg_operations = {
        'orderValue': 'sum',
        'unitPrice': 'mean',
        'buyQuantity': 'mean'
    }
    grouped_df = df.groupby(['orderType', 'itemName', 'classId', 'updateDateDay']).agg(agg_operations)
    grouped_df.columns = ['Sum Total', 'unitPrice', 'buyQuantity']  # Rename aggregated columns
    grouped_df = grouped_df.reset_index()

    # Convert DataFrame to dictionaries for output
    metrics_data = grouped_df.to_dict('records')

    # Assuming head-table requires the first few records
    head_data = grouped_df.head().to_dict('records')

    def calculate_trend(x):
        # Ensure there are at least two unique data points
        if len(x['updateDateDay'].unique()) < 2:
            return 'Not enough data'

        # Perform linear regression
        try:
            coefficients, residuals, _, _, _ = np.polyfit(
                x['updateDateDay'].apply(lambda d: pd.to_datetime(d).date().toordinal()),
                x['unitPrice'], 1, full=True
            )
            slope = coefficients[0]
        except FloatingPointError:
            return 'Calculation error'

        coefficients, residuals, _, _, _ = np.polyfit(
            x['updateDateDay'].apply(lambda d: pd.to_datetime(d).date().toordinal()),
            x['unitPrice'], 1, full=True
        )
        slope = coefficients[0]

        # Determining the strength of the trend
        magnitude = abs(slope)
        trend_strength = ''
        if magnitude > 0.2: trend_strength = 'Strongly '
        elif magnitude > 0.1: trend_strength = 'Moderately '
        elif magnitude > 0.05: trend_strength = 'Slightly '

        # Determining direction of the trend
        direction = 'Upward' if slope > 0 else 'Downward' if slope < 0 else 'Stable'

        # Calculating confidence interval
        mean_x = np.mean(x['updateDateDay'].apply(lambda d: pd.to_datetime(d).date().toordinal()))
        n = len(x)
        t_value = stats.t.ppf(1-0.025, n-2)  # 95% confidence interval, adjust as needed

        conf_interval = 0.5 # t_value * np.sqrt((np.sum(residuals) / (n-2)) * (1/n + (mean_x - np.mean(mean_x))**2 / np.sum((mean_x - np.mean(mean_x))**2)))
        
        return f'{trend_strength}{direction}'# (Confidence Interval: ±{conf_interval:.2f})'

    # Calculate trend for each group defined by 'orderType', 'itemName', 'classId'
    trend_df = df.groupby(['orderType', 'itemName', 'classId']).apply(calculate_trend).reset_index(name='trend')

    # Merging trend into the main DataFrame
    results_df = pd.merge(grouped_df, trend_df, on=['orderType', 'itemName', 'classId'], how='left')

    # Sorting and saving to CSV
    results_df = results_df.sort_values(by=['classId', 'updateDateDay'])
    results_df.to_csv("market_metrics.csv", index=False)
    return results_df.to_dict('records')

@app.callback(
    Output('head-table', 'data'),
    [Input('metrics-table', 'data'), Input('tabs6', 'value')]
)
def update_head(metrics_data, trend_tab_value=None):
    # Convert the data back to DataFrame
    df = pd.DataFrame(metrics_data)

    # Reduce dataframe to latest date for each group
    head = df.sort_values(by='updateDateDay', ascending=False).groupby(['orderType', 'itemName', 'classId']).head(1)

    # Filter based on the trend
    tab_value = trend_tab_value
    if tab_value != 'all-tab':
        trend_map = {
            'strongly-down': 'Strongly Downward',
            'moderately-down': 'Moderately Downward',
            'slightly-down': 'Slightly Downward',
            'stable': 'Stable',
            'slightly-up': 'Slightly Upward',
            'moderately-up': 'Moderately Upward',
            'strongly-up': 'Strongly Upward'
        }
        head = head[head['trend'] == trend_map.get(tab_value, '')]

    head.to_csv("head_market_metrics.csv", index=False)
    return head.to_dict('records')

@app.callback(
    Output('price_graph', 'figure'),
    Input('metrics-table', 'data')
)
def update_trends_figure(data):
    data = pd.DataFrame(data)
    if data.empty:
        logger.error("No data for graph")
        return None

    # Define a window size for the moving average
    window_size = 6 # This is an example, you can adjust the size

    # Separate data for Buy and Sell orders
    buy_orders = data[data['orderType'] == 'Buy']
    sell_orders = data[data['orderType'] == 'Sell']

    # Grouping and aggregating the required columns
    buy_prices = buy_orders.groupby(['itemName', 'updateDateDay'])['unitPrice'].mean().reset_index()
    sell_prices = sell_orders.groupby(['itemName', 'updateDateDay'])['unitPrice'].mean().reset_index()
    #weighted_mean = buy_prices.groupby(['itemName', 'orderType', 'updateDateDay'])['Weighted Mean'].mean().reset_index()

    # Apply a moving average to the price data
    buy_prices['Price'] = buy_prices['unitPrice'].rolling(window=window_size, min_periods=5).mean()
    sell_prices['Price'] = sell_prices['unitPrice'].rolling(window=window_size, min_periods=5).mean()
    #weighted_mean['Price'] = weighted_mean['Weighted Mean'].mean()

    # Renaming the price columns was already done above during moving average

    # Adding a column to indicate the type of order or metric
    buy_prices['OrderType'] = 'Buy'
    sell_prices['OrderType'] = 'Sell'
    #weighted_mean['OrderType'] = 'Weighted Mean'

    # Merging the datasets
    final_data = pd.concat([buy_prices, sell_prices]) #weighted_mean

    # Creating the line plot
    #fig = px.line(final_data, x='updateDateDay', y='Price', color='itemName', line_dash='OrderType',
    #            title='Prices of Items Over Time by Order Type',
    #            labels={'updateDateDay': 'Update Date', 'Price': 'Price ($)'},
    #            category_orders={"OrderType": ["Buy", "Sell", "Weighted Mean"]})
    
    # Creating the line plot with a black background and neon colors
    fig = px.line(final_data, x='updateDateDay', y='Price', color='itemName', line_dash='OrderType',
                #title='Prices of Items Over Time by Order Type',
                #labels={'updateDateDay': 'Update Date', 'Price': 'Price ($)'},
                #text='unitPrice',
                #connectgaps=True,
                line_shape='linear',
                category_orders={"OrderType": ["Buy", "Sell"]})
    
    #fig.update_traces(textposition="bottom right")

    # Updating the layout to have a black background
    fig.update_layout({
        'plot_bgcolor': '#110000',
        'paper_bgcolor': '#110000',
        'font_color': 'lime',
    })
                    

    # Customizing the color of the lines to be more neon
    fig.update_traces(line=dict(width=1.5))
    colors = ['#00FF00', '#00EE00', '#FF00FF', '#EE00EE', '#00FFFF', '#00EEEE', '#FFFF00', '#EEEE00'] 
    # Example neon colors: green, lighter green, magenta, lighter magenta, cyan, lighter cyan, yellow, orange
    for i, trace in enumerate(fig.data):
        trace.line.color = colors[i % len(colors)]
    return fig

# Run the app
if __name__ == '__main__':
    update()
    app.run_server(debug=True)
