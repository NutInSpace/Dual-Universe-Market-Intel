import os
import re
import requests

import pandas as pd
from market import columns

# Create a cache for storing the items data
items_cache = None
market_cache = None

# Regular expression pattern
market_pattern = r'MarketOrder:\[marketId = (\d+), orderId = (\d+), itemType = (\d+), buyQuantity = (.*?), expirationDate = @\(\d+\) (.*?), updateDate = @\(\d+\) (.*?), unitPrice = Currency:\[amount = (\d+).*?](?:, ownerId = EntityId:\[(.*?)\], ownerName = (.*?))?\]'
owner_pattern = r"ownerId = EntityId:\[playerId = (\d+) organizationId = (\d+)\] ownerName = (.+)"

def parse_fix_owner(items):
    fixed = []
    for item in items:
        item = list(item)
        if "ownerId" in item[5]: 
            data = item[5].split(",")
            item[5] = data[0]
            rem = "".join(data[1:])
            owners = re.findall(owner_pattern, rem)
            owners = list(owners[0])
            #print(f"Owners: {owners} Data[1]: {rem}")

            # player + org 
            playerId = int(owners[0]) + int(owners[1])
            # shrugs
            playerName = owners[2]
            item[7] = playerId
            item[8] = playerName
                
        item = tuple(item)
        fixed.append(item)
    return fixed

def lookupItem(id, source='https://raw.githubusercontent.com/NutInSpace/DualUniverse-GPT/main/items.json'):
    global items_cache

    # Only load data if the cache is empty
    if items_cache is None:
        response = requests.get(source)
        items_cache = response.json()
    #else:
    #    print(f"Cached Items : {len(items_cache)}")

    # Search in the cached data
    for search in items_cache:
        #print(f"id = {type(search['id'])}; search = {type(id)}")
        if int(search['id']) == int(id):
            return search['displayNameWithSize']

    return "UnknownItem"
 
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
    markets = {'46174' : 'Market 12', '46129' : 'Market 6', '46136' : 'Aegis'}
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
    for item in items:
        item = list(item)
        # buyQuantity * unitPrice
        value = int(item[3]) * int(item[6]) / 100
        item.append(value)
        item = tuple(item)
        updated.append(item)
    return updated
       
def process_file(file_path):
    compiled_pattern = re.compile(market_pattern)
    matches = []
    print(f"Reading {file_path}")
    with open(file_path, 'r') as file:
        content = file.read()
        print(f"Matching {file_path}")
        matches.extend(re.findall(market_pattern, content))
        
    # Fixup 1, Index 8
    matches = parse_fix_owner(matches)
    # Fixup 2, Index 9
    matches = parse_add_itemName(matches)
    # Fixup 3, Index 10
    matches = parse_add_marketName(matches)
    # Fixup 9, Index 11
    matches = parse_add_orderValue(matches)

    
    return matches

def update():
    # Step 1: Retrieve log files
    log_directory = os.path.expandvars(r'%localappdata%\NQ\DualUniverse\log')
    print(f"Log Directory: {log_directory}")
    log_files = [os.path.join(log_directory, lf) for lf in os.listdir(log_directory)]
    print("Log files:")

    # Process each file and collect data
    data = []
    for lf in log_files:
        data.extend(process_file(lf))

    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Cleanup
    df = df.drop_duplicates()
    
    # Step 2: Save the DataFrame
    df.to_csv('market_orders.csv', index=False)
    print("DataFrame saved as 'market_orders.csv'")

def apply_format(df):
    df['Value'] = df['orderValue'].apply(lambda x: f"${x:,.2f}")
    return df