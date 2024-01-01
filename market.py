columns = [
    'marketId',         # 0  : from log
    'orderId',          # 1  : from log
    'itemType',         # 2  : from log
    'buyQuantity',      # 3  : from log
    'expirationDate',   # 4  : from log
    'updateDate',       # 5  : from log
    'unitPrice',        # 6  : from log
    'ownerId',          # 7  : needs cleanup
    'ownerName',        # 8  : fixup 1, from log may contain Entity
    'itemName',         # 9  : fixup 2, lookUp
    'marketName',       # 10 : fixup 3, lookUp
    'orderValue',       # 11 : fixup 4, Maths
]
currency = {'specifier': '$,.2f'}
display_columns = [
    {'id' : 'buyQuantity', 'name' : 'Order Quantity'},
    {'id' : 'expirationDate', 'name' : 'Expires'},
    {'id' : 'updateDate', 'name' : 'Updated'}, 
    {'id' : 'ownerName', 'name' : 'Seller'}, 
    {'id' : 'itemName', 'name' : 'Item'}, 
    {'id' : 'marketName', 'name' : 'Market'}, 
    {'id' : 'orderValue', 'name' : 'Value', 'format': currency, 'type': 'numeric'}, 
]