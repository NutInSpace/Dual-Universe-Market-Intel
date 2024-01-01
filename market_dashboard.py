from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
from market import columns, display_columns
from market_log_to_csv import update, apply_format

# Initialize the Dash app
app = Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Market Orders Data"),
    dcc.Tabs(id='tabs', value='buy-tab', children=[  # Added default value for 'tabs'
        dcc.Tab(label='Buy Orders', value='buy-tab'),  # Changed id to value
        dcc.Tab(label='Sell Orders', value='sell-tab')  # Changed id to value
    ]),
    dash_table.DataTable(
        id='table',
        columns=display_columns, ###[{"name": i, "id": i} for i in display_columns],
        data=[],
        filter_action='native',
        sort_action='native',
        page_action='native',
        page_size=25,
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
            {'if': {'column_id': 'some_column'}, 'backgroundColor': 'rgb(227, 227, 227)', 'color': 'black'}
        ]
    ),
])

# Callback for tabs
@app.callback(
    Output('table', 'data'),
    [Input('tabs', 'value')],  # Changed 'tab' to 'value'
    prevent_initial_call=False
)
def update_table(tab_value):  # Changed parameter name for clarity
    update()  # Rebuild the CSV
    
    df = pd.read_csv('market_orders.csv')
    
    df = apply_format(df)
    
    if tab_value == 'buy-tab':
        filtered_df = df[df['buyQuantity'] > 0]  # Changed to filter for buy orders correctly
    elif tab_value == 'sell-tab':
        filtered_df = df[df['buyQuantity'] <= 0]  # Changed to filter for sell orders correctly
    else:
        filtered_df = df
    
    return filtered_df.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
