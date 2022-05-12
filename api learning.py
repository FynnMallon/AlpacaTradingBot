
# # Get daily price data for AAPL over the last 5 trading days.

# aapl_bars = api.get_bars("AAPL", TimeFrame.Hour, "2021-06-08", "2021-06-08", adjustment='raw').df
# print(aapl_bars)
# # See how much AAPL moved in that timeframe.
# week_open = aapl_bars[0].o
# week_close = aapl_bars[-1].c
# percent_change = (week_close - week_open) / week_open * 100
# print('AAPL moved {}% over the last 5 days'.format(percent_change))


# #Get Account info
# account = api.get_account()

# # Check our current balance vs. our balance at the last market close
# balance_change = float(account.equity) - float(account.last_equity)
# print(f'Today\'s portfolio balance change: ${balance_change}')

# #Retrieves Active Assets
# active_assets = api.list_assets(status='active')

# #Filter for live NASDAQ Assets
# nasdaq_assets = [a for a in active_assets if a.exchange == 'NASDAQ']

# #Check if asset is tradeable
# aapl_asset = api.get_asset('AAPL')
# if aapl_asset.tradeable:
#     print("we can trade AAPL")

# class AlpacaPaperSocket(alpaca.REST):
#     def __init__(self):
#         super().__init__(
#             key_id='PK1GNWZ4WG3CFDX64EY0',
#             secret_key='E3GbOVdeLp5GR6HgOW0dxmzZtqsicsBn7DcXNUFz',
#             base_url='https://paper-api.alpaca.markets'
#         )