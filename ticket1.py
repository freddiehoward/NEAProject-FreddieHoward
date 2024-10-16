'''this is the library offered by yahoo finance that allows me to fetch data
using their api but without having to hard code the api call in myself
'''
import yfinance as yf


stockChosenTicker = "NVDA"

'''converting from a string to a Ticker object allows yfinance to access
the correct data'''
stockChosenTicker = yf.Ticker(stockChosenTicker)


'''
df stands for dataframe, and the .history attribute of the Ticker object
is what contains the historical data values, ie storing a data value per day
over a period of time for different pieces of information, such as open price,
close price, volume and volatility. 10y is the period of time for which we
want the historical data to go back for
'''
df = stockChosenTicker.history(period="10y")


'''
df.drop removes the specified columns/pieces of information from the dataframe
'''
df.drop(columns=['Stock Splits'], inplace=True)

df.drop(columns=['Dividends'], inplace=True)


'''
here I am separating the data into input data for the LSTM, as X; output data
for the LSTM, as y. since the close prices of days in the future are what we
want the LSTM to predict, I have made y, the output values, all of the close
values. X is made up of all the other values such as volatility, open price etc
'''
X, y = df.drop(columns=['Close']), df.Close.values

